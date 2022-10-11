import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import fetch
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, DADSReplayBuffer
from video import TrainVideoRecorder, VideoRecorder
import wandb
import time

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS
import envs.plane as plane
import matplotlib.pyplot as plt

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        torch.cuda.set_device(cfg.device)
        
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb,use_wandb=cfg.use_wandb)

        # create envs
        time_limit = cfg.time_limit if cfg.use_time_limit else None
        if 'plane' in cfg.domain:
            self.train_env = plane.make(cfg.domain, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, time_limit)
            self.eval_env = plane.make(cfg.domain, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, time_limit)
        elif 'fetch' in cfg.domain:
            self.train_env = fetch.make(cfg.domain, cfg.obs_type,
                                    cfg.frame_stack, cfg.action_repeat,
                                    cfg.seed, cfg.random_reset, time_limit)
            self.eval_env = fetch.make(cfg.domain, cfg.obs_type,
                                    cfg.frame_stack, cfg.action_repeat,
                                    cfg.seed, cfg.random_reset,time_limit)
        else: 
            # DMC setup
            task = PRIMAL_TASKS[self.cfg.domain]
            self.train_env = dmc.make(task, cfg.obs_type,
                                    cfg.frame_stack, cfg.action_repeat,
                                    cfg.seed, use_custom=cfg.use_custom, time_limit=time_limit,
                                    random_reset=cfg.random_reset)
            self.eval_env = dmc.make(task, cfg.obs_type,
                                    cfg.frame_stack, cfg.action_repeat,
                                    cfg.seed, use_custom=cfg.use_custom, time_limit=time_limit,
                                    random_reset=cfg.random_reset)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # initialize from pretrained
        if cfg.restore_snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)
            print(f"initialized agent from {self.cfg.restore_snapshot_dir}")
        else:
            print(f"training from scratch")

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        self.data_specs, self.meta_specs = data_specs, meta_specs

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        if self.cfg.agent.name == "dads":
            self.replay_loader = DADSReplayBuffer(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size, cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount, cfg.batch_size)
        else: 
            self.replay_loader = make_replay_loader(
                self.replay_storage, cfg.replay_buffer_size,
                cfg.batch_size, cfg.replay_buffer_num_workers,
                False, cfg.nstep, cfg.discount)
            self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id= 0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb, frame_lst= 'fetch' in self.cfg.domain)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id= 0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb, frame_lst= 'fetch' in self.cfg.domain)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # plotting
        self.plot_2d, self.draw_plot = True, True
        if self.cfg.domain == "plane":
            self.xlim1, self.ylim1 = -128, -128
            self.xlim2, self.ylim2 = 128, 128
        elif self.cfg.domain == "fetch_push":
            self.xlim1, self.ylim1 = 0.7, 0.5
            self.xlim2, self.ylim2 = 1.7, 1.2
        elif self.cfg.domain == "fetch_reach":
            self.xlim1, self.ylim1 = 0.5, 0.25
            self.xlim2, self.ylim2 = 1.5, 1
            self.plot_2d = False
        elif self.cfg.domain == "quadruped":
            self.xlim1, self.ylim1 = -10, -10
            self.xlim2, self.ylim2 = 10, 10
        elif self.cfg.domain == "jaco":
            self.xlim1, self.ylim1 = -1, -1
            self.xlim2, self.ylim2 = 1, 1
        elif self.cfg.domain == "walker":
            self.xlim1, self.ylim1 = -1, -1
            self.xlim2, self.ylim2 = 0, 1
            self.bounds = dict(min=-1, max=1)
            self.train_env.step_limit = self.cfg.time_limit * 4
            self.plot_2d = False
        else:
            self.draw_plot = False

        if self.draw_plot:
            self.plot_dir = self.work_dir / 'plot'
            self.plot_dir.mkdir(exist_ok=True)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        ep_observations = []
        while eval_until_episode(episode): 
            out = self.run_skill(meta['skill'], video=(episode == 0))
            total_reward += out['reward']
            step += out['step']
            episode += 1
            ep_observations.append(out['ep_obs'])

        if self.cfg.agent.name == "cic":
            positive_kq_pair, negative_kq_pair = self.eval_cic(ep_observations, out['skill'])
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / episode)
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)
                log('positive_kq', positive_kq_pair)
                log('negative_kq', negative_kq_pair)
                log('delta_kq', (positive_kq_pair - negative_kq_pair))
        elif self.cfg.agent.name == "dads":
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / episode)
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)
        else: 
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / episode)
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)
        
        if self.draw_plot:
            # run trajectories
            step, episode = 0, 0
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
            ep_neg_observations = []
            skill_lst = []
            while eval_until_episode(episode):
                # sample diff negative skills every time because all of them are 
                # negative trajectories (also for later plotting purposes)
                out = self.run_skill(None, video=(episode == 0))
                skill_lst.append(out['skill'])
                step += out['step']
                episode += 1
                ep_neg_observations.append(out['ep_obs'])

            # plot
            color_map = ['b', 'g', 'r', 'c', 'm', 'y']
            style_map = []
            for line_style in ['-', '--', '-.', ':']:
              style_map += [color + line_style for color in color_map]

            self.plot_traj(skill_lst, ep_neg_observations, str(self.global_step), style_map=style_map)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                if self.cfg.agent.name == "dads":
                    if self.global_step % self.cfg.agent.update_dads_every_steps == 0:
                        update_time = time.time()
                        metrics = self.agent.update(self.replay_loader, self.global_step, self.logger)
                        self.replay_storage = ReplayBufferStorage(self.data_specs, self.meta_specs,
                                                  self.work_dir / f'buffer')
                        self.replay_loader = DADSReplayBuffer(self.replay_storage,
                                                self.cfg.replay_buffer_size,
                                                self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
                                                False, self.cfg.nstep, self.cfg.discount, self.cfg.batch_size)
                        metrics['update_time'] = time.time() - update_time
                else:
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # try to save snapshot
            if self.global_frame in self.cfg.snapshots:
                self.save_snapshot()

            # take env step
            if "fetch" in self.cfg.domain:
                time_step = self.train_env.step(action, make_video=self.cfg.save_train_video)
            else:
                time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def run_skill(self, skill, video=False, video_name=None):
        step, episode, total_reward = 0, 0, 0
        ep_obs, ep_action = [], []
        meta = self.agent.init_meta(skill)

        time_step = self.eval_env.reset()
        ep_obs.append(time_step.observation)
        self.video_recorder.init(self.eval_env, enabled=video)
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=True)  
            if "fetch" in self.cfg.domain:
                time_step = self.eval_env.step(action, make_video=video)
            else:
                time_step = self.eval_env.step(action)
            self.video_recorder.record(self.eval_env)
            ep_obs.append(time_step.observation)
            ep_action.append(action)
            total_reward += time_step.reward
            step += 1
        if video:
            if video_name is None:
                video_name = f"{self.global_frame}.mp4"
            self.video_recorder.save(video_name)
        return dict(ep_obs=ep_obs, ep_action=ep_action, reward=total_reward, step=step, skill=meta['skill'])

    def eval_cic(self, ep_observations, skill):
        positive_kq_pair = 0
        negative_kq_pair = 0
        # eval trajectory corresp to skill
        pos_cic_skill = torch.from_numpy(skill).to(self.cfg.device).unsqueeze(0)
        neg_cic_skill = torch.from_numpy(np.random.uniform(0,1,self.cfg.agent.skill_dim).astype(np.float32)).to(self.cfg.device).unsqueeze(0)
        for ep_obs in ep_observations:
            ep_obs = [torch.from_numpy(obs).to(self.cfg.device).unsqueeze(0) for obs in ep_obs]
            obs_pt, _ = self.agent.process_observation(torch.cat(ep_obs, dim=0))
            cic_obs = obs_pt[:len(ep_obs)-self.cfg.agent.nstep]
            cic_next_obs = obs_pt[self.cfg.agent.nstep:]
            pos_loss = self.agent.compute_inner_product(cic_obs, cic_next_obs, pos_cic_skill)
            neg_loss = self.agent.compute_inner_product(cic_obs, cic_next_obs, neg_cic_skill)
            negative_kq_pair += neg_loss.mean()
            positive_kq_pair += pos_loss.mean()
        positive_kq_pair /= len(ep_observations)
        negative_kq_pair /= len(ep_observations)
        return positive_kq_pair, negative_kq_pair

    def plot_traj(self, skills, all_traj, name, style_map=None, title=None):
        style_map = ['maroon', 'orangered', 'darkorange', 'gold', 'yellow', 
                    'lawngreen', 'aquamarine', 'deepskyblue', 'blue', 'darkblue',
                    'slateblue', 'rebeccapurple', 'indigo', 'blueviolet', 'mediumorchid',
                    'violet', 'magenta', 'orchid', 'deeppink', 'pink',
                    'gainsboro', 'silver', 'darkgray', 'dimgray', 'black']

        plt.xlim(self.xlim1, self.xlim2)
        plt.ylim(self.ylim1, self.ylim2)

        if not self.plot_2d:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        for i in range(len(all_traj)):
            traj_coord = np.array(all_traj[i])
            traj_coord, _ = self.agent.process_observation(torch.from_numpy(traj_coord).to(self.cfg.device))
            traj_coord = traj_coord.cpu().numpy()
            if self.plot_2d:
                plt.plot(traj_coord[:, 0], traj_coord[:, 1], c=style_map[i])
            else:
                plt.plot(traj_coord[:, 0], traj_coord[:, 1], traj_coord[:, 2], c=style_map[i])
        if title is not None:
            plt.title(title)
        plot_name = f"{self.plot_dir}/{name}.png"
        plt.savefig(plot_name)
        plt.clf()

        # write meta
        with open(f"{self.plot_dir}/meta_{name}.txt", "w") as f:
            for i in range(len(all_traj)):
                f.write(f"{style_map[i]}: {skills[i]}")
                f.write('\n')
        print(f"saved plot to {plot_name}")

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot_dir = Path(self.cfg.restore_snapshot_dir)
        snapshot = snapshot_dir / f'snapshot_{self.cfg.restore_snapshot_ts}.pt'

        with snapshot.open('rb') as f:
            payload = torch.load(f)
        return payload


@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    from pretrain import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()

    # create logger
    if cfg.use_wandb:
        import omegaconf
        wandb.init(entity="url",project="0714_plane",group=cfg.experiment_folder,name=cfg.experiment_name,tags=[cfg.experiment_folder], sync_tensorboard=True)
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

    workspace.train()


if __name__ == '__main__':
    main()
