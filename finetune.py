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
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
import envs.plane as plane
import matplotlib.pyplot as plt
import math

torch.backends.cudnn.benchmark = True

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
        self.device = torch.cuda.set_device(cfg.device)

        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([cfg.experiment,cfg.agent.name,cfg.domain,cfg.obs_type,str(cfg.seed)])
            wandb.init(project="urlb",group=cfg.agent.name,name=exp_name)

        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb,use_wandb=cfg.use_wandb)

        # create envs
        time_limit = cfg.time_limit if cfg.use_time_limit else None
        if 'plane' in cfg.task:
            self.train_env = plane.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, time_limit)
            self.eval_env = plane.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, time_limit)
        elif 'fetch' in cfg.task:
            self.train_env = fetch.make(cfg.task, cfg.obs_type,
                                    cfg.frame_stack, cfg.action_repeat,
                                    cfg.seed, cfg.random_reset, time_limit)
            self.eval_env = fetch.make(cfg.task, cfg.obs_type,
                                    cfg.frame_stack, cfg.action_repeat,
                                    cfg.seed, cfg.random_reset, time_limit)
        else: 
            self.train_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                    cfg.action_repeat, cfg.seed, use_custom=cfg.use_custom,
                                    random_reset=cfg.random_reset, time_limit=time_limit)
            self.eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                    cfg.action_repeat, cfg.seed, use_custom=cfg.use_custom,
                                    random_reset=cfg.random_reset, time_limit=time_limit)

        # create agent
        num_expl_steps = 0
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                num_expl_steps,
                                cfg.agent)

        # initialize from pretrained
        if self.cfg.restore_snapshot_ts > 0 and os.path.exists(self.cfg.restore_snapshot_dir):
            if self.cfg.agent.name in ["cic", "dads"]:
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

        # create video recorders
        if self.cfg.task in ["fetch_push", "fetch_barrier", "fetch_barrier2"]:
            self.video_recorder = VideoRecorder(
                self.work_dir if cfg.save_video else None,
                camera_id= 0 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb, frame_lst=True)
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if cfg.save_train_video else None,
                camera_id= 0 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb, frame_lst=True)
        else: 
            self.video_recorder = VideoRecorder(
                self.work_dir if cfg.save_video else None,
                camera_id= 0 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb)
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if cfg.save_train_video else None,
                camera_id= 0 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # Initialize IRM module
        self.irm = hydra.utils.instantiate(cfg.irm)
        self.setup_plot_parameters()
        self.irm.init_IRM(self.eval_env, self.agent, cfg.task, self.video_recorder, self.logger)
        if self.cfg.hrl:
            import agent.manager as manager
            self.agent = manager.ManagerAgent(worker_agent=self.agent, obs_shape=self.train_env.observation_spec().shape)
            self.irm.agent = self.agent

        # create data storage
        if os.path.exists(cfg.replay_dir):
            replay_dir = Path(cfg.replay_dir) / 'buffer'
        else:
            replay_dir = self.work_dir / 'buffer'
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  replay_dir, cfg.irm.name == 'reward_relabel')
        # create replay buffer
        self.replay_loader, self.replay_buffer = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount,
                                                skill_duration=self.agent.skill_duration)
        self._replay_iter = None
        if self.cfg.irm.name == 'reward_relabel':
            self.irm.replay_buffer = self.replay_buffer


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

    def eval(self, skill=None):
        episode = 1 
        tls = [self.agent.skill_duration for _ in range(self.irm.n_rewards)]
        if self.cfg.hrl:
            traj_out = self.run_metas([{'skill': None} for _ in range(self.irm.n_rewards)], tl_lst=tls, video=True, video_name=str(self.global_frame), use_handcrafted=True)
        elif "goal" in self.irm.extr_reward[0] and self.irm.n_rewards > 1:
            traj_out = self.run_metas(self.agent.ft_skills, tl_lst=tls, video=True, video_name=str(self.global_frame), use_handcrafted=True)
        else:
            traj_out = self.run_metas(self.agent.ft_skills, tl_lst=tls, video=True, video_name=str(self.global_frame))
        total_reward_lst, ep_obs, step = traj_out['reward'], traj_out['ep_obs'], traj_out['step']

        total_reward = sum(total_reward_lst)
        if isinstance(step, list):
            step = step[-1]
        if "barrier" in self.cfg.task:
            # early terminations don't count
            total_reward = total_reward / step 

        if self.cfg.agent.name == "cic":
            # positive_kq_pair, negative_kq_pair = self.eval_cic(ep_obs)
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / episode)
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)
                for i, rew in enumerate(total_reward_lst):
                    log(f'episode_reward_{i}', rew)
        elif self.cfg.agent.name == "dads":
            # logp, logp_altz = self.eval_dads(ep_obs)
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / episode)
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)
                # log('logp', logp)
                # log('logp_altz', logp_altz)
                # log('delta_logp', (logp - logp_altz))
        else: 
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / episode)
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)

        if self.draw_plot:
            self.save_plots()

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        # collect_until_step = utils.Until(self.cfg.frames_to_collect, 
        #                                self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()

        # Intrinsic Reward Matching
        self.irm.run_skill_selection() 
        meta = self.irm.get_ft_meta(episode_step)

        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None

        while train_until_step(self.global_step):
            meta = self.irm.get_ft_meta(episode_step)

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

                episode_step = 0
                episode_reward = 0
                meta = self.irm.get_ft_meta(episode_step)
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                
            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
       
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # try to save snapshot
            if self.global_frame in self.cfg.snapshots:
                self.save_snapshot()

            # take env step
            prev_time_step = time_step
            time_step = self.train_env.step(action)
            
            # to avoid additional environment wrapper, IRM deals with changing reward functions (sequential goals)
            time_step = self.irm.recompute_reward(prev_time_step, time_step, episode_step)

            episode_reward += time_step.reward
            if self.cfg.hrl:
                self.replay_storage.add(time_step, self.agent.meta_action)
            else:
                self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def run_metas(self, metas, **kwargs):
        skill_lst = [m['skill'] for m in metas]
        return self.irm.run_skills(skill_lst, **kwargs)

    def save_plots(self):
        # run trajectories
        episode = 0
        eval_until_episode = utils.Until(self.cfg.num_plot)
        all_ep_obs = []
        skill_lst = []
        tl_lst = [self.agent.skill_duration for _ in range(self.irm.n_rewards)]
        while eval_until_episode(episode):
            if episode == 0:
                if self.cfg.hrl:
                    # raise NotImplementedError
                    traj_out = self.run_metas([{'skill': None} for _ in range(self.irm.n_rewards)], tl_lst=tl_lst)
                    # traj_out = self.run_metas([self.agent.get_ft_meta()], tl_lst=tl_lst, video=True)
                else:
                    traj_out = self.run_metas(self.agent.ft_skills, tl_lst=tl_lst)
            else: 
                traj_out = self.irm.run_skills([None for _ in range(self.irm.n_rewards)], tl_lst)
            ep_obs = np.array(traj_out['ep_obs'])
            ep_obs = ep_obs.reshape(-1, ep_obs.shape[-1])
            all_ep_obs.append(ep_obs)
            if "skill" in traj_out:
                # for non-ddpg
                skill_lst.append(traj_out['skill'])
            episode += 1
        self.plot_traj(skill_lst, all_ep_obs, str(self.global_frame), title="ft skill (maroon) + others", goal=self.irm.get_goal())

    def plot_traj(self, skills, all_traj, name, style_map=None, title=None, goal=None, labels=None):
        if style_map is None:
            color_map = ['maroon', 'orangered', 'darkorange', 'gold', 'yellow', 
                        'lawngreen', 'aquamarine', 'deepskyblue', 'blue', 'darkblue',
                        'slateblue', 'rebeccapurple', 'indigo', 'blueviolet', 'mediumorchid',
                        'violet', 'magenta', 'orchid', 'deeppink', 'pink',
                        'gainsboro', 'silver', 'darkgray', 'dimgray', 'black']
            style_map = color_map
            
        plt.xlim(self.xlim1, self.xlim2)
        plt.ylim(self.ylim1, self.ylim2)

        label = None
        if not self.plot_2d:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        for i in range(len(all_traj)):
            if labels is not None:
                label = labels[i]
            traj_coord = np.array(all_traj[i])
            try:
                traj_coord, _ = self.agent.process_observation(torch.from_numpy(traj_coord).to(self.cfg.device))
                traj_coord = traj_coord.cpu().numpy()
                
                if self.plot_2d:
                    plt.plot(traj_coord[:, 0], traj_coord[:, 1], c=style_map[i], label=label)
                else:
                    plt.plot(traj_coord[:, 0], traj_coord[:, 1], traj_coord[:, 2], c=style_map[i], label=label)
            except:
                continue

        for g in goal:
            if g is None:
                pass
            elif self.plot_2d:
                plt.plot(g[0], g[1], marker="*", markersize="20")
            else:
                plt.plot(g[0], g[1], g[2], marker="*", markersize="20") 
        if title is not None:
            plt.title(title)
        if labels is not None:
            plt.legend(loc="upper right")
        plot_name = f"{self.plot_dir}/{name}.png"
        plt.savefig(plot_name)
        plt.clf()

        # write meta
        with open(f"{self.plot_dir}/meta_{name}.txt", "w") as f:
            for i in range(len(skills)):
                f.write(f"{style_map[i]}: {skills[i]}")
                f.write('\n')
        print(f"saved plot to {plot_name}")

    def eval_cic(self, ep_ob):
        positive_kq_pair = 0
        negative_kq_pair = 0
        total_pairs = 0

        # make sure ep_ob is a list of obs 
        if len(ep_ob) != self.irm.n_rewards:
            ep_ob = [ep_ob]

        for r in range(self.irm.n_rewards):
            # get skills corresp to reward
            pos_cic_skill = torch.from_numpy(self.agent.ft_skills[r]['skill']).to(self.cfg.device).unsqueeze(0)
            neg_cic_skill = torch.from_numpy(np.random.uniform(0,1,self.cfg.agent.skill_dim).astype(np.float32)).to(self.cfg.device).unsqueeze(0)

            # get obs corresp to reward
            ep_obs = torch.from_numpy(np.array(ep_ob[r])).to(self.cfg.device)
            obs_pt, _ = self.agent.process_observation(ep_obs)
            start, end = self.agent.skill_duration * r, self.agent.skill_duration * (r+1)
            cic_obs = obs_pt[:len(ep_obs)-self.cfg.agent.nstep]
            cic_next_obs = obs_pt[self.cfg.agent.nstep:]

            # calculate cic rewards
            pos_loss = self.agent.compute_inner_product(cic_obs, cic_next_obs, pos_cic_skill)
            neg_loss = self.agent.compute_inner_product(cic_obs, cic_next_obs, neg_cic_skill)
            negative_kq_pair += neg_loss.mean()
            positive_kq_pair += pos_loss.mean()
            total_pairs += obs_pt.shape[0]
        positive_kq_pair /= total_pairs
        negative_kq_pair /= total_pairs
        return positive_kq_pair.item(), negative_kq_pair.item()

    def eval_dads(self, ep_observations, skill):
        logp = 0
        logp_altz = 0
        # eval trajectory corresp to skill
        skill = torch.from_numpy(skill).to(self.cfg.device).unsqueeze(0).repeat(len(ep_observations[0]) - self.cfg.agent.nstep, 1)
        for ep_obs in ep_observations:
            ep_obs = [torch.from_numpy(obs).to(self.cfg.device).unsqueeze(0) for obs in ep_obs]
            obs_pt, _ = self.agent.process_observation(torch.cat(ep_obs, dim=0))
            dads_obs = obs_pt[:len(ep_obs)-self.cfg.agent.nstep]
            dads_next_obs = obs_pt[self.cfg.agent.nstep:]
            _, rew_info = self.agent.compute_intr_reward(dads_obs, skill, dads_next_obs, None, keep_grad=False)
            logp += rew_info['logp'].mean().item() 
            logp_altz += rew_info['logp_altz'].mean().item() 
        logp /= len(ep_observations)
        logp_altz /= len(ep_observations)
        return logp, logp_altz

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
            payload = torch.load(f, map_location=f"cuda:{self.cfg.device}")
        return payload

    def setup_plot_parameters(self):
        self.plot_2d, self.draw_plot = True, True
        if "plane" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = -140, -140
            self.xlim2, self.ylim2 = 140, 140
            self.irm.bounds = dict(min=-128, max=128)
            self.train_env.step_limit = cfg.time_limit if cfg.use_time_limit else 200
        elif "fetch_push" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = 0.7, 0.5
            self.xlim2, self.ylim2 = 1.7, 1.2
            self.irm.bounds = dict(min=0.5, max=1.7)
            self.video_recorder = VideoRecorder(
                self.work_dir if self.cfg.save_video else None, frame_lst=True)
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if self.cfg.save_train_video else None, frame_lst=True)
            self.train_env.step_limit = self.train_env._max_episode_steps
        elif "fetch_barrier" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = 0.7, 0.5
            self.xlim2, self.ylim2 = 1.7, 1.2
            self.train_env.step_limit = self.train_env.env._max_episode_steps
            min_th, max_th = torch.zeros(1, 28).to(self.device), torch.ones(1, 28).to(self.device)
            min_th[:, 3] = 0.96
            min_th[:, 4] = 0.5
            max_th[:, 3] = 1.4
            max_th[:, 4] = 1
        elif "fetch_reach" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = 0.7, 0.5
            self.xlim2, self.ylim2 = 1.7, 1.2
            self.plot_2d = False 
            self.train_env.step_limit = self.cfg.time_limit if self.cfg.use_time_limit else 200
        elif "jaco" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = -1, -1
            self.xlim2, self.ylim2 = 1, 1
            self.plot_2d = False
            self.train_env.step_limit = 125
        elif "walker" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = -1, -1
            self.xlim2, self.ylim2 = 0, 1
            self.train_env.step_limit = self.cfg.time_limit * 4 if self.cfg.use_time_limit else -1 # fix
        elif "quadruped" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = -1, -1
            self.xlim2, self.ylim2 = 0, 1
            self.train_env.step_limit = self.cfg.time_limit * 4 if self.cfg.use_time_limit else -1 # fix
        else:
            self.draw_plot = False
        if self.draw_plot:
            self.plot_dir = self.work_dir / 'plot'
            self.plot_dir.mkdir(exist_ok=True)
        self.eval_env.step_limit = self.train_env.step_limit


@hydra.main(config_path='.', config_name='finetune')
def main(cfg):
    from finetune import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
