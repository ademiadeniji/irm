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

        
        assert self.cfg.z_id in ['irm_random', 'irm_cem', 'irm_gradient_descent', 'irm_random_iter', 'random_skill', 'env_rollout', 'env_rollout_cem', 'grid_search', 'env_rollout_iter']
        self.z_id = self.cfg.z_id

        num_expl_steps = 0
        self.best_skill = None

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                num_expl_steps,
                                cfg.agent)

        # initialize from pretrained
        if self.cfg.restore_snapshot_ts > 0:
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

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create video recorders
        if self.cfg.task in ["fetch_push", "fetch_barrier"]:
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
        self.plot_2d, self.draw_plot = True, True
        if "plane" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = -140, -140
            self.xlim2, self.ylim2 = 140, 140
            self.bounds = dict(min=-128, max=128)
            self.train_env.step_limit = cfg.time_limit if cfg.use_time_limit else 200
        elif "fetch_push" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = 0.7, 0.5
            self.xlim2, self.ylim2 = 1.7, 1.2
            self.bounds = dict(min=0.5, max=1.7)
            self.video_recorder = VideoRecorder(
                self.work_dir if cfg.save_video else None, frame_lst=True)
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if cfg.save_train_video else None, frame_lst=True)
            self.train_env.step_limit = self.train_env._max_episode_steps
        elif "fetch_barrier" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = 0.7, 0.5
            self.xlim2, self.ylim2 = 1.7, 1.2
            self.train_env.step_limit = self.train_env.env._max_episode_steps
            min_th, max_th = torch.zeros(1, 28).to(cfg.device), torch.ones(1, 28).to(cfg.device)
            min_th[:, 3] = 0.96
            min_th[:, 4] = 0.5
            max_th[:, 3] = 1.4
            max_th[:, 4] = 1
            self.bounds = {'min': min_th, 'max': max_th}
        elif "fetch_reach" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = 0.7, 0.5
            self.xlim2, self.ylim2 = 1.7, 1.2
            self.plot_2d = False 
            self.train_env.step_limit = cfg.time_limit if cfg.use_time_limit else 200
            self.bounds = dict(min=0.5, max=1.7)
        elif "jaco" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = -1, -1
            self.xlim2, self.ylim2 = 1, 1
            self.bounds = dict(min=-1, max=1)
            self.plot_2d = False
            self.train_env.step_limit = 125
        elif "walker" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = -1, -1
            self.xlim2, self.ylim2 = 0, 1
            self.bounds = dict(min=-1, max=1)
            self.train_env.step_limit = self.cfg.time_limit * 4 if cfg.use_time_limit else -1 # fix
        elif "quadruped" in self.cfg.task:
            self.draw_plot = True 
            self.xlim1, self.ylim1 = -1, -1
            self.xlim2, self.ylim2 = 0, 1
            self.bounds = dict(min=-1, max=1)
            self.train_env.step_limit = self.cfg.time_limit * 4 if cfg.use_time_limit else -1 # fix
        else:
            self.draw_plot = False
        if self.draw_plot:
            self.plot_dir = self.work_dir / 'plot'
            self.plot_dir.mkdir(exist_ok=True)

        # commonly used vars
        self.agent.extr_reward_id = 0
        self.n_rewards = max(1, len(self.agent.extr_reward_seq))
        self.agent.skill_duration = self.train_env.step_limit // self.n_rewards

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount,
                                                skill_duration=self.agent.skill_duration)
        self._replay_iter = None


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

    def get_extr_reward(self, step, reward, prev_obs, curr_obs, action):
        if "goal" in self.cfg.agent.extr_reward or self.cfg.use_handcrafted_task_reward:
            # use hand-crafted reward
            return self.agent.get_extr_rew(step)(prev_obs, curr_obs, action)
        else: 
            # hand-crafted reward is not a substitute for true reward
            return reward

    def eval(self, skill=None):
        episode = 1 
        tls = [self.agent.skill_duration for _ in range(self.n_rewards)]
        if "goal" in self.agent.extr_reward and self.n_rewards > 1:
            traj_out = self.run_metas(self.agent.ft_skills, tl_lst=tls, extr_reward_lst=range(self.n_rewards), video=True, use_handcrafted=True)
        else:
            traj_out = self.run_metas(self.agent.ft_skills, tl_lst=tls, video=True)
        total_reward, ep_obs, step = traj_out['reward'], traj_out['ep_obs'], traj_out['step']

        if isinstance(total_reward, list):
            total_reward = sum(total_reward)
        if isinstance(step, list):
            step = sum(step)
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
                # log('positive_kq', positive_kq_pair)
                # log('negative_kq', negative_kq_pair)
                # log('delta_kq', (positive_kq_pair - negative_kq_pair))
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
        collect_until_step = utils.Until(self.cfg.frames_to_collect, 
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.agent.find_ft_meta(self.bounds) 
        if self.cfg.agent.name != "ddpg":
            if self.cfg.agent.z_id == "env_rollout":
                self.find_best_skill()
            if self.cfg.agent.z_id == "env_rollout_cem":
                self.find_best_skill_cem()
            elif self.cfg.agent.z_id == "grid_search":
                self.grid_search()
            elif self.cfg.agent.z_id == "irm_random_iter":
                self.irm_random_iter()
            elif self.cfg.agent.z_id == "env_rollout_iter":
                self.env_rollout_iter()

        meta = self.agent.get_ft_meta(episode_step)
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None

        while train_until_step(self.global_step):
            meta = self.agent.get_ft_meta(episode_step)
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

                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                episode_step = 0
                episode_reward = 0

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
            proc_prev_obs = self.agent.process_observation(torch.from_numpy(prev_time_step.observation).unsqueeze(0))[0][0].numpy()
            proc_obs = self.agent.process_observation(torch.from_numpy(time_step.observation).unsqueeze(0))[0][0].numpy()
            relevant_reward = self.get_extr_reward(episode_step, time_step.reward, proc_prev_obs, proc_obs, None)
            time_step = dmc.update_time_step_reward(time_step, relevant_reward) # only important for replay buffer
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def find_best_skill(self):
        max_sk, max_rew = None, float('-inf')
        for i in range(self.cfg.num_env_skill_rollouts):
            traj_out = self.run_skill(None)
            if (max_rew < traj_out['reward']):
                max_rew = traj_out['reward'] 
                max_sk = traj_out['skill'] 
        self.agent.ft_skills = [dict(skill=max_sk)]
    
    def find_best_skill_cem(self):
        with torch.no_grad():
            mean = torch.zeros(self.agent.skill_dim, requires_grad=False, device=self.device) + 0.5
            std = torch.zeros(self.agent.skill_dim, requires_grad=False, device=self.device) + 0.25
            for iter in range(self.agent.num_cem_iterations):
                samples = torch.normal(mean.repeat(self.cfg.num_env_skill_rollouts, 1), std.repeat(self.cfg.num_env_skill_rollouts, 1))
                rewards = []
                for sk in range(self.cfg.num_env_skill_rollouts):
                    if isinstance(self.run_skill(samples[sk])['reward'], float):
                        rewards.append(self.run_skill(samples[sk])['reward'])
                    else:
                        rewards.append(self.run_skill(samples[sk])['reward'].item())
                sorted_rewards = np.flip(np.argsort(rewards))
                elite_idxs = sorted_rewards[:self.agent.num_cem_elites]
                elites = samples[elite_idxs.copy()]
                mean = torch.mean(elites, dim=0)
                std = torch.std(elites, dim=0)
            self.agent.ft_skills = [dict(skill=elites[0].cpu().numpy())]
    
    def grid_search(self):
        max_sk, max_rew = None, float('-inf')
        curr_sk = np.zeros(self.cfg.agent.skill_dim, dtype=np.float32)
        for i in range(math.floor(1 / self.cfg.grid_search_size) + 1):
            traj_out = self.run_skill(curr_sk)
            if (max_rew < traj_out['reward']):
                max_rew = traj_out['reward'] 
                max_sk = traj_out['skill']
            curr_sk = curr_sk + self.cfg.agent.grid_search_size
        self.agent.ft_skills = [dict(skill=max_sk)]

    def get_epic_obs(self, out, r):
        if self.cfg.agent.pearson_setting_sequencing == "prev_rollout":
            epic_obs = np.array(out['ep_obs'][max(r-1, 0)])
            epic_actions = np.array(out['ep_action'][max(r-1, 0)])
        if self.cfg.agent.pearson_setting_sequencing == "prev_rollout_last":
            epic_obs = np.array(out['ep_obs'][max(r-1, 0)])
            epic_actions = np.array(out['ep_action'][max(r-1, 0)])
            if r > 1:
                epic_obs = np.tile(epic_obs[epic_obs.shape[0] // 2:], (2, 1))
                epic_actions = np.tile(epic_actions[epic_actions.shape[0] // 2:], (2, 1))
        else:
            epic_obs = np.array(out['ep_obs'][r])
            epic_actions = np.array(out['ep_action'][r])
        obs, next_obs = torch.from_numpy(epic_obs[:-1, :]).to(self.cfg.device), torch.from_numpy(epic_obs[1:, :]).to(self.cfg.device)
        return obs, next_obs

    def irm_random_iter(self):    
        best_skills = []
        tl_lst = [self.agent.skill_duration for _ in range(self.n_rewards)]

        # select first best skill with cfg's pearson / canonical settings
        self.agent.extr_reward_id = 0
        min_sk = self.agent.irm_random_search(self.bounds).cpu().numpy()
        best_skills.append(min_sk)

        # rollout first skill. select second best skill with possibly new canonical setting
        for r in range(1, self.n_rewards):
            self.agent.extr_reward_id = r
            prev_rollout = self.run_skills(best_skills, tl_lst[:r], use_handcrafted=True)
            obs_th, next_obs_th = self.get_epic_obs(prev_rollout, r-1) # obs from prev rollout
            
            # take care of prev_rollout_last
            min_sk = self.agent.irm_random_search(self.bounds, obs_th, next_obs_th).cpu().numpy()
            best_skills.append(min_sk)

        self.agent.ft_skills = [dict(skill=sk) for sk in best_skills]

    def env_rollout_iter(self):    
        best_skills = []
        tl_lst = [self.agent.skill_duration for _ in range(self.n_rewards)]
        extr_reward_lst = range(self.n_rewards)
        num_per_skill = self.cfg.num_env_skill_rollouts // self.n_rewards

        for r in range(self.n_rewards):
            max_sk, max_rew = None, float('-inf')
            for i in range(num_per_skill):
                traj_out = self.run_skills(best_skills + [None], tl_lst[:r+1], extr_reward_lst, use_handcrafted=True)
                if (max_rew < traj_out['reward'][r]):
                    max_rew = traj_out['reward'][r] 
                    max_sk = traj_out['skill'][r] 
            best_skills.append(max_sk)

        self.agent.ft_skills = [dict(skill=sk) for sk in best_skills]

    def run_metas(self, metas, **kwargs):
        skill_lst = [m['skill'] for m in metas]
        return self.run_skills(skill_lst, **kwargs)

    def run_skills(self, skill_lst, tl_lst, extr_reward_lst=None, **kwargs):
        out = dict(time_step=self.eval_env.reset(), step=0)
        final_out = dict()
        if 'video' in kwargs and kwargs['video'] is True:
            kwargs['init_video'] = False 
            self.video_recorder.init(self.eval_env, enabled=True)
        for idx, (sk, tl) in enumerate(zip(skill_lst, tl_lst)):
            if extr_reward_lst is not None: 
                self.agent.extr_reward_id = extr_reward_lst[idx]
            out = self.run_skill(sk, num_timesteps=tl, time_step=out['time_step'], **kwargs)
            for key, val in out.items():
                if key not in final_out:
                    final_out[key] = [val]
                else:
                    final_out[key].append(val)
        if 'video' in kwargs and kwargs['video'] is True:
            video_name = f"{self.global_frame}.mp4"
            self.video_recorder.save(video_name)
        return final_out 

    def run_skill(self, skill, video=False, init_video=True, video_name=None, num_timesteps=None, time_step=None, use_handcrafted=False, step=0):
        step, total_reward = step, 0
        ep_obs, ep_action = [], []
        meta = self.agent.init_meta(skill)

        if num_timesteps is None:
            time_step = self.eval_env.reset()
            ep_obs.append(time_step.observation)

        if init_video:
            self.video_recorder.init(self.eval_env, enabled=video)
        while not time_step.last() and (num_timesteps is None or step < num_timesteps):
            prev_time_step = time_step
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=True)
            if "fetch" in self.cfg.task:
                time_step = self.eval_env.step(action, make_video=video)
            else:
                time_step = self.eval_env.step(action)
            self.video_recorder.record(self.eval_env)
            ep_obs.append(time_step.observation)
            ep_action.append(action)
            prev_obs, _ = self.agent.process_observation(torch.from_numpy(prev_time_step.observation).unsqueeze(0).to(self.cfg.device))
            curr_obs, _ = self.agent.process_observation(torch.from_numpy(time_step.observation).unsqueeze(0).to(self.cfg.device))
            if use_handcrafted:
                total_reward += self.agent.get_extr_rew()(prev_obs, curr_obs, None)
            else:
                total_reward += self.get_extr_reward(step, time_step.reward, prev_obs, curr_obs, None)
            step += 1
        if video:
            if video_name is None:
                video_name = f"{self.global_frame}.mp4"
            self.video_recorder.save(video_name)

        out = dict(ep_obs=ep_obs, ep_action=ep_action, reward=total_reward, step=step, time_step=time_step)
        if "skill" in meta:
            out['skill'] = meta['skill']
        return out

    def save_plots(self):
        # run trajectories
        episode = 0
        eval_until_episode = utils.Until(self.cfg.num_plot)
        all_ep_obs = []
        skill_lst = []
        tl_lst = [self.agent.skill_duration for _ in range(self.n_rewards)]
        while eval_until_episode(episode):
            if episode == 0:
                traj_out = self.run_metas(self.agent.ft_skills, tl_lst=tl_lst, video=True)
            else: 
                traj_out = self.run_skills([None for _ in range(self.n_rewards)], tl_lst)
            ep_obs = np.array(traj_out['ep_obs'])
            ep_obs = ep_obs.reshape(-1, ep_obs.shape[-1])
            all_ep_obs.append(ep_obs)
            if "skill" in traj_out:
                # for non-ddpg
                skill_lst.append(traj_out['skill'])
            episode += 1
        self.plot_traj(skill_lst, all_ep_obs, str(self.global_frame), title="ft skill (maroon) + others", goal=self.agent.get_goal())

    def plot_traj(self, skills, all_traj, name, style_map=None, title=None, goal=None, labels=None):
        if style_map is None:
            color_map = ['maroon', 'orangered', 'darkorange', 'gold', 'yellow', 
                        'lawngreen', 'aquamarine', 'deepskyblue', 'blue', 'darkblue',
                        'slateblue', 'rebeccapurple', 'indigo', 'blueviolet', 'mediumorchid',
                        'violet', 'magenta', 'orchid', 'deeppink', 'pink',
                        'gainsboro', 'silver', 'darkgray', 'dimgray', 'black']
            style_map = color_map
            # for line_style in ['-', '--', '-.', ':']:
            #   style_map += [(color, line_style) for color in color_map] 

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
                # uneven padding
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
        if len(ep_ob) != self.n_rewards:
            ep_ob = [ep_ob]

        for r in range(self.n_rewards):
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


@hydra.main(config_path='.', config_name='finetune')
def main(cfg):
    from finetune import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
