import torch
import numpy as np

import utils
import dmc

class IRM:
    def __init__(self, name, extr_reward, pearson_setting, canonical_setting, 
				pearson_setting_sequencing, canonical_setting_sequencing, 
				pearson_samples, canonical_samples, noise, learnable_reward_scale, 
                matching_metric, use_handcrafted_task_reward, hrl,
                device, discount, print_every_step, eval_extr_every_step):
        self.extr_reward = extr_reward
        self.pearson_setting = pearson_setting
        self.canonical_setting = canonical_setting
        self.pearson_setting_sequencing = pearson_setting_sequencing
        self.canonical_setting_sequencing = canonical_setting_sequencing
        self.pearson_samples = pearson_samples
        self.canonical_samples = canonical_samples
        self.noise = noise
        self.learnable_reward_scale = learnable_reward_scale
        self.matching_metric = matching_metric
        self.use_handcrafted_task_reward = use_handcrafted_task_reward
        self.hrl = hrl
        self.device = device
        self.discount = discount
        self.print_every_step = utils.Every(print_every_step)
        self.eval_extr_every_step = utils.Every(eval_extr_every_step)

    def init_IRM(self, env, agent, task, video_recorder, logger):
        self.eval_env = env 
        self.agent = agent 
        self.task = task 
        self.video_recorder = video_recorder
        self.logger = logger
        
        self.timer = utils.Timer()
        self.global_step = 0
        self.process_env_info()

        self.n_rewards = len(self.extr_reward)
        self.agent.skill_duration = self.eval_env.step_limit // self.n_rewards
        self.skill_based_agent = self.agent.name in ["cic", "dads"]

    def run_skill_selection(self):
        if self.skill_based_agent:
            self.agent.ft_skills = self.run_skill_selection_method()
        else:
            self.agent.ft_skills = [dict(skill=None)]

    def run_skill_selection_method(self):
        # should be implemented by child classes
        raise NotImplementedError

    def get_ft_meta(self, episode_step):
        if len(self.extr_reward) == 1:
            return self.agent.ft_skills[0]
        else:
            ind = min(episode_step // self.agent.skill_duration, len(self.extr_reward)-1)
            return self.agent.ft_skills[ind]

    def get_extr_reward_function(self, episode_step=None, extr_reward_id=None):
        # either specify episode step or desired reward id
        assert not (episode_step is not None and extr_reward_id is not None)

        if extr_reward_id is None:
            if episode_step is None:
                extr_reward_id = 0
            else:
                extr_reward_id =  min(episode_step // self.agent.skill_duration, len(self.extr_reward)-1)
        extr_reward_name = self.extr_reward[extr_reward_id]
        return utils.get_extr_reward_function(extr_reward_name, self.device)

    def recompute_reward(self, prev_time_step, time_step, step, use_handcrafted=False):
        """
        use_handcrafted should be true during skill selection process, 
        where we assume access only to a handcrafted reward, even when running environment rollouts.
        This is for fair comparison between interaction-free methods, which rely on handcrafted rewards, 
        and interaction-based methods.
        """
        if self.use_handcrafted_task_reward or "goal" in self.extr_reward[0] or use_handcrafted:
            proc_prev_obs = self.agent.process_observation(torch.from_numpy(prev_time_step.observation).unsqueeze(0))[0][0].numpy()
            proc_obs = self.agent.process_observation(torch.from_numpy(time_step.observation).unsqueeze(0))[0][0].numpy()
            handcrafted_reward = self.get_extr_reward_function(step)(proc_prev_obs, proc_obs, None)
            time_step = dmc.update_time_step_reward(time_step, handcrafted_reward) # only important for replay buffer
            return time_step
        else:
            return time_step

    def compute_l1_loss(self, intr_rew, extr_rew, skill_th, pearson_obs_th=None, pearson_next_obs_th=None):
        pearson_obs, pearson_next_obs = self.get_pearson_samples(pearson_obs_th, pearson_next_obs_th)
        skill_pearson = skill_th.expand((obs.shape[0], -1))
        
        reward1 = intr_rew(obs, next_obs, skill_pearson).unsqueeze(1)
        reward2 = extr_rew(obs, next_obs, skill_pearson)
        if self.learnable_reward_scale:
            reward1 = self.alpha_reward_scale * reward1
        return F.l1_loss(reward1, reward2)

    def compute_l2_loss(self, intr_rew, extr_rew, skill_th, pearson_obs_th=None, pearson_next_obs_th=None):
        pearson_obs, pearson_next_obs = self.get_pearson_samples(pearson_obs_th, pearson_next_obs_th)
        skill_pearson = skill_th.expand((obs.shape[0], -1))
        
        reward1 = intr_rew(obs, next_obs, skill_pearson).unsqueeze(1)
        reward2 = extr_rew(obs, next_obs, skill_pearson)
        if self.learnable_reward_scale:
            reward1 = self.alpha_reward_scale * reward1
        return F.mse_loss(reward1, reward2)

    def compute_epic_loss(self, intr_rew, extr_rew, skill_th, pearson_obs_th=None, pearson_next_obs_th=None):
        pearson_obs, pearson_next_obs = self.get_pearson_samples(pearson_obs_th, pearson_next_obs_th)
        
        skill_pearson = skill_th.expand((pearson_obs.shape[0], -1))

        reward1 = intr_rew(pearson_obs, pearson_next_obs, skill_pearson).unsqueeze(1)
        reward2 = extr_rew(pearson_obs, pearson_next_obs, skill_pearson)

        pearson_obs = torch.repeat_interleave(pearson_obs, self.canonical_samples, dim=0)
        pearson_next_obs = torch.repeat_interleave(pearson_next_obs, self.canonical_samples, dim=0)

        canonical_next_obs_samples = self.get_canonical_samples(pearson_obs)

        skill_canon = skill_th.expand((canonical_next_obs_samples.shape[0], -1))
        
        with torch.no_grad():
            reward1_1 = intr_rew(pearson_next_obs, canonical_next_obs_samples, skill_canon).unsqueeze(1)
            reward1_2 = intr_rew(pearson_obs, canonical_next_obs_samples, skill_canon).unsqueeze(1)
            reward1_1 = torch.mean(reward1_1.reshape((skill_pearson.shape[0], -1, 1)), axis=1)
            reward1_2 = torch.mean(reward1_2.reshape((skill_pearson.shape[0], -1, 1)), axis=1)

            reward2_1 = extr_rew(pearson_next_obs, canonical_next_obs_samples, skill_canon)
            reward2_2 = extr_rew(pearson_obs, canonical_next_obs_samples, skill_canon)
            reward2_1 = torch.mean(reward2_1.reshape((skill_pearson.shape[0], -1, 1)), axis=1)
            reward2_2 = torch.mean(reward2_2.reshape((skill_pearson.shape[0], -1, 1)), axis=1)

        canonical_reward1 = reward1 + self.discount * reward1_1 \
            - reward1_2 
        canonical_reward2 = reward2 + self.discount * reward2_1 \
            - reward2_2 

        return self.compute_pearson_distance(canonical_reward1, 
            canonical_reward2)

    def compute_pearson_distance(self, rew_1, rew_2):
        rew_1 = rew_1 - torch.mean(rew_1)
        rew_2 = rew_2 - torch.mean(rew_2)
        var_1 = rew_1**2
        var_2 = rew_2**2
        cov = torch.sum(rew_1 * rew_2)
        corr = cov / (torch.sqrt(torch.sum(var_1) * torch.sum(var_2)))
        if (corr > 1.0):
            return torch.tensor([1.0])
        return torch.sqrt(0.5 * (1 - corr))

    def get_canonical_samples(self, pearson_obs):
        if self.canonical_setting == "uniform":
            next_obs_samples = torch.rand_like(pearson_obs)
        elif self.canonical_setting == "full_rand":
            if isinstance(self.bounds['min'], torch.Tensor):
                min_b, max_b = self.agent.process_observation(self.bounds['min'])[0], self.agent.process_observation(self.bounds['max'])[0]
            else:
                min_b, max_b = self.bounds['min'], self.bounds['max']
            next_obs_samples = torch.rand_like(pearson_obs) * (max_b - min_b) + min_b
        elif self.canonical_setting == "gaussian":
            next_obs_samples = pearson_obs + self.noise * torch.randn_like(pearson_obs)
        elif "gaussian" in self.canonical_setting:
            _, noise = self.canonical_setting.split("_")
            next_obs_samples = pearson_obs + float(noise) * torch.randn_like(pearson_obs)
        else:
            raise NotImplementedError
        return next_obs_samples

    def get_pearson_samples(self, obs_th, next_obs_th):
        if self.pearson_setting in ["onpolicy", "prev_rollout", "prev_rollout_last"]:
            obs, next_obs = obs_th, next_obs_th
        else:
            obs_dim = self.agent.obs_dim - self.agent.skill_dim
            s1 = torch.rand(self.pearson_samples, obs_dim, device=self.device)
            s2 = torch.rand(self.pearson_samples, obs_dim, device=self.device)
            if self.pearson_setting == "full_rand":
                obs = s1 * (self.bounds['max'] - self.bounds['min']) + self.bounds['min']
                next_obs = s2 * (self.bounds['max'] - self.bounds['min']) + self.bounds['min']
                actions = next_obs - obs
            elif self.pearson_setting == "uniform":
                obs = s1
                next_obs = s2
            
            else:
                raise NotImplementedError
        obs = self.agent.process_observation(obs)[0]
        next_obs = self.agent.process_observation(next_obs)[0]
        return obs, next_obs

    def run_skills(self, skill_lst, tl_lst=[float('inf')], 
    				video=False, video_name=None,
    				use_handcrafted=False):

        traj_data = dict(time_step=self.eval_env.reset(), step=0)
        entire_traj_data = dict()

        if video:
            self.video_recorder.init(self.eval_env, enabled=True)

        for idx, sk in enumerate(skill_lst):
            traj_data = self._run_skill(sk, use_handcrafted=use_handcrafted,
            					num_timesteps=sum(tl_lst[:idx+1]), time_step=traj_data['time_step'], step=traj_data['step'],
            					video=video, video_name=video_name)
            for key, val in traj_data.items():
                if key not in entire_traj_data:
                    entire_traj_data[key] = [val]
                else:
                    entire_traj_data[key].append(val)

        if video:
            self.video_recorder.save(f"{video_name}.mp4")
        return entire_traj_data 

    def _run_skill(self, skill, use_handcrafted=False, 
    				num_timesteps=None, time_step=None, step=0,
    				video=False, video_name=None):
        """
        - use_handcrafted: use handcrafted reward (e.g. when running skill selection)
        - num_timesteps, time_step: run for `num_timesteps`, given the current env `time_step`
        - step: current environment step (used for determining which extr. reward for sequential rewards)
        - video, video_name: video args
        """

        step, total_reward = step, 0
        ep_obs, ep_action = [], []
        meta = self.agent.init_meta(skill)

        while not time_step.last() and (num_timesteps is None or step < num_timesteps):
            prev_time_step = time_step
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        step,
                                        eval_mode=True)
            if "fetch" in self.task:
                time_step = self.eval_env.step(action, make_video=video)
            else:
                time_step = self.eval_env.step(action)
            if video:
                self.video_recorder.record(self.eval_env)
            ep_obs.append(time_step.observation)
            ep_action.append(action)
            time_step = self.recompute_reward(prev_time_step, time_step, step, use_handcrafted)
            total_reward += time_step.reward
            step += 1
        if video:
            self.video_recorder.save(f"{video_name}.mp4")

        traj_data = dict(ep_obs=ep_obs, ep_action=ep_action, reward=total_reward, step=step, time_step=time_step)
        if self.hrl:
            traj_data['skill'] = self.agent.meta_action
        elif "skill" in meta:
            traj_data['skill'] = meta['skill']
        return traj_data

    def process_env_info(self):
        if "plane" in self.task:
            self.bounds = dict(min=-128, max=128)
        elif "fetch_push" in self.task:
            self.bounds = dict(min=0.5, max=1.7)
        elif "fetch_barrier" in self.task:
            min_th, max_th = torch.zeros(1, 28).to(self.device), torch.ones(1, 28).to(self.device)
            min_th[:, 3] = 0.96
            min_th[:, 4] = 0.5
            max_th[:, 3] = 1.4
            max_th[:, 4] = 1
            self.bounds = {'min': min_th, 'max': max_th}
        elif "fetch_reach" in self.task:
            self.bounds = dict(min=0.5, max=1.7)
        elif "jaco" in self.task:
            self.bounds = dict(min=-1, max=1)
        elif "walker" in self.task:
            self.bounds = dict(min=-1, max=1)
        elif "quadruped" in self.task:
            self.bounds = dict(min=-1, max=1)

    def get_goal(self):
        goals = []
        for rew in self.extr_reward:
            if rew == "goal_top_right":
                goals.append(np.array([128, 128]))
            elif rew == "goal_top_left":
                goals.append(np.array([-128, 128]))
            elif rew == "goal_1.4_0.8":
                goals.append(np.array([1.4, 0.8]))
            elif rew == "goal_1.2_1":
                goals.append(np.array([1.2, 1]))
            elif rew == "goal_1.3_0.9":
                goals.append(np.array([1.3, 0.9]))
            elif rew == "goal_-0.75_0.5":
                goals.append(np.array([-0.75, 0.5]))
            elif rew == "goal_-0.75_0.5_0.5":
                goals.append(np.array([-0.75, 0.5, 0.5]))
            elif rew == "goal_0.5_0.5_0.5":
                goals.append(np.array([0.5, 0.5, 0.5]))
            elif rew == "goal_0_0_0.5":
                goals.append(np.array([0, 0, 0.5]))
            elif rew == "goal_1_1.2_1":
                goals.append(np.array([1, 1.2, 1]))
            elif rew == "goal_1_1_0.5":
                goals.append(np.array([1, 1, 0.5]))
            elif rew == "goal_1_0.8_0.5":
                goals.append(np.array([1, 0.8, 0.5]))
            elif rew == "goal_barrier1":
                goals.append(np.array([1.31, 0.6]))
            elif rew == "goal_barrier2":
                goals.append(np.array([1.31, 0.9]))
            elif rew == "goal_barrier3":
                goals.append(np.array([1.16, 0.9]))
            elif rew == "goal_barrier4":
                goals.append(np.array([1.6, 0.4]))
            elif rew == "goal_tunnel1":
                goals.append(np.array([1.21, 0.65]))
            elif rew == "goal_tunnel2":
                goals.append(np.array([1.41, 0.65]))
            elif rew == "goal_tunnel3":
                goals.append(np.array([1.41, 0.95]))
            elif rew == "goal_reach1":
                goals.append(np.array([1.5, 0.3, 0.45]))
            elif rew in ["walker_right", "walker_left"]:
                goals.append(None)
            elif "reach" in rew:
                from utils import jaco_reach
                goals.append(jaco_reach(self.device, rew).target_pos)
            elif "jaco" in rew or "quadruped" in rew or "walker" in rew:
                goals.append(None)
            else:
                raise NotImplementedError
        return goals
            
