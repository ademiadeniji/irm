import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2


class DDPGAgent:
    def __init__(self, name, reward_free, obs_type, obs_shape, action_shape,
                 device, lr, feature_dim, hidden_dim, critic_target_tau,
                 num_expl_steps, stddev_schedule, nstep, extr_reward, extr_reward_seq,
                 batch_size, stddev_clip, init_critic, use_tb, use_wandb, noise, 
                 epic_gradient_descent_steps, z_lr, learnable_reward_scale, 
                 matching_metric, num_cem_iterations, num_cem_elites, num_cem_samples,
                 update_every_steps=None,
                 meta_dim=0):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.z_lr = z_lr
        self.learnable_reward_scale = learnable_reward_scale
        self.matching_metric = matching_metric
        self.epic_gradient_descent_steps = epic_gradient_descent_steps
        self.num_cem_iterations = num_cem_iterations
        self.num_cem_elites = num_cem_elites
        self.num_cem_samples = num_cem_samples

        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None

        self.batch_size = batch_size
        self.extr_reward = extr_reward
        self.extr_reward_seq = extr_reward_seq
        self.noise = noise

        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim

        if "walker" in self.obs_type:
            self.obs_dim -= 3 # remove global x, y, z from obs

        self.actor = Actor(obs_type, self.obs_dim, self.action_dim,
                           feature_dim, hidden_dim).to(device)

        self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

        self.discr_obs_dim = self.obs_dim 
        if self.obs_type in ["fetch_push_xy"]:
            self.discr_obs_dim = 2 
        elif self.obs_type in ["jaco_xyz", "walker_delta_xyz", "fetch_reach_xyz"]:
            self.discr_obs_dim = 3


    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        self.encoder.to(self.device)
        self.actor.to(self.device)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
            self.critic.to(self.device)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self, time_step=None):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def find_ft_meta(self, bounds=None):
        self.ft_skills = [dict(skill=None)]

    def get_ft_meta(self, step):
        return OrderedDict()

    def act(self, obs, meta, step, eval_mode):
        if "walker" in self.obs_type:
            obs = obs[3:]

        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def get_epic_obs(self, obs_th, next_obs_th, bounds):
        if self.pearson_setting in ["onpolicy", "prev_rollout", "prev_rollout_last"]:
            obs, next_obs = obs_th, next_obs_th
        else:
            obs_dim = self.obs_dim - self.skill_dim
            s1 = torch.rand(self.pearson_samples, obs_dim, device=self.device)
            s2 = torch.rand(self.pearson_samples, obs_dim, device=self.device)
            if self.pearson_setting == "full_rand":
                obs = s1 * (bounds['max'] - bounds['min']) + bounds['min']
                next_obs = s2 * (bounds['max'] - bounds['min']) + bounds['min']
                actions = next_obs - obs
            elif self.pearson_setting == "uniform":
                obs = s1
                next_obs = s2
            
            else:
                raise NotImplementedError
        obs = self.process_observation(obs)[0]
        next_obs = self.process_observation(next_obs)[0]
        return obs, next_obs

    def irm_random_search(self, bounds, obs_th=None, next_obs_th=None):
        min_sk, min_loss = None, float('inf')
        for i in range(self.num_epic_skill):
            if self.learnable_reward_scale:
                self.alpha_reward_scale = torch.rand((1), device=self.device) * 100
            skill = torch.rand((self.skill_dim), device=self.device)
            with torch.no_grad():
                if self.matching_metric == "epic":
                    loss = self.compute_epic_loss_ft(skill, bounds=bounds,
                                        obs_th=obs_th, next_obs_th=next_obs_th)
                elif self.matching_metric == 'l2':
                    loss = self.compute_l2_loss_ft(skill, bounds=bounds,
                                        obs_th=obs_th, next_obs_th=next_obs_th)
                elif self.matching_metric == 'l1':
                    loss = self.compute_l1_loss_ft(skill, bounds=bounds,
                                        obs_th=obs_th, next_obs_th=next_obs_th)
                else:
                    raise ValueError("Invalid reward matching metric specified")
                if (loss < min_loss):
                    min_loss = loss 
                    min_sk = skill 
        return min_sk

    def irm_gradient_descent(self, bounds):
        z = torch.full([self.skill_dim], 0.5, requires_grad=True, device=self.device) 
        self.z_optimizer = torch.optim.Adam([z], lr=self.z_lr)
        if self.learnable_reward_scale:
            self.alpha_reward_scale = torch.ones(1, requires_grad=True, device=self.device)
            self.alpha_reward_scale_optimizer = torch.optim.Adam([self.alpha_reward_scale], lr=self.z_lr)
        for step in range(self.epic_gradient_descent_steps):
            if self.matching_metric == "epic":
                loss = self.compute_epic_loss_ft(z, bounds=bounds)
            elif self.matching_metric == 'l2':
                loss = self.compute_l2_loss_ft(z, bounds=bounds)
            elif self.matching_metric == 'l1':
                loss = self.compute_l1_loss_ft(z, bounds=bounds)
            else:
                raise ValueError("Invalid reward matching metric specified")
            self.z_optimizer.zero_grad()
            if self.learnable_reward_scale:
                self.alpha_reward_scale_optimizer.zero_grad()
            loss.backward()
            self.z_optimizer.step()
            if self.learnable_reward_scale:
                self.alpha_reward_scale_optimizer.zero_grad()
        return z
    
    def irm_cem(self, bounds):
        if self.learnable_reward_scale:
            with torch.no_grad():
                mean = torch.zeros(self.skill_dim+1, requires_grad=False, device=self.device) + 0.5
                std = torch.zeros(self.skill_dim+1, requires_grad=False, device=self.device) + 0.25
                for iter in range(self.num_cem_iterations):
                    samples = torch.normal(mean.repeat(self.num_cem_samples, 1), std.repeat(self.num_cem_samples, 1))
                    if self.learnable_reward_scale:
                        skill_samples = samples[:, :-1]
                        scale_samples = samples[:, -1]
                    losses = []
                    for sk in range(self.num_cem_samples):
                        if self.learnable_reward_scale:
                            self.alpha_reward_scale = scale_samples[sk]
                        if self.matching_metric == "epic":
                            losses.append(self.compute_epic_loss_ft(skill_samples[sk], bounds=bounds).item())
                        elif self.matching_metric == 'l2':
                            losses.append(self.compute_l2_loss_ft(skill_samples[sk], bounds=bounds).item())
                        elif self.matching_metric == 'l1':
                            losses.append(self.compute_l1_loss_ft(skill_samples[sk], bounds=bounds).item())
                        else:
                            raise ValueError("Invalid reward matching metric specified")
                    sorted_losses = np.argsort(losses)
                    elite_idxs = sorted_losses[:self.num_cem_elites]
                    elites = samples[elite_idxs]
                    mean = torch.mean(elites, dim=0)
                    std = torch.std(elites, dim=0)
            return elites[0][:-1]
        else:
            print(self.num_cem_iterations)
            with torch.no_grad():
                mean = torch.zeros(self.skill_dim, requires_grad=False, device=self.device) + 0.5
                std = torch.zeros(self.skill_dim, requires_grad=False, device=self.device) + 0.25
                for iter in range(self.num_cem_iterations):
                    samples = torch.normal(mean.repeat(self.num_cem_samples, 1), std.repeat(self.num_cem_samples, 1))
                    losses = []
                    for sk in range(self.num_cem_samples):
                        if self.matching_metric == "epic":
                            losses.append(self.compute_epic_loss_ft(samples[sk], bounds=bounds).item())
                        elif self.matching_metric == 'l2':
                            losses.append(self.compute_l2_loss_ft(samples[sk], bounds=bounds).item())
                        elif self.matching_metric == 'l1':
                            losses.append(self.compute_l1_loss_ft(samples[sk], bounds=bounds).item())
                        else:
                            raise ValueError("Invalid reward matching metric specified")
                    sorted_losses = np.argsort(losses)
                    elite_idxs = sorted_losses[:self.num_cem_elites]
                    elites = samples[elite_idxs]
                    mean = torch.mean(elites, dim=0)
                    std = torch.std(elites, dim=0)
            return elites[0]

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

    def get_extr_rew(self, episode_step=None):
        if episode_step is not None:
            self.extr_reward_id =  min(episode_step // self.skill_duration, len(self.extr_reward_seq)-1)
        if len(self.extr_reward_seq) > 0:
            self.extr_reward = self.extr_reward_seq[self.extr_reward_id]
        return utils.get_extr_rew(self.extr_reward, self.device)

    def get_goal(self):
        if len(self.extr_reward_seq) > 0:
            extr_rewards = self.extr_reward_seq
        else: 
            extr_rewards = [self.extr_reward]
        goals = []
        for extr_reward in extr_rewards:
            if extr_reward == "goal_top_right":
                goals.append(np.array([128, 128]))
            elif extr_reward == "goal_top_left":
                goals.append(np.array([-128, 128]))
            elif extr_reward == "goal_0.5_0.5_0.5":
                goals.append(np.array([0.5, 0.5, 0.5]))
            elif extr_reward == "goal_1_1.2_1":
                goals.append(np.array([1, 1.2, 1]))
            elif extr_reward == "goal_1.5_0.3_0.45":
                goals.append(np.array([1.5, 0.3, 0.45]))
            elif extr_reward == "goal_barrier1":
                goals.append(np.array([1.31, 0.6]))
            elif extr_reward == "goal_barrier2":
                goals.append(np.array([1.31, 0.9]))
            elif extr_reward == "goal_barrier3":
                goals.append(np.array([1.16, 0.9]))
            elif extr_reward in ["walker_right", "walker_left"]:
                goals.append(None)
            elif "reach" in extr_reward:
                from utils import jaco_reach
                goals.append(jaco_reach(self.device, extr_reward).target_pos)
            elif "jaco" in extr_reward or "quadruped" in extr_reward or "walker" in extr_reward:
                goals.append(None)
            else:
                raise NotImplementedError
        return goals

    def process_observation(self, obs):
        aug_obs = self.aug_and_encode(obs)
        if self.obs_type == "jaco_xyz":
            proc_obs = aug_obs[:, 30:33]
        elif self.obs_type == "fetch_push_xy":
            proc_obs = aug_obs[:, 3:5]
        elif self.obs_type == "fetch_reach_xyz":
            proc_obs = aug_obs[:, :3]
        elif self.obs_type == "quad_velocity":
            proc_obs = aug_obs[:, 32:35]
        elif self.obs_type == "walker_delta_xyz":
            proc_obs = aug_obs[:, :3]
            aug_obs = aug_obs[:, 3:]
        else:
            proc_obs = aug_obs
        return proc_obs, aug_obs

    def compute_l1_loss_ft(self, skill1_th, obs_th=None, next_obs_th=None, skill2_th=None, bounds=None):
        f_rew1 = self.compute_epic_rew
        f_rew2 = self.get_extr_rew()

        obs, next_obs = self.get_epic_obs(obs_th, next_obs_th, bounds)
        skill1_pearson = skill1_th.unsqueeze(0).repeat((obs.shape[0], 1))
        if skill2_th is None:
            skill2_pearson = skill1_pearson 
        else:
            skill2_pearson = skill2_th.unsqueeze(0).repeat((obs.shape[0], 1)) 
       
        reward1 = f_rew1(obs, next_obs, skill1_pearson).unsqueeze(1)
        reward2 = f_rew2(obs, next_obs, skill2_pearson)
        if self.learnable_reward_scale:
            reward1 = self.alpha_reward_scale * reward1
        return F.l1_loss(reward1, reward2)

    def compute_l2_loss_ft(self, skill1_th, obs_th=None, next_obs_th=None, skill2_th=None, bounds=None):
        f_rew1 = self.compute_epic_rew
        f_rew2 = self.get_extr_rew()

        obs, next_obs = self.get_epic_obs(obs_th, next_obs_th, bounds)
        skill1_pearson = skill1_th.unsqueeze(0).repeat((obs.shape[0], 1))
        if skill2_th is None:
            skill2_pearson = skill1_pearson 
        else:
            skill2_pearson = skill2_th.unsqueeze(0).repeat((obs.shape[0], 1)) 
       
        reward1 = f_rew1(obs, next_obs, skill1_pearson).unsqueeze(1)
        reward2 = f_rew2(obs, next_obs, skill2_pearson)
        if self.learnable_reward_scale:
            reward1 = self.alpha_reward_scale * reward1
        return F.mse_loss(reward1, reward2)

    def compute_epic_loss_ft(self, skill1_th, obs_th=None, next_obs_th=None, skill2_th=None, bounds=None):
        f_rew1 = self.compute_epic_rew
        f_rew2 = self.get_extr_rew()

        obs, next_obs = self.get_epic_obs(obs_th, next_obs_th, bounds)
        skill1_pearson = skill1_th.unsqueeze(0).repeat((obs.shape[0], 1))
        if skill2_th is None:
            skill2_pearson = skill1_pearson 
        else:
            skill2_pearson = skill2_th.unsqueeze(0).repeat((obs.shape[0], 1)) 
       
        reward1 = f_rew1(obs, next_obs, skill1_pearson).unsqueeze(1)
        reward2 = f_rew2(obs, next_obs, skill2_pearson)

        obs = torch.repeat_interleave(obs, self.canonical_samples, dim=0)
        next_obs = torch.repeat_interleave(next_obs, self.canonical_samples, dim=0)

        if self.canonical_setting == "uniform":
            next_obs_samples = torch.rand_like(obs)
        elif self.canonical_setting == "full_rand":
            if isinstance(bounds['min'], torch.Tensor):
                min_b, max_b = self.process_observation(bounds['min'])[0], self.process_observation(bounds['max'])[0]
            else:
                min_b, max_b = bounds['min'], bounds['max']
            next_obs_samples = torch.rand_like(obs) * (max_b - min_b) + min_b
        elif self.canonical_setting == "gaussian":
            next_obs_samples = obs + self.noise * torch.randn_like(obs)
        elif "gaussian" in self.canonical_setting:
            _, noise = self.canonical_setting.split("_")
            next_obs_samples = obs + float(noise) * torch.randn_like(obs)
        else:
            raise NotImplementedError

        skill1_canon = skill1_th.repeat((next_obs_samples.shape[0], 1))
        if skill2_th is None:
            skill2_canon = skill1_canon 
        else:
            skill2_canon = skill2_th.repeat((next_obs_samples.shape[0], 1))
        with torch.no_grad():
            reward1_1 = f_rew1(next_obs, next_obs_samples, skill1_canon).unsqueeze(1)
            reward1_2 = f_rew1(obs, next_obs_samples, skill1_canon).unsqueeze(1)
            reward1_1 = torch.mean(reward1_1.reshape((skill1_pearson.shape[0], -1, 1)), axis=1)
            reward1_2 = torch.mean(reward1_2.reshape((skill1_pearson.shape[0], -1, 1)), axis=1)

            reward2_1 = f_rew2(next_obs, next_obs_samples, skill2_canon)
            reward2_2 = f_rew2(obs, next_obs_samples, skill2_canon)
            reward2_1 = torch.mean(reward2_1.reshape((skill1_pearson.shape[0], -1, 1)), axis=1)
            reward2_2 = torch.mean(reward2_2.reshape((skill1_pearson.shape[0], -1, 1)), axis=1)

        canonical_reward1 = reward1 + self.discount * reward1_1 \
            - reward1_2 
        canonical_reward2 = reward2 + self.discount * reward2_1 \
            - reward2_2 

        return self.compute_pearson_distance(canonical_reward1, 
            canonical_reward2)

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, replay_id = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
