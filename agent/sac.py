"""
Adapted from https://github.com/denisyarats/pytorch_sac
"""
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import math
import utils
from collections import OrderedDict

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = MLP(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

class SACAgent:
    """SAC algorithm."""
    def __init__(self, name, reward_free, obs_type, obs_shape, action_shape,
                 device, lr, actor_lr, critic_lr, feature_dim, hidden_dim, critic_target_tau,
                 num_expl_steps, stddev_schedule, nstep, num_rl_updates, epic_gradient_descent_steps, z_lr,
                 learnable_reward_scale, num_cem_iterations, num_cem_elites, num_cem_samples,
                 batch_size, stddev_clip, init_critic, use_tb, use_wandb,
                 matching_metric,
                 meta_dim=0, learnable_temperature=True, init_temperature=0.1,
                 actor_update_frequency=1, critic_target_update_frequency=2):
        super().__init__()
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None
        self.batch_size = batch_size
        self._num_rl_updates = num_rl_updates

        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature

        self.epic_gradient_descent_steps = epic_gradient_descent_steps
        self.z_lr = z_lr
        self.learnable_reward_scale = learnable_reward_scale
        self.num_cem_iterations = num_cem_iterations
        self.num_cem_elites = num_cem_elites 
        self.num_cem_samples = num_cem_samples
        self.matching_metric = matching_metric

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

        # set hidden_depth = 2 for actor, critic
        self.actor = DiagGaussianActor(self.obs_dim, self.action_dim,
                           hidden_dim, 2, [-5,2]).to(device)

        self.critic = DoubleQCritic(self.obs_dim, self.action_dim,
                             hidden_dim, 2).to(device)
        self.critic_target = DoubleQCritic(self.obs_dim, self.action_dim,
                                    hidden_dim, 2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -self.action_dim

        # optimizers
        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=lr)
        else:
            self.encoder_opt = None
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=[0.9, 0.999])

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=[0.9, 0.999])

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=lr,
                                                    betas=[0.9, 0.999])

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
        if self.init_critic:
            utils.hard_update_params(other.critic.Q1, self.critic.Q1)
            utils.hard_update_params(other.critic.Q2, self.critic.Q2)

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

    def get_extr_rew(self, episode_step=None):
        if episode_step is not None:
            self.extr_reward_id =  min(episode_step // self.skill_duration, len(self.extr_reward_seq)-1)
        if len(self.extr_reward_seq) > 0:
            self.extr_reward = self.extr_reward_seq[self.extr_reward_id]
        return utils.get_extr_rew(self.extr_reward, self.device)

    def get_goal(self):
        if len(self.extr_reward_seq) > 0:
            # self.extr_reward = self.extr_reward_seq[self.extr_reward_id]
            extr_rewards = self.extr_reward_seq
        else: 
            extr_rewards = [self.extr_reward]
        goals = []
        for extr_reward in extr_rewards:
            if extr_reward == "goal_top_right":
                goals.append(np.array([128, 128]))
            elif extr_reward == "goal_top_left":
                goals.append(np.array([-128, 128]))
            elif extr_reward == "goal_1.4_0.8":
                goals.append(np.array([1.4, 0.8]))
            elif extr_reward == "goal_1.2_1":
                goals.append(np.array([1.2, 1]))
            elif extr_reward == "goal_1.3_0.9":
                goals.append(np.array([1.3, 0.9]))
            elif extr_reward == "goal_-0.75_0.5":
                goals.append(np.array([-0.75, 0.5]))
            elif extr_reward == "goal_-0.75_0.5_0.5":
                goals.append(np.array([-0.75, 0.5, 0.5]))
            elif extr_reward == "goal_0.5_0.5_0.5":
                goals.append(np.array([0.5, 0.5, 0.5]))
            elif extr_reward == "goal_0_0_0.5":
                goals.append(np.array([0, 0, 0.5]))
            elif extr_reward == "goal_1_1.2_1":
                goals.append(np.array([1, 1.2, 1]))
            elif extr_reward == "goal_1_1_0.5":
                goals.append(np.array([1, 1, 0.5]))
            elif extr_reward == "goal_1_0.8_0.5":
                goals.append(np.array([1, 0.8, 0.5]))
            elif extr_reward == "goal_barrier1":
                goals.append(np.array([1.31, 0.6]))
            elif extr_reward == "goal_barrier2":
                goals.append(np.array([1.31, 0.9]))
            elif extr_reward == "goal_barrier3":
                goals.append(np.array([1.16, 0.9]))
            elif extr_reward == "goal_reach1":
                goals.append(np.array([1.5, 0.3, 0.45]))
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


    @property
    def alpha(self):
        return self.log_alpha.exp()

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
        dist = self.actor(inpt)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict() 
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = current_Q1.mean().item()
            metrics['critic_q2'] = current_Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics 

    def update_actor(self, obs, step):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['target_entropy'] = self.target_entropy
            metrics['mu'] = torch.abs(self.actor.outputs['mu']).mean().item()
            metrics['std'] = self.actor.outputs['std'].mean().item()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            if self.use_tb or self.use_wandb:
                metrics['alpha_loss'] = alpha_loss.item()
                metrics['alpha_value'] = self.alpha
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        return metrics

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


    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def process_observation(self, obs):
        aug_obs = self.aug_and_encode(obs)
        if self.obs_type == "walker_delta_xyz":
            proc_obs = aug_obs[:, :3]
            aug_obs = aug_obs[:, 3:]
        elif self.obs_type == "fetch_push_xy":
            proc_obs = aug_obs[:, 3:5]
        elif self.obs_type == "fetch_reach_xyz":
            proc_obs = aug_obs[:, :3]
        else:
            proc_obs = aug_obs
        return proc_obs, aug_obs

    def update(self, replay_buffer, logger, step):
        metrics = dict()

        if self % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                           logger, step)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic, critic target
        if step % self.critic_target_update_frequency == 0:
            metrics.update(
                self.update_critic(obs, action, reward, discount, next_obs, step))
            utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        # update actor
        if step % self.actor_update_frequency == 0:
            metrics.update(self.update_actor(obs.detach(), step))

        return metrics