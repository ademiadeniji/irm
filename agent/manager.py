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

class ManagerAgent:
    def __init__(self, worker_agent, obs_shape, name='manager', lr=1e-4):

        self.name = name
        self.worker_agent = worker_agent
        self.lr = lr
        self.meta_action = None
        self.obs_shape = obs_shape
        self.skill_duration = worker_agent.skill_duration

        # models
        if worker_agent.obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(worker_agent.device)
            self.obs_dim = self.encoder.repr_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0]
        
        self.actor = Actor(worker_agent.obs_type, self.obs_dim, worker_agent.skill_dim,
                           worker_agent.feature_dim, worker_agent.hidden_dim).to(worker_agent.device)

        self.critic = Critic(worker_agent.obs_type, self.obs_dim, worker_agent.skill_dim,
                             worker_agent.feature_dim, worker_agent.hidden_dim).to(worker_agent.device)
        self.critic_target = Critic(worker_agent.obs_type, self.obs_dim, worker_agent.skill_dim,
                                    worker_agent.feature_dim, worker_agent.hidden_dim).to(worker_agent.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if worker_agent.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        self.encoder.to(self.worker_agent.device)
        self.actor.to(self.worker_agent.device)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
            self.critic.to(self.worker_agent.device)
    
    def get_meta_specs(self):
        return self.worker_agent.get_meta_specs()
    
    def init_meta(self, skill=None):
        return None
    
    def find_ft_meta(self, bounds=None):
        return None
    
    def get_ft_meta(self, episode_step=None):
        return dict(skill=np.zeros(self.worker_agent.skill_dim).astype(np.float32))
    
    def update_meta(self, meta, global_step, time_step):
        return None

    def compute_inner_product(self, obs, next_obs, skill):
        return self.worker_agent.compute_inner_product(obs, next_obs, skill)

    def compute_intr_reward(self, obs, skill, next_obs, step, keep_grad=False):
        return self.worker_agent.compute_intr_reward(obs, skill, next_obs, step, keep_grad)
    
    def process_observation(self, obs):
        return self.worker_agent.process_observation(obs) 

    def get_extr_rew(self, episode_step=None):
        return self.worker_agent.get_extr_rew(episode_step)
    
    def get_goal(self):
        return self.worker_agent.get_goal()

    def act(self, obs, meta, step, eval_mode):
        if step % self.skill_duration == 0:
            obs_th = torch.as_tensor(obs, device=self.worker_agent.device).unsqueeze(0)
            inpt = self.encoder(obs_th)
            stddev = utils.schedule(self.worker_agent.stddev_schedule, step)
            dist = self.actor(inpt, stddev)
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
                if step < self.worker_agent.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
            self.meta_action = dict(skill=action.squeeze(0).detach().cpu().numpy())
        return self.worker_agent.act(obs, self.meta_action, step, eval_mode)

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.worker_agent.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.worker_agent.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.worker_agent.use_tb or self.worker_agent.use_wandb:
            metrics['manager_critic_target_q'] = target_Q.mean().item()
            metrics['manager_critic_q1'] = Q1.mean().item()
            metrics['manager_critic_q2'] = Q2.mean().item()
            metrics['manager_critic_loss'] = critic_loss.item()

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

        stddev = utils.schedule(self.worker_agent.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.worker_agent.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.worker_agent.use_tb or self.worker_agent.use_wandb:
            metrics['manager_actor_loss'] = actor_loss.item()
            metrics['manager_actor_logprob'] = log_prob.mean().item()
            metrics['manager_actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)
    
    def update(self, replay_iter, step, dem=None):
        metrics = dict()

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill, replay_id = utils.to_torch(
            batch, self.worker_agent.device)

        aug_obs = self.aug_and_encode(obs)
        with torch.no_grad():
            aug_next_obs = self.aug_and_encode(next_obs)
        
        reward = extr_reward

        if self.worker_agent.use_tb or self.worker_agent.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        if not self.worker_agent.freeze_rl:

            # update manager RL
            # update critic
            metrics.update(
                self.update_critic(aug_obs, skill, reward, discount, aug_next_obs, step))

            # update actor
            metrics.update(self.update_actor(aug_obs, step))

            # update critic target
            utils.soft_update_params(self.critic, self.critic_target,
                                    self.worker_agent.critic_target_tau)

            # update worker RL
            # extend observations with skill
            worker_obs = torch.cat([aug_obs, skill], dim=1)
            worker_next_obs = torch.cat([aug_next_obs, skill], dim=1)

            # update critic
            metrics.update(
                self.worker_agent.update_critic(worker_obs, action, reward, discount, worker_next_obs, step))

            # update actor
            metrics.update(self.worker_agent.update_actor(worker_obs, step))

            # update critic target
            utils.soft_update_params(self.worker_agent.critic, self.worker_agent.critic_target,
                                    self.worker_agent.critic_target_tau)

        return metrics
