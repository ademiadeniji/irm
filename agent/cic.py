import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

import utils

from agent.ddpg import DDPGAgent

import torch 
import numpy as np


class CIC(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, project_skill, obs_type='states', p=0, use_batchnorm=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.obs_type = obs_type
        self.use_batchnorm = use_batchnorm

        if self.obs_type in ["jaco_xyz", "fetch_reach_xyz"]:
            self.state_net = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), 
                                            nn.Dropout(p=p),
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p),
                                            nn.Linear(hidden_dim, self.skill_dim))

            self.next_state_net = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), 
                                            nn.Dropout(p=p),
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                            nn.Dropout(p=p),
                                        nn.Linear(hidden_dim, self.skill_dim)) # trying out self.skill_dim
            self.pred_net = nn.Sequential(nn.Linear(2*self.skill_dim, hidden_dim), nn.ReLU(), 
                                        nn.Dropout(p=p),
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(p=p), 
                                        nn.Linear(hidden_dim, self.skill_dim))
            bnorm_dim = 3
        elif self.obs_type in ["fetch_push_xy"]:
            self.state_net = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), 
                                            nn.Dropout(p=p),
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p),
                                            nn.Linear(hidden_dim, self.skill_dim))

            self.next_state_net = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), 
                                            nn.Dropout(p=p),
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                            nn.Dropout(p=p),
                                        nn.Linear(hidden_dim, self.skill_dim)) # trying out self.skill_dim
            self.pred_net = nn.Sequential(nn.Linear(2*self.skill_dim, hidden_dim), nn.ReLU(), 
                                        nn.Dropout(p=p),
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(p=p), 
                                        nn.Linear(hidden_dim, self.skill_dim))
            bnorm_dim = 2
        elif self.obs_type == "states":
            self.state_net = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p), 
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                            nn.Dropout(p=p),
                                            nn.Linear(hidden_dim, self.skill_dim))

            self.next_state_net = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p), 
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                            nn.Dropout(p=p),
                                        nn.Linear(hidden_dim, self.skill_dim))
        
            self.pred_net = nn.Sequential(nn.Linear(2 * self.skill_dim, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p), 
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p), 
                                            nn.Linear(hidden_dim, self.skill_dim))
            bnorm_dim = self.obs_dim
        elif self.obs_type == "quad_velocity":
            self.state_net = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p), 
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                            nn.Dropout(p=p),
                                            nn.Linear(hidden_dim, self.skill_dim))
        
            self.pred_net = nn.Sequential(nn.Linear(self.skill_dim, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p), 
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p), 
                                            nn.Linear(hidden_dim, self.skill_dim))
            bnorm_dim = 3
        elif self.obs_type in ["walker_delta_xyz"]:
            self.state_net = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p), 
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                            nn.Dropout(p=p),
                                            nn.Linear(hidden_dim, self.skill_dim))
        
            self.pred_net = nn.Sequential(nn.Linear(self.skill_dim, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p), 
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                            nn.Dropout(p=p), 
                                            nn.Linear(hidden_dim, self.skill_dim))
            bnorm_dim = 3
        else:
            raise NotImplementedError

        if self.use_batchnorm:
            self.preprocess_state = nn.BatchNorm1d(bnorm_dim, affine=False)

        if project_skill:
            self.skill_net = nn.Sequential(nn.Linear(self.skill_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                            nn.Linear(hidden_dim, self.skill_dim))
        else:
            self.skill_net = nn.Identity()  
   
        self.apply(utils.weight_init)

    def forward(self,state,next_state,skill):
        assert len(state.size()) == len(next_state.size())
        if "delta" in self.obs_type:
            if self.use_batchnorm:
                delta = self.preprocess_state(next_state - state)
            else:
                delta = next_state - state
            state = self.state_net(delta)
            query = self.pred_net(state)
            key = self.skill_net(skill) 
        elif self.obs_type == "quad_velocity":
            if self.use_batchnorm:
                next_state = self.preprocess_state(next_state)
            next_state = self.state_net(next_state)
            query = self.pred_net(next_state)
            key = self.skill_net(skill)
        else:
            if self.use_batchnorm:
                state = self.preprocess_state(state)
                next_state = self.preprocess_state(next_state)
            state = self.state_net(state)
            next_state = self.next_state_net(next_state)
            query = self.pred_net(torch.cat([state,next_state],1))
            key = self.skill_net(skill)
        return query, key


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RMS(object):
    def __init__(self, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

class APTArgs:
    def __init__(self,knn_k=16,knn_avg=True, rms=True,knn_clip=0.0005):
        self.knn_k = knn_k 
        self.knn_avg = knn_avg 
        self.rms = rms 
        self.knn_clip = knn_clip

rms = RMS()


def compute_apt_reward(source, target, args, device):

    b1, b2 = source.size(0), target.size(0)
    # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
    sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)
    reward, _ = sim_matrix.topk(args.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)
    if not args.knn_avg:  # only keep k-th nearest neighbor
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(device))  # (b1, )
    else:  # average over all k nearest neighbors
        reward = reward.reshape(-1, 1)  # (b1 * k, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(device))
        reward = reward.reshape((b1, args.knn_k))  # (b1, k)
        reward = reward.mean(dim=1)  # (b1,)
    reward = torch.log(reward + 1.0)
    return reward

class Reward(nn.Module):
    def __init__(self, obs_dim, skill_dim, action_dim, hidden_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.pred_net = nn.Sequential(nn.Linear(2*(obs_dim - skill_dim) + action_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, 1)) 
   
        self.apply(utils.weight_init)

    def forward(self,state,action,next_state):
        assert len(state.size()) == len(next_state.size())
        reward_pred = self.pred_net(torch.concat([state, action, next_state], axis=-1))
        return reward_pred

class CICAgent(DDPGAgent):
    # Contrastive Intrinsic Control (CIC)
    def __init__(self, update_skill_every_step, skill_dim, project_skill, 
                 alpha, z_id, freeze_rl, freeze_cic, num_seed_frames, 
                 action_repeat, p, init_rl, pearson_setting, canonical_setting, 
                 pearson_setting_sequencing, canonical_setting_sequencing, num_epic_skill, 
                 use_batchnorm, pearson_samples, canonical_samples, discount, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.project_skill = project_skill
        self.alpha = alpha
        self.z_id = z_id
        self.freeze_rl = freeze_rl
        self.freeze_cic = freeze_cic
        self.init_rl = init_rl
        kwargs["meta_dim"] = self.skill_dim

        # epic args
        self.pearson_setting = pearson_setting 
        self.canonical_setting = canonical_setting
        self.pearson_setting_sequencing = pearson_setting_sequencing
        self.canonical_setting_sequencing = canonical_setting_sequencing
        self.num_epic_skill = num_epic_skill 
        self.pearson_samples = pearson_samples
        self.canonical_samples = canonical_samples
        self.discount = discount
        self.compute_epic_rew = self.compute_inner_product 

        # create actor and critic
        super().__init__(**kwargs)

        # create cic first
        self.cic = CIC(self.obs_dim - skill_dim, skill_dim,
                           kwargs['hidden_dim'], project_skill, self.obs_type, p,
                           use_batchnorm).to(kwargs['device'])

        # optimizers
        self.cic_optimizer = torch.optim.Adam(self.cic.parameters(), 
                                                lr = self.lr)

        self.cic.train()

        rms.M = rms.M.to(self.device)
        rms.S = rms.S.to(self.device)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.cic, self.cic)
        self.cic.to(self.device)
        if self.init_rl:
            super().init_from(other)

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self, skill=None):
        if skill is not None:
            meta = OrderedDict()
            meta['skill'] = skill
            return meta

        skill = np.random.uniform(0,1,self.skill_dim).astype(np.float32)
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def find_ft_meta(self, bounds=None):
        self.extr_rew_fn = self.get_extr_rew()
        if self.z_id == "random_skill":
            if len(self.extr_reward_seq) == 0:
                skill = np.random.uniform(0,1,self.skill_dim).astype(np.float32)
            else:
                self.ft_skills = [dict(skill=np.random.uniform(0,1,self.skill_dim).astype(np.float32)) for _ in range(len(self.extr_reward_seq))]
                return 
        elif self.z_id == "irm_random":
            assert len(self.extr_reward_seq) == 0
            skill = self.irm_random_search(bounds).cpu().numpy()
        elif self.z_id == "irm_cem":
            skill = self.irm_cem(bounds).cpu().numpy()
        elif self.z_id == "irm_gradient_descent":
            skill = self.irm_gradient_descent(bounds).cpu().detach().numpy()
        elif self.z_id in ["env_rollout", "irm_random_iter", "grid_search", "env_rollout_cem", "env_rollout_iter", "reward_relabel"]:
            return # need to take env steps
        else:
            raise ValueError('Must select a finetuning mode')
        self.ft_skills = [dict(skill = skill)]

    def get_ft_meta(self, episode_step):
        if len(self.extr_reward_seq) == 0:
            return self.ft_skills[0]
        else:
            ind = min(episode_step // self.skill_duration, len(self.extr_reward_seq)-1)
            return self.ft_skills[ind]

    def update_meta(self, meta, global_step, time_step):
        if self.z_id == "random_skill":
            if global_step % self.update_skill_every_step == 0:
                return self.init_meta()
        else:
            raise ValueError('Cannot update skill when finetuning')
        return meta

    def compute_cpc_loss(self,obs,next_obs,skill):
        temperature = 0.5
        eps = 1e-6
        query, key = self.cic.forward(obs,next_obs,skill)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query,key.T) # (b,b)
        sim = torch.exp(cov / temperature) 
        neg = sim.mean(dim=-1) # take some negative samples
        pos = torch.exp(torch.sum(key * query, dim=-1) / temperature) 
        loss = -torch.log(pos) + torch.log(neg + eps)
        return loss, cov / temperature, pos, neg

    def compute_inner_product(self, obs, next_obs, skill):
        temperature = 0.5
        eps = 1e-6
        query, key = self.cic.forward(obs,next_obs,skill)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        sim = torch.sum(key * query, dim=-1) / temperature
        pos = torch.exp(torch.sum(key * query, dim=-1) / temperature) 
        return pos

    def update_cic(self, obs, skill, next_obs, step):
        metrics = dict()

        loss, logits, pos, neg = self.compute_cpc_loss(obs, next_obs, skill)
        loss = loss.mean()
        if not self.freeze_cic:
            self.cic_optimizer.zero_grad()
            loss.backward()
            self.cic_optimizer.step()

        metrics['cic_loss'] = loss.item()
        metrics['cic_logits'] = logits.norm()
        metrics['positive_kqpair_value'] = torch.mean(pos)
        metrics['negative_kqpair_value'] = torch.mean(neg)
        metrics['delta_kqpair_value'] = metrics['positive_kqpair_value'] - metrics['negative_kqpair_value']
        return metrics

    def compute_intr_reward(self, obs, skill, next_obs, step, keep_grad=False):
        if keep_grad:
            reward, _, pos, _ = self.compute_cpc_loss(obs, next_obs, skill)
            reward = reward.unsqueeze(-1)
        else:
            with torch.no_grad():
                reward, _, pos, _ = self.compute_cpc_loss(obs, next_obs, skill)

            reward = reward.clone().detach().unsqueeze(-1)
        return -reward, pos

    @torch.no_grad()
    def compute_apt_reward(self, obs, next_obs):
        args = APTArgs()
        source = self.cic.state_net(obs)
        target = self.cic.state_net(next_obs)
        reward = compute_apt_reward(source, target, args, self.device) # (b,)
        return reward.unsqueeze(-1) # (b,1)

    def update(self, replay_iter, step, dem=None):
        metrics = dict()

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill, replay_id = utils.to_torch(
            batch, self.device)

        if self.reward_free:
            with torch.no_grad():
                cic_obs, aug_obs = self.process_observation(obs)
                cic_next_obs, aug_next_obs = self.process_observation(next_obs)
        else:
            aug_obs = self.aug_and_encode(obs)
            with torch.no_grad():
                aug_next_obs = self.aug_and_encode(next_obs)
            if "walker" in self.obs_type:
                aug_obs = aug_obs[:, 3:]
                aug_next_obs = aug_next_obs[:, 3:]
            

        if self.reward_free:
            if self.alpha < 1.0:
                metrics.update(self.update_cic(cic_obs, skill, cic_next_obs, step))

            apt_reward = self.compute_apt_reward(cic_obs, cic_next_obs)
            cpc_reward, _ = self.compute_intr_reward(cic_obs, skill, cic_next_obs, step)
            intr_reward = self.alpha * apt_reward + (1 - self.alpha) * cpc_reward

            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            if self.reward_free:
                metrics['extr_reward'] = extr_reward.mean().item()
                metrics['cpc_reward'] = cpc_reward.mean().item()
                metrics['apt_reward'] = apt_reward.mean().item()
                metrics['intr_reward'] = intr_reward.mean().item()
                metrics['replay_id_0'] = (replay_id == 0).float().mean()
                metrics['replay_id_100'] = (replay_id == 100).float().mean()
            metrics['batch_reward'] = reward.mean().item()

        if not self.freeze_rl: 
            # extend observations with skill
            obs = torch.cat([aug_obs, skill], dim=1)
            next_obs = torch.cat([aug_next_obs, skill], dim=1)

            # update critic
            metrics.update(
                self.update_critic(obs, action, reward, discount, next_obs, step))

            # update actor
            metrics.update(self.update_actor(obs, step))

            # update critic target
            utils.soft_update_params(self.critic, self.critic_target,
                                    self.critic_target_tau)

        return metrics

    def compute_reward(self, obs, skill, action, next_obs, step, batch_size, obs_dim):
        reward, _ = self.compute_intr_reward(obs, torch.unsqueeze(skill, 0),
            next_obs, step, keep_grad=True)
        return torch.mean(reward.reshape((batch_size, -1, 1)), axis=1)
