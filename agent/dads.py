import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from dm_env import specs
import math
from collections import OrderedDict
import time 

import utils

from agent.sac import SACAgent

import torch 
import numpy as np

class DADS(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, num_components, distribution, obs_type='states', p=0, variance=1):
        super().__init__()
        self.obs_dim = obs_dim 
        self.skill_dim = skill_dim
        self.obs_type = obs_type
        self.num_components = num_components  
        self.variance = variance
        self.distribution = distribution 
        assert self.distribution in ["mixture_gaussian", "gaussian"]

        if self.obs_type in ["walker_delta_xyz", "jaco_xyz"]:
            self.net = nn.Sequential(nn.Linear(3 + self.skill_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            self.preprocess = nn.BatchNorm1d(3, affine=False)
        elif self.obs_type in ["fetch_reach_xyz"]:
            self.net = nn.Sequential(nn.Linear(3 + self.skill_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            self.preprocess = nn.BatchNorm1d(3, affine=False)
        else:
            raise NotImplementedError

        obs_predict = self.obs_dim
        if self.distribution == "mixture_gaussian":
            self.mix_net = nn.Linear(hidden_dim, self.num_components)
            self.distr_nets = nn.ModuleList([nn.Linear(hidden_dim, obs_predict).cuda() for _ in range(self.num_components)])
        elif self.distribution == "gaussian":
            self.distr_net = nn.Linear(hidden_dim, obs_predict).cuda()
   
        self.apply(utils.weight_init)

    def forward(self,state,next_state,skill):
        assert len(state.size()) == len(next_state.size())
        info = dict() 
        if "delta" in self.obs_type:
            out = self.net(torch.cat((state, skill), dim=1))
        else:
            out = self.net(torch.cat((state, skill), dim=1))

        if self.distribution == "mixture_gaussian":
            mix_out = self.mix_net(out)
            means = []
            for i in range(self.num_components):
                means.append(self.distr_nets[i].forward(out).unsqueeze(1)) # linear out -> s' (observation size)
            all_means = torch.cat(means, dim=1) # (num_components, obs_size)
            all_diags = torch.ones_like(all_means) * self.variance # (num_components, obs_size)
            
            # Mixture of Gaussians with all_means, all_variance
            # (batch, n_components, distribution_dim)
            mix = D.Categorical(logits=mix_out)
            comp = D.Independent(D.Normal(all_means, all_diags), 1)
            gmm = D.MixtureSameFamily(mix, comp) 
            s_predict = self.process_obs(state, next_state)
            prob_delta = gmm.log_prob(s_predict)
            info['dads_mean'] = all_means.abs().mean()
        elif self.distribution == "gaussian":
            mean = self.distr_net.forward(out) 
            diag = torch.ones_like(mean) * self.variance # (num_components, obs_size)

            # Mixture of Gaussians with all_means, all_variance
            # (batch, n_components, distribution_dim)
            comp = D.Normal(mean, diag)
            s_predict = self.process_obs(state, next_state)
            prob_delta = comp.log_prob(s_predict)
            prob_delta = prob_delta.sum(1)
            info['dads_mean'] = mean.abs().mean()
        info['s_predict'] = s_predict.abs().mean()
        info['s_predict_raw'] = (next_state-state).abs().mean()
        return prob_delta, info

    def process_obs(self, s, n_s):
        out = n_s - s 
        out = self.preprocess(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DADSAgent(SACAgent):
    def __init__(self, update_skill_every_step, skill_dim, scale, 
                 z_id, freeze_rl, freeze_dads, num_seed_frames, action_repeat, 
                 p, init_rl, pearson_setting, canonical_setting, num_epic_skill, 
                 pearson_samples, canonical_samples, discount, update_dads_every_steps, 
                 num_neg_samples, update_rl_every_steps, num_dads_updates, variance, 
                 num_components, max_batch, distribution, extr_reward, extr_reward_seq, 
                 **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.update_rl_every_steps = update_rl_every_steps
        self.scale = scale
        self.z_id = z_id
        self.freeze_rl = freeze_rl
        self.freeze_dads = freeze_dads
        self.init_rl = init_rl
        self.update_dads_every_steps = update_dads_every_steps
        self.num_neg_samples = num_neg_samples
        self.num_dads_updates = num_dads_updates
        self.num_components = num_components
        self.max_batch = max_batch
        self.grad_steps = 0
        kwargs["meta_dim"] = self.skill_dim

        # epic args
        self.pearson_setting = pearson_setting 
        self.canonical_setting = canonical_setting
        self.num_epic_skill = num_epic_skill 
        self.pearson_samples = pearson_samples
        self.canonical_samples = canonical_samples
        self.discount = discount
        self.compute_epic_rew = self.compute_log_prob_noinfo
        self.extr_reward = extr_reward 
        self.extr_reward_seq = extr_reward_seq

        # create actor and critic
        super().__init__(**kwargs)

        # create dads first
        self.dads = DADS(self.discr_obs_dim, skill_dim,
                           kwargs['hidden_dim'], num_components, distribution, self.obs_type, 
                           p, variance).to(kwargs['device'])

        # optimizers
        self.dads_optimizer = torch.optim.Adam(self.dads.parameters(), 
                                                lr = self.lr)

        self.dads.train()

    def get_ft_meta(self, episode_step):
        if len(self.extr_reward_seq) == 0:
            return self.ft_skills[0]
        else:
            ind = min(episode_step // self.skill_duration, len(self.extr_reward_seq)-1)
            return self.ft_skills[ind]

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.dads, self.dads)
        self.dads.to(self.device)
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
        elif self.z_id in ["env_rollout", "irm_random_iter", "grid_search", "env_rollout_cem", "env_rollout_iter"]:
            return # need to take env steps
        else:
            raise ValueError('Must select a finetuning mode')
        self.ft_skills = [dict(skill = skill)]

    def update_meta(self, meta, global_step, time_step):
        if self.z_id == "random_skill":
            if global_step % self.update_skill_every_step == 0:
                return self.init_meta()
        else:
            raise ValueError('Cannot update skill when finetuning')
        return meta

    def compute_dads_reward(self,obs,next_obs,skill):
        logp, _ = self.compute_log_prob(obs, next_obs, skill)
        obs_altz = torch.cat([obs] * self.num_neg_samples)
        next_obs_altz = torch.cat([next_obs] * self.num_neg_samples)
        neg_skills = torch.rand(obs_altz.shape[0], skill.shape[1], device=obs_altz.device)

        if obs_altz.shape[0] <= self.max_batch:
            logp_altz, _ = self.compute_log_prob(obs_altz, next_obs_altz, neg_skills)
        else:
            logp_altz = []
            for split_idx in range(obs_altz.shape[0] // self.max_batch):
                start_split = split_idx * self.max_batch
                end_split = (split_idx+1) * self.max_batch
                prob, _ = self.compute_log_prob(obs_altz[start_split:end_split], 
                                            next_obs_altz[start_split:end_split], 
                                            neg_skills[start_split:end_split])
                logp_altz.append(prob)
            if obs_altz.shape[0] % self.max_batch:
                start_split = obs_altz.shape[0] % self.max_batch
                prob, _ = self.compute_log_prob(obs_altz[-start_split:], 
                                            next_obs_altz[-start_split:], 
                                            neg_skills[-start_split:])
                logp_altz.append(prob) 
            logp_altz = torch.cat(logp_altz)

        logp_altz = torch.cat(torch.tensor_split(logp_altz.unsqueeze(1), self.num_neg_samples), dim=1)

        logp = logp.unsqueeze(1)
        rew = -torch.log(1 + torch.exp(torch.clamp(logp_altz - logp, -50, 50)).sum(1))
        rew += np.log(self.num_neg_samples + 1)
        info = dict(logp_altz=logp_altz, logp=logp, percent_correct=(logp - logp_altz > 0).float().mean())
        return rew, info

    def compute_log_prob(self, obs, next_obs, skill):
        prob, info = self.dads.forward(obs,next_obs,skill)
        return prob, info

    def compute_log_prob_noinfo(self, obs, next_obs, skill):
        return self.compute_log_prob(obs, next_obs, skill)[0]

    def update_dads(self, obs, skill, next_obs, step):
        metrics = dict()
        prob, dads_info = self.compute_log_prob(obs, next_obs, skill)
        loss = -prob.mean()
        if not self.freeze_dads:
            self.dads_optimizer.zero_grad()
            loss.backward()
            self.dads_optimizer.step()

        metrics['dads_logprob'] = prob.mean().item()
        metrics['dads_loss'] = loss.item()
        metrics.update(dads_info)
        return metrics

    def compute_intr_reward(self, obs, skill, next_obs, step, keep_grad=False):
        if keep_grad:
            reward, info = self.compute_dads_reward(obs, next_obs, skill)
            reward = reward.unsqueeze(-1)
        else:
            with torch.no_grad():
                reward, info = self.compute_dads_reward(obs, next_obs, skill)
                
            reward = reward.clone().detach().unsqueeze(-1)
        return reward * self.scale, info

    def get_obs(self, replay_loader):
        batch = replay_loader.get_all_transitions()

        obs, action, extr_reward, discount, next_obs, skill, replay_id = utils.to_torch(
            batch, self.device)

        with torch.no_grad():
            dads_obs, aug_obs = self.process_observation(obs)
            dads_next_obs, aug_next_obs = self.process_observation(next_obs)
        return dict(dads_obs=dads_obs, dads_next_obs=dads_next_obs, action=action,
            extr_reward=extr_reward, skill=skill, replay_id=replay_id,
            aug_obs=aug_obs, aug_next_obs=aug_next_obs, discount=discount)

    def get_batch(self, all_batch, batch_size):
        perm = torch.randperm(all_batch['action'].shape[0])
        batch = dict()
        for key in all_batch.keys():
            val = all_batch[key]
            new_val = val[perm[:batch_size]]
            batch[key] = new_val
        return batch

    def relabel_reward(self, all_batch):
        dads_obs, skill, dads_next_obs = all_batch['dads_obs'], all_batch['skill'], all_batch['dads_next_obs'] 
        if self.reward_free:
            intr_reward, rew_info = self.compute_intr_reward(dads_obs, skill, dads_next_obs, None)
            reward = intr_reward
        else:
            reward = extr_reward
            rew_info = None
        all_batch['reward'] = reward
        return rew_info

    def update(self, replay_loader, step, logger, dem=None):
        metrics = dict()

        # process batch 
        all_batch = self.get_obs(replay_loader)

        if self.reward_free:
            if step % self.update_dads_every_steps == 0:
                for _ in range(self.num_dads_updates):
                    batch = self.get_batch(all_batch, replay_loader.batch_size)
                    metrics.update(self.update_dads(batch['dads_obs'], 
                            batch['skill'], batch['dads_next_obs'], step))
                    logger.log_metrics(metrics, self.grad_steps, ty='train')
                    self.grad_steps += 1

        if step % self.update_rl_every_steps == 0:
            if not self.freeze_rl: 
                rew_info = self.relabel_reward(all_batch)
                for _ in range(self._num_rl_updates):
                    batch = self.get_batch(all_batch, replay_loader.batch_size)
                    aug_obs, aug_next_obs, skill = batch['aug_obs'], batch['aug_next_obs'], batch['skill']
                    dads_obs, dads_next_obs, action = batch['dads_obs'], batch['dads_next_obs'], batch['action']
                    discount, reward = batch['discount'], batch['reward']

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

                    if self.reward_free:
                        metrics['logp_altz'] = rew_info['logp_altz'].mean().item()
                        metrics['logp'] = rew_info['logp'].mean().item()
                        metrics['percent_correct'] = rew_info['percent_correct'].mean().item()
                        metrics['delta_logp'] = metrics['logp'] - metrics['logp_altz']
                        metrics['intr_reward'] = batch['reward'].mean().item()
                        metrics['batch_reward'] = reward.mean().item()
                        metrics['extr_reward'] = batch['extr_reward'].mean().item() 
                        metrics['replay_id_0'] = (batch['replay_id'] == 0).float().mean()
                        metrics['replay_id_100'] = (batch['replay_id'] == 100).float().mean()
                    logger.log_metrics(metrics, self.grad_steps, ty='train')
                    self.grad_steps += 1
        return metrics


    def compute_pearson_distance(self, rew_1, rew_2):
        rew_1 = rew_1 - torch.mean(rew_1)
        rew_2 = rew_2 - torch.mean(rew_2)
        var_1 = rew_1**2
        var_2 = rew_2**2
        cov = torch.sum(rew_1 * rew_2)
        corr = cov / (torch.sqrt(torch.sum(var_1) * torch.sum(var_2)))
        corr = min(corr, 1.0)  
        return torch.sqrt(0.5 * (1 - corr))

    def compute_reward(self, obs, skill, action, next_obs, step, batch_size, obs_dim):
        reward, rew_info = self.compute_intr_reward(obs, torch.unsqueeze(skill, 0),
            next_obs, step, keep_grad=True)
        return torch.mean(reward.reshape((batch_size, -1, 1)), axis=1)
