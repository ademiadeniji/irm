import random
import re
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from typing import Any, NamedTuple
from dm_env import StepType, specs
from dm_control.utils import rewards


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def chain(*iterables):
    for it in iterables:
        yield from it


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update_params(net, target_net):
    for (n, param), (tn, target_param) in zip(net.named_parameters(), target_net.named_parameters()):
        target_param.data.copy_(param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def grad_norm(params, norm_type=2.0):
    params = [p for p in params if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


def param_norm(params, norm_type=2.0):
    total_norm = torch.norm(
        torch.stack([torch.norm(p.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


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


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + torch.square(delta) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device

    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)
        reward, _ = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)
        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(reward - self.knn_clip, torch.zeros_like(reward).to(self.device)) if self.knn_clip >= 0.0 else reward  # (b1, 1)
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(reward - self.knn_clip, torch.zeros_like(reward).to(self.device)) if self.knn_clip >= 0.0 else reward
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward

class PBKL(object):
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device
    def __call__(self, source, source_demo):
        """Computes the KL divergence between the source distribution and source_demo distribution"""
        n = source.size(0) # number of samples from distr p
        m = source_demo.size(0) # number of samples from distr q
        d = source.size(1) # dimension of the observation
        v_k = compute_v_k(source, source_demo, self.knn_k)
        p_k = compute_p_k(source, self.knn_k)
        log_ratio = torch.log(v_k + 1) - torch.log(p_k + 1)
        if self.knn_rms: # normalize to be good scale
            log_ratio = log_ratio / self.rms(log_ratio)[0]
        reward = log_ratio # + math.log(m / (n-1))
        return reward, {'v_k_mean': v_k.mean(), 'v_k_std': v_k.std(), 'p_k_mean': p_k.mean(), 'p_k_std': p_k.std()}

def compute_v_k(source, source_demo, knn_k):
    n = source.size(0) # number of samples from distr p
    m = source_demo.size(0) # number of samples from distr q
    d = source.size(1) # dimension of the observation
    kl_sim_matrix = torch.norm(source[:, None, :].view(n, 1, -1) - source_demo[None, :, :].view(1, m, -1), dim=-1, p=2)
    v_k, _ = kl_sim_matrix.topk(knn_k + 1, dim=1, largest=False, sorted=True)
    v_k = v_k[:, -1] # (n, )
    return v_k

def compute_p_k(source, knn_k):
    n = source.size(0) # number of samples from distr p
    d = source.size(1) # dimension of the observation
    # compute kth distance
    sim_matrix = torch.norm(source[:, None, :].view(n, 1, -1) - source[None, :, :].view(1, n, -1), dim=-1, p=2)
    p_k, _ = sim_matrix.topk(knn_k + 1, dim=1, largest=False, sorted=True)
    p_k = p_k[:, -1] # (n, )
    return p_k

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)

class goal_top_right(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, obs, next_obs, skill):
        # scale between -1, 1
        if (len(obs.shape) == 1):
            return -np.sum(np.square(obs/128 - np.array([1, 1])))
        else:
            with torch.no_grad():
                goal = torch.tensor([[1, 1]], device=self.device) # not sure if data efficient
                return -torch.square(next_obs/128 - goal).sum(dim=1).unsqueeze(1)
        return reward

class goal_top_left(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, obs, next_obs, skill):
        # scale between -1, 1
        if (len(obs.shape) == 1):
            return -np.sum(np.square(obs/128 - np.array([-1, 1])))
        else:
            with torch.no_grad():
                goal = torch.tensor([[-1, 1]], device=self.device) # not sure if data efficient
                return -torch.square(next_obs/128 - goal).sum(dim=1).unsqueeze(1)
        return reward

class goal_reward(object):
    def __init__(self, device, goal):
        self.goal = goal
        self.device = device

    def __call__(self, obs, next_obs, skill):
        if (len(obs.shape) == 1):
            return -np.sum(np.square(obs - np.array(self.goal)))
        else:
            with torch.no_grad():
                goal_th = torch.tensor([self.goal], device=self.device) # not sure if data efficient
                return -torch.square(next_obs - goal_th).sum(dim=1).unsqueeze(1)

class walk_right(object):
    def __call__(self, obs, next_obs, skill):
        if (len(obs.shape) == 1):
            return next_obs[0] - obs[0]
        else:
            with torch.no_grad():
                return next_obs[:, 0] - obs[:, 0]

class walk_left(object):
    def __call__(self, obs, next_obs, skill):
        if (len(obs.shape) == 1):
            return -(next_obs[0] - obs[0])
        else:
            with torch.no_grad():
                return -(next_obs[:, 0] - obs[:, 0])

class quadruped_stand(object):
    def __call__(self, obs, next_obs, skill):
        # penalize all velocities (?) 
        if (len(obs.shape) == 1):
            return -np.sum(np.abs(obs))
        else:
            with torch.no_grad():
                return -torch.sum(torch.abs(obs), axis=-1).unsqueeze(-1)

class quadruped_jump(object):
    def __call__(self, obs, next_obs, skill):
        # reward upward velocities
        if (len(obs.shape) == 1):
            return obs[-1] 
        else:
            with torch.no_grad():
                return obs[:, -1].unsqueeze(-1)

class quadruped_move(object):
    def __init__(self, desired_speed):
        self._desired_speed = desired_speed
        self.value_at_margin = 0.5

    def __call__(self, obs, next_obs, skill):
        if (len(obs.shape) == 1):
            velo = obs[0] 
            move_reward = rewards.tolerance(
                velo,
                bounds=(self._desired_speed, float('inf')),
                margin=self._desired_speed,
                value_at_margin=self.value_at_margin,
                sigmoid='linear')
            return move_reward
        else:
            velo = obs[:, 0]
            in_bounds = self._desired_speed <= velo
            if self._desired_speed == 0:
                value = torch.where(in_bounds, 1.0, 0.0) 
            else:
                d = torch.where(velo < self._desired_speed, self._desired_speed-velo, velo) / self._desired_speed
                sig = self._sigmoids(d, self.value_at_margin)

            value = in_bounds * 1 + (1 - in_bounds.long()) * sig
            value = value.type(torch.float32).unsqueeze(1)
            return value

    def _sigmoids(self, x, value_at_1):
        # linear
        scale = 1-value_at_1
        scaled_x = x*scale

        cond = torch.abs(scaled_x) < 1
        val1 = (1 - scaled_x).float()
        val2 = float(0)
        final = cond * val1 + (1 - cond.long()) * val2
        return final

class walker_move(object):
    def __init__(self, desired_speed):
        self._desired_speed = desired_speed
        self._STAND_HEIGHT = 1.2
        self.value_at_margin = 0.1


    def __call__(self, obs, next_obs, skill):
        if (len(obs.shape) == 1):
            standing = rewards.tolerance(obs[-1],
                                 bounds=(self._STAND_HEIGHT, float('inf')),
                                 margin=self._STAND_HEIGHT/2)
            if self._desired_speed == 0:
                return 3*standing / 4 # similar scaling
            else:
                horizontal_velocity = next_obs[0] - obs[0]
                move_reward = rewards.tolerance(
                    horizontal_velocity,
                    bounds=(self._desired_speed, float('inf')),
                    margin=self._desired_speed / 2,
                    value_at_margin=0.5,
                    sigmoid='linear')
                return 3 * standing / 4 * (5 * move_reward + 1) / 6
        else:
            z = obs[:, -1]
            in_bounds = z > self._STAND_HEIGHT
            d = torch.where(z < self._STAND_HEIGHT, self._STAND_HEIGHT-z, z) / (self._STAND_HEIGHT / 2)
            value = torch.where(in_bounds, 1.0, 0.0) 
            
            sig = self._sigmoids(d, self.value_at_margin)

            value = in_bounds * 1 + (1 - in_bounds.long()) * sig
            standing = value.type(torch.float32).unsqueeze(1)

            if self._desired_speed == 0:
                return 3*standing / 4 # scaling according to walker.py
            else:
                velo = (next_obs[:, 0] - obs[:, 0]) * 15 + 1e-3 # manual scaling
                in_bounds = velo > self._desired_speed
                d = torch.where(velo < self._desired_speed, self._desired_speed-velo, velo) / (self._desired_speed / 2)
                value = torch.where(in_bounds, 1.0, 0.0) 
                
                sig = self._sigmoids2(d, 0.5)

                move_reward = in_bounds * 1 + (1 - in_bounds.long()) * sig
                move_reward = move_reward.type(torch.float32).unsqueeze(1)

                return 3 * standing / 4 * (5 * move_reward + 1) / 6

    def _sigmoids(self, x, value_at_1):
        # gaussian
        scale = np.sqrt(-2 * np.log(value_at_1))
        return torch.exp(-0.5 * (x*scale)**2)

    def _sigmoids2(self, x, value_at_1):
        # linear
        scale = 1-value_at_1
        scaled_x = x*scale

        cond = torch.abs(scaled_x) < 1
        val1 = (1 - scaled_x).float()
        val2 = float(0)
        final = cond * val1 + (1 - cond.long()) * val2
        return final

class walker_flip(object):
    def __init__(self):
        self._STAND_HEIGHT = 1.5
        self._SPIN_SPEED = 1.2
        self.value_at_margin = 0.1

    def __call__(self, obs, next_obs, skill):
        if (len(obs.shape) == 1):
            standing = rewards.tolerance(obs[-1],
                                 bounds=(self._STAND_HEIGHT, float('inf')),
                                 margin=self._STAND_HEIGHT/2)
            return standing
        else:
            z = obs[:, -1]
            in_bounds = z > self._STAND_HEIGHT
            d = torch.where(z < self._STAND_HEIGHT, self._STAND_HEIGHT-z, z) / (self._STAND_HEIGHT / 2)
            value = torch.where(in_bounds, 1.0, 0.0) 
            
            sig = self._sigmoids(d, self.value_at_margin)

            value = in_bounds * 1 + (1 - in_bounds.long()) * sig
            standing = value.type(torch.float32).unsqueeze(1)

            return standing

    def _sigmoids(self, x, value_at_1):
        # gaussian
        scale = np.sqrt(-2 * np.log(value_at_1))
        return torch.exp(-0.5 * (x*scale)**2)

class jaco_reach(object):
    def __init__(self, device, task):
        self.task = task 
        _PROP_Z_OFFSET = 0
        if task == "jaco_reach_top_left":
            self.target_pos = np.array([-0.09, 0.09, _PROP_Z_OFFSET]) 
        elif task == "jaco_reach_top_right":
            self.target_pos = np.array([0.09, 0.09, _PROP_Z_OFFSET])  
        elif task == "jaco_reach_bottom_left":
            self.target_pos = np.array([-0.09, -0.09, _PROP_Z_OFFSET])  
        elif task == "jaco_reach_bottom_right":
            self.target_pos = np.array([0.09, -0.09, _PROP_Z_OFFSET]) 
        else:
            raise ValueError("Invalid jaco reach task specified", task)
        self.target_pos_th = torch.tensor(self.target_pos).unsqueeze(0).to(device)
        self._TARGET_RADIUS = 0.05 
        self.value_at_margin = 0.1

    def __call__(self, obs, next_obs, skill):
        if (len(obs.shape) == 1):
            distance = next_obs - self.target_pos
            distance = np.linalg.norm(distance)
            out = rewards.tolerance(
                distance, bounds=(0, self._TARGET_RADIUS), margin=self._TARGET_RADIUS)
            return out
        else:
            distance = next_obs - self.target_pos_th
            distance = torch.linalg.norm(distance, dim=1)
            in_bounds = torch.logical_and(0 <= distance, distance <= self._TARGET_RADIUS)
            d = torch.where(distance < 0, -distance, distance - self._TARGET_RADIUS) / self._TARGET_RADIUS
            value = torch.where(in_bounds, 1.0, self._sigmoids(d, self.value_at_margin))
            value = value.type(torch.float32).unsqueeze(1)
            return value

    def _sigmoids(self, x, value_at_1):
        # gaussian
        scale = np.sqrt(-2 * np.log(value_at_1))
        return torch.exp(-0.5 * (x*scale)**2)
        

def get_extr_rew(rew, device):
    # global reward
    if rew == "goal_top_right":
        return goal_top_right(device)
    if rew == "goal_top_left":
        return goal_top_left(device)
    elif "goal" in rew:
        if rew == "goal_0_0_0.5":
            goal = [0, 0, 0.5]
        elif rew == "goal_1_1.2_1":
            goal = [1, 1.2, 1]
        elif rew == "goal_1.5_0.3_0.45":
            goal = [1.5, 0.3, 0.45]
        elif rew == "goal_barrier1":
            goal = [1.31, 0.6]
        elif rew == "goal_barrier2":
            goal = [1.31, 0.9]
        elif rew == "goal_barrier3":
            goal = [1.16, 0.9]
        elif rew == "goal_1_1_0.5":
            goal = [1, 1, 0.5]
        return goal_reward(device, goal)
    elif rew == "walker_right":
        return walk_right()
    elif rew == "walker_left":
        return walk_left()
    elif rew == "quadruped_walk":
        return quadruped_move(0.5)
    elif rew == "quadruped_run":
        return quadruped_move(5)
    elif rew == "quadruped_stand":
        return quadruped_stand()
    elif rew == "quadruped_jump":
        return quadruped_jump()
    elif rew == "walker_stand":
        return walker_move(0)
    elif rew == "walker_walk":
        return walker_move(1)
    elif rew == "walker_run":
        return walker_move(8)
    elif rew == "walker_flip":
        return walker_flip()
    elif "reach" in rew:
        return jaco_reach(device, rew)
    else:
        raise NotImplementedError
