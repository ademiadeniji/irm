import numpy as np

from gym.wrappers import RescaleAction
import gym

import dm_env
from dm_env import StepType, specs
from dmc import ExtendedTimeStep

class DMEnvWrapper(dm_env.Environment):
    def __init__(self, env, time_limit=200):
        self._env = env 
        self.time_limit = time_limit
        self.current_step = 0
        action_space = env.action_space
        self._action_spec = specs.BoundedArray(action_space.shape,
                                               np.dtype('float32'),
                                               action_space.low,
                                               action_space.high,
                                               'action')
        
        observation_space = env.observation_space.spaces

        # Fetch original observation space is a dict. Concatenate here into a single array.
        concat_obs_space_low = []
        concat_obs_space_high = []
        for key, value in observation_space.items():
            if not 'desired_goal' in key:
                concat_obs_space_low.append(value.low)
                concat_obs_space_high.append(value.high)

        concat_obs_space_low = np.concatenate(concat_obs_space_low)
        concat_obs_space_high = np.concatenate(concat_obs_space_high)

        self._observation_spec = specs.BoundedArray(concat_obs_space_low.shape,
                                               np.dtype('float32'),
                                               concat_obs_space_low,
                                               concat_obs_space_high,
                                               'observation')
    
    def concatenate_obs(self, obs):
        concat_obs = []
        for key, value in obs.items():
            if 'desired_goal' not in key:
                concat_obs.append(value)

        concat_obs = np.concatenate(concat_obs)
        concat_obs = concat_obs.astype(np.dtype('float32'))
        return concat_obs

    def step(self, action, make_video=False):
        action = action.astype(self._action_spec.dtype)

        obs, env_rew, done, info = self._env.step(action, make_video=make_video)
        obs = self.concatenate_obs(obs)

        self.current_step += 1
        if self.current_step == self.time_limit:
            done = True
        
        timestep = ExtendedTimeStep(step_type=StepType.MID if not done else StepType.LAST,
                    reward=env_rew,
                    discount=1.0,
                    observation=obs,
                    action=action
                    )

        return timestep

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode='rgb_array', height=432, width=432):
        return self._env.render(mode=mode, height=height, width=width)


    def reset(self):
        obs = self._env.reset()
        obs = self.concatenate_obs(obs)
        self.current_step = 0

        timestep = ExtendedTimeStep(step_type=StepType.FIRST,
                    reward=0.0,
                    discount=1.0,
                    observation=obs,
                    action=np.zeros(self._action_spec.shape).astype(self._action_spec.dtype)
                    )
        return timestep

    def __getattr__(self, name):
        return getattr(self._env, name)

def make(name, obs_type, frame_stack, action_repeat, seed, random_reset=False, time_limit=50):
    if 'push' in name:
        env = gym.make("FetchPushPrimitives-v1")
    elif 'reach' in name:
        env = gym.make("FetchReach-v1")
    elif 'barrier' in name:
        env = gym.make("FetchBarrierPrimitives-v1")
    else:
        raise NotImplementedError

    env.seed(seed)
    if time_limit is not None:
        env._max_episode_steps = time_limit
        env.env.randomize_object = random_reset

    env = RescaleAction(env, min_action=-1.0, max_action=+1.0)
    env = DMEnvWrapper(env, time_limit=time_limit)
    return env
