import gym
from gym import spaces
from envs import RescaleAction
import numpy as np

from dm_env import specs
import dm_env
from dm_env import StepType, specs
from dmc import ExtendedTimeStep

class DMEnvWrapper(dm_env.Environment):
    def __init__(self, env, time_limit=200):
        self._env = env 
        self.time_limit = time_limit
        self.current_step = 0
        action_space = env.action_space
        self._action_spec = specs.Array((2,), np.float32, 'action')
        self._observation_spec = specs.Array((2,), np.float32, 'observation')

    def step(self, action):
        action = action.astype(self._action_spec.dtype)
        
        obs, env_rew, done, info = self._env.step(action)
        obs = obs.astype(self._observation_spec.dtype)

        self.current_step += 1
        if self.current_step >= self.time_limit:
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

    def render(self):
        return self._env.render()


    def reset(self):
        obs = self._env.reset()
        obs = obs.astype(self._observation_spec.dtype)
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

class PlaneEnv(gym.Env):
  """2D continuous plane Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(PlaneEnv, self).__init__()
    self.max_obs = 128
    self.max_action = 10

    self.reward_range = (0, 10)
    self.action_space = spaces.Box(low=np.array([-self.max_action, -self.max_action]), high=np.array([self.max_action, self.max_action]))
    self.observation_space = spaces.Box(low=np.array([-self.max_obs, -self.max_obs]), high=np.array([self.max_obs, self.max_obs]))
    self.goal_pos = np.array([-50, 50])

  def step(self, action, make_video=None):
    posbefore = np.copy(self._obs)
    self._obs = np.clip(self._obs + action, -self.max_obs, self.max_obs)
    posafter = self._obs
    obs = self._obs

    if np.max(np.abs(posafter - self.goal_pos)) < 5:
      reward = 1000
    else:
      reward = -1
    done = False
    return obs, reward, done, {}

  def reset(self):
    self._obs = np.array([0.0, 0.0])
    return self._obs
  
  def render(self, mode='rgb_array', close=False):
    arr = np.zeros((self.max_obs*2, self.max_obs*2, 3)).astype(np.uint8)
    x, y = int(self._obs[0] + self.max_obs), int(self._obs[1] + self.max_obs)

    width = 4
    min_x, max_x = max(x-width, 0), min(x+width, self.max_obs*2)
    min_y, max_y = max(y-width, 0), min(y+width, self.max_obs*2)
    arr[min_x:max_x, min_y:max_y, :] = 255
    return arr

def make(name, obs_type, frame_stack, action_repeat, seed, time_limit):
    assert obs_type in ['states', 'pixels', 'delta_states']

    env = PlaneEnv()

    env.seed(seed)

    env = RescaleAction(env, min_action=-1.0, max_action=+1.0)
    if time_limit is None:
        env = DMEnvWrapper(env)
    else:
        env = DMEnvWrapper(env, time_limit)
    return env