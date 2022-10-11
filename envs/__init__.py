import numpy as np
import gym
from gym import spaces
from collections import deque
from typing import Any, NamedTuple
import dm_env
from dm_env import StepType, specs

# since had to download specific commit of gym to get rllab working
class RescaleAction(gym.ActionWrapper):
    r"""Rescales the continuous action space of the environment to a range [min_action, max_action].
    Example::
        >>> RescaleAction(env, min_action, max_action).action_space == Box(min_action, max_action)
        True
    """

    def __init__(self, env, min_action, max_action):
        assert isinstance(
            env.action_space, spaces.Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        super().__init__(env)
        self.min_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + min_action
        )
        self.max_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + max_action
        )
        self.action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        assert np.all(np.greater_equal(action, self.min_action)), (
            action,
            self.min_action,
        )
        assert np.all(np.less_equal(action, self.max_action)), (action, self.max_action)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = np.clip(action, low, high)
        return action

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
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        if type(time_step.observation) is dict:
            pixels = time_step.observation[self._pixels_key]
        else:
            pixels = time_step.observation
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY = (
    "`env.action_spec()` must return a single `BoundedArray`, got: {}.")
_MUST_BE_FINITE = "All values in `{name}` must be finite, got: {bounds}."
_MUST_BROADCAST = (
    "`{name}` must be broadcastable to shape {shape}, got: {bounds}.")


class ActionScaleWrapper(dm_env.Environment):
    """Wraps a control environment to rescale actions to a specific range."""
    __slots__ = ("_action_spec", "_env", "_transform")

    def __init__(self, env, minimum, maximum):
        """Initializes a new action scale Wrapper.

        Args:
          env: Instance of `dm_env.Environment` to wrap. Its `action_spec` must
            consist of a single `BoundedArray` with all-finite bounds.
          minimum: Scalar or array-like specifying element-wise lower bounds
            (inclusive) for the `action_spec` of the wrapped environment. Must be
            finite and broadcastable to the shape of the `action_spec`.
          maximum: Scalar or array-like specifying element-wise upper bounds
            (inclusive) for the `action_spec` of the wrapped environment. Must be
            finite and broadcastable to the shape of the `action_spec`.

        Raises:
          ValueError: If `env.action_spec()` is not a single `BoundedArray`.
          ValueError: If `env.action_spec()` has non-finite bounds.
          ValueError: If `minimum` or `maximum` contain non-finite values.
          ValueError: If `minimum` or `maximum` are not broadcastable to
            `env.action_spec().shape`.
        """
        action_spec = env.action_spec()
        if not isinstance(action_spec, specs.BoundedArray):
            raise ValueError(
                _ACTION_SPEC_MUST_BE_BOUNDED_ARRAY.format(action_spec))

        minimum = np.array(minimum)
        maximum = np.array(maximum)
        shape = action_spec.shape
        orig_minimum = action_spec.minimum
        orig_maximum = action_spec.maximum
        orig_dtype = action_spec.dtype

        def validate(bounds, name):
            if not np.all(np.isfinite(bounds)):
                raise ValueError(
                    _MUST_BE_FINITE.format(name=name, bounds=bounds))
            try:
                np.broadcast_to(bounds, shape)
            except ValueError:
                raise ValueError(_MUST_BROADCAST.format(
                    name=name, bounds=bounds, shape=shape))

        validate(minimum, "minimum")
        validate(maximum, "maximum")
        validate(orig_minimum, "env.action_spec().minimum")
        validate(orig_maximum, "env.action_spec().maximum")

        scale = (orig_maximum - orig_minimum) / (maximum - minimum)

        def transform(action):
            new_action = orig_minimum + scale * (action - minimum)
            return new_action.astype(orig_dtype, copy=False)

        dtype = np.result_type(minimum, maximum, orig_dtype)
        self._action_spec = action_spec.replace(
            minimum=minimum, maximum=maximum, dtype=dtype)
        self._env = env
        self._transform = transform

    def step(self, action):
        return self._env.step(self._transform(action))

    def reset(self):
        return self._env.reset()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)