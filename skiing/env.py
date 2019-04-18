"""A wrapper around the Skiing-v0 environment."""
import numpy as np
import gym
from gym import spaces


HEIGHT = 144
WIDTH = 144


def _process_obs(obs):
    return obs[58:202, 8:-8, :]


class SkiingWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SkiingWrapper, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(HEIGHT, WIDTH, 3),
                                            dtype=np.uint8)

    def step(self, action):
        obs, reward, done, info = super(SkiingWrapper, self).step(action)
        return _process_obs(obs), reward, done, info

    def reset(self):
        obs = super(SkiingWrapper, self).reset()
        return _process_obs(obs)
