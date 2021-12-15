'''
   This code was written by following the following PyTorch tutorial
   Tutorial Title: TRAIN A MARIO-PLAYING RL AGENT
   Project Title: MadMario
   Author: Yuansong Feng, Suraj Subramanian, Howard Wang, Steven Guo
   Date: June 2020
   Code version: 2.0
   Availability: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html#train-a-mario-playing-rl-agent
                 https://github.com/YuansongFeng/MadMario
'''

import gym
import torch
import random, datetime, numpy as np
from skimage import transform

from gym.spaces import Box

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
