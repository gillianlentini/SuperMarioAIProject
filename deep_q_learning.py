import torch
from pathlib import Path
import random, datetime, numpy as np, cv2
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from agents.deep_q_learning_agent import DeepQLearningMario

#NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

# Initialize Super Mario environment
from util.deep_q_logging import MetricLogger
from util.wrappers import SkipFrame, ResizeObservation

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path('./data/checkpoints')

mario = DeepQLearningMario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 40000

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )