from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.reset()

number_of_random_games = 100
done = False
for _ in range(number_of_random_games):
    while True:
        if done:
            env.reset()
            print(info)
            done = False
            break
        state, reward, done, info = env.step(env.action_space.sample())

env.close()