import numpy
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from agents import q_learning
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
agent = q_learning.QLearningMarioAgent(len(SIMPLE_MOVEMENT))

done = True
episodes = 10
curr_state = env.reset()
for i in range(episodes):
    for step in range(20000):
        if done:
            curr_state = env.reset()
        action = agent.get_action(numpy.array2string(curr_state))
        next_state, reward, done, info = env.step(action)
        agent.update_q_values(numpy.array2string(curr_state), action, numpy.array2string(next_state), reward)
        curr_state = next_state
        env.render()


    print(f"EPISODE {i+1}.")

learned_q_values = agent.get_q_values()
agent = q_learning.QLearningMarioAgent(len(SIMPLE_MOVEMENT), learned_q_values, exploration_rate=0.0, learning_rate=0.0, discount=1.0, decay=1.0)
for step in range(20000):
    if done:
        curr_state = env.reset()
    action = agent.get_action(numpy.array2string(curr_state))
    next_state, reward, done, info = env.step(action)
    curr_state = next_state
    env.render()

#
# for step in range(20000):
#     if done:
#         curr_state = env.reset()
#     action = agent.get_action(numpy.array2string(curr_state))
#     next_state, reward, done, info = env.step(action)
#     agent.update_q_values(numpy.array2string(curr_state), action, numpy.array2string(next_state), reward)
#     curr_state = next_state

env.close()