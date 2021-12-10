import numpy
import csv
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from agents.q_learning_agent import QLearningMarioAgent
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = False
episodes = 100
games_per_episode = 20
info = {}
curr_state = env.reset()
q_values = None
data_from_training = []
fields = [
    "Episode", "Game", "Coins", "Score", "World", "Level", "Time"
]
print(env._env._get_info())

for i in range(episodes):
    q_learning_agent = QLearningMarioAgent(len(SIMPLE_MOVEMENT), q_values=q_values)
    for j in range(games_per_episode):
        while True:
            if done:
                data_from_training.append([i+1, j+1, info["coins"], info["score"], info["world"], info["stage"], info["time"]])
                curr_state = env.reset()
                done = False
                break
            action = q_learning_agent.get_action(numpy.array2string(curr_state))
            next_state, reward, done, info = env.step(action)
            q_learning_agent.update(numpy.array2string(curr_state), action, numpy.array2string(next_state), reward)
            curr_state = next_state
            env.render()
    q_values = q_learning_agent.get_q_values()

with open("q_learning_data.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(data_from_training)

env.close()