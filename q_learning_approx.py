"""
Q-Learning Algorithm where the representation of state is an abstract list of features.
"""
import csv
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from agents.approximate_q_learning_agent import ApproxQLearningMarioAgent
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make("SuperMarioBros-v3")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = False
episodes = 100
games_per_episode = 20
info = {"coins": 0, "flag_get": False, "life":3, "score":0, "stage": 1, "status": "small","time": 400, "world": 1, "x_pos":40, "y_pos": 79}
weights = None
data_from_training = []
fields = ["Episode", "Game", "Coins", "Score", "World", "Level", "Time"]

env.reset()
for i in range(episodes):
    print(f"CURRENT EPISODE: {i+1}")
    q_learning_agent = ApproxQLearningMarioAgent(len(SIMPLE_MOVEMENT), weights=weights)
    for j in range(games_per_episode):
        print(f"game:{j+1}")
        while True:
            if done:
                data_from_training.append(
                    [
                        i + 1,
                        j + 1,
                        info["coins"],
                        info["score"],
                        info["world"],
                        info["stage"],
                        info["time"],
                    ]
                )
                env.reset()
                info = {"coins": 0, "flag_get": False, "life": 3, "score": 0, "stage": 1, "status": "small",
                        "time": 400, "world": 1, "x_pos": 40, "y_pos": 79}
                done = False
                break

            action = q_learning_agent.get_action(info)
            state, reward, done, next_info = env.step(action)
            q_learning_agent.update(info, action, next_info, reward)
            info = next_info
    q_values = q_learning_agent.get_q_values()

with open("data/q_learning_approx.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(data_from_training)

file_best_sequence = open("data/approx_policy", "w")
file_best_sequence.write(f"f{q_values}\n")
