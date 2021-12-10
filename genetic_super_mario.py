import random
import heapq
import csv
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from agents import genetic
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
sequence_length = 20000
generations = 100
number_of_sequences = 100
number_to_breed = 20
number_of_crossovers = 1
mutation_rate = .5
mutation_rate_decay = .95
curr_state = env.reset()

# CSV INFORMATION
fields = [
    "Generation", "Sequence", "Fitness", "Score", "World", "Stage"
]
data_from_ga = []

sequences = [[random.randint(0, len(SIMPLE_MOVEMENT) - 1) for _ in range(sequence_length)] for _ in range(number_of_sequences)]
for i in range(generations):
    best_sequences = []
    for j in range(sequences):
        sequence = sequences[j]
        max_reached, info = genetic.run_sequence(sequence, 'SuperMarioBros-v0')
        fitness = genetic.fitness_of_sequence_all_levels(info)
        best_sequences.append((-fitness, max_reached, sequence))
        data_from_ga.append([i+1, j+1, fitness, info["score"], info["world"], info["stage"]])

    heapq.heapify(best_sequences)
    to_breed = []
    max_steps = 0
    for _ in range(number_to_breed):
        neg_fitness, max_reached, sequence = heapq.heappop(best_sequences)
        max_steps = max(max_steps, max_reached)
        to_breed.append(sequence)

    # crossover
    sequences = genetic.mutate(genetic.crossover(max_steps, number_of_crossovers, to_breed, number_of_sequences), max_steps, mutation_rate)
    mutation_rate *= mutation_rate_decay

with open("ga_data.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(data_from_ga)