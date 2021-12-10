import random

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def crossover(max_steps, number_of_crossovers, to_breed, number_of_children):
    crossover_points = [random.randint(0, max_steps) for _ in range(number_of_crossovers)]
    list.sort(crossover_points)
    buckets = [[]] * (number_of_crossovers + 1)
    for parent in to_breed:
        crossover_left = 0
        for i in range(number_of_crossovers):
            crossover_right = crossover_points[i]
            buckets[i].append(parent[crossover_left:crossover_right])
            crossover_left = crossover_right
        buckets[number_of_crossovers].append(parent[crossover_left:])

    sequences = []
    for _ in range(number_of_children):
        new_sequence = []
        for bucket in buckets:
            seq_piece = random.choice(bucket)
            new_sequence.extend(seq_piece)
        sequences.append(new_sequence)
    return sequences


def mutate(sequences, max_reached, mutation_rate):
    sequences = list.copy(sequences)
    for sequence in sequences:
        max_possible_mutations = int(max_reached * mutation_rate)
        number_of_mutations = random.randint(0, max_possible_mutations)
        for _ in range(number_of_mutations):
            place_to_mutate = random.randint(0, max_reached)
            action = random.randint(0, len(SIMPLE_MOVEMENT) - 1)
            sequence[place_to_mutate] = action
    return sequences

def run_sequence(sequence, env_name):
    # env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.reset()
    info = None
    for i in range(len(sequence)):
        curr_action = sequence[i]
        state, reward, done, info = env.step(curr_action)
        if done:
            return i, info
    return len(sequence) - 1, info

def run_sequence_parallel(seq, episode, seq_num):
    max_reached, info = run_sequence(seq, 'SuperMarioBros-v0')
    fitness = fitness_of_sequence_all_levels(info)
    data_tuple = ([episode + 1, seq_num + 1, fitness, info["score"], info["world"], info["stage"]])
    return -fitness, max_reached, data_tuple, seq


def fitness_of_sequence_all_levels(info):
    # max_score_ever = 1435100
    score = info["score"]
    coins = info["coins"]
    stage = info["stage"] # 1-4
    world = info["world"] # 1-8
    level_factor = 10 * world + stage
    if info["flag_get"]:
        level_factor = 100
    return level_factor * score + coins

def fitness_of_sequence_one_level(info):
    score = info["score"]
    coins = info["coins"]
    return score + (coins / 10)
