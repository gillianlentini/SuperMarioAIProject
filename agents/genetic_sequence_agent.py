import random

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def crossover(max_steps, number_of_crossovers, to_breed, number_of_children):
    """
    Performs the crossover functionality for the GA. For more than one crossover, we do not maintain
    the same parents! Can be combo of parents.

    :param max_steps: Furthest reach action in sequence (We only crossover before here)
    :param number_of_crossovers: Number of crossovers to perform
    :param to_breed: Parent sequences
    :param number_of_children: Number of children to make
    :return: Children sequences of len(number_of_children)
    """
    crossover_points = [
        random.randint(0, max_steps - 1) for _ in range(number_of_crossovers)
    ]
    list.sort(crossover_points)
    buckets = []
    for _ in range(number_of_crossovers + 1):
        buckets.append([])
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
    """
    Mutates genes (actions) at some rate.

    :param sequences: The sequences to mutate
    :param max_reached: Mutates only reached actions (only genes before this one)
    :param mutation_rate: Rate at which genes (actions) are mutated
    :return: A new list of sequences with mutations
    """
    sequences = list.copy(sequences)
    for sequence in sequences:
        max_possible_mutations = int(max_reached * mutation_rate)
        number_of_mutations = random.randint(0, max_possible_mutations)
        for _ in range(number_of_mutations):
            place_to_mutate = random.randint(0, max_reached - 1)
            action = random.randint(0, len(SIMPLE_MOVEMENT) - 1)
            sequence[place_to_mutate] = action
    return sequences


def run_sequence(sequence, env_name, stop_when_dead):
    """
    Runs the sequence and returns info on it.

    :param sequence: The sequence to run
    :param env_name: The env to make to run the sequence
    :param stop_when_dead: Whether we should stop after first death.
    :return: Max step reached, Info on run
    """
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.reset()
    info = None
    for i in range(len(sequence)):
        curr_action = sequence[i]
        state, reward, done, info = env.step(curr_action)
        if done or (stop_when_dead and (info["life"] < 2 or info["time"] == 0)):
            env.close()
            return i, info
    env.close()
    return len(sequence) - 1, info


def run_sequence_parallel(seq, episode, seq_num, stop_when_dead, world, stage):
    """
    Function to support parallelization of running sequences. Runs a specified sequence and gets its fitness.

    :param seq: The sequence to run
    :param episode: The episode of training we are on (generation)
    :param seq_num: The generation number
    :param stop_when_dead: Whether to stop run of sequence after first life lost
    :param world: The world to run sequence in, False if all
    :param stage: The stage to run sequence in, False if all
    :return: -fitness, max reached step, data on run, sequence
    """
    if world and stage:
        stage_name = f"SuperMarioBros-{world}-{stage}-v3"
    else:
        stage_name = "SuperMarioBros-v3"
    max_reached, info = run_sequence(seq, stage_name, stop_when_dead)
    if world and stage:
        fitness = fitness_of_sequence_one_level(info)
    else:
        fitness = fitness_of_sequence_all_levels(info)
    data = [
        episode + 1,
        seq_num + 1,
        fitness,
        info["score"],
        info["world"],
        info["stage"],
        info["x_pos"],
    ]
    return -fitness, max_reached, data, seq


def fitness_of_sequence_all_levels(info):
    """
    Returns fitness score based on GA for all levels

    :param info: Info about sequence run
    :return: Fitness score. Score is weighted by level reached. This number will likely be largest.
    There then is a bias meant to decide between sequences within the same stage for furthest right.
    Then there is a coins bias
    """
    # max_score_ever = 1435100
    score = info["score"]
    stage = info["stage"]  # 1-4
    world = info["world"]  # 1-8
    x_pos = info["x_pos"]
    level_factor = 10 * world + stage
    if info["flag_get"]:
        level_factor = 100
    score_level_factor = level_factor * (score / 100)
    right_bias = x_pos * 10
    return score_level_factor + right_bias


def fitness_of_sequence_one_level(info):
    """
    Returns the fitness of the sequence for one level.

    :param info: Info on sequence run
    :return: Fitness of sequence
    """
    score = info["score"]
    x_pos = info["x_pos"]
    return x_pos * 10 + score


def get_initial_sequences(sequence_length, num_sequences, number_actions):
    """
    Returns a list of sequences completely randomized.

    :param sequence_length: Length of each sequence
    :param num_sequences: Number of sequences to make
    :param number_actions: Number of action that are possible
    :return: Randomized list of sequences
    """
    return [
        [random.randint(0, number_actions - 1) for _ in range(sequence_length)]
        for _ in range(num_sequences)
    ]
