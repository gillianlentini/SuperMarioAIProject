import math
import multiprocessing
import random

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def get_starting_generation(number_in_generation, max_x, max_y):
    """
    Creates a starting generation (list of policies)

    :param number_in_generation: Number of policies per generation
    :param max_x: Max x position possible
    :param max_y: Max y position possible
    :return: List of policies
    """
    # parallelized for faster computation
    pool = multiprocessing.Pool()
    policies = [
        pool.apply(_get_random_policy, args=(max_x, max_y))
        for _ in range(number_in_generation)
    ]
    pool.close()
    return policies


def _get_random_policy(max_x, max_y):
    """
    Gets a random policy

    :param max_x: Max x to consider in policy
    :param max_y: Max y to consider in policy
    :return: Random policy
    """
    policy = {}
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            policy[(x, y)] = random.randint(0, len(SIMPLE_MOVEMENT) - 1)
    return policy


def run_policy(policy, generation, number, world, stage):
    """
    Runs the given policy and returns information on it

    :param policy: the policy to evaluate
    :param generation: the generation this policy belongs to
    :param number: the id of this policy in the generation
    :param world: the world this policy is being evaluated in
    :param stage: the stage this policy is being evaluated in
    :return: Negated fitness level, data on the run, and policy
    """
    stage_name = f"SuperMarioBros-{world}-{stage}-v3"
    env = gym_super_mario_bros.make(stage_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.reset()
    info = {
        "x_pos": 40,
        "y_pos": 79,
        "time": 400,
    }
    max_x = 0
    while True:
        x = info["x_pos"]
        y = info["y_pos"]
        max_x = max(x, max_x)
        curr_action = policy.get((x, y), False)
        if not curr_action:
            curr_action = random.randint(0, len(SIMPLE_MOVEMENT) - 1)
            policy[(x, y)] = curr_action
        state, reward, done, info = env.step(curr_action)
        if done:
            break
    env.close()
    fitness = _get_fitness(info["score"], max_x)
    data = [
        generation + 1,
        number + 1,
        fitness,
        info["score"],
        info["world"],
        info["stage"],
        max_x,
    ]
    return -fitness, data, policy


def _get_fitness(score, max_x):
    """
    Returns a value of fitness for a policy based on score and max_x returned from run.

    :param score: The score the policy finished with
    :param max_x: the max x pos the policy reached in its run
    :return: The fitness score
    """
    return max_x * 5 + score


def do_crossover(policies_to_breed, number_to_make, max_x, max_y):
    """
    Does the crossover portion of GA. We have made a design decision here to use a less biologically informed
    "crossover." Instead of a crossover "point" we take a random move from the best policies for each "gene."

    :param policies_to_breed: The policies that will "breed"
    :param number_to_make: The number of policies to create
    :param max_x: The max x for this level
    :param max_y: The max y for this level
    :return:
    """
    pool = multiprocessing.Pool()
    new_generation = [
        pool.apply(_create_one_child, args=(policies_to_breed, max_x, max_y))
        for _ in range(number_to_make)
    ]
    pool.close()
    return new_generation


def _create_one_child(policies_to_breed, max_x, max_y):
    """
    Creates a single child policy from the parents given.

    :param policies_to_breed: Policies to breed
    :param max_x: Max x in the policy
    :param max_y: Max y in the policy
    :return: The "child" policy
    """
    child = {}
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            child[(x, y)] = random.choice(policies_to_breed)[(x, y)]
    return child


def mutate(policies, mutation_rate, max_x, max_y):
    """
    Mutates some number of genes (based on rate) in the given policies.

    :param policies: The policies to mutate
    :param mutation_rate: The rate to make mutations at
    :param max_x: The max x in the policies
    :param max_y: The max y in the policies
    :return: The policy list with mutations
    """
    number_mutations = math.ceil(mutation_rate * max_x * max_y)
    for policy in policies:
        for _ in range(number_mutations):
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            policy[(x, y)] = random.randint(0, len(SIMPLE_MOVEMENT) - 1)
    return policies
