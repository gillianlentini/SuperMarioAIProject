"""
Script for running Genetic Algorithm that learns an Optimal Policy. A unique state is represented by an (x, y) pair on
the map. Although time contributes to making a state "unique", There are 255+ y-positions, ~5,000 x-positions, and
400 ticks on the clock in the game. This means random policies with 510,000,000 moves must be generated. Removing this
dimension (as many things aren't time dependant) makes computational space and time more realistic
(though there is a trade off with finding a more optimal policy).
"""
import argparse
import heapq
import multiprocessing

from agents import genetic_policy_agent
from util.file_utils import add_csv_rows, create_csv_file, overwrite_policy_file


def run_genetic_algorithm(
    world,
    stage,
    number_of_generations,
    number_per_gen,
    number_to_breed,
    mutation_rate,
    mutation_rate_decay,
    csv_file_name,
    policy_file_name,
    max_x,
    max_y,
):
    """
    Runs the Genetic Algorithm.
    :param world: The world to run the GA on (1-8)
    :param stage: The stage within the world to run the GA on (1-4)
    :param number_of_generations: Number of generations to run
    :param number_per_gen: Number of policies per generation
    :param number_to_breed: Number of policies allowed to breed
    :param mutation_rate: Mutation rate
    :param mutation_rate_decay: Decay of mutation rate over time
    :param csv_file_name: CSV file to store data in
    :param policy_file_name: TXT file to store policy in
    :param max_x: Max x to generate a policy for
    :param max_y: Max y to generate a policy for
    """
    print("STARTING...")
    fields = [
        "Generation",
        "Sequence",
        "Fitness",
        "Score",
        "World",
        "Stage",
        "Max_X_Position",
    ]
    create_csv_file(fields, csv_file_name)

    current_generation = genetic_policy_agent.get_starting_generation(
        number_per_gen, max_x, max_y
    )
    max_fitness_encountered = 0
    print(f"Created initial sequences!")

    for i in range(number_of_generations):
        print(f"STARTING GENERATION: {i + 1}")
        # runs and evaluates fitness of each policy
        multiprocessing_pool = multiprocessing.Pool()
        best_in_generation = [
            multiprocessing_pool.apply(
                genetic_policy_agent.run_policy,
                args=(current_generation[j], i, j, world, stage),
            )
            for j in range(number_per_gen)
        ]
        multiprocessing_pool.close()
        print(f"Finished processing sequences for gen: {i + 1}")

        # save data for this generation
        data_from_ga = []
        for neg_fitness, data, policy in best_in_generation:
            data_from_ga.append(data)
        add_csv_rows(data_from_ga, csv_file_name)

        # finds the most fit in this generation to breed
        heapq.heapify(best_in_generation)
        to_breed = []
        for parent_index in range(number_to_breed):
            neg_fitness, data, policy = heapq.heappop(best_in_generation)
            if parent_index == 0 and abs(neg_fitness) > max_fitness_encountered:
                max_fitness_encountered = max(max_fitness_encountered, abs(neg_fitness))
                overwrite_policy_file(policy, i + 1, policy_file_name)
            to_breed.append(policy)

        # crossover
        print(f"Crossover for Generation: {i + 1}")
        current_generation = genetic_policy_agent.do_crossover(
            to_breed, number_per_gen, max_x, max_y
        )
        # mutate, and decrement mutation rate
        print(f"Mutations for Generation: {i + 1}")
        current_generation = genetic_policy_agent.mutate(
            current_generation, mutation_rate, max_x, max_y
        )
        mutation_rate *= mutation_rate_decay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algo for SMB.")
    parser.add_argument("--num_in_gen", required=True, type=int)
    parser.add_argument("--generations", required=True, type=int)
    parser.add_argument("--world", required=True, type=int)
    parser.add_argument("--stage", required=True, type=int)
    args = parser.parse_args()
    number_per_gen = max(2, args.num_in_gen)
    number_of_generations = args.generations
    world = args.world
    stage = args.stage
    csv_file_name = f"ga_policy_data_{world}_{stage}.csv"
    policy_file_name = f"best_seq_ga_policy_{world}_{stage}.txt"

    max_y = 255
    max_x = 5000  # this is a guess
    number_to_breed = max(2, int(args.num_in_gen * 0.1))
    mutation_rate = 0.4
    mutation_rate_decay = 0.95
    world = 1
    stage = 1

    run_genetic_algorithm(
        world,
        stage,
        number_of_generations,
        number_per_gen,
        number_to_breed,
        mutation_rate,
        mutation_rate_decay,
        csv_file_name,
        policy_file_name,
        max_x,
        max_y,
    )
