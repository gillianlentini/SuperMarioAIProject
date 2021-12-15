"""
Script for running Genetic Algorithm based on Sequences. Random sequences of moves are created, evaluated for fitness,
sorted by fitness level, and then allowed to "breed."
"""
import argparse
import heapq
import multiprocessing
from agents import genetic_sequence_agent
from util.file_utils import add_csv_rows, create_csv_file, overwrite_seq_file
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def run_genetic_algorithm(
    world,
    stage,
    generations,
    number_of_sequences,
    sequence_length,
    number_to_breed,
    number_of_crossovers,
    mutation_rate,
    mutation_rate_decay,
    csv_file_name,
    seq_file_name,
):
    """
    Runs the genetic Algorithm

    :param world: World to run it on (False if all worlds)
    :param stage: Stage to run it on (False if all stages
    :param generations: Number of generations
    :param number_of_sequences: Number of sequences to make
    :param sequence_length: Length of actions in sequence
    :param number_to_breed: Number of sequences (parents) allowed to breed
    :param number_of_crossovers: Number of crossovers in breeding
    :param mutation_rate: Rate of mutations
    :param mutation_rate_decay: Rate mutation rate decays at
    :param csv_file_name: CSV file to output info to
    :param seq_file_name: text file to output best sequence to
    :return:
    """
    fields = [
        "Generation",
        "Sequence",
        "Fitness",
        "Score",
        "World",
        "Stage",
        "X Position",
    ]
    create_csv_file(fields, csv_file_name)
    # initial sequences are random moves
    sequences = genetic_sequence_agent.get_initial_sequences(
        sequence_length, number_of_sequences, len(SIMPLE_MOVEMENT)
    )
    max_fitness_encountered = 0

    for i in range(generations):
        print(f"STARTING GENERATION: {i + 1}")

        # run each member of the generation
        multiprocessing_pool = multiprocessing.Pool()
        best_sequences = [
            multiprocessing_pool.apply(
                genetic_sequence_agent.run_sequence_parallel,
                args=(sequences[j], i, j, stop_when_dead, world, stage),
            )
            for j in range(number_of_sequences)
        ]
        multiprocessing_pool.close()
        print(f"Finished Initial Processing for Generation: {i + 1}")

        data_from_ga = []
        # save data for this generation
        for neg_fitness, max_reached, data, seq in best_sequences:
            data_from_ga.append(data)
        add_csv_rows(data_from_ga, csv_file_name)
        # reset for next generation

        # order sequences
        heapq.heapify(best_sequences)
        to_breed = []
        max_steps = 0
        # choose the most fit for breeding
        for seq_num in range(number_to_breed):
            neg_fitness, max_reached, data, sequence = heapq.heappop(best_sequences)
            if seq_num == 0:
                # if this is the best so far we want to save this!
                if abs(neg_fitness) > max_fitness_encountered:
                    max_fitness_encountered = max(
                        max_fitness_encountered, abs(neg_fitness)
                    )
                    overwrite_seq_file(sequence, i + 1, seq_file_name)
            max_steps = max(max_steps, max_reached)
            to_breed.append(sequence)

        # crossover
        print(f"Crossover for Generation: {i + 1}")
        sequences = genetic_sequence_agent.crossover(
            max_steps, number_of_crossovers, to_breed, number_of_sequences
        )
        # mutate, and decrement mutation rate
        print(f"Mutations for Generation: {i + 1}")
        sequences = genetic_sequence_agent.mutate(sequences, max_steps, mutation_rate)
        mutation_rate *= mutation_rate_decay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algo for SMB.")
    parser.add_argument("--num_seq", required=True, type=int)
    parser.add_argument("--generations", required=True, type=int)
    parser.add_argument("--world", default=False, type=int)
    parser.add_argument("--stage", default=False, type=int)
    args = parser.parse_args()
    number_of_sequences = max(2, args.num_seq)
    generations = args.generations
    world = args.world
    stage = args.stage
    if not (world and stage):
        sequence_length = 20000
        csv_file_name = "ga_data_all_stages.csv"
        seq_file_name = "best_seq_all.txt"
    else:
        sequence_length = 5000
        csv_file_name = f"ga_data_{world}_{stage}.csv"
        seq_file_name = f"best_seq_{world}_{stage}.txt"

    number_to_breed = max(2, int(number_of_sequences * 0.1))
    number_of_crossovers = 1
    mutation_rate = 0.4
    mutation_rate_decay = 0.95
    stop_when_dead = True

    run_genetic_algorithm(
        world,
        stage,
        generations,
        number_of_sequences,
        sequence_length,
        number_to_breed,
        number_of_crossovers,
        mutation_rate,
        mutation_rate_decay,
        csv_file_name,
        seq_file_name,
    )
