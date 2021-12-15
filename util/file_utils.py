import csv
import json


def add_csv_rows(rows_to_add, file_name):
    """
    Adds rows to existing CSV file
    """
    try:
        with open(f"./data/{file_name}", "a") as csv_write:
            writer = csv.writer(csv_write)
            writer.writerows(rows_to_add)
        return True
    except Exception as e:
        print(f"Error adding data to file {file_name}.\n Failed with exception: {e}")
        return False


def create_csv_file(headers, file_name):
    """
    Creates csv file with the given headers and file name
    """
    with open(f"./data/{file_name}", "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)


def overwrite_seq_file(best_sequence, generation, filename):
    """
    Overwrites the best sequence file
    """
    try:
        with open(f"./data/{filename}", "w") as file_to_write:
            file_to_write.write(f"{generation}\n")
            file_to_write.write(", ".join(str(action) for action in best_sequence))
        return True
    except Exception as e:
        print(
            f"Error adding data to file {filename}. For gen {generation}\n Failed with exception: {e}"
        )
        return False


def overwrite_policy_file(policy, generation, filename):
    """
    Overwrites the best policy file
    """
    try:
        with open(f"./data/{filename}", "w") as file_to_write:
            file_to_write.write(f"{generation}\n")
            str_to_add = json.dumps(policy)
            file_to_write.write(str_to_add)
        return True
    except Exception as e:
        print(
            f"Error adding data to file {filename}. For gen {generation}\n Failed with exception: {e}"
        )
        return False
