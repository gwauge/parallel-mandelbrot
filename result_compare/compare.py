import numpy as np
import hashlib
import os
from collections import defaultdict

results_folder = "results"
baseline_file_sp = "baseline_sp_O0"
baseline_file_dp = "baseline_dp_O0"


def read_result(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        matrix = np.array([list(map(int, row.split(','))) for row in data.strip().split('\n')])
        return matrix

# Optimized function to compute the hash of a file using a larger buffer size
def compute_file_hash(file_path, hash_algorithm='sha256', chunk_size=65536):
    hash_func = hashlib.new(hash_algorithm)  # Choose hash algorithm
    
    with open(file_path, 'rb') as file:
        while chunk := file.read(chunk_size):  # Read in larger chunks (e.g., 64KB)
            hash_func.update(chunk)
    
    return hash_func.hexdigest()  # Return the hexadecimal representation of the hash

file_hashes = defaultdict(list) 
# Loop through all files in the results folder (excluding the baseline file)
for filename in sorted(os.listdir(results_folder)):
    file_hashes[compute_file_hash(os.path.join(results_folder, filename))].append(filename)


baseline_sp = read_result(os.path.join(results_folder, baseline_file_sp))
baseline_dp = read_result(os.path.join(results_folder, baseline_file_dp))
sp_groups = 0
dp_groups = 0

baseline_difference = np.abs(baseline_dp - baseline_sp)
baseline_difference_sum = np.sum(baseline_difference)
baseline_difference_average = np.mean(baseline_difference)
print(f"Difference baseline_dp baseline_sp: {baseline_difference_sum} {baseline_difference_average}")

for i, hash_group in enumerate(sorted(file_hashes.items(), key=lambda x: len(x[1]))):
    _hash, group = hash_group
    # Construct the full path of the current result file
    file_path = os.path.join(results_folder, group[0])
    
    # Read the current result file into a NumPy array
    result_matrix = read_result(file_path)

    baseline_matrix = baseline_dp
    if  "_sp_" in group[0]:
        baseline_matrix = baseline_sp
        sp_groups += 1
    else:
        dp_groups += 1

    difference = np.abs(baseline_matrix - result_matrix)
    difference_sum = np.sum(difference)
    difference_average = np.mean(difference)
    print(f"Group {_hash} {i}:")
    print(f'\t{"\n\t".join(group)}')
    print(f"\tDifference Sum: {difference_sum} {difference_average}")


print(f"\nTotal groups: {len(file_hashes)} sp: {sp_groups} dp: {dp_groups}")