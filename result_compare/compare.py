import numpy as np
import os

results_folder = "results"
baseline_file = "baseline_dp_O3"

def read_result(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        matrix = np.array([list(map(int, row.split(','))) for row in data.strip().split('\n')])
        return matrix

baseline_matrix = read_result(os.path.join(results_folder, baseline_file))
print(f"Baseline file is {baseline_file}")

# Loop through all files in the results folder (excluding the baseline file)
for filename in sorted(os.listdir(results_folder)):
    
    # Construct the full path of the current result file
    file_path = os.path.join(results_folder, filename)
    
    # Read the current result file into a NumPy array
    result_matrix = read_result(file_path)

    # Compare the result matrix with the baseline matrix
    if np.array_equal(result_matrix, baseline_matrix):
        print(f"File {filename} matches the baseline.")
    #else:
    #    print(f"File {filename} does NOT match the baseline.")

