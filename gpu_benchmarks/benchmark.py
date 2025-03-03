import os
import subprocess
import csv
import sys

def is_executable(file_path):
    return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

def run_binary(binary_path, num_runs, capture_output=True):
    outputs = []
    for i in range(num_runs):
        print(f"Run {i}")
        try:
            result = subprocess.run([binary_path], capture_output=capture_output, text=True, check=True)
            if capture_output:
                # Strip any whitespace/newlines from stdout.
                outputs.append(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error running {binary_path} on run {i+1}: {e}")
    return outputs

def main():
    binaries_dir = "../build/bin/rocm"
    csv_filename = "results.csv"
    
    # Open CSV file for writing the results.
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header row.
        csv_writer.writerow(['filename', 'output'])
        
        # Iterate over each file in the directory.
        for filename in os.listdir(binaries_dir):
            file_path = os.path.join(binaries_dir, filename)
            if is_executable(file_path):
                print(f"Processing {filename}...")
                # Warmup: run 5 times (outputs are discarded).
                run_binary(file_path, 5, capture_output=False)
                # Measurement: run 10 times and capture output.
                measurement_outputs = run_binary(file_path, 10)
                # Write each measurement result to the CSV.
                for output in measurement_outputs:
                    csv_writer.writerow([filename, output])
    
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    main()
