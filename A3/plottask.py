#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt

# X_LABEL = "Size of list of matrices (Elements)"
# Y_LABEL = "Application Compute Bandwidth (GB/s)"
# TITLE = "Finite Element Bandwidth Comparison between CPU and GPU, float and double"
# OUTPUT_FILE = "plottask2gbps1.png"
# FILES = ["outputs/task2cpu.csv", "outputs/task2gpu.csv", "outputs/task2cpud.csv", "outputs/task2gpud.csv"]
# LABELS = ["CPU Bandwidth (float)", "GPU Bandwidth (float)", "CPU Bandwidth (double)", "GPU Bandwidth (double)"]

# Y_COLUMN = 3 # GB/s

# X_LABEL = "Size of list of matrices (Elements)"
# Y_LABEL = "Application Performance (Million Elements/s)"
# TITLE = "Finite Element Performance Comparison between CPU and GPU, float and double"
# OUTPUT_FILE = "plottask2meps.png"
# FILES = ["outputs/task2cpu.csv", "outputs/task2gpu.csv", "outputs/task2cpud.csv", "outputs/task2gpud.csv"]
# LABELS = ["CPU Performance (float)", "GPU Performance (float)", "CPU Performance (double)", "GPU Performance (double)"]

# Y_COLUMN = 1  # M Elements/s


X_LABEL = "Size of list of matrices (Elements)"
Y_LABEL = "Floating Point Performance (GFlop/s)"
TITLE = "Finite Element FLOPS Comparison between CPU and GPU, float and double"
OUTPUT_FILE = "plottask2gflops.png"
FILES = ["outputs/task2cpu.csv", "outputs/task2gpu.csv", "outputs/task2cpud.csv", "outputs/task2gpud.csv"]
LABELS = ["CPU Performance (float)", "GPU Performance (float)", "CPU Performance (double)", "GPU Performance (double)"]

Y_COLUMN = 2  # GFlop/s


def main():

    plt.figure(figsize=(8,6))

    for csv_file, label in zip(FILES, LABELS):
        df = pd.read_csv(csv_file)
        x = df.iloc[:,0]  # N_elements
        y = df.iloc[:,Y_COLUMN]  # M Elements/s
        plt.plot(x, y, marker='o', label=label)

    plt.xscale('log') 
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title(TITLE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(OUTPUT_FILE, dpi=300)
    print("Plot saved as", OUTPUT_FILE)

if __name__ == "__main__":
    main()
