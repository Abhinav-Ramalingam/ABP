#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as mticker




# Cache sizes in floats
CACHE_FLOATS = {
    "L1 cache (32K)": 32*1024 // 12,
    "L2 cache (256K)": 256*1024 // 12,
    "L3 cache (20480K)": 20480*1024 // 12
}


def plot_csvs(csv_files):
    plt.figure(figsize=(10,6))
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "MatrixSize" not in df.columns or "GFlops" not in df.columns:
            print(f"Skipping {csv_file}, missing required columns")
            continue
        label = input(f"Enter label for {Path(csv_file).name}: ")
        plt.plot(df["MatrixSize"], df["GFlops"], marker='o', label=label)

    # # Add vertical lines for cache sizes
    # for cache, size in CACHE_FLOATS.items():
    #     plt.axvline(x=size, color='red', linestyle='--', alpha=0.7)
    #     plt.text(size, plt.ylim()[1]*0.95, cache, rotation=90, verticalalignment='top', color='red')

    
    plt.xlabel("MatrixSize") 
    plt.ylabel("GFlops")
    title = input("Enter plot title: ")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    ax = plt.gca()
    TICK_INTERVAL = int(input("Enter x-axis tick interval (e.g., 500): "))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(TICK_INTERVAL))
    plt.tight_layout()
    output_file = input("Enter output filename (e.g., plot.png): ")
    output_file = "./plots/" + output_file
    plt.savefig(output_file, dpi=300)
    print(f"Saved combined plot to {output_file}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv1> <csv2> ...")
        sys.exit(1)

    csv_files = sys.argv[1:]
    plot_csvs(csv_files)

