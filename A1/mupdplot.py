#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
        if "Size" not in df.columns or "MUPD/s" not in df.columns:
            print(f"Skipping {csv_file}, missing required columns")
            continue
        label = input(f"Enter label for {Path(csv_file).name}: ")
        plt.plot(df["Size"], df["MUPD/s"], marker='o', label=label)

    # Optional: Add vertical lines for cache sizes
    # for cache, size in CACHE_FLOATS.items():
    #     plt.axvline(x=size, color='red', linestyle='--', alpha=0.7)
    #     plt.text(size, plt.ylim()[1]*0.95, cache, rotation=90, verticalalignment='top', color='red')

    plt.xlabel("Size")
    plt.ylabel("MUPD/s")
    title = input("Enter plot title: ")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.xscale("log")  # optional: log scale for size
    plt.tight_layout()
    output_file = input("Enter output filename (e.g., plot.png): ")
    plt.savefig(output_file, dpi=300)
    print(f"Saved combined plot to {output_file}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv1> <csv2> ...")
        sys.exit(1)

    csv_files = sys.argv[1:]
    plot_csvs(csv_files)
