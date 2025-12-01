#!/usr/bin/env python3
import sys
import csv
import re
from pathlib import Path

def parse_stream_out(filename):
    sizes = []
    gbs = []

    pattern = re.compile(r"STREAM triad of size\s+(\d+).*?([\d.]+)\s*GB/s")

    with open(filename, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                size = int(m.group(1))
                gb_value = float(m.group(2))
                sizes.append(size)
                gbs.append(gb_value)

    return sizes, gbs

def write_csv(output_filename, sizes, gbs):
    with open(output_filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Size", "GB/s"])
        for s, g in zip(sizes, gbs):
            writer.writerow([s, g])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_file.out>")
        sys.exit(1)

    input_file = sys.argv[1]
    input_path = Path(input_file)
    output_file = input_path.with_suffix(".csv")  # sto2.out -> sto2.csv

    sizes, gbs = parse_stream_out(input_file)
    write_csv(output_file, sizes, gbs)
    print(f"Saved {len(sizes)} entries to {output_file}")
