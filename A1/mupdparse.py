#!/usr/bin/env python3
import sys
import csv
import re
from pathlib import Path

def parse_stream_out(filename):
    sizes = []
    mupd = []

    # Regex captures size and MUPD/s
    pattern = re.compile(r"STREAM triad of size\s+(\d+).*?([\d.eE+-]+)\s*MUPD/s")

    with open(filename, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                size = int(m.group(1))
                mupd_value = float(m.group(2))
                sizes.append(size)
                mupd.append(mupd_value)

    return sizes, mupd

def write_csv(output_filename, sizes, mupd):
    with open(output_filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Size", "MUPD/s"])
        for s, m in zip(sizes, mupd):
            writer.writerow([s, m])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_file.out>")
        sys.exit(1)

    input_file = sys.argv[1]
    input_path = Path(input_file)
    output_file = input_path.stem + "MUPD.csv"  # something.out -> somethingMUPD.csv

    sizes, mupd = parse_stream_out(input_file)
    write_csv(output_file, sizes, mupd)
    print(f"Saved {len(sizes)} entries to {output_file}")
