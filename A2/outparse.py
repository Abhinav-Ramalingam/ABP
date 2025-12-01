import sys
import re
import csv

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input_log> <output_csv>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

pattern = re.compile(r"Matrix size:\s*(\d+)\s*x\s*\d+.*Bandwidth:\s*([\d.eE+-]+) GB/s")

data = []

with open(input_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            size = int(match.group(1))
            bw = float(match.group(2))
            data.append((size, bw))

# Write CSV
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["MatrixSize", "Bandwidth_GBps"])
    writer.writerows(data)

print(f"CSV written to {output_file} with {len(data)} rows.")
