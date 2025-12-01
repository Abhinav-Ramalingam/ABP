#!/usr/bin/env python3
import sys

def read_numbers(filename):
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def main():
    if len(sys.argv) != 3:
        print("Usage: {} file1 file2".format(sys.argv[0]))
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    nums1 = read_numbers(file1)
    nums2 = read_numbers(file2)

    if len(nums1) != len(nums2):
        print("Files have different number of lines!")
        sys.exit(1)

    diffs = [abs(a - b) for a, b in zip(nums1, nums2)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    num_nonzero = sum(1 for d in diffs if d > 1e-6)

    print("Max absolute difference: {:e}".format(max_diff))
    print("Mean absolute difference: {:e}".format(mean_diff))
    print("Number of differing lines (i.e., Difference <= 1e-6): {} / {}".format(num_nonzero, len(nums1)))


if __name__ == "__main__":
    main()