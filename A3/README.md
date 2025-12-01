# README — Running the Finite Element Matrix Benchmark

## Synopsis

This benchmark computes small ( 4 \times 4 ) element matrices for a given number of finite elements using Kokkos.
It measures computation throughput and host–device transfer bandwidth.

## Usage

```bash
./fed [options]
```
(Provided you compiled the file)

### Options

| Option           | Description                                                                   | Default      |
| :--------------- | :---------------------------------------------------------------------------- | :----------- |
| `-N <num>`       | Starting number of elements                                                   | `1000`       |
| `-Nlimit <num>`  | Maximum number of elements (upper bound for sweep). If omitted, same as `-N`. | same as `-N` |
| `-dtype <type>`  | Data type for computation: `float` or `double`                                | `float`      |
| `-layout <type>` | Memory layout for Kokkos Views: `left` or `right`                             | `left`       |
| `-h`             | Show help                                                                     | —            |

## Additional files
1. `Assigment_3_ABP.pdf` - Report for the assignment
2. `plottask.py` - Script used to generate the plots