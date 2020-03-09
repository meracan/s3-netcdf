### Memory Profiling and Optimization

```bash
conda install memory_profiler
PYTHONPATH=../s3-netcdf/ python3 bench/bench_1.py
```

##### Writing
1M x 1M
| Partition size  | step | hrs  |
|-----------------|------|------|
|[1 250000] | 1 |52.6hrs|
|[1 250000] | 4 |59.5hrs|
|[1 250000] | 8 |70.4hrs|
|[ 2 1000000] | 1 |61.6hrs|
|[ 2 1000000] | 2 |51.2hrs|
|[ 2 1000000] | 4 |52.1hrs|
|[ 2 1000000] | 16 |57.1hrs|
|[ 2 1000000] | 32 |63.2hrs|
|[ 3 1000000] | 1 |71.4hrs|
|[ 3 1000000] | 3 |52.3hrs|
|[ 3 1000000] | 6 |52.3hrs|
|[ 3 1000000] | 9 |52.9hrs|
|[ 3 1000000] | 27 |59.0hrs|
|[27 1000000] | 1 |357.4hrs|
|[27 1000000] | 27 |51.5hrs|

##### Reading

##### Memory consumptions

1x1E6 =>  0.2GB

3x1E6 =>  0.4GB

6x1E6 =>  0.5GB

9x1E6 =>  0.7GB

27x1E6 => 2.5GB

36x1E6 => 3.0GB


### Single and multi-threaded Application
The application is single threaded. A threaded application is possible by modifying the `f()` function in `__getitem__` and `__setitem__`. However,the IO (reading-uncompressing and writing-compressing) NetCDF files locally is the limited factor.

### Single-Core
#### Writing
The following sample performance test was done by generating  arrays using numpy and writing locally using different ec2.

| Test            | r5.large | r5.xlarge | r5.2xlarge |
|-----------------|----------|-----------|------------|
| [0.001M x 0.1M] |          |           |            |
| [0.001M x 1.0M] |          |           |            |
| [0.01M x 0.1M]  |          |           |            |
| [0.01M x 1.0M]  |          |           |            |
| [0.1M x 0.1M]   |          |           |            |
| [0.1M x 1.0M]   |          |           |            |
| [1.0M x 0.1M]   |          |           |            |
| [1.0M x 1.0M]   |          |           |            |

#### Reading
| Test         | r5.large | r5.xlarge | r5.2xlarge |
|--------------|----------|-----------|------------|
| [1 x 0.1M]   |          |           |            |
| [1 x 1.0M]   |          |           |            |
| [10 x 0.1M]  |          |           |            |
| [10 x 1.0M]  |          |           |            |

### Multiple-Core

TODO : Example script
#### Writing
#### Reading