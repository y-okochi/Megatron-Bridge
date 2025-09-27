# Performance

This page provides performance benchmarks for large language models using Megatron-Bridge across different GPU systems and configurations.

## Performance Summary for Large Language Models

Below are performance benchmarks for various large language models. These results were obtained using a version of the performance recipes available [here](https://github.com/NVIDIA/Megatron-Bridge/tree/main/scripts/performance).

The performance data includes:

- **Pre-training Performance**: Throughput metrics for various model sizes and architectures
- **System Configurations**: Results across different GPU systems (DGX-GB200, DGX-B200, DGX-H100)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8, MXFP8)

### Nomenclature

- **GBS**: Global Batch Size
- **MBS**: Micro Batch Size
- **FSDP**: Fully Sharded Data Parallel
  - FSDP = 1: use FSDP
  - FSDP = 0: use DDP (Distributed Data Parallel)
- **TP**: Tensor Parallel Size
- **PP**: Pipeline Parallel Size
- **CP**: Context Parallel Size
- **VP**: Virtual Pipeline Parallel Size
- **EP**: Expert Parallel Size
- **GA**: Number of Gradient Accumulations

### Performance Metrics

Performance is measured using:
- **Tokens/sec/GPU**: Throughput per GPU
- **Model TFLOP/sec/GPU**: Model floating-point operations per second per GPU

### Pre-Training Performance

Coming soon..
