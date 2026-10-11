# Compile profile (E12)

## Abstract

Build tips for throughput: `-O2`/`-O3`, OpenMP, optional `ARCH=-mavx2`, `make` (CPU stub) vs `make cuda` / `make cuda-autotrain`.

## Commands

```bash
make                 # CPU ./ann
make cuda            # CUDA ./ann + cuBLAS
make autotrain       # CPU ./autotrain
make cuda-autotrain  # CUDA ./autotrain
make test            # CPU tests
```

Override: `OPT=-O3 ARCH='-march=native' make cuda`
