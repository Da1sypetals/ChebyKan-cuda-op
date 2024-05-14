# CudaDemo

> References:
    - [知乎: 用 Cuda 实现 PyTorch 算子](https://zhuanlan.zhihu.com/p/595851188).
    - [Original implementation of Chebyshev KAN](https://github.com/SynodicMonth/ChebyKAN/tree/7eb83592042e5d23c2aa338a0d3df9b54b5b6b19).

## Start

1. Install

```bash
pip install -e .
```

> Make sure the version of `nvcc` in `PATH` is compatible with your current PyTorch version (it seems minor version difference is OK).

2. Run

```bash
python cheby_test.py
```