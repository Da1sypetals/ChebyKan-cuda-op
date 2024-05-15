# Chebyshev KAN CUDA implementation

## References:

- [知乎: 用 Cuda 实现 PyTorch 算子](https://zhuanlan.zhihu.com/p/595851188).
- [Original implementation of Chebyshev KAN](https://github.com/SynodicMonth/ChebyKAN/tree/7eb83592042e5d23c2aa338a0d3df9b54b5b6b19).

## Note

There is a problem with backprop when replacing `bmm` with `einsum` [see here](https://github.com/Da1sypetals/ChebyKan-cuda-op/blob/6f96f2c0084feef4730c21ba63f2fd69e5a35ec6/dfReturnChebyKan/layer.py#L36C12-L36C13), probably because I created tensors the wrong way. If anyone know how to fix that, please help, a lot of thanks.

## Start

1. Install

```bash
pip install -e .
```

> Make sure the version of `nvcc` in `PATH` is compatible with your current PyTorch version (it seems minor version difference is OK).

2. Run

- Run test on MNIST;

```bash
python cheby_test.py
```

- Or you can make your own net:

```python
from fasterCuChebyKan.layer import ChebyKANLayer
import torch.nn as nn

class ChebyNet(nn.Module):
    def __init__(self):
        super(ChebyNet, self).__init__()
        self.layer1 = ChebyKANLayer(28*28, 256, 4)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = ChebyKANLayer(256, 10, 4)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.lalyer1(x)
        x = self.ln1(x)
        x = self.layer2(x)
        return x
```
