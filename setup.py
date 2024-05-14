from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ['CUDA_HOME'] = '/usr/local/cuda-12'


setup(
    name='CudaDemo',
    packages=find_packages(),
    version='0.1.0',
    author='Yuxue Yang',
    ext_modules=[
        CUDAExtension(
            'sum_single', # operator name
            ['./ops/src/reduce_sum/sum.cpp',
             './ops/src/reduce_sum/sum_cuda.cu',]
        ),
        CUDAExtension(
            'sum_double',
            ['./ops/src/sum_two_arrays/two_sum.cpp',
             './ops/src/sum_two_arrays/two_sum_cuda.cu',]
        ),
        CUDAExtension(
            'linear_op', # operator name
            ['./ops/src/linear/linear.cpp',
             './ops/src/linear/linear_cuda.cu',]
        ),
        CUDAExtension(
            'cheby_ops', # operator name
            ['./ops/src/cheby/cheby.cpp',
             './ops/src/cheby/cheby_cuda.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)