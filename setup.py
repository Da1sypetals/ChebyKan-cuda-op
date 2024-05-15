from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ['CUDA_HOME'] = '/usr/local/cuda-12'


setup(
    name='cuChebyKan',
    packages=find_packages(),
    version='0.0.0',
    author='Yuxue Yang',
    ext_modules=[
        CUDAExtension(
            'cheby_ops', # operator name
            ['./cpp/cheby.cpp',
             './cpp/cheby_cuda.cu',]
        ),
        CUDAExtension(
            'deg_first_cheby_ops', # operator name
            ['./deg_first_cpp/cheby.cpp',
             './deg_first_cpp/cheby_cuda.cu',]
        ),
        CUDAExtension(
            'dfr_cheby_ops', # operator name
            ['./df_return_cpp/cheby.cpp',
             './df_return_cpp/cheby_cuda.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)