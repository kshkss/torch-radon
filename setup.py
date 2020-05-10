from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
        packages=find_packages('src'),
        package_dir={'': 'src'},
        ext_modules=[
            CUDAExtension('radon_cuda', ['lib/radon_cuda.cpp', 'lib/radon_forward.cu', 'lib/radon_backward.cu'])],
        cmdclass={'build_ext': BuildExtension},
        python_requires='>=3.6, <4',
        install_requires=['torch>=1.5.0'],
)

