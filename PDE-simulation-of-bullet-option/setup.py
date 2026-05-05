from setuptools import setup
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_extension',
    ext_modules=cythonize([
        CUDAExtension(
                name='cuda_extension',
                sources=['wrapper.pyx', 'wrappers.cu', 'MC.cu', 'utils.cu'],
                extra_compile_args={'cxx': ['-g'],
                                    'nvcc': ['-O2']},
                language='c++',
                include_dirs=["./include"],
                extra_link_args=['-Wl,--no-as-needed', '-lcuda'])
    ]),
    cmdclass={
        'build_ext': BuildExtension
    })