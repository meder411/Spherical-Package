import torch
import os
import os.path as osp
import pathlib
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

module = 'mapped-conv'  # ['tangent-images', 'mapped-conv']
compute_arch = 'compute_61'  # Compute architecture for GPU
use_ninja = False  # [True, False]
additional_includes = []  # List of any additional include directories

# Default search paths
abs_path_here = pathlib.Path(__file__).parent.absolute()
include_dir = osp.join(abs_path_here, 'ext_modules/include')
src_cpp_dir = osp.join(abs_path_here, 'ext_modules/src/cpp')
src_cuda_dir = osp.join(abs_path_here, 'ext_modules/src/cuda')
eigen_include_dir = ['/usr/include/eigen3', '/usr/local/include/eigen3']


def extension(name,
              source_basename,
              src_cpp_dir,
              include_dir,
              src_cuda_dir=None,
              compute_arch='compute_61',
              additional_include_dirs=[],
              cxx_compile_args=[],
              nvcc_compile_args=[]):
    '''Create a build extension. Use CUDA if available, otherwise C++ only'''

    if torch.cuda.is_available() and src_cuda_dir is not None:
        return CUDAExtension(
            name=name,
            sources=[
                osp.join(src_cpp_dir, source_basename + '.cpp'),
                osp.join(src_cuda_dir, source_basename + '.cu'),
            ],
            include_dirs=[include_dir] + additional_include_dirs,
            extra_compile_args={
                'cxx': ['-fopenmp', '-O3'] + cxx_compile_args,
                'nvcc':
                ['--gpu-architecture=' + compute_arch] + nvcc_compile_args
            },
        )
    else:
        return CppExtension(
            name=name,
            sources=[
                osp.join(src_cpp_dir, source_basename + '.cpp'),
            ],
            include_dirs=[include_dir] + additional_include_dirs,
            define_macros=[('__NO_CUDA__', None)],
            extra_compile_args={
                'cxx': ['-fopenmp', '-O3'] + cxx_compile_args,
                'nvcc': [] + nvcc_compile_args
            },
        )


basic_modules = [
    extension('_enums', 'enum_export', src_cpp_dir, include_dir),

    # ------------------------------------------------
    # Resample operations
    # ------------------------------------------------
    extension(
        '_resample',
        'resample_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
    extension(
        '_weighted_resample',
        'weighted_resample_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
    extension(
        '_voting_resample',
        'voting_resample_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
    extension(
        '_uv_resample',
        'uv_resample_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),

    # ------------------------------------------------
    # Miscellaneous operations
    # ------------------------------------------------
    extension(
        '_distort',
        'distortion_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*eigen_include_dir, *additional_includes]),
    extension(
        '_mesh',
        'triangle_mesh',
        src_cpp_dir,
        include_dir,
        additional_include_dirs=[*eigen_include_dir, *additional_includes],
        cxx_compile_args=['-DCGAL_HEADER_ONLY'])
]

mapped_conv_modules = [

    # ------------------------------------------------
    # Standard CNN operations
    # ------------------------------------------------
    extension(
        '_convolution',
        'convolution_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
    extension(
        '_transposed_convolution',
        'transposed_convolution_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),

    # ------------------------------------------------
    # Mapped CNN operations
    # ------------------------------------------------
    extension(
        '_mapped_convolution',
        'mapped_convolution_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
    extension(
        '_mapped_transposed_convolution',
        'mapped_transposed_convolution_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
    extension(
        '_mapped_max_pooling',
        'mapped_max_pooling_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
    extension(
        '_mapped_avg_pooling',
        'mapped_avg_pooling_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
    extension(
        '_weighted_mapped_convolution',
        'weighted_mapped_convolution_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
    extension(
        '_weighted_mapped_transposed_convolution',
        'weighted_mapped_transposed_convolution_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
    extension(
        '_weighted_mapped_max_pooling',
        'weighted_mapped_max_pooling_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
    extension(
        '_weighted_mapped_avg_pooling',
        'weighted_mapped_avg_pooling_layer',
        src_cpp_dir,
        include_dir,
        src_cuda_dir,
        additional_include_dirs=[*additional_includes]),
]

if module == 'tangent-images':
    ext_modules = basic_modules
if module == 'mapped-conv':
    ext_modules = basic_modules + mapped_conv_modules

setup(
    name='Spherical Distortion',
    version='1.0.0',
    author='Marc Eder',
    author_email='meder@cs.unc.edu',
    description=
    'A PyTorch module for my dissertation on mitigating spherical distortion',
    ext_package='_spherical_distortion_ext',
    ext_modules=ext_modules,
    packages=[
        'spherical_distortion',
        'spherical_distortion.nn',
        'spherical_distortion.transforms',
        'spherical_distortion.functional',
        'spherical_distortion.util',
        'spherical_distortion.metrics',
        'spherical_distortion.loss',
    ],
    package_dir={
        'spherical_distortion': 'layers',
        'spherical_distortion.nn': 'layers/nn',
        'spherical_distortion.transforms': 'layers/transforms',
        'spherical_distortion.functional': 'layers/functional',
        'spherical_distortion.util': 'layers/util',
        'spherical_distortion.metrics': 'layers/metrics',
        'spherical_distortion.loss': 'layers/loss',
    },
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=use_ninja)},
)
