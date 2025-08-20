import sys
from setuptools import setup, find_packages, Extension
import numpy
import pybind11

# Set compiler arguments based on the operating system
if sys.platform == 'win32':
    # MSVC compiler arguments
    cpp_args = ['/std:c++17', '/openmp']
else:
    # GCC/Clang compiler arguments
    cpp_args = ['-std=c++17', '-fopenmp']

extensions = [
    Extension('mtflib.backends.cpp.mtf_cpp',
              ['src/mtflib/backends/cpp/mtf_ops.cpp', 'src/mtflib/backends/cpp/pybind_wrapper.cpp'],
              include_dirs=[pybind11.get_include(), numpy.get_include(), 'src/mtflib/backends/cpp'],
              language='c++',
              extra_compile_args=cpp_args,
              extra_link_args=cpp_args),
]

setup(
    name='mtflib',
    version='2.0.0',
    description='Multivariate Taylor Expansion Library with C++ backend',
    author='Shashikant Manikonda',
    author_email='manikonda@outlook.com',
    license='MIT',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=['numpy', 'pandas'],
    setup_requires=['numpy', 'pybind11'],
    extras_require={
        'test': [
            'pytest',
            'matplotlib',
            'nbconvert',
            'ipykernel',
        ]
    },
    ext_modules=extensions,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
    ],
    zip_safe=False,
)
