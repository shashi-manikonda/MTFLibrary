from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import pybind11

cpp_args = ['-std=c++17', '-fopenmp']

extensions = [
    Extension(
        "mtflib.backends.cython.mtf_cython",
        ["src/mtflib/backends/cython/mtf_cython.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=cpp_args,
        extra_link_args=cpp_args,
    ),
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
    setup_requires=['cython', 'numpy', 'pybind11'],
    ext_modules=cythonize(extensions),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
    ],
    zip_safe=False,
)
