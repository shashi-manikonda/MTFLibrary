from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import pybind11

cpp_args = ['-std=c++17', '-fopenmp']

extensions = [
    Extension(
        "MTFLibrary.mtf_cython",
        ["MTFLibrary/mtf_cython.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=cpp_args,
        extra_link_args=cpp_args,
    ),
    Extension('MTFLibrary.mtf_cpp',
              ['cpp_src/mtf_ops.cpp', 'cpp_src/pybind_wrapper.cpp'],
              include_dirs=[pybind11.get_include(), numpy.get_include()],
              language='c++',
              extra_compile_args=cpp_args,
              extra_link_args=cpp_args),
]

setup(
    name='MTFLibrary',
    version='1.2.0',
    description='Multivariate Taylor Expansion Library with C++ backend',
    author='Shashikant Manikonda',
    author_email='manikonda@outlook.com',
    license='MIT',
    packages=find_packages(),
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
