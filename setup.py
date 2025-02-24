# setup.py
from setuptools import setup, find_packages

setup(
    name='MTFLibrary',
    version='0.1.0',
    description='Multivariate Taylor Expansion Library',
    author='Shashikant Manikonda',
    author_email='manikonda@outlook.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
    ],
)