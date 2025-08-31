#!/bin/bash
echo "--- Tiny Size (10 terms, 50 points) ---"
echo "Python:"
python benchmarks/benchmark.py --implementation python --num-terms 10 --num-points 50
echo "C:"
python benchmarks/benchmark.py --implementation c --num-terms 10 --num-points 50
echo "C++:"
python benchmarks/benchmark.py --implementation cpp --num-terms 10 --num-points 50
