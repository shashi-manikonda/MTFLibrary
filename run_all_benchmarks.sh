#!/bin/bash
echo "--- Small Size (20 terms, 100 points) ---"
echo "Python:"
python benchmarks/benchmark.py --implementation python --num-terms 20 --num-points 100
echo "C:"
python benchmarks/benchmark.py --implementation c --num-terms 20 --num-points 100
echo "C++:"
python benchmarks/benchmark.py --implementation cpp --num-terms 20 --num-points 100

echo ""
echo "--- Mid Size (100 terms, 500 points) ---"
echo "Python:"
python benchmarks/benchmark.py --implementation python --num-terms 100 --num-points 500
echo "C:"
python benchmarks/benchmark.py --implementation c --num-terms 100 --num-points 500
echo "C++:"
python benchmarks/benchmark.py --implementation cpp --num-terms 100 --num-points 500

echo ""
echo "--- Large Size (500 terms, 1000 points) ---"
echo "Python:"
python benchmarks/benchmark.py --implementation python --num-terms 500 --num-points 1000
echo "C:"
python benchmarks/benchmark.py --implementation c --num-terms 500 --num-points 1000
echo "C++:"
python benchmarks/benchmark.py --implementation cpp --num-terms 500 --num-points 1000
