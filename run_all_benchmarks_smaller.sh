#!/bin/bash
echo "--- Small Size (20 terms, 100 points) ---"
echo "Python:"
python benchmarks/benchmark.py --implementation python --num-terms 20 --num-points 100
echo "C:"
python benchmarks/benchmark.py --implementation c --num-terms 20 --num-points 100
echo "C++:"
python benchmarks/benchmark.py --implementation cpp --num-terms 20 --num-points 100

echo ""
echo "--- Mid Size (50 terms, 200 points) ---"
echo "Python:"
python benchmarks/benchmark.py --implementation python --num-terms 50 --num-points 200
echo "C:"
python benchmarks/benchmark.py --implementation c --num-terms 50 --num-points 200
echo "C++:"
python benchmarks/benchmark.py --implementation cpp --num-terms 50 --num-points 200

echo ""
echo "--- Large Size (100 terms, 300 points) ---"
echo "Python:"
python benchmarks/benchmark.py --implementation python --num-terms 100 --num-points 300
echo "C:"
python benchmarks/benchmark.py --implementation c --num-terms 100 --num-points 300
echo "C++:"
python benchmarks/benchmark.py --implementation cpp --num-terms 100 --num-points 300
