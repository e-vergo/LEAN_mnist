#!/bin/bash
# Performance benchmarking script for VerifiedNN

set -e

echo "VerifiedNN Performance Benchmarks"
echo "=================================="
echo ""

# Build the project first
echo "Building project..."
lake build

echo ""
echo "Running benchmarks..."
echo ""

# Benchmark simple example
echo "1. Simple Example Benchmark:"
echo "----------------------------"
time lake exe simpleExample || echo "Simple example not yet implemented"

echo ""
echo "2. MNIST Training Benchmark (1 epoch):"
echo "---------------------------------------"
time lake exe mnistTrain --epochs 1 --batch-size 32 --lr 0.01 || echo "MNIST training not yet implemented"

echo ""
echo "Benchmarks complete!"
