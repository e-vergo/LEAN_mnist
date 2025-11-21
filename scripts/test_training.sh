#!/bin/bash

set -e  # Exit on error

echo "========================================"
echo "MNIST Training Validation Script"
echo "========================================"
echo ""

# Check if MNIST data exists
echo "Checking for MNIST data..."
if [ ! -f "data/train-images-idx3-ubyte" ]; then
    echo "Error: MNIST data not found!"
    echo "Please run: ./scripts/download_mnist.sh"
    exit 1
fi
echo "✓ MNIST data found"
echo ""

# Build everything
echo "Building project..."
lake build VerifiedNN.Examples.MiniTraining
lake build VerifiedNN.Examples.MNISTTrain
echo "✓ Build complete"
echo ""

# Run mini training test (quick validation)
echo "========================================"
echo "Running Mini Training Test (~30s)"
echo "========================================"
echo ""
lake exe miniTraining

echo ""
echo "========================================"
echo "Mini training test passed!"
echo "========================================"
echo ""

# Ask if user wants to run full training
echo "Mini training successful. Ready for full MNIST training?"
echo "Full training: 60,000 samples, 10 epochs, ~5-10 minutes"
echo ""
read -p "Run full training now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "========================================"
    echo "Running Full MNIST Training"
    echo "========================================"
    echo ""
    echo "Training with defaults:"
    echo "  - Epochs: 10"
    echo "  - Batch size: 32"
    echo "  - Learning rate: 0.01"
    echo ""
    echo "Output will be saved to training_log.txt"
    echo ""

    # Run training and save output
    lake exe mnistTrain --epochs 10 --batch-size 32 --lr 0.01 | tee training_log.txt

    echo ""
    echo "========================================"
    echo "Training complete! Results saved to training_log.txt"
    echo "========================================"
else
    echo ""
    echo "Skipping full training. You can run it later with:"
    echo "  lake exe mnistTrain --epochs 10"
    echo ""
fi

echo ""
echo "✓ All tests complete!"
