#!/bin/bash
# Download MNIST dataset from Yann LeCun's website

set -e

# Create data directory if it doesn't exist
mkdir -p data
cd data

echo "Downloading MNIST dataset..."

# MNIST URLs (using mirror since original is no longer available)
BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

FILES=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

# Download each file
for file in "${FILES[@]}"; do
  if [ ! -f "${file%.gz}" ]; then
    echo "Downloading $file..."
    curl -O "$BASE_URL/$file"
    echo "Extracting $file..."
    gunzip -f "$file"
  else
    echo "$file already exists, skipping..."
  fi
done

echo "MNIST dataset downloaded successfully!"
echo ""
echo "Files in data directory:"
ls -lh

cd ..
