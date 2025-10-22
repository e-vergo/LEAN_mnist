# VerifiedNN.Data

Data loading, preprocessing, and iteration utilities for MNIST neural network training.

## Overview

This directory provides the data pipeline for the verified neural network project:
* **Loading**: Parse MNIST IDX binary format
* **Preprocessing**: Normalize and transform pixel data
* **Iteration**: Efficient batch-wise dataset traversal with shuffling

All implementations are **unverified** - they operate on `Float` values without formal correctness proofs. These are practical utilities for the training pipeline.

## Files

### Iterator.lean (286 lines)

Memory-efficient batch iteration for training.

**Main definitions:**
* `DataIterator`: Stateful iterator for MNIST datasets (784-dimensional vectors with labels)
  - Supports shuffling with Fisher-Yates algorithm and linear congruential generator
  - Configurable batch size, seed, and shuffle behavior
  - Methods: `new`, `nextBatch`, `nextFullBatch`, `reset`, `resetWithShuffle`, `hasNext`
  - Utilities: `progress`, `remainingBatches`, `collectBatches`
* `GenericIterator α`: Polymorphic iterator for arbitrary data types
  - Simpler alternative without shuffling support
  - Methods: `new`, `nextBatch`, `reset`, `hasNext`

**Status:** Fully implemented
**Use cases:** Training loops, epoch management, mini-batch SGD

### MNIST.lean (267 lines)

MNIST IDX binary format parser.

**Main definitions:**
* `loadMNISTImages`: Parse IDX image file → `Array (Vector 784)`
* `loadMNISTLabels`: Parse IDX label file → `Array Nat`
* `loadMNIST`: Combine images and labels → `Array (Vector 784 × Nat)`
* `loadMNISTTrain`: Load standard training set (60,000 samples)
* `loadMNISTTest`: Load standard test set (10,000 samples)

**IDX format details:**
* Images: Magic number 2051, dimensions 28×28, pixel data 0-255 (1 byte each)
* Labels: Magic number 2049, label data 0-9 (1 byte each)
* All multi-byte integers are big-endian

**Status:** Fully implemented
**Error handling:** Returns empty arrays on failure, logs errors to stderr

### Preprocessing.lean (304 lines)

Normalization and data transformation utilities.

**Main definitions:**
* `normalizePixels`: Scale [0, 255] → [0, 1] (standard MNIST preprocessing)
* `normalizeBatch`: Batch version of pixel normalization
* `standardizePixels`: Z-score normalization (zero mean, unit variance)
* `centerPixels`: Subtract mean (zero-center data)
* `flattenImage`: Convert 28×28 `Array (Array Float)` → `Vector 784`
* `flattenImagePure`: Pure version assuming valid dimensions
* `reshapeToImage`: Convert `Vector 784` → 28×28 `Array (Array Float)`
* `normalizeDataset`: Apply normalization to entire dataset
* `clipPixels`: Clamp values to [min, max] range
* `addGaussianNoise`: **TODO - Placeholder** (requires RNG implementation)

**Status:** Mostly implemented (missing RNG-based augmentation)

## Current Status

**Implementation:** Complete for core functionality
**Verification:** None - these are practical utilities without formal proofs
**Testing:** Manual testing via MNIST training pipeline

## Usage Examples

### Loading MNIST

```lean
import VerifiedNN.Data.MNIST

def main : IO Unit := do
  -- Load training set
  let trainData ← loadMNISTTrain (System.FilePath.mk "data/mnist")
  IO.println s!"Loaded {trainData.size} training samples"

  -- Load test set
  let testData ← loadMNISTTest (System.FilePath.mk "data/mnist")
  IO.println s!"Loaded {testData.size} test samples"
```

### Preprocessing

```lean
import VerifiedNN.Data.Preprocessing

-- Normalize dataset (0-255 → 0-1)
let normalizedData := normalizeDataset trainData

-- Standardize individual image
let image : Vector 784 := trainData[0]!.1
let standardized := standardizePixels image

-- Clip values to valid range
let clipped := clipPixels image 0.0 1.0
```

### Batch Iteration

```lean
import VerifiedNN.Data.Iterator

-- Create iterator with shuffling
let iter := DataIterator.new trainData batchSize shuffle:=true seed:=42

-- Iterate over batches
let mut currentIter := iter
while currentIter.hasNext do
  match currentIter.nextBatch with
  | none => break
  | some (batch, newIter) =>
    -- Process batch
    IO.println s!"Batch size: {batch.size}"
    currentIter := newIter

-- Reset for next epoch (with shuffle)
let nextEpochIter := currentIter.resetWithShuffle
```

### Complete Training Pipeline

```lean
-- Load data
let trainData ← loadMNISTTrain dataDir
let testData ← loadMNISTTest dataDir

-- Preprocess
let trainData := normalizeDataset trainData
let testData := normalizeDataset testData

-- Create iterator
let iter := DataIterator.new trainData batchSize:=32 shuffle:=true

-- Training loop
for epoch in [0:numEpochs] do
  let mut epochIter := iter.resetWithShuffle
  while epochIter.hasNext do
    match epochIter.nextBatch with
    | none => break
    | some (batch, newIter) =>
      -- Forward pass, compute loss, backward pass, update weights
      epochIter := newIter
```

## Dependencies

* `VerifiedNN.Core.DataTypes`: Vector and Batch type definitions
* `SciLean`: Array operations and indexing (`Float^[n]`, `⊞` notation)

## Implementation Notes

### Fisher-Yates Shuffle

`DataIterator` uses the Fisher-Yates algorithm with a linear congruential generator (LCG):
* Parameters: a = 1664525, c = 1013904223, m = 2^32
* Provides deterministic, reproducible shuffling
* Not cryptographically secure (not needed for ML training)
* Seed increments after each epoch for different shuffles

### Memory Efficiency

* Iterators maintain reference to original data array (no copying)
* Batches are extracted via `Array.extract` (slice operation)
* Shuffling creates a new permuted array (necessary for randomization)
* For large datasets, use iteration rather than `collectBatches`

### Type Safety

* `DataIterator` is specialized to `Vector 784 × Nat` (MNIST-sized)
* `GenericIterator α` works with arbitrary types
* No dependent types - batch sizes are not tracked at type level

### Float vs ℝ Gap

All preprocessing operates on `Float` values. The project acknowledges but does not bridge the Float ↔ ℝ verification gap. Pixel normalization and standardization are implemented pragmatically without formal correctness proofs.

## Planned Improvements

1. **RNG-based augmentation**: Implement `addGaussianNoise` with proper random number generation
2. **Data augmentation**: Random rotations, translations, elastic distortions
3. **Validation split**: Utility to split training data into train/validation sets
4. **Caching**: Preprocessed data persistence to avoid recomputation
5. **Parallel loading**: Speed up MNIST loading with parallel file I/O

## Testing

Currently tested indirectly through the full MNIST training pipeline. Future work:
* Unit tests for IDX parser edge cases
* Property tests for iterator invariants (correct batch counts, no data loss)
* Gradient checking for preprocessing differentiability
* Benchmarks for iteration performance

## References

* MNIST Database: http://yann.lecun.com/exdb/mnist/
* IDX File Format: http://yann.lecun.com/exdb/mnist/ (see "File formats" section)
* Fisher-Yates Shuffle: Knuth, TAOCP Vol 2, Algorithm P (Shuffling)

---

**Last Updated:** 2025-10-21
**Status:** ✅ Cleanup complete - mathlib quality standards maintained

**Final Cleanup Summary:**
- **Documentation:** All module-level and function-level docstrings at mathlib submission quality
  - Module docstrings enhanced with context, workflow descriptions, and references
  - All 33 public definitions have comprehensive parameter/return/implementation documentation
  - TODO for `addGaussianNoise` fully documented (26 lines) with implementation strategy
- **Code Quality:**
  - Zero compilation errors ✅
  - Zero Lean warnings ✅ (only expected OpenBLAS linker warnings)
  - Zero sorries ✅
  - Zero axioms ✅
  - No commented-out code ✅
  - All imports minimal and necessary ✅
- **Build Status:** All files compile successfully
  - Iterator.lean (286 lines, 16 docstrings)
  - MNIST.lean (267 lines, 7 docstrings)
  - Preprocessing.lean (304 lines, 10 docstrings)
- **Cross-module Verification:** Import structure validated, no external breakage
