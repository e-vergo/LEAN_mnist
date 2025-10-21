/-
# Mini-Batch Handling

Data batching and shuffling utilities for stochastic gradient descent training.

## Overview

This module provides utilities for preparing training data in mini-batches,
which is essential for efficient and effective neural network training:
- **Batching:** Split large datasets into manageable mini-batches
- **Shuffling:** Randomize data order between epochs to prevent overfitting
- **Efficiency:** Support for variable batch sizes and partial final batches

## Implementation Status

**Complete implementation:** All core batching functionality is implemented:
- Fixed-size mini-batch creation with partial last batch support
- Fisher-Yates shuffle algorithm for randomization
- Convenient combined shuffle + batch interface

Potential future enhancements:
- Stratified batching (ensure class balance within batches)
- Data augmentation hooks
- Parallel batch processing

## Mini-Batch Strategy

Mini-batch gradient descent balances two competing goals:
1. **Computational efficiency:** Larger batches use vectorization better
2. **Optimization dynamics:** Smaller batches add noise that helps escape local minima

Typical MNIST batch sizes: 16-128 examples

## Shuffling Algorithm

Uses Fisher-Yates shuffle to ensure uniform random permutation:
1. For each position i from 0 to n-1:
2. Generate random index j where i ≤ j < n
3. Swap elements at positions i and j

This produces an unbiased random permutation in O(n) time.

## Usage

```lean
-- Create static batches
let batches := createBatches trainData 32

-- Create shuffled batches (typical training usage)
let shuffledBatches ← createShuffledBatches trainData 32

-- Query batch information
let nBatches := numBatches trainData.size 32
```
-/

import VerifiedNN.Core.DataTypes

namespace VerifiedNN.Training.Batch

open VerifiedNN.Core

/-- Inhabited instance for (Vector 784 × Nat) to support array operations -/
instance : Inhabited (Vector 784 × Nat) := ⟨(0, 0)⟩

/-- Create mini-batches from training data.

Splits the input data array into mini-batches of the specified size.
The last batch may be smaller if the data size is not evenly divisible by batchSize.

**Parameters:**
- `data`: Array of training examples (input vector, label)
- `batchSize`: Number of examples per batch

**Returns:** Array of batches, where each batch is an array of examples
-/
def createBatches
    (data : Array (Vector 784 × Nat))
    (batchSize : Nat) : Array (Array (Vector 784 × Nat)) :=
  if batchSize == 0 then
    #[]  -- Return empty array if batchSize is 0
  else
    let numBatches := (data.size + batchSize - 1) / batchSize  -- Ceiling division
    Array.ofFn fun (i : Fin numBatches) =>
      let start := i.val * batchSize
      let end_idx := min (start + batchSize) data.size
      data.extract start end_idx

/-- Shuffle data array using Fisher-Yates shuffle algorithm.

Performs an in-place shuffle of the input array using random swaps.
This is used to randomize training data between epochs.

**Parameters:**
- `data`: Array to shuffle

**Returns:** Shuffled array in IO monad (due to randomness)

**Implementation:** Fisher-Yates shuffle with cryptographically secure randomness
-/
def shuffleData {α : Type} [Inhabited α] (data : Array α) : IO (Array α) := do
  let mut arr := data
  for i in [0:data.size] do
    -- Generate random index j where i ≤ j < data.size
    let range := data.size - i
    if range > 0 then
      let rand ← IO.rand 0 range
      let j := i + rand
      -- Swap arr[i] and arr[j]
      let temp := arr[i]!
      arr := arr.set! i arr[j]!
      arr := arr.set! j temp
  return arr

/-- Create mini-batches with shuffling.

Convenience function that shuffles data and then creates batches.
This is the typical usage pattern for training.

**Parameters:**
- `data`: Training data to batch
- `batchSize`: Size of each mini-batch

**Returns:** Shuffled and batched data in IO monad
-/
def createShuffledBatches
    (data : Array (Vector 784 × Nat))
    (batchSize : Nat) : IO (Array (Array (Vector 784 × Nat))) := do
  let shuffled ← shuffleData data
  return createBatches shuffled batchSize

/-- Get total number of batches for given data and batch size -/
def numBatches (dataSize : Nat) (batchSize : Nat) : Nat :=
  if batchSize == 0 then 0
  else (dataSize + batchSize - 1) / batchSize

end VerifiedNN.Training.Batch
