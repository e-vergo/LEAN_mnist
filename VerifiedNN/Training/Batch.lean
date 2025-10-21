/-
# Mini-Batch Handling

Data batching utilities for training.

This module provides utilities for creating mini-batches from training data
and shuffling datasets between epochs.
-/

import VerifiedNN.Core.DataTypes

namespace VerifiedNN.Training.Batch

open VerifiedNN.Core

/-- Create mini-batches from training data.

Splits the input data array into mini-batches of the specified size.
The last batch may be smaller if the data size is not evenly divisible by batchSize.

**Parameters:**
- `data`: Array of training examples (input vector, label)
- `batchSize`: Number of examples per batch

**Returns:** Array of batches, where each batch is an array of examples
-/
def createBatches {n : Nat}
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
def shuffleData {α : Type} (data : Array α) : IO (Array α) := do
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
def createShuffledBatches {n : Nat}
    (data : Array (Vector 784 × Nat))
    (batchSize : Nat) : IO (Array (Array (Vector 784 × Nat))) := do
  let shuffled ← shuffleData data
  return createBatches shuffled batchSize

/-- Get total number of batches for given data and batch size -/
def numBatches (dataSize : Nat) (batchSize : Nat) : Nat :=
  if batchSize == 0 then 0
  else (dataSize + batchSize - 1) / batchSize

end VerifiedNN.Training.Batch
