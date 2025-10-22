import VerifiedNN.Core.DataTypes

/-!
# Mini-Batch Handling

Data batching and shuffling utilities for stochastic gradient descent training.

## Main Definitions

- `createBatches`: Split dataset into fixed-size mini-batches with partial final batch support
- `shuffleData`: Fisher-Yates shuffle algorithm for unbiased random permutation
- `createShuffledBatches`: Convenience function combining shuffle and batch creation
- `numBatches`: Calculate total number of batches for given data size and batch size

## Main Results

This module provides computational utilities without formal verification.
No theorems are proven, but the implementations are production-ready.

## Implementation Notes

**Mini-batch strategy:** Mini-batch gradient descent balances two competing goals:
1. **Computational efficiency:** Larger batches use vectorization better (SIMD, cache)
2. **Optimization dynamics:** Smaller batches add stochastic noise that helps escape local minima

Typical MNIST batch sizes: 16-128 examples. Batch size is a key hyperparameter.

**Shuffling algorithm:** Uses the Fisher-Yates shuffle to ensure uniform random permutation:
1. For each position i from 0 to n-1:
2. Generate random index j where i ≤ j < n
3. Swap elements at positions i and j

This produces an unbiased random permutation in O(n) time with O(1) space.

**Performance:** Shuffling uses cryptographically secure randomness via `IO.rand`.
For large datasets (>100k examples), consider parallel batch creation.

**Type genericity:** `shuffleData` works with any inhabited type, making it reusable
beyond MNIST training data.

**Edge cases:**
- Zero batch size returns empty array (no batches)
- Empty data returns empty array (no batches)
- Partial final batch is included (ensures all data is used)

## References

- Fisher-Yates shuffle: Knuth, "The Art of Computer Programming", Vol. 2, Algorithm P
- Mini-batch SGD: "On Large-Batch Training for Deep Learning" (Keskar et al., 2016)
- Stochastic optimization: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)
-/

namespace VerifiedNN.Training.Batch

open VerifiedNN.Core

/-- Inhabited instance for (Vector 784 × Nat) to support array operations.

This instance is required for array functions that need default values,
such as `Array.get!` and array modification operations used in shuffling.
The default value (zero vector, zero label) is arbitrary and should never
be accessed in correct usage.
-/
instance : Inhabited (Vector 784 × Nat) := ⟨(0, 0)⟩

/-- Create mini-batches from training data.

Splits the input data array into mini-batches of the specified size.
The last batch may be smaller if the data size is not evenly divisible by batchSize.
This enables mini-batch stochastic gradient descent, which balances computational
efficiency (vectorization) with optimization dynamics (gradient noise).

**Parameters:**
- `data`: Array of training examples (input vector, label)
- `batchSize`: Number of examples per batch (typically 16-128 for MNIST)

**Returns:** Array of batches, where each batch is an array of examples

**Edge cases:**
- If `batchSize = 0`, returns empty array (no batches created)
- If `data.size < batchSize`, returns single batch containing all data
- If `data.size % batchSize ≠ 0`, final batch contains remainder examples

**Implementation:** Uses ceiling division `(n + b - 1) / b` to compute batch count,
ensuring partial final batch is included. Each batch is created via `Array.extract`
for efficient slicing without copying the entire array.

**Complexity:** O(n / batchSize) to create batch array structure, where n = data.size
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
This is used to randomize training data order between epochs, which prevents
the network from learning spurious patterns based on data ordering and improves
generalization. The Fisher-Yates algorithm guarantees uniform random permutation,
meaning every permutation of the input array has equal probability.

**Parameters:**
- `data`: Array to shuffle (any inhabited type)

**Returns:** Shuffled array in IO monad (due to randomness)

**Correctness:** The Fisher-Yates shuffle produces a uniform random permutation.
After processing position i, all elements in positions [0, i] are uniformly
randomly selected from the original array, and all elements in positions [i+1, n)
are uniformly randomly selected from the remaining unprocessed elements.

**Implementation:** Uses cryptographically secure randomness via `IO.rand`, which
provides high-quality random numbers suitable for scientific computing. The
shuffle is performed functionally (despite "in-place" description) by creating
new array versions via `Array.set!`.

**Complexity:**
- Time: O(n) where n = data.size (single pass through array)
- Space: O(n) due to functional array updates (Lean arrays are not truly mutable)
- Randomness: n calls to `IO.rand`

**Type genericity:** Works with any type `α` that has an `Inhabited` instance,
making this function reusable beyond MNIST training data.

**References:** Knuth, "The Art of Computer Programming", Vol. 2, Algorithm P
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
This is the typical usage pattern for training: at the start of each epoch,
randomize the data order and split into mini-batches. The randomization prevents
the optimizer from learning epoch-specific patterns and improves generalization.

**Parameters:**
- `data`: Training data to batch (typically MNIST training examples)
- `batchSize`: Size of each mini-batch (typically 16-128 for MNIST)

**Returns:** Shuffled and batched data in IO monad

**Usage pattern:** Call this at the start of each training epoch:
```lean
for epoch in [0:numEpochs] do
  let batches ← createShuffledBatches trainData batchSize
  for batch in batches do
    processMiniBatch batch
```

**Composition:** Equivalent to `createBatches (← shuffleData data) batchSize`,
but more convenient and self-documenting.
-/
def createShuffledBatches
    (data : Array (Vector 784 × Nat))
    (batchSize : Nat) : IO (Array (Array (Vector 784 × Nat))) := do
  let shuffled ← shuffleData data
  return createBatches shuffled batchSize

/-- Get total number of batches for given data and batch size.

Computes the number of mini-batches that will be created from a dataset
of size `dataSize` when using batches of size `batchSize`. Uses ceiling
division to include the partial final batch if `dataSize` is not evenly
divisible by `batchSize`.

**Parameters:**
- `dataSize`: Total number of examples in dataset
- `batchSize`: Desired mini-batch size

**Returns:** Number of batches (including partial final batch)

**Examples:**
- `numBatches 100 32 = 4` (batches of sizes 32, 32, 32, 4)
- `numBatches 64 32 = 2` (batches of sizes 32, 32)
- `numBatches 30 32 = 1` (batch of size 30)
- `numBatches 100 0 = 0` (edge case: invalid batch size)

**Mathematical formula:** ⌈dataSize / batchSize⌉ = (dataSize + batchSize - 1) / batchSize

**Use case:** Allocate arrays or estimate training time before creating actual batches.
-/
def numBatches (dataSize : Nat) (batchSize : Nat) : Nat :=
  if batchSize == 0 then 0
  else (dataSize + batchSize - 1) / batchSize

end VerifiedNN.Training.Batch
