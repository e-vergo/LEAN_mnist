import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Data Iterator

Memory-efficient batch iteration for neural network training.

Provides iteration utilities for processing datasets in mini-batches with support for
shuffling, epoch management, and flexible batch sizing.

## Main definitions

* `DataIterator`: Stateful iterator for MNIST-sized data (784-dimensional vectors)

## Implementation notes

This is a pure implementation without formal verification. The Fisher-Yates shuffle uses
a basic linear congruential generator (LCG) for reproducibility, not cryptographic security.
-/

namespace VerifiedNN.Data.Iterator

open VerifiedNN.Core
open SciLean

/-- Data iterator state for batch-wise iteration over MNIST datasets.

This structure maintains iteration state for efficient mini-batch training loops.
Supports configurable shuffling using Fisher-Yates algorithm with deterministic seeding.

**Fields:**
- `data`: Complete dataset as array of (image, label) pairs where images are 784-dimensional Float vectors
- `currentIdx`: Current position in the dataset (0 ≤ currentIdx ≤ data.size)
- `batchSize`: Number of samples per batch (typically 32, 64, or 128)
- `shuffle`: Whether to shuffle data at the start of each epoch (default: false)
- `seed`: Random seed for reproducible shuffling using LCG (default: 42)

**Implementation notes:**
- Iterator is stateful and pure (no IO effects during iteration)
- Shuffling increments seed each epoch for different permutations
- Batch extraction uses array slicing (no data copying) -/
structure DataIterator where
  data : Array (Vector 784 × Nat)
  currentIdx : Nat
  batchSize : Nat
  shuffle : Bool := false
  seed : UInt32 := 42

/-- Create a new data iterator.

**Parameters:**
- `data`: Dataset to iterate over (array of (image, label) pairs)
- `batchSize`: Number of samples per batch (must be > 0 for meaningful iteration)
- `shuffle`: Whether to shuffle data between epochs (default: false)
- `seed`: Random seed for shuffling (default: 42)

**Returns:** Fresh iterator positioned at the start of the dataset (currentIdx = 0) -/
def DataIterator.new (data : Array (Vector 784 × Nat)) (batchSize : Nat)
    (shuffle : Bool := false) (seed : UInt32 := 42) : DataIterator :=
  { data := data
    currentIdx := 0
    batchSize := batchSize
    shuffle := shuffle
    seed := seed }

/-- Check if iterator has more data available.

**Returns:** `true` if currentIdx < data.size, meaning at least one sample remains -/
def DataIterator.hasNext (iter : DataIterator) : Bool :=
  iter.currentIdx < iter.data.size

/-- Get next batch from iterator.

Returns a batch of data and an updated iterator. If insufficient samples remain
for a full batch, returns a partial batch with remaining samples.

**Returns:**
- `none` if no data remains (currentIdx ≥ data.size)
- `some (batch, newIter)` where batch contains up to `batchSize` samples and newIter
  has updated currentIdx

**Implementation notes:**
Partial batches are returned at end of epoch. Use `nextFullBatch` to skip these. -/
def DataIterator.nextBatch (iter : DataIterator) :
    Option (Array (Vector 784 × Nat) × DataIterator) :=
  if iter.currentIdx >= iter.data.size then
    none
  else
    let remaining := iter.data.size - iter.currentIdx
    let batchLen := min iter.batchSize remaining
    let endIdx := iter.currentIdx + batchLen

    let batch := iter.data.extract iter.currentIdx endIdx
    let newIter := { iter with currentIdx := endIdx }

    some (batch, newIter)

/-- Reset iterator to beginning of dataset (does not shuffle).

**Returns:** Iterator with currentIdx = 0 and original data order preserved

**Note:** Use `resetWithShuffle` to reset with shuffling enabled -/
def DataIterator.reset (iter : DataIterator) : DataIterator :=
  { iter with currentIdx := 0 }

/-- Shuffle an array using Fisher-Yates algorithm with linear congruential generator.

Implements the unbiased Fisher-Yates shuffle algorithm with a deterministic LCG for random number generation.

**Parameters:**
- `arr`: Array to shuffle (must have Inhabited instance for safe indexing)
- `seed`: Initial seed for LCG

**Returns:** Permuted array with same elements in random order

**Algorithm:** Fisher-Yates shuffle (Knuth Algorithm P)
- LCG parameters: a = 1664525, c = 1013904223, m = 2^32 (Numerical Recipes)
- Time complexity: O(n), Space complexity: O(1) extra (modifies copy)

**Implementation notes:**
This provides deterministic, reproducible shuffling for ML training, not cryptographic security. -/
private def shuffleArray {α : Type} [Inhabited α] (arr : Array α) (seed : UInt32) : Array α :=
  let n := arr.size
  if n ≤ 1 then arr
  else Id.run do
    let mut result := arr
    let mut rng := seed

    -- Fisher-Yates shuffle
    for i in [0:n-1] do
      -- LCG: next = (a * current + c) mod m
      rng := 1664525 * rng + 1013904223
      let j := i + (rng % (n - i).toUInt32).toNat

      -- Swap result[i] and result[j]
      let temp := result[i]!
      result := result.set! i result[j]!
      result := result.set! j temp

    pure result

/-- Reset iterator and optionally shuffle data.

Resets currentIdx to 0 and optionally applies Fisher-Yates shuffle with the iterator's seed.

**Parameters:**
- `iter`: Iterator to reset
- `doShuffle`: Override shuffle behavior (none = use iter.shuffle, some true/false = force enable/disable)

**Returns:** Iterator with currentIdx = 0, optionally shuffled data, and incremented seed (if shuffled)

**Implementation notes:**
- Uses the iterator's seed for reproducibility
- Increments seed after shuffling so next epoch has different permutation
- Seed increment ensures different shuffles across epochs with same initial seed -/
def DataIterator.resetWithShuffle (iter : DataIterator) (doShuffle : Option Bool := none) :
    DataIterator :=
  let shouldShuffle := doShuffle.getD iter.shuffle
  if shouldShuffle then
    { iter with
      currentIdx := 0
      data := shuffleArray iter.data iter.seed
      seed := iter.seed + 1 }  -- Increment seed for next epoch
  else
    iter.reset

end VerifiedNN.Data.Iterator
