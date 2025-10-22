import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Data Iterator

Memory-efficient batch iteration for neural network training.

Provides iteration utilities for processing datasets in mini-batches with support for
shuffling, epoch management, and flexible batch sizing.

## Main definitions

* `DataIterator`: Stateful iterator for MNIST-sized data (784-dimensional vectors)
* `GenericIterator α`: Polymorphic iterator for arbitrary data types

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

/-- Get next full batch, skipping incomplete batches.

Returns only complete batches of size `batchSize`. Useful when training requires fixed batch sizes.

**Returns:**
- `none` if insufficient data for a full batch (currentIdx + batchSize > data.size)
- `some (batch, newIter)` where batch has exactly `batchSize` samples

**Use case:** Training algorithms that cannot handle variable batch sizes -/
def DataIterator.nextFullBatch (iter : DataIterator) :
    Option (Array (Vector 784 × Nat) × DataIterator) :=
  if iter.currentIdx + iter.batchSize > iter.data.size then
    none
  else
    let endIdx := iter.currentIdx + iter.batchSize
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

/-- Get current progress through the dataset as a fraction from 0.0 to 1.0.

**Returns:** Progress as Float in [0.0, 1.0] where 0.0 = start, 1.0 = end (or empty dataset) -/
def DataIterator.progress (iter : DataIterator) : Float :=
  if iter.data.size == 0 then 1.0
  else iter.currentIdx.toFloat / iter.data.size.toFloat

/-- Get number of complete batches remaining in the dataset.

**Returns:** Number of full batches that can be extracted from current position (partial batches not counted) -/
def DataIterator.remainingBatches (iter : DataIterator) : Nat :=
  let remaining := iter.data.size - iter.currentIdx
  remaining / iter.batchSize

/-- Collect all batches from iterator into an array.

Consumes the iterator and collects all remaining batches into an array of arrays.
Useful for small datasets or when you need to precompute all batches.

**Returns:** Array of batches, where each batch is an array of (image, label) pairs.
May include partial batch at end if dataset size not divisible by batchSize.

**Warning:** Loads all data into memory. Use iteration for large datasets. -/
def DataIterator.collectBatches (iter : DataIterator) :
    Array (Array (Vector 784 × Nat)) := Id.run do
  let mut batches : Array (Array (Vector 784 × Nat)) := #[]
  let mut current := iter

  while current.hasNext do
    match current.nextBatch with
    | none => break
    | some (batch, newIter) =>
      batches := batches.push batch
      current := newIter

  pure batches

/-- Generic data iterator for arbitrary data types.

More flexible version that works with any data type, not just MNIST pairs.
Provides basic iteration without shuffling support.

**Type parameters:**
- `α`: Type of data elements (can be any type)

**Fields:**
- `data`: Array of elements to iterate over
- `currentIdx`: Current position in the dataset
- `batchSize`: Number of elements per batch

**Use cases:** Non-MNIST datasets, custom data types, when shuffling not needed -/
structure GenericIterator (α : Type) where
  data : Array α
  currentIdx : Nat
  batchSize : Nat

/-- Create a generic iterator.

**Parameters:**
- `data`: Array of elements to iterate over
- `batchSize`: Number of elements per batch

**Returns:** Fresh iterator positioned at the start (currentIdx = 0) -/
def GenericIterator.new {α : Type} (data : Array α) (batchSize : Nat) :
    GenericIterator α :=
  { data := data, currentIdx := 0, batchSize := batchSize }

/-- Get next batch from generic iterator (may return partial batch at end).

**Returns:**
- `none` if no data remains
- `some (batch, newIter)` where batch contains up to `batchSize` elements -/
def GenericIterator.nextBatch {α : Type} (iter : GenericIterator α) :
    Option (Array α × GenericIterator α) :=
  if iter.currentIdx >= iter.data.size then
    none
  else
    let remaining := iter.data.size - iter.currentIdx
    let batchLen := min iter.batchSize remaining
    let endIdx := iter.currentIdx + batchLen

    let batch := iter.data.extract iter.currentIdx endIdx
    let newIter := { iter with currentIdx := endIdx }

    some (batch, newIter)

/-- Reset generic iterator to beginning.

**Returns:** Iterator with currentIdx = 0 -/
def GenericIterator.reset {α : Type} (iter : GenericIterator α) : GenericIterator α :=
  { iter with currentIdx := 0 }

/-- Check if generic iterator has more data available.

**Returns:** `true` if currentIdx < data.size -/
def GenericIterator.hasNext {α : Type} (iter : GenericIterator α) : Bool :=
  iter.currentIdx < iter.data.size

end VerifiedNN.Data.Iterator
