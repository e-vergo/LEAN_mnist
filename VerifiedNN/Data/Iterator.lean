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

Fields:
* `data`: Complete dataset as array of (image, label) pairs
* `currentIdx`: Current position in the dataset
* `batchSize`: Number of samples per batch
* `shuffle`: Whether to shuffle data at the start of each epoch
* `seed`: Random seed for reproducible shuffling (if shuffle enabled) -/
structure DataIterator where
  data : Array (Vector 784 × Nat)
  currentIdx : Nat
  batchSize : Nat
  shuffle : Bool := false
  seed : UInt32 := 42

/-- Create a new data iterator.

Parameters:
* `data`: Dataset to iterate over
* `batchSize`: Number of samples per batch
* `shuffle`: Whether to shuffle data between epochs (default: false)
* `seed`: Random seed for shuffling (default: 42) -/
def DataIterator.new (data : Array (Vector 784 × Nat)) (batchSize : Nat)
    (shuffle : Bool := false) (seed : UInt32 := 42) : DataIterator :=
  { data := data
    currentIdx := 0
    batchSize := batchSize
    shuffle := shuffle
    seed := seed }

/-- Check if iterator has more data available. -/
def DataIterator.hasNext (iter : DataIterator) : Bool :=
  iter.currentIdx < iter.data.size

/-- Get next batch from iterator.

Returns a batch of data and an updated iterator. If insufficient samples remain
for a full batch, returns a partial batch with remaining samples.

Returns `none` if no data remains, otherwise `some (batch, newIter)` where batch
contains up to `batchSize` samples.

Note: Partial batches are returned at end of epoch. Use `nextFullBatch` to skip these. -/
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

Returns `none` if insufficient data for a full batch, otherwise `some (batch, newIter)`
where batch has exactly `batchSize` samples. -/
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

Use `resetWithShuffle` to reset with shuffling enabled. -/
def DataIterator.reset (iter : DataIterator) : DataIterator :=
  { iter with currentIdx := 0 }

/-- Shuffle an array using Fisher-Yates algorithm with linear congruential generator.

Uses LCG parameters: a = 1664525, c = 1013904223, m = 2^32.
This is a basic deterministic shuffle for reproducibility, not cryptographic security. -/
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

Uses the iterator's seed for reproducibility and increments it for the next epoch. -/
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

/-- Get current progress through the dataset as a fraction from 0.0 to 1.0. -/
def DataIterator.progress (iter : DataIterator) : Float :=
  if iter.data.size == 0 then 1.0
  else iter.currentIdx.toFloat / iter.data.size.toFloat

/-- Get number of complete batches remaining in the dataset. -/
def DataIterator.remainingBatches (iter : DataIterator) : Nat :=
  let remaining := iter.data.size - iter.currentIdx
  remaining / iter.batchSize

/-- Collect all batches from iterator into an array.

Useful for small datasets. May include partial batch at end.

Warning: Loads all data into memory. Use iteration for large datasets. -/
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

Type parameters:
* `α`: Type of data elements -/
structure GenericIterator (α : Type) where
  data : Array α
  currentIdx : Nat
  batchSize : Nat

/-- Create a generic iterator. -/
def GenericIterator.new {α : Type} (data : Array α) (batchSize : Nat) :
    GenericIterator α :=
  { data := data, currentIdx := 0, batchSize := batchSize }

/-- Get next batch from generic iterator (may return partial batch at end). -/
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

/-- Reset generic iterator to beginning. -/
def GenericIterator.reset {α : Type} (iter : GenericIterator α) : GenericIterator α :=
  { iter with currentIdx := 0 }

/-- Check if generic iterator has more data available. -/
def GenericIterator.hasNext {α : Type} (iter : GenericIterator α) : Bool :=
  iter.currentIdx < iter.data.size

end VerifiedNN.Data.Iterator
