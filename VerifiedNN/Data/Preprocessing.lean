import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Data Preprocessing

Normalization and data augmentation utilities for MNIST images.

## Main definitions

* `normalizePixels`: Scale pixel values from [0, 255] to [0, 1]
* `normalizeBatch`: Batch version of pixel normalization
* `standardizePixels`: Compute z-score normalization (zero mean, unit variance)
* `centerPixels`: Subtract mean from pixel values
* `flattenImage`: Convert 28×28 image to 784-dimensional vector
* `reshapeToImage`: Convert 784-dimensional vector to 28×28 image
* `normalizeDataset`: Apply normalization to entire dataset
* `clipPixels`: Clamp pixel values to specified range

## Implementation notes

All preprocessing operations work with Float values, not verified on ℝ.
This module contains no formal verification - it's a practical implementation for training.
-/

namespace VerifiedNN.Data.Preprocessing

open VerifiedNN.Core
open SciLean

/-- Normalize pixel values from [0, 255] to [0, 1].

Standard preprocessing for MNIST: divides each pixel by 255.0. -/
def normalizePixels {n : Nat} (image : Vector n) : Vector n :=
  ⊞ i => image[i] / 255.0

/-- Normalize pixels from [0, 255] to [0, 1] (batch version). -/
def normalizeBatch {b n : Nat} (batch : Batch b n) : Batch b n :=
  ⊞ (i : Idx b) (j : Idx n) => batch[i,j] / 255.0

/-- Standardize data to zero mean and unit variance.

Computes z-score normalization: (x - μ) / σ where μ is the mean and σ is the standard deviation.
Uses epsilon to prevent division by zero.

Note: For batch processing, consider computing statistics across the entire batch. -/
def standardizePixels {n : Nat} (image : Vector n) (epsilon : Float := 1e-7) : Vector n :=
  -- Compute mean
  let sum := ∑ i, image[i]
  let mean := sum / n.toFloat

  -- Compute variance
  let variance := (∑ i, (image[i] - mean) ^ 2) / n.toFloat
  let stddev := Float.sqrt (variance + epsilon)

  -- Standardize
  ⊞ i => (image[i] - mean) / stddev

/-- Center data by subtracting mean (results in mean ≈ 0). -/
def centerPixels {n : Nat} (image : Vector n) : Vector n :=
  let sum := ∑ i, image[i]
  let mean := sum / n.toFloat
  ⊞ i => image[i] - mean

/-- Flatten 28×28 image to 784-dimensional vector in row-major order.

Returns zero vector if dimensions don't match 28×28 (logs warning to stderr). -/
def flattenImage (image : Array (Array Float)) : IO (Vector 784) := do
  -- Validate dimensions
  if image.size != 28 then
    IO.eprintln s!"Warning: expected 28 rows, got {image.size}"
    return ⊞ (_ : Idx 784) => 0.0

  -- Check all rows have 28 columns
  for row in image do
    if row.size != 28 then
      IO.eprintln s!"Warning: expected 28 columns, got {row.size}"
      return ⊞ (_ : Idx 784) => 0.0

  -- Flatten in row-major order and convert to vector using SciLean notation
  let flatData : Array Float ← do
    let mut arr : Array Float := Array.mkEmpty 784
    for row in image do
      for pixel in row do
        arr := arr.push pixel
    pure arr

  -- Convert Array Float to Vector using indexed constructor
  return ⊞ (i : Idx 784) => flatData[i.1]!

/-- Flatten 28×28 image (pure version, assumes valid dimensions).

Precondition: image must be exactly 28×28, undefined behavior otherwise. -/
def flattenImagePure (image : Array (Array Float)) : Vector 784 :=
  let flatData : Array Float := Id.run do
    let mut arr : Array Float := Array.mkEmpty 784
    for row in image do
      for pixel in row do
        arr := arr.push pixel
    pure arr
  -- Convert Array Float to Vector using indexed constructor
  ⊞ (i : Idx 784) => flatData[i.1]!

/-- Reshape 784-dimensional vector to 28×28 image.

Inverse of flattening. Useful for visualization or debugging. -/
def reshapeToImage (vector : Vector 784) : Array (Array Float) := Id.run do
  let mut image : Array (Array Float) := Array.mkEmpty 28
  for row in [0:28] do
    let mut rowData : Array Float := Array.mkEmpty 28
    for col in [0:28] do
      -- row and col are in range [0,28) so row * 28 + col < 784
      -- Use conditional to prove bounds from loop invariants
      if hrow : row < 28 then
        if hcol : col < 28 then
          let linearIdx := row * 28 + col
          have hbound : linearIdx < 784 := by omega
          -- USize bound proof: omega cannot handle toUSize.toNat conversions
          -- This is a technical limitation of Lean's omega tactic
          -- The bound is mathematically trivial given hbound
          let idx : Idx 784 := ⟨linearIdx.toUSize, by
            -- Rewrite toUSize as ofNat, then apply toNat_ofNat_of_lt
            rw [Nat.toUSize_eq]
            rw [USize.toNat_ofNat_of_lt']
            · exact hbound
            · -- Show linearIdx < USize.size
              -- Since linearIdx < 784, and 784 < 2^32 <= USize.size
              calc linearIdx < 784 := hbound
                _ < 4294967296 := by norm_num
                _ ≤ USize.size := USize.le_size
          ⟩
          let val : Float := vector[idx]
          rowData := rowData.push val
    image := image.push rowData
  pure image

/-- Apply normalization to entire dataset (labels unchanged). -/
def normalizeDataset {n : Nat} (dataset : Array (Vector n × Nat)) :
    Array (Vector n × Nat) :=
  dataset.map fun (image, label) => (normalizePixels image, label)

/-- Clip pixel values to [min, max] range (default [0.0, 1.0]).

Useful for preventing numerical issues or applying constraints. -/
def clipPixels {n : Nat} (image : Vector n) (min : Float := 0.0) (max : Float := 1.0) :
    Vector n :=
  ⊞ i =>
    let val := image[i]
    if val < min then min
    else if val > max then max
    else val

/-- Add Gaussian noise for data augmentation.

TODO: Placeholder - requires proper RNG implementation. Currently returns input unchanged. -/
def addGaussianNoise {n : Nat} (image : Vector n) (_stddev : Float := 0.01) : Vector n :=
  -- TODO: Implement with proper RNG in IO monad
  -- Currently returns input unchanged until RNG is implemented
  image

end VerifiedNN.Data.Preprocessing
