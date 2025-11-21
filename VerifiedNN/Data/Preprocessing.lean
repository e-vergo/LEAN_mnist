import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Data Preprocessing

Normalization and data augmentation utilities for MNIST images.

Provides functions for transforming raw MNIST data into formats suitable for neural network
training, including scaling, standardization, and format conversion between 1D vectors and 2D arrays.

## Main definitions

* `normalizePixels`: Scale pixel values from [0, 255] to [0, 1]
* `standardizePixels`: Compute z-score normalization (zero mean, unit variance)
* `centerPixels`: Subtract mean from pixel values
* `flattenImage`: Convert 28×28 image to 784-dimensional vector
* `reshapeToImage`: Convert 784-dimensional vector to 28×28 image
* `normalizeDataset`: Apply normalization to entire dataset
* `clipPixels`: Clamp pixel values to specified range

## Implementation notes

All preprocessing operations work with `Float` values, not verified on ℝ.
This module contains no formal verification - it's a practical implementation for training.

Standard MNIST preprocessing workflow:
1. Load raw images with `loadMNISTImages` (pixels in [0, 255])
2. Apply `normalizePixels` to scale to [0, 1]
3. Optionally apply `standardizePixels` for zero mean/unit variance
4. Use `flattenImage`/`reshapeToImage` for format conversion as needed

**Verification status:** Implementation only, no formal proofs of numerical correctness.
-/

namespace VerifiedNN.Data.Preprocessing

open VerifiedNN.Core
open SciLean

/-- Normalize pixel values from [0, 255] to [0, 1].

Standard preprocessing for MNIST: scales raw pixel values to unit interval by dividing by 255.

**Parameters:**
- `image`: Vector of pixel values (typically in range [0, 255] from raw MNIST data)

**Returns:** Vector with values scaled to [0, 1]

**Mathematical operation:** For each pixel p, computes p / 255.0

**Use case:** Apply after loading MNIST data to prepare for neural network input -/
def normalizePixels {n : Nat} (image : Vector n) : Vector n :=
  ⊞ i => image[i] / 255.0

/-- Standardize data to zero mean and unit variance.

Computes z-score normalization: transforms data to have mean ≈ 0 and standard deviation ≈ 1.

**Parameters:**
- `image`: Vector of pixel values
- `epsilon`: Small constant to prevent division by zero (default: 1e-7)

**Returns:** Standardized vector where each pixel x becomes (x - μ) / σ

**Mathematical operation:**
- μ = mean of pixel values
- σ = sqrt(variance + ε)
- Result: (x - μ) / σ for each pixel x

**Implementation notes:**
- Computes statistics per-image (not across batch)
- For batch normalization, compute statistics across entire batch instead
- Epsilon prevents numerical instability when variance ≈ 0 -/
def standardizePixels {n : Nat} (image : Vector n) (epsilon : Float := 1e-7) : Vector n :=
  -- Compute mean
  let sum := ∑ i, image[i]
  let mean := sum / n.toFloat

  -- Compute variance
  let variance := (∑ i, (image[i] - mean) ^ 2) / n.toFloat
  let stddev := Float.sqrt (variance + epsilon)

  -- Standardize
  ⊞ i => (image[i] - mean) / stddev

/-- Center data by subtracting mean (results in mean ≈ 0).

Applies mean centering: shifts all pixel values so the image has zero mean.

**Parameters:**
- `image`: Vector of pixel values

**Returns:** Centered vector where each pixel x becomes x - μ (μ = mean)

**Use case:** Simpler alternative to full standardization when only centering needed -/
def centerPixels {n : Nat} (image : Vector n) : Vector n :=
  let sum := ∑ i, image[i]
  let mean := sum / n.toFloat
  ⊞ i => image[i] - mean

/-- Flatten 28×28 image to 784-dimensional vector in row-major order.

Converts 2D array representation to 1D vector suitable for neural network input.

**Parameters:**
- `image`: 2D array of Float values (expected to be 28×28)

**Returns:** 784-dimensional vector in IO monad:
- Success: Flattened pixels in row-major order (row 0, row 1, ..., row 27)
- Error: Zero vector with warning logged to stderr

**Error handling:**
- Validates dimensions are exactly 28×28
- Logs warning and returns zero vector on dimension mismatch

**Use case:** Converting external 2D image formats to Vector for processing -/
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

/-- Reshape 784-dimensional vector to 28×28 image.

Inverse operation of flattening: converts 1D vector to 2D array representation.

**Parameters:**
- `vector`: 784-dimensional Float vector (flattened image)

**Returns:** 28×28 2D array reconstructed in row-major order

**Properties:**
- Inverse of `flattenImage`: `flattenImage (reshapeToImage v) ≈ v` (modulo IO wrapper)
- First 28 elements → row 0, next 28 → row 1, etc.

**Use cases:**
- Visualization of network activations
- Debugging image transformations
- Converting internal representation to displayable format

**Implementation notes:**
Contains explicit bounds proofs using `omega` and USize arithmetic to satisfy Lean's termination checker -/
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

/-- Apply normalization to entire dataset (labels unchanged).

Maps `normalizePixels` over all images in the dataset while preserving labels.

**Parameters:**
- `dataset`: Array of (image, label) pairs

**Returns:** Dataset with normalized images and original labels

**Use case:** Preprocessing entire training or test set before training -/
def normalizeDataset {n : Nat} (dataset : Array (Vector n × Nat)) :
    Array (Vector n × Nat) :=
  dataset.map fun (image, label) => (normalizePixels image, label)

/-- Clip pixel values to [min, max] range (default [0.0, 1.0]).

Clamps all pixel values to specified bounds, useful for preventing outliers or enforcing constraints.

**Parameters:**
- `image`: Vector of pixel values
- `min`: Lower bound (default: 0.0)
- `max`: Upper bound (default: 1.0)

**Returns:** Vector where each pixel p is clamped to [min, max]

**Mathematical operation:** For each pixel p:
- If p < min, result is min
- If p > max, result is max
- Otherwise, result is p

**Use cases:**
- Prevent numerical issues from extreme values
- Enforce valid pixel range after transformations
- Remove outliers from noisy data -/
def clipPixels {n : Nat} (image : Vector n) (min : Float := 0.0) (max : Float := 1.0) :
    Vector n :=
  ⊞ i =>
    let val := image[i]
    if val < min then min
    else if val > max then max
    else val

end VerifiedNN.Data.Preprocessing
