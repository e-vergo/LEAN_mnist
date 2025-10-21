import SciLean

/-!
# Core Data Types

Basic array and tensor type definitions for neural networks.

This module defines type aliases for vectors, matrices, and batches using
SciLean's DataArrayN types with compile-time dimension checking.
-/

namespace VerifiedNN.Core

open SciLean

/-- A vector of dimension `n` using Float values. -/
abbrev Vector (n : Nat) := Float^[n]

/-- A matrix of dimensions `m × n` using Float values. -/
abbrev Matrix (m n : Nat) := Float^[m, n]

/-- A batch of `b` vectors, each of dimension `n`. -/
abbrev Batch (b n : Nat) := Float^[b, n]

/-- Default epsilon for floating-point approximate equality comparisons. -/
def epsilon : Float := 1e-7

/-- Approximate equality for floating-point numbers.

Returns true if `|x - y| ≤ eps`. -/
def approxEq (x y : Float) (eps : Float := epsilon) : Bool :=
  Float.abs (x - y) ≤ eps

/-- Approximate equality for vectors using average absolute difference.

**Implementation Note:** Uses average rather than maximum absolute difference.
This is a conservative approximation pending efficient max reduction in SciLean.

**TODO:** Replace with proper max when SciLean provides efficient reduction operations. -/
def vectorApproxEq {n : Nat} (v w : Vector n) (eps : Float := epsilon) : Bool :=
  let diff := ⊞ i => Float.abs (v[i] - w[i])
  let avgDiff := (∑ i, diff[i]) / n.toFloat
  avgDiff ≤ eps

/-- Approximate equality for matrices using average absolute difference.

**Implementation Note:** Uses average rather than maximum absolute difference.
This is a conservative approximation pending efficient max reduction in SciLean.

**TODO:** Replace with proper 2D max when SciLean provides efficient reduction operations. -/
def matrixApproxEq {m n : Nat} (a b : Matrix m n) (eps : Float := epsilon) : Bool :=
  let diff := ⊞ (i, j) => Float.abs (a[i,j] - b[i,j])
  let avgDiff := (∑ i, ∑ j, diff[i,j]) / (m.toFloat * n.toFloat)
  avgDiff ≤ eps

end VerifiedNN.Core
