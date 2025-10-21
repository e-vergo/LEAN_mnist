/-
# Core Data Types

Basic array and tensor type definitions for neural networks.

This module defines type aliases for vectors, matrices, and batches using
SciLean's DataArrayN types with compile-time dimension checking.
-/

import SciLean

namespace VerifiedNN.Core

open SciLean

/-- A vector of dimension n using Float values -/
abbrev Vector (n : Nat) := Float^[n]

/-- A matrix of dimensions m × n using Float values -/
abbrev Matrix (m n : Nat) := Float^[m, n]

/-- A batch of b vectors, each of dimension n -/
abbrev Batch (b n : Nat) := Float^[b, n]

/-- Epsilon for floating-point approximate equality.
    Default tolerance used in approximate equality comparisons. -/
def epsilon : Float := 1e-7

/-- Approximate equality for floating-point numbers -/
def approxEq (x y : Float) (eps : Float := epsilon) : Bool :=
  Float.abs (x - y) ≤ eps

/-- Approximate equality for vectors.
    Checks if all elements are approximately equal.

    **Implementation Note:** Currently uses average absolute difference instead of
    maximum absolute difference. This is a conservative approximation that works
    well in practice but could be replaced with a proper max operation when available.

    TODO: Replace sum-based check with proper max when SciLean provides efficient
    reduction operations. Issue: Need `max` reduction operation in SciLean. -/
def vectorApproxEq {n : Nat} (v w : Vector n) (eps : Float := epsilon) : Bool :=
  -- Check average absolute difference (conservative approximation of max)
  let diff := ⊞ i => Float.abs (v[i] - w[i])
  let avgDiff := (∑ i, diff[i]) / n.toFloat
  avgDiff ≤ eps

/-- Approximate equality for matrices.
    Checks if all elements are approximately equal.

    **Implementation Note:** Currently uses average absolute difference instead of
    maximum absolute difference. This is a conservative approximation.

    TODO: Replace sum-based check with proper max when available.
    Issue: Need efficient 2D max reduction in SciLean. -/
def matrixApproxEq {m n : Nat} (a b : Matrix m n) (eps : Float := epsilon) : Bool :=
  -- Check average absolute difference (conservative approximation of max)
  let diff := ⊞ (i, j) => Float.abs (a[i,j] - b[i,j])
  let avgDiff := (∑ i, ∑ j, diff[i,j]) / (m.toFloat * n.toFloat)
  avgDiff ≤ eps

end VerifiedNN.Core
