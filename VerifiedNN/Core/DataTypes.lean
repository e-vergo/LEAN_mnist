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

/-- Epsilon for floating-point approximate equality -/
def epsilon : Float := 1e-7

/-- Approximate equality for floating-point numbers -/
def approxEq (x y : Float) (eps : Float := epsilon) : Bool :=
  Float.abs (x - y) ≤ eps

/-- Approximate equality for vectors -/
def vectorApproxEq {n : Nat} (v w : Vector n) (eps : Float := epsilon) : Bool :=
  sorry -- TODO: implement element-wise comparison

/-- Approximate equality for matrices -/
def matrixApproxEq {m n : Nat} (a b : Matrix m n) (eps : Float := epsilon) : Bool :=
  sorry -- TODO: implement element-wise comparison

end VerifiedNN.Core
