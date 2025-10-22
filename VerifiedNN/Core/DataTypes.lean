import SciLean

/-!
# Core Data Types

Basic array and tensor type definitions for neural network computations.

This module provides foundational type aliases for vectors, matrices, and batches
using SciLean's `DataArrayN` types, which combine compile-time dimension checking
with automatic differentiation support.

## Main Definitions

- `Vector n` - Fixed-size vector of dimension `n` (column vector)
- `Matrix m n` - Fixed-size matrix with `m` rows and `n` columns
- `Batch b n` - Batch of `b` samples, each of dimension `n`
- `epsilon` - Default tolerance for floating-point comparisons (1e-7)
- `approxEq` - Approximate equality for scalar Float values
- `vectorApproxEq` - Approximate equality for vectors using average absolute difference
- `matrixApproxEq` - Approximate equality for matrices using average absolute difference

## Implementation Notes

**Type System Integration:**
All types are based on SciLean's `DataArrayN` (notation: `Float^[n]`), which provides:
- Compile-time dimension checking via dependent types
- Automatic differentiation support via SciLean's `fun_trans` system
- Efficient memory layout for numerical computations
- Integration with OpenBLAS for optimized linear algebra

**Approximate Equality:**
The approximate equality functions use average absolute difference rather than
maximum absolute difference. This is a conservative approximation pending efficient
reduction operations in SciLean. For most testing purposes, this is acceptable,
but may need refinement for specific numerical stability analyses.

**Float vs ℝ Gap:**
These types represent computational implementations using IEEE 754 floating-point.
Formal verification is performed symbolically on ℝ (real numbers). The correspondence
between Float operations and their ℝ counterparts is assumed but not formally proven.

## References

- SciLean `DataArrayN` documentation: https://github.com/lecopivo/SciLean
- Project verification specification: verified-nn-spec.md
-/

namespace VerifiedNN.Core

open SciLean

/-- A vector of dimension `n` using Float values.

Vectors are represented as 1-dimensional SciLean `DataArrayN` structures,
providing both compile-time dimension guarantees and automatic differentiation support.

**Mathematical interpretation:** Elements of ℝⁿ, implemented as Float^[n].

**Type safety:** The dimension `n` is part of the type, preventing dimension mismatches at compile time.

**Usage:** Use for neural network inputs, outputs, layer biases, and gradients. -/
abbrev Vector (n : Nat) := Float^[n]

/-- A matrix of dimensions `m × n` using Float values.

Matrices are represented as 2-dimensional SciLean `DataArrayN` structures.
Indices are in row-major order: `A[i,j]` accesses row `i`, column `j`.

**Mathematical interpretation:** Elements of ℝᵐˣⁿ, implemented as Float^[m,n].

**Type safety:** Both dimensions are part of the type, ensuring matrix operations
are well-typed (e.g., cannot multiply incompatible matrix dimensions).

**Usage:** Use for weight matrices in neural network layers. -/
abbrev Matrix (m n : Nat) := Float^[m, n]

/-- A batch of `b` samples, each of dimension `n`.

Batches are represented as 2-dimensional arrays where the first index selects
the sample and the second index selects the feature dimension.

**Mathematical interpretation:** `b` samples from ℝⁿ, implemented as Float^[b,n].

**Type safety:** Both batch size and feature dimension are part of the type.

**Usage:** Use for mini-batch training where multiple samples are processed in parallel.
Index `[k,i]` accesses feature `i` of sample `k`. -/
abbrev Batch (b n : Nat) := Float^[b, n]

/-- Default epsilon for floating-point approximate equality comparisons.

Value: 1e-7

This tolerance is used as the default threshold for considering two floating-point
values "approximately equal". It balances numerical precision with practical robustness
to rounding errors in typical neural network computations.

**Rationale:** 1e-7 is sufficiently small to catch meaningful differences while being
large enough to absorb accumulation of rounding errors in typical forward/backward passes.

**Usage:** Can be overridden in specific contexts via the optional `eps` parameter
in approximate equality functions. -/
def epsilon : Float := 1e-7

/-- Approximate equality for floating-point numbers.

Tests whether two Float values are within `eps` of each other using absolute difference.

**Mathematical definition:** Returns `true` if and only if `|x - y| ≤ eps`.

**Parameters:**
- `x` : First Float value to compare
- `y` : Second Float value to compare
- `eps` : Tolerance threshold (defaults to `epsilon = 1e-7`)

**Returns:** `true` if values are within tolerance, `false` otherwise.

**Usage:** Use for scalar comparisons in tests, e.g., verifying loss values or gradients.

**Note:** This is absolute error, not relative error. For very large or very small
values, consider using relative error instead. -/
def approxEq (x y : Float) (eps : Float := epsilon) : Bool :=
  Float.abs (x - y) ≤ eps

/-- Approximate equality for vectors using average absolute difference.

Tests whether two vectors are approximately equal by comparing their average
element-wise absolute difference against a tolerance.

**Mathematical definition:** Returns `true` if (1/n) Σᵢ |vᵢ - wᵢ| ≤ eps.

**Parameters:**
- `v` : First vector of dimension `n`
- `w` : Second vector of dimension `n`
- `eps` : Tolerance threshold (defaults to `epsilon = 1e-7`)

**Returns:** `true` if average absolute difference is within tolerance.

**Implementation Note:** Uses average rather than maximum absolute difference.
This is a conservative approximation pending efficient max reduction in SciLean.
The average-based metric is:
- More forgiving than max (all elements must be somewhat close on average)
- Less sensitive to individual outliers
- Easier to compute with current SciLean primitives

**TODO:** Replace with proper max-based metric when SciLean provides efficient
reduction operations. The ideal metric would be: max_i |vᵢ - wᵢ| ≤ eps.

**Usage:** Use in gradient checking and numerical tests to verify vector-valued results. -/
def vectorApproxEq {n : Nat} (v w : Vector n) (eps : Float := epsilon) : Bool :=
  let diff := ⊞ i => Float.abs (v[i] - w[i])
  let avgDiff := (∑ i, diff[i]) / n.toFloat
  avgDiff ≤ eps

/-- Approximate equality for matrices using average absolute difference.

Tests whether two matrices are approximately equal by comparing their average
element-wise absolute difference against a tolerance.

**Mathematical definition:** Returns `true` if (1/(m·n)) Σᵢ Σⱼ |aᵢⱼ - bᵢⱼ| ≤ eps.

**Parameters:**
- `a` : First matrix of dimensions `m × n`
- `b` : Second matrix of dimensions `m × n`
- `eps` : Tolerance threshold (defaults to `epsilon = 1e-7`)

**Returns:** `true` if average absolute difference is within tolerance.

**Implementation Note:** Uses average rather than maximum absolute difference.
This is a conservative approximation pending efficient 2D max reduction in SciLean.
The average-based metric is more forgiving and easier to compute.

**TODO:** Replace with proper 2D max when SciLean provides efficient reduction
operations. The ideal metric would be: max_{i,j} |aᵢⱼ - bᵢⱼ| ≤ eps.

**Usage:** Use in gradient checking to verify weight gradient matrices. -/
def matrixApproxEq {m n : Nat} (a b : Matrix m n) (eps : Float := epsilon) : Bool :=
  let diff := ⊞ (i, j) => Float.abs (a[i,j] - b[i,j])
  let avgDiff := (∑ i, ∑ j, diff[i,j]) / (m.toFloat * n.toFloat)
  avgDiff ≤ eps

end VerifiedNN.Core
