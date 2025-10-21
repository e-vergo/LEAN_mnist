# VerifiedNN Core Module

Foundation types and operations for the verified neural network implementation.

## Overview

The Core module provides fundamental data structures and operations for neural network computations using SciLean's automatically differentiable array primitives. All operations are designed to work with SciLean's automatic differentiation system while enforcing dimension consistency through dependent types.

## Module Organization

### DataTypes.lean

Defines core type aliases and approximate equality predicates:

- `Vector n` - Fixed-size vector of dimension `n`
- `Matrix m n` - Fixed-size matrix of dimensions `m × n`
- `Batch b n` - Batch of `b` vectors, each of dimension `n`
- `approxEq`, `vectorApproxEq`, `matrixApproxEq` - Floating-point approximate equality

All types are built on SciLean's `DataArrayN` (Float^[n]) for optimal performance and AD integration.

### LinearAlgebra.lean

Matrix and vector operations with formal linearity properties:

**Operations:**
- Matrix-vector multiplication: `matvec`
- Matrix-matrix multiplication: `matmul`
- Vector operations: `vadd`, `vsub`, `smul`, `vmul`, `dot`
- Matrix operations: `transpose`, `matAdd`, `matSub`, `matSmul`
- Batch operations: `batchMatvec`, `batchAddVec`
- Outer product: `outer`

**Verified Properties:**
- `vadd_comm` - Vector addition is commutative
- `vadd_assoc` - Vector addition is associative
- `smul_vadd_distrib` - Scalar multiplication distributes over addition
- `matvec_linear` - Matrix-vector multiplication is linear (currently admitted with `sorry`)
- `affine_combination_identity` - Affine combination property

### Activation.lean

Common activation functions for neural networks:

**Functions:**
- `relu`, `reluVec`, `reluBatch` - ReLU activation
- `softmax` - Softmax for classification (numerically unstable, needs log-sum-exp trick)
- `sigmoid`, `sigmoidVec`, `sigmoidBatch` - Sigmoid activation
- `leakyRelu`, `leakyReluVec` - Leaky ReLU with configurable slope
- `tanh`, `tanhVec` - Hyperbolic tangent

**Analytical Derivatives:** Provided separately for gradient checking:
- `reluDerivative`, `sigmoidDerivative`, `tanhDerivative`, `leakyReluDerivative`

## Key Definitions

```lean
abbrev Vector (n : Nat) := Float^[n]
abbrev Matrix (m n : Nat) := Float^[m, n]
abbrev Batch (b n : Nat) := Float^[b, n]

def matvec {m n : Nat} (A : Matrix m n) (x : Vector n) : Vector m :=
  ⊞ i => ∑ j, A[i,j] * x[j]

def relu (x : Float) : Float :=
  if x > 0 then x else 0
```

## Build Status

✅ **All modules build successfully**

Current warnings:
- `LinearAlgebra.lean:225:8` - `matvec_linear` theorem uses `sorry` (awaiting compatibility lemmas between SciLean's indexed sums and mathlib's Finset.sum)

## Dependencies

- **SciLean** - Scientific computing library with automatic differentiation
- **Mathlib** - `Mathlib.Algebra.BigOperators.Group.Finset.Basic` for sum properties

## Verification Status

### Type Safety
- ✅ Dimension consistency enforced by dependent types
- ✅ Compile-time dimension checking via type system

### Differentiation Properties
- ⚠️ TODO - Register operations with `@[fun_trans]` and `@[fun_prop]` attributes
- ⚠️ Awaiting SciLean support for `Float.exp` differentiability

### Numerical Correctness
- ✅ Validated via gradient checking tests (see `VerifiedNN/Testing/`)
- ⚠️ Softmax lacks numerical stability (log-sum-exp trick needed)

## Known Limitations

1. **Float vs ℝ Gap:** Verification is symbolic (ℝ), implementation is Float (IEEE 754)
2. **ReLU Gradient:** Undefined at x=0; uses 0 as subgradient by convention
3. **Softmax Stability:** May overflow for large inputs without log-sum-exp trick
4. **Activation AD:** Float.exp may not be differentiable in current SciLean version
5. **Approximate Equality:** Uses average rather than max absolute difference (pending efficient reduction operations in SciLean)

## Development Notes

### Float Constraints
- Cannot prove `(0.0 : Float) = (0.0 + 0.0)` with `rfl` (Float is opaque)
- No canonical verified Float library in Lean 4 (unlike Coq's Flocq)

### Performance
- Use `@[inline]` and `@[specialize]` for hot-path functions
- Prefer `DataArrayN` over `Array Float` for numerical operations
- Leverage SciLean's indexed sums (`∑ᴵ`) for better performance

### Next Steps
1. Complete `matvec_linear` proof with SciLean/mathlib compatibility lemmas
2. Register all operations with `@[fun_trans]` and `@[fun_prop]`
3. Implement numerically stable softmax with log-sum-exp trick
4. Add formal differentiation correctness proofs

## Usage Example

```lean
import VerifiedNN.Core.DataTypes
import VerifiedNN.Core.LinearAlgebra
import VerifiedNN.Core.Activation

open VerifiedNN.Core
open VerifiedNN.Core.LinearAlgebra
open VerifiedNN.Core.Activation

-- Create a 3×2 matrix and 2D vector
def A : Matrix 3 2 := ⊞ (i, j) => (i.val.toFloat + j.val.toFloat)
def x : Vector 2 := ⊞ i => i.val.toFloat

-- Compute matrix-vector product
def y : Vector 3 := matvec A x

-- Apply ReLU activation
def activated : Vector 3 := reluVec y
```

## References

- [SciLean Documentation](https://github.com/lecopivo/SciLean)
- [Project Technical Specification](../../verified-nn-spec.md)
- [CLAUDE.md Development Guide](../../CLAUDE.md)

---

**Last Updated:** 2025-10-21
