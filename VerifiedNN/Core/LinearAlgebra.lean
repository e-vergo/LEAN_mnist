import VerifiedNN.Core.DataTypes
import SciLean
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Linear Algebra Operations

Matrix and vector operations using SciLean primitives with automatic differentiation support.

This module provides fundamental linear algebra operations for neural network computations,
including matrix-vector multiplication, vector arithmetic, batch operations, and transpose.
All operations preserve dimension information through dependent types and are designed to
work with SciLean's automatic differentiation system.

## Main Definitions

**Vector Operations:**
- `vadd` - Vector addition
- `vsub` - Vector subtraction
- `smul` - Scalar-vector multiplication
- `vmul` - Element-wise vector multiplication (Hadamard product)
- `dot` - Vector dot product
- `normSq` - Squared L2 norm
- `norm` - L2 norm

**Matrix Operations:**
- `matvec` - Matrix-vector multiplication
- `matmul` - Matrix-matrix multiplication
- `transpose` - Matrix transpose
- `matAdd` - Matrix addition
- `matSub` - Matrix subtraction
- `matSmul` - Scalar-matrix multiplication
- `outer` - Outer product of two vectors

**Batch Operations:**
- `batchMatvec` - Batch matrix-vector multiplication (efficient forward pass)
- `batchAddVec` - Add vector to each row of batch (bias addition)

## Main Results

**Linearity Properties (Proven):**
- `vadd_comm` - Vector addition is commutative
- `vadd_assoc` - Vector addition is associative
- `smul_vadd_distrib` - Scalar multiplication distributes over vector addition
- `matvec_linear` - Matrix-vector multiplication is linear
- `affine_combination_identity` - Affine combination property for scaled vectors

All theorems are proven with zero sorries using mathlib's `Finset.sum` properties.

## Implementation Notes

**Performance:**
- Uses SciLean's `DataArrayN` (Float^[n]) for efficient memory layout
- Operations marked `@[inline]` for performance
- Indexed sums (`∑`) and array constructors (`⊞`) are SciLean primitives optimized for AD
- Integration with OpenBLAS for optimized linear algebra (via SciLean)

**Automatic Differentiation:**
- All operations are automatically differentiable via SciLean's `fun_trans` system
- TODO: Register operations with `@[fun_trans]` and `@[fun_prop]` attributes
- Gradients computed symbolically on ℝ, executed on Float

**Type Safety:**
- Dimension consistency enforced by dependent types at compile time
- Type system prevents dimension mismatches (e.g., cannot multiply incompatible matrices)

## Verification Status

- **Linearity proofs:** ✅ Complete (5 theorems proven)
- **Differentiation properties:** ⚠️ TODO - Register with `@[fun_trans]` and `@[fun_prop]`
- **Dimension consistency:** ✅ Enforced by type system
- **Numerical correctness:** ✅ Validated via gradient checking tests

## References

- SciLean DataArrayN documentation: https://github.com/lecopivo/SciLean
- Mathlib BigOperators: `Mathlib.Algebra.BigOperators.Group.Finset.Basic`
- Project specification: verified-nn-spec.md
-/

namespace VerifiedNN.Core.LinearAlgebra

open SciLean
open VerifiedNN.Core
open BigOperators

/-- Matrix-vector multiplication: `A * x`.

Computes the product of matrix `A` with vector `x`, producing vector `y` where
`y[i] = Σⱼ A[i,j] * x[j]` for each row `i`.

**Mathematical interpretation:** Standard matrix-vector product from linear algebra.

**Type safety:** Dimensions are checked at compile time - matrix columns must match vector dimension.

**Parameters:**
- `A` : Matrix of dimensions `m × n`
- `x` : Vector of dimension `n`

**Returns:** Vector of dimension `m`

**Verified properties:**
- Linearity: See `matvec_linear` theorem

**TODO:** Register differentiation properties:
- `@[fun_prop] theorem matvec_differentiable`
- `@[fun_trans] theorem matvec_fderiv`

**Usage:** Forward pass in dense layers, computing `W * x` for weight matrix `W`. -/
@[inline]
def matvec {m n : Nat} (A : Matrix m n) (x : Vector n) : Vector m :=
  ⊞ i => ∑ j, A[i,j] * x[j]

/-- Matrix-matrix multiplication: `A * B`.

Computes the product of matrices `A` and `B`, producing matrix `C` where
`C[i,k] = Σⱼ A[i,j] * B[j,k]` for each row `i` and column `k`.

**Mathematical interpretation:** Standard matrix multiplication from linear algebra.

**Type safety:** Dimensions are checked at compile time - `A` must have as many columns as `B` has rows.

**Parameters:**
- `A` : Matrix of dimensions `m × n`
- `B` : Matrix of dimensions `n × p`

**Returns:** Matrix of dimensions `m × p`

**TODO:** Register differentiation properties:
- `@[fun_prop] theorem matmul_differentiable`
- `@[fun_trans] theorem matmul_fderiv`

**Usage:** Composing weight matrices or computing products in backpropagation. -/
@[inline]
def matmul {m n p : Nat} (A : Matrix m n) (B : Matrix n p) : Matrix m p :=
  ⊞ (i, k) => ∑ j, A[i,j] * B[j,k]

/-- Vector addition: `x + y`.

Element-wise addition of two vectors, producing `z[i] = x[i] + y[i]` for each index `i`.

**Mathematical interpretation:** Vector addition in ℝⁿ.

**Verified properties:**
- Commutativity: `vadd_comm`
- Associativity: `vadd_assoc`
- Distributivity with scalar multiplication: `smul_vadd_distrib`

**Parameters:**
- `x` : Vector of dimension `n`
- `y` : Vector of dimension `n`

**Returns:** Vector of dimension `n`

**TODO:** Register differentiation properties (gradient is identity for both arguments).

**Usage:** Adding gradients, bias terms, or residual connections. -/
@[inline]
def vadd {n : Nat} (x y : Vector n) : Vector n :=
  ⊞ i => x[i] + y[i]

/-- Vector subtraction: `x - y`.

Element-wise subtraction of two vectors, producing `z[i] = x[i] - y[i]` for each index `i`.

**Mathematical interpretation:** Vector subtraction in ℝⁿ.

**Parameters:**
- `x` : Vector of dimension `n`
- `y` : Vector of dimension `n`

**Returns:** Vector of dimension `n`

**TODO:** Register differentiation properties (gradient is +1 for x, -1 for y).

**Usage:** Computing prediction errors, gradient differences, or residuals. -/
@[inline]
def vsub {n : Nat} (x y : Vector n) : Vector n :=
  ⊞ i => x[i] - y[i]

/-- Scalar-vector multiplication: `c * x`.

Multiply each element of vector `x` by scalar `c`, producing `y[i] = c * x[i]` for each index `i`.

**Mathematical interpretation:** Scalar multiplication in ℝⁿ.

**Verified properties:**
- Distributivity: `smul_vadd_distrib` shows `c * (x + y) = c * x + c * y`

**Parameters:**
- `c` : Scalar multiplier (Float)
- `x` : Vector of dimension `n`

**Returns:** Vector of dimension `n`

**TODO:** Register differentiation properties (gradient w.r.t. x is c, w.r.t. c is x).

**Usage:** Scaling gradients in optimization, learning rate application. -/
@[inline]
def smul {n : Nat} (c : Float) (x : Vector n) : Vector n :=
  ⊞ i => c * x[i]

/-- Element-wise vector multiplication (Hadamard product): `x ⊙ y`.

Multiply corresponding elements of two vectors, producing `z[i] = x[i] * y[i]` for each index `i`.

**Mathematical interpretation:** Hadamard (element-wise) product, not to be confused with dot product.

**Parameters:**
- `x` : Vector of dimension `n`
- `y` : Vector of dimension `n`

**Returns:** Vector of dimension `n`

**TODO:** Register differentiation properties (gradient w.r.t. x is y, w.r.t. y is x).

**Usage:** Applying activation derivatives in backpropagation, gating mechanisms. -/
@[inline]
def vmul {n : Nat} (x y : Vector n) : Vector n :=
  ⊞ i => x[i] * y[i]

/-- Vector dot product: `⟨x, y⟩`.

Computes the inner product of two vectors: `⟨x, y⟩ = Σᵢ x[i] * y[i]`.

**Mathematical interpretation:** Standard inner product in ℝⁿ.

**Parameters:**
- `x` : Vector of dimension `n`
- `y` : Vector of dimension `n`

**Returns:** Scalar (Float) representing the inner product

**TODO:** Register differentiation properties (gradient w.r.t. x is y, w.r.t. y is x).

**Usage:** Computing similarities, loss functions, or vector projections. -/
@[inline]
def dot {n : Nat} (x y : Vector n) : Float :=
  ∑ i, x[i] * y[i]

/-- Vector L2 norm squared: `‖x‖²`.

Computes the squared L2 norm of vector `x`: `‖x‖² = Σᵢ x[i]²`.

**Mathematical interpretation:** Squared Euclidean norm in ℝⁿ.

**Parameters:**
- `x` : Vector of dimension `n`

**Returns:** Scalar (Float) representing `‖x‖²`

**TODO:** Register differentiation properties (gradient is `2 * x`).

**Usage:** Regularization terms, gradient magnitude checks, avoiding sqrt for efficiency. -/
@[inline]
def normSq {n : Nat} (x : Vector n) : Float :=
  ∑ i, x[i] * x[i]

/-- Vector L2 norm: `‖x‖`.

Computes the L2 norm of vector `x`: `‖x‖ = √(Σᵢ x[i]²)`.

**Mathematical interpretation:** Euclidean norm (length) in ℝⁿ.

**Parameters:**
- `x` : Vector of dimension `n`

**Returns:** Scalar (Float) representing `‖x‖`

**TODO:** Register differentiation properties.
Gradient is `x / ‖x‖` when `x ≠ 0`, undefined at zero.

**Note:** Gradient clipping and some regularization terms use this.
Consider `normSq` for efficiency when the square root is not needed.

**Usage:** Gradient clipping, normalization, computing distances. -/
@[inline]
def norm {n : Nat} (x : Vector n) : Float :=
  Float.sqrt (normSq x)

/-- Matrix transpose: `Aᵀ`.

Swaps rows and columns of matrix `A`, producing `Aᵀ[j,i] = A[i,j]` for all indices.

**Mathematical interpretation:** Standard matrix transpose from linear algebra.

**Parameters:**
- `A` : Matrix of dimensions `m × n`

**Returns:** Matrix of dimensions `n × m`

**TODO:** Register differentiation properties (gradient w.r.t. A is also transposed).

**Usage:** Backpropagation through dense layers, computing `Wᵀ * δ`. -/
@[inline]
def transpose {m n : Nat} (A : Matrix m n) : Matrix n m :=
  ⊞ (j, i) => A[i,j]

/-- Matrix addition: `A + B`.

Element-wise addition of two matrices, producing `C[i,j] = A[i,j] + B[i,j]` for all indices.

**Mathematical interpretation:** Matrix addition in ℝᵐˣⁿ.

**Parameters:**
- `A` : Matrix of dimensions `m × n`
- `B` : Matrix of dimensions `m × n`

**Returns:** Matrix of dimensions `m × n`

**TODO:** Register differentiation properties (gradient is identity for both arguments).

**Usage:** Accumulating weight gradients, residual connections. -/
@[inline]
def matAdd {m n : Nat} (A B : Matrix m n) : Matrix m n :=
  ⊞ (i, j) => A[i,j] + B[i,j]

/-- Matrix subtraction: `A - B`.

Element-wise subtraction of two matrices, producing `C[i,j] = A[i,j] - B[i,j]` for all indices.

**Mathematical interpretation:** Matrix subtraction in ℝᵐˣⁿ.

**Parameters:**
- `A` : Matrix of dimensions `m × n`
- `B` : Matrix of dimensions `m × n`

**Returns:** Matrix of dimensions `m × n`

**TODO:** Register differentiation properties (gradient is +1 for A, -1 for B).

**Usage:** Computing weight differences, gradient updates. -/
@[inline]
def matSub {m n : Nat} (A B : Matrix m n) : Matrix m n :=
  ⊞ (i, j) => A[i,j] - B[i,j]

/-- Scalar-matrix multiplication: `c * A`.

Multiply each element of matrix `A` by scalar `c`, producing `B[i,j] = c * A[i,j]` for all indices.

**Mathematical interpretation:** Scalar multiplication in ℝᵐˣⁿ.

**Parameters:**
- `c` : Scalar multiplier (Float)
- `A` : Matrix of dimensions `m × n`

**Returns:** Matrix of dimensions `m × n`

**TODO:** Register differentiation properties (gradient w.r.t. A is c, w.r.t. c is A).

**Usage:** Scaling weight matrices, applying learning rates to weight gradients. -/
@[inline]
def matSmul {m n : Nat} (c : Float) (A : Matrix m n) : Matrix m n :=
  ⊞ (i, j) => c * A[i,j]

/-- Batch matrix-vector multiplication: `X * Aᵀ`.

For each row in batch, compute matrix-vector product with transpose:
`Y[k,i] = Σⱼ A[i,j] * X[k,j]` which is equivalent to `Y = X * Aᵀ`.

**Mathematical interpretation:** Batched application of linear transformation Aᵀ.

**Type safety:** Batch feature dimension must match matrix columns.

**Parameters:**
- `A` : Matrix of dimensions `m × n`
- `X` : Batch of `b` samples, each of dimension `n`

**Returns:** Batch of `b` samples, each of dimension `m`

**TODO:** Register differentiation properties.

**Usage:** Efficient forward pass in batched neural network training.
This computes `W * xᵢ` for all `b` samples simultaneously. -/
@[inline]
def batchMatvec {b m n : Nat} (A : Matrix m n) (X : Batch b n) : Batch b m :=
  ⊞ (k, i) => ∑ j, A[i,j] * X[k,j]

/-- Add a vector to each row of a batch (broadcasting): `X .+ v`.

Adds vector `v` to each row of batch `X`, producing `Y[k,j] = X[k,j] + v[j]` for all indices.

**Mathematical interpretation:** Broadcasting addition - adds same vector to all samples.

**Parameters:**
- `X` : Batch of `b` samples, each of dimension `n`
- `v` : Vector of dimension `n`

**Returns:** Batch of `b` samples, each of dimension `n`

**TODO:** Register differentiation properties.
Gradient w.r.t. X is identity, gradient w.r.t. v is sum over batch.

**Usage:** Adding bias vectors in neural network layers during forward pass.
Computes `W * x + b` for all batch samples efficiently. -/
@[inline]
def batchAddVec {b n : Nat} (X : Batch b n) (v : Vector n) : Batch b n :=
  ⊞ (k, j) => X[k,j] + v[j]

/-- Outer product of two vectors: `x ⊗ y`.

Creates matrix from two vectors where `A[i,j] = x[i] * y[j]` for all indices.

**Mathematical interpretation:** Outer product (tensor product) in ℝᵐ ⊗ ℝⁿ.

**Parameters:**
- `x` : Vector of dimension `m`
- `y` : Vector of dimension `n`

**Returns:** Matrix of dimensions `m × n`

**TODO:** Register differentiation properties.
Gradient w.r.t. x is `A_grad * y`, gradient w.r.t. y is `Aᵀ_grad * x`.

**Usage:** Computing weight gradients in backpropagation.
The gradient of a dense layer weight is `δ ⊗ input` where δ is the output gradient.

**Example:** For layer `f(x) = W * x`, the weight gradient is `∂L/∂W = (∂L/∂f) ⊗ x`. -/
@[inline]
def outer {m n : Nat} (x : Vector m) (y : Vector n) : Matrix m n :=
  ⊞ (i, j) => x[i] * y[j]

-- ============================================================================
-- Linearity Properties
-- ============================================================================

/-- Vector addition is commutative.

Mathematical Statement: x + y = y + x

**Proof Strategy:** Use funext to prove element-wise equality, then apply Float commutativity.
-/
theorem vadd_comm {n : Nat} (x y : Vector n) : vadd x y = vadd y x := by
  unfold vadd; congr; funext i; ring

/-- Vector addition is associative.

Mathematical Statement: (x + y) + z = x + (y + z)

**Proof Strategy:** Use funext to prove element-wise equality, then apply Float associativity.
-/
theorem vadd_assoc {n : Nat} (x y z : Vector n) :
  vadd (vadd x y) z = vadd x (vadd y z) := by
  unfold vadd; congr; funext i; simp only [SciLean.getElem_ofFn]; ring

/-- Scalar multiplication distributes over vector addition.

Mathematical Statement: α * (x + y) = α * x + α * y

**Proof Strategy:** Use funext to prove element-wise equality, then apply Float distributivity.
-/
theorem smul_vadd_distrib {n : Nat} (α : Float) (x y : Vector n) :
  smul α (vadd x y) = vadd (smul α x) (smul α y) := by
  unfold smul vadd; congr; funext i; simp only [SciLean.getElem_ofFn]; ring

/-- Matrix-vector multiplication is linear.

Mathematical Statement:
  A @ (α * x + β * y) = α * (A @ x) + β * (A @ y)

This is the fundamental linearity property of matrix-vector multiplication.

**Proof Strategy:**
Unfold definitions and prove element-wise equality.
For each row i:
  LHS[i] = Σⱼ A[i,j] * (α*x[j] + β*y[j])
         = Σⱼ (A[i,j] * α * x[j] + A[i,j] * β * y[j])
         = α * Σⱼ A[i,j]*x[j] + β * Σⱼ A[i,j]*y[j]
         = α * (A@x)[i] + β * (A@y)[i]
         = RHS[i]
-/
theorem matvec_linear {m n : Nat} (A : Matrix m n) (x y : Vector n) (α β : Float) :
  matvec A (vadd (smul α x) (smul β y)) =
  vadd (smul α (matvec A x)) (smul β (matvec A y)) := by
  unfold matvec vadd smul
  congr
  funext i
  simp only [SciLean.getElem_ofFn]
  -- The goal is to show: ∑ j, A[i,j] * (α * x[j] + β * y[j]) = α * ∑ j, A[i,j] * x[j] + β * ∑ j, A[i,j] * y[j]
  -- This is a direct application of sum distributivity and scalar factoring
  calc ∑ j, A[(i,j)] * (α * x[j] + β * y[j])
      = ∑ j, (A[(i,j)] * α * x[j] + A[(i,j)] * β * y[j]) := by congr 1; funext j; ring
    _ = (∑ j, A[(i,j)] * α * x[j]) + (∑ j, A[(i,j)] * β * y[j]) := Finset.sum_add_distrib
    _ = (∑ j, α * (A[(i,j)] * x[j])) + (∑ j, β * (A[(i,j)] * y[j])) := by congr 1 <;> (congr 1; funext j; ring)
    _ = α * (∑ j, A[(i,j)] * x[j]) + β * (∑ j, A[(i,j)] * y[j]) := by rw [← Finset.mul_sum, ← Finset.mul_sum]

/-- When scalars sum to one, a vector equals the sum of scaled copies.

Mathematical Statement: When α + β = 1, then b = α·b + β·b

This is a fundamental property of affine combinations.
-/
theorem affine_combination_identity {n : Nat} (α β : Float) (b : Vector n) (h : α + β = 1) :
  b = vadd (smul α b) (smul β b) := by
  unfold vadd smul; ext i
  simp only [SciLean.getElem_ofFn]
  calc b[i]
      = 1 * b[i] := by ring
    _ = (α + β) * b[i] := by rw [← h]
    _ = α * b[i] + β * b[i] := by ring
    _ = (⊞ i => α * b[i] + β * b[i])[i] := by simp

end VerifiedNN.Core.LinearAlgebra
