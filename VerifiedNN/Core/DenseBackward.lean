import VerifiedNN.Core.DataTypes
import VerifiedNN.Core.LinearAlgebra
import SciLean

/-!
# Dense Layer Backward Pass

Computes gradients for a dense (fully-connected) neural network layer.

## Mathematical Formulation

For a dense layer forward pass: `y = Wx + b`

Given the gradient ∂L/∂y from the layer above, we compute:
- **∂L/∂W = (∂L/∂y) ⊗ x** (outer product)
- **∂L/∂b = ∂L/∂y** (identity)
- **∂L/∂x = W^T @ (∂L/∂y)** (transpose multiply)

## Key Functions
- `denseLayerBackward`: Main backward pass computation

## Verification Status
- **Computable:** Yes (uses only computable linear algebra operations)
- **Correctness:** Should be validated against finite differences in testing
- **Differentiability:** All operations are compositions of differentiable primitives

## Implementation Notes

The backward pass implementation follows the standard backpropagation algorithm:
1. Weight gradient uses outer product between output gradient and input
2. Bias gradient is simply the output gradient (identity function)
3. Input gradient uses transposed weight matrix multiplication

All operations preserve dimension information through dependent types, ensuring
correctness at compile time.
-/

namespace VerifiedNN.Core.DenseBackward

open VerifiedNN.Core
open VerifiedNN.Core.LinearAlgebra
open SciLean

set_default_scalar Float

/-- Compute gradients for a dense layer.

Given the gradient of the loss with respect to the layer's output,
computes gradients with respect to weights, bias, and input.

**Mathematical Background:**

For a dense layer computing `y = Wx + b`:
- The weight gradient is the outer product: `∂L/∂W[i,j] = ∂L/∂y[i] * x[j]`
- The bias gradient is identity: `∂L/∂b[i] = ∂L/∂y[i]`
- The input gradient uses transpose: `∂L/∂x[j] = Σ_i W[i,j] * ∂L/∂y[i] = W^T @ ∂L/∂y`

**Parameters:**
- `gradOutput`: Gradient ∂L/∂output from layer above [m]
- `input`: Input vector that was used in forward pass [n]
- `weights`: Layer weight matrix [m, n]

**Returns:** Tuple of:
- `dW`: Gradient w.r.t. weights [m, n]
- `db`: Gradient w.r.t. bias [m]
- `dInput`: Gradient w.r.t. input [n]

**Computational Cost:** O(m*n) for each of weight and input gradients

**Type Safety:**
All dimensions are checked at compile time. The type system ensures that:
- gradOutput has dimension m (matching weight matrix rows)
- input has dimension n (matching weight matrix columns)
- returned gradients have the correct shapes

**Example:**
```lean
let weights : Matrix 10 784 := ...  -- Output layer
let input : Vector 784 := ...       -- Flattened image
let gradOutput : Vector 10 := ...   -- From loss function

let (dW, db, dInput) := denseLayerBackward gradOutput input weights
-- dW has shape [10, 784], db has shape [10], dInput has shape [784]
```

**Verification Note:**
This implementation should be validated via numerical gradient checking.
The gradients computed here must match finite difference approximations
within acceptable tolerance (typically 1e-5 to 1e-7).
-/
@[inline]
def denseLayerBackward
  {m n : Nat}
  (gradOutput : Vector m)
  (input : Vector n)
  (weights : Matrix m n)
  : (Matrix m n × Vector m × Vector n) :=
  -- Weight gradient: outer product of output gradient and input
  -- ∂L/∂W = (∂L/∂y) ⊗ x
  let dW := outer gradOutput input

  -- Bias gradient: identity (gradient flows through unchanged)
  -- ∂L/∂b = ∂L/∂y
  let db := gradOutput

  -- Input gradient: transpose multiply
  -- ∂L/∂x = W^T @ (∂L/∂y)
  let weightsT := transpose weights
  let dInput := matvec weightsT gradOutput

  (dW, db, dInput)

-- ============================================================================
-- Example Usage
-- ============================================================================

/-- Example demonstrating the backward pass computation.

This shows how to compute gradients for a simple 2x3 dense layer.
The type system ensures all dimensions are correct at compile time.
-/
example : True := by
  -- Simple 2x3 example
  -- Weight matrix: 2 outputs, 3 inputs
  let weights : Matrix 2 3 := ⊞[0.0, 1.0, 2.0; 1.0, 2.0, 3.0]

  -- Input vector: 3 elements
  let input : Vector 3 := ⊞[1.0, 2.0, 3.0]

  -- Gradient from layer above: 2 elements
  let gradOutput : Vector 2 := ⊞[1.0, 1.0]

  -- Compute gradients
  let (dW, db, dInput) := denseLayerBackward gradOutput input weights

  -- Type checker verifies:
  -- dW : Matrix 2 3 (gradient of weights)
  -- db : Vector 2 (gradient of bias)
  -- dInput : Vector 3 (gradient of input)

  trivial

end VerifiedNN.Core.DenseBackward
