/-
# Dense Layer

Dense (fully-connected) layer implementation with type-safe dimensions.

This module implements dense (fully-connected) layers with support for both
single-sample and batched forward passes. The implementation leverages dependent
types to ensure dimension compatibility at compile time.

**Verification Status:**
- Forward pass implementation: Working (uses SciLean primitives)
- Type safety: Enforced via dependent types
- Differentiability: Planned (requires Core.LinearAlgebra completion)

**Implementation Notes:**
- Forward pass computes: output = activation(weights @ input + bias)
- Batched operations process multiple samples efficiently
- Uses SciLean's DataArrayN for performance
-/

import VerifiedNN.Core.DataTypes
import VerifiedNN.Core.LinearAlgebra
import VerifiedNN.Core.Activation
import SciLean

namespace VerifiedNN.Layer

open VerifiedNN.Core
open VerifiedNN.Core.LinearAlgebra
open VerifiedNN.Core.Activation
open SciLean

/-- Dense layer structure with weights and biases.

A dense layer transforms an input vector of dimension `inDim` to an output
vector of dimension `outDim` via: output = activation(W @ input + b)

**Type Parameters:**
- `inDim`: Input dimension
- `outDim`: Output dimension

**Fields:**
- `weights`: Weight matrix of shape (outDim, inDim)
- `bias`: Bias vector of shape (outDim,)

**Example:**
```lean
let layer : DenseLayer 784 128 := {
  weights := ...,  -- 128×784 matrix
  bias := ...      -- 128-dimensional vector
}
```
-/
structure DenseLayer (inDim outDim : Nat) where
  weights : Matrix outDim inDim
  bias : Vector outDim

/-- Forward pass through a dense layer without activation.

Computes the linear transformation: output = W @ x + b

**Parameters:**
- `layer`: Dense layer with dimensions (n → m)
- `x`: Input vector of dimension n

**Returns:** Output vector of dimension m

**Note:** This is the pre-activation output. For typical neural network layers,
apply an activation function (e.g., ReLU) to the result.
-/
def DenseLayer.forwardLinear {m n : Nat} (layer : DenseLayer n m) (x : Vector n) : Vector m :=
  -- Linear transformation: Wx + b
  -- Note: This depends on LinearAlgebra implementations
  let wx := matvec layer.weights x
  vadd wx layer.bias

/-- Forward pass through a dense layer with optional activation.

Computes: output = activation(W @ x + b)

**Parameters:**
- `layer`: Dense layer with dimensions (n → m)
- `x`: Input vector of dimension n
- `activation`: Optional activation function to apply (default: identity)

**Returns:** Output vector of dimension m

**Verification:** Type system ensures dimension compatibility.
-/
def DenseLayer.forward {m n : Nat}
    (layer : DenseLayer n m)
    (x : Vector n)
    (activation : Vector m → Vector m := id) : Vector m :=
  let linear := layer.forwardLinear x
  activation linear

/-- Forward pass with ReLU activation.

Convenience function for: ReLU(W @ x + b)

**Parameters:**
- `layer`: Dense layer with dimensions (n → m)
- `x`: Input vector of dimension n

**Returns:** Output vector of dimension m with ReLU applied element-wise
-/
def DenseLayer.forwardReLU {m n : Nat} (layer : DenseLayer n m) (x : Vector n) : Vector m :=
  layer.forward x reluVec

/-- Batched forward pass through a dense layer without activation.

Processes multiple input samples in parallel for efficiency.
Computes: output[i] = W @ X[i] + b for each sample i

**Parameters:**
- `layer`: Dense layer with dimensions (n → m)
- `X`: Batch of b input vectors, each of dimension n

**Returns:** Batch of b output vectors, each of dimension m

**Performance:** More efficient than processing samples individually due to
vectorized operations and better cache utilization.
-/
def DenseLayer.forwardBatchLinear {b m n : Nat}
    (layer : DenseLayer n m)
    (X : Batch b n) : Batch b m :=
  -- Batched matrix-vector multiplication followed by bias addition
  -- Each row of X is multiplied by weights and bias is added
  let wx := batchMatvec layer.weights X
  batchAddVec wx layer.bias

/-- Batched forward pass with optional activation.

Applies the layer transformation to a batch of inputs with optional activation.

**Parameters:**
- `layer`: Dense layer with dimensions (n → m)
- `X`: Batch of b input vectors, each of dimension n
- `activation`: Optional activation function (default: identity)

**Returns:** Batch of b output vectors, each of dimension m
-/
def DenseLayer.forwardBatch {b m n : Nat}
    (layer : DenseLayer n m)
    (X : Batch b n)
    (activation : Batch b m → Batch b m := id) : Batch b m :=
  let linear := layer.forwardBatchLinear X
  activation linear

/-- Batched forward pass with ReLU activation.

Convenience function for batched ReLU layer.

**Parameters:**
- `layer`: Dense layer with dimensions (n → m)
- `X`: Batch of b input vectors, each of dimension n

**Returns:** Batch of b output vectors with ReLU applied
-/
def DenseLayer.forwardBatchReLU {b m n : Nat}
    (layer : DenseLayer n m)
    (X : Batch b n) : Batch b m :=
  layer.forwardBatch X reluBatch

-- Differentiability proofs (to be added when Core modules are complete)
-- These will establish that the forward pass is differentiable with respect to:
-- 1. Input vector x
-- 2. Weight matrix W
-- 3. Bias vector b

-- @[fun_prop]
-- theorem DenseLayer.forward_differentiable {m n : Nat} (layer : DenseLayer n m) :
--   Differentiable Float (fun x => layer.forward x) := by
--   sorry

-- @[fun_trans]
-- theorem DenseLayer.forward_fderiv {m n : Nat} (layer : DenseLayer n m) :
--   fderiv Float (fun x => layer.forward x) = fun x dx => layer.weights @ dx := by
--   sorry

end VerifiedNN.Layer
