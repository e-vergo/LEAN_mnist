import VerifiedNN.Core.DataTypes
import VerifiedNN.Core.LinearAlgebra
import VerifiedNN.Core.Activation
import SciLean
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic

/-!
# Dense Layer

Dense (fully-connected) layer implementation with compile-time dimension safety.

## Main Definitions

* `DenseLayer inDim outDim`: Dense layer structure transforming `inDim`-dimensional inputs to `outDim`-dimensional outputs
* `DenseLayer.forwardLinear`: Linear transformation `Wx + b` without activation
* `DenseLayer.forward`: Forward pass `activation(Wx + b)` with optional activation function
* `DenseLayer.forwardReLU`: Forward pass with ReLU activation
* `DenseLayer.forwardBatchLinear`: Batched linear transformation for multiple samples
* `DenseLayer.forwardBatch`: Batched forward pass with optional activation
* `DenseLayer.forwardBatchReLU`: Batched forward pass with ReLU activation

## Main Results

Type-level dimension safety is guaranteed by Lean's dependent type system:
* If `layer : DenseLayer n m` and `x : Vector n` type-check, then `layer.forward x : Vector m`
* Batch operations preserve both batch size and output dimension
* Composition with other layers enforces intermediate dimension matching

## Implementation Notes

The forward pass computes `activation(Wx + b)` where:
- `W : Matrix outDim inDim` is the weight matrix
- `b : Vector outDim` is the bias vector
- `activation : Vector outDim → Vector outDim` is an optional activation function

Batched operations use SciLean's `DataArrayN` for efficient vectorized computations.
All functions are marked `@[inline]` for performance in hot paths.

The `forwardLinear` variant computes the pre-activation output `Wx + b`, which is useful
for analyzing the affine transformation properties or when activation will be applied separately.

## Verification Status

- **Build status:** ✅ Compiles with zero errors
- **Sorries:** 0
- **Axioms:** 0
- **Type safety:** ✅ Enforced via dependent types (compile-time dimension checking)
- **Forward pass correctness:** ✅ Implemented using verified SciLean primitives
- **Differentiability:** Planned (requires integration with SciLean's automatic differentiation)
- **Gradient correctness:** Planned (see VerifiedNN/Verification/GradientCorrectness.lean)

## References

- SciLean documentation: https://github.com/lecopivo/SciLean
- Layer composition properties: VerifiedNN.Layer.Composition
- Mathematical properties: VerifiedNN.Layer.Properties
-/

namespace VerifiedNN.Layer

open VerifiedNN.Core
open VerifiedNN.Core.LinearAlgebra
open VerifiedNN.Core.Activation
open SciLean

/-- A dense (fully-connected) layer with learnable weights and biases.

Represents an affine transformation followed by an optional activation function.
The forward pass computes `activation(W @ input + b)` where `W` is the weight matrix
and `b` is the bias vector.

**Mathematical formulation:**
For input `x : ℝ^inDim`, the layer computes `σ(Wx + b) : ℝ^outDim` where:
- `W : ℝ^(outDim × inDim)` is the weight matrix
- `b : ℝ^outDim` is the bias vector
- `σ : ℝ^outDim → ℝ^outDim` is the activation function (e.g., ReLU, softmax, identity)

**Type safety:**
Dimension compatibility is enforced at compile time through dependent types.
If `layer : DenseLayer n m` type-checks, then `n` and `m` are the correct dimensions
for all layer operations.

**Fields:**
- `weights : Matrix outDim inDim`: Weight matrix with shape `(outDim, inDim)`
- `bias : Vector outDim`: Bias vector with shape `(outDim,)`

**Usage:**
```lean
-- MNIST hidden layer: 784 input features → 128 hidden units
def hiddenLayer : DenseLayer 784 128 := {
  weights := initializeXavier 784 128  -- Use proper initialization
  bias := ⊞ _ => 0.0
}
```

**Implementation notes:**
Uses SciLean's `DataArrayN` for efficient array operations. Weights and biases are
mutable in the training loop but immutable within forward/backward passes.

**References:**
- Weight initialization: VerifiedNN.Network.Initialization
- Forward pass operations: `forwardLinear`, `forward`, `forwardReLU`
- Batched operations: `forwardBatch`, `forwardBatchReLU` -/
structure DenseLayer (inDim outDim : Nat) where
  weights : Matrix outDim inDim
  bias : Vector outDim

/-- Linear transformation without activation: computes `Wx + b`.

Performs the affine transformation that forms the core of a dense layer,
without applying any activation function. This is the pre-activation output.

**Mathematical specification:**
Given `layer : DenseLayer n m` and `x : Vector n`, computes:
```
forwardLinear(x) = Wx + b
```
where `W = layer.weights` and `b = layer.bias`.

**Parameters:**
- `layer : DenseLayer n m`: Dense layer with weight matrix `W : ℝ^(m×n)` and bias `b : ℝ^m`
- `x : Vector n`: Input vector

**Returns:**
- `Vector m`: Pre-activation output `Wx + b`

**Verified properties:**
- Output dimension matches layer's output dimension (type-level guarantee)
- Computes an affine transformation (proven in VerifiedNN.Layer.Properties)
- Preserves affine combinations when coefficient sum equals 1

**Usage:**
```lean
-- Get pre-activation values for analysis or custom activation
let preActivation := layer.forwardLinear input
let customOutput := myActivation preActivation
```

**Implementation notes:**
Uses `matvec` for matrix-vector multiplication and `vadd` for vector addition,
both from VerifiedNN.Core.LinearAlgebra. Marked `@[inline]` for performance.

**References:**
- Activation functions: VerifiedNN.Core.Activation
- Full forward pass: `forward`, `forwardReLU`
- Affine property proof: VerifiedNN.Layer.Properties.forwardLinear_is_affine -/
@[inline]
def DenseLayer.forwardLinear {m n : Nat} (layer : DenseLayer n m) (x : Vector n) : Vector m :=
  let wx := matvec layer.weights x
  vadd wx layer.bias

/-- Forward pass with optional activation: computes `activation(Wx + b)`.

Performs the complete dense layer transformation: affine transformation followed
by activation function. This is the standard layer operation in neural networks.

**Mathematical specification:**
Given `layer : DenseLayer n m`, input `x : Vector n`, and activation `σ`, computes:
```
forward(x, σ) = σ(Wx + b)
```

**Parameters:**
- `layer : DenseLayer n m`: Dense layer with weights and biases
- `x : Vector n`: Input vector
- `activation : Vector m → Vector m`: Activation function (default: identity)

**Returns:**
- `Vector m`: Activated output `σ(Wx + b)`

**Type safety:**
Dimension compatibility is enforced at compile time. If this function type-checks,
then the input dimension matches the layer's input dimension, and the activation
function signature matches the layer's output dimension.

**Verified properties:**
- Output dimension equals layer's output dimension (type-level guarantee)
- When `activation = id`, this reduces to `forwardLinear`
- Differentiability follows from composition of differentiable functions (planned proof)

**Usage:**
```lean
-- With ReLU activation
let output1 := layer.forward input reluVec

-- With identity (no activation)
let output2 := layer.forward input id

-- Custom activation
let output3 := layer.forward input mySigmoid
```

**Implementation notes:**
Computes affine transformation first (`forwardLinear`), then applies activation.
This separation allows for easy proof of differentiability via chain rule.

**References:**
- Pre-activation output: `forwardLinear`
- Common activations: VerifiedNN.Core.Activation (ReLU, softmax)
- Convenience function: `forwardReLU` -/
@[inline]
def DenseLayer.forward {m n : Nat}
    (layer : DenseLayer n m)
    (x : Vector n)
    (activation : Vector m → Vector m := id) : Vector m :=
  let linear := layer.forwardLinear x
  activation linear

/-- Forward pass with ReLU activation: computes `ReLU(Wx + b)`.

Convenience function equivalent to `forward x reluVec`. Applies ReLU element-wise to the
pre-activation output.

**Mathematical specification:** `forwardReLU(x) = max(0, Wx + b)` element-wise

**Parameters:**
- `layer : DenseLayer n m`: Dense layer
- `x : Vector n`: Input vector

**Returns:** `Vector m`: ReLU-activated output

**References:** See `forward` for detailed documentation -/
@[inline]
def DenseLayer.forwardReLU {m n : Nat} (layer : DenseLayer n m) (x : Vector n) : Vector m :=
  layer.forward x reluVec

/-- Batched linear transformation: computes `WX + b` for each sample in the batch.

Processes multiple samples simultaneously using vectorized operations, which is significantly
more efficient than processing samples individually.

**Mathematical specification:**
For batch `X : ℝ^(b×n)` with `b` samples, computes `WX + b` where bias is broadcast:
```
forwardBatchLinear(X)[i] = W @ X[i] + b  for i ∈ [0, b)
```

**Parameters:**
- `layer : DenseLayer n m`: Dense layer
- `X : Batch b n`: Batch of `b` input vectors, each of dimension `n`

**Returns:** `Batch b m`: Batch of `b` output vectors, each of dimension `m`

**Type safety:** Batch size `b` is preserved through the transformation.

**Performance:** Leverages SciLean's `batchMatvec` for efficient batch processing.

**References:** Single-sample version: `forwardLinear` -/
@[inline]
def DenseLayer.forwardBatchLinear {b m n : Nat}
    (layer : DenseLayer n m)
    (X : Batch b n) : Batch b m :=
  let wx := batchMatvec layer.weights X
  batchAddVec wx layer.bias

/-- Batched forward pass with optional activation.

Applies the layer transformation to a batch of inputs, with optional activation function.

**Mathematical specification:** `forwardBatch(X, σ)[i] = σ(W @ X[i] + b)` for each sample `i`

**Parameters:**
- `layer : DenseLayer n m`: Dense layer
- `X : Batch b n`: Batch of input vectors
- `activation : Batch b m → Batch b m`: Activation function (default: identity)

**Returns:** `Batch b m`: Activated batch output

**Usage:** This is the standard operation for training with mini-batches.

**References:** Single-sample version: `forward` -/
@[inline]
def DenseLayer.forwardBatch {b m n : Nat}
    (layer : DenseLayer n m)
    (X : Batch b n)
    (activation : Batch b m → Batch b m := id) : Batch b m :=
  let linear := layer.forwardBatchLinear X
  activation linear

/-- Batched forward pass with ReLU activation.

Convenience function for batched ReLU activation, equivalent to `forwardBatch X reluBatch`.

**Mathematical specification:** `forwardBatchReLU(X)[i] = max(0, W @ X[i] + b)` element-wise

**Parameters:**
- `layer : DenseLayer n m`: Dense layer
- `X : Batch b n`: Batch of input vectors

**Returns:** `Batch b m`: ReLU-activated batch output

**Usage:** Standard for training hidden layers with ReLU activation. -/
@[inline]
def DenseLayer.forwardBatchReLU {b m n : Nat}
    (layer : DenseLayer n m)
    (X : Batch b n) : Batch b m :=
  layer.forwardBatch X reluBatch

/-! ## Examples -/

/-- Example dense layer for MNIST hidden layer (784 → 128).

In practice, use `Network.Initialization` for proper weight initialization. -/
example : DenseLayer 784 128 := {
  weights := ⊞ (_ : Idx 128) (_ : Idx 784) => 0.01  -- Placeholder initialization
  bias := ⊞ (_ : Idx 128) => 0.0
}

/-- Type system guarantees dimension compatibility through layers. -/
example (layer : DenseLayer 784 128) (input : Vector 784) :
  ∃ (output : Vector 128), output = layer.forwardReLU input := by
  exists layer.forwardReLU input

end VerifiedNN.Layer
