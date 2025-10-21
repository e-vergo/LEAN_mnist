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

* `DenseLayer n m`: A dense layer transforming `n`-dimensional inputs to `m`-dimensional outputs
* `DenseLayer.forward`: Forward pass with optional activation function
* `DenseLayer.forwardLinear`: Linear transformation without activation
* `DenseLayer.forwardBatch`: Batched forward pass for efficient training

## Implementation Notes

Forward pass computes `activation(Wx + b)` where `W` is the weight matrix and `b` is the bias.
Batched operations process multiple samples efficiently using SciLean's `DataArrayN`.

## Verification Status

- Forward pass implementation: Working (uses SciLean primitives) ✓
- Type safety: Enforced via dependent types ✓
- Differentiability: Planned (requires Core.LinearAlgebra completion)
-/

namespace VerifiedNN.Layer

open VerifiedNN.Core
open VerifiedNN.Core.LinearAlgebra
open VerifiedNN.Core.Activation
open SciLean

/-- A dense (fully-connected) layer with learnable weights and biases.

Transforms input vectors of dimension `inDim` to output vectors of dimension `outDim`
via `activation(W @ input + b)`.

**Fields:**
- `weights`: Weight matrix of shape `(outDim, inDim)`
- `bias`: Bias vector of shape `(outDim,)` -/
structure DenseLayer (inDim outDim : Nat) where
  weights : Matrix outDim inDim
  bias : Vector outDim

/-- Linear transformation without activation: computes `Wx + b`.

This is the pre-activation output. Apply an activation function (e.g., ReLU) to get
the final layer output. -/
@[inline]
def DenseLayer.forwardLinear {m n : Nat} (layer : DenseLayer n m) (x : Vector n) : Vector m :=
  let wx := matvec layer.weights x
  vadd wx layer.bias

/-- Forward pass with optional activation: computes `activation(Wx + b)`.

The type system ensures dimension compatibility. -/
@[inline]
def DenseLayer.forward {m n : Nat}
    (layer : DenseLayer n m)
    (x : Vector n)
    (activation : Vector m → Vector m := id) : Vector m :=
  let linear := layer.forwardLinear x
  activation linear

/-- Forward pass with ReLU activation: computes `ReLU(Wx + b)`. -/
@[inline]
def DenseLayer.forwardReLU {m n : Nat} (layer : DenseLayer n m) (x : Vector n) : Vector m :=
  layer.forward x reluVec

/-- Batched linear transformation: computes `WX + b` for each sample in the batch.

More efficient than processing samples individually due to vectorized operations. -/
@[inline]
def DenseLayer.forwardBatchLinear {b m n : Nat}
    (layer : DenseLayer n m)
    (X : Batch b n) : Batch b m :=
  let wx := batchMatvec layer.weights X
  batchAddVec wx layer.bias

/-- Batched forward pass with optional activation.

Applies the layer transformation to a batch of inputs. -/
@[inline]
def DenseLayer.forwardBatch {b m n : Nat}
    (layer : DenseLayer n m)
    (X : Batch b n)
    (activation : Batch b m → Batch b m := id) : Batch b m :=
  let linear := layer.forwardBatchLinear X
  activation linear

/-- Batched forward pass with ReLU activation. -/
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
