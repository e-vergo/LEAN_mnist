import VerifiedNN.Layer.Dense
import VerifiedNN.Core.Activation
import SciLean
import Mathlib.Analysis.Calculus.FDeriv.Basic

/-!
# Layer Composition

Utilities for composing dense layers to build multi-layer neural networks.

## Main Definitions

* `stack`: Compose two layers sequentially with optional activations
* `stackLinear`: Compose two layers without activations
* `stackReLU`: Compose two layers with ReLU activations
* `stackBatch`: Batched composition for efficient training
* `stack3`: Three-layer composition

## Implementation Notes

Type-safe composition prevents dimension mismatches at compile time. The intermediate
dimension must match between layers for composition to type-check.

Batched operations provide better throughput than per-sample processing.

## Verification Status

- Type safety: Dimension compatibility enforced by type system ✓
- Composition correctness: Proven by construction ✓
- Differentiability: Chain rule applies (proof planned)
-/

namespace VerifiedNN.Layer

open VerifiedNN.Core
open VerifiedNN.Core.Activation
open SciLean

/-- Compose two dense layers sequentially with optional activations.

Applies `layer1` followed by `layer2`. The intermediate dimension `d2` must match between layers.

Type safety: If this function type-checks, dimension compatibility is guaranteed. -/
@[inline]
def stack {d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (x : Vector d1)
    (activation1 : Vector d2 → Vector d2 := id)
    (activation2 : Vector d3 → Vector d3 := id) : Vector d3 :=
  let h := layer1.forward x activation1
  layer2.forward h activation2

/-- Compose two layers without activations.

Pure affine composition: `layer2(layer1(x))` with no activation functions.
Useful for analyzing linear transformation properties. -/
@[inline]
def stackLinear {d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (x : Vector d1) : Vector d3 :=
  let h := layer1.forwardLinear x
  layer2.forwardLinear h

/-- Compose two layers with ReLU activations.

Convenience function: `ReLU(W2 @ ReLU(W1 @ x + b1) + b2)`. -/
@[inline]
def stackReLU {d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (x : Vector d1) : Vector d3 :=
  stack layer1 layer2 x reluVec reluVec

/-- Batched composition of two dense layers.

Applies layer composition to a batch of inputs efficiently. -/
@[inline]
def stackBatch {b d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (X : Batch b d1)
    (activation1 : Batch b d2 → Batch b d2 := id)
    (activation2 : Batch b d3 → Batch b d3 := id) : Batch b d3 :=
  let H := layer1.forwardBatch X activation1
  layer2.forwardBatch H activation2

/-- Batched composition with ReLU activations. -/
@[inline]
def stackBatchReLU {b d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (X : Batch b d1) : Batch b d3 :=
  stackBatch layer1 layer2 X reluBatch reluBatch

/-- Compose three layers sequentially.

Extends `stack` to three layers for building deeper networks. -/
@[inline]
def stack3 {d1 d2 d3 d4 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (layer3 : DenseLayer d3 d4)
    (x : Vector d1)
    (act1 : Vector d2 → Vector d2 := id)
    (act2 : Vector d3 → Vector d3 := id)
    (act3 : Vector d4 → Vector d4 := id) : Vector d4 :=
  let h1 := layer1.forward x act1
  let h2 := layer2.forward h1 act2
  layer3.forward h2 act3

/-! ## Examples -/

/-- Two-layer network composition for MNIST (784 → 128 → 10). -/
example (layer1 : DenseLayer 784 128) (layer2 : DenseLayer 128 10)
    (input : Vector 784) :
  ∃ (output : Vector 10), output = stackReLU layer1 layer2 input := by
  exists stackReLU layer1 layer2 input

/-- Type-safe composition ensures dimension compatibility. -/
example (layer1 : DenseLayer 784 128) (layer2 : DenseLayer 128 10) :
  ∀ (x : Vector 784), ∃ (y : Vector 10), y = stack layer1 layer2 x id id := by
  intro x
  exists stack layer1 layer2 x id id

/-- Batched composition maintains batch size and output dimension. -/
example (layer1 : DenseLayer 784 128) (layer2 : DenseLayer 128 10)
    (batch : Batch 32 784) :
  ∃ (output : Batch 32 10), output = stackBatchReLU layer1 layer2 batch := by
  exists stackBatchReLU layer1 layer2 batch

end VerifiedNN.Layer
