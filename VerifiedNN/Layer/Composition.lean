import Mathlib.Analysis.Calculus.FDeriv.Basic
import SciLean
import VerifiedNN.Core.Activation
import VerifiedNN.Layer.Dense

/-!
# Layer Composition

Utilities for composing dense layers to build multi-layer neural networks with compile-time
dimension safety guarantees.

## Main Definitions

* `stack`: Compose two layers sequentially with optional activations
* `stackLinear`: Compose two layers without activations (pure affine composition)
* `stackReLU`: Compose two layers with ReLU activations
* `stackBatch`: Batched composition with optional activations
* `stackBatchReLU`: Batched composition with ReLU activations
* `stack3`: Three-layer composition with optional activations

## Main Results

**Type-level dimension safety:**
* Intermediate dimensions must match for composition to type-check
* If `layer1 : DenseLayer d1 d2` and `layer2 : DenseLayer d2 d3`, then `stack` produces `Vector d3`
* Dimension mismatches are caught at compile time, not runtime

**Verified properties (proven in VerifiedNN.Layer.Properties):**
* Composition of affine maps preserves affine combinations
* Type system enforces dimension compatibility through all compositions

## Implementation Notes

All composition functions enforce dimension compatibility through dependent types.
The intermediate dimension parameter (e.g., `d2` in `DenseLayer d1 d2` → `DenseLayer d2 d3`)
must match exactly for the code to type-check.

Batched operations (`stackBatch`, `stackBatchReLU`) process multiple samples simultaneously
for better throughput during training. These are essential for efficient mini-batch gradient
descent.

The `stackLinear` variant composes layers without activation functions, creating a pure
affine transformation. This is useful for analyzing mathematical properties and proving
composition theorems.

## Verification Status

- **Build status:** ✅ Compiles with zero errors
- **Sorries:** 0
- **Axioms:** 0
- **Type safety:** ✅ Dimension compatibility enforced by type system (compile-time checking)
- **Composition correctness:** ✅ Proven by construction (definitions are correct by design)
- **Affine preservation:** ✅ Proven in
  VerifiedNN.Layer.Properties.stackLinear_preserves_affine_combination
- **Differentiability:** Planned (chain rule for composition of differentiable functions)
- **Gradient correctness:** Planned (see VerifiedNN/Verification/GradientCorrectness.lean)

## References

- Dense layer definitions: VerifiedNN.Layer.Dense
- Mathematical properties: VerifiedNN.Layer.Properties
- Full network architecture: VerifiedNN.Network.Architecture
-/

namespace VerifiedNN.Layer

open VerifiedNN.Core
open VerifiedNN.Core.Activation
open SciLean

/-- Compose two dense layers sequentially with optional activations.

Applies `layer1` followed by `layer2`, creating a two-layer neural network.
The intermediate dimension `d2` must match between layers for type-checking.

**Mathematical specification:**
```
stack(x, σ₁, σ₂) = σ₂(W₂ @ σ₁(W₁ @ x + b₁) + b₂)
```
where `layer1 = (W₁, b₁)`, `layer2 = (W₂, b₂)`, and `σ₁, σ₂` are activations.

**Parameters:**
- `layer1 : DenseLayer d1 d2`: First layer
- `layer2 : DenseLayer d2 d3`: Second layer (input dimension must match `layer1`'s output)
- `x : Vector d1`: Input vector
- `activation1 : Vector d2 → Vector d2`: Activation after first layer (default: identity)
- `activation2 : Vector d3 → Vector d3`: Activation after second layer (default: identity)

**Returns:** `Vector d3`: Output of two-layer composition

**Type safety:** If this function type-checks, dimension compatibility is guaranteed at
compile time.

**Verified properties:**
- Output dimension equals `d3` (type-level guarantee)
- Preserves affine combinations when activations are identity (proven in Properties.lean)

**References:** `stackLinear`, `stackReLU`, `stackBatch` -/
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

Pure affine composition: applies two affine transformations sequentially without
any activation functions. This is useful for analyzing mathematical properties.

**Mathematical specification:**
```
stackLinear(x) = W₂ @ (W₁ @ x + b₁) + b₂
               = (W₂ @ W₁) @ x + (W₂ @ b₁ + b₂)
```
This is equivalent to a single affine transformation with combined weights and biases.

**Parameters:**
- `layer1 : DenseLayer d1 d2`: First layer
- `layer2 : DenseLayer d2 d3`: Second layer
- `x : Vector d1`: Input vector

**Returns:** `Vector d3`: Composed affine transformation output

**Verified properties:**
- Preserves affine combinations (proven: `stackLinear_preserves_affine_combination`)
- Composition of affine maps is affine

**Usage:** Primarily for mathematical analysis and proofs, not for training.

**References:** Mathematical properties in VerifiedNN.Layer.Properties -/
@[inline]
def stackLinear {d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (x : Vector d1) : Vector d3 :=
  let h := layer1.forwardLinear x
  layer2.forwardLinear h

/-- Compose two layers with ReLU activations.

Convenience function for the common pattern of ReLU activation after each layer.
Equivalent to `stack layer1 layer2 x reluVec reluVec`.

**Mathematical specification:**
```
stackReLU(x) = max(0, W₂ @ max(0, W₁ @ x + b₁) + b₂)
```
where `max(0, ·)` is applied element-wise.

**Parameters:**
- `layer1 : DenseLayer d1 d2`: First layer
- `layer2 : DenseLayer d2 d3`: Second layer
- `x : Vector d1`: Input vector

**Returns:** `Vector d3`: ReLU-activated output

**Usage:** Standard for hidden layers in feedforward networks.

**References:** `stack`, `reluVec` -/
@[inline]
def stackReLU {d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (x : Vector d1) : Vector d3 :=
  stack layer1 layer2 x reluVec reluVec

/-- Batched composition of two dense layers.

Applies layer composition to a batch of inputs efficiently using vectorized operations.

**Mathematical specification:**
For batch `X : ℝ^(b×d1)`, computes for each sample `i ∈ [0, b)`:
```
stackBatch(X, σ₁, σ₂)[i] = σ₂(W₂ @ σ₁(W₁ @ X[i] + b₁) + b₂)
```

**Parameters:**
- `layer1 : DenseLayer d1 d2`: First layer
- `layer2 : DenseLayer d2 d3`: Second layer
- `X : Batch b d1`: Batch of `b` input vectors
- `activation1 : Batch b d2 → Batch b d2`: Activation after first layer (default: identity)
- `activation2 : Batch b d3 → Batch b d3`: Activation after second layer (default: identity)

**Returns:** `Batch b d3`: Batch of outputs

**Type safety:** Batch size `b` is preserved through composition.

**Performance:** Significantly faster than processing samples individually.

**References:** Single-sample version: `stack` -/
@[inline]
def stackBatch {b d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (X : Batch b d1)
    (activation1 : Batch b d2 → Batch b d2 := id)
    (activation2 : Batch b d3 → Batch b d3 := id) : Batch b d3 :=
  let H := layer1.forwardBatch X activation1
  layer2.forwardBatch H activation2

/-- Batched composition with ReLU activations.

Convenience function for batched two-layer composition with ReLU activations.
Equivalent to `stackBatch layer1 layer2 X reluBatch reluBatch`.

**Parameters:**
- `layer1 : DenseLayer d1 d2`: First layer
- `layer2 : DenseLayer d2 d3`: Second layer
- `X : Batch b d1`: Batch of input vectors

**Returns:** `Batch b d3`: ReLU-activated batch output

**Usage:** Standard for mini-batch training with ReLU hidden layers. -/
@[inline]
def stackBatchReLU {b d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (X : Batch b d1) : Batch b d3 :=
  stackBatch layer1 layer2 X reluBatch reluBatch

/-- Compose three layers sequentially.

Extends `stack` to three layers for building deeper neural networks. Type system
enforces that intermediate dimensions match.

**Mathematical specification:**
```
stack3(x, σ₁, σ₂, σ₃) = σ₃(W₃ @ σ₂(W₂ @ σ₁(W₁ @ x + b₁) + b₂) + b₃)
```

**Parameters:**
- `layer1 : DenseLayer d1 d2`: First layer
- `layer2 : DenseLayer d2 d3`: Second layer
- `layer3 : DenseLayer d3 d4`: Third layer
- `x : Vector d1`: Input vector
- `act1, act2, act3`: Activation functions (default: identity)

**Returns:** `Vector d4`: Output of three-layer composition

**Type safety:** All intermediate dimensions (`d2`, `d3`) must match for type-checking.

**Usage:** For three-layer networks; extend pattern for deeper networks if needed.

**References:** `stack` for two-layer composition -/
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
