/-
# Layer Composition

Utilities for composing layers to build neural networks.

This module provides functions for sequentially composing dense layers to create
multi-layer networks. The composition utilities leverage dependent types to ensure
dimension compatibility at compile time.

**Verification Status:**
- Type safety: Dimension compatibility enforced by type system
- Composition correctness: Proven by construction
- Differentiability: Chain rule applies (proof planned)

**Design Philosophy:**
- Type-safe composition prevents dimension mismatches
- Sequential composition builds networks layer-by-layer
- Supports both single-sample and batched operations
-/

import VerifiedNN.Layer.Dense
import VerifiedNN.Core.Activation
import SciLean

namespace VerifiedNN.Layer

open VerifiedNN.Core
open VerifiedNN.Core.Activation
open SciLean

/-- Compose two dense layers sequentially.

Applies layer1 followed by layer2, automatically handling dimension compatibility
through the type system. The intermediate dimension d2 must match between layers.

**Type Parameters:**
- `d1`: Input dimension
- `d2`: Intermediate dimension (output of layer1, input of layer2)
- `d3`: Output dimension

**Parameters:**
- `layer1`: First layer transforming d1 → d2
- `layer2`: Second layer transforming d2 → d3
- `x`: Input vector of dimension d1

**Returns:** Output vector of dimension d3

**Type Safety:** If this function type-checks, dimension compatibility is guaranteed.

**Mathematical Formulation:**
```
output = layer2(layer1(x))
       = layer2(activation1(W1 @ x + b1))
       = activation2(W2 @ activation1(W1 @ x + b1) + b2)
```

**Example:**
```lean
let layer1 : DenseLayer 784 128 := ...
let layer2 : DenseLayer 128 10 := ...
let input : Vector 784 := ...
let output : Vector 10 := stack layer1 layer2 input
```
-/
def stack {d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (x : Vector d1)
    (activation1 : Vector d2 → Vector d2 := id)
    (activation2 : Vector d3 → Vector d3 := id) : Vector d3 :=
  let h := layer1.forward x activation1
  layer2.forward h activation2

/-- Compose two layers without activation functions.

Pure linear composition: layer2(layer1(x)) without any activations.
Useful for analyzing the linear transformation properties.

**Parameters:**
- `layer1`: First layer transforming d1 → d2
- `layer2`: Second layer transforming d2 → d3
- `x`: Input vector of dimension d1

**Returns:** Output vector of dimension d3
-/
def stackLinear {d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (x : Vector d1) : Vector d3 :=
  let h := layer1.forwardLinear x
  layer2.forwardLinear h

/-- Compose two layers with ReLU activations.

Convenience function for the common pattern of stacking layers with ReLU.

**Parameters:**
- `layer1`: First layer transforming d1 → d2
- `layer2`: Second layer transforming d2 → d3
- `x`: Input vector of dimension d1

**Returns:** Output vector of dimension d3

**Computation:** ReLU(W2 @ ReLU(W1 @ x + b1) + b2)
-/
def stackReLU {d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (x : Vector d1) : Vector d3 :=
  stack layer1 layer2 x reluVec reluVec

/-- Batched composition of two dense layers.

Applies layer composition to a batch of inputs efficiently.

**Parameters:**
- `layer1`: First layer transforming d1 → d2
- `layer2`: Second layer transforming d2 → d3
- `X`: Batch of b input vectors, each of dimension d1
- `activation1`: Activation after first layer (default: identity)
- `activation2`: Activation after second layer (default: identity)

**Returns:** Batch of b output vectors, each of dimension d3

**Performance:** More efficient than processing samples individually.
-/
def stackBatch {b d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (X : Batch b d1)
    (activation1 : Batch b d2 → Batch b d2 := id)
    (activation2 : Batch b d3 → Batch b d3 := id) : Batch b d3 :=
  let H := layer1.forwardBatch X activation1
  layer2.forwardBatch H activation2

/-- Batched composition with ReLU activations.

Applies two layers with ReLU activations to a batch of inputs.

**Parameters:**
- `layer1`: First layer transforming d1 → d2
- `layer2`: Second layer transforming d2 → d3
- `X`: Batch of b input vectors, each of dimension d1

**Returns:** Batch of b output vectors, each of dimension d3
-/
def stackBatchReLU {b d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (X : Batch b d1) : Batch b d3 :=
  stackBatch layer1 layer2 X reluBatch reluBatch

/-- Compose three layers sequentially.

Extends stack to three layers for building deeper networks.

**Parameters:**
- `layer1`: First layer transforming d1 → d2
- `layer2`: Second layer transforming d2 → d3
- `layer3`: Third layer transforming d3 → d4
- `x`: Input vector of dimension d1
- `act1`, `act2`, `act3`: Activation functions for each layer

**Returns:** Output vector of dimension d4
-/
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

/-- Infix operator for layer composition.

Allows writing: layer1 |> layer2 |> layer3

Note: This is experimental syntax sugar. Type inference may require
explicit type annotations in complex scenarios.
-/
def compose {d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3) : Vector d1 → Vector d3 :=
  fun x => stack layer1 layer2 x

-- Differentiability proofs for composition
-- These establish that composition preserves differentiability via chain rule

-- @[fun_prop]
-- theorem stack_differentiable {d1 d2 d3 : Nat}
--     (layer1 : DenseLayer d1 d2)
--     (layer2 : DenseLayer d2 d3)
--     (hf : Differentiable Float (layer1.forward · act1))
--     (hg : Differentiable Float (layer2.forward · act2)) :
--   Differentiable Float (fun x => stack layer1 layer2 x act1 act2) := by
--   sorry

-- @[fun_trans]
-- theorem stack_fderiv {d1 d2 d3 : Nat}
--     (layer1 : DenseLayer d1 d2)
--     (layer2 : DenseLayer d2 d3) :
--   fderiv Float (fun x => stack layer1 layer2 x act1 act2) =
--     fun x dx => fderiv Float (layer2.forward · act2) (layer1.forward x act1)
--                   (fderiv Float (layer1.forward · act1) x dx) := by
--   sorry -- Chain rule

end VerifiedNN.Layer
