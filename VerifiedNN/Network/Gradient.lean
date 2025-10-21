/-
# Network Gradient Computation

Gradient computation using SciLean's automatic differentiation.

This module provides utilities for:
1. Flattening network parameters into a single vector for optimization
2. Unflattening parameter vectors back to network structure
3. Computing gradients with respect to all network parameters

## Memory Layout

Parameters are flattened in this order:
- **Indices 0..100351** (100,352 elements): Layer 1 weights (784 × 128)
- **Indices 100352..100479** (128 elements): Layer 1 bias
- **Indices 100480..101759** (1,280 elements): Layer 2 weights (128 × 10)
- **Indices 101760..101769** (10 elements): Layer 2 bias
- **Total: 101,770 parameters**

## Verification Status

**8 sorries remaining** - all related to index arithmetic proofs:
- 1 sorry in `flattenParams`: Proving final bias index < nParams
- 4 sorries in `unflattenParams`: Proving computed indices < nParams for each component
- 2 sorries in round-trip theorems: Structural extensionality and index arithmetic
- 2 sorries in batch functions: Proving loop indices < batch size

All sorries are mathematically trivial but require detailed omega-style arithmetic
reasoning that interacts poorly with Lean's type conversion (USize ↔ Nat ↔ Idx).
-/

import VerifiedNN.Network.Architecture
import VerifiedNN.Loss.CrossEntropy
import SciLean

namespace VerifiedNN.Network.Gradient

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Loss
open SciLean

set_default_scalar Float

/-- Helper to convert Nat with bound proof to Idx.
    Uses Idx.finEquiv internally to avoid USize conversion proofs. -/
private def natToIdx (n : Nat) (i : Nat) (h : i < n) : Idx n :=
  (Idx.finEquiv n).invFun ⟨i, h⟩

/-- Technical lemma: Idx bounds convert correctly to Nat bounds.

    If i : Idx n, then i.toNat < n by construction.
    Proven using Idx.finEquiv which establishes Idx n ≃ Fin n. -/
private theorem idx_toNat_lt {n : Nat} (i : Idx n) : i.1.toNat < n := by
  -- Use the equivalence Idx n ≃ Fin n
  let fin_i := (Idx.finEquiv n) i
  have h : fin_i.val < n := fin_i.isLt
  -- i.1 is the USize component, convert via the equivalence
  exact h

/-- Total number of parameters in the network.

Breakdown:
- Layer 1 weights: 784 × 128 = 100,352
- Layer 1 bias: 128
- Layer 2 weights: 128 × 10 = 1,280
- Layer 2 bias: 10
- Total: 101,770 parameters
-/
def nParams : Nat := 784 * 128 + 128 + 128 * 10 + 10

/-- Compile-time verification that nParams is correct -/
theorem nParams_value : nParams = 101770 := by rfl

/-- Flatten network parameters into a single vector.

Concatenates all parameters in order:
1. Layer 1 weights (flattened row-major)
2. Layer 1 bias
3. Layer 2 weights (flattened row-major)
4. Layer 2 bias

**Parameters:**
- `net`: The MLP network structure

**Returns:** Flattened parameter vector of dimension nParams
-/
def flattenParams (net : MLPArchitecture) : Vector nParams :=
  ⊞ (i : Idx nParams) =>
    let idx := i.1.toNat
    -- Layer 1 weights: indices 0..100351 (784 * 128 = 100352 elements)
    if h : idx < 784 * 128 then
      let row := idx / 784
      let col := idx % 784
      have hrow : row < 128 := by omega
      have hcol : col < 784 := Nat.mod_lt idx (by omega : 0 < 784)
      net.layer1.weights[natToIdx 128 row hrow, natToIdx 784 col hcol]
    -- Layer 1 bias: indices 100352..100479 (128 elements)
    else if h2 : idx < 784 * 128 + 128 then
      let bidx := idx - 784 * 128
      have hb : bidx < 128 := by omega
      net.layer1.bias[natToIdx 128 bidx hb]
    -- Layer 2 weights: indices 100480..101759 (128 * 10 = 1280 elements)
    else if h3 : idx < 784 * 128 + 128 + 128 * 10 then
      let offset := idx - (784 * 128 + 128)
      let row := offset / 128
      let col := offset % 128
      have hrow : row < 10 := by omega
      have hcol : col < 128 := Nat.mod_lt offset (by omega : 0 < 128)
      net.layer2.weights[natToIdx 10 row hrow, natToIdx 128 col hcol]
    -- Layer 2 bias: indices 101760..101769 (10 elements)
    else
      let bidx := idx - (784 * 128 + 128 + 128 * 10)
      have hb : bidx < 10 := by
        -- In the else branch, we know: ¬(idx < 784*128), ¬(idx < 784*128+128), ¬(idx < 784*128+128+128*10)
        -- And we have idx < nParams (from i : Idx nParams)
        -- Therefore bidx < 10
        have hidx : idx < nParams := idx_toNat_lt i
        unfold nParams at hidx
        omega
      net.layer2.bias[natToIdx 10 bidx hb]

/-- Unflatten parameter vector back to network structure.

Reconstructs the network from a flattened parameter vector.
This is the inverse operation of flattenParams.

**Parameters:**
- `params`: Flattened parameter vector of dimension nParams

**Returns:** MLPArchitecture with parameters extracted from the vector
-/
def unflattenParams (params : Vector nParams) : MLPArchitecture :=
  let w1 : Matrix 128 784 := ⊞ ((i, j) : Idx 128 × Idx 784) =>
    let idx := i.1.toNat * 784 + j.1.toNat
    have h : idx < nParams := by
      have hi : i.1.toNat < 128 := idx_toNat_lt i
      have hj : j.1.toNat < 784 := idx_toNat_lt j
      unfold nParams
      omega
    params[natToIdx nParams idx h]
  let b1 : Vector 128 := ⊞ (i : Idx 128) =>
    let idx := 784 * 128 + i.1.toNat
    have h : idx < nParams := by
      have hi : i.1.toNat < 128 := idx_toNat_lt i
      unfold nParams
      omega
    params[natToIdx nParams idx h]
  let w2 : Matrix 10 128 := ⊞ ((i, j) : Idx 10 × Idx 128) =>
    let idx := 784 * 128 + 128 + i.1.toNat * 128 + j.1.toNat
    have h : idx < nParams := by
      have hi : i.1.toNat < 10 := idx_toNat_lt i
      have hj : j.1.toNat < 128 := idx_toNat_lt j
      unfold nParams
      omega
    params[natToIdx nParams idx h]
  let b2 : Vector 10 := ⊞ (i : Idx 10) =>
    let idx := 784 * 128 + 128 + 128 * 10 + i.1.toNat
    have h : idx < nParams := by
      have hi : i.1.toNat < 10 := idx_toNat_lt i
      unfold nParams
      omega
    params[natToIdx nParams idx h]
  { layer1 := { weights := w1, bias := b1 }
    layer2 := { weights := w2, bias := b2 } }

/-- Theorem: Flattening then unflattening is identity.

This is a critical property for gradient descent to work correctly.

This theorem requires detailed DataArrayN extensionality and index arithmetic,
which depends on SciLean's internal Idx/USize representation. Axiomatized as
technically correct but requiring extensive boilerplate.

TODO: Prove using structural extensionality, DataArrayN extensionality, and index arithmetic.
-/
theorem unflatten_flatten_id (net : MLPArchitecture) :
    unflattenParams (flattenParams net) = net := by
  sorry
  -- Proof obligation: Show that round-trip preserves network structure
  -- Strategy:
  --   1. Apply MLPArchitecture extensionality (layer1 = layer1, layer2 = layer2)
  --   2. Apply DenseLayer extensionality (weights = weights, bias = bias)
  --   3. Apply DataArrayN extensionality (pointwise equality at all indices)
  --   4. For each index (i,j), show that:
  --      unflattenParams(flattenParams(net))[i,j] = net[i,j]
  --      by unfolding definitions and simplifying index arithmetic
  -- Blocked by: Requires custom DataArrayN extensionality lemma and tedious case analysis
  -- Note: Mathematically obvious by construction, but Lean needs explicit proof

/-- Theorem: Unflattening then flattening is identity.

Another critical property for optimization correctness.
-/
theorem flatten_unflatten_id (params : Vector nParams) :
    flattenParams (unflattenParams params) = params := by
  sorry
  -- Proof obligation: Show that round-trip preserves parameter vector
  -- Strategy:
  --   1. Apply DataArrayN extensionality (prove elementwise equality)
  --   2. For each index k : Idx nParams, show:
  --      flattenParams(unflattenParams(params))[k] = params[k]
  --   3. Case split on k's range (which layer/component it belongs to)
  --   4. In each case, unfold definitions and verify index arithmetic cancels
  -- Blocked by: Requires explicit case analysis on 4 ranges and index arithmetic
  -- Note: Dual of unflatten_flatten_id, same level of detail required

/-- Helper function to compute loss for a single sample.

Used for numerical gradient checking and testing.

**Parameters:**
- `params`: Flattened network parameters
- `input`: Input vector of dimension 784
- `target`: Target class (0-9)

**Returns:** Scalar loss value
-/
def computeLoss (params : Vector nParams)
    (input : Vector 784) (target : Nat) : Float :=
  let net := unflattenParams params
  let output := net.forward input
  crossEntropyLoss output target

/-- Compute gradient of loss with respect to network parameters.

Uses SciLean's automatic differentiation to compute the gradient.
This is the core of backpropagation in the network.

**Parameters:**
- `params`: Flattened network parameters
- `input`: Input vector of dimension 784
- `target`: Target class (0-9 for MNIST)

**Returns:** Gradient vector of dimension nParams

**Verification Status:** This function will be proven to compute the
mathematically correct gradient in VerifiedNN.Verification.GradientCorrectness
-/
noncomputable def networkGradient (params : Vector nParams)
    (input : Vector 784) (target : Nat) : Vector nParams :=
  -- Use SciLean's automatic differentiation to compute gradient
  let lossFunc := fun p => computeLoss p input target
  (∇ p, lossFunc p) params

/-- Compute gradient for a mini-batch of samples.

Computes the average gradient over a batch of training examples.
This is more efficient than computing individual gradients.

**Parameters:**
- `params`: Flattened network parameters
- `inputs`: Batch of b input vectors, each of dimension 784
- `targets`: Array of b target classes

**Returns:** Average gradient vector of dimension nParams
-/
noncomputable def networkGradientBatch {b : Nat} (params : Vector nParams)
    (inputs : Batch b 784) (targets : Array Nat) : Vector nParams :=
  -- Compute gradient for each sample and average them
  Id.run do
    let mut gradSum : Vector nParams := ⊞ (_ : Idx nParams) => (0.0 : Float)
    for i in Array.range b do
      -- i is in range [0, b) from Array.range membership
      have hi : i < b := by
        sorry
        -- Proof obligation: Show that i < b for i ∈ Array.range b
        -- Given: Array.range b produces array [0, 1, ..., b-1]
        -- Hence: For all i in the array, i < b holds
        -- Blocked by: Requires lemma about Array.range membership
        -- Note: Should be provable with: Array.mem_range_iff_mem_finRange
      if h : i < targets.size then
        -- Extract input sample from batch - convert Nat to Idx
        let idxI : Idx b := (Idx.finEquiv b).invFun ⟨i, hi⟩
        let inputSample : Vector 784 := ⊞ j => inputs[idxI, j]
        let target := targets[i]
        let grad := networkGradient params inputSample target
        -- Accumulate gradient
        gradSum := ⊞ k => gradSum[k] + grad[k]
    -- Average the accumulated gradients
    let bFloat := b.toFloat
    return ⊞ k => gradSum[k] / bFloat

/-- Helper function to compute batched loss.

**Parameters:**
- `params`: Flattened network parameters
- `inputs`: Batch of b input vectors
- `targets`: Array of b target classes

**Returns:** Average loss over the batch
-/
def computeLossBatch {b : Nat} (params : Vector nParams)
    (inputs : Batch b 784) (targets : Array Nat) : Float :=
  -- Compute loss for each sample and average
  Id.run do
    let mut lossSum := 0.0
    for i in Array.range b do
      -- i is in range [0, b) from Array.range membership
      have hi : i < b := by
        sorry
        -- Proof obligation: Show that i < b for i ∈ Array.range b
        -- Given: Array.range b produces array [0, 1, ..., b-1]
        -- Hence: For all i in the array, i < b holds
        -- Blocked by: Requires lemma about Array.range membership (same as in networkGradientBatch)
        -- Note: Should be provable with: Array.mem_range_iff_mem_finRange
      if h : i < targets.size then
        -- Extract input sample from batch - convert Nat to Idx
        let idxI : Idx b := (Idx.finEquiv b).invFun ⟨i, hi⟩
        let inputSample : Vector 784 := ⊞ j => inputs[idxI, j]
        let target := targets[i]
        let loss := computeLoss params inputSample target
        lossSum := lossSum + loss
    return lossSum / b.toFloat

end VerifiedNN.Network.Gradient
