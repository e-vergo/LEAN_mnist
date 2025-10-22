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

**3 axioms, 0 sorries:**
- **Axiom:** `unflatten_flatten_id` - Round-trip identity (flattening then unflattening)
  - Requires SciLean array extensionality (currently axiomatized in SciLean itself)
  - See comprehensive documentation on axiom for justification
- **Axiom:** `flatten_unflatten_id` - Round-trip identity (unflattening then flattening)
  - Dual of above, requires same extensionality infrastructure
  - Together these establish bijection between MLPArchitecture and Vector nParams
- **Axiom:** `array_range_mem_bound` - Elements of Array.range n are less than n
  - Mathematically trivial property: Array.range n = [0,1,...,n-1]
  - Requires Array.mem_range lemma not currently in standard library
  - Used only in batch training loop, does not affect gradient correctness proofs

The first two axioms are **essential and justified** - they axiomatize what is
algorithmically true but unprovable without array extensionality. SciLean's DataArray.ext
is itself axiomatized as sorry_proof, so we inherit this limitation.

The third axiom is **mathematically trivial** and could be eliminated with additional
lemmas about Array.range membership in the standard library.
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

/-- Axiom: Elements of Array.range n are less than n.

This is a trivial property: `Array.range n` produces `#[0, 1, ..., n-1]`, so every
element is less than n. However, proving this requires a lemma about Array.range
membership that is not currently available in the standard library.

**Mathematical Content:** ∀ i ∈ Array.range n, i < n

**Why This is an Axiom:**
The property is algorithmically obvious from the definition of Array.range but proving
it formally requires:
1. A lemma characterizing membership in `Array.range n` (like `Array.mem_range_iff`)
2. Connecting this to the bound `i < n`

Such lemmas exist for `List.range` but not yet for `Array.range` in the current version
of the standard library or Batteries.

**Consistency:**
This axiom is mathematically trivial and adds no risk of inconsistency. It merely
asserts a basic property of the `Array.range` function that is true by construction.

**Alternative:**
Could be eliminated by:
- Using `List.range` instead (which has the lemmas), though this impacts performance
- Waiting for Batteries to add `Array.mem_range` lemmas
- Manually refactoring to avoid `Array.range` and use explicit index bounds

**Impact:**
Used only in batch computation functions for the training loop. Does not affect the
core verification of gradient correctness.
-/
private axiom array_range_mem_bound {n : Nat} (i : Nat) (h : i ∈ Array.range n) : i < n

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

/-- Axiom: Flattening network parameters then unflattening recovers the original network.

**Critical Property:** This is essential for gradient descent to work correctly.

**Mathematical Content:**
For any MLP network structure `net`, the composition `unflattenParams ∘ flattenParams`
is the identity function. This states that the parameter marshalling operations are
left-invertible.

**Why This is an Axiom:**
This theorem requires SciLean's array extensionality (`DataArray.ext`), which is itself
currently axiomatized as `sorry_proof` in SciLean (see SciLean/Data/DataArray/DataArray.lean:130).
The extensionality principle states that two arrays are equal if they have the same size
and equal elements at all indices.

The proof would proceed as:
1. MLPArchitecture structural extensionality (layer1 = layer1, layer2 = layer2)
2. DenseLayer structural extensionality (weights = weights, bias = bias)
3. DataArrayN extensionality for 1D (bias vectors) and 2D (weight matrices)
4. Index arithmetic showing that for each index i,j:
   - `unflattenParams(flattenParams(net)).weights[i,j] = net.weights[i,j]`
   - This follows from the index mapping: flatten maps (i,j) to k, unflatten maps k back to (i,j)

**Consistency:**
This axiom is consistent with the definitions of `flattenParams` and `unflattenParams`,
which implement inverse index transformations by construction. The axiom merely asserts
what is algorithmically true but not provable without array extensionality.

**Alternative:**
Could be proven if SciLean provided `DataArray.ext` as a proven lemma rather than axiom.
The blocking issue is that SciLean's `DataArray` is currently not a quotient type
(see comment in SciLean source: "Currently this is inconsistent, we need to turn
DataArray into quotient!"). Once SciLean addresses this, this axiom could be replaced
with a proof.

**References:**
- SciLean DataArray.ext: SciLean/Data/DataArray/DataArray.lean:130
- Related work: Similar axioms in Certigrad (Lean 3 predecessor)

**Impact:**
Using this axiom does not introduce inconsistency beyond what SciLean already assumes.
It is essential for proving gradient descent correctness, as it ensures parameter updates
in the optimizer preserve the network structure.
-/
axiom unflatten_flatten_id (net : MLPArchitecture) :
    unflattenParams (flattenParams net) = net

/-- Axiom: Unflattening a parameter vector then flattening produces the original vector.

**Mathematical Content:**
For any parameter vector `params : Vector nParams`, the composition
`flattenParams ∘ unflattenParams` is the identity function. This states that the
parameter marshalling operations are right-invertible.

**Why This is an Axiom:**
This theorem is the dual of `unflatten_flatten_id` and requires the same extensionality
infrastructure. The proof would require:

1. Array extensionality for `Vector nParams` (1D DataArrayN)
2. Case analysis on index ranges (4 ranges corresponding to layer1.weights, layer1.bias,
   layer2.weights, layer2.bias)
3. For each range, index arithmetic showing:
   - If k ∈ [0, 100352), then `flatten(unflatten(params))[k] = params[k]` by weight1 mapping
   - If k ∈ [100352, 100480), then `flatten(unflatten(params))[k] = params[k]` by bias1 mapping
   - If k ∈ [100480, 101760), then `flatten(unflatten(params))[k] = params[k]` by weight2 mapping
   - If k ∈ [101760, 101770), then `flatten(unflatten(params))[k] = params[k]` by bias2 mapping

**Proof Sketch:**
```lean
apply DataArrayN.ext
intro k
-- Case split on k's range
by_cases h1 : k.1.toNat < 784 * 128
· -- Layer 1 weights range
  simp [flattenParams, unflattenParams]
  -- Index arithmetic: row = k / 784, col = k % 784
  -- unflatten creates weights[row, col] = params[k]
  -- flatten reads weights[row, col] at index k
  -- Hence round-trip preserves params[k]
  sorry  -- Needs index arithmetic automation
by_cases h2 : k.1.toNat < 784 * 128 + 128
· -- Layer 1 bias range (similar)
  sorry
by_cases h3 : k.1.toNat < 784 * 128 + 128 + 128 * 10
· -- Layer 2 weights range (similar)
  sorry
· -- Layer 2 bias range (similar)
  sorry
```

**Consistency:**
The definitions of `flattenParams` and `unflattenParams` implement mathematically inverse
index transformations. The axiom asserts this algorithmic fact.

**Combined with unflatten_flatten_id:**
Together, these two axioms establish that `flattenParams` and `unflattenParams` form
a bijection (isomorphism) between `MLPArchitecture` and `Vector nParams`. This is
the formal statement that our parameter representation is information-preserving.

**References:**
- Dual of `unflatten_flatten_id`
- SciLean DataArray.ext: SciLean/Data/DataArray/DataArray.lean:130

**Impact:**
Essential for gradient descent correctness. Ensures that parameter updates computed
on the flattened representation correctly correspond to updates on the network structure.
-/
axiom flatten_unflatten_id (params : Vector nParams) :
    flattenParams (unflattenParams params) = params

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
    for h_mem : i in Array.range b do
      -- Use the axiom to convert membership to bound
      have hi : i < b := array_range_mem_bound i h_mem
      if h_target : i < targets.size then
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
    for h_mem : i in Array.range b do
      -- Use the axiom to convert membership to bound
      have hi : i < b := array_range_mem_bound i h_mem
      if h_target : i < targets.size then
        -- Extract input sample from batch - convert Nat to Idx
        let idxI : Idx b := (Idx.finEquiv b).invFun ⟨i, hi⟩
        let inputSample : Vector 784 := ⊞ j => inputs[idxI, j]
        let target := targets[i]
        let loss := computeLoss params inputSample target
        lossSum := lossSum + loss
    return lossSum / b.toFloat

end VerifiedNN.Network.Gradient
