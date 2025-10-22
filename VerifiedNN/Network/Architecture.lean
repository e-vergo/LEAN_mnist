import VerifiedNN.Layer.Dense
import VerifiedNN.Core.Activation
import SciLean

/-!
# Network Architecture

MLP architecture definition and forward pass implementation.

This module defines a two-layer multilayer perceptron (MLP) for MNIST classification.
The architecture uses dependent types to enforce dimension consistency at compile time,
preventing shape mismatches between layers.

## Main Definitions

- `MLPArchitecture`: Network structure containing two dense layers (784 → 128 → 10)
- `MLPArchitecture.forward`: Single-sample forward pass with ReLU and Softmax activations
- `MLPArchitecture.forwardBatch`: Efficient batched forward pass for training
- `MLPArchitecture.predict`: Class prediction from network output (argmax)
- `MLPArchitecture.forwardLogits`: Forward pass without final softmax (for numerical stability)
- `MLPArchitecture.predictBatch`: Batched prediction for multiple inputs
- `argmax`: Functional argmax implementation finding maximum element index
- `softmaxBatch`: Row-wise softmax for batch processing

## Architecture Details

**Layer Structure:** 784 → 128 → 10
- Input layer: 784 dimensions (28×28 pixel images flattened)
- Hidden layer: 128 dimensions with ReLU activation
- Output layer: 10 dimensions (digit classes 0-9) with Softmax activation

**Type Safety:**
The type system enforces that `layer1.outDim = layer2.inDim`, preventing
dimension mismatches at compile time. All batch operations maintain dimension
consistency through dependent type specifications.

## Implementation Notes

- `argmax` uses functional recursion over `Fin n` indices to avoid `Idx` type complexity
- `softmaxBatch` applies softmax independently to each row of a batch
- `forwardLogits` provides unnormalized scores for numerically stable loss computation
- All batch operations use DataArrayN matrix notation (⊞) for efficiency

## Verification Status

- **Sorries:** 0 (complete implementation)
- **Axioms:** 0 (no axiomatized properties)
- **Build status:** ✅ Compiles successfully
- **Type safety:** All dimension specifications enforced by dependent types

## References

- Dense layer implementation: `VerifiedNN.Layer.Dense`
- Activation functions: `VerifiedNN.Core.Activation`
- Training usage: `VerifiedNN.Training.Loop`
-/

namespace VerifiedNN.Network

open VerifiedNN.Core
open VerifiedNN.Layer
open VerifiedNN.Core.Activation
open SciLean

/-- MLP architecture: 784 → 128 → 10

Two-layer fully-connected neural network for MNIST classification.

**Architecture:**
- Input layer: 784 dimensions (28×28 pixel images flattened)
- Hidden layer: 128 dimensions with ReLU activation
- Output layer: 10 dimensions (digit classes 0-9) with Softmax activation

**Type Safety:** Layer dimensions are enforced at compile time through dependent types.
The type checker ensures that layer1.outDim (128) matches layer2.inDim (128),
preventing dimension mismatches.
-/
structure MLPArchitecture where
  layer1 : DenseLayer 784 128
  layer2 : DenseLayer 128 10

/-- Forward pass through the network.

Computes the forward pass through a two-layer MLP:
1. First dense layer (784 -> 128)
2. ReLU activation
3. Second dense layer (128 -> 10)
4. Softmax activation (for probability distribution)

**Parameters:**
- `net`: The MLP network structure
- `x`: Input vector of dimension 784

**Returns:** Output probability distribution of dimension 10
-/
def MLPArchitecture.forward (net : MLPArchitecture) (x : Vector 784) : Vector 10 :=
  let hidden := net.layer1.forward x  -- Dense layer: 784 -> 128
  let activated := reluVec hidden      -- ReLU activation
  let logits := net.layer2.forward activated  -- Dense layer: 128 -> 10
  softmax logits                       -- Softmax for probabilities

/-- Batched softmax activation.

Applies softmax to each row of a batch independently.

**Parameters:**
- `X`: Batch of b vectors, each of dimension n

**Returns:** Batch of b probability distributions, each of dimension n
-/
def softmaxBatch {b n : Nat} (X : Batch b n) : Batch b n :=
  ⊞ (k, j) =>
    -- Extract row k, apply softmax, get element j
    let row : Vector n := ⊞ j' => X[k, j']
    (softmax row)[j]

/-- Batched forward pass.

Processes a batch of inputs through the network in parallel.
More efficient than individual forward passes for training.

**Parameters:**
- `net`: The MLP network structure
- `X`: Batch of b input vectors, each of dimension 784

**Returns:** Batch of b output probability distributions, each of dimension 10
-/
def MLPArchitecture.forwardBatch {b : Nat} (net : MLPArchitecture) (X : Batch b 784) : Batch b 10 :=
  let hidden := net.layer1.forwardBatch X  -- Batched dense layer: (b, 784) -> (b, 128)
  let activated := reluBatch hidden         -- Batched ReLU activation
  let logits := net.layer2.forwardBatch activated  -- Batched dense layer: (b, 128) -> (b, 10)
  softmaxBatch logits  -- Row-wise softmax for each sample

/-- Find the index of the maximum element in a vector (argmax).

**Parameters:**
- `v`: Input vector of dimension n

**Returns:** Index of the maximum element (0-indexed)

**Implementation:** Uses a functional fold pattern to avoid Idx type complexity.
Iterates through all indices [0, n), tracking the maximum value and its index.

**Edge Case:** Returns 0 for empty vectors (n = 0).

**Approach:** Functional recursion over Fin n indices, avoiding imperative loops
and Idx type construction issues encountered in previous attempts.
-/
def argmax {n : Nat} (v : Vector n) : Nat :=
  if h : 0 < n then
    -- Helper function: recursively find argmax from index i onwards
    -- Returns (maxIndex, maxValue) pair
    let rec findMaxFrom (i : Fin n) (currentMaxIdx : Nat) (currentMaxVal : Float) : Nat × Float :=
      -- Convert Fin n to Idx n using SciLean's equivalence
      let idx : Idx n := (Idx.finEquiv n).invFun i
      let val := v[idx]
      let (newMaxIdx, newMaxVal) :=
        if val > currentMaxVal then (i.val, val) else (currentMaxIdx, currentMaxVal)
      if h' : i.val + 1 < n then
        findMaxFrom ⟨i.val + 1, h'⟩ newMaxIdx newMaxVal
      else
        (newMaxIdx, newMaxVal)

    -- Start from index 0
    let firstIdx : Fin n := ⟨0, h⟩
    let firstIdxIdx : Idx n := (Idx.finEquiv n).invFun firstIdx
    let result := findMaxFrom firstIdx 0 v[firstIdxIdx]
    result.1
  else
    0  -- Empty vector case

/-- Predict class from network output.

Returns the index of the maximum probability in the output vector,
which corresponds to the predicted class (0-9 for MNIST).

**Parameters:**
- `net`: The MLP network structure
- `x`: Input vector of dimension 784

**Returns:** Predicted class index (0-9)
-/
def MLPArchitecture.predict (net : MLPArchitecture) (x : Vector 784) : Nat :=
  let output := net.forward x
  argmax output

/-- Compute network output logits (before softmax).

Sometimes useful for numerical stability in loss computation,
as we can combine softmax + cross-entropy more efficiently.

**Parameters:**
- `net`: The MLP network structure
- `x`: Input vector of dimension 784

**Returns:** Output logits of dimension 10 (unnormalized scores)
-/
def MLPArchitecture.forwardLogits (net : MLPArchitecture) (x : Vector 784) : Vector 10 :=
  let hidden := net.layer1.forward x
  let activated := reluVec hidden
  net.layer2.forward activated

/-- Batched prediction for multiple inputs.

**Parameters:**
- `net`: The MLP network structure
- `X`: Batch of b input vectors, each of dimension 784

**Returns:** Array of predicted class indices

**Implementation Note:** Uses functional approach with Array.ofFn to avoid
imperative loop complexities. Extracts each row from batch output and applies argmax.
-/
def MLPArchitecture.predictBatch {b : Nat} (net : MLPArchitecture) (X : Batch b 784) : Array Nat :=
  let outputs := net.forwardBatch X  -- outputs : Batch b 10
  -- Use Array.ofFn to functionally create array of predictions
  Array.ofFn (fun (i : Fin b) =>
    -- Extract row i from outputs as a Vector 10
    -- Convert Fin b to Idx b using SciLean's equivalence
    let row : Vector 10 := ⊞ (j : Idx 10) => outputs[(Idx.finEquiv b).invFun i, j]
    argmax row
  )

end VerifiedNN.Network
