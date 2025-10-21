/-
# Network Gradient Computation

Gradient computation using SciLean's automatic differentiation.

This module provides utilities for:
1. Flattening network parameters into a single vector for optimization
2. Unflattening parameter vectors back to network structure
3. Computing gradients with respect to all network parameters
-/

import VerifiedNN.Network.Architecture
import VerifiedNN.Loss.CrossEntropy
import SciLean

namespace VerifiedNN.Network.Gradient

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Loss
open SciLean

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

**Implementation Note:** This currently requires manual implementation of
matrix/vector flattening and concatenation. The order must match unflattenParams.
-/
def flattenParams (net : MLPArchitecture) : Vector nParams :=
  -- TODO: Implement flattening
  -- 1. Flatten layer1.weights (Matrix 128 784) -> 100352 elements
  -- 2. Concatenate layer1.bias (Vector 128) -> 128 elements
  -- 3. Flatten layer2.weights (Matrix 10 128) -> 1280 elements
  -- 4. Concatenate layer2.bias (Vector 10) -> 10 elements
  -- Total: 101770 elements
  sorry -- Requires matrix flattening and array concatenation

/-- Unflatten parameter vector back to network structure.

Reconstructs the network from a flattened parameter vector.
This is the inverse operation of flattenParams.

**Parameters:**
- `params`: Flattened parameter vector of dimension nParams

**Returns:** MLPArchitecture with parameters extracted from the vector

**Implementation Note:** Must extract parameters in the same order as flattenParams:
1. Extract layer 1 weights (first 100,352 elements)
2. Extract layer 1 bias (next 128 elements)
3. Extract layer 2 weights (next 1,280 elements)
4. Extract layer 2 bias (last 10 elements)
-/
def unflattenParams (params : Vector nParams) : MLPArchitecture :=
  -- TODO: Implement unflattening
  -- Split params into:
  --   params[0:100352]     -> reshape to Matrix 128 784
  --   params[100352:100480] -> Vector 128
  --   params[100480:101760] -> reshape to Matrix 10 128
  --   params[101760:101770] -> Vector 10
  sorry -- Requires array slicing and matrix reshaping

/-- Theorem: Flattening then unflattening is identity.

This is a critical property for gradient descent to work correctly.
-/
theorem unflatten_flatten_id (net : MLPArchitecture) :
    unflattenParams (flattenParams net) = net := by
  sorry -- TODO: Prove once flatten/unflatten are implemented

/-- Theorem: Unflattening then flattening is identity.

Another critical property for optimization correctness.
-/
theorem flatten_unflatten_id (params : Vector nParams) :
    flattenParams (unflattenParams params) = params := by
  sorry -- TODO: Prove once flatten/unflatten are implemented

/-- Compute gradient of loss with respect to network parameters.

Uses SciLean's automatic differentiation to compute the gradient.
This is the core of backpropagation in the network.

**Parameters:**
- `params`: Flattened network parameters
- `input`: Input vector of dimension 784
- `target`: Target class (0-9 for MNIST)

**Returns:** Gradient vector of dimension nParams

**Implementation Strategy:**
1. Unflatten params to get network structure
2. Compute forward pass through network
3. Compute loss using cross-entropy
4. Use SciLean's ∇ operator to differentiate with respect to params
5. Apply fun_trans to simplify the gradient expression

**Verification Status:** This function will be proven to compute the
mathematically correct gradient in VerifiedNN.Verification.GradientCorrectness
-/
def networkGradient (params : Vector nParams)
    (input : Vector 784) (target : Nat) : Vector nParams :=
  -- TODO: Implement using SciLean's automatic differentiation
  --
  -- Pseudocode:
  -- let lossFunc := fun p =>
  --   let net := unflattenParams p
  --   let output := net.forward input
  --   crossEntropyLoss output target
  -- (∇ p, lossFunc p) params
  --   |>.rewrite_by fun_trans (disch := aesop)
  sorry -- Requires SciLean gradient computation

/-- Compute gradient for a mini-batch of samples.

Computes the average gradient over a batch of training examples.
This is more efficient than computing individual gradients.

**Parameters:**
- `params`: Flattened network parameters
- `inputs`: Batch of b input vectors, each of dimension 784
- `targets`: Array of b target classes

**Returns:** Average gradient vector of dimension nParams

**Implementation Note:** Can be implemented by either:
1. Computing individual gradients and averaging
2. Using batched forward pass and loss, then differentiating
-/
def networkGradientBatch {b : Nat} (params : Vector nParams)
    (inputs : Batch b 784) (targets : Array Nat) : Vector nParams :=
  -- TODO: Implement batched gradient computation
  -- Option 1: Average individual gradients
  -- Option 2: Compute batch loss and differentiate
  sorry -- Requires batched gradient computation

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

/-- Helper function to compute batched loss.

**Parameters:**
- `params`: Flattened network parameters
- `inputs`: Batch of b input vectors
- `targets`: Array of b target classes

**Returns:** Average loss over the batch
-/
def computeLossBatch {b : Nat} (params : Vector nParams)
    (inputs : Batch b 784) (targets : Array Nat) : Float :=
  -- TODO: Implement batched loss computation
  sorry -- Requires batched loss computation

end VerifiedNN.Network.Gradient
