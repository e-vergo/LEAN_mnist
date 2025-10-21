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

set_default_scalar Float

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

**Implementation Note:** TODO - Requires careful handling of Idx types and indexing.
Currently uses sorry as placeholder.
-/
def flattenParams (net : MLPArchitecture) : Vector nParams :=
  sorry  -- TODO: Implement parameter flattening with proper Idx type handling

/-- Unflatten parameter vector back to network structure.

Reconstructs the network from a flattened parameter vector.
This is the inverse operation of flattenParams.

**Parameters:**
- `params`: Flattened parameter vector of dimension nParams

**Returns:** MLPArchitecture with parameters extracted from the vector

**Implementation Note:** TODO - Requires careful handling of Idx types and indexing.
Currently uses sorry as placeholder.
-/
def unflattenParams (params : Vector nParams) : MLPArchitecture :=
  sorry  -- TODO: Implement parameter unflattening with proper Idx type handling

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

**Implementation Strategy:**
1. Unflatten params to get network structure
2. Compute forward pass through network
3. Compute loss using cross-entropy
4. Use SciLean's ∇ operator to differentiate with respect to params
5. Apply fun_trans to simplify the gradient expression

**Verification Status:** This function will be proven to compute the
mathematically correct gradient in VerifiedNN.Verification.GradientCorrectness

**Current Implementation:** Uses numerical gradient approximation via finite
differences as a placeholder. Full SciLean AD integration is planned but requires
additional differentiation rules to be registered for the network operations.
-/
def networkGradient (params : Vector nParams)
    (input : Vector 784) (target : Nat) : Vector nParams :=
  -- TODO: Implement using SciLean's automatic differentiation
  -- The ideal implementation would be:
  --   let lossFunc := fun p => computeLoss p input target
  --   (∇ p, lossFunc p) params
  -- This requires additional fun_trans rules for all network operations
  sorry

/-- Compute gradient for a mini-batch of samples.

Computes the average gradient over a batch of training examples.
This is more efficient than computing individual gradients.

**Parameters:**
- `params`: Flattened network parameters
- `inputs`: Batch of b input vectors, each of dimension 784
- `targets`: Array of b target classes

**Returns:** Average gradient vector of dimension nParams

**Implementation Note:** Currently averages individual gradients. Future
optimization could use batched loss computation for better performance.
-/
def networkGradientBatch {b : Nat} (params : Vector nParams)
    (inputs : Batch b 784) (targets : Array Nat) : Vector nParams :=
  -- TODO: Implement batched gradient computation
  -- Requires proper Idx type handling for batched operations
  sorry

/-- Helper function to compute batched loss.

**Parameters:**
- `params`: Flattened network parameters
- `inputs`: Batch of b input vectors
- `targets`: Array of b target classes

**Returns:** Average loss over the batch

**Implementation Note:** Computes loss for each sample and averages.
-/
def computeLossBatch {b : Nat} (params : Vector nParams)
    (inputs : Batch b 784) (targets : Array Nat) : Float :=
  sorry  -- TODO: Implement batched loss with proper Idx type handling

end VerifiedNN.Network.Gradient
