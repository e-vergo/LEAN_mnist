/-
# Stochastic Gradient Descent

SGD optimizer implementation for neural network parameter updates.

This module implements the basic stochastic gradient descent update rule:
  θ_{t+1} = θ_t - η * ∇L(θ_t)
where θ are the parameters, η is the learning rate, and ∇L is the gradient.

**Verification Status:** Implementation complete. Formal verification of convergence
properties is out of scope (optimization theory), but dimension consistency is
maintained by construction via dependent types.
-/

import VerifiedNN.Core.DataTypes
import SciLean

namespace VerifiedNN.Optimizer

open VerifiedNN.Core
open SciLean

set_default_scalar Float

/-- SGD optimizer state containing parameters, learning rate, and epoch counter.

The state tracks:
- `params`: Current parameter vector of dimension nParams
- `learningRate`: Step size for gradient updates (η)
- `epoch`: Current training epoch (for monitoring and scheduling)
-/
structure SGDState (nParams : Nat) where
  params : Vector nParams
  learningRate : Float
  epoch : Nat

/-- Single SGD step: θ_{t+1} = θ_t - η * ∇L(θ_t)

Given the current state and computed gradient, performs one parameter update step
following the standard SGD update rule.

**Parameters:**
- `state`: Current SGD state with parameters θ_t
- `gradient`: Computed gradient ∇L(θ_t) of the same dimension

**Returns:** Updated SGD state with new parameters θ_{t+1}

**Dimension safety:** Type system ensures gradient and parameters have matching dimensions.

**Performance:** Marked inline for hot-path optimization.
-/
@[inline]
def sgdStep {n : Nat} (state : SGDState n) (gradient : Vector n) : SGDState n :=
  { state with
    params := state.params - state.learningRate • gradient
    epoch := state.epoch + 1
  }

/-- SGD step with gradient clipping to prevent gradient explosion.

Clips gradient norm to specified maximum before applying update. If ‖∇L‖ > maxNorm,
the gradient is rescaled to have norm maxNorm while preserving direction.

**Parameters:**
- `state`: Current SGD state
- `gradient`: Computed gradient (potentially large)
- `maxNorm`: Maximum allowed gradient norm

**Returns:** Updated SGD state with clipped gradient applied

**Performance:** Marked inline for hot-path optimization.
-/
@[inline]
def sgdStepClipped {n : Nat} (state : SGDState n) (gradient : Vector n) (maxNorm : Float) : SGDState n :=
  let gradNorm := ‖gradient‖₂²
  let scaleFactor := if gradNorm > maxNorm * maxNorm then
    maxNorm / Float.sqrt gradNorm
  else
    1.0
  sgdStep state (scaleFactor • gradient)

/-- Update learning rate in SGD state.

Used for learning rate scheduling during training.

**Parameters:**
- `state`: Current SGD state
- `newLR`: New learning rate value

**Returns:** SGD state with updated learning rate
-/
def updateLearningRate {n : Nat} (state : SGDState n) (newLR : Float) : SGDState n :=
  { state with learningRate := newLR }

/-- Initialize SGD state with given parameters and learning rate.

**Parameters:**
- `initialParams`: Initial parameter vector (typically from weight initialization)
- `lr`: Learning rate η for SGD updates

**Returns:** Initial SGD state ready for training
-/
def initSGD {n : Nat} (initialParams : Vector n) (lr : Float) : SGDState n :=
  { params := initialParams
    learningRate := lr
    epoch := 0
  }

end VerifiedNN.Optimizer
