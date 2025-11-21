import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Stochastic Gradient Descent

SGD optimizer implementation for neural network parameter updates.

## Main Definitions

- `SGDState n`: Optimizer state containing parameters, learning rate, and epoch counter
- `sgdStep`: Basic SGD parameter update step
- `sgdStepClipped`: SGD step with gradient norm clipping
- `initSGD`: Initialize SGD state from parameters
- `updateLearningRate`: Update learning rate for scheduling

## Algorithm

This module implements the basic stochastic gradient descent update rule:

  **θ_{t+1} = θ_t - η · ∇L(θ_t)**

where:
- θ ∈ ℝⁿ are the model parameters
- η > 0 is the learning rate (step size)
- ∇L(θ_t) is the gradient of the loss function at parameters θ_t
- t is the iteration/epoch counter

## Main Results

- Dimension consistency: Type system guarantees parameter-gradient dimension matching
- Gradient clipping preserves direction while bounding magnitude
- Updates are deterministic and dimension-preserving by construction

## Implementation Notes

- Uses dependent types to enforce compile-time dimension safety
- Functions marked `@[inline]` for hot-path optimization
- Gradient clipping uses squared norm comparison to avoid unnecessary sqrt
- Learning rate scheduling handled via `updateLearningRate`

## Verification Status

**Verified:**
- Dimension consistency (by dependent types)
- Type safety of all operations

**Out of scope:**
- Convergence properties (optimization theory)
- Numerical stability of Float operations (ℝ vs Float gap)

## Production Usage Status

**Actively used in production:**
- `sgdStep` - Core SGD parameter update (Training/Loop.lean lines 403-413)
- Used in: MNISTTrainFull, MNISTTrainMedium, MiniTraining
- Achieved: 93% MNIST accuracy (60K samples, 3.3 hours, 50 epochs)

**Production-ready but currently unused:**
- `sgdStepClipped` - SGD with gradient clipping
- Tested extensively (Testing/OptimizerTests.lean), mathematically correct
- Consider for training robustness improvements (prevents gradient explosion)

**Implementation note:** Production training uses constant learning rate (config.learningRate).
Learning rate scheduling available in Update.lean but not currently adopted.

See Training/Loop.lean for production usage patterns.

## References

- Robbins, H., & Monro, S. (1951). "A Stochastic Approximation Method". *Annals of Mathematical Statistics*.
- Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent". *COMPSTAT*.
-/

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

Clips gradient norm to specified maximum before applying update. The clipping operation is:

  **g_clipped = g · min(1, maxNorm / ‖g‖₂)**

If ‖g‖₂ ≤ maxNorm, the gradient is unchanged. Otherwise, it is rescaled to have
norm exactly maxNorm while preserving direction.

**Parameters:**
- `state`: Current SGD state
- `gradient`: Computed gradient (potentially large)
- `maxNorm`: Maximum allowed gradient norm (typically 1.0 or 5.0)

**Returns:** Updated SGD state with clipped gradient applied

**Note:** Uses squared norm comparison to avoid unnecessary sqrt computation in common case.

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
