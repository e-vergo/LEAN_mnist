/-
# SGD with Momentum

SGD optimizer with momentum for improved convergence and reduced oscillation.

This module implements SGD with classical momentum (also known as heavy ball method):
  v_{t+1} = β * v_t + ∇L(θ_t)
  θ_{t+1} = θ_t - η * v_{t+1}
where v is the velocity, β is the momentum coefficient (typically 0.9), and η is the learning rate.

Momentum helps accelerate SGD in relevant directions and dampens oscillations, leading to
faster convergence especially in the presence of high curvature or noisy gradients.

**Verification Status:** Implementation complete. Convergence properties verified
informally but not formally proven (optimization theory out of scope). Dimension
consistency maintained by dependent types.
-/

import VerifiedNN.Core.DataTypes
import SciLean

namespace VerifiedNN.Optimizer.Momentum

open VerifiedNN.Core
open SciLean

set_default_scalar Float

/-- Momentum optimizer state containing parameters, velocity, and hyperparameters.

The state tracks:
- `params`: Current parameter vector θ_t
- `velocity`: Accumulated velocity v_t (exponential moving average of gradients)
- `learningRate`: Step size η for parameter updates
- `momentum`: Momentum coefficient β (typically 0.9)
- `epoch`: Current training epoch
-/
structure MomentumState (n : Nat) where
  params : Vector n
  velocity : Vector n
  learningRate : Float
  momentum : Float
  epoch : Nat

/-- Momentum update step implementing the classical momentum algorithm.

Update rules:
  v_{t+1} = β * v_t + ∇L(θ_t)
  θ_{t+1} = θ_t - η * v_{t+1}

**Parameters:**
- `state`: Current momentum optimizer state
- `gradient`: Computed gradient ∇L(θ_t) of the loss

**Returns:** Updated state with new parameters and velocity

**Note:** This is "classical momentum" where the velocity is updated before
applying to parameters. An alternative is Nesterov momentum which looks ahead.

**Performance:** Marked inline for hot-path optimization.
-/
@[inline]
def momentumStep {n : Nat} (state : MomentumState n) (gradient : Vector n) : MomentumState n :=
  let newVelocity := state.momentum • state.velocity + gradient
  { state with
    velocity := newVelocity
    params := state.params - state.learningRate • newVelocity
    epoch := state.epoch + 1
  }

/-- Momentum step with gradient clipping to prevent gradient explosion.

Clips gradient norm before incorporating into velocity accumulation.

**Parameters:**
- `state`: Current momentum state
- `gradient`: Computed gradient (potentially large)
- `maxNorm`: Maximum allowed gradient norm

**Returns:** Updated state with clipped gradient applied

**Performance:** Marked inline for hot-path optimization.
-/
@[inline]
def momentumStepClipped {n : Nat} (state : MomentumState n) (gradient : Vector n) (maxNorm : Float) : MomentumState n :=
  let gradNorm := ‖gradient‖₂²
  let scaleFactor := if gradNorm > maxNorm * maxNorm then
    maxNorm / Float.sqrt gradNorm
  else
    1.0
  momentumStep state (scaleFactor • gradient)

/-- Update learning rate in momentum state.

Used for learning rate scheduling during training.

**Parameters:**
- `state`: Current momentum state
- `newLR`: New learning rate value

**Returns:** Momentum state with updated learning rate
-/
def updateLearningRate {n : Nat} (state : MomentumState n) (newLR : Float) : MomentumState n :=
  { state with learningRate := newLR }

/-- Update momentum coefficient.

Allows dynamic adjustment of momentum during training (rarely used but available).

**Parameters:**
- `state`: Current momentum state
- `newMomentum`: New momentum coefficient β (should be in [0, 1))

**Returns:** Momentum state with updated momentum coefficient
-/
def updateMomentum {n : Nat} (state : MomentumState n) (newMomentum : Float) : MomentumState n :=
  { state with momentum := newMomentum }

/-- Initialize momentum optimizer state.

**Parameters:**
- `initialParams`: Initial parameter vector (from weight initialization)
- `lr`: Learning rate η for parameter updates
- `beta`: Momentum coefficient β (typically 0.9)

**Returns:** Initial momentum state with zero velocity, ready for training
-/
def initMomentum {n : Nat} (initialParams : Vector n) (lr : Float) (beta : Float := 0.9) : MomentumState n :=
  { params := initialParams
    velocity := 0  -- SciLean's zero for DataArrayN
    learningRate := lr
    momentum := beta
    epoch := 0
  }

/-- Nesterov momentum update (look-ahead variant).

Update rules:
  θ_lookahead = θ_t - β * v_t
  v_{t+1} = β * v_t + ∇L(θ_lookahead)
  θ_{t+1} = θ_t - η * v_{t+1}

Nesterov momentum evaluates the gradient at the "look-ahead" position, which
can provide better convergence properties.

**Note:** Requires re-computation of gradient at look-ahead position, so this
function takes a gradient computation function rather than a pre-computed gradient.

**Parameters:**
- `state`: Current momentum state
- `computeGrad`: Function to compute gradient at a given parameter vector

**Returns:** Updated state with Nesterov momentum applied
-/
def nesterovStep {n : Nat} (state : MomentumState n) (computeGrad : Vector n → Vector n) : MomentumState n :=
  let lookahead := state.params - state.momentum • state.velocity
  let gradLookahead := computeGrad lookahead
  let newVelocity := state.momentum • state.velocity + gradLookahead
  { state with
    velocity := newVelocity
    params := state.params - state.learningRate • newVelocity
    epoch := state.epoch + 1
  }

end VerifiedNN.Optimizer.Momentum
