/-
# Parameter Update Logic

Unified parameter update interface and learning rate scheduling.

This module provides:
- Learning rate scheduling strategies (constant, step decay, exponential decay)
- Gradient accumulation for effective large batch training
- Utility functions for optimizer state management
- Generic parameter update interfaces

**Verification Status:** Implementation complete. Scheduling logic is deterministic
and dimension-preserving by construction.
-/

import VerifiedNN.Optimizer.SGD
import VerifiedNN.Optimizer.Momentum
import VerifiedNN.Core.DataTypes
import SciLean

namespace VerifiedNN.Optimizer.Update

open VerifiedNN.Core
open VerifiedNN.Optimizer
open SciLean

/-- Learning rate scheduling strategies.

Common schedules include:
- `constant`: Fixed learning rate throughout training
- `step`: Reduce learning rate by a factor every N epochs
- `exponential`: Exponential decay of learning rate
- `cosine`: Cosine annealing (smooth decay to zero)
-/
inductive LRSchedule where
  | constant : Float → LRSchedule
  | step : Float → Nat → Float → LRSchedule  -- initial, step size, decay factor
  | exponential : Float → Float → LRSchedule  -- initial, decay rate
  | cosine : Float → Nat → LRSchedule  -- initial, total epochs
  deriving Repr

/-- Apply learning rate schedule at a given epoch.

**Parameters:**
- `schedule`: The learning rate schedule strategy
- `epoch`: Current epoch number (0-indexed)

**Returns:** Learning rate to use for the current epoch

**Examples:**
- `constant 0.01` returns 0.01 for all epochs
- `step 0.1 30 0.1` returns 0.1 for epochs 0-29, 0.01 for 30-59, 0.001 for 60+, etc.
- `exponential 0.1 0.95` returns 0.1 * 0.95^epoch
-/
def applySchedule (schedule : LRSchedule) (epoch : Nat) : Float :=
  match schedule with
  | LRSchedule.constant lr => lr
  | LRSchedule.step initialLR stepSize decayFactor =>
      let numSteps := epoch / stepSize
      initialLR * Float.pow decayFactor numSteps.toFloat
  | LRSchedule.exponential initialLR decayRate =>
      initialLR * Float.pow decayRate epoch.toFloat
  | LRSchedule.cosine initialLR totalEpochs =>
      let progress := Float.min 1.0 (epoch.toFloat / totalEpochs.toFloat)
      let cosineDecay := (1.0 + Float.cos (Float.pi * progress)) / 2.0
      initialLR * cosineDecay

/-- Warmup schedule that linearly increases learning rate from 0 to target over N epochs.

Useful for stabilizing training at the beginning, especially with large batch sizes.

**Parameters:**
- `targetLR`: Target learning rate to reach after warmup
- `warmupEpochs`: Number of epochs for warmup phase
- `epoch`: Current epoch

**Returns:** Learning rate with warmup applied
-/
def warmupSchedule (targetLR : Float) (warmupEpochs : Nat) (epoch : Nat) : Float :=
  if epoch < warmupEpochs then
    targetLR * ((epoch.toFloat + 1.0) / warmupEpochs.toFloat)
  else
    targetLR

/-- Combine warmup with a main schedule.

**Parameters:**
- `warmupEpochs`: Number of warmup epochs
- `mainSchedule`: Schedule to use after warmup
- `epoch`: Current epoch

**Returns:** Combined learning rate
-/
def warmupThenSchedule (warmupEpochs : Nat) (mainSchedule : LRSchedule) (epoch : Nat) : Float :=
  if epoch < warmupEpochs then
    let baseLR := applySchedule mainSchedule 0
    warmupSchedule baseLR warmupEpochs epoch
  else
    applySchedule mainSchedule (epoch - warmupEpochs)

/-- Gradient accumulator for simulating larger batch sizes.

Accumulates gradients over multiple mini-batches before applying parameter update.
This is useful when memory constraints prevent using desired batch size directly.
-/
structure GradientAccumulator (n : Nat) where
  accumulated : Vector n
  count : Nat
  deriving Repr

/-- Initialize gradient accumulator with zero gradients.

**Parameters:**
- `n`: Parameter dimension

**Returns:** Fresh accumulator ready to accumulate gradients
-/
def initAccumulator (n : Nat) : GradientAccumulator n :=
  { accumulated := (0 : Float^[n])
    count := 0
  }

/-- Add a gradient to the accumulator.

**Parameters:**
- `acc`: Current accumulator state
- `gradient`: Gradient from one mini-batch

**Returns:** Updated accumulator with gradient added
-/
def addGradient {n : Nat} (acc : GradientAccumulator n) (gradient : Vector n) : GradientAccumulator n :=
  { accumulated := acc.accumulated + gradient
    count := acc.count + 1
  }

/-- Get the accumulated gradient average and reset the accumulator.

**Parameters:**
- `acc`: Accumulator with gradients from multiple mini-batches

**Returns:**
- Average gradient (accumulated / count)
- Fresh accumulator reset to zero

**Note:** Returns zero vector if no gradients were accumulated.
-/
def getAndReset {n : Nat} (acc : GradientAccumulator n) : Vector n × GradientAccumulator n :=
  if acc.count > 0 then
    let avgGradient := (1.0 / acc.count.toFloat) • acc.accumulated
    (avgGradient, initAccumulator n)
  else
    (0, acc)

/-- Generic optimizer state that can hold either SGD or Momentum.

This provides a unified interface for different optimizer types.
-/
inductive OptimizerState (n : Nat) where
  | sgd : SGDState n → OptimizerState n
  | momentum : Momentum.MomentumState n → OptimizerState n
  deriving Repr

/-- Apply optimizer step using the appropriate update rule.

**Parameters:**
- `state`: Optimizer state (SGD or Momentum)
- `gradient`: Computed gradient

**Returns:** Updated optimizer state
-/
def optimizerStep {n : Nat} (state : OptimizerState n) (gradient : Vector n) : OptimizerState n :=
  match state with
  | OptimizerState.sgd s => OptimizerState.sgd (sgdStep s gradient)
  | OptimizerState.momentum s => OptimizerState.momentum (Momentum.momentumStep s gradient)

/-- Extract parameters from optimizer state.

**Parameters:**
- `state`: Optimizer state (SGD or Momentum)

**Returns:** Current parameter vector
-/
def getParams {n : Nat} (state : OptimizerState n) : Vector n :=
  match state with
  | OptimizerState.sgd s => s.params
  | OptimizerState.momentum s => s.params

/-- Update learning rate in optimizer state.

**Parameters:**
- `state`: Optimizer state
- `newLR`: New learning rate value

**Returns:** Optimizer state with updated learning rate
-/
def updateOptimizerLR {n : Nat} (state : OptimizerState n) (newLR : Float) : OptimizerState n :=
  match state with
  | OptimizerState.sgd s => OptimizerState.sgd (updateLearningRate s newLR)
  | OptimizerState.momentum s => OptimizerState.momentum (Momentum.updateLearningRate s newLR)

/-- Get current epoch from optimizer state.

**Parameters:**
- `state`: Optimizer state

**Returns:** Current epoch number
-/
def getEpoch {n : Nat} (state : OptimizerState n) : Nat :=
  match state with
  | OptimizerState.sgd s => s.epoch
  | OptimizerState.momentum s => s.epoch

end VerifiedNN.Optimizer.Update
