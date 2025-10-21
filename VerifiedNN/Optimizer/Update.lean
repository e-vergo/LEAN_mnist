/-
# Parameter Update Logic

Unified parameter update interface and learning rate scheduling.

## Overview

This module provides infrastructure for flexible optimizer configuration and training:

1. **Learning Rate Scheduling:** Multiple strategies for dynamic learning rate adjustment
2. **Gradient Accumulation:** Simulate large batch sizes with limited memory
3. **Unified Optimizer Interface:** Generic API for different optimizer types
4. **State Management:** Utilities for accessing and modifying optimizer state

## Learning Rate Schedules

### Constant Schedule
  **η(t) = η₀**

  Fixed learning rate throughout training.

### Step Decay
  **η(t) = η₀ · γ^⌊t/s⌋**

  Reduces learning rate by factor γ every s epochs.

### Exponential Decay
  **η(t) = η₀ · γ^t**

  Smooth exponential decay at rate γ.

### Cosine Annealing
  **η(t) = η₀ · (1 + cos(π·t/T)) / 2**

  Smooth decay to zero over T epochs following cosine curve.

### Warmup
  **η(t) = η_target · min(1, (t+1)/N)**

  Linear increase from 0 to target over N epochs, useful for training stability.

## Gradient Accumulation

Accumulate gradients over K mini-batches before updating parameters. This simulates
effective batch size of K × batch_size with memory usage of single batch:

  **g_eff = (1/K) · Σᵢ₌₁ᴷ g_i**

where g_i is the gradient from mini-batch i.

## Verification Status

Implementation complete. Scheduling logic is deterministic and dimension-preserving
by construction. All operations maintain type-level dimension guarantees.
-/

import VerifiedNN.Optimizer.SGD
import VerifiedNN.Optimizer.Momentum
import VerifiedNN.Core.DataTypes
import SciLean

namespace VerifiedNN.Optimizer.Update

open VerifiedNN.Core
open VerifiedNN.Optimizer
open SciLean

set_default_scalar Float

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

/-- Apply learning rate schedule at a given epoch.

**Parameters:**
- `schedule`: The learning rate schedule strategy
- `epoch`: Current epoch number (0-indexed)

**Returns:** Learning rate to use for the current epoch

**Mathematical formulas:**
- `constant α`: Returns α for all epochs
- `step α₀ s γ`: Returns α₀ · γ^⌊epoch/s⌋ (step decay every s epochs by factor γ)
- `exponential α₀ γ`: Returns α₀ · γ^epoch (exponential decay)
- `cosine α₀ T`: Returns α₀ · (1 + cos(π·epoch/T))/2 (cosine annealing over T epochs)

**Examples:**
- `constant 0.01` returns 0.01 for all epochs
- `step 0.1 30 0.1` returns 0.1 for epochs 0-29, 0.01 for 30-59, 0.001 for 60+, etc.
- `exponential 0.1 0.95` returns 0.1 * 0.95^epoch
- `cosine 0.1 100` smoothly decays from 0.1 to ~0 over 100 epochs
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
      -- Safety: Handle totalEpochs = 0 case
      if totalEpochs = 0 then
        initialLR
      else
        let progress := min 1.0 (epoch.toFloat / totalEpochs.toFloat)
        let pi : Float := 3.141592653589793
        let cosineDecay := (1.0 + Float.cos (pi * progress)) / 2.0
        initialLR * cosineDecay

/-- Warmup schedule that linearly increases learning rate from 0 to target over N epochs.

Useful for stabilizing training at the beginning, especially with large batch sizes.

**Mathematical formula:**
- For epoch < N: α · (epoch + 1) / N  (linear warmup)
- For epoch ≥ N: α  (constant at target)

where α is the target learning rate and N is warmupEpochs.

**Parameters:**
- `targetLR`: Target learning rate to reach after warmup
- `warmupEpochs`: Number of epochs for warmup phase
- `epoch`: Current epoch

**Returns:** Learning rate with warmup applied
-/
def warmupSchedule (targetLR : Float) (warmupEpochs : Nat) (epoch : Nat) : Float :=
  -- Safety: Handle warmupEpochs = 0 case
  if warmupEpochs = 0 then
    targetLR
  else if epoch < warmupEpochs then
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

Computes the mean gradient over all accumulated mini-batches:

  **g_mean = (1/K) · Σᵢ₌₁ᴷ g_i**

where K is the number of accumulated gradients.

**Parameters:**
- `acc`: Accumulator with gradients from multiple mini-batches

**Returns:**
- Average gradient (accumulated / count)
- Fresh accumulator reset to zero

**Safety:** Returns zero vector if no gradients were accumulated (avoids division by zero).

**Note:** The average is computed as accumulated / count only when count > 0.

**Usage Pattern:**
```lean
let acc := initAccumulator n
let acc := addGradient acc grad1
let acc := addGradient acc grad2
let (avgGrad, freshAcc) := getAndReset acc
-- avgGrad = (grad1 + grad2) / 2
```
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

/-- Apply optimizer step using the appropriate update rule.

**Parameters:**
- `state`: Optimizer state (SGD or Momentum)
- `gradient`: Computed gradient

**Returns:** Updated optimizer state

**Performance:** Marked inline for hot-path optimization.
-/
@[inline]
def optimizerStep {n : Nat} (state : OptimizerState n) (gradient : Vector n) : OptimizerState n :=
  match state with
  | OptimizerState.sgd s => OptimizerState.sgd (sgdStep s gradient)
  | OptimizerState.momentum s => OptimizerState.momentum (Momentum.momentumStep s gradient)

/-- Extract parameters from optimizer state.

**Parameters:**
- `state`: Optimizer state (SGD or Momentum)

**Returns:** Current parameter vector

**Performance:** Marked inline for hot-path optimization.
-/
@[inline]
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
