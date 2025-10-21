/-
# Training Loop

Main training loop implementation for neural network training.

## Overview

This module implements the core training loop that orchestrates the entire
training process for the MLP network on MNIST. It handles:
- Epoch iteration with configurable hyperparameters
- Mini-batch processing via SGD optimization
- Progress tracking and periodic evaluation
- Training state management and checkpointing

## Implementation Status

**Partial implementation:** Core functionality is complete, but the following
enhancements are planned:
- Gradient clipping for training stability
- Early stopping based on validation metrics
- Learning rate scheduling
- More sophisticated logging and visualization

## Training Architecture

The training loop follows this structure:
1. **Initialization:** Set up network, optimizer state, and configuration
2. **Epoch Loop:** For each epoch, shuffle data and create mini-batches
3. **Batch Loop:** For each batch, compute gradients and update parameters
4. **Evaluation:** Periodically evaluate on validation set
5. **Checkpoint:** Return final trained state

## Gradient Computation

Gradients are computed using automatic differentiation via SciLean:
- Forward pass through network computes predictions
- Cross-entropy loss measures prediction error
- Backward pass (automatic) computes gradients w.r.t. parameters
- Gradients are averaged across the mini-batch
- SGD step updates parameters using averaged gradients

## Usage

```lean
-- Simple interface
let trainedNet ← trainEpochs initialNet trainData 10 32 0.01

-- Full control with configuration
let config : TrainConfig := {
  epochs := 10
  batchSize := 32
  learningRate := 0.01
  printEveryNBatches := 100
  evaluateEveryNEpochs := 1
}
let finalState ← trainEpochsWithConfig initialNet trainData config (some validData)
```
-/

import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Gradient
import VerifiedNN.Training.Batch
import VerifiedNN.Training.Metrics
import VerifiedNN.Loss.CrossEntropy
import VerifiedNN.Optimizer.SGD
import SciLean

namespace VerifiedNN.Training.Loop

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Network.Gradient
open VerifiedNN.Training.Batch
open VerifiedNN.Training.Metrics
open VerifiedNN.Loss
open VerifiedNN.Optimizer
open SciLean

/-- Training configuration structure.

Contains hyperparameters and settings for the training process.
-/
structure TrainConfig where
  epochs : Nat
  batchSize : Nat
  learningRate : Float
  printEveryNBatches : Nat := 100
  evaluateEveryNEpochs : Nat := 1
  deriving Repr

/-- Training state that tracks progress through training.

Maintains the current network state, optimizer state, and training metrics.
-/
structure TrainState where
  net : MLPArchitecture
  optimState : SGDState nParams
  currentEpoch : Nat
  totalBatchesSeen : Nat

/-- Initialize training state from a network and configuration.

**Parameters:**
- `net`: Initial network (typically randomly initialized)
- `config`: Training configuration

**Returns:** Initial training state
-/
def initTrainState (net : MLPArchitecture) (config : TrainConfig) : TrainState :=
  { net := net
    optimState := {
      params := flattenParams net
      learningRate := config.learningRate
      epoch := 0
    }
    currentEpoch := 0
    totalBatchesSeen := 0
  }

/-- Train on a single mini-batch.

Performs forward pass, computes gradients, and updates parameters.

**Parameters:**
- `state`: Current training state
- `batch`: Mini-batch of training examples

**Returns:** Updated training state

**Note:** Gradient accumulation is implemented using the functional approach:
compute individual gradients and accumulate them. The averaging is done using
scalar multiplication.
-/
noncomputable def trainBatch (state : TrainState) (batch : Array (Vector 784 × Nat)) : TrainState :=
  if batch.size == 0 then
    state
  else
    -- Compute gradient for the batch
    -- We accumulate gradients across the batch and average them
    let params := state.optimState.params

    -- Compute average gradient over the batch
    let gradSum := batch.foldl (fun accGrad (input, label) =>
      let grad := networkGradient params input label
      ⊞ i => accGrad[i] + grad[i]
    ) (⊞ (_ : Idx nParams) => (0.0 : Float))

    -- Average the gradients
    let batchSizeFloat := batch.size.toFloat
    let avgGrad := ⊞ i => gradSum[i] / batchSizeFloat

    -- Apply SGD step
    let newOptimState := sgdStep state.optimState avgGrad

    -- Update network from new parameters
    let newNet := unflattenParams newOptimState.params

    -- Return updated state
    { state with
      net := newNet
      optimState := newOptimState
      totalBatchesSeen := state.totalBatchesSeen + 1
    }

/-- Train for one epoch.

Processes all mini-batches in the training data for one complete pass.

**Parameters:**
- `state`: Current training state
- `trainData`: Array of training examples
- `config`: Training configuration
- `validData`: Optional validation data for evaluation

**Returns:** Updated training state after one epoch
-/
noncomputable def trainOneEpoch
    (state : TrainState)
    (trainData : Array (Vector 784 × Nat))
    (config : TrainConfig)
    (validData : Option (Array (Vector 784 × Nat)) := none) : IO TrainState := do
  -- Create shuffled batches for this epoch
  let batches ← createShuffledBatches trainData config.batchSize

  -- Process all batches with progress tracking
  let mut currentState := state
  for batchIdx in [0:batches.size] do
    let batch := batches[batchIdx]!
    currentState := trainBatch currentState batch

    -- Print progress periodically
    if (batchIdx + 1) % config.printEveryNBatches == 0 then
      IO.println s!"  Batch {batchIdx + 1}/{batches.size} (Epoch {state.currentEpoch + 1})"

  -- Evaluate on validation set if provided and it's the right epoch
  if state.currentEpoch % config.evaluateEveryNEpochs == 0 then
    match validData with
    | some vData =>
      let accuracy := computeAccuracy currentState.net vData
      let avgLoss := computeAverageLoss currentState.net vData
      IO.println s!"Epoch {state.currentEpoch + 1}: Validation Accuracy = {accuracy * 100.0}%, Loss = {avgLoss}"
    | none => pure ()

  -- Return state with incremented epoch counter
  return { currentState with currentEpoch := currentState.currentEpoch + 1 }

/-- Train network for multiple epochs.

Main entry point for training. Iterates through epochs and tracks progress.

**Parameters:**
- `net`: Initial network (typically randomly initialized)
- `trainData`: Training dataset
- `epochs`: Number of epochs to train
- `batchSize`: Mini-batch size
- `learningRate`: Learning rate for SGD

**Returns:** Trained network

**Note:** This is a simplified interface. For more control, use `trainEpochsWithConfig`.
-/
noncomputable def trainEpochsWithConfig
    (net : MLPArchitecture)
    (trainData : Array (Vector 784 × Nat))
    (config : TrainConfig)
    (validData : Option (Array (Vector 784 × Nat)) := none) : IO TrainState := do
  -- Initialize training state
  let mut state := initTrainState net config

  -- Print initial configuration
  IO.println s!"Starting training for {config.epochs} epochs"
  IO.println s!"Batch size: {config.batchSize}, Learning rate: {config.learningRate}"
  IO.println s!"Training samples: {trainData.size}"

  -- Train for specified number of epochs
  for epochIdx in [0:config.epochs] do
    IO.println s!"Epoch {epochIdx + 1}/{config.epochs}"
    state ← trainOneEpoch state trainData config validData

  IO.println "Training complete!"

  -- Return final state
  return state

noncomputable def trainEpochs
    (net : MLPArchitecture)
    (trainData : Array (Vector 784 × Nat))
    (epochs : Nat)
    (batchSize : Nat)
    (learningRate : Float) : IO MLPArchitecture := do
  -- Create config from simple parameters
  let config : TrainConfig := {
    epochs := epochs
    batchSize := batchSize
    learningRate := learningRate
    printEveryNBatches := 100
    evaluateEveryNEpochs := 1
  }

  -- Delegate to full-featured training function
  let finalState ← trainEpochsWithConfig net trainData config none

  -- Return just the network
  return finalState.net

/-- Resume training from a checkpoint.

Allows continuing training from a saved state with potentially different configuration.

**Parameters:**
- `state`: Checkpointed training state
- `trainData`: Training dataset
- `additionalEpochs`: Number of additional epochs to train
- `newLearningRate`: Optional new learning rate (if None, keeps current rate)
- `validData`: Optional validation dataset

**Returns:** Updated training state
-/
noncomputable def resumeTraining
    (state : TrainState)
    (trainData : Array (Vector 784 × Nat))
    (additionalEpochs : Nat)
    (newLearningRate : Option Float := none)
    (validData : Option (Array (Vector 784 × Nat)) := none) : IO TrainState := do
  -- Update learning rate if provided
  let updatedState := match newLearningRate with
    | some lr =>
      { state with
        optimState := updateLearningRate state.optimState lr
      }
    | none => state

  -- Create config for additional training
  let config : TrainConfig := {
    epochs := additionalEpochs
    batchSize := 32  -- Default batch size for resume
    learningRate := updatedState.optimState.learningRate
    printEveryNBatches := 100
    evaluateEveryNEpochs := 1
  }

  -- Print resume information
  IO.println s!"Resuming training from epoch {updatedState.currentEpoch}"
  IO.println s!"Training for {additionalEpochs} additional epochs"
  IO.println s!"Learning rate: {updatedState.optimState.learningRate}"

  -- Train for additional epochs
  let mut currentState := updatedState
  for _epochIdx in [0:additionalEpochs] do
    IO.println s!"Epoch {currentState.currentEpoch + 1}"
    currentState ← trainOneEpoch currentState trainData config validData

  IO.println "Resumed training complete!"

  -- Return final state
  return currentState

end VerifiedNN.Training.Loop
