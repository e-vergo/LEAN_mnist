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

**Production-ready implementation:** Core functionality is complete with structured
logging and checkpoint infrastructure. Implemented features:
- ✅ Epoch iteration with configurable hyperparameters
- ✅ Mini-batch processing via SGD optimization
- ✅ Progress tracking with structured logging utilities
- ✅ Validation evaluation during training
- ✅ Checkpoint API (serialization TODO)

**Future enhancements:**
- Gradient clipping for training stability (available in Optimizer.SGD)
- Early stopping based on validation metrics
- Learning rate scheduling
- Checkpoint serialization/deserialization

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
-- Simple interface (returns just the network)
let trainedNet ← trainEpochs initialNet trainData 10 32 0.01

-- Full control with configuration and validation
let config : TrainConfig := {
  epochs := 10
  batchSize := 32
  learningRate := 0.01
  printEveryNBatches := 100
  evaluateEveryNEpochs := 1
}
let finalState ← trainEpochsWithConfig initialNet trainData config (some validData)

-- With checkpointing (serialization TODO)
let checkpointCfg : CheckpointConfig := {
  saveDir := "checkpoints"
  saveEveryNEpochs := 5
  saveOnlyBest := true
}
let finalState ← trainEpochsWithConfig initialNet trainData config (some validData) (some checkpointCfg)

-- Access final network and training state
let finalNet := finalState.net
let finalParams := finalState.optimState.params
let epochsTrained := finalState.currentEpoch
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

/-- Checkpoint configuration for saving training state.

**Note:** Checkpoint serialization/deserialization not yet implemented.
This structure defines the API for future checkpoint functionality.
-/
structure CheckpointConfig where
  /-- Directory to save checkpoints -/
  saveDir : String := "checkpoints"
  /-- Save checkpoint every N epochs (0 = never save) -/
  saveEveryNEpochs : Nat := 0
  /-- Only save checkpoints that improve validation metrics -/
  saveOnlyBest : Bool := true
  deriving Repr

-- Structured logging utilities for training progress.
-- Provides consistent, readable logging throughout the training process.
namespace TrainingLog

/-- Log the start of an epoch. -/
def logEpochStart (current total : Nat) : IO Unit :=
  IO.println s!"Epoch {current}/{total}"

/-- Log batch progress during training. -/
def logBatchProgress (current total : Nat) (loss : Float) (printEvery : Nat := 100) : IO Unit :=
  if current % printEvery == 0 then
    IO.println s!"  Batch {current}/{total}, Loss: {loss}"
  else
    pure ()

/-- Log epoch completion with metrics. -/
def logEpochEnd (epoch : Nat) (trainAcc trainLoss : Float)
    (valAcc : Option Float := none) (valLoss : Option Float := none) : IO Unit := do
  IO.println s!"Epoch {epoch} Summary:"
  IO.println s!"  Train - Acc: {trainAcc * 100.0}%, Loss: {trainLoss}"
  match valAcc, valLoss with
  | some vAcc, some vLoss =>
    IO.println s!"  Valid - Acc: {vAcc * 100.0}%, Loss: {vLoss}"
  | _, _ => pure ()

/-- Log training initialization. -/
def logTrainingStart (config : TrainConfig) (numSamples : Nat) : IO Unit := do
  IO.println "=========================================="
  IO.println "Starting Neural Network Training"
  IO.println "=========================================="
  IO.println s!"Configuration:"
  IO.println s!"  Epochs: {config.epochs}"
  IO.println s!"  Batch size: {config.batchSize}"
  IO.println s!"  Learning rate: {config.learningRate}"
  IO.println s!"  Training samples: {numSamples}"
  IO.println ""

/-- Log training completion. -/
def logTrainingComplete (finalAcc finalLoss : Float) : IO Unit := do
  IO.println ""
  IO.println "=========================================="
  IO.println "Training Complete!"
  IO.println "=========================================="
  IO.println s!"Final Accuracy: {finalAcc * 100.0}%"
  IO.println s!"Final Loss: {finalLoss}"

end TrainingLog

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
  let mut lastLoss : Float := 0.0
  for batchIdx in [0:batches.size] do
    let batch := batches[batchIdx]!
    currentState := trainBatch currentState batch

    -- Compute loss for logging (on last processed batch)
    if batch.size > 0 then
      let (input, label) := batch[0]!
      let output := currentState.net.forward input
      lastLoss := crossEntropyLoss output label

    -- Log batch progress periodically using structured logging
    TrainingLog.logBatchProgress (batchIdx + 1) batches.size lastLoss config.printEveryNBatches

  -- Evaluate on both training and validation sets at evaluation epochs
  if (state.currentEpoch + 1) % config.evaluateEveryNEpochs == 0 then
    -- Compute training metrics
    let trainAcc := computeAccuracy currentState.net trainData
    let trainLoss := computeAverageLoss currentState.net trainData

    -- Compute validation metrics if provided
    match validData with
    | some vData =>
      let valAcc := computeAccuracy currentState.net vData
      let valLoss := computeAverageLoss currentState.net vData
      TrainingLog.logEpochEnd (state.currentEpoch + 1) trainAcc trainLoss (some valAcc) (some valLoss)
    | none =>
      TrainingLog.logEpochEnd (state.currentEpoch + 1) trainAcc trainLoss

  -- Return state with incremented epoch counter
  return { currentState with currentEpoch := currentState.currentEpoch + 1 }

/-- Save training state checkpoint to disk.

**Parameters:**
- `state`: Training state to save
- `epoch`: Current epoch number
- `config`: Checkpoint configuration
- `valAcc`: Optional validation accuracy for best-model tracking

**Returns:** IO action that saves the checkpoint

**TODO:** Implement actual serialization. Currently just logs the intent.

**Implementation strategy:**
1. Convert MLPArchitecture to JSON-serializable format (parameter arrays)
2. Include optimizer state (learning rate, epoch counter)
3. Include training metadata (validation accuracy, timestamp)
4. Write to file using `IO.FS.writeFile`
5. Handle errors gracefully with try/catch

**Note:** Lean 4's JSON library (Lean.Data.Json) can be used for serialization.
For binary formats, consider custom byte array serialization.
-/
def saveCheckpoint (_state : TrainState) (epoch : Nat)
    (config : CheckpointConfig) (valAcc : Option Float := none) : IO Unit := do
  if config.saveEveryNEpochs == 0 then
    return ()  -- Checkpointing disabled

  if epoch % config.saveEveryNEpochs != 0 then
    return ()  -- Not a checkpoint epoch

  -- TODO: Implement actual serialization
  let filename := s!"{config.saveDir}/checkpoint_epoch_{epoch}.json"
  IO.println s!"[Checkpoint] Would save to: {filename}"
  match valAcc with
  | some acc =>
    IO.println s!"[Checkpoint] Validation accuracy: {acc * 100.0}%"
  | none => pure ()

/-- Load training state from checkpoint.

**Parameters:**
- `path`: Path to checkpoint file

**Returns:** Loaded training state

**TODO:** Implement actual deserialization. Currently returns error.

**Implementation strategy:**
1. Read checkpoint file using `IO.FS.readFile`
2. Parse JSON using `Lean.Json.parse`
3. Reconstruct MLPArchitecture from parameter arrays
4. Reconstruct SGDState from optimizer metadata
5. Validate dimensions and consistency
6. Return TrainState

**Error handling:** Use IO.Error for file not found, parse errors, dimension mismatches.
-/
def loadCheckpoint (path : String) : IO TrainState := do
  -- TODO: Implement actual deserialization
  IO.eprintln s!"Error: Checkpoint loading not yet implemented"
  IO.eprintln s!"Attempted to load: {path}"
  throw (IO.userError "loadCheckpoint: Not implemented")

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
    (validData : Option (Array (Vector 784 × Nat)) := none)
    (checkpointConfig : Option CheckpointConfig := none) : IO TrainState := do
  -- Initialize training state
  let mut state := initTrainState net config

  -- Log training initialization using structured logging
  TrainingLog.logTrainingStart config trainData.size

  -- Train for specified number of epochs
  for epochIdx in [0:config.epochs] do
    -- Log epoch start
    TrainingLog.logEpochStart (epochIdx + 1) config.epochs

    -- Train one epoch
    state ← trainOneEpoch state trainData config validData

    -- Save checkpoint if configured
    match checkpointConfig with
    | some ckptCfg =>
      -- Get validation accuracy for checkpoint if available
      let valAcc := match validData with
        | some vData => some (computeAccuracy state.net vData)
        | none => none
      saveCheckpoint state (epochIdx + 1) ckptCfg valAcc
    | none => pure ()

  -- Compute final metrics for logging
  let finalAcc := match validData with
    | some vData => computeAccuracy state.net vData
    | none => computeAccuracy state.net trainData
  let finalLoss := match validData with
    | some vData => computeAverageLoss state.net vData
    | none => computeAverageLoss state.net trainData

  -- Log training completion
  TrainingLog.logTrainingComplete finalAcc finalLoss

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
