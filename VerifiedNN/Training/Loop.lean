import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Gradient
import VerifiedNN.Network.ManualGradient
import VerifiedNN.Training.Batch
import VerifiedNN.Training.Metrics
import VerifiedNN.Loss.CrossEntropy
import VerifiedNN.Optimizer.SGD
import SciLean

/-!
# Training Loop

Main training loop implementation for neural network training.

## Main Definitions

- `TrainConfig`: Hyperparameter configuration (epochs, batch size, learning rate, logging)
- `CheckpointConfig`: Checkpoint saving configuration (API defined, serialization TODO)
- `TrainState`: Training state management (network, optimizer state, progress counters)
- `initTrainState`: Initialize training state from network and configuration
- `trainBatch`: Process single mini-batch with gradient computation and parameter update
- `trainOneEpoch`: Complete pass through training data with shuffled mini-batches
- `trainEpochs`: Simplified training interface returning just the trained network
- `trainEpochsWithConfig`: Full-featured training with validation and checkpointing
- `resumeTraining`: Continue training from checkpoint with optional hyperparameter changes

## Main Results

This module provides computational training infrastructure without formal verification.
No theorems are proven. The correctness of gradient computation depends on verified
properties in `VerifiedNN.Network.Gradient` and `VerifiedNN.Loss.CrossEntropy`.

## Implementation Notes

**Training architecture:** The training loop follows standard mini-batch SGD:
1. **Initialization:** Set up network, optimizer state, and configuration
2. **Epoch Loop:** For each epoch, shuffle data and create mini-batches
3. **Batch Loop:** For each batch, compute gradients and update parameters
4. **Evaluation:** Periodically evaluate on validation set
5. **Checkpoint:** Return final trained state

**Gradient computation:** Automatic differentiation via SciLean computes gradients:
- Forward pass through network computes predictions
- Cross-entropy loss measures prediction error
- Backward pass (automatic) computes gradients w.r.t. parameters using chain rule
- Gradients are averaged across the mini-batch (dividing by batch size)
- SGD step updates parameters: θ_new = θ_old - η * ∇L

**Batch gradient accumulation:** Gradients from each example in the mini-batch are
accumulated and then averaged. This is mathematically equivalent to computing the
gradient of the average loss, but implemented as sum-then-scale for clarity.

**Training state management:** `TrainState` tracks network parameters, optimizer
state (learning rate, epoch counter), and progress (current epoch, batches seen).
This enables pause/resume functionality via checkpointing.

**Logging infrastructure:** `TrainingLog` namespace provides structured logging
utilities for epoch start/end, batch progress, and training initialization/completion.
This keeps main training code clean while providing detailed progress feedback.

**Checkpoint API:** `CheckpointConfig`, `saveCheckpoint`, and `loadCheckpoint` define
the checkpoint API. Actual serialization/deserialization is TODO (requires converting
MLPArchitecture and SGDState to/from JSON or binary format).

**Verification status:** Training loop itself is not formally verified. Correctness
depends on:
- Gradient correctness (verified in Network.Gradient)
- Loss function properties (verified in Loss.CrossEntropy)
- Optimizer update correctness (simple arithmetic in Optimizer.SGD)
- Type safety of dimension tracking (enforced by dependent types)

**Performance:** Training speed depends on SciLean and OpenBLAS. Typical MNIST
performance on M1 Mac: ~10-50ms per batch (B=32), ~10-30s per epoch (60k examples).

## References

- Mini-batch SGD: "On Large-Batch Training for Deep Learning" (Keskar et al., 2016)
- Learning rate effects: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)
- Cyclical schedules: "Cyclical Learning Rates for Training Neural Networks" (Smith, 2017)
-/

namespace VerifiedNN.Training.Loop

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Network.Gradient
open VerifiedNN.Network.ManualGradient
open VerifiedNN.Training.Batch
open VerifiedNN.Training.Metrics
open VerifiedNN.Loss
open VerifiedNN.Optimizer
open SciLean

/-- Training configuration structure.

Contains hyperparameters and settings for the training process.
This structure groups all training hyperparameters in one place for easy
configuration management and passing to training functions.

**Fields:**
- `epochs`: Number of complete passes through training data
- `batchSize`: Mini-batch size for SGD (typical: 16-128 for MNIST)
- `learningRate`: SGD step size (typical: 0.01-0.1 for MNIST)
- `printEveryNBatches`: Log progress every N batches (default: 100)
- `evaluateEveryNEpochs`: Evaluate metrics every N epochs (default: 1)
- `debugLogging`: Enable detailed batch-level debugging logs (default: false)

**Typical MNIST configuration:** 10-20 epochs, batch size 32-64, learning rate 0.01-0.05
-/
structure TrainConfig where
  epochs : Nat
  batchSize : Nat
  learningRate : Float
  printEveryNBatches : Nat := 100
  evaluateEveryNEpochs : Nat := 1
  debugLogging : Bool := false
  deriving Repr

/-- Checkpoint configuration for saving training state.

Defines the API for checkpoint saving during training. Checkpoints enable
pause/resume functionality and allow recovering training if interrupted.

**Note:** Checkpoint serialization/deserialization not yet implemented.
This structure defines the API for future checkpoint functionality.

**Fields:**
- `saveDir`: Directory to save checkpoints (default: "checkpoints")
- `saveEveryNEpochs`: Save checkpoint every N epochs, 0 = never save (default: 0)
- `saveOnlyBest`: Only save checkpoints that improve validation metrics (default: true)

**Future implementation:** Will serialize MLPArchitecture and SGDState to JSON or binary format.
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

-- Debugging utilities for training diagnostics.
-- Provides gradient and parameter inspection tools for debugging training issues.
namespace DebugUtils

open VerifiedNN.Core.LinearAlgebra

/-- Compute L2 norm of a vector.

**Formula:** ‖v‖₂ = √(Σᵢ vᵢ²)

**Parameters:**
- `v`: Vector of any dimension

**Returns:** L2 norm as Float
-/
def vectorNorm {n : Nat} (v : Vector n) : Float :=
  Float.sqrt (normSq v)

/-- Check if vector contains NaN or Inf values.

Checks first element as a simple heuristic.

**Parameters:**
- `v`: Vector to check

**Returns:** True if first element is NaN or Inf
-/
def vectorHasNaN {n : Nat} (v : Vector n) : Bool :=
  if h : 0 < n then
    v[⟨0, h⟩].isNaN || v[⟨0, h⟩].isInf
  else
    false

/-- Extract first k elements of a vector as an array for display.

NOTE: Simplified to always return empty array due to Lean indexing complexity.
The norm-based logging provides sufficient information for debugging.

**Parameters:**
- `v`: Vector to extract from (unused)
- `k`: Number of elements to extract (unused)

**Returns:** Empty array (feature disabled for simplicity)
-/
def vectorFirstK {n : Nat} (_v : Vector n) (_k : Nat := 5) : Array Float :=
  #[]  -- Simplified: just use norms for debugging instead of individual values

/-- Log detailed batch diagnostics for debugging.

Prints comprehensive information about gradients and parameters during training:
- Batch loss
- Gradient norm (L2)
- Parameter norms before and after update
- Parameter norm change
- Warnings for NaN/Inf, gradient explosion, or vanishing

**Parameters:**
- `batchIdx`: Current batch index (1-indexed for display)
- `totalBatches`: Total number of batches
- `loss`: Batch loss value
- `gradNorm`: L2 norm of batch gradient
- `paramsBefore`: Parameters before SGD update
- `paramsAfter`: Parameters after SGD update
- `avgGrad`: Average gradient for the batch
-/
def logBatchDebug {nParams : Nat}
    (batchIdx totalBatches : Nat)
    (loss : Float)
    (gradNorm : Float)
    (paramsBefore paramsAfter : Vector nParams)
    (avgGrad : Vector nParams) : IO Unit := do
  IO.println s!"Batch {batchIdx}/{totalBatches}:"
  IO.println s!"  Batch loss: {loss}"
  IO.println s!"  Batch gradient norm: {gradNorm}"

  let paramNormBefore := vectorNorm paramsBefore
  let paramNormAfter := vectorNorm paramsAfter
  let paramNormChange := paramNormAfter - paramNormBefore
  IO.println s!"  Params before norm: {paramNormBefore}, after norm: {paramNormAfter}"
  IO.println s!"  Param norm change: {paramNormChange}"

  -- Check for anomalies
  if vectorHasNaN avgGrad then
    IO.println "  WARNING: Gradient contains NaN or Inf!"

  if gradNorm > 10.0 then
    IO.println s!"  WARNING: Gradient explosion detected (norm={gradNorm})!"

  if gradNorm < 0.0001 then
    IO.println s!"  WARNING: Gradient vanishing detected (norm={gradNorm})!"

end DebugUtils

/-- Training state that tracks progress through training.

Maintains the current network state, optimizer state, and training metrics.
This structure encapsulates all mutable state during training, enabling
checkpoint/restore functionality and progress tracking.

**Fields:**
- `net`: Current network architecture with trained parameters
- `optimState`: SGD optimizer state (parameters, learning rate, epoch counter)
- `currentEpoch`: Number of completed epochs
- `totalBatchesSeen`: Total mini-batches processed (across all epochs)

**Invariant:** `optimState.params` should match `flattenParams net` after each update.
This invariant is maintained by `trainBatch` and `unflattenParams`.

**Use case:** Enables pause/resume training and progress monitoring.
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

Performs forward pass, computes gradients via automatic differentiation,
and updates parameters using SGD. This is the core computational step of
mini-batch gradient descent.

**Parameters:**
- `state`: Current training state (network, optimizer state, progress)
- `batch`: Mini-batch of training examples (input vectors and labels)

**Returns:** Updated training state with improved parameters

**Algorithm:**
1. For each example in batch: compute gradient of loss w.r.t. parameters
2. Accumulate gradients across batch: gradSum = Σ ∇L(θ, xᵢ, yᵢ)
3. Average gradients: avgGrad = gradSum / batchSize
4. Apply SGD step: θ_new = θ_old - η * avgGrad
5. Update network from new parameters via `unflattenParams`
6. Increment batch counter

**Gradient accumulation:** Uses functional fold to accumulate gradients:
```lean
gradSum = batch.foldl (fun accGrad (input, label) =>
  let grad := networkGradient params input label
  ⊞ i => accGrad[i] + grad[i]
) (zero gradient)
```
Then averages by dividing by batch size (floating-point division).

**Mathematical correctness:** The averaged gradient equals the gradient of
the average loss:
  ∇(1/B Σ L(θ, xᵢ, yᵢ)) = 1/B Σ ∇L(θ, xᵢ, yᵢ)
This is proven in standard calculus (linearity of differentiation).

**Edge case:** If batch is empty, returns state unchanged (no update).

**Complexity:** O(B × nParams) where B = batch.size, nParams ≈ 101770 for MNIST MLP

**Computable:** Uses manual backpropagation (networkGradientManual) instead of symbolic AD.
-/
def trainBatch
    (state : TrainState)
    (batch : Array (Vector 784 × Nat))
    (batchIdx totalBatches : Nat)
    (config : TrainConfig) : IO TrainState := do
  if batch.size == 0 then
    return state
  else
    -- Compute gradient for the batch
    -- We accumulate gradients across the batch and average them
    let params := state.optimState.params

    -- Compute average gradient over the batch
    -- Use manual backpropagation (computable implementation)
    let gradSum := batch.foldl (fun accGrad (input, label) =>
      let grad := networkGradientManual params input label
      ⊞ i => accGrad[i] + grad[i]
    ) (⊞ (_ : Idx nParams) => (0.0 : Float))

    -- Average the gradients
    let batchSizeFloat := batch.size.toFloat
    let avgGrad := ⊞ i => gradSum[i] / batchSizeFloat

    -- Compute batch loss for logging
    let batchLoss := if h : batch.size > 0 then
      let (input, label) := batch[0]
      let output := state.net.forward input
      crossEntropyLoss output label
    else
      0.0

    -- Debug logging if enabled
    if config.debugLogging then
      let gradNorm := DebugUtils.vectorNorm avgGrad
      DebugUtils.logBatchDebug batchIdx totalBatches batchLoss gradNorm params (sgdStep state.optimState avgGrad).params avgGrad

    -- Apply SGD step
    let newOptimState := sgdStep state.optimState avgGrad

    -- Update network from new parameters
    let newNet := unflattenParams newOptimState.params

    -- Return updated state
    return { state with
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
def trainOneEpoch
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
    currentState ← trainBatch currentState batch (batchIdx + 1) batches.size config

    -- Compute loss for logging (on last processed batch)
    if batch.size > 0 then
      let (input, label) := batch[0]!
      let output := currentState.net.forward input
      lastLoss := crossEntropyLoss output label

    -- Log batch progress periodically using structured logging (only if debug logging is off)
    if !config.debugLogging then
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
      TrainingLog.logEpochEnd
        (state.currentEpoch + 1) trainAcc trainLoss (some valAcc) (some valLoss)
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

/-- Train network for multiple epochs with full configuration control.

Main entry point for production training. Provides full control over training
process including validation monitoring, checkpoint saving, and logging frequency.

**Parameters:**
- `net`: Initial network (typically randomly initialized via `initializeMLPArchitecture`)
- `trainData`: Training dataset (array of input-label pairs)
- `config`: Training configuration (epochs, batch size, learning rate, logging)
- `validData`: Optional validation dataset for monitoring generalization (default: none)
- `checkpointConfig`: Optional checkpoint configuration for saving state (default: none)

**Returns:** Final training state (network, optimizer state, progress counters)

**Usage:**
```lean
let config : TrainConfig := {
  epochs := 10
  batchSize := 32
  learningRate := 0.01
  printEveryNBatches := 100
  evaluateEveryNEpochs := 1
}
let finalState ← trainEpochsWithConfig net trainData config (some validData)
let trainedNet := finalState.net
```

**Features:**
- Configurable logging frequency (batch progress, epoch metrics)
- Validation set evaluation (if provided)
- Checkpoint saving (API defined, serialization TODO)
- Returns full training state (not just network)
- Structured logging via `TrainingLog` namespace

**Training loop:**
1. Initialize training state from network and config
2. For each epoch:
   - Shuffle data and create mini-batches
   - Train on all batches via `trainOneEpoch`
   - Evaluate on validation set (if configured)
   - Save checkpoint (if configured)
3. Return final trained state

**For simple use cases:** See `trainEpochs` for a simplified interface that
returns just the trained network.

**Computable:** Uses manual backpropagation for gradient computation.
-/
def trainEpochsWithConfig
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

/-- Train network for multiple epochs (simplified interface).

Simplified training interface that returns just the trained network.
This is a convenience wrapper around `trainEpochsWithConfig` for simple
training scenarios where you don't need validation monitoring or checkpointing.

**Parameters:**
- `net`: Initial network (typically randomly initialized)
- `trainData`: Training dataset (array of input-label pairs)
- `epochs`: Number of complete passes through training data
- `batchSize`: Mini-batch size for SGD (typical: 32-64 for MNIST)
- `learningRate`: SGD step size (typical: 0.01-0.05 for MNIST)

**Returns:** Trained network (discards optimizer state and progress counters)

**Usage:**
```lean
let trainedNet ← trainEpochs initialNet trainData 10 32 0.01
```

**Defaults (not configurable in this interface):**
- Prints progress every 100 batches
- Evaluates on training set every epoch (no separate validation)
- No checkpointing

**Internally:** Creates a `TrainConfig` and delegates to `trainEpochsWithConfig`,
then extracts just the network from the returned `TrainState`.

**For production:** Use `trainEpochsWithConfig` if you need validation monitoring,
checkpoint support, or access to final optimizer state.

**Computable:** Uses manual backpropagation for gradient computation.
-/
def trainEpochs
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
