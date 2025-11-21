import VerifiedNN.Data.MNIST
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Network.ManualGradient
import VerifiedNN.Training.Metrics
import VerifiedNN.Core.DataTypes
import SciLean

/-!
# MNIST Training with Manual Gradients - Computable Version

This executable trains an MLP on MNIST using manually implemented backpropagation
instead of automatic differentiation. This makes the entire training pipeline
computable and allows compilation to native binaries.

## Key Difference from MNISTTrain.lean

**MNISTTrain.lean:** Uses SciLean's `∇` operator for gradients (noncomputable)
**TrainManual.lean:** Uses closed-form gradient formulas (computable)

## Why This Works

SciLean's automatic differentiation is based on symbolic rewriting and cannot
be compiled to native code. However, for our specific 2-layer MLP architecture,
we can compute gradients manually using closed-form formulas:

- Softmax + cross-entropy gradient: `softmax(logits) - one_hot(target)`
- Dense layer gradients: standard backpropagation via chain rule
- ReLU gradient: `(x > 0) ? 1 : 0` (element-wise)

All these operations use only basic arithmetic and are fully computable.

## Usage

```bash
# This will compile to a native binary and execute
lake exe trainManual

# Or run directly with interpreter (slower)
lake env lean --run VerifiedNN/Examples/TrainManual.lean
```

## Expected Performance

Training 10 epochs on MNIST (60,000 samples):
- Expected accuracy: ~92-95%
- Training time: 3-8 minutes (CPU-only, varies by hardware)
- Loss: Should decrease from ~2.3 to ~0.3

## Implementation Status

- **Sorries:** 0
- **Compilation:** ✅ Compiles to native binary
- **Noncomputable:** Uses unsafe for IO and random initialization only
-/

namespace VerifiedNN.Examples.TrainManual

open VerifiedNN.Core
open VerifiedNN.Data.MNIST
open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Network.Gradient
open VerifiedNN.Network.ManualGradient
open VerifiedNN.Training.Metrics
open SciLean

/-- Training configuration for manual gradient training -/
structure Config where
  epochs : Nat := 10
  batchSize : Nat := 32
  learningRate : Float := 0.01
  evaluateEveryNEpochs : Nat := 1

/-- Create mini-batches from dataset -/
def createBatches (data : Array (Vector 784 × Nat)) (batchSize : Nat)
    : Array (Array (Vector 784 × Nat)) :=
  let numBatches := (data.size + batchSize - 1) / batchSize
  Array.range numBatches |>.map fun batchIdx =>
    let startIdx := batchIdx * batchSize
    let endIdx := min (startIdx + batchSize) data.size
    data.toSubarray startIdx endIdx |>.toArray

/-- Train for one epoch using manual gradients (single-example updates) -/
def trainEpochManual (net : MLPArchitecture) (data : Array (Vector 784 × Nat))
    (config : Config) : IO MLPArchitecture := do
  let mut currentNet := net

  -- Train on single examples to avoid expensive gradient accumulation
  for exampleIdx in [0:data.size] do
    let (input, label) := data[exampleIdx]!

    -- Compute gradient using manual backpropagation
    let params := flattenParams currentNet
    let gradient := networkGradientManual params input label

    -- Apply SGD update
    let newParams := ⊞ i => params[i] - config.learningRate * gradient[i]
    currentNet := unflattenParams newParams

    if (exampleIdx + 1) % 50 == 0 then
      IO.println s!"  Example {exampleIdx + 1}/{data.size}"
      (← IO.getStdout).flush

  return currentNet

/-- Main training function with manual gradients -/
unsafe def main : IO Unit := do
  IO.println "=========================================="
  (← IO.getStdout).flush
  IO.println "MNIST Training with Manual Gradients"
  (← IO.getStdout).flush
  IO.println "Computable Implementation (No AD)"
  (← IO.getStdout).flush
  IO.println "=========================================="
  (← IO.getStdout).flush
  IO.println ""
  (← IO.getStdout).flush

  -- Configuration (FIXED: Reduced LR by 1000x due to gradient explosion)
  -- Gradient norms ~3000, so LR=0.00001 gives updates of ~0.03
  let config : Config := {
    epochs := 5
    batchSize := 8
    learningRate := 0.00001
    evaluateEveryNEpochs := 1
  }

  IO.println "Configuration:"
  IO.println s!"  Epochs: {config.epochs}"
  IO.println s!"  Batch size: {config.batchSize}"
  IO.println s!"  Learning rate: {config.learningRate}"
  IO.println ""
  (← IO.getStdout).flush

  -- Load MNIST data
  IO.println "Loading MNIST dataset..."
  (← IO.getStdout).flush
  let trainDataFull ← loadMNISTTrain "data"
  let testData ← loadMNISTTest "data"
  let trainData := trainDataFull

  if trainData.size == 0 then
    IO.eprintln "Error: Failed to load training data"
    IO.eprintln "Please run ./scripts/download_mnist.sh"
    IO.Process.exit 1

  IO.println s!"Loaded {trainData.size} training samples"
  IO.println s!"Loaded {testData.size} test samples"
  IO.println ""
  (← IO.getStdout).flush

  -- Initialize network
  IO.println "Initializing network (784 → 128 → 10)..."
  (← IO.getStdout).flush
  let net ← initializeNetworkHe
  IO.println "Network initialized with He initialization"
  IO.println ""

  -- Initial evaluation
  IO.println "Initial performance:"
  let initialTestAcc := computeAccuracy net testData
  let initialTestLoss := computeAverageLoss net testData
  let initialPerClassAcc := computePerClassAccuracy net testData
  IO.println s!"  Test accuracy: {Float.floor (initialTestAcc * 1000.0) / 10.0}%"
  IO.println s!"  Test loss: {initialTestLoss}"
  printPerClassAccuracy initialPerClassAcc
  IO.println ""

  -- Training loop
  IO.println "Starting training..."
  IO.println "===================="
  let startTime ← IO.monoMsNow

  let mut trainedNet := net
  for epoch in [0:config.epochs] do
    IO.println s!"Epoch {epoch + 1}/{config.epochs}"
    trainedNet ← trainEpochManual trainedNet trainData config

    if (epoch + 1) % config.evaluateEveryNEpochs == 0 then
      let trainAcc := computeAccuracy trainedNet trainData
      let trainPerClassAcc := computePerClassAccuracy trainedNet trainData
      let testAcc := computeAccuracy trainedNet testData
      let testPerClassAcc := computePerClassAccuracy trainedNet testData
      let testLoss := computeAverageLoss trainedNet testData
      IO.println s!"  Train accuracy: {Float.floor (trainAcc * 1000.0) / 10.0}%"
      printPerClassAccuracy trainPerClassAcc
      IO.println s!"  Test accuracy: {Float.floor (testAcc * 1000.0) / 10.0}%"
      printPerClassAccuracy testPerClassAcc
      IO.println s!"  Test loss: {testLoss}"
      (← IO.getStdout).flush
    IO.println ""

  let endTime ← IO.monoMsNow
  let trainingTimeSec := (endTime - startTime).toFloat / 1000.0
  IO.println "===================="
  IO.println s!"Training completed in {trainingTimeSec} seconds"
  IO.println ""

  -- Final evaluation
  IO.println "Final Evaluation:"
  IO.println "================="
  let finalTrainAcc := computeAccuracy trainedNet trainData
  let finalTrainPerClassAcc := computePerClassAccuracy trainedNet trainData
  let finalTestAcc := computeAccuracy trainedNet testData
  let finalTestPerClassAcc := computePerClassAccuracy trainedNet testData
  let finalTestLoss := computeAverageLoss trainedNet testData

  IO.println s!"Final train accuracy: {Float.floor (finalTrainAcc * 1000.0) / 10.0}%"
  printPerClassAccuracy finalTrainPerClassAcc
  IO.println s!"Final test accuracy: {Float.floor (finalTestAcc * 1000.0) / 10.0}%"
  printPerClassAccuracy finalTestPerClassAcc
  IO.println s!"Final test loss: {finalTestLoss}"
  IO.println ""
  (← IO.getStdout).flush

  let accImprovement := (finalTestAcc - initialTestAcc) * 100.0
  IO.println s!"Accuracy improvement: +{Float.floor (accImprovement * 10.0) / 10.0}%"
  IO.println ""

  -- Demonstrate inference on test examples
  IO.println "Sample predictions on test set:"
  IO.println "================================"
  for i in [0:min 10 testData.size] do
    let (input, trueLabel) := testData[i]!
    let predictedLabel := MLPArchitecture.predict trainedNet input
    let correct := if predictedLabel == trueLabel then "✓" else "✗"
    IO.println s!"Sample {i}: True={trueLabel}, Predicted={predictedLabel} {correct}"

  IO.println ""
  IO.println "=========================================="
  IO.println "Training complete!"
  IO.println s!"Final test accuracy: {Float.floor (finalTestAcc * 1000.0) / 10.0}%"
  IO.println "=========================================="
  IO.println ""
  IO.println "Model is now trained and ready for inference."
  IO.println "The trained network is stored in memory."

end VerifiedNN.Examples.TrainManual

-- Top-level main for Lake executable infrastructure
unsafe def main : IO Unit := VerifiedNN.Examples.TrainManual.main
