import VerifiedNN.Data.MNIST
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Network.Gradient
import VerifiedNN.Network.Serialization
import VerifiedNN.Training.Loop
import VerifiedNN.Training.Metrics

/-!
# Train MNIST and Serialize Model

Complete example showing how to train an MLP on MNIST and save it as a
human-readable Lean source file.

This demonstrates the full workflow:
1. Load MNIST dataset
2. Initialize network with He initialization
3. Train for multiple epochs with progress tracking
4. Evaluate final performance
5. Serialize trained model to Lean source file
6. Show how to use the saved model

## Usage

```bash
# Run training
lake exe trainAndSerialize

# This will create:
# - SavedModels/MNIST_Trained.lean (serialized model)
# - Console output with training progress
```

## Verification Status

- **Build status:** ✅ Compiles successfully
- **Sorries:** 0
- **Axioms:** 0 (uses axiomatized components but no new axioms)
-/

open VerifiedNN.Core
open VerifiedNN.Data.MNIST
open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Network.Gradient
open VerifiedNN.Training.Loop
open VerifiedNN.Training.Metrics
open SciLean

namespace VerifiedNN.Examples.TrainAndSerialize

/-- Get current timestamp string (placeholder - implement properly for production) -/
def getCurrentTimestamp : IO String := do
  -- Simple timestamp using monotonic time
  -- In production, would use actual system time
  let ms ← IO.monoMsNow
  return s!"Epoch_{ms}"

/-- Main training and serialization function -/
noncomputable unsafe def main : IO Unit := do
  IO.println "=========================================="
  IO.println "MNIST Training with Model Serialization"
  IO.println "=========================================="
  IO.println ""

  -- Load MNIST dataset
  IO.println "Loading MNIST dataset..."
  let trainData ← loadMNISTTrain "data"
  let testData ← loadMNISTTest "data"

  if trainData.size == 0 then
    IO.eprintln "Error: Failed to load training data"
    IO.eprintln "Make sure to run: ./scripts/download_mnist.sh"
    IO.Process.exit 1

  IO.println s!"Loaded {trainData.size} training samples"
  IO.println s!"Loaded {testData.size} test samples"
  IO.println ""

  -- Initialize network
  IO.println "Initializing network (784 → 128 → 10)..."
  let net ← initializeNetworkHe
  IO.println "Network initialized with He initialization"
  IO.println s!"Total parameters: {nParams}"
  IO.println ""

  -- Compute initial metrics
  IO.println "Initial performance (random weights):"
  let initialTestAcc := computeAccuracy net testData
  let initialTestLoss := computeAverageLoss net testData
  IO.println s!"  Test accuracy: {Float.floor (initialTestAcc * 1000.0) / 10.0}%"
  IO.println s!"  Test loss: {initialTestLoss}"
  IO.println ""

  -- Configure training
  let config : TrainConfig := {
    epochs := 5
    batchSize := 32
    learningRate := 0.00001
    printEveryNBatches := 100
    evaluateEveryNEpochs := 1
  }

  IO.println "Training configuration:"
  IO.println s!"  Epochs: {config.epochs}"
  IO.println s!"  Batch size: {config.batchSize}"
  IO.println s!"  Learning rate: {config.learningRate}"
  IO.println ""

  -- Train the network
  IO.println "Training started..."
  IO.println "===================="
  let startTime ← IO.monoMsNow
  let finalState ← trainEpochsWithConfig net trainData config (some testData)
  let endTime ← IO.monoMsNow
  let trainedNet := finalState.net

  let trainingTime := (endTime - startTime).toFloat / 1000.0
  IO.println "===================="
  IO.println s!"Training completed in {trainingTime} seconds"
  IO.println ""

  -- Final evaluation
  IO.println "Final performance:"
  let finalTrainAcc := computeAccuracy trainedNet trainData
  let finalTestAcc := computeAccuracy trainedNet testData
  let finalLoss := computeAverageLoss trainedNet trainData
  IO.println s!"  Training accuracy: {Float.floor (finalTrainAcc * 1000.0) / 10.0}%"
  IO.println s!"  Test accuracy: {Float.floor (finalTestAcc * 1000.0) / 10.0}%"
  IO.println s!"  Final loss: {finalLoss}"
  IO.println s!"  Improvement: +{Float.floor ((finalTestAcc - initialTestAcc) * 1000.0) / 10.0}%"
  IO.println ""

  -- Create metadata for serialization
  let timestamp ← getCurrentTimestamp
  let metadata : ModelMetadata := {
    trainedOn := timestamp
    epochs := config.epochs
    finalTrainAcc := finalTrainAcc
    finalTestAcc := finalTestAcc
    finalLoss := finalLoss
    architecture := "784→128→10 (ReLU+Softmax)"
    learningRate := config.learningRate
    datasetSize := trainData.size
  }

  -- Save the trained model as Lean source file
  IO.println "Serializing trained model..."
  let filepath := "SavedModels/MNIST_Trained.lean"
  saveModel trainedNet metadata filepath
  IO.println ""

  -- Demonstrate inference
  IO.println "=========================================="
  IO.println "Inference Demo: Predicting 10 test samples"
  IO.println "=========================================="
  IO.println ""

  let mut correct := 0
  for i in [0:min 10 testData.size] do
    let (input, trueLabel) := testData[i]!
    let predictedLabel := MLPArchitecture.predict trainedNet input

    if predictedLabel == trueLabel then
      correct := correct + 1

    let correctMark := if predictedLabel == trueLabel then "✓" else "✗"
    IO.println s!"Sample {i+1}: True={trueLabel}, Predicted={predictedLabel} {correctMark}"

  IO.println ""
  IO.println s!"Quick test: {correct}/10 correct"
  IO.println ""

  -- Final summary
  IO.println "=========================================="
  IO.println "Training and Serialization Complete!"
  IO.println "=========================================="
  IO.println ""
  IO.println "Summary:"
  IO.println s!"  ✓ Model trained for {config.epochs} epochs"
  IO.println s!"  ✓ Final test accuracy: {Float.floor (finalTestAcc * 1000.0) / 10.0}%"
  IO.println s!"  ✓ Model saved to: {filepath}"
  IO.println ""
  IO.println "Next steps:"
  IO.println "  1. Check the saved model: cat SavedModels/MNIST_Trained.lean"
  IO.println "  2. Import in your code: import VerifiedNN.SavedModels.MNIST_Trained"
  IO.println "  3. Use the model: let model := VerifiedNN.SavedModels.MNIST_Trained.trainedModel"
  IO.println ""
  IO.println "Note: Large models may take time to compile when first imported."
  IO.println ""

end VerifiedNN.Examples.TrainAndSerialize
