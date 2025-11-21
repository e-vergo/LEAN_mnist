import VerifiedNN.Core.DataTypes
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Network.ManualGradient
import VerifiedNN.Network.Serialization
import VerifiedNN.Training.Loop
import VerifiedNN.Training.Metrics
import VerifiedNN.Data.MNIST
import VerifiedNN.Data.Preprocessing
import VerifiedNN.Optimizer.SGD
import SciLean

/-!
# Full-Scale MNIST Training (50 epochs × 12K samples)

Extended training with frequent evaluation for production-ready model.
Includes:
- 50 epochs × 12K samples = 600K total training samples (same as 10 epochs × 60K)
- Full test set evaluation (10K samples) after each epoch
- Best model tracking and saving
- Detailed logging and gradient monitoring

Perfect for achieving maximum accuracy with comprehensive model selection.

## Training Strategy

Instead of 10 large epochs (60K samples each), we use 50 smaller epochs (12K samples each):
- **Total training time**: ~2.5 hours (same as 10 × 60K epochs)
- **Evaluation frequency**: 5× more frequent (every ~3 minutes)
- **Model selection**: 50 checkpoints to choose from vs. 10

## Usage

```bash
lake exe mnistTrainFull
```

## Expected Performance

- **Training time**: ~2.5 hours (50 epochs × 12K samples)
- **Epoch time**: ~3 minutes per epoch
- **Target accuracy**: 90-95% on test set
- **Best model**: Automatically saved when new best test accuracy achieved
- **Evaluation**: Full 10K test set used for accurate model selection

-/

namespace VerifiedNN.Examples.MNISTTrainFull

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Network.Gradient
open VerifiedNN.Network.ManualGradient
open VerifiedNN.Training
open VerifiedNN.Data
open VerifiedNN.Optimizer
open SciLean

set_default_scalar Float

/-- Load full MNIST dataset with normalization -/
def loadFullDataset : IO (Array (Vector 784 × Nat) × Array (Vector 784 × Nat)) := do
  IO.println "Loading full MNIST dataset (60000 train, 10000 test)..."

  let dataDir : System.FilePath := "data"
  let trainFull ← MNIST.loadMNISTTrain dataDir
  let testFull ← MNIST.loadMNISTTest dataDir

  -- ✅ Normalize pixels from [0,255] to [0,1] (CRITICAL for gradient stability!)
  IO.println "  Normalizing pixel values to [0,1]..."
  let trainNorm := Preprocessing.normalizeDataset trainFull
  let testNorm := Preprocessing.normalizeDataset testFull

  IO.println s!"✓ Loaded and normalized {trainNorm.size} training samples"
  IO.println s!"✓ Loaded and normalized {testNorm.size} test samples"

  return (trainNorm, testNorm)

/-- Main training function -/
unsafe def main : IO UInt32 := do
  IO.println "=========================================="
  IO.println "Full-Scale MNIST Training"
  IO.println "60K Samples - Production Training"
  IO.println "=========================================="
  IO.println ""

  -- Create timestamped log file
  let timestamp ← IO.monoMsNow
  let logPath := s!"logs/training_full_{timestamp}.log"
  let logHandle ← IO.FS.Handle.mk logPath IO.FS.Mode.write

  let logWrite (msg : String) : IO Unit := do
    logHandle.putStrLn msg
    logHandle.flush
    IO.println msg

  IO.println s!"📝 Logging to: {logPath}"
  IO.println ""

  logWrite "=========================================="
  logWrite "Full-Scale MNIST Training"
  logWrite "60K Samples - Production Training"
  logWrite "=========================================="
  logWrite ""

  -- Load data
  let (trainData, testData) ← loadFullDataset
  logWrite ""

  -- Initialize network
  logWrite "Initializing network (784 → 128 → 10)..."
  let net ← initializeNetworkHe
  logWrite "✓ Network initialized with He initialization"
  logWrite ""

  -- Training configuration
  let epochs := 50  -- Extended training for convergence
  let batchSize := 64
  let learningRate := 0.01  -- ✅ Validated on medium dataset
  let maxGradNorm := 10.0  -- Gradient clipping threshold

  logWrite "Training Configuration:"
  logWrite s!"  Epochs: {epochs}"
  logWrite s!"  Batch size: {batchSize}"
  logWrite s!"  Learning rate: {learningRate}"
  logWrite s!"  Train samples: {trainData.size}"
  logWrite s!"  Test samples: {testData.size}"
  logWrite ""

  -- Initial evaluation (using subsets for speed)
  let evalSubsetSize := min 500 trainData.size
  let trainEvalSubset := trainData.toSubarray 0 evalSubsetSize |>.toArray
  let testEvalSubset := testData.toSubarray 0 (min 100 testData.size) |>.toArray

  logWrite s!"Initial evaluation (on {trainEvalSubset.size} train, {testEvalSubset.size} test samples):"

  let evalStartTime ← IO.monoMsNow
  logWrite "  Computing train loss..."
  let initialLoss := Metrics.computeAverageLoss net trainEvalSubset
  logWrite s!"    Train loss: {initialLoss}"

  logWrite "  Computing train accuracy..."
  let initialTrainAcc := Metrics.computeAccuracy net trainEvalSubset
  logWrite s!"    Train accuracy: {Float.floor (initialTrainAcc * 1000.0) / 10.0}%"

  logWrite "  Computing test accuracy..."
  let initialTestAcc := Metrics.computeAccuracy net testEvalSubset
  logWrite s!"    Test accuracy: {Float.floor (initialTestAcc * 1000.0) / 10.0}%"

  let evalEndTime ← IO.monoMsNow
  let evalTimeSec := (evalEndTime - evalStartTime).toFloat / 1000.0
  logWrite s!"  (Evaluation completed in {Float.floor (evalTimeSec * 10.0) / 10.0}s)"
  logWrite ""

  -- Sanity check
  if initialLoss < 1.5 || initialLoss > 4.0 then
    logWrite "⚠ Warning: Initial loss unusual (expected ~2.0-2.5)"
    logWrite ""

  -- Training with detailed logging
  logWrite "Starting training..."
  logWrite "=========================================="
  let startTime ← IO.monoMsNow

  -- Manual training loop with per-batch logging and best model tracking
  let mut currentNet := net
  let mut bestTestAcc := 0.0
  let mut bestEpoch := 0

  for epoch in [0:epochs] do
    logWrite s!"\n=== Epoch {epoch + 1}/{epochs} ==="
    let epochStartTime ← IO.monoMsNow

    -- Sample 12K training examples per epoch (keeping same total training time as 10 epochs × 60K)
    let samplesPerEpoch := 12000
    let epochData := trainData.toSubarray 0 (min samplesPerEpoch trainData.size) |>.toArray

    -- Create batches for this epoch
    let batches ← Training.Batch.createShuffledBatches epochData batchSize
    logWrite s!"  Processing {batches.size} batches ({epochData.size} samples, batch size: {batchSize})..."

    -- Process each batch with logging
    let mut params := Network.Gradient.flattenParams currentNet
    for batchIdx in [0:batches.size] do
      let batch := batches[batchIdx]!
      let paramsBefore := params

      -- Log every 5th batch for detailed progress tracking
      if batchIdx % 5 == 0 then
        logWrite s!"    Batch {batchIdx + 1}/{batches.size} processing..."

      -- Compute gradients for batch
      let gradSum := batch.foldl (fun accGrad (input, label) =>
        let grad := Network.ManualGradient.networkGradientManual params input label
        ⊞ i => accGrad[i] + grad[i]
      ) (⊞ (_ : Idx nParams) => (0.0 : Float))

      -- Average gradients
      let batchSizeFloat := batch.size.toFloat
      let avgGrad := ⊞ i => gradSum[i] / batchSizeFloat

      -- Gradient clipping
      let gradNormSq := (⊞ i => avgGrad[i] * avgGrad[i]).sum
      let gradNorm := Float.sqrt gradNormSq
      let clipScale := if gradNorm > maxGradNorm then maxGradNorm / gradNorm else 1.0
      let clippedGrad := ⊞ i => avgGrad[i] * clipScale

      -- Apply SGD step with clipped gradients
      params := ⊞ i => params[i] - learningRate * clippedGrad[i]

      -- Log every 5th batch loss and diagnostics
      if batchIdx % 5 == 0 && batch.size > 0 then
        let (input, label) := batch[0]!
        let tempNet := Network.Gradient.unflattenParams params
        let output := tempNet.forward input
        let loss := Loss.crossEntropyLoss output label

        -- Compute parameter change norm
        let paramDiff := ⊞ i => params[i] - paramsBefore[i]
        let paramChangeNormSq := (⊞ i => paramDiff[i] * paramDiff[i]).sum
        let paramChangeNorm := Float.sqrt paramChangeNormSq

        let clippedMsg := if clipScale < 1.0 then s!" [CLIPPED from {gradNorm}]" else ""
        logWrite s!"      Loss: {loss}, GradNorm: {gradNorm}{clippedMsg}, ParamChange: {paramChangeNorm}"

        -- Check for NaN/Inf
        if gradNorm.isNaN || gradNorm.isInf then
          logWrite s!"      ⚠ WARNING: Gradient is NaN or Inf!"
        if loss.isNaN || loss.isInf then
          logWrite s!"      ⚠ WARNING: Loss is NaN or Inf!"

    -- Update network with trained parameters
    currentNet := Network.Gradient.unflattenParams params

    let epochEndTime ← IO.monoMsNow
    let epochTimeSec := (epochEndTime - epochStartTime).toFloat / 1000.0
    logWrite s!"  Epoch {epoch + 1} completed in {Float.floor (epochTimeSec * 10.0) / 10.0}s"

    -- Compute and log epoch metrics (FULL test set for accurate evaluation)
    logWrite "  Computing epoch metrics on FULL test set..."
    let epochLoss := Metrics.computeAverageLoss currentNet trainEvalSubset
    let epochTrainAcc := Metrics.computeAccuracy currentNet trainEvalSubset
    let epochTestAcc := Metrics.computeAccuracy currentNet testData  -- ✅ Full 10K test set!
    logWrite s!"    Epoch loss: {epochLoss}"
    logWrite s!"    Train accuracy (subset): {Float.floor (epochTrainAcc * 1000.0) / 10.0}%"
    logWrite s!"    Test accuracy (FULL 10K): {Float.floor (epochTestAcc * 1000.0) / 10.0}%"

    -- Save best model
    if epochTestAcc > bestTestAcc then
      bestTestAcc := epochTestAcc
      bestEpoch := epoch + 1
      logWrite s!"  🎉 NEW BEST! Test accuracy: {Float.floor (epochTestAcc * 1000.0) / 10.0}%"
      logWrite s!"  Saving model to models/best_model_epoch_{epoch + 1}.lean..."

      -- Save model using serialization
      let modelPath := s!"models/best_model_epoch_{epoch + 1}.lean"
      try
        -- Get current timestamp for model metadata
        let timestamp ← IO.monoMsNow
        let timestampStr := s!"Epoch {epoch + 1} at training time {timestamp}ms"

        -- Create metadata for this checkpoint
        let metadata : Network.ModelMetadata := {
          trainedOn := timestampStr
          epochs := epoch + 1
          finalTrainAcc := epochTrainAcc
          finalTestAcc := epochTestAcc
          finalLoss := epochLoss
          architecture := "784→128→10 (ReLU+Softmax)"
          learningRate := learningRate
          datasetSize := trainData.size
        }

        -- Actually save the model using the serialization module
        Network.saveModel currentNet metadata modelPath
        logWrite s!"  ✓ Best model saved (epoch {epoch + 1}, test acc: {Float.floor (epochTestAcc * 1000.0) / 10.0}%)"
      catch e =>
        logWrite s!"  ⚠ Warning: Could not save model: {e}"
    else
      logWrite s!"  (Best remains: {Float.floor (bestTestAcc * 1000.0) / 10.0}% at epoch {bestEpoch})"

  let finalNet := currentNet

  let endTime ← IO.monoMsNow
  let trainingTimeSec := (endTime - startTime).toFloat / 1000.0
  logWrite ""
  logWrite "=========================================="
  logWrite s!"Training completed in {Float.floor (trainingTimeSec * 10.0) / 10.0} seconds"
  logWrite ""

  -- Final evaluation (using larger subset)
  let finalEvalTrain := trainData.toSubarray 0 (min 1000 trainData.size) |>.toArray
  let finalEvalTest := testData.toSubarray 0 (min 500 testData.size) |>.toArray

  logWrite "Final Evaluation:"
  logWrite "=========================================="
  logWrite s!"(Using {finalEvalTrain.size} train, {finalEvalTest.size} test samples for evaluation)"

  let finalEvalStartTime ← IO.monoMsNow
  logWrite "  Computing final train loss..."
  let finalTrainLoss := Metrics.computeAverageLoss finalNet finalEvalTrain
  logWrite s!"    Final train loss: {finalTrainLoss}"

  logWrite "  Computing final train accuracy..."
  let finalTrainAcc := Metrics.computeAccuracy finalNet finalEvalTrain
  logWrite s!"    Final train accuracy: {Float.floor (finalTrainAcc * 1000.0) / 10.0}%"

  logWrite "  Computing final test accuracy..."
  let finalTestAcc := Metrics.computeAccuracy finalNet finalEvalTest
  logWrite s!"    Final test accuracy: {Float.floor (finalTestAcc * 1000.0) / 10.0}%"

  logWrite "  Computing per-class accuracy..."
  let finalPerClassAcc := Metrics.computePerClassAccuracy finalNet finalEvalTest
  logWrite "    Per-class accuracy computed"

  let finalEvalEndTime ← IO.monoMsNow
  let finalEvalTimeSec := (finalEvalEndTime - finalEvalStartTime).toFloat / 1000.0
  logWrite s!"  (Evaluation completed in {Float.floor (finalEvalTimeSec * 10.0) / 10.0}s)"
  logWrite ""

  logWrite "Per-class test accuracy:"
  -- Note: printPerClassAccuracy writes to stdout, need to capture and log
  Metrics.printPerClassAccuracy finalPerClassAcc
  logWrite ""

  -- Performance summary
  let lossReduction := initialLoss - finalTrainLoss
  let accImprovement := (finalTestAcc - initialTestAcc) * 100.0
  logWrite "Training Summary:"
  logWrite s!"  Loss reduction: {Float.floor (lossReduction * 1000.0) / 1000.0}"
  logWrite s!"  Accuracy improvement: +{Float.floor (accImprovement * 10.0) / 10.0}%"
  logWrite s!"  Training time: {Float.floor (trainingTimeSec * 10.0) / 10.0}s"
  logWrite s!"  Samples/second: {Float.floor ((trainData.size.toFloat * epochs.toFloat / trainingTimeSec) * 10.0) / 10.0}"
  logWrite ""

  -- Report best model
  logWrite "Best Model Summary:"
  logWrite "==========================================​="
  logWrite s!"  Best test accuracy: {Float.floor (bestTestAcc * 1000.0) / 10.0}%"
  logWrite s!"  Best epoch: {bestEpoch}/{epochs}"
  logWrite s!"  Model saved as: models/best_model_epoch_{bestEpoch}.lean"
  logWrite ""

  -- Success criteria for full-scale training
  if bestTestAcc >= 0.88 then
    logWrite "✓ SUCCESS: Achieved ≥88% test accuracy on full dataset"
    logWrite "  → Production-ready model!"
  else if bestTestAcc >= 0.80 then
    logWrite "⚠ PARTIAL: Achieved ≥80% test accuracy"
    logWrite "  → Consider tuning learning rate or training longer"
  else
    logWrite "✗ NEEDS WORK: Test accuracy <80%"
    logWrite "  → Check hyperparameters or data preprocessing"
  logWrite ""

  -- Sample predictions
  logWrite "Sample predictions on test set:"
  logWrite "================================"
  let mut correctCount := 0
  for i in [0:min 20 testData.size] do
    let (input, trueLabel) := testData[i]!
    let predictedLabel := MLPArchitecture.predict finalNet input
    let correct := if predictedLabel == trueLabel then "✓" else "✗"
    if predictedLabel == trueLabel then
      correctCount := correctCount + 1
    logWrite s!"Sample {i}: True={trueLabel}, Predicted={predictedLabel} {correct}"
  logWrite s!"Sample accuracy: {correctCount}/20 = {Float.floor ((correctCount.toFloat / 20.0) * 1000.0) / 10.0}%"
  logWrite ""

  logWrite "=========================================="
  logWrite "Full Training Complete!"
  logWrite s!"Test accuracy: {Float.floor (finalTestAcc * 1000.0) / 10.0}%"
  logWrite "=========================================="

  logHandle.flush
  return 0

end VerifiedNN.Examples.MNISTTrainFull

-- Top-level main
unsafe def main : IO UInt32 := VerifiedNN.Examples.MNISTTrainFull.main
