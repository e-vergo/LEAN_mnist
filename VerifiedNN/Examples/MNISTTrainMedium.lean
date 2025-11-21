import VerifiedNN.Core.DataTypes
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Network.Gradient
import VerifiedNN.Network.ManualGradient
import VerifiedNN.Training.Loop
import VerifiedNN.Training.Metrics
import VerifiedNN.Data.MNIST
import VerifiedNN.Data.Preprocessing
import VerifiedNN.Optimizer.SGD
import SciLean

/-!
# Medium-Scale MNIST Training (5K samples)

Fast training on 5,000 samples for hyperparameter tuning and validation.
This provides a good balance between:
- Fast iteration (< 1 minute training time)
- Representative performance (enough data to see real learning)
- Quick feedback for tuning hyperparameters

Perfect for testing different learning rates, batch sizes, and epochs
before committing to full 60K training.

## Usage

```bash
lake exe mnistTrainMedium
```

## Expected Performance

- **Training time**: 30-60 seconds
- **Target accuracy**: 75-85% on test set
- **Loss curve**: Should decrease from ~2.3 to ~0.5-0.8

-/

namespace VerifiedNN.Examples.MNISTTrainMedium

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

/-- Load medium-sized MNIST subset for fast training -/
def loadMediumDataset : IO (Array (Vector 784 × Nat) × Array (Vector 784 × Nat)) := do
  IO.println "Loading medium dataset (5000 train, 1000 test)..."

  let dataDir : System.FilePath := "data"
  let trainFull ← MNIST.loadMNISTTrain dataDir
  let testFull ← MNIST.loadMNISTTest dataDir

  -- Take medium-sized subsets
  let trainMedium := trainFull.toSubarray 0 (min 5000 trainFull.size) |>.toArray
  let testMedium := testFull.toSubarray 0 (min 1000 testFull.size) |>.toArray

  -- ✅ Normalize pixels from [0,255] to [0,1] (CRITICAL for gradient stability!)
  IO.println "  Normalizing pixel values to [0,1]..."
  let trainNorm := Preprocessing.normalizeDataset trainMedium
  let testNorm := Preprocessing.normalizeDataset testMedium

  IO.println s!"✓ Loaded and normalized {trainNorm.size} training samples"
  IO.println s!"✓ Loaded and normalized {testNorm.size} test samples"

  return (trainNorm, testNorm)

/-- Main training function -/
unsafe def main : IO UInt32 := do
  IO.println "=========================================="
  IO.println "Medium-Scale MNIST Training"
  IO.println "5K Samples - Fast Hyperparameter Tuning"
  IO.println "=========================================="
  IO.println ""

  -- Create timestamped log file
  let timestamp ← IO.monoMsNow
  let logPath := s!"logs/training_{timestamp}.log"
  let logHandle ← IO.FS.Handle.mk logPath IO.FS.Mode.write

  let logWrite (msg : String) : IO Unit := do
    logHandle.putStrLn msg
    logHandle.flush
    IO.println msg

  IO.println s!"📝 Logging to: {logPath}"
  IO.println ""

  logWrite "=========================================="
  logWrite "Medium-Scale MNIST Training"
  logWrite "5K Samples - Fast Hyperparameter Tuning"
  logWrite "=========================================="
  logWrite ""

  -- Load data
  let (trainData, testData) ← loadMediumDataset
  logWrite ""

  -- Initialize network
  logWrite "Initializing network (784 → 128 → 10)..."
  let net ← initializeNetworkHe
  logWrite "✓ Network initialized with He initialization"
  logWrite ""

  -- Training configuration
  let epochs := 10
  let batchSize := 64
  let learningRate := 0.01  -- ✅ Increased after fixing data normalization
  let maxGradNorm := 10.0  -- Gradient clipping threshold

  logWrite "Training Configuration:"
  logWrite s!"  Epochs: {epochs}"
  logWrite s!"  Batch size: {batchSize}"
  logWrite s!"  Learning rate: {learningRate}"
  logWrite s!"  Train samples: {trainData.size}"
  logWrite s!"  Test samples: {testData.size}"
  logWrite ""

  -- Initial evaluation (using subsets for speed)
  let evalSubsetSize := min 50 trainData.size
  let trainEvalSubset := trainData.toSubarray 0 evalSubsetSize |>.toArray
  let testEvalSubset := testData.toSubarray 0 (min 20 testData.size) |>.toArray

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

  -- Manual training loop with per-batch logging
  let mut currentNet := net
  let numBatches := (trainData.size + batchSize - 1) / batchSize

  for epoch in [0:epochs] do
    logWrite s!"\n=== Epoch {epoch + 1}/{epochs} ==="
    let epochStartTime ← IO.monoMsNow

    -- Create batches for this epoch
    let batches ← Training.Batch.createShuffledBatches trainData batchSize
    logWrite s!"  Processing {batches.size} batches (batch size: {batchSize})..."

    -- Process each batch with logging
    let mut params := Network.Gradient.flattenParams currentNet
    for batchIdx in [0:batches.size] do
      let batch := batches[batchIdx]!
      let paramsBefore := params

      -- Log every 10th batch
      if batchIdx % 10 == 0 then
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

      -- Log every 10th batch loss and diagnostics
      if batchIdx % 10 == 0 && batch.size > 0 then
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

    -- Compute and log epoch metrics (on small subset for speed)
    logWrite "  Computing epoch metrics..."
    let epochLoss := Metrics.computeAverageLoss currentNet trainEvalSubset
    let epochTrainAcc := Metrics.computeAccuracy currentNet trainEvalSubset
    let epochTestAcc := Metrics.computeAccuracy currentNet testEvalSubset
    logWrite s!"    Epoch loss: {epochLoss}"
    logWrite s!"    Train accuracy: {Float.floor (epochTrainAcc * 1000.0) / 10.0}%"
    logWrite s!"    Test accuracy: {Float.floor (epochTestAcc * 1000.0) / 10.0}%"

  let finalNet := currentNet

  let endTime ← IO.monoMsNow
  let trainingTimeSec := (endTime - startTime).toFloat / 1000.0
  logWrite ""
  logWrite "=========================================="
  logWrite s!"Training completed in {Float.floor (trainingTimeSec * 10.0) / 10.0} seconds"
  logWrite ""

  -- Final evaluation (using subsets for speed)
  logWrite "Final Evaluation:"
  logWrite "=========================================="
  logWrite s!"(Using {trainEvalSubset.size} train, {testEvalSubset.size} test samples for fast evaluation)"

  let finalEvalStartTime ← IO.monoMsNow
  logWrite "  Computing final train loss..."
  let finalTrainLoss := Metrics.computeAverageLoss finalNet trainEvalSubset
  logWrite s!"    Final train loss: {finalTrainLoss}"

  logWrite "  Computing final train accuracy..."
  let finalTrainAcc := Metrics.computeAccuracy finalNet trainEvalSubset
  logWrite s!"    Final train accuracy: {Float.floor (finalTrainAcc * 1000.0) / 10.0}%"

  logWrite "  Computing final test accuracy..."
  let finalTestAcc := Metrics.computeAccuracy finalNet testEvalSubset
  logWrite s!"    Final test accuracy: {Float.floor (finalTestAcc * 1000.0) / 10.0}%"

  logWrite "  Computing per-class accuracy..."
  let finalPerClassAcc := Metrics.computePerClassAccuracy finalNet testEvalSubset
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

  -- Success criteria for medium-scale training
  if finalTestAcc >= 0.75 then
    logWrite "✓ SUCCESS: Achieved ≥75% test accuracy on medium dataset"
    logWrite "  → Ready to scale to full 60K training"
  else if finalTestAcc >= 0.60 then
    logWrite "⚠ PARTIAL: Achieved ≥60% test accuracy"
    logWrite "  → Consider tuning learning rate or epochs"
  else
    logWrite "✗ NEEDS WORK: Test accuracy <60%"
    logWrite "  → Check hyperparameters, initialization, or gradients"
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
  logWrite "Medium Training Complete!"
  logWrite s!"Test accuracy: {Float.floor (finalTestAcc * 1000.0) / 10.0}%"
  logWrite "=========================================="

  logHandle.flush
  return 0

end VerifiedNN.Examples.MNISTTrainMedium

-- Top-level main
unsafe def main : IO UInt32 := VerifiedNN.Examples.MNISTTrainMedium.main
