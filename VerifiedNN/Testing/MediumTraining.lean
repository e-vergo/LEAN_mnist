import VerifiedNN.Core.DataTypes
import VerifiedNN.Data.MNIST
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Training.Loop
import VerifiedNN.Training.Metrics

/-!
# Medium-Scale Training Validation

Tests training on 1,000 samples for 5 epochs to validate the learning rate fix
before proceeding to full 60K sample training.

## Purpose

Validates that the corrected learning rate (0.001) produces stable, effective
learning on a medium-scale dataset. This acts as a checkpoint before committing
to expensive full-scale training.

## Expected Results

With learning rate = 0.001:
- Epoch 1: loss ~2.0 → ~1.5, accuracy 30% → 50%
- Epoch 2: loss ~1.5 → ~1.2, accuracy 50% → 60%
- Epoch 3: loss ~1.2 → ~1.0, accuracy 60% → 68%
- Epoch 4: loss ~1.0 → ~0.9, accuracy 68% → 72%
- Epoch 5: loss ~0.9 → ~0.8, accuracy 72% → 75%

## Validation Criteria

1. Loss decreases from initial to final
2. Loss improves by >50%
3. Final accuracy >70%
4. No NaN/Inf in parameters

## References

- Agent 13 debugging: Identified lr=0.01 bug, fixed to lr=0.001
- Agent 14 validation: This test validates fix at medium scale
-/

namespace VerifiedNN.Testing.MediumTraining

open VerifiedNN.Core
open VerifiedNN.Data
open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Training

/-- Load medium-scale dataset (1,000 training samples).

Takes the first 1,000 samples from the full MNIST training set.
This provides a manageable dataset for validating the learning rate fix
before committing to full 60K sample training.

**Returns:** Array of 1,000 (image, label) pairs
-/
def loadMediumDataset : IO (Array (Vector 784 × Nat)) := do
  IO.println "Loading medium dataset (1,000 samples)..."

  -- Load full training dataset
  let dataDir : System.FilePath := "data"
  let trainFull ← MNIST.loadMNISTTrain dataDir

  -- Take first 1,000 samples
  let mediumData := trainFull.toSubarray 0 1000 |>.toArray

  IO.println s!"✓ Loaded {mediumData.size} training samples"

  return mediumData

/-- Run medium-scale training validation.

Trains on 1,000 samples for 5 epochs with fixed learning rate 0.001.
Validates that training is stable and effective before scaling to full dataset.

**Success Criteria:**
1. Loss decreases (initial > final)
2. Loss improves by >50%
3. Final accuracy >70%
4. No NaN/Inf in parameters
-/
def runMediumTraining : IO Unit := do
  IO.println "=========================================="
  IO.println "Medium-Scale Training Validation"
  IO.println "=========================================="
  IO.println ""

  -- Load medium dataset
  let trainData ← loadMediumDataset
  IO.println ""

  -- Initialize network
  IO.println "Initializing network..."
  let net ← initializeNetworkHe
  IO.println "✓ Network initialized (784 → 128 → 64 → 10)"
  IO.println ""

  -- Training configuration
  let epochs := 5
  let batchSize := 32
  let learningRate := 0.001  -- FIXED: was 0.01 (10x too high)

  IO.println "Training Configuration:"
  IO.println s!"  Samples: {trainData.size}"
  IO.println s!"  Epochs: {epochs}"
  IO.println s!"  Batch size: {batchSize}"
  IO.println s!"  Learning rate: {learningRate}"
  IO.println ""

  -- Initial evaluation
  IO.println "Initial Evaluation (before training):"
  let initialLoss := Metrics.computeAverageLoss net trainData
  let initialAcc := Metrics.computeAccuracy net trainData
  IO.println s!"  Initial loss: {initialLoss}"
  IO.println s!"  Initial accuracy: {initialAcc * 100.0}%"
  IO.println ""

  -- Train!
  IO.println "Starting training..."
  IO.println "========================================\n"

  let startTime ← IO.monoMsNow
  let finalNet ← Loop.trainEpochs net trainData epochs batchSize learningRate
  let endTime ← IO.monoMsNow
  let duration := (endTime - startTime).toFloat / 1000.0

  IO.println "\n========================================"
  IO.println "Training Complete!"
  IO.println s!"Total time: {duration}s"
  IO.println ""

  -- Final evaluation
  IO.println "Final Evaluation (after training):"
  let finalLoss := Metrics.computeAverageLoss finalNet trainData
  let finalAcc := Metrics.computeAccuracy finalNet trainData
  IO.println s!"  Final loss: {finalLoss}"
  IO.println s!"  Final accuracy: {finalAcc * 100.0}%"
  IO.println ""

  -- Compute improvements
  let lossReduction := initialLoss - finalLoss
  let lossImprovement := (lossReduction / initialLoss) * 100.0
  let accImprovement := (finalAcc - initialAcc) * 100.0

  IO.println "Improvement:"
  IO.println s!"  Loss reduction: {lossReduction} ({lossImprovement}%)"
  IO.println s!"  Accuracy gain: +{accImprovement} percentage points"
  IO.println ""

  -- Validation criteria
  IO.println "=========================================="
  IO.println "Validation Results"
  IO.println "=========================================="

  let mut allPassed := true

  -- Test 1: Loss decreased
  if finalLoss < initialLoss then
    IO.println "✓ Test 1: Loss decreased"
  else
    IO.println "✗ Test 1: Loss did not decrease!"
    allPassed := false

  -- Test 2: Loss improved by >50%
  if lossImprovement > 50.0 then
    IO.println s!"✓ Test 2: Loss improved by {lossImprovement}% (>50% target)"
  else
    IO.println s!"✗ Test 2: Loss improved only {lossImprovement}% (<50% target)"
    allPassed := false

  -- Test 3: Final accuracy >70%
  if finalAcc > 0.70 then
    IO.println s!"✓ Test 3: Final accuracy {finalAcc * 100.0}% (>70% target)"
  else
    IO.println s!"✗ Test 3: Final accuracy {finalAcc * 100.0}% (<70% target)"
    allPassed := false

  -- Test 4: Network is reasonable (not NaN)
  -- Check via forward pass on first example and checking if loss is finite
  if finalLoss.isNaN || finalLoss.isInf then
    IO.println "✗ Test 4: NaN/Inf detected in final loss!"
    allPassed := false
  else
    IO.println "✓ Test 4: Final loss is finite (no NaN/Inf)"

  IO.println ""
  if allPassed then
    IO.println "=========================================="
    IO.println "✅ ALL VALIDATION CRITERIA PASSED"
    IO.println "=========================================="
    IO.println "Ready to proceed to full-scale training!"
    IO.println ""
  else
    IO.println "=========================================="
    IO.println "❌ VALIDATION FAILED"
    IO.println "=========================================="
    IO.println "Further debugging needed before full training."
    IO.println ""
    throw (IO.userError "Medium-scale validation failed")

end VerifiedNN.Testing.MediumTraining

-- Top-level main for Lake executable infrastructure
unsafe def main : IO Unit := VerifiedNN.Testing.MediumTraining.runMediumTraining
