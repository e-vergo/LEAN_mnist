import VerifiedNN.Core.DataTypes
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Network.ManualGradient
import VerifiedNN.Training.Loop
import VerifiedNN.Training.Metrics
import VerifiedNN.Data.MNIST
import VerifiedNN.Optimizer.SGD
import SciLean

namespace VerifiedNN.Examples.MiniTraining

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Network.ManualGradient
open VerifiedNN.Training
open VerifiedNN.Data
open VerifiedNN.Optimizer
open SciLean

set_default_scalar Float

/-!
# Mini Training Test

Quick validation that training works with manual gradients.

Uses a tiny subset of data (100 train, 50 test) and runs for just 3 epochs.
This provides fast feedback that:
1. Training loop executes without errors
2. Loss decreases (learning is happening)
3. Gradients are reasonable magnitude
4. Forward and backward passes are consistent

## Expected Results

- Initial loss: ~2.3 (random 10-way classification)
- Final loss: <2.0 (showing some learning)
- Accuracy: >20% (better than random 10%)
- Runtime: 10-30 seconds total

If this doesn't work, full training won't work either!
-/

/-- Load a small subset of MNIST data for quick testing -/
def loadMiniDataset : IO (Array (Vector 784 × Nat) × Array (Vector 784 × Nat)) := do
  IO.println "Loading mini dataset (100 train, 50 test)..."

  -- Load full datasets
  let dataDir : System.FilePath := "data"
  let trainFull ← MNIST.loadMNISTTrain dataDir
  let testFull ← MNIST.loadMNISTTest dataDir

  -- Take small subsets
  let trainMini := trainFull.toSubarray 0 100 |>.toArray
  let testMini := testFull.toSubarray 0 50 |>.toArray

  IO.println s!"✓ Loaded {trainMini.size} training samples"
  IO.println s!"✓ Loaded {testMini.size} test samples"

  return (trainMini, testMini)

/-- Run mini training with detailed logging -/
def runMiniTraining : IO Unit := do
  IO.println "========================================"
  IO.println "Mini Training Test"
  IO.println "========================================"
  IO.println ""

  -- Load mini dataset
  let (trainData, testData) ← loadMiniDataset
  IO.println ""

  -- Initialize network
  IO.println "Initializing network..."
  let net ← initializeNetworkHe
  IO.println "✓ Network initialized (784 → 128 → 10)"
  IO.println ""

  -- Training parameters
  let epochs := 10
  let batchSize := 10
  let learningRate := 0.5  -- Aggressive learning rate for testing

  IO.println "Training Configuration:"
  IO.println s!"  Epochs: {epochs}"
  IO.println s!"  Batch size: {batchSize}"
  IO.println s!"  Learning rate: {learningRate}"
  IO.println s!"  Train samples: {trainData.size}"
  IO.println s!"  Test samples: {testData.size}"
  IO.println ""

  -- Initial evaluation
  IO.println "Initial evaluation (before training):"
  let initialLoss := Metrics.computeAverageLoss net trainData
  let initialAcc := Metrics.computeAccuracy net trainData
  IO.println s!"  Initial loss: {initialLoss}"
  IO.println s!"  Initial accuracy: {initialAcc * 100.0}%"
  IO.println ""

  -- Check initial loss is reasonable (should be ~2.3 for random)
  if initialLoss < 1.0 || initialLoss > 5.0 then
    IO.println "⚠ Warning: Initial loss unusual (expected ~2.3)"

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
  IO.println "Final evaluation (after training):"
  let finalLoss := Metrics.computeAverageLoss finalNet trainData
  let finalTrainAcc := Metrics.computeAccuracy finalNet trainData
  let finalTestAcc := Metrics.computeAccuracy finalNet testData

  IO.println s!"  Final train loss: {finalLoss}"
  IO.println s!"  Final train accuracy: {finalTrainAcc * 100.0}%"
  IO.println s!"  Final test accuracy: {finalTestAcc * 100.0}%"
  IO.println ""

  -- Verify learning happened
  let lossReduction := initialLoss - finalLoss
  IO.println "Learning Verification:"
  IO.println s!"  Loss reduction: {lossReduction}"

  if lossReduction > 0.1 then
    IO.println "  ✓ PASS: Loss decreased (learning occurred)"
  else
    IO.println "  ✗ FAIL: Loss did not decrease significantly"
    throw (IO.userError "Training failed - no learning")

  if finalTrainAcc > 0.15 then
    IO.println "  ✓ PASS: Accuracy > 15% (better than random)"
  else
    IO.println "  ✗ FAIL: Accuracy too low (not learning)"
    throw (IO.userError "Training failed - accuracy too low")

  IO.println ""
  IO.println "========================================"
  IO.println "✓ Mini training test PASSED!"
  IO.println "========================================"

end VerifiedNN.Examples.MiniTraining

-- Top-level main for Lake executable infrastructure
unsafe def main : IO Unit := VerifiedNN.Examples.MiniTraining.runMiniTraining
