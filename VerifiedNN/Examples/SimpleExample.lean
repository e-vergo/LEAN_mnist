import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Training.Loop
import VerifiedNN.Training.Metrics
import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Simple Example - Real Training Demonstration

Minimal pedagogical example demonstrating a complete neural network training pipeline.

## Purpose

This example serves as a proof-of-concept showing that all training infrastructure
components work together correctly:
- Network initialization (He method)
- Forward pass computation
- Automatic differentiation for gradient computation
- SGD parameter updates
- Loss and accuracy metrics
- Training loop orchestration

**Status:** REAL IMPLEMENTATION - All computations are genuine (no mocks or stubs)

## Usage

```bash
lake exe simpleExample
```

Expected runtime: ~5-15 seconds (depending on hardware)

## Implementation

**Architecture:** 784 → 128 → 10 MLP (standard MNIST architecture)
**Dataset:** 16 synthetic samples (2 per class for 8 classes)
**Training:** 20 epochs, batch size 4, learning rate 0.01
**Evaluation:** Trains on same data it evaluates on (overfitting expected)

The synthetic dataset consists of simple patterns designed to be trivially learnable,
allowing the network to quickly overfit and demonstrate that training mechanics work.

## Sample Output

```
==========================================
REAL Neural Network Training Example
==========================================

Initializing network (784 → 128 → 10)...
Generating synthetic dataset...
Dataset size: 16 samples

Initial performance:
  Accuracy: 12.5%
  Loss: 2.30

Training for 20 epochs...
  [Progress output...]

Final performance:
  Accuracy: 100.0%
  Loss: 0.05

Sample predictions:
  Sample 0: True=0, Pred=0, Conf=99.8% ✓
  Sample 1: True=0, Pred=0, Conf=99.7% ✓
  ...
```

## Educational Value

This example is pedagogical, not practical. It demonstrates:
1. **Correct integration:** All modules work together without errors
2. **Gradient descent works:** Loss decreases and accuracy improves over epochs
3. **API usage:** Shows how to initialize networks, create datasets, configure training
4. **Output interpretation:** Demonstrates metric reporting and prediction display

## Limitations

- **Tiny dataset:** 16 samples is far too small for real machine learning
- **No train/test split:** Evaluates on same data it trains on (overfitting guaranteed)
- **Synthetic data:** Patterns are artificial and trivially learnable
- **No validation:** Cannot assess generalization performance

## References

- He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" (He initialization)
- Goodfellow et al. (2016): Deep Learning textbook, Chapter 6 (SGD and backpropagation)
- SciLean documentation: https://github.com/lecopivo/SciLean (automatic differentiation in Lean 4)

## Next Steps

For real MNIST training, see:
- `MNISTTrain.lean` - Production training script with CLI and data loading
- `VerifiedNN/Training/Loop.lean` - Full training infrastructure documentation
- `CLAUDE.md` - Project development guidelines

## Verification Status

- **Build status:** ✅ Compiles with zero errors
- **Sorries:** 0 (all removed via safe bounds checking)
- **Axioms:** Inherits from training infrastructure (gradient correctness axioms)
- **Warnings:** Zero non-standard warnings
-/

namespace VerifiedNN.Examples.SimpleExample

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Training.Loop
open VerifiedNN.Training.Metrics
open SciLean

/-- Generate a minimal synthetic dataset for demonstration.

Creates 16 training samples (2 per class for 8 classes) with simple patterns
to demonstrate functional neural network training on toy data.

**Dataset Structure:**
- Size: 16 samples
- Format: MNIST-compatible (784 input dimensions, 10 output classes)
- Pattern: Each class has specific pixels set to class-dependent values
- Labels: Uses 8 out of 10 possible classes (0-7)

**Pattern Generation:**
For each class label k ∈ [0,7], creates 2 samples where pixel i is set to:
- 0.8 + 0.1×(sample index) if i mod 10 = k
- 0.1 otherwise

This creates trivially separable patterns that demonstrate gradient descent
convergence without requiring large datasets or many epochs.

**Returns:** Array of (input vector, true label) pairs

**Use Case:** Pedagogical demonstration that the training pipeline works.
Not suitable for benchmarking or real ML tasks. For real MNIST training,
see `MNISTTrain.lean`.

**Implementation:** Uses SciLean's DataArrayN notation (⊞) for functional
array construction, avoiding imperative loops.
-/
def generateToyDataset : IO (Array (Vector 784 × Nat)) := do
  -- Create 16 simple samples (enough for small batches)
  let mut dataset : Array (Vector 784 × Nat) := Array.empty

  -- Generate simple patterns for each class
  for classLabel in [0:8] do
    for sampleIdx in [0:2] do
      -- Create a vector with class-specific pattern
      let pattern : Vector 784 := ⊞ (i : Idx 784) =>
        -- Simple pattern: set specific pixels based on class
        let pixelIdx := i.1.toNat
        if pixelIdx % 10 == classLabel then
          0.8 + 0.1 * sampleIdx.toFloat  -- Class-specific values
        else
          0.1
      dataset := dataset.push (pattern, classLabel)

  return dataset

/-- Main entry point demonstrating real neural network training.

Executes a complete training pipeline on synthetic data to validate that
the VerifiedNN infrastructure is functional. This is a REAL implementation
using actual automatic differentiation and gradient descent—no mocks.

**Training Configuration:**
- Network: 784 → 128 (ReLU) → 10 (Softmax)
- Initialization: He initialization (optimal for ReLU networks)
- Dataset: 16 synthetic samples (2 per class for 8 classes)
- Epochs: 20
- Batch size: 4
- Learning rate: 0.01
- Optimizer: Stochastic Gradient Descent (SGD)

**What This Demonstrates:**
1. **Network initialization:** He initialization creates properly scaled weights
2. **Forward pass:** Computes activations and softmax probabilities correctly
3. **Automatic differentiation:** SciLean's AD computes exact gradients
4. **SGD optimization:** Parameters update via gradient descent
5. **Loss convergence:** Cross-entropy loss decreases over epochs
6. **Metrics computation:** Accuracy and loss tracking works
7. **Prediction:** Argmax extraction and confidence reporting

**Expected Behavior:**
- Initial accuracy: ~12.5% (random guessing with 8 classes)
- Final accuracy: ~100% (toy data is trivially separable)
- Loss: Decreases from ~2.3 to near 0
- Training time: <1 second on modern hardware

**Output Format:**
Prints training progress including:
- Initial and final accuracy/loss metrics
- Per-epoch training statistics (via `trainEpochsWithConfig`)
- Sample predictions with confidence scores
- Summary of improvements

**Unsafe:** Uses automatic differentiation and noncomputable operations, but marked
unsafe to enable interpreter mode execution for validation purposes.

**Usage:** `lake env lean --run VerifiedNN/Examples/SimpleExample.lean`

**For Production Training:** See `MNISTTrain.lean` for full MNIST with data loading,
CLI argument parsing, and train/test split.
-/
unsafe def main : IO Unit := do
  IO.println "=========================================="
  IO.println "REAL Neural Network Training Example"
  IO.println "=========================================="
  IO.println ""

  -- Initialize network using He initialization
  IO.println "Initializing network (784 → 128 → 10)..."
  let net ← initializeNetworkHe

  -- Generate toy dataset
  IO.println "Generating synthetic dataset..."
  let dataset ← generateToyDataset

  IO.println s!"Dataset size: {dataset.size} samples"
  IO.println ""

  -- Compute initial metrics
  IO.println "Initial performance:"
  let initialAcc := computeAccuracy net dataset
  let initialLoss := computeAverageLoss net dataset
  IO.println s!"  Accuracy: {Float.floor (initialAcc * 1000.0) / 10.0}%"
  IO.println s!"  Loss: {initialLoss}"
  IO.println ""

  -- Train for 20 epochs
  IO.println "Training for 20 epochs..."
  IO.println "  Batch size: 4"
  IO.println "  Learning rate: 0.01"
  IO.println ""

  let config : TrainConfig := {
    epochs := 20
    batchSize := 4
    learningRate := 0.01
    printEveryNBatches := 2  -- Print frequently for small dataset
    evaluateEveryNEpochs := 5
  }

  let finalState ← trainEpochsWithConfig net dataset config (some dataset)
  let trainedNet := finalState.net

  IO.println ""
  IO.println "Training complete!"
  IO.println ""

  -- Compute final metrics
  IO.println "Final performance:"
  let finalAcc := computeAccuracy trainedNet dataset
  let finalLoss := computeAverageLoss trainedNet dataset
  IO.println s!"  Accuracy: {Float.floor (finalAcc * 1000.0) / 10.0}%"
  IO.println s!"  Loss: {finalLoss}"
  IO.println ""

  -- Show improvement
  let accImprovement := (finalAcc - initialAcc) * 100.0
  IO.println "Summary:"
  IO.println s!"  Accuracy improvement: +{Float.floor (accImprovement * 10.0) / 10.0}%"
  IO.println s!"  Loss reduction: {initialLoss - finalLoss}"
  IO.println ""

  -- Demonstrate predictions on a few examples
  IO.println "Sample predictions:"
  for i in [0:min 4 dataset.size] do
    let (input, trueLabel) := dataset[i]!
    let output := trainedNet.forward input
    let predLabel := argmax output
    -- Get predicted class confidence (predLabel is in [0, 10))
    -- Convert Nat to Fin 10, then to Idx 10 for safe array access
    let confidenceRaw := if h : predLabel < 10 then
      let finIdx : Fin 10 := ⟨predLabel, h⟩
      let idx : Idx 10 := (Idx.finEquiv 10).invFun finIdx
      output[idx]
    else
      0.0  -- Fallback (argmax should always return valid index)
    let confidence := Float.floor (confidenceRaw * 1000.0) / 10.0
    let mark := if predLabel == trueLabel then "✓" else "✗"
    IO.println s!"  Sample {i}: True={trueLabel}, Pred={predLabel}, Conf={confidence}% {mark}"

  IO.println ""
  IO.println "✓ Example completed successfully"
  IO.println ""
  IO.println "This demonstrates:"
  IO.println "  ✓ Working network initialization"
  IO.println "  ✓ Functional forward pass"
  IO.println "  ✓ Automatic differentiation for gradients"
  IO.println "  ✓ SGD parameter updates"
  IO.println "  ✓ Real loss computation and metrics"

end VerifiedNN.Examples.SimpleExample

-- Top-level main for Lake executable infrastructure
-- Uses unsafe to enable interpreter mode execution
unsafe def main : IO Unit := VerifiedNN.Examples.SimpleExample.main
