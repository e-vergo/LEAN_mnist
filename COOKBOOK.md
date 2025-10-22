# VerifiedNN Cookbook

Practical recipes and examples for common tasks in the VerifiedNN project.

## Table of Contents

1. [Data Operations](#data-operations)
2. [Network Construction](#network-construction)
3. [Training Recipes](#training-recipes)
4. [Verification Patterns](#verification-patterns)
5. [Testing Recipes](#testing-recipes)
6. [Debugging Techniques](#debugging-techniques)
7. [Performance Optimization](#performance-optimization)

---

## Data Operations

### Recipe 1: Load and Visualize MNIST Data

**Problem:** You want to load MNIST data and see sample images.

**Solution:**

```bash
# Download MNIST dataset
./scripts/download_mnist.sh

# Visualize 5 random test images
lake exe renderMNIST --count 5

# Visualize training set
lake exe renderMNIST --count 10 --train

# Inverted colors for light terminals
lake exe renderMNIST --count 3 --inverted
```

**Expected Output:**
```
Sample 0 | Ground Truth: 7
----------------------------

      :*++:.
      #%%%%%*********.
      :=:=+%%#%%%%%%%=
            : :::: %%-
                  :%#
                  %@:
```

### Recipe 2: Load MNIST Programmatically

**Problem:** Load MNIST data in your own Lean program.

**Solution:**

```lean
import VerifiedNN.Data.MNIST
import VerifiedNN.Data.Preprocessing

open VerifiedNN.Data

def loadAndPreprocessMNIST : IO (Array (Vector 784) × Array Nat) := do
  -- Load training images and labels
  let images ← loadMNISTImages "data/train-images-idx3-ubyte"
  let labels ← loadMNISTLabels "data/train-labels-idx1-ubyte"

  -- Normalize pixels to [0, 1]
  let normalizedImages := images.map normalize

  -- Verify data
  IO.println s!"Loaded {images.size} training samples"
  IO.println s!"First label: {labels[0]!}"

  return (normalizedImages, labels)
```

### Recipe 3: Create Mini-Batches

**Problem:** Split dataset into mini-batches for training.

**Solution:**

```lean
import VerifiedNN.Training.Batch

def batchDataExample : IO Unit := do
  -- Load data
  let (images, labels) ← loadAndPreprocessMNIST

  -- Create batches of size 32
  let batchSize := 32
  let batches := createBatches images labels batchSize

  IO.println s!"Created {batches.size} batches of size {batchSize}"

  -- Process first batch
  let (batchX, batchY) := batches[0]!
  IO.println s!"Batch shape: {batchX.size} samples"
```

### Recipe 4: Generate Synthetic Data for Testing

**Problem:** Create synthetic data for unit tests.

**Solution:**

```lean
def generateSyntheticData (numSamples : Nat) (inputDim : Nat) (numClasses : Nat)
    : Array (Vector inputDim) × Array Nat := Id.run do
  let mut inputs := #[]
  let mut labels := #[]

  for i in [0:numSamples] do
    -- Generate random input
    let input := ⊞ (j : Fin inputDim) =>
      Float.sin (i.toFloat + j.val.toFloat)

    -- Assign cyclic labels
    let label := i % numClasses

    inputs := inputs.push input
    labels := labels.push label

  return (inputs, labels)

-- Usage
def testWithSyntheticData := do
  let (inputs, labels) := generateSyntheticData 100 784 10
  IO.println s!"Generated {inputs.size} synthetic samples"
```

---

## Network Construction

### Recipe 5: Initialize a Standard MLP

**Problem:** Create and initialize a 2-layer MLP for MNIST.

**Solution:**

```lean
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization

def createMNISTNetwork : IO MLPArchitecture := do
  -- Define architecture (784 → 128 → 10)
  let arch := {
    inputDim := 784,
    hiddenDim := 128,
    outputDim := 10
  }

  -- Initialize with He initialization (best for ReLU)
  let net ← initializeNetwork arch

  IO.println "Network initialized:"
  IO.println s!"  Layer 1: {arch.inputDim} → {arch.hiddenDim} (ReLU)"
  IO.println s!"  Layer 2: {arch.hiddenDim} → {arch.outputDim} (Softmax)"

  return net
```

### Recipe 6: Custom Weight Initialization

**Problem:** Initialize weights with a custom strategy.

**Solution:**

```lean
import VerifiedNN.Layer.Dense

def initializeLayerCustom (inDim outDim : Nat) : IO (DenseLayer inDim outDim) := do
  -- Xavier/Glorot initialization: U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
  let bound := Float.sqrt (6.0 / (inDim.toFloat + outDim.toFloat))

  -- Initialize weights uniformly in [-bound, bound]
  let weights ← IO.rand bound (-bound) >>= fun _ =>
    return ⊞ (i : Fin outDim, j : Fin inDim) =>
      (IO.rand (-bound) bound).run'  -- Simplified for illustration

  -- Initialize biases to zero
  let bias := ⊞ (_ : Fin outDim) => 0.0

  return { weights := weights, bias := bias }
```

### Recipe 7: Forward Pass Through Network

**Problem:** Compute network output for a single input.

**Solution:**

```lean
def forwardPassExample : IO Unit := do
  -- Initialize network
  let net ← createMNISTNetwork

  -- Create sample input (all zeros)
  let input : Vector 784 := ⊞ _ => 0.5

  -- Forward pass
  let output := net.forwardPass input

  -- Get prediction
  let prediction := argmax output
  let confidence := output[prediction]

  IO.println s!"Prediction: {prediction}"
  IO.println s!"Confidence: {confidence}"
  IO.println s!"Output probabilities sum: {output.sum}"
```

### Recipe 8: Batch Forward Pass

**Problem:** Process multiple samples efficiently.

**Solution:**

```lean
def batchForwardPassExample : IO Unit := do
  let net ← createMNISTNetwork

  -- Create batch of 32 samples
  let batchSize := 32
  let batch : Batch batchSize 784 := ⊞ (i, j) =>
    Float.sin (i.val.toFloat + j.val.toFloat)

  -- Batch forward pass
  let outputs := net.forwardPassBatch batch

  -- Process predictions
  for i in [0:batchSize] do
    let sampleOutput := ⊞ j => outputs[i, j]
    let prediction := argmax sampleOutput
    IO.println s!"Sample {i}: Predicted class {prediction}"
```

---

## Training Recipes

### Recipe 9: Simple Training Loop

**Problem:** Train a network on synthetic data.

**Solution:**

```lean
import VerifiedNN.Training.Loop
import VerifiedNN.Optimizer.SGD

def simpleTrainingExample : IO Unit := do
  -- 1. Generate synthetic data
  let (trainData, trainLabels) := generateSyntheticData 100 784 10

  -- 2. Initialize network
  let net ← createMNISTNetwork

  -- 3. Training configuration
  let epochs := 10
  let batchSize := 16
  let learningRate := 0.01

  -- 4. Training loop
  for epoch in [0:epochs] do
    -- Train one epoch
    let updatedNet ← trainEpoch net trainData trainLabels batchSize learningRate

    -- Evaluate
    let (loss, acc) := evaluateFull updatedNet trainData trainLabels
    IO.println s!"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.2%}"

    net := updatedNet

  IO.println "Training complete!"
```

### Recipe 10: Training with Validation Set

**Problem:** Monitor validation metrics during training.

**Solution:**

```lean
def trainingWithValidation : IO Unit := do
  -- Load data
  let (trainData, trainLabels) ← loadAndPreprocessMNIST
  let (valData, valLabels) := splitDataset trainData trainLabels 0.2

  -- Initialize
  let net ← createMNISTNetwork

  -- Training loop with validation
  for epoch in [0:20] do
    -- Train
    let trainedNet ← trainEpoch net trainData trainLabels 32 0.01

    -- Evaluate on both sets
    let (trainLoss, trainAcc) := evaluateFull trainedNet trainData trainLabels
    let (valLoss, valAcc) := evaluateFull trainedNet valData valLabels

    IO.println s!"Epoch {epoch}:"
    IO.println s!"  Train: Loss={trainLoss:.4f}, Acc={trainAcc:.2%}"
    IO.println s!"  Val:   Loss={valLoss:.4f}, Acc={valAcc:.2%}"

    -- Early stopping (optional)
    if valAcc > 0.95 then
      IO.println "Target accuracy reached!"
      break

    net := trainedNet
```

### Recipe 11: Learning Rate Scheduling

**Problem:** Adjust learning rate during training.

**Solution:**

```lean
import VerifiedNN.Optimizer.LRSchedule

def trainingWithLRSchedule : IO Unit := do
  let net ← createMNISTNetwork
  let (trainData, trainLabels) ← loadAndPreprocessMNIST

  -- Initial learning rate
  let initialLR := 0.1

  for epoch in [0:30] do
    -- Step decay: halve LR every 10 epochs
    let lr := if epoch % 10 == 0 && epoch > 0 then
      initialLR * (0.5 ^ (epoch / 10))
    else
      initialLR

    IO.println s!"Epoch {epoch}: Learning rate = {lr}"

    -- Train with current LR
    let trainedNet ← trainEpoch net trainData trainLabels 32 lr
    net := trainedNet
```

### Recipe 12: Gradient Accumulation

**Problem:** Simulate large batch sizes with limited memory.

**Solution:**

```lean
def trainingWithGradientAccumulation : IO Unit := do
  let net ← createMNISTNetwork
  let (trainData, trainLabels) ← loadAndPreprocessMNIST

  let microBatchSize := 8
  let accumulationSteps := 4
  let effectiveBatchSize := microBatchSize * accumulationSteps  -- 32

  IO.println s!"Effective batch size: {effectiveBatchSize}"

  for epoch in [0:10] do
    let mut accumulatedGrad := zeroVector nParams

    -- Accumulate gradients over micro-batches
    for step in [0:accumulationSteps] do
      let microBatch := getMiniBatch trainData step microBatchSize
      let grad := computeGradient net microBatch
      accumulatedGrad := vadd accumulatedGrad grad

    -- Average accumulated gradient
    accumulatedGrad := smul (1.0 / accumulationSteps.toFloat) accumulatedGrad

    -- Update parameters
    let updatedNet := updateParameters net accumulatedGrad 0.01
    net := updatedNet

    IO.println s!"Epoch {epoch} complete"
```

---

## Verification Patterns

### Recipe 13: Verify Gradient Correctness

**Problem:** Check that automatic differentiation is correct.

**Solution:**

```lean
import VerifiedNN.Testing.GradientCheck

def verifyGradientCorrectness : IO Unit := do
  -- Define test function
  let f (x : Vector 3) : Float := x[0]^2 + 2.0 * x[1]^2 + 3.0 * x[2]^2

  -- Test point
  let x := ![1.0, 2.0, 3.0]

  -- Analytical gradient: [2x₀, 4x₁, 6x₂]
  let analyticalGrad := ![2.0 * x[0], 4.0 * x[1], 6.0 * x[2]]

  -- Numerical gradient
  let numericalGrad := finiteDifferenceGradient f x

  -- Compare
  if vectorApproxEq analyticalGrad numericalGrad 1e-5 then
    IO.println "✓ Gradient is correct"
  else
    IO.println "✗ Gradient mismatch detected"
    IO.println s!"Analytical: {analyticalGrad}"
    IO.println s!"Numerical: {numericalGrad}"
```

### Recipe 14: Prove a Simple Property

**Problem:** Formally prove that ReLU is non-negative.

**Solution:**

```lean
import VerifiedNN.Core.Activation
import Mathlib.Data.Real.Basic

-- Theorem statement
theorem relu_nonneg (x : ℝ) : relu x ≥ 0 := by
  unfold relu
  split_ifs
  · -- Case: x > 0
    linarith
  · -- Case: x ≤ 0
    linarith

-- Use in code
example : ∀ x : ℝ, relu x ≥ 0 := relu_nonneg
```

### Recipe 15: Type-Level Dimension Verification

**Problem:** Prove that layer composition maintains dimensions.

**Solution:**

```lean
import VerifiedNN.Layer.Dense

-- This is proven by construction via dependent types
theorem layer_composition_dimension_safe
  {d1 d2 d3 : Nat}
  (layer1 : DenseLayer d1 d2)
  (layer2 : DenseLayer d2 d3)
  (x : Vector d1) :
  (layer2.forward (layer1.forward x)).size = d3 := by
  -- Proof is trivial: type system guarantees this
  rfl

-- Using the theorem
def composedLayers : IO Unit := do
  let layer1 : DenseLayer 784 128 := ...
  let layer2 : DenseLayer 128 10 := ...
  let input : Vector 784 := ...

  -- This type-checks, so dimensions are correct by construction
  let output : Vector 10 := layer2.forward (layer1.forward input)
  IO.println s!"Output dimension: {output.size}"  -- Always 10
```

---

## Testing Recipes

### Recipe 16: Write a Unit Test

**Problem:** Test a single function.

**Solution:**

```lean
import LSpec
import VerifiedNN.Core.LinearAlgebra

open LSpec

def testVectorAddition := test "vector addition" $ do
  let v1 := ![1.0, 2.0, 3.0]
  let v2 := ![4.0, 5.0, 6.0]
  let result := vadd v1 v2
  let expected := ![5.0, 7.0, 9.0]

  check ("vadd computes correctly" :
    vectorApproxEq result expected 1e-7)

-- Run test
#eval testVectorAddition.run
```

### Recipe 17: Test Multiple Properties

**Problem:** Test several properties of a function.

**Solution:**

```lean
def testSoftmaxProperties := test "softmax properties" $ do
  let input := ![2.0, 1.0, 0.1]
  let output := softmax input

  -- Property 1: Outputs are non-negative
  for i in [0:3] do
    check (s!"softmax[{i}] ≥ 0" : output[i] ≥ 0.0)

  -- Property 2: Outputs sum to 1
  let sum := output[0] + output[1] + output[2]
  check ("softmax sums to 1" : approxEq sum 1.0 1e-6)

  -- Property 3: Outputs are in [0, 1]
  for i in [0:3] do
    check (s!"softmax[{i}] ≤ 1" : output[i] ≤ 1.0)

  -- Property 4: Largest input has largest output
  let argmaxInput := if input[0] > input[1] && input[0] > input[2] then 0
                     else if input[1] > input[2] then 1
                     else 2
  let argmaxOutput := if output[0] > output[1] && output[0] > output[2] then 0
                      else if output[1] > output[2] then 1
                      else 2
  check ("softmax preserves argmax" : argmaxInput == argmaxOutput)
```

---

## Debugging Techniques

### Recipe 18: Debug Training Issues

**Problem:** Loss is not decreasing during training.

**Diagnostic Checklist:**

```lean
def debugTraining : IO Unit := do
  let net ← createMNISTNetwork
  let (trainData, trainLabels) := generateSyntheticData 100 784 10

  -- Check 1: Data is normalized
  IO.println "=== Data Check ==="
  let sample := trainData[0]!
  let minVal := sample.min
  let maxVal := sample.max
  IO.println s!"Data range: [{minVal}, {maxVal}]"
  if minVal < -1.0 || maxVal > 2.0 then
    IO.println "⚠️ Data may need normalization"

  -- Check 2: Initial forward pass works
  IO.println "\n=== Forward Pass Check ==="
  let output := net.forwardPass sample
  IO.println s!"Output sum: {output.sum}"
  if Float.abs (output.sum - 1.0) > 0.01 then
    IO.println "⚠️ Softmax may not be summing to 1"

  -- Check 3: Gradient exists and is non-zero
  IO.println "\n=== Gradient Check ==="
  let grad := computeGradient net sample trainLabels[0]!
  let gradNorm := norm grad
  IO.println s!"Gradient norm: {gradNorm}"
  if gradNorm < 1e-8 then
    IO.println "⚠️ Gradient is too small (vanishing gradient?)"
  if gradNorm > 100.0 then
    IO.println "⚠️ Gradient is too large (exploding gradient?)"

  -- Check 4: Learning rate is reasonable
  IO.println "\n=== Learning Rate Check ==="
  let lr := 0.01
  let updateMagnitude := lr * gradNorm
  IO.println s!"Update magnitude: {updateMagnitude}"
  if updateMagnitude < 1e-6 then
    IO.println "⚠️ Updates too small, increase learning rate"
  if updateMagnitude > 10.0 then
    IO.println "⚠️ Updates too large, decrease learning rate"

  -- Check 5: Loss is being computed correctly
  IO.println "\n=== Loss Check ==="
  let loss := computeLoss net sample trainLabels[0]!
  IO.println s!"Initial loss: {loss}"
  if loss < 0.0 then
    IO.println "⚠️ Negative loss (bug in loss function)"
  if loss > 100.0 then
    IO.println "⚠️ Very large loss (numerical instability?)"
```

### Recipe 19: Debug Dimension Mismatches

**Problem:** Getting dimension mismatch errors.

**Solution:**

```lean
def debugDimensions : IO Unit := do
  IO.println "=== Network Architecture ==="

  let net ← createMNISTNetwork

  -- Print layer shapes
  IO.println s!"Layer 1:"
  IO.println s!"  Weights: {net.layer1.weights.shape}"
  IO.println s!"  Bias: {net.layer1.bias.size}"

  IO.println s!"Layer 2:"
  IO.println s!"  Weights: {net.layer2.weights.shape}"
  IO.println s!"  Bias: {net.layer2.bias.size}"

  -- Test forward pass dimensions
  IO.println "\n=== Forward Pass Dimensions ==="
  let input : Vector 784 := ⊞ _ => 1.0
  IO.println s!"Input: {input.size}"

  let h1 := net.layer1.forwardReLU input
  IO.println s!"Hidden: {h1.size}"

  let output := net.layer2.forward h1
  IO.println s!"Output: {output.size}"

  -- Check compatibility
  if net.layer1.bias.size != net.layer1.weights.nrows then
    IO.println "⚠️ Layer 1 bias dimension mismatch"

  if net.layer2.weights.ncols != net.layer1.weights.nrows then
    IO.println "⚠️ Layer 1→2 dimension incompatible"
```

---

## Performance Optimization

### Recipe 20: Profile Code Performance

**Problem:** Identify performance bottlenecks.

**Solution:**

```lean
-- Add to your Lean file
set_option profiler true
set_option trace.profiler.threshold 10  -- Show operations >10ms

def profileTraining : IO Unit := do
  IO.println "Profiling training performance..."

  let net ← createMNISTNetwork
  let (trainData, trainLabels) := generateSyntheticData 1000 784 10

  -- Profile forward pass
  let startForward ← IO.monoMsNow
  let _ := net.forwardPass trainData[0]!
  let endForward ← IO.monoMsNow
  IO.println s!"Forward pass: {endForward - startForward}ms"

  -- Profile gradient computation
  let startGrad ← IO.monoMsNow
  let _ := computeGradient net trainData[0]! trainLabels[0]!
  let endGrad ← IO.monoMsNow
  IO.println s!"Gradient computation: {endGrad - startGrad}ms"

  -- Profile batch processing
  let batchSize := 32
  let startBatch ← IO.monoMsNow
  for i in [0:batchSize] do
    let _ := net.forwardPass trainData[i]!
  let endBatch ← IO.monoMsNow
  IO.println s!"Batch ({batchSize} samples): {endBatch - startBatch}ms"
  IO.println s!"Average per sample: {(endBatch - startBatch) / batchSize}ms"
```

### Recipe 21: Optimize Memory Usage

**Problem:** Reduce memory consumption during training.

**Solution:**

```lean
-- Use smaller batches
def memoryEfficientTraining : IO Unit := do
  let net ← createMNISTNetwork
  let (trainData, trainLabels) ← loadAndPreprocessMNIST

  -- Small batch size to reduce memory
  let batchSize := 8  -- Instead of 32 or 64

  -- Process in chunks
  let numBatches := trainData.size / batchSize

  for epoch in [0:10] do
    for batchIdx in [0:numBatches] do
      -- Process batch
      let batchStart := batchIdx * batchSize
      let batchEnd := min (batchStart + batchSize) trainData.size

      let batch := trainData.extract batchStart batchEnd
      let labels := trainLabels.extract batchStart batchEnd

      -- Train on batch
      let updatedNet ← trainEpochBatch net batch labels 0.01
      net := updatedNet

      -- Clear intermediate results (if applicable)
      -- Lean's GC should handle this automatically

    IO.println s!"Epoch {epoch} complete"
```

---

## Quick Reference

### Common Function Signatures

```lean
-- Data loading
def loadMNISTImages (path : FilePath) : IO (Array (Vector 784))
def loadMNISTLabels (path : FilePath) : IO (Array Nat)

-- Network operations
def initializeNetwork (arch : MLPArchitecture) : IO MLPArchitecture
def forwardPass (net : MLPArchitecture) (x : Vector 784) : Vector 10
def computeGradient (net : MLPArchitecture) (x : Vector 784) (y : Nat) : Vector nParams

-- Training
def trainEpoch (net : MLPArchitecture) (data : Array (Vector 784))
               (labels : Array Nat) (batchSize : Nat) (lr : Float) : IO MLPArchitecture
def evaluateFull (net : MLPArchitecture) (data : Array (Vector 784))
                 (labels : Array Nat) : (Float × Float)  -- (loss, accuracy)

-- Testing
def finiteDifferenceGradient (f : Vector n → Float) (x : Vector n)
                              (ε : Float := 1e-5) : Vector n
def vectorApproxEq (v1 v2 : Vector n) (eps : Float := 1e-7) : Bool
```

### Common Patterns

**Pattern: Check if code compiles (type-level verification)**
```lean
-- If this type-checks, dimensions are correct
def verified_operation : Vector 10 :=
  let layer : DenseLayer 784 128 := ...
  let input : Vector 784 := ...
  let intermediate : Vector 128 := layer.forward input
  -- Type error if dimensions don't match
```

**Pattern: Approximate equality for floats**
```lean
-- Never use == for Float comparisons
if approxEq result expected 1e-6 then
  IO.println "Match!"
```

**Pattern: Iterate over array safely**
```lean
-- Use indexed loops to avoid out-of-bounds
for i in [0:array.size] do
  process array[i]!
```

---

## Common Pitfalls

❌ **Don't:** Use `==` for Float comparisons
✅ **Do:** Use `approxEq` with appropriate tolerance

❌ **Don't:** Assume gradient computation is computable
✅ **Do:** Accept that AD is noncomputable in SciLean

❌ **Don't:** Forget to normalize MNIST pixel values
✅ **Do:** Scale pixels to [0, 1] before training

❌ **Don't:** Use large learning rates (>0.1) without testing
✅ **Do:** Start with 0.01 and adjust based on loss behavior

❌ **Don't:** Assume dimensions match without type checking
✅ **Do:** Use dependent types to enforce dimensions at compile time

---

**Last Updated:** 2025-10-22
**Maintained by:** Project contributors
**More Examples:** See `VerifiedNN/Examples/` directory
