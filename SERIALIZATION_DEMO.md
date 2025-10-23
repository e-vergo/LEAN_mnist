# Model Serialization Demo

Quick demonstration of the model serialization feature.

## What Was Implemented

A complete system for saving trained neural networks as human-readable Lean source files that can be imported and used in other programs.

## Quick Demo

### 1. Run the Serialization Example

```bash
# Create a sample model with random initialization
lake build VerifiedNN.Examples.SerializationExample

# This will:
# - Initialize a network with Xavier initialization
# - Create example metadata
# - Save to SavedModels/Example_MNIST.lean
# - Print usage instructions
```

### 2. Inspect the Generated File

```bash
# View the saved model (first 50 lines)
head -50 SavedModels/Example_MNIST.lean

# You'll see:
# - Import statements
# - Training metadata in docstring
# - Weight and bias definitions
# - Assembled network definition
```

### 3. Use the Saved Model

Create a new file that imports the saved model:

```lean
import VerifiedNN.SavedModels.Example_MNIST

def testSavedModel (input : Vector 784) : Nat :=
  VerifiedNN.SavedModels.Example_MNIST.trainedModel.predict input
```

## Full Training Example

### Run Training with Serialization

```bash
# Train on MNIST and save the model
lake build VerifiedNN.Examples.TrainAndSerialize

# This will:
# - Load MNIST dataset
# - Train for 5 epochs
# - Evaluate performance
# - Save trained model to SavedModels/MNIST_Trained.lean
# - Show usage instructions
```

### Check Training Results

The saved file contains all training metadata:

```lean
/-!
# Trained MNIST Model: Epoch_12345678

## Training Configuration
- Architecture: 784→128→10 (ReLU+Softmax)
- Epochs: 5
- Learning rate: 0.00001
- Final test accuracy: 60.9%
- Final loss: 0.453
-/
```

## File Format Example

Here's what a serialized matrix looks like:

```lean
def layer1Weights : Matrix 128 784 :=
  ⊞ (i : Idx 128, j : Idx 784) =>
    match i.1.toNat, j.1.toNat with
    | 0, 0 => -0.051235
    | 0, 1 => 0.023457
    | 0, 2 => 0.012345
    ...
    | 127, 783 => 0.043210
    | _, _ => 0.0  -- Default case (should never occur)
```

And a vector:

```lean
def layer1Bias : Vector 128 :=
  ⊞ (i : Idx 128) =>
    match i.1.toNat with
    | 0 => 0.001234
    | 1 => -0.002345
    | 2 => 0.000123
    ...
    | 127 => 0.001111
    | _ => 0.0  -- Default case
```

## Key Features

### ✅ Human-Readable
- Plain text Lean source code
- Can view/edit in any text editor
- Use `diff` to compare models

### ✅ Type-Safe
- Compile-time dimension checking
- Type errors if you try to use incompatible dimensions
- No runtime parsing overhead

### ✅ Version Control Friendly
- Track model changes in git
- Review parameter changes in PRs
- Merge different model versions

### ✅ Verifiable
- Models are Lean code
- Can prove properties about them
- Type checker ensures correctness

## Performance

### File Sizes

For MNIST (784→128→10 architecture):
- **Total parameters:** 101,770
- **Expected file size:** 10-20 MB
- **First compilation:** 10-60 seconds
- **Subsequent builds:** Fast (cached)

### Optimization

Speed up compilation:
```bash
# Cache precompiled dependencies
lake exe cache get

# Build once
lake build VerifiedNN.SavedModels.Example_MNIST

# Cache your build
lake exe cache put
```

## API Summary

```lean
-- Save a model
saveModel : (net : MLPArchitecture) → (metadata : ModelMetadata) →
            (filepath : String) → IO Unit

-- Metadata structure
structure ModelMetadata where
  trainedOn : String        -- Timestamp
  epochs : Nat              -- Training epochs
  finalTrainAcc : Float     -- Final train accuracy
  finalTestAcc : Float      -- Final test accuracy
  finalLoss : Float         -- Final loss
  architecture : String     -- Architecture description
  learningRate : Float      -- Learning rate
  datasetSize : Nat         -- Dataset size

-- Load (use static imports instead)
-- import VerifiedNN.SavedModels.YourModel
-- let model := VerifiedNN.SavedModels.YourModel.trainedModel
```

## Use Cases

### 1. Checkpointing During Training

```lean
-- Save after each epoch
for epoch in [1:numEpochs] do
  trainedNet ← trainOneEpoch trainedNet data config
  let metadata := createMetadata trainedNet epoch
  saveModel trainedNet metadata s!"SavedModels/Checkpoint_Epoch{epoch}.lean"
```

### 2. Model Versioning

```lean
-- Save different versions
saveModel v1 metadata1 "SavedModels/MNIST_v1.lean"
saveModel v2 metadata2 "SavedModels/MNIST_v2.lean"

-- Compare in code
import VerifiedNN.SavedModels.MNIST_v1
import VerifiedNN.SavedModels.MNIST_v2

def compareModels (input : Vector 784) : Bool :=
  MNIST_v1.trainedModel.predict input ==
  MNIST_v2.trainedModel.predict input
```

### 3. Sharing Models

```lean
-- Commit to git
git add SavedModels/BestModel.lean
git commit -m "Add best performing model (86.2% accuracy)"
git push

-- Colleague can use it
import VerifiedNN.SavedModels.BestModel
let production := VerifiedNN.SavedModels.BestModel.trainedModel
```

## Advantages Over Binary Formats

| Feature | Lean Source | Binary |
|---------|-------------|---------|
| Human-readable | ✅ Yes | ❌ No |
| Diff-able | ✅ Yes | ❌ No |
| Type-checked | ✅ Yes | ❌ No |
| Verifiable | ✅ Yes | ❌ No |
| File size | ⚠️ Larger | ✅ Smaller |
| Load speed | ⚠️ Slower | ✅ Faster |

For this project, the advantages of Lean source format outweigh the disadvantages.

## Next Steps

1. **Try the examples:**
   ```bash
   lake build VerifiedNN.Examples.SerializationExample
   lake build VerifiedNN.Examples.TrainAndSerialize
   ```

2. **Integrate into your training loop:**
   - Add `saveModel` calls after training
   - Create meaningful metadata
   - Use descriptive filenames

3. **Explore the saved models:**
   - View the generated files
   - Import them in your code
   - Compare different models

4. **Read the docs:**
   - `VerifiedNN/Network/SERIALIZATION_USAGE.md` - Complete guide
   - `MODEL_SERIALIZATION_SUMMARY.md` - Implementation details
   - `SavedModels/README.md` - Directory usage guide

## Questions?

See the comprehensive documentation:
- **Usage Guide:** `VerifiedNN/Network/SERIALIZATION_USAGE.md`
- **Implementation:** `VerifiedNN/Network/Serialization.lean`
- **Examples:** `VerifiedNN/Examples/SerializationExample.lean`
- **Summary:** `MODEL_SERIALIZATION_SUMMARY.md`

---

**Status:** ✅ Implementation complete and ready to use!
