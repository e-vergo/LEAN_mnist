# Model Serialization Usage Guide

Complete guide to saving and loading trained neural networks as Lean source files.

## Quick Start

### 1. Save a Trained Model

```lean
import VerifiedNN.Network.Serialization

-- After training
let metadata : ModelMetadata := {
  trainedOn := "2025-10-22 23:59:45"
  epochs := 5
  finalTrainAcc := 0.862
  finalTestAcc := 0.609
  finalLoss := 0.453
  architecture := "784→128→10 (ReLU+Softmax)"
  learningRate := 0.00001
  datasetSize := 60000
}

-- Save to file
saveModel trainedNetwork metadata "SavedModels/MNIST_20251022_235945.lean"
```

### 2. Use the Saved Model

The saved model is a complete Lean source file that can be imported directly:

```lean
-- Import the generated module
import VerifiedNN.SavedModels.MNIST_20251022_235945

-- Use the trained model
let model := VerifiedNN.SavedModels.MNIST_20251022_235945.trainedModel
let prediction := model.predict inputImage
```

## File Format

Generated files are human-readable Lean source code:

```lean
import VerifiedNN.Network.Architecture
import VerifiedNN.Core.DataTypes

namespace VerifiedNN.SavedModels

/-!
# Trained MNIST Model: MNIST_20251022_235945

## Training Configuration
- **Architecture:** 784→128→10 (ReLU+Softmax)
- **Epochs:** 5
- **Learning rate:** 0.00001
- **Dataset size:** 60000 samples

## Training Results
- **Final training accuracy:** 86.2%
- **Final test accuracy:** 60.9%
- **Final loss:** 0.453
- **Trained:** 2025-10-22 23:59:45
-/

def layer1Weights : Matrix 128 784 :=
  ⊞ (i : Idx 128, j : Idx 784) =>
    match i.1.toNat, j.1.toNat with
    | 0, 0 => -0.051235
    | 0, 1 => 0.023457
    ...
    | _, _ => 0.0

def layer1Bias : Vector 128 := ...
def layer2Weights : Matrix 10 128 := ...
def layer2Bias : Vector 10 := ...

def trainedModel : MLPArchitecture := {
  layer1 := { weights := layer1Weights, bias := layer1Bias }
  layer2 := { weights := layer2Weights, bias := layer2Bias }
}

end VerifiedNN.SavedModels
```

## Integration with Training Loop

### Typical Workflow

```lean
-- 1. Train the network
let trainedNet ← trainNetwork dataset config

-- 2. Evaluate performance
let trainAcc ← evaluateAccuracy trainedNet trainData
let testAcc ← evaluateAccuracy trainedNet testData
let finalLoss ← computeLoss trainedNet trainData

-- 3. Create metadata
let timestamp := getCurrentTimestamp  -- implement as needed
let metadata : ModelMetadata := {
  trainedOn := timestamp
  epochs := config.epochs
  finalTrainAcc := trainAcc
  finalTestAcc := testAcc
  finalLoss := finalLoss
  architecture := "784→128→10 (ReLU+Softmax)"
  learningRate := config.learningRate
  datasetSize := trainData.size
}

-- 4. Save model
let filename := s!"SavedModels/MNIST_{timestamp}.lean"
saveModel trainedNet metadata filename
```

## File Organization

Recommended directory structure:

```
project/
├── VerifiedNN/
│   ├── Network/
│   │   └── Serialization.lean       # Serialization module
│   └── SavedModels/                 # Directory for saved models
│       ├── MNIST_20251022_120000.lean
│       ├── MNIST_20251022_180000.lean
│       └── BestModel.lean           # Symlink or copy of best model
├── lakefile.lean                    # Add SavedModels library
└── ...
```

## File Sizes and Performance

### Expected File Sizes

For MNIST architecture (784→128→10):
- **Total parameters:** 101,770 (weights + biases)
- **File size:** 10-20 MB
- **Compilation time:** 10-60 seconds (first compile)

Parameter breakdown:
- Layer 1 weights: 784 × 128 = 100,352
- Layer 1 bias: 128
- Layer 2 weights: 128 × 10 = 1,280
- Layer 2 bias: 10

### Optimization Tips

1. **Use Lake cache:**
   ```bash
   lake build SavedModels/MNIST_20251022_120000
   lake exe cache put  # Cache compiled artifacts
   ```

2. **Git LFS for large files:**
   ```bash
   git lfs track "SavedModels/*.lean"
   git lfs track "SavedModels/*.olean"
   ```

3. **Selective importing:**
   ```lean
   -- Only import when needed
   section ModelLoading
   import VerifiedNN.SavedModels.BestModel
   end ModelLoading
   ```

## Advantages of Lean Source Format

### Human-Readable
- View weights directly in text editor
- Debug by inspecting specific parameter values
- Compare models using diff tools

### Version Control Friendly
- Track model evolution in git
- Review parameter changes in pull requests
- Merge different model versions

### Type-Safe
- Compile-time dimension checking
- No runtime parsing overhead
- Guaranteed correctness via Lean's type system

### Verifiable
- Models are Lean code, can be formally verified
- Type checker ensures architectural consistency
- Can prove properties about loaded models

## Alternative: Binary Format

If file size or loading speed becomes an issue, consider implementing binary serialization:

```lean
-- Future enhancement (not implemented)
def serializeModelBinary (net : MLPArchitecture) (filepath : String) : IO Unit
def loadModelBinary (filepath : String) : IO MLPArchitecture
```

Binary format would:
- Reduce file size by ~10x
- Load faster (no compilation needed)
- Lose human-readability
- Require runtime parsing validation

## Troubleshooting

### Issue: File too large for git

**Solution:** Use Git LFS
```bash
git lfs install
git lfs track "SavedModels/*.lean"
git add .gitattributes
```

### Issue: Slow compilation

**Solution:** Use lake cache
```bash
lake exe cache get  # Download precompiled dependencies
lake build          # Build once
lake exe cache put  # Cache your builds
```

### Issue: Import not found

**Solution:** Check lakefile.lean includes:
```lean
lean_lib VerifiedNN.SavedModels {
  roots := #[`VerifiedNN.SavedModels]
  globs := #[.submodules `VerifiedNN.SavedModels]
}
```

### Issue: Out of memory during compilation

**Solution:** Reduce file size or use binary format (future)
```bash
# Current workaround: increase Lean memory limit
LEAN_MEMORY_MB=8192 lake build
```

## Examples

### Save After Training

```lean
def main : IO Unit := do
  let trainedNet ← trainMNIST
  let metadata := createMetadata trainedNet
  saveModel trainedNet metadata "SavedModels/Trained_MNIST.lean"
  IO.println "Model saved successfully!"
```

### Load and Evaluate

```lean
import VerifiedNN.SavedModels.Trained_MNIST

def evaluateSavedModel : IO Unit := do
  let model := VerifiedNN.SavedModels.Trained_MNIST.trainedModel
  let testData ← loadMNISTTest
  let accuracy ← evaluateAccuracy model testData
  IO.println s!"Test accuracy: {accuracy * 100.0}%"
```

### Compare Models

```bash
# Compare two saved models
diff SavedModels/Model_A.lean SavedModels/Model_B.lean
```

## See Also

- `VerifiedNN.Network.Architecture` - MLP architecture definition
- `VerifiedNN.Training.Loop` - Training implementation
- `VerifiedNN.Examples.SerializationExample` - Complete example
- `CLAUDE.md` - Project documentation and conventions
