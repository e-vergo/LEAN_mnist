# Saved Models Directory

This directory contains serialized neural network models saved as Lean source files.

## Purpose

Models saved here are complete Lean modules that can be imported and used directly in your code. Each file contains:
- Training metadata (accuracy, loss, hyperparameters)
- Complete network parameters (weights and biases)
- Ready-to-use `trainedModel` definition

## Usage

### Saving a Model

After training, save your model here:

```lean
import VerifiedNN.Network.Serialization

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

saveModel trainedNet metadata "SavedModels/MNIST_20251022_235945.lean"
```

### Loading a Model

Import the generated module:

```lean
import VerifiedNN.SavedModels.MNIST_20251022_235945

def myInference (input : Vector 784) : Nat :=
  VerifiedNN.SavedModels.MNIST_20251022_235945.trainedModel.predict input
```

## File Naming Convention

Recommended naming scheme:
- **Timestamp-based:** `MNIST_YYYYMMDD_HHMMSS.lean`
- **Version-based:** `MNIST_v1.lean`, `MNIST_v2.lean`
- **Performance-based:** `MNIST_acc_86.2.lean`
- **Purpose-based:** `BestModel.lean`, `ProductionModel.lean`

## File Structure

Each saved model follows this structure:

```lean
import VerifiedNN.Network.Architecture
import VerifiedNN.Core.DataTypes

namespace VerifiedNN.SavedModels

/-!
# Trained MNIST Model: [Model Name]

Training metadata and configuration...
-/

def layer1Weights : Matrix 128 784 := ...
def layer1Bias : Vector 128 := ...
def layer2Weights : Matrix 10 128 := ...
def layer2Bias : Vector 10 := ...

def trainedModel : MLPArchitecture := {
  layer1 := { weights := layer1Weights, bias := layer1Bias }
  layer2 := { weights := layer2Weights, bias := layer2Bias }
}

end VerifiedNN.SavedModels
```

## Version Control

### Git LFS Recommended

For large models (>10 MB), use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track Lean model files
git lfs track "SavedModels/*.lean"

# Commit .gitattributes
git add .gitattributes
git commit -m "Track SavedModels with Git LFS"
```

### .gitignore

If you don't want to commit models:

```
# Add to .gitignore
SavedModels/*.lean
!SavedModels/README.md
```

## File Sizes

Typical file sizes for MNIST architecture (784→128→10):
- **Parameters:** 101,770 values
- **File size:** 10-20 MB
- **Compilation time:** 10-60 seconds (first time)

## Performance

### Compilation

First import of a large model may take time:
```bash
# First build
lake build VerifiedNN.SavedModels.MyModel  # May take 30-60s

# Subsequent builds
lake build VerifiedNN.SavedModels.MyModel  # Fast (cached)
```

### Caching

Use lake cache to speed up builds:
```bash
lake exe cache get   # Download precompiled dependencies
lake build          # Build your models
lake exe cache put  # Cache your builds
```

## Examples

See these files for complete examples:
- `VerifiedNN/Examples/SerializationExample.lean` - Basic serialization
- `VerifiedNN/Examples/TrainAndSerialize.lean` - Full training + save workflow
- `VerifiedNN/Network/SERIALIZATION_USAGE.md` - Comprehensive usage guide

## Troubleshooting

### Issue: Model file too large for git

**Solution:** Use Git LFS (see above) or exclude from version control

### Issue: Slow compilation

**Solution:**
1. Use lake cache: `lake exe cache get`
2. Only import models when needed
3. Consider binary format (future enhancement)

### Issue: Import not found

**Solution:** Make sure lakefile.lean includes:
```lean
lean_lib VerifiedNN.SavedModels {
  roots := #[`VerifiedNN.SavedModels]
  globs := #[.submodules `VerifiedNN.SavedModels]
}
```

### Issue: Out of memory during compilation

**Solution:** Increase Lean memory limit:
```bash
LEAN_MEMORY_MB=8192 lake build
```

## Maintenance

### Cleanup Old Models

Remove outdated models periodically:
```bash
# List models by size
ls -lhS SavedModels/*.lean

# Remove old models
rm SavedModels/MNIST_old_*.lean
```

### Model Registry

Keep a log of important models:
```
SavedModels/MODELS.txt:
- BestModel.lean (86.2% acc, 2025-10-22) - Production model
- MNIST_20251022_120000.lean (84.1% acc) - Initial training
- MNIST_20251022_180000.lean (85.7% acc) - Improved hyperparams
```

## See Also

- `VerifiedNN/Network/Serialization.lean` - Serialization implementation
- `VerifiedNN/Network/SERIALIZATION_USAGE.md` - Complete usage guide
- `MODEL_SERIALIZATION_SUMMARY.md` - Implementation summary
