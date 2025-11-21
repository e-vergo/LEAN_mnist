# VerifiedNN Examples Directory

Complete collection of executable examples demonstrating verified neural network training in Lean 4.

## Directory Overview

**Purpose:** Pedagogical examples, production training scripts, and utility tools
**Status:** 9 example files, 7 with executable targets
**Build Status:** ✅ All 9 files compile with zero errors

---

## Quick Start Guide

**Want to train a model and see results quickly?**
```bash
lake exe miniTraining      # 100 samples, 10 epochs, ~30 seconds
lake exe mnistTrainMedium  # 5K samples, 12 epochs, ~12 minutes
```

**Want maximum accuracy?**
```bash
lake exe mnistTrainFull    # 60K samples, 50 epochs, ~3.3 hours, 93% accuracy
```

**Want to understand manual backpropagation?**
```bash
lake exe trainManual       # Manual gradient implementation with detailed logging
```

**Want to visualize MNIST digits?**
```bash
lake exe renderMNIST --count 10
```

---

## Production Training Examples (Use These!)

### MNISTTrainFull.lean - Production-Proven Training
- **Purpose:** Full-scale MNIST training (60,000 samples, 50 epochs)
- **Accuracy:** 93% test accuracy (empirically validated)
- **Runtime:** ~3.3 hours on CPU
- **Approach:** Manual backpropagation (fully computable)
- **Command:** `lake exe mnistTrainFull`
- **Features:**
  - Full dataset training (60K train, 10K test)
  - Automatic model checkpointing (saves best model)
  - Progress tracking and metrics logging
  - Final accuracy and loss reporting
- **Use when:** You want production-quality results and have time for full training

### MNISTTrainMedium.lean - Recommended for Development
- **Purpose:** Medium-scale training for quick iteration (5,000 samples, 12 epochs)
- **Accuracy:** 75-85% expected
- **Runtime:** ~12 minutes on CPU
- **Approach:** Manual backpropagation
- **Command:** `lake exe mnistTrainMedium`
- **Features:**
  - Subset training for faster hyperparameter tuning
  - Same architecture as full training (784→128→10)
  - Epoch-by-epoch progress reporting
- **Use when:** You want to test changes quickly without waiting hours

### MiniTraining.lean - Quick Validation
- **Purpose:** Minimal smoke test (100 train, 50 test, 10 epochs)
- **Runtime:** 10-30 seconds
- **Approach:** Manual backpropagation
- **Command:** `lake exe miniTraining`
- **Features:**
  - Aggressive learning rate (0.5) for fast convergence
  - Tiny dataset for rapid iteration
  - Validates entire training pipeline
- **Use when:** You want to verify code changes didn't break training

---

## Command-Line Interface Example

### MNISTTrain.lean - Production CLI with Training
- **Purpose:** Feature-rich command-line interface with real MNIST training
- **Status:** ✅ Fully functional (uses manual backpropagation via `trainEpochsWithConfig`)
- **Command:** `lake exe mnistTrain [OPTIONS]`
- **Options:**
  - `--epochs N` - Number of training epochs (default: 10)
  - `--batch-size N` - Mini-batch size (default: 32)
  - `--lr FLOAT` - Learning rate (default: 0.01, parsing TODO)
  - `--quiet` - Reduce output verbosity
  - `--help` - Show help message
- **Features:**
  - Complete argument parsing and validation
  - Real MNIST data loading (60K train, 10K test)
  - Training progress monitoring
  - Final evaluation statistics
  - Training time measurement
- **Example usage:**
```bash
lake exe mnistTrain --epochs 5 --batch-size 64
```
- **Use when:** You need a production CLI for MNIST training

---

## Pedagogical Examples

### TrainManual.lean - Manual Gradient Reference
- **Purpose:** Educational example showing explicit manual backpropagation
- **Approach:** Manual gradients with detailed documentation
- **Command:** `lake exe trainManual`
- **Features:**
  - Explicit gradient computation without automatic differentiation
  - Detailed logging of gradient norms and parameter updates
  - Uses very low learning rate (0.00001) for numerical stability demonstration
  - Full MNIST dataset (60K train, 10K test)
- **Note:** Currently limited to 500 samples by DEBUG flag (see Task 6.6)
- **Use when:** You want to understand how manual backprop works step-by-step

### SimpleExample.lean - Minimal Toy Example (⚠️ REFERENCE ONLY)
- **Purpose:** Minimal pedagogical example on synthetic data
- **Status:** Uses automatic differentiation (marked `noncomputable unsafe`)
- **Command:** `lake exe simpleExample`
- **Dataset:** 16 synthetic samples (2 per class for 8 classes)
- **Training:** 20 epochs, batch size 4, learning rate 0.01
- **Runtime:** ~5-15 seconds
- **⚠️ NOTE:** This example uses SciLean's `∇` operator for automatic differentiation. For executable training with production-quality results, use `MiniTraining.lean` instead.
- **Use when:** You want to understand the AD-based approach (for comparison/reference)

---

## Utilities

### RenderMNIST.lean - ASCII Visualization Tool
- **Purpose:** Visualize MNIST digits as ASCII art
- **Command:** `lake exe renderMNIST [OPTIONS]`
- **Options:**
  - `--index N` - Display digit at specific index (default: 0)
  - `--count N` - Display first N digits (overrides --index)
  - `--help` - Show usage information
- **Features:**
  - High-quality ASCII rendering with grayscale characters
  - Grid layout for multiple digits
  - Label and pixel statistics display
  - Loads from real MNIST dataset
- **Example usage:**
```bash
lake exe renderMNIST --count 5     # Show first 5 digits
lake exe renderMNIST --index 100   # Show digit at index 100
```
- **Use when:** You want to inspect MNIST data visually

---

## Serialization Examples

### SerializationExample.lean - Model Save/Load Demo
- **Purpose:** Minimal example of model persistence
- **Status:** ⚠️ No lakefile executable entry (see Task 6.3)
- **Features:**
  - Creates network with random initialization
  - Demonstrates serialization to human-readable Lean source
  - Shows how to load saved models
- **Note:** Uses outdated `initializeNetwork` (should use `initializeNetworkHe`)
- **File size:** 89 lines (minimal implementation)

### TrainAndSerialize.lean - Train + Save Workflow (⚠️ REFERENCE ONLY)
- **Purpose:** Complete train→save workflow demonstration
- **Status:** Uses automatic differentiation (marked `noncomputable unsafe`)
- **Status:** ⚠️ No lakefile executable entry (see Task 6.3)
- **Features:**
  - Loads MNIST dataset
  - Trains network with AD-based gradients
  - Saves trained model to Lean source file
  - Demonstrates model reloading
- **⚠️ NOTE:** This example uses automatic differentiation and may not execute. For production training with serialization, use `MNISTTrainFull` which saves checkpoints with manual backpropagation.
- **File size:** 6,444 lines
- **Recommendation:** Consider DELETE (redundant with MNISTTrainFull which does train+save with manual backprop)

---

## Manual Backprop vs Automatic Differentiation

**This project uses two approaches for gradient computation:**

### Manual Backpropagation (Recommended for Production)
- ✅ **Fully computable and executable**
- ✅ **Achieves 93% MNIST accuracy** (empirically validated)
- ✅ **Production-ready and reliable**
- ✅ **Used in all recommended examples**

**Examples using manual backprop:**
- `MNISTTrainFull.lean` (93% accuracy, 3.3 hours)
- `MNISTTrainMedium.lean` (75-85% accuracy, 12 minutes)
- `MiniTraining.lean` (quick validation, 30 seconds)
- `TrainManual.lean` (educational reference with detailed logging)
- `MNISTTrain.lean` (production CLI)

**Implementation:** Uses `networkGradientManual` from `VerifiedNN.Network.ManualGradient`

### Automatic Differentiation (Reference Only)
- ❌ **Uses SciLean's `∇` operator** (noncomputable in current implementation)
- ℹ️ **Useful for formal verification** (specification of correct gradients)
- ℹ️ **May not execute in all contexts** (depends on Lean compiler optimizations)
- ⚠️ **Not recommended for production training**

**Examples using automatic differentiation:**
- `SimpleExample.lean` (toy example on 16 synthetic samples)
- `TrainAndSerialize.lean` (train+save workflow, may not execute)

**Implementation:** Uses SciLean's automatic differentiation via `∇` operator

**Why this distinction matters:**
- Manual backprop is the **proven, working approach** for training
- Automatic differentiation provides the **mathematical specification** for verification
- Gradient correctness theorems prove manual backprop matches AD specification
- See [CLAUDE.md](/Users/eric/LEAN_mnist/CLAUDE.md) for full rationale

---

## Complete File Listing

| File | LOC | Executable | Approach | Status | Purpose |
|------|-----|------------|----------|--------|---------|
| MNISTTrainFull.lean | 15,312 | ✅ `mnistTrainFull` | Manual | Production | Full 60K training, 93% accuracy |
| MNISTTrainMedium.lean | 12,709 | ✅ `mnistTrainMedium` | Manual | Production | 5K training, 12 min |
| MiniTraining.lean | 5,164 | ✅ `miniTraining` | Manual | Production | 100 samples, 30 sec smoke test |
| MNISTTrain.lean | 16,502 | ✅ `mnistTrain` | Manual | Production | CLI with arg parsing |
| TrainManual.lean | 8,916 | ✅ `trainManual` | Manual | Pedagogical | Manual backprop reference |
| RenderMNIST.lean | 12,064 | ✅ `renderMNIST` | N/A | Utility | ASCII visualization |
| SimpleExample.lean | 10,682 | ✅ `simpleExample` | AD | Reference | Toy example with AD |
| TrainAndSerialize.lean | 6,444 | ❌ None | AD | Reference | Train+save (redundant) |
| SerializationExample.lean | 2,879 | ❌ None | N/A | Minimal | Save/load demo |

**Legend:**
- **Approach:** Manual = manual backpropagation (computable), AD = automatic differentiation (noncomputable)
- **Status:** Production = recommended for real training, Pedagogical = educational, Reference = comparison/deprecated, Utility = tools

---

## Accuracy Claims (Empirically Validated)

**MNISTTrainFull:**
- **Claimed:** 93% test accuracy
- **Validation:** Achieved in production run (60K samples, 50 epochs, 3.3 hours)
- **Evidence:** 29 saved model checkpoints, best model at epoch 49

**MNISTTrainMedium:**
- **Expected:** 75-85% test accuracy
- **Validation:** To be empirically verified
- **Note:** Subset training (5K samples) limits maximum achievable accuracy

**MiniTraining:**
- **Expected:** Variable (quick convergence validation only)
- **Validation:** Not designed for high accuracy (tiny dataset)
- **Purpose:** Smoke test to verify training pipeline works

---

## Build and Run

**Build all examples:**
```bash
lake build VerifiedNN.Examples
```

**Build specific example:**
```bash
lake build VerifiedNN.Examples.MNISTTrainFull
```

**Run executable:**
```bash
lake exe [executable-name]
```

**Available executables (from lakefile.lean):**
```bash
lake exe mnistTrainFull      # Production: 60K samples, 93% accuracy
lake exe mnistTrainMedium    # Production: 5K samples, 12 minutes
lake exe miniTraining        # Production: 100 samples, 30 seconds
lake exe mnistTrain          # Production: CLI with options
lake exe trainManual         # Pedagogical: manual backprop reference
lake exe renderMNIST         # Utility: ASCII visualization
lake exe simpleExample       # Reference: AD-based toy example
```

**Missing executables (not in lakefile):**
- `serializationExample` - (Task 6.3: Add to lakefile or delete)
- `trainAndSerialize` - (Task 6.3: Add to lakefile or DELETE recommended)

---

## Prerequisites

**Required:**
- MNIST dataset in `data/` directory
  ```bash
  ./scripts/download_mnist.sh
  ```
- Lean 4 (v4.23.0 as specified in lean-toolchain)
- Lake build system

**Optional but recommended:**
- OpenBLAS for performance (system package)
- Mathlib cache: `lake exe cache get` (saves compilation time)

---

## Common Issues

**Issue:** MNIST data files not found
**Solution:** Run `./scripts/download_mnist.sh` to download dataset to `data/` directory

**Issue:** Training is very slow
**Expected:** Lean training is ~400× slower than PyTorch (CPU-only, no SIMD optimization). This is a research prototype, not production ML infrastructure.

**Issue:** `simpleExample` or `trainAndSerialize` fail to execute
**Explanation:** These use automatic differentiation which is noncomputable. Use manual backprop examples instead (MiniTraining, MNISTTrainMedium, MNISTTrainFull).

**Issue:** Build errors about missing modules
**Solution:** Run `lake update` to fetch dependencies, then `lake build`

**Issue:** SciLean compilation takes forever
**Solution:** Use `lake exe cache get` to download precompiled mathlib (~1GB download)

---

## Code Quality Standards

All examples maintain research-quality standards:

**Documentation:**
- ✅ Module-level docstrings (/-! format)
- ✅ Function-level docstrings (/-- format)
- ✅ Clear status indicators (Production/Pedagogical/Reference/Utility)
- ✅ Usage examples and expected output

**Build Quality:**
- ✅ Zero compilation errors
- ✅ Zero linter warnings (except expected sorry warnings in GradientFlattening)
- ✅ All public definitions documented
- ✅ Naming conventions followed (Lean 4 standards)

**Verification Status:**
- ✅ 0 sorries in all example files
- ✅ All axioms inherited from training infrastructure (documented in Loss/Properties.lean and Verification/Convergence/Axioms.lean)

---

## Related Documentation

- [Project README](/Users/eric/LEAN_mnist/README.md) - Overall project overview
- [CLAUDE.md](/Users/eric/LEAN_mnist/CLAUDE.md) - Development guidelines (see "Manual Backprop vs AD" section)
- [GETTING_STARTED.md](/Users/eric/LEAN_mnist/GETTING_STARTED.md) - Setup and onboarding
- [verified-nn-spec.md](/Users/eric/LEAN_mnist/verified-nn-spec.md) - Technical specification
- [lakefile.lean](/Users/eric/LEAN_mnist/lakefile.lean) - Build configuration

**Directory-specific documentation:**
- [Network/README.md](/Users/eric/LEAN_mnist/VerifiedNN/Network/README.md) - See ManualGradient.lean vs Gradient.lean comparison
- [Training/README.md](/Users/eric/LEAN_mnist/VerifiedNN/Training/README.md) - Training loop implementation details
- [Verification/README.md](/Users/eric/LEAN_mnist/VerifiedNN/Verification/README.md) - Gradient correctness theorems

---

## Contributing

When adding new examples:

1. **Choose implementation approach:**
   - **Manual backpropagation** for executable production examples
   - **Automatic differentiation** only for reference/comparison (will be noncomputable)

2. **Follow naming conventions:**
   - Files: PascalCase (e.g., `MNISTTrainFull.lean`)
   - Executables: camelCase (e.g., `mnistTrainFull`)

3. **Add lakefile entry:**
```lean
lean_exe yourExecutableName where
  root := `VerifiedNN.Examples.YourFileName
  supportInterpreter := true
```

4. **Maintain documentation standards:**
   - Module docstring with /-! format
   - Clear status (Production/Pedagogical/Reference/Utility)
   - Usage instructions and expected runtime
   - Update this README with new entry

5. **Update this README:**
   - Add to appropriate section (Production/Pedagogical/Reference/Utility)
   - Update file listing table
   - Add to "Available executables" section

---

**Last Updated:** 2025-11-21
**Files:** 9 example files (7 with executables, 2 without)
**Status:** ✅ All files compile, 6 production-ready examples, 93% accuracy achieved
**Next Actions:** Task 6.3 (add lakefile entries), Task 6.5 (deprecation notices), Task 6.6 (remove DEBUG limit)
