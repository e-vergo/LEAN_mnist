# VerifiedNN Examples

Runnable examples demonstrating the VerifiedNN training pipeline.

## Directory Overview

**Purpose:** Provide pedagogical examples and production CLI demonstrations
**Status:** Mixed - SimpleExample is REAL, MNISTTrain is MOCK
**Build Status:** ✅ Both files compile with zero errors

---

## Available Examples

### 1. SimpleExample.lean ✅ REAL IMPLEMENTATION

**Status:** Fully functional with real training (no mocks)

Minimal pedagogical example demonstrating end-to-end training on synthetic data.
Serves as proof-of-concept that all training infrastructure components work correctly.

**Usage:**
```bash
lake exe simpleExample
```

**What it demonstrates:**
- Network initialization (He method)
- Forward pass computation
- Automatic differentiation for gradients
- SGD parameter updates
- Real loss and accuracy metrics
- Training loop orchestration

**Configuration:**
- Architecture: 784 → 128 → 10 MLP
- Dataset: 16 synthetic samples (2 per class for 8 classes)
- Training: 20 epochs, batch size 4, learning rate 0.01
- Expected runtime: ~5-15 seconds

**Limitations:**
- Tiny dataset (16 samples, pedagogical only)
- No train/test split (overfitting guaranteed)
- Synthetic patterns (trivially learnable)

**Best for:** Understanding how all components integrate and seeing gradient descent work

**Verification Status:**
- Build: ✅ Zero errors
- Sorries: 0 (all removed via safe bounds checking)
- Warnings: 0
- Axioms: Inherits from training infrastructure

---

### 2. MNISTTrain.lean ⚠️ MOCK BACKEND

**Status:** Production CLI with simulated training backend

Demonstrates production-ready command-line interface for MNIST training.
While the CLI is complete, the backend currently simulates training with realistic output.

**Usage:**
```bash
# Run with defaults
lake exe mnistTrain

# Custom configuration
lake exe mnistTrain --epochs 20 --batch-size 64

# See all options
lake exe mnistTrain --help
```

**Command-line options:**
- `--epochs N` - Number of training epochs (default: 10)
- `--batch-size N` - Mini-batch size (default: 32)
- `--lr FLOAT` - Learning rate (parsing TODO, uses default 0.01)
- `--quiet` - Reduce output verbosity
- `--help` - Show help message

**What it demonstrates:**

**Currently Working (Real):**
- ✅ Command-line argument parsing and validation
- ✅ Error handling with clear messages
- ✅ Help text and usage information
- ✅ Configuration display
- ✅ Training time measurement

**Currently Simulated (Mock):**
- ⚠️ MNIST data loading (simulates 60k train, 10k test)
- ⚠️ Network training (linear extrapolation for metrics)
- ⚠️ Loss and accuracy computation (synthetic values)

**Best for:** Understanding production CLI design and user experience

**Verification Status:**
- Build: ✅ Zero errors
- Sorries: 0
- Warnings: 0
- Axioms: None (pure CLI code, no numerical computation)

---

## Module Descriptions

### SimpleExample.lean

**Purpose:** Pedagogical demonstration of real training
**Lines of code:** ~240 (includes enhanced docstrings)
**Key definitions:** `generateToyDataset`, `main`
**Dependencies:** Network.Architecture, Network.Initialization, Training.Loop, Training.Metrics
**Verification status:** ✅ Fully functional, zero sorries
**Documentation:** 100% coverage with comprehensive mathlib-quality docstrings

### MNISTTrain.lean

**Purpose:** Production CLI demonstration with mock backend
**Lines of code:** ~395 (includes enhanced docstrings)
**Key definitions:** `TrainingConfig`, `parseArgs`, `formatFloat`, `runTraining`, `main`
**Dependencies:** SciLean (minimal, just for imports)
**Verification status:** ✅ CLI complete, backend mocked
**Documentation:** 100% coverage with detailed API documentation for all functions

---

## Implementation Roadmap

### SimpleExample (COMPLETE)
- ✅ Network initialization
- ✅ Training loop integration
- ✅ Gradient computation via AD
- ✅ Metrics computation
- ✅ Sample prediction display
- ✅ Safe array access (removed sorries)

### MNISTTrain (IN PROGRESS)

**Phase 1: Data Loading (Next Priority)**
- [ ] Implement `VerifiedNN/Data/MNIST.lean` with IDX parser
- [ ] Add CSV fallback for debugging
- [ ] Integrate into `runTraining`

**Phase 2: Connect Training Infrastructure**
- [ ] Replace mock training with `trainEpochsWithConfig`
- [ ] Use real `initializeMLPArchitecture`
- [ ] Connect `computeAccuracy` and `computeAverageLoss`

**Phase 3: Enhanced Features**
- [ ] Implement Float parsing for `--lr` flag
- [ ] Add `--data-dir` for custom MNIST location
- [ ] Add `--checkpoint-dir` for model saving

---

## Testing the Examples

### Building
```bash
# Build specific examples
lake build VerifiedNN.Examples.SimpleExample
lake build VerifiedNN.Examples.MNISTTrain

# Or build entire project
lake build
```

### Running
```bash
# SimpleExample - Real training
lake exe simpleExample

# MNISTTrain - Mock training with various configs
lake exe mnistTrain
lake exe mnistTrain --epochs 5
lake exe mnistTrain --epochs 10 --batch-size 16
lake exe mnistTrain --quiet
lake exe mnistTrain --help
```

### Expected Output

**SimpleExample:**
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
  [Progress...]

Final performance:
  Accuracy: 100.0%
  Loss: 0.05

Sample predictions:
  Sample 0: True=0, Pred=0, Conf=99.8% ✓
  ...
```

**MNISTTrain:**
```
==========================================
MNIST Neural Network Training
==========================================

NOTE: This is a MOCK implementation

Configuration:
  Epochs: 10
  Batch size: 32
  Learning rate: 0.01

Mock: Loaded 60000 training samples
Mock: Loaded 10000 test samples

[Simulated training progress...]

Training Summary
================
Train accuracy improvement: +69.77%
Test accuracy improvement: +69.85%
```

---

## Common Issues

**Issue:** `lake exe` command not found
**Solution:** Install Lean 4 via `curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh`

**Issue:** Build errors about missing modules
**Solution:** Run `lake update` to fetch dependencies

**Issue:** SciLean compilation takes forever
**Solution:** Use `lake exe cache get` to download precompiled mathlib (~1GB)

---

## Code Quality Standards

All examples maintain mathlib submission quality:

**Documentation:**
- ✅ Module-level docstrings (`/-!` format)
- ✅ Function-level docstrings (`/--` format)
- ✅ Clear status indicators (REAL vs MOCK)
- ✅ Usage examples and expected output

**Code Quality:**
- ✅ Zero compilation errors
- ✅ Zero linter warnings
- ✅ Zero non-documented sorries
- ✅ All axioms justified (inherited from training infrastructure)
- ✅ Naming conventions followed (Lean 4 standards)

**User Experience:**
- ✅ Clear, well-formatted output
- ✅ Helpful error messages
- ✅ Progress monitoring
- ✅ Timing information

---

## Contributing

When adding new examples:

1. **Choose implementation status:**
   - Real implementation (like SimpleExample) if dependencies exist
   - Mock implementation (like MNISTTrain) to design interface first

2. **Follow documentation standards:**
   - Module docstring with `/-!` format
   - Status clearly marked (REAL vs MOCK)
   - Usage instructions with examples
   - Sample output showing expected behavior

3. **Maintain quality gates:**
   - Zero compilation errors
   - Zero linter warnings
   - Docstrings on all public definitions
   - README entry with status

4. **Update lakefile.lean:**
   - Add new executable target
   - Follow naming convention: camelCase for executables

---

## Related Documentation

- [Project README](../../README.md) - Overall project overview
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines
- [verified-nn-spec.md](../../verified-nn-spec.md) - Technical specification
- [CLEANUP_SUMMARY.md](../../CLEANUP_SUMMARY.md) - Codebase cleanup report
- [lakefile.lean](../../lakefile.lean) - Build configuration

---

## Last Cleanup

**Date:** 2025-10-21
**Changes Made:**
- Added References sections to both module docstrings (SimpleExample.lean, MNISTTrain.lean)
- Verified zero diagnostics in both files
- Confirmed all documentation meets mathlib submission standards
- All files already had comprehensive docstrings meeting quality requirements

**Verification Status:** ✅ All files compile, SimpleExample fully functional, MNISTTrain CLI complete
**Next Priority:** Implement MNIST data loading to connect MNISTTrain to real training infrastructure
