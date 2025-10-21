# VerifiedNN Examples

This directory contains runnable examples demonstrating the VerifiedNN training pipeline.

## Status

**Current:** MOCK IMPLEMENTATIONS - All examples are runnable but use simulated training
**Purpose:** Demonstrate project structure, CLI design, and expected behavior
**Next:** Replace with real implementations as core modules are developed

---

## Available Examples

### 1. SimpleExample.lean

**Purpose:** Minimal pedagogical example showing basic training flow

**Usage:**
```bash
lake exe simpleExample
```

**What it demonstrates:**
- Basic training configuration
- Training loop structure with epochs
- Progress monitoring (loss and accuracy)
- Training completion summary

**Output:**
- Simulates 5 epochs of training on 100 samples
- Shows loss decreasing from 2.30 to 1.70
- Shows accuracy improving from 12% to 85%

**Best for:** Understanding the overall training pipeline structure

---

### 2. MNISTTrain.lean

**Purpose:** Production-ready MNIST training script with full CLI

**Usage:**
```bash
# Run with defaults (10 epochs, batch size 32, lr 0.01)
lake exe mnistTrain

# Custom configuration
lake exe mnistTrain --epochs 20 --batch-size 64

# See all options
lake exe mnistTrain --help
```

**Command-line options:**
- `--epochs N` - Number of training epochs (default: 10)
- `--batch-size N` - Mini-batch size (default: 32)
- `--lr FLOAT` - Learning rate (default: 0.01, parsing not yet implemented)
- `--quiet` - Reduce output verbosity
- `--help` - Show help message

**What it demonstrates:**
- Command-line argument parsing
- Full MNIST training pipeline structure
- Data loading simulation (60k train, 10k test)
- Network initialization (784→128→10 MLP)
- Per-epoch progress with metrics
- Training time measurement
- Final evaluation and summary

**Output:**
- Simulates realistic MNIST training
- Shows per-epoch loss and accuracy
- Displays training time and performance metrics

**Best for:** Understanding production training scripts and CLI design

---

## Implementation Roadmap

These examples will become fully functional as the following modules are implemented:

### Core Dependencies (for both examples)
1. **VerifiedNN/Core/LinearAlgebra.lean** - Matrix operations with SciLean
2. **VerifiedNN/Core/Activation.lean** - ReLU and softmax activations
3. **VerifiedNN/Layer/Dense.lean** - Dense layer forward/backward passes
4. **VerifiedNN/Network/Architecture.lean** - MLP composition
5. **VerifiedNN/Training/Loop.lean** - SGD training loop

### MNIST-Specific Dependencies
6. **VerifiedNN/Data/MNIST.lean** - MNIST data loading (IDX or CSV format)
7. **VerifiedNN/Data/Preprocessing.lean** - Normalization and batching
8. **VerifiedNN/Training/Metrics.lean** - Accuracy and loss computation

See [verified-nn-spec.md](../../verified-nn-spec.md) for detailed implementation specifications.

---

## Development Guidelines

### When to Update Examples

**Update SimpleExample when:**
- Core training loop structure changes
- New configuration options are added
- Output format is redesigned

**Update MNISTTrain when:**
- CLI arguments change
- New hyperparameters are supported
- Progress monitoring format changes
- MNIST-specific logic is added

### Transitioning from Mock to Real

When implementing real functionality:
1. Keep mock version as backup (rename to `MNISTTrain.Mock.lean`)
2. Replace mock implementations incrementally
3. Test each component replacement independently
4. Validate against gradient checks and finite differences
5. Document verification status in docstrings

### Code Quality Standards

Both examples should maintain:
- **Clear documentation:** Module-level and function-level docstrings
- **Mock status visibility:** Clearly marked in comments and output
- **User-friendly output:** Well-formatted progress and results
- **Error handling:** Graceful failure with helpful messages
- **Lean conventions:** Follow project naming and style guidelines

---

## Testing the Examples

### Building
```bash
# Build all examples
lake build VerifiedNN.Examples.SimpleExample
lake build VerifiedNN.Examples.MNISTTrain

# Or build entire project
lake build
```

### Running
```bash
# SimpleExample - no arguments
lake exe simpleExample

# MNISTTrain - test different configurations
lake exe mnistTrain
lake exe mnistTrain --epochs 5
lake exe mnistTrain --epochs 10 --batch-size 16
lake exe mnistTrain --quiet
lake exe mnistTrain --help
```

### Expected Behavior (Current Mock)

**SimpleExample:**
- Prints training progress for 5 epochs
- Shows decreasing loss
- Shows increasing accuracy
- Completes successfully with summary

**MNISTTrain:**
- Parses command-line arguments correctly
- Shows configuration
- Simulates data loading
- Shows per-epoch progress
- Displays final metrics and timing
- Exits cleanly (exit code 0)

### Common Issues

**Issue:** `lake exe` command not found
**Solution:** Ensure Lean 4 and Lake are installed via `elan`

**Issue:** Build errors about missing modules
**Solution:** Run `lake update` to fetch dependencies

**Issue:** SciLean compilation takes forever
**Solution:** Use `lake exe cache get` to download precompiled mathlib

---

## Contributing

When adding new examples:
1. Follow the module docstring format (`/-!` block)
2. Include clear usage instructions
3. Document mock vs. real implementation status
4. Add entry to this README
5. Update lakefile.lean with new executable
6. Test with `lake exe <name>`

---

## Related Documentation

- [Project README](../../README.md) - Overall project documentation
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines for Claude Code
- [verified-nn-spec.md](../../verified-nn-spec.md) - Detailed technical specification
- [lakefile.lean](../../lakefile.lean) - Build configuration

---

**Last Updated:** 2025-10-21
**Status:** Mock implementations - functional examples demonstrating structure
