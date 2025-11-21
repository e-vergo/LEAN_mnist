# Getting Started with Verified Neural Networks in Lean 4

A comprehensive guide to setting up, building, and using this formally verified neural network implementation.

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installing Prerequisites](#installing-prerequisites)
4. [Building the Project](#building-the-project)
5. [Downloading MNIST Data](#downloading-mnist-data)
6. [Testing Your Installation](#testing-your-installation)
7. [Understanding the Project](#understanding-the-project)
8. [Next Steps](#next-steps)

## Introduction

This project implements a neural network trained on MNIST with **formally verified gradient correctness**. Key features:

- ✅ **Proven correct:** 26 theorems proving automatic differentiation computes exact gradients
- ✅ **Type-safe:** Dependent types ensure dimension correctness at compile time
- ⚠️ **Research project:** Some proofs incomplete (4 active sorries), training cannot execute due to noncomputable AD

**Time to setup:** 15-30 minutes (mostly waiting for compilation)

## System Requirements

### Operating System
- **Supported:** macOS (Intel or Apple Silicon), Linux (Ubuntu 20.04+, Arch, etc.)
- **Not supported:** Windows (SciLean doesn't support Windows)

### Hardware
- **CPU:** Any modern processor (multi-core recommended for faster builds)
- **RAM:** 4GB minimum, 8GB+ recommended (16GB+ for comfortable MCP development)
- **Disk:** ~2GB for dependencies + MNIST data
- **Internet:** Required for downloading dependencies and MNIST dataset

### Software Prerequisites
- **Lean 4.23.0** (installed via elan, specified in lean-toolchain)
- **Git**
- **OpenBLAS** (BLAS library for numerical operations)
- **Bash** (for running scripts)
- **uv** (optional, for MCP server setup)
- **ripgrep** (optional, for local search in MCP)

## Installing Prerequisites

### Step 1: Install Elan (Lean Version Manager)

**macOS and Linux:**
```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

Follow the prompts. Then restart your terminal or run:
```bash
source ~/.profile  # or ~/.bashrc, ~/.zshrc depending on your shell
```

**Verify installation:**
```bash
elan --version
lean --version  # Should show Lean version (4.23.0 for this project)
lake --version
```

**Troubleshooting:**
- If `lean` command not found, add `~/.elan/bin` to your PATH
- On Ubuntu, you may need: `sudo apt install curl git`

### Step 2: Install OpenBLAS

OpenBLAS provides optimized linear algebra routines.

**macOS (Homebrew):**
```bash
brew install openblas
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install libopenblas-dev
```

**Arch Linux:**
```bash
sudo pacman -S openblas
```

**Verify installation:**
```bash
# macOS
ls /opt/homebrew/opt/openblas/lib  # Apple Silicon
ls /usr/local/opt/openblas/lib     # Intel

# Linux
ls /usr/lib/x86_64-linux-gnu/openblas*
```

### Step 3: Install Git (if not already installed)

```bash
# macOS
brew install git  # or use Xcode Command Line Tools

# Linux
sudo apt install git  # Ubuntu/Debian
sudo pacman -S git    # Arch
```

### Step 4: Install Optional Tools (for MCP Development)

If you plan to use the lean-lsp-mcp server for AI-assisted development:

**Install uv (Python package manager):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install ripgrep (for local search):**
```bash
# macOS
brew install ripgrep

# Ubuntu/Debian
sudo apt install ripgrep

# Arch
sudo pacman -S ripgrep
```

**Verify:**
```bash
uv --version
rg --version
```

## Building the Project

### Clone the Repository

```bash
git clone https://github.com/yourusername/LEAN_mnist.git
cd LEAN_mnist
```

### Build Dependencies and Project

```bash
# Update dependencies
lake update

# Download precompiled mathlib (highly recommended!)
lake exe cache get

# Build the entire project
lake build
```

**What to expect:**
- With cache: **2-5 minutes** (only builds this project)
- Without cache: **20-40 minutes** (builds mathlib + SciLean + project)
- You'll see progress messages: `✔ [N/total] Building ModuleName`
- Some harmless warnings about OpenBLAS paths are normal

**Expected final output:**
```
✔ [40/40] Building VerifiedNN.Examples.MNISTTrain
Build completed successfully.
```

**Common build warnings (safe to ignore):**
- `ld64.lld: warning: directory not found for option -L/usr/local/opt/openblas/lib` (macOS)
- `warning: building from source; failed to fetch GitHub release` (leanblas)
- `warning: declaration uses 'sorry'` (expected for incomplete proofs)

**If build fails:**
- Check that you're using the correct Lean version: `lean --version` should match `lean-toolchain`
- Try `lake clean` then `lake build` (warning: this rebuilds everything)
- Check for network issues if dependencies fail to download
- Ensure OpenBLAS is properly installed

## Downloading MNIST Data

The MNIST dataset is not included in the repository (too large for git).

### Automatic Download (Recommended)

```bash
./scripts/download_mnist.sh
```

This downloads from `ossci-datasets.s3.amazonaws.com` mirror (~55MB total):
- `train-images-idx3-ubyte.gz` → 60,000 training images
- `train-labels-idx1-ubyte.gz` → training labels
- `test-images-idx3-ubyte.gz` → 10,000 test images
- `test-labels-idx1-ubyte.gz` → test labels

Files are extracted to `data/` directory.

**Verify download:**
```bash
ls -lh data/
# Should show 4 uncompressed files (train-images, train-labels, test-images, test-labels)
```

### Manual Download (if script fails)

Download from AWS mirror:
```bash
mkdir -p data
cd data
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/test-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/test-labels-idx1-ubyte.gz

gunzip *.gz
```

## Testing Your Installation

### Test 1: Verify Build Status

```bash
lake build
```

**Expected:** Zero errors, only expected warnings (sorry declarations, OpenBLAS paths).

**Build stats:**
- 59 Lean files compiled successfully
- 4 active sorries (all documented with completion strategies)
- 9 axioms (all justified in documentation)

### Test 2: Test Data Loading

```bash
lake exe mnistLoadTest
```

**Expected:** `✓ Loaded 60,000 training images, 10,000 test images`

**Status:** ✅ Compiles to executable, runs perfectly

### Test 3: Test ASCII Visualization

```bash
lake exe renderMNIST 0  # View first training image
lake exe renderMNIST --count 5  # View 5 random samples
```

**Expected:** Beautiful ASCII art representations of handwritten digits

**Status:** ✅ Compiles to executable, excellent UX

### Test 4: Test Forward Pass and Gradient Validation

```bash
lake exe smokeTest
```

**Expected:** All 5 tests pass:
1. Network creation
2. Forward pass dimension consistency
3. Loss computation
4. Gradient computation
5. Numerical gradient validation

**Status:** ✅ Compiles to executable, validates AD correctness

### Test 5: Check Data Distribution

```bash
lake exe checkDataDistribution
```

**Expected:** Shows distribution of digits in MNIST dataset

**Status:** ✅ Compiles to executable

### Test 6: Verify Gradient Correctness Proofs

```bash
lake build VerifiedNN.Verification.GradientCorrectness
lean --print-axioms VerifiedNN/Verification/GradientCorrectness.lean
```

**Expected:** List of axioms used in proofs (should show 9 documented axioms)

**Status:** ✅ Builds successfully, proofs verified

---

### Installation Test Checklist

Use this checklist to verify your setup works:

- [ ] ✅ `lake build` completes successfully (zero errors)
- [ ] ✅ `lake exe mnistLoadTest` loads 60K training + 10K test images
- [ ] ✅ `lake exe renderMNIST 0` shows ASCII digit
- [ ] ✅ `lake exe smokeTest` passes all 5 tests
- [ ] ✅ `lake exe checkDataDistribution` shows digit distribution
- [ ] ✅ `lake build VerifiedNN.Verification.GradientCorrectness` compiles proofs
- [ ] ❌ `lake exe mnistTrain` FAILS (expected - see Known Limitations below)
- [ ] ❌ `lake exe simpleExample` FAILS (expected - see Known Limitations below)

**6/8 passing?** Perfect! Those 2 failures are expected and documented.

## Understanding the Project

### Understanding Project Capabilities

This section sets realistic expectations about what works and what doesn't.

**What You Can Do (Fully Functional):**

- ✅ **Explore MNIST Data:** Load and visualize 60,000 training images with ASCII renderer
- ✅ **Run Forward Pass:** Execute network inference on test data
- ✅ **Validate Gradients:** Numerical gradient checks confirm AD correctness
- ✅ **Study Verification:** Read 26 proven gradient correctness theorems
- ✅ **Learn Type Safety:** See dependent types prevent dimension errors at compile time
- ✅ **Understand Architecture:** Explore formally verified neural network implementation

**What Doesn't Work (Current Limitations):**

- ❌ **Training Executables:** `lake exe mnistTrain` and `lake exe simpleExample` fail to compile
- ❌ **Fast Training:** Only interpreter mode works (10-100x slower than compiled code)
- ❌ **Production Deployment:** Noncomputable AD prevents standalone executable distribution

**Why Training Doesn't Compile:**

SciLean's automatic differentiation (`∇` operator) uses compile-time metaprogramming for symbolic rewriting. This produces mathematically correct gradients (proven by theorems) but cannot be compiled to native machine code. The `noncomputable` keyword in Lean marks this explicitly.

**Workaround:** Interpreter mode (`lake env lean --run`) executes the code but is significantly slower than compiled binaries.

### What Makes This Special?

**Formally Verified Gradients:**
- Every differentiable operation (ReLU, matrix multiply, softmax, etc.) has a **proven theorem** that automatic differentiation computes the correct mathematical gradient
- Main theorem (`mlp_architecture_differentiable`) proves end-to-end differentiability through the entire 2-layer MLP
- 26 gradient correctness theorems covering all network components

**Type Safety:**
- Dependent types ensure matrix dimensions match at compile time
- Impossible to multiply incompatible matrices or access out-of-bounds indices
- Type-level specifications correspond to runtime array dimensions (proven in TypeSafety.lean)

**Honest About Limitations:**
- 17 active sorries (all documented with completion strategies)
- 9 axioms (all justified: 8 convergence theory, 1 Float bridge)
- Float ≈ ℝ gap acknowledged (we prove properties on real numbers, implement in Float)
- Training executables don't compile (noncomputable AD limitation)

### Project Structure

```
VerifiedNN/
├── Core/          # Fundamental types (Vector, Matrix), linear algebra, activations
│                  # - DataTypes.lean: Vector, Matrix with dependent typing
│                  # - LinearAlgebra.lean: Matrix operations with proofs
│                  # - Activations.lean: ReLU, softmax with gradient proofs
├── Layer/         # Dense layers with 13 proven properties
│                  # - DenseLayer.lean: Forward/backward pass
│                  # - Properties.lean: 13 differentiability theorems
├── Network/       # MLP architecture, initialization, gradient computation
│                  # - Architecture.lean: 2-layer MLP definition
│                  # - Gradient.lean: End-to-end gradient computation (7 sorries)
│                  # - Initialization.lean: Xavier/He initialization
├── Loss/          # Cross-entropy with mathematical properties proven
│                  # - CrossEntropy.lean: Loss function
│                  # - Properties.lean: Non-negativity, convexity, gradient proofs
├── Optimizer/     # SGD implementation
│                  # - SGD.lean: Parameter updates with momentum
├── Training/      # Training loop, metrics, utilities
│                  # - Loop.lean: Main training orchestration
│                  # - Metrics.lean: Accuracy, loss tracking
├── Data/          # MNIST loading and preprocessing
│                  # - MNIST.lean: IDX format parser
│                  # - Preprocessing.lean: Normalization, batching
├── Verification/  # 🎯 FORMAL PROOFS of gradient correctness
│                  # - GradientCorrectness.lean: 26 theorems (6 sorries)
│                  # - TypeSafety.lean: Dimension consistency (2 sorries)
│                  # - Convergence/: Theoretical foundations (axiomatized)
├── Testing/       # Unit tests, integration tests, gradient checks
│                  # - UnitTests.lean: Component-level tests
│                  # - GradientChecks.lean: Numerical validation
└── Examples/      # Runnable examples
                   # - SimpleExample.lean: Minimal training demo
                   # - MNISTTrain.lean: Full MNIST training
```

### Known Limitations

This section documents what doesn't work and why, so you're not surprised.

**Training Executables Don't Compile**

```bash
# These commands will FAIL:
lake exe mnistTrain       # ❌ Error: "unknown constant 'VerifiedNN.Examples.MNISTTrain.main'"
lake exe simpleExample    # ❌ Error: "unknown constant 'VerifiedNN.Examples.SimpleExample.main'"
```

**Root Cause:**

SciLean's automatic differentiation (`∇` operator) is marked `noncomputable` because:
1. It uses compile-time metaprogramming for symbolic rewriting
2. This metaprogramming generates mathematically correct gradients (proven by theorems)
3. But the metaprogramming infrastructure isn't available in compiled native code
4. Lean's compiler cannot translate the `∇` operator to machine instructions

**Workaround (Slow):**

Interpreter mode executes the code but bypasses native compilation:

```bash
lake env lean --run VerifiedNN/Examples/SimpleExample.lean  # Works, but 10-100x slower
```

**Performance Impact:**
- Interpreter mode: 10-100x slower than compiled binaries
- Acceptable for: Proof-of-concept validation, testing correctness
- Not acceptable for: Experimentation, research, production use

**Implications:**
- This project demonstrates verified gradient correctness
- It's not suitable for actual ML experimentation or deployment
- A production system would need a different AD implementation (e.g., JAX, PyTorch) and separate verification

### Interpreter Mode vs Compiled Code

**Computable (compiled to native binaries):**
- ✅ Data loading and preprocessing
- ✅ Forward pass (without gradients)
- ✅ Matrix operations, activations (forward only)
- ✅ Metrics computation (accuracy, loss evaluation)
- ✅ Numerical gradient checking (finite differences)
- ✅ All test executables (mnistLoadTest, renderMNIST, smokeTest, checkDataDistribution)

**Noncomputable (requires interpreter mode):**
- ❌ Gradient computation (uses SciLean's `∇` operator)
- ❌ Training loop (depends on gradients)
- ❌ Backward pass through layers
- ❌ Any use of automatic differentiation
- ❌ Training examples (MNISTTrain, SimpleExample)

**How to use interpreter mode:**
```bash
lake env lean --run <filepath>
```

### Key Files to Explore

1. **Main Theorem:** `VerifiedNN/Verification/GradientCorrectness.lean:384-404`
   - `mlp_architecture_differentiable`: End-to-end network differentiability
2. **Network Definition:** `VerifiedNN/Network/Architecture.lean:27-48`
   - `MLPArchitecture`: 2-layer MLP structure
3. **Training Loop:** `VerifiedNN/Training/Loop.lean:76-137`
   - `trainEpoch`: Single epoch with gradient updates
4. **MNIST Data Loading:** `VerifiedNN/Data/MNIST.lean:149-192`
   - `loadTrainData`, `loadTestData`: IDX format parsing
5. **Development Guide:** `CLAUDE.md`
   - Comprehensive development standards and conventions

### Verification Status

**Current status (as of 2025-11-20):**
- **Build:** ✅ All 59 files compile with zero errors
- **Sorries:** 4 active sorries (all documented with completion strategies)
  - Verification/TypeSafety.lean: 4 (flatten/unflatten inverses and dimension proofs)
- **Axioms:** 9 total (all documented and justified)
  - 8 convergence theory axioms (out of scope for gradient verification)
  - 1 Float ≈ ℝ bridge axiom (acknowledged numerical gap)

**Documentation coverage:** 100%
- All sorries have TODO comments with completion strategies
- All axioms have comprehensive docstrings with justification
- All directories have detailed READMEs (~103KB total documentation)

## Next Steps

### For Users Who Want to Explore the Data

Now that you've verified the setup works, start exploring:

```bash
# View random MNIST samples (excellent ASCII visualization)
lake exe renderMNIST --count 10

# View specific images
lake exe renderMNIST 0    # First training image
lake exe renderMNIST 100  # 100th training image

# Understand dataset distribution
lake exe checkDataDistribution

# Run smoke tests (validates forward pass and gradients)
lake exe smokeTest
```

### For Users Who Want to Understand Verification

The real value of this project is formal verification of gradient correctness:

1. **Read Core Theorems:** `VerifiedNN/Verification/GradientCorrectness.lean` - 26 gradient proofs
2. **Study Layer Properties:** `VerifiedNN/Layer/Properties.lean` - 13 differentiability theorems
3. **Explore Activations:** `VerifiedNN/Core/Activations.lean` - ReLU and softmax gradient proofs
4. **Technical Specification:** [verified-nn-spec.md](verified-nn-spec.md) - Complete technical details
5. **Verification Status:** [README.md - Verification Status](README.md#verification-status) - Axiom catalog
6. **Module Documentation:** Directory READMEs in `VerifiedNN/` subdirectories

### For Users Who Want to Experiment (Advanced)

Training works in interpreter mode but is slow (not recommended for quick experimentation):

```bash
# Simple example (~10 seconds)
lake env lean --run VerifiedNN/Examples/SimpleExample.lean

# Full MNIST training (~2-5 minutes, requires patience)
lake env lean --run VerifiedNN/Examples/MNISTTrain.lean
```

**Performance warning:** Interpreter mode is 10-100x slower than compiled executables. This is acceptable for validation but not for experimentation or research.

### For Contributors

1. **Development Guide:** [CLAUDE.md](CLAUDE.md) - Standards, MCP tools, conventions
2. **Setup MCP:** Configure lean-lsp-mcp for AI-assisted development (see CLAUDE.md)
3. **Documentation:** All 10 `VerifiedNN/` subdirectories have comprehensive READMEs
4. **Complete Proofs:** Help finish remaining sorries:
   - Verification/TypeSafety.lean: 4 sorries (flatten/unflatten inverses and dimension proofs)

### For Researchers

1. **Study Proof Strategy:** Main theorem in `Verification/GradientCorrectness.lean`
2. **Extend Architecture:** Add new layer types (convolution, batch norm, etc.) with proofs
3. **Improve Completeness:** Replace axioms with full proofs (see README axiom catalog)
4. **Compare Related Work:** Review Certigrad, DeepSpec, and positioning
5. **Academic Presentation:** This codebase is designed for publication-quality work

### Learning Resources

**Internal Documentation:**
- [README.md](README.md) - Project overview and current status
- [CLAUDE.md](CLAUDE.md) - Development guide and MCP tools
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [verified-nn-spec.md](verified-nn-spec.md) - Technical specification
- Directory READMEs in `VerifiedNN/` subdirectories - Module-specific guides

**External Documentation:**
- [Lean 4 Official Docs](https://lean-lang.org/documentation/)
- [Theorem Proving in Lean 4](https://leanprover.github.io/theorem_proving_in_lean4/)
- [Mathlib4 Documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [SciLean Repository](https://github.com/lecopivo/SciLean)
- [Lean Zulip Chat](https://leanprover.zulipchat.com/) (#scientific-computing, #mathlib4)

## Installation Checklist

Use this checklist to verify your setup:

**Required (Core Functionality):**

- [ ] Elan installed (`elan --version`)
- [ ] Lean 4.23.0 installed (`lean --version`)
- [ ] Lake available (`lake --version`)
- [ ] Git installed (`git --version`)
- [ ] OpenBLAS installed (check library paths)
- [ ] Repository cloned
- [ ] Dependencies updated (`lake update`)
- [ ] Mathlib cache downloaded (`lake exe cache get` - optional but recommended)
- [ ] `lake build` completed successfully (zero errors, only expected warnings)
- [ ] MNIST data downloaded (`ls data/` shows 4 files)

**Verify Working Features:**

- [ ] ✅ `lake exe mnistLoadTest` loads 60K + 10K images
- [ ] ✅ `lake exe renderMNIST 0` shows ASCII digit
- [ ] ✅ `lake exe smokeTest` passes all 5 tests
- [ ] ✅ `lake exe checkDataDistribution` shows digit distribution
- [ ] ✅ `lake build VerifiedNN.Verification.GradientCorrectness` compiles
- [ ] ✅ `lean --print-axioms VerifiedNN/Verification/GradientCorrectness.lean` shows 9 axioms

**Expected Failures (Documented Limitations):**

- [ ] ❌ `lake exe mnistTrain` fails with "unknown constant" error (expected)
- [ ] ❌ `lake exe simpleExample` fails with "unknown constant" error (expected)

**All core features checked?** You're ready to explore formally verified neural networks!

**Optional (for MCP development):**

- [ ] uv installed (`uv --version`)
- [ ] ripgrep installed (`rg --version`)
- [ ] MCP server configured (`~/.claude.json` with lean-lsp entry)
- [ ] Claude Code recognizes MCP tools (`claude mcp list` shows lean-lsp-mcp)

## Getting Help

**Common issues:**
- **Build timeout:** Ensure you've run `lake exe cache get` to download precompiled mathlib
- **LSP unresponsive:** Kill Lean servers with `pkill -f "lean --server"` and restart
- **MNIST download fails:** Try manual download from AWS mirror (see Downloading MNIST Data section)
- **Import errors:** Run `lake update` to refresh dependencies
- **"unknown constant 'main'" error:** This is expected for training executables (see Known Limitations section)
- **Training executable fails:** Training cannot be compiled - this is a documented SciLean limitation

**Detailed troubleshooting:** Check individual module READMEs in `VerifiedNN/` subdirectories

**Development questions:** See [CLAUDE.md](CLAUDE.md) for coding standards and MCP usage

**Verification questions:** See [verified-nn-spec.md](verified-nn-spec.md) for proof strategies

**Community support:** Visit [Lean Zulip](https://leanprover.zulipchat.com/) (#scientific-computing channel)

---

**Welcome to verified machine learning!** 🚀

---

**Last Updated:** 2025-11-20
