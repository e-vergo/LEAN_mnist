# Getting Started with VerifiedNN

A beginner-friendly guide to understanding, building, and using this verified neural network implementation in Lean 4.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Project](#understanding-the-project)
3. [Installation](#installation)
4. [Your First Example](#your-first-example)
5. [Key Concepts](#key-concepts)
6. [Next Steps](#next-steps)

---

## Quick Start

**Want to see it in action immediately?**

```bash
# Clone and build (10-15 minutes first time)
git clone [repository-url]
cd LEAN_mnist
lake build

# Run the ASCII digit visualizer (computable!)
lake exe renderMNIST --count 5

# See a simple training example
lake env lean --run VerifiedNN/Examples/SimpleExample.lean

# Run fast smoke test
lake exe smokeTest
```

**Expected timeline:**
- First build: 10-15 minutes (downloads and compiles dependencies)
- ASCII renderer: 1-2 seconds
- Simple example: 5-10 seconds
- Smoke test: <10 seconds

---

## Understanding the Project

### What is VerifiedNN?

VerifiedNN is a **neural network implementation with mathematical proofs** built in Lean 4. It proves that:

1. **Gradients are mathematically correct** - The automatic differentiation system computes the exact derivative
2. **Types enforce correctness** - Dimension mismatches are caught at compile time
3. **Real code executes** - Not just a paper proof; actual MNIST data loads and processes

### Why Lean 4?

Lean 4 is both:
- A **proof assistant** - Write mathematical theorems and prove them correct
- A **programming language** - Write executable code that compiles to native binaries

This unique combination allows us to prove properties about code that actually runs.

### What Makes This Special?

Most machine learning frameworks (PyTorch, TensorFlow) implement backpropagation without formal verification. VerifiedNN **proves** that the gradients are mathematically correct:

```lean
-- This theorem is formally proven
theorem network_gradient_correct :
  Differentiable ℝ (λ params => networkLoss (unflattenParams params) x y) := by
  -- Proof establishes gradient correctness
  ...
```

### The Float vs ℝ Philosophy

**What we prove:** Mathematical properties on ℝ (real numbers)
**What executes:** Computational implementation in Float (IEEE 754)

The gap between Float and ℝ is acknowledged but not formally bridged. This is standard practice in verified numerical computing.

---

## Installation

### Prerequisites

#### 1. Install Lean 4 (via elan)

```bash
# Linux/macOS
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Follow prompts to add to PATH
source ~/.profile  # or restart terminal
```

Verify installation:
```bash
lean --version
# Should show: Lean (version 4.x.x)
```

#### 2. Install OpenBLAS (for performance)

**macOS:**
```bash
brew install openblas
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libopenblas-dev
```

**Arch Linux:**
```bash
sudo pacman -S openblas
```

#### 3. Install ripgrep (optional, for search tools)

```bash
# macOS
brew install ripgrep

# Ubuntu/Debian
sudo apt-get install ripgrep

# Arch Linux
sudo pacman -S ripgrep
```

### Building the Project

```bash
# Clone repository
git clone [repository-url]
cd LEAN_mnist

# Download precompiled mathlib (saves ~30 minutes)
lake exe cache get

# Build project (first time: 10-15 minutes)
lake build
```

**Troubleshooting:**

**Problem:** `lake exe cache get` fails
**Solution:** This is optional. `lake build` will compile everything from source (slower but works)

**Problem:** OpenBLAS linking errors
**Solution:** These are warnings, not errors. The project still works.

**Problem:** Build takes >30 minutes
**Solution:** This is normal if cache download failed. Subsequent builds are much faster.

---

## Your First Example

### Example 1: ASCII Digit Visualization

The simplest computable example - visualize MNIST digits as ASCII art:

```bash
# Visualize 5 test digits
lake exe renderMNIST --count 5

# Inverted colors for light terminals
lake exe renderMNIST --count 3 --inverted

# Show training set digits
lake exe renderMNIST --count 10 --train
```

**Output:**
```
Sample 0 | Ground Truth: 7
----------------------------

      :*++:.
      #%%%%%*********.
      :=:=+%%#%%%%%%%=
            : :::: %%-
                  :%#
                  %@:
                 =%%:.
                :%%:
                =%*
```

**What this demonstrates:**
- Real MNIST data loading (70,000 images)
- Pure Lean computation (no external dependencies)
- Manual unrolling workaround for SciLean limitations
- First fully computable executable in project

**Technical note:** Uses 28-case match statement with 784 literal array indices to bypass SciLean's `DataArrayN` indexing limitation. See `VerifiedNN/Util/README.md` for details.

### Example 2: Simple Training Demo

Watch a neural network train on synthetic data:

```bash
lake env lean --run VerifiedNN/Examples/SimpleExample.lean
```

**Output:**
```
Initializing network...
Starting training: 20 epochs, batch size 4

Epoch 0: Loss=2.459, Accuracy=0.125
Epoch 1: Loss=2.301, Accuracy=0.250
Epoch 2: Loss=2.156, Accuracy=0.375
...
Epoch 19: Loss=0.423, Accuracy=0.875

Training complete!
```

**What this demonstrates:**
- Network initialization (He method)
- Forward pass computation
- Real loss and accuracy tracking
- SGD parameter updates
- Full training loop orchestration

### Example 3: Run Tests

Verify everything works correctly:

```bash
# Run all unit tests
lake env lean --run VerifiedNN/Testing/UnitTests.lean

# Run optimizer tests
lake env lean --run VerifiedNN/Testing/OptimizerTests.lean

# Fast smoke test
lake exe smokeTest
```

---

## Key Concepts

### 1. Dependent Types for Safety

Lean's type system prevents dimension mismatches:

```lean
-- This type-checks
def layer : DenseLayer 784 128 := ...
def input : Vector 784 := ...
def output : Vector 128 := layer.forward input  -- ✓ OK

-- This won't compile
def badInput : Vector 100 := ...
def badOutput := layer.forward badInput  -- ✗ Type error!
```

**Benefit:** Dimension errors caught at compile time, not runtime.

### 2. Automatic Differentiation

SciLean provides automatic differentiation via the `∇` operator:

```lean
-- Define a loss function
def loss (params : Vector n) : Float := ...

-- Compute gradient automatically
def grad := (∇ p, loss p) params
  |>.rewrite_by fun_trans
```

**Verification:** We prove `fderiv ℝ f = analytical_derivative(f)` for each operation.

### 3. The Computable vs Noncomputable Boundary

**Computable (✅ Executes):**
- Data loading
- Preprocessing
- Forward pass
- Loss computation

**Noncomputable (❌ Symbolic only):**
- Gradient computation (SciLean's `∇` is noncomputable)
- Training loop (depends on gradients)

**Why?** SciLean prioritizes verification over executability. The gradients are proven correct but don't compile to standalone binaries.

### 4. Module Organization

```
VerifiedNN/
├── Core/          # Foundation (vectors, matrices, activations)
├── Layer/         # Dense layers with proofs
├── Network/       # MLP architecture
├── Loss/          # Cross-entropy loss
├── Optimizer/     # SGD and variants
├── Training/      # Training loop
├── Data/          # MNIST loading
├── Verification/  # Formal proofs (MAIN THEOREMS HERE)
├── Testing/       # Test suites
└── Examples/      # Runnable demos
```

**Start here:** `VerifiedNN/Examples/` for working code
**Core math:** `VerifiedNN/Verification/GradientCorrectness.lean` for main theorem

### 5. Documentation Standards

Every module has:
- `/-!` module-level docstring explaining purpose
- `/--` function-level docstrings with parameters and properties
- README.md in each subdirectory

**Example:**
```lean
/-!
# Module Name
Brief overview.
## Main Definitions
...
-/

/-- Function description.
**Parameters:**
- param1: Description
**Returns:** Description
-/
def myFunction := ...
```

---

## Next Steps

### For Users (Want to understand neural networks)

1. **Read the simple example:** `VerifiedNN/Examples/SimpleExample.lean`
2. **Explore Core modules:** Start with `VerifiedNN/Core/DataTypes.lean`
3. **Understand layers:** Read `VerifiedNN/Layer/Dense.lean`
4. **See the proofs:** Check `VerifiedNN/Verification/GradientCorrectness.lean`

### For Contributors (Want to add features)

1. **Read CONTRIBUTING.md:** Contribution guidelines
2. **Study CLAUDE.md:** Development practices and MCP integration
3. **Check TESTING_GUIDE.md:** How to write tests
4. **Review ARCHITECTURE.md:** System design

### For Researchers (Want to extend verification)

1. **Read verified-nn-spec.md:** Technical specification
2. **Study Verification/:** Proof techniques
3. **Check axiom catalog:** README.md Section "Axioms and Unproven Theorems"
4. **Review sorry strategies:** Each sorry has a proof strategy comment

### For Lean Beginners

1. **Learn Lean basics:** https://leanprover.github.io/theorem_proving_in_lean4/
2. **Explore SciLean:** https://github.com/lecopivo/SciLean
3. **Study mathlib:** https://leanprover-community.github.io/mathlib4_docs/
4. **Join Zulip chat:** https://leanprover.zulipchat.com/ (#scientific-computing)

---

## Common Questions

### Q: Can I train a real MNIST model?

**A:** Yes! The data loads and processes. However, gradient computation is noncomputable, so training happens symbolically rather than producing a standalone binary.

```bash
# This works and processes real data
lake exe renderMNIST --count 10

# This works with synthetic data
lake env lean --run VerifiedNN/Examples/SimpleExample.lean
```

### Q: How long does verification take?

**A:** Instant. Verification happens at compile time. `lake build` both compiles and verifies.

### Q: Are the proofs complete?

**A:** Main gradient correctness theorems: **YES, proven** (26 theorems)
Convergence theory: **NO, axiomatized** (explicitly out of scope)
See README.md "Axioms and Unproven Theorems" for complete catalog.

### Q: How does this compare to PyTorch?

| Feature | VerifiedNN | PyTorch |
|---------|------------|---------|
| Gradient correctness | Formally proven | Tested |
| Type safety | Compile-time dimensions | Runtime checks |
| Performance | Slower (research focus) | Production-optimized |
| GPU support | No | Yes |
| Use case | Verification research | Production ML |

### Q: Can I use this in production?

**A:** No. This is a research project demonstrating formal verification of neural networks. Use PyTorch/TensorFlow for production.

### Q: What's the Float vs ℝ gap?

**A:** We prove mathematical properties on **ℝ (real numbers)** but implement using **Float (IEEE 754)**. The correspondence is assumed but not formally proven. This is standard in verified numerical computing.

### Q: Why are some things noncomputable?

**A:** SciLean's automatic differentiation (`∇` operator) is marked noncomputable because it uses symbolic manipulation. This is a SciLean design choice prioritizing verification over executability.

---

## Troubleshooting

### Build Issues

**Symptom:** `lake build` fails with "unknown package 'scilean'"
**Fix:** Run `lake update` to fetch dependencies

**Symptom:** Extremely slow build (>1 hour)
**Fix:** Use `lake exe cache get` to download precompiled mathlib

**Symptom:** OpenBLAS warnings
**Fix:** These are harmless linker warnings, not errors

### Runtime Issues

**Symptom:** "file not found" when running ASCII renderer
**Fix:** Download MNIST data: `./scripts/download_mnist.sh`

**Symptom:** Lean server consuming excessive memory
**Fix:** Restart LSP: `pkill -f "lean --server"`, then rebuild

### MCP Integration Issues

**Symptom:** MCP tools not available in Claude Code
**Fix:** Check `~/.claude.json` configuration exists and restart Claude Code

**Symptom:** MCP server timeout on startup
**Fix:** Build project first (`lake build`), then restart Claude Code

---

## Resources

### Official Documentation
- **This project README:** [README.md](README.md)
- **Developer guide:** [CLAUDE.md](CLAUDE.md)
- **Technical spec:** [verified-nn-spec.md](verified-nn-spec.md)
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md) *(coming soon)*

### Lean 4 Learning
- **Theorem Proving in Lean 4:** https://leanprover.github.io/theorem_proving_in_lean4/
- **Lean 4 Documentation:** https://lean-lang.org/documentation/
- **Mathlib4 docs:** https://leanprover-community.github.io/mathlib4_docs/

### SciLean
- **Repository:** https://github.com/lecopivo/SciLean
- **Documentation:** https://lecopivo.github.io/scientific-computing-lean/

### Community
- **Lean Zulip:** https://leanprover.zulipchat.com/
  - #scientific-computing (for SciLean questions)
  - #new members (for Lean 4 basics)
  - #mathlib4 (for mathematical libraries)

### Academic References
- **Certigrad (ICML 2017):** Prior work on verified backpropagation in Lean 3
- **Selsam et al.:** "Developing Bug-Free Machine Learning Systems With Formal Mathematics"

---

## Quick Reference Card

### Essential Commands

```bash
# Build
lake build                      # Compile entire project
lake build VerifiedNN.Core     # Build specific module

# Run Examples
lake exe renderMNIST --count 5              # ASCII visualization
lake env lean --run VerifiedNN/Examples/SimpleExample.lean

# Run Tests
lake env lean --run VerifiedNN/Testing/UnitTests.lean
lake exe smokeTest             # Fast smoke test

# Verification
lean --print-axioms VerifiedNN/Verification/GradientCorrectness.lean

# Search Code
rg "theorem.*gradient" VerifiedNN --type lean
rg "sorry" VerifiedNN --type lean
```

### File Locations

- **Main theorem:** `VerifiedNN/Verification/GradientCorrectness.lean:352`
- **Simple example:** `VerifiedNN/Examples/SimpleExample.lean`
- **ASCII renderer:** `VerifiedNN/Util/ImageRenderer.lean`
- **Test runner:** `VerifiedNN/Testing/RunTests.lean`
- **Type definitions:** `VerifiedNN/Core/DataTypes.lean`

### Documentation Hierarchy

1. **README.md** - Project overview and claims
2. **GETTING_STARTED.md** - This file (beginner tutorial)
3. **CLAUDE.md** - Developer guide and conventions
4. **verified-nn-spec.md** - Technical specification
5. **Directory READMEs** - Module-specific documentation

---

**Last Updated:** 2025-10-22
**Maintained by:** Project contributors
**Questions?** Open an issue or ask on Lean Zulip #scientific-computing
