# Verified Neural Network Training in Lean 4

This project implements and formally verifies a multilayer perceptron (MLP) trained on MNIST handwritten digits using stochastic gradient descent (SGD) with backpropagation, entirely within Lean 4.

## Verification Philosophy

Mathematical properties proven on ℝ (real numbers), computational implementation in Float (IEEE 754). The Float→ℝ gap is acknowledged—we verify symbolic correctness, not floating-point numerics.

## Project Status

**Current Status:** Initial setup complete - Project structure created, awaiting dependency stabilization

### Completed

- ✅ Repository initialized with Lean 4 project structure
- ✅ Created modular directory structure for all components
- ✅ Configured lakefile.lean with SciLean and LSpec dependencies
- ✅ Created placeholder files for all planned modules
- ✅ Set up lean-toolchain (currently v4.24.0, auto-updated by SciLean/mathlib)

### In Progress

- ⚠️ Dependency resolution - mathlib/SciLean versions are actively being updated
- ⚠️ Build system stabilization - some transitive dependency issues

### Next Steps

1. Wait for SciLean/mathlib compatibility to stabilize (currently Lean 4.24.0)
2. Implement Core modules (DataTypes, LinearAlgebra, Activation)
3. Begin vertical slice implementation (Phase 1 from spec)

## Project Structure

```text
LEAN_mnist/
├── lean-toolchain           # Lean version (auto-managed by dependencies)
├── lakefile.lean            # Build configuration
├── VerifiedNN.lean          # Main library file
├── VerifiedNN/
│   ├── Core/                # Fundamental types, linear algebra, activations
│   │   ├── DataTypes.lean
│   │   ├── LinearAlgebra.lean
│   │   └── Activation.lean
│   ├── Layer/               # Dense layers with differentiability proofs
│   │   ├── Dense.lean
│   │   ├── Composition.lean
│   │   └── Properties.lean
│   ├── Network/             # MLP architecture, initialization, gradients
│   │   ├── Architecture.lean
│   │   ├── Initialization.lean
│   │   └── Gradient.lean
│   ├── Loss/                # Cross-entropy with mathematical properties
│   │   ├── CrossEntropy.lean
│   │   ├── Properties.lean
│   │   └── Gradient.lean
│   ├── Optimizer/           # SGD implementation
│   │   ├── SGD.lean
│   │   ├── Momentum.lean
│   │   └── Update.lean
│   ├── Training/            # Training loop, batching, metrics
│   │   ├── Loop.lean
│   │   ├── Batch.lean
│   │   └── Metrics.lean
│   ├── Data/                # MNIST loading and preprocessing
│   │   ├── MNIST.lean
│   │   ├── Preprocessing.lean
│   │   └── Iterator.lean
│   ├── Verification/        # Formal proofs
│   │   ├── GradientCorrectness.lean
│   │   ├── TypeSafety.lean
│   │   ├── Convergence.lean
│   │   └── Tactics.lean
│   ├── Testing/             # Unit tests, integration tests, gradient checking
│   │   ├── GradientCheck.lean
│   │   ├── UnitTests.lean
│   │   └── Integration.lean
│   └── Examples/            # Minimal examples and full MNIST training
│       ├── SimpleExample.lean
│       └── MNISTTrain.lean
├── scripts/
│   ├── download_mnist.sh    # MNIST dataset retrieval
│   └── benchmark.sh         # Performance benchmarks
├── CLAUDE.md                # Development guide for Claude Code
├── verified-nn-spec.md      # Full technical specification
└── README.md                # This file
```

## Dependencies

- **Lean 4.x** (project uses whatever version SciLean requires, currently 4.24.0)
- **SciLean** (latest compatible, main branch) - Automatic differentiation and scientific computing
- **mathlib4** (via SciLean) - Mathematical foundations
- **LSpec** (testing framework)
- **OpenBLAS** (optional, for performance)

## Build Commands

Once dependencies stabilize:

```bash
# Setup
lake update                    # Update dependencies
lake exe cache get             # Download precompiled mathlib

# Build
lake build                     # Build entire project
lake build VerifiedNN.Core.DataTypes  # Build specific module

# Clean build
lake clean
lake build

# Execute
lake exe simpleExample         # Run minimal example
lake exe mnistTrain --epochs 10 --batch-size 32 --lr 0.01

# Test
lake build VerifiedNN.Testing.UnitTests
lake env lean --run VerifiedNN/Testing/UnitTests.lean

# Verify proofs
lake build VerifiedNN.Verification.GradientCorrectness
lean --print-axioms VerifiedNN/Verification/GradientCorrectness.lean
```

## Technical Approach

### Target Architecture

- **Input layer:** 784 dimensions (28×28 flattened images)
- **Hidden layer:** 128 neurons with ReLU activation
- **Output layer:** 10 neurons (digit classes 0-9)
- **Loss function:** Cross-entropy loss
- **Optimizer:** Mini-batch SGD

### Success Criteria

**Functional Goals:**

- MLP trains on MNIST dataset
- Achieves meaningful test accuracy
- Automatic differentiation integrated with SciLean
- Code compiles and runs successfully

**Verification Goals:**

- Type-level dimension safety for core operations
- Gradient correctness proven symbolically for key components
- Convergence properties stated formally (proofs may be axiomatized)
- Gradient validation through numerical methods

**Code Quality Goals:**

- Public functions documented with docstrings
- Code follows Lean 4 style conventions
- Modular architecture with clear separation of concerns
- Incomplete verification documented with TODO comments

**Reproducibility Goals:**

- Complete build instructions in README
- Dependency management via Lake
- MNIST dataset acquisition documented
- Example runs with expected outcomes

## Development Approach

### Iterative Development

Development follows an iterative pattern focused on building working implementations first, then adding verification as understanding deepens:

1. Implement computational code (Float) with basic tests
2. Iterate until functionality works as expected
3. Add formal verification (ℝ) when design stabilizes
4. Document with docstrings explaining verification scope
5. Code with incomplete proofs (`sorry`) is acceptable during development—mark with TODO comments

See `verified-nn-spec.md` Section 8 for the detailed 5-iteration implementation plan.

## Current Limitations

1. **Build system**: Dependencies are still stabilizing in the Lean 4.24.0 ecosystem
2. **SciLean**: Early-stage library, API may change, performance being optimized
3. **Performance**: Expected to be slower than PyTorch/JAX (this is a proof-of-concept)
4. **No GPU**: CPU-only via OpenBLAS
5. **Float arithmetic unverified**: Acknowledged gap between ℝ and Float

## Development Notes

All module files have been created with placeholder implementations marked with `sorry`. The structure follows the technical specification in `verified-nn-spec.md`, which details:

- Implementation phases (10 phases from setup to documentation)
- Task breakdown for each component
- Verification requirements (core goals and aspirational)
- Testing strategy
- Performance expectations

See `CLAUDE.md` for detailed development conventions, SciLean integration patterns, and guidance for working with this codebase.

## External Resources

### Lean 4 Documentation

- Official docs: <https://lean-lang.org/documentation/>
- Theorem Proving in Lean 4: <https://leanprover.github.io/theorem_proving_in_lean4/>
- Mathlib docs: <https://leanprover-community.github.io/mathlib4_docs/>
- Functional Programming in Lean: <https://lean-lang.org/functional_programming_in_lean/>

### SciLean Resources

- Repository: <https://github.com/lecopivo/SciLean>
- Documentation (WIP): <https://lecopivo.github.io/scientific-computing-lean/>
- Zulip #scientific-computing: <https://leanprover.zulipchat.com/>

### Community

- Lean Zulip chat: <https://leanprover.zulipchat.com/>
- Relevant channels: #scientific-computing, #mathlib4, #new members

### References

- **Technical Specification:** `verified-nn-spec.md`
- **Development Guide:** `CLAUDE.md`
- Certigrad (ICML 2017): Prior work on verified backpropagation in Lean 3
- "Developing Bug-Free Machine Learning Systems With Formal Mathematics"

## License

MIT License - See LICENSE file for details
