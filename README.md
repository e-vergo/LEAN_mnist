# Verified Neural Network Training in Lean 4

This project proves that automatic differentiation computes mathematically correct gradients for neural network training. We implement an MLP trained on MNIST using SGD with backpropagation in Lean 4, and formally verify that the computed gradients equal the analytical derivatives.

## Core Contribution

**Primary Goal:** Prove gradient correctness throughout the neural network. For every differentiable operation, formally verify that `fderiv ℝ f = analytical_derivative(f)`, and prove that composition via chain rule preserves correctness end-to-end.

**Secondary Goal:** Leverage dependent types to prove that type-checked operations maintain correct tensor dimensions at runtime.

## Verification Scope

**What We Prove:**
- Gradient correctness for each operation (ReLU, matrix multiply, softmax, cross-entropy)
- Chain rule preservation through layer composition
- Type-level dimension specifications match runtime array sizes
- End-to-end: network gradient computation is mathematically sound

**Acknowledged Gaps:**
- Float arithmetic (we prove on ℝ, implement in Float)
- Convergence properties of SGD (out of scope)
- Generalization bounds (out of scope)

This follows Certigrad's precedent of proving backpropagation correctness, executed in modern Lean 4 with SciLean.

## Project Status

**Current Status:** ✅ Building successfully - Core modules implemented, verification in progress

### Completed
- ✅ Repository initialized with Lean 4 project structure (v4.20.1)
- ✅ Created modular directory structure for all components
- ✅ Configured lakefile.lean with SciLean and LSpec dependencies
- ✅ **All core modules build successfully**
- ✅ Core data types and operations implemented
- ✅ Layer implementations (Dense, Composition)
- ✅ Network architecture definitions
- ✅ Verification module structure established
- ✅ Testing infrastructure in place

### In Progress (with `sorry` placeholders)
- ⚠️ Training loop implementation (placeholder functions)
- ⚠️ Gradient checking tests (structure complete, implementations pending)
- ⚠️ Data preprocessing utilities
- ⚠️ Network initialization strategies
- ⚠️ Formal verification proofs (theorems stated, proofs in progress)

### Verification Status
- **Type Safety:** Core theorems proven, demonstrates dependent type safety
- **Gradient Correctness:** Theorem statements complete, proofs use axioms/sorries for complex cases
- **Chain Rule:** Proven using mathlib's composition theorem
- **Operation-level gradients:** Stated, awaiting proof completion

### Next Steps
1. Complete training loop implementation
2. Finish gradient correctness proofs (currently axiomatized)
3. Implement gradient checking numerical validation
4. Add MNIST data loading
5. Run end-to-end training with verification

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
│   ├── Verification/        # Formal proofs (CORE CONTRIBUTION)
│   │   ├── GradientCorrectness.lean
│   │   ├── TypeSafety.lean
│   │   └── Tactics.lean
│   ├── Testing/             # Gradient validation, unit tests
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
- **SciLean** (latest compatible, main branch) - Automatic differentiation framework
- **mathlib4** (via SciLean) - Mathematical foundations for calculus and analysis
- **LSpec** (testing framework)
- **OpenBLAS** (optional, for numerical performance)

## Build Commands

Once dependencies stabilize:

```bash
# Setup
lake update                    # Update dependencies
lake exe cache get             # Download precompiled mathlib

# Build
lake build                     # Build entire project
lake build VerifiedNN.Verification.GradientCorrectness  # Build verification

# Verify proofs
lean --print-axioms VerifiedNN/Verification/GradientCorrectness.lean

# Test
lake build VerifiedNN.Testing.GradientCheck
lake env lean --run VerifiedNN/Testing/UnitTests.lean

# Execute training
lake exe mnistTrain --epochs 10 --batch-size 32 --lr 0.01
```

## Technical Approach

### Network Architecture
- **Input:** 784 dimensions (28×28 flattened images)
- **Hidden:** 128 neurons with ReLU activation
- **Output:** 10 neurons (softmax + cross-entropy loss)
- **Training:** Mini-batch SGD

### Verification Strategy

**Gradient Correctness Theorems:**
```lean
-- Per-operation correctness
theorem relu_gradient_correct : 
  fderiv ℝ relu x = if x > 0 then id else 0

theorem matmul_gradient_correct :
  fderiv ℝ (λ w, w * x) = λ dw, dw * x

theorem cross_entropy_gradient_correct :
  ∂L/∂ŷ = ŷ - y  -- for one-hot target y

-- Composition preserves correctness
theorem chain_rule_preserves_correctness
  (hf : fderiv ℝ f = f') (hg : fderiv ℝ g = g') :
  fderiv ℝ (g ∘ f) = g' ∘ f
```

**Type Safety Theorems:**
```lean
-- Runtime dimensions match type specifications
theorem layer_output_dim {m n : Nat} (layer : DenseLayer m n) (x : Vector n) :
  (layer.forward x).size = m

-- Composition preserves dimensions
theorem composition_type_safe {d1 d2 d3 : Nat} :
  ∀ (l1 : DenseLayer d1 d2) (l2 : DenseLayer d2 d3),
    type-safe composition guaranteed
```

## Success Criteria

### Primary: Gradient Correctness Proofs
- ✅ Prove `fderiv ℝ f = analytical_derivative(f)` for ReLU, matrix ops, softmax, cross-entropy
- ✅ Prove chain rule preserves correctness through composition
- ✅ Establish end-to-end theorem: computed gradient = mathematical gradient

### Secondary: Type Safety Verification
- ✅ Prove type-level dimensions match runtime array sizes
- ✅ Demonstrate type system prevents dimension mismatches
- ✅ Show operations preserve dimension invariants

### Implementation Validation
- Network trains on MNIST
- Gradient checks validate AD against finite differences
- Code compiles and executes

## Development Approach

Development prioritizes working implementations first, then formal verification as design stabilizes:

1. Implement computational code (Float) with basic functionality
2. Iterate until operations work correctly
3. Add formal verification (ℝ) proving mathematical properties
4. Document verification scope in docstrings
5. Incomplete proofs (`sorry`) acceptable during development with TODO comments

See `verified-nn-spec.md` for the detailed implementation plan organized into 10 phases.

## Current Limitations

1. **Implementation completeness:** Many functions use `sorry` placeholders during iterative development
2. **Verification completeness:** Gradient correctness proofs axiomatized (full proofs require deeper mathlib integration)
3. **SciLean:** Early-stage library, API evolving, limited documentation
4. **Performance:** Slower than PyTorch/JAX (proof-of-concept focus, not production-ready)
5. **Float verification:** ℝ vs Float gap acknowledged—we verify symbolic correctness on real numbers
6. **Convergence:** No proofs of SGD convergence (optimization theory out of scope)
7. **Training:** Training loop structure in place but not fully implemented

## Development Notes

Module files created with placeholder implementations marked `sorry`. Structure follows `verified-nn-spec.md` detailing:

- Gradient correctness proof requirements per operation
- Type safety proof requirements
- Implementation phases and task breakdown
- Testing strategy for numerical validation

See `CLAUDE.md` for:
- Development conventions and code style
- SciLean integration patterns
- Proof patterns and tactics
- Known issues and workarounds

## External Resources

### Lean 4 Documentation
- Official docs: https://lean-lang.org/documentation/
- Theorem Proving in Lean 4: https://leanprover.github.io/theorem_proving_in_lean4/
- Mathlib docs: https://leanprover-community.github.io/mathlib4_docs/

### SciLean Resources
- Repository: https://github.com/lecopivo/SciLean
- Documentation (WIP): https://lecopivo.github.io/scientific-computing-lean/

### References
- **Technical Specification:** `verified-nn-spec.md` - Complete implementation roadmap
- **Development Guide:** `CLAUDE.md` - Conventions and patterns
- Certigrad (ICML 2017): Prior work verifying backpropagation in Lean 3
- "Developing Bug-Free Machine Learning Systems With Formal Mathematics" (Selsam et al.)

## License

MIT License - See LICENSE file for details