# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# Verified Neural Network Training in Lean 4

## Project Overview

**Primary Goal:** Prove that automatic differentiation computes mathematically correct gradients. For every differentiable operation in the network, formally verify that `fderiv ℝ f = analytical_derivative(f)`, and prove that composition via chain rule preserves correctness through the entire network.

**Secondary Goal:** Leverage dependent types to enforce dimension consistency at compile time, proving that type-checked operations maintain correct tensor dimensions at runtime.

**Implementation:** Train an MLP on MNIST using Lean 4, SciLean, and SGD with backpropagation, where the core gradient computation is formally verified.

**Verification Philosophy:** Mathematical properties proven on ℝ (real numbers), computational implementation in Float (IEEE 754). The Float→ℝ gap is acknowledged—we verify symbolic correctness, not floating-point numerics.

## Current Implementation Status

Most modules contain placeholder implementations marked with `sorry`. The project structure is complete and ready for iterative development following the [verified-nn-spec.md](verified-nn-spec.md) roadmap. Focus on building working implementations first, then adding verification as design stabilizes.

## Tech Stack

```
Lean: v4.23.0 (specified in lean-toolchain)
SciLean: latest compatible version (master branch)
mathlib4: (via SciLean)
LSpec: testing framework
OpenBLAS: system package (for performance)
Platform: Linux/macOS preferred (Windows support depends on SciLean)
```

## Build Commands

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

# Data and benchmarking
./scripts/download_mnist.sh    # Download MNIST dataset
./scripts/benchmark.sh          # Run performance benchmarks
```

## Project Structure

```
VerifiedNN/
├── Core/              # Fundamental types, linear algebra, activations
├── Layer/             # Dense layers with differentiability proofs
├── Network/           # MLP architecture, initialization, gradients
├── Loss/              # Cross-entropy with mathematical properties
├── Optimizer/         # SGD implementation
├── Training/          # Training loop, batching, metrics
├── Data/              # MNIST loading and preprocessing
├── Verification/      # Formal proofs (gradient correctness, type safety, convergence)
├── Testing/           # Unit tests, integration tests, gradient checking
└── Examples/          # Minimal examples and full MNIST training

scripts/
├── download_mnist.sh  # MNIST dataset retrieval
└── benchmark.sh       # Performance benchmarks
```

## Lean 4 Conventions

### Naming
- **Structures:** `PascalCase` (e.g., `DenseLayer`, `MLPArchitecture`)
- **Functions:** `camelCase` (e.g., `forwardPass`, `computeGradient`)
- **Theorems:** `snake_case` (e.g., `gradient_correct`, `layer_composition_type_safe`)
- **Type parameters:** lowercase single letters or descriptive (e.g., `{n : Nat}`, `{inDim outDim : Nat}`)

### Import Style
```lean
import SciLean
import Mathlib.Analysis.Calculus.FDeriv.Basic
import VerifiedNN.Core.DataTypes

set_default_scalar Float  -- For numerical code
```

### Type Signatures
- Always include explicit type signatures for public definitions
- Use dependent types for dimension tracking: `Vector (n : Nat) := Float^[n]`
- Leverage type inference in local definitions only

### Proof Style
- Use `by` blocks with explicit tactic sequences
- Prefer `fun_trans` for differentiation proofs
- Use `fun_prop` for differentiability/continuity
- Apply `simp` with explicit lemma lists when possible
- Document proof strategy in comments for complex theorems

## SciLean Integration Patterns

### Differentiation
```lean
-- Compute gradient using SciLean's AD
def computeGradient (f : Float^[n] → Float) (x : Float^[n]) : Float^[n] :=
  (∇ x', f x') x
    |>.rewrite_by fun_trans (disch := aesop)

-- Register custom differentiable operation
@[fun_prop]
theorem myFunc_differentiable : Differentiable Float myFunc := by
  unfold myFunc
  fun_prop

@[fun_trans]
theorem myFunc_fderiv : fderiv Float myFunc = ... := by
  unfold myFunc
  fun_trans
```

### Performance
- Use `Float^[n]` (DataArrayN) not `Array Float` for numerical arrays
- Mark hot-path functions `@[inline]` and `@[specialize]`
- Use `IndexType` instead of `Fintype` for better performance
- Leverage OpenBLAS via SciLean's matrix operations

### Common Pitfalls
- **Don't** use division without proving denominator nonzero
- **Don't** use `sqrt` or `log` without positivity proofs
- **Do** use regularized versions: `sqrt (ε + x²)` for small ε > 0
- **Don't** branch in differentiable functions—breaks AD

## Code Style Guidelines

### General
- Maximum line length: 100 characters
- Use 2-space indentation
- Add docstrings to all public definitions
- Group imports by: Lean stdlib, mathlib, SciLean, project modules

### Docstrings
```lean
/-- Compute forward pass through dense layer.

Given a layer with weight matrix `W : Float^[m,n]` and bias `b : Float^[m]`,
computes `σ(Wx + b)` where σ is the activation function.

**Verified properties:** Dimension consistency, differentiability.

**Parameters:**
- `layer`: Dense layer structure
- `x`: Input vector of dimension n

**Returns:** Output vector of dimension m -/
def DenseLayer.forward {m n : Nat}
  (layer : DenseLayer m n) (x : Float^[n]) : Float^[m] := ...
```

### Mathematical Comments
```lean
-- Gradient of cross-entropy: ∂L/∂ŷ = ŷ - y (one-hot target)
-- This is proven in Verification/Loss.lean theorem ce_gradient_correct
def crossEntropyGrad := ...
```

### Error Handling
- Use `Option` for operations that may fail (division, indexing)
- Use `IO` for operations with effects (file reading, randomness)
- Document failure conditions in docstrings

## Verification Workflow

### Project Goals

**Primary: Gradient Correctness**
- For each differentiable operation (ReLU, matrix multiply, softmax, cross-entropy), prove `fderiv ℝ f = analytical_derivative(f)`
- Prove chain rule application preserves correctness through layer composition
- Verify end-to-end: the gradient computed by automatic differentiation equals the mathematical gradient

**Secondary: Type Safety**
- Prove type-level dimension specifications correspond to runtime array dimensions
- Show operations on `DataArrayN` preserve size invariants
- Demonstrate type system prevents dimension mismatches by construction

**Implementation Validation:**
- Train functional MLP on MNIST with reasonable accuracy
- Numerically validate AD against finite differences
- Confirm implementation matches verified specification

**Out of Scope:**
- Floating-point numerical stability (ℝ vs Float gap acknowledged)
- Convergence properties of SGD (optimization theory)
- Generalization bounds or learning theory

### Proof Patterns
```lean
-- Pattern 1: Gradient correctness
theorem activation_gradient_correct (f : Float → Float) :
  fderiv Float f = ... := by
  unfold f
  fun_trans
  simp [...]

-- Pattern 2: Dimension consistency
theorem layer_output_dim {m n : Nat} (layer : DenseLayer m n) (x : Float^[n]) :
  (layer.forward x).size = m := by
  unfold DenseLayer.forward
  simp [DataArrayN.size]

-- Pattern 3: Composition preserves properties
theorem composition_differentiable {f g : Float^[n] → Float^[m]}
  (hf : Differentiable Float f) (hg : Differentiable Float g) :
  Differentiable Float (g ∘ f) := by
  apply Differentiable.comp hg hf
```

### Axiom Usage
- **Acceptable for research:** Convergence proofs, Float ≈ ℝ correspondence statements
- **Minimize where practical:** Core type safety and gradient correctness proofs
- Check axioms: `#print axioms theorem_name` or `lean --print-axioms file.lean`
- Document rationale for axiomatized proofs in comments

## Mathematical Notation

### Lean vs LaTeX
```
ℝ         → \R (Real)
∇ f       → \nabla (gradient)
‖x‖       → \| (norm)
⟪x, y⟫    → \< \> (inner product)
∑ i, x[i] → \Sum (sum)
Float^[n] → exponent [n] for fixed-size array
```

### Common Definitions
- `Vector n := Float^[n]` (column vectors)
- `Matrix m n := Float^[m, n]` (m×n matrices)
- `Batch b n := Float^[b, n]` (mini-batch, b samples)

## Development Workflow

### Iterative Development Approach
Development follows an iterative pattern focused on building working implementations first, then adding verification as understanding deepens:

1. Create feature branch: `git checkout -b feature/layer-batch-norm`
2. Implement computational code (Float) with basic tests
3. Iterate until functionality works as expected
4. Add formal verification (ℝ) when design stabilizes
5. Document with docstrings explaining verification scope
6. PR when ready (can include `sorry` for incomplete proofs if documented)

Note: Code with incomplete proofs is acceptable during development—mark with TODO comments explaining what needs verification.

### Debugging Proofs
- Use `#check` to inspect types
- Use `#print` to see definitions
- Use `trace.Meta.Tactic.simp` for simp debugging
- Use `set_option trace.fun_trans true` for AD debugging
- Check `sorry` locations: `grep -r "sorry" VerifiedNN/`

### Performance Profiling
```lean
set_option profiler true
set_option trace.profiler.threshold 10  -- Show >10ms operations

#eval timeit "gradient computation" do
  let grad := computeGradient loss params
  pure ()
```

### Memory Debugging
```lean
-- Check if value is shared (RC > 1)
let x := dbgTraceIfShared "x is shared!" someArray
```

## Production Readiness Guidelines

These represent the ultimate standard for production-quality code. During development, deviations are acceptable and should be documented with TODO comments.

### Critical Standards
- **Type Safety:** Use dependent types for dimension tracking where it enhances correctness
- **Numerical Arrays:** Prefer `Float^[n]` (DataArrayN) over `Array Float` for performance
- **Differentiability:** Register new differentiable operations with `@[fun_trans]` and `@[fun_prop]`
- **Verification Scope:** Clearly document what is proven vs. tested vs. axiomatized
- **Float vs ℝ Gap:** Acknowledge the verification boundary in docstrings

### Quality Markers
- Gradient checks validate symbolic derivatives match numerical approximations
- Loss decreases during training (basic sanity check)
- Code compiles without warnings
- Proofs minimize axiom usage (when practical)
- Docstrings explain verification status

### Anti-Patterns to Avoid
- Claiming Float properties are proven (only ℝ properties are verified)
- Using `Array Float` in hot paths (performance penalty)
- Branching within differentiable code paths (breaks automatic differentiation)
- Division/sqrt/log without nonzero/positive handling
- Premature optimization before correctness is established

## Known Limitations

### SciLean (Early Stage Library)
- API may change, performance is being optimized
- CPU-only via OpenBLAS (no GPU support)
- Function inversion very limited (mostly for sum reindexing)
- Many examples in SciLean's `examples/` directory don't compile

### Lean 4 Numerical Computing
- **Float is opaque:** Cannot prove `(0.0 : Float) = (0.0 + 0.0)` with `rfl`
- **No Float theory:** No canonical verified Float library (unlike Coq's Flocq)
- **Compilation time:** mathlib is large, use `lake exe cache get`
- **Type class slowness:** Complex numerical types may timeout

### Common Errors
- **Dimension mismatch:** Use dependent types to catch at compile time
- **Division by zero:** Prove denominator nonzero or use regularization
- **AD through branches:** Avoid `if` in differentiable code paths
- **Accidental array copies:** Maintain unique references for in-place updates
- **Slow sums:** Use `∑ᴵ` indexed sums instead of `∑` with `Fintype`

## Performance Expectations

### Build Times
- Clean build: Depends on mathlib cache availability
- Incremental: Typically fast for single module changes
- Use `lake exe cache get` to download precompiled mathlib binaries

### Runtime Performance
- Gradient computation slower than PyTorch (acceptable for proof-of-concept)
- OpenBLAS integration helps numerical operations
- Profile with `timeit` and `set_option profiler true` to identify bottlenecks

## External Resources

### Essential Documentation
- Lean 4 Official: https://lean-lang.org/documentation/
- Theorem Proving in Lean 4: https://leanprover.github.io/theorem_proving_in_lean4/
- Mathlib4 docs: https://leanprover-community.github.io/mathlib4_docs/
- SciLean Repository: https://github.com/lecopivo/SciLean
- Lean Zulip chat: https://leanprover.zulipchat.com/ (channels: #scientific-computing, #mathlib4, #new members)

### References
- Certigrad (ICML 2017): Prior work on verified backpropagation in Lean 3
- "Developing Bug-Free Machine Learning Systems With Formal Mathematics" (Selsam et al.)
- Full technical specification: [verified-nn-spec.md](verified-nn-spec.md)

## Critical Reminders for Claude Code

### During Active Development
- Incomplete proofs (`sorry`) are acceptable with TODO comments explaining what needs verification
- Focus on building working implementations before perfecting proofs
- Iterate on design before committing to formal verification
- Document verification scope clearly in docstrings
- Flag areas where verification is aspirational vs. complete

### When in Doubt
- Consult SciLean examples and documentation
- Check mathlib for existing analysis lemmas
- Reference [verified-nn-spec.md](verified-nn-spec.md) for detailed implementation guidance
- Ask on Lean Zulip #scientific-computing (Tomáš Skřivan, SciLean author, is responsive)

---

**Last Updated:** 2024-10-20
**Maintained by:** Project contributors
