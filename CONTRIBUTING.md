# Contributing to VerifiedNN

Thank you for your interest in contributing to the VerifiedNN project. This document provides guidelines for contributing to this formally verified neural network implementation in Lean 4.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Code Standards](#code-standards)
5. [Verification Standards](#verification-standards)
6. [Testing Requirements](#testing-requirements)
7. [Documentation Standards](#documentation-standards)
8. [Pull Request Process](#pull-request-process)
9. [Common Contribution Types](#common-contribution-types)
10. [Getting Help](#getting-help)

---

## Code of Conduct

This project follows the [Lean Community Code of Conduct](https://leanprover-community.github.io/code-of-conduct.html). Be respectful, constructive, and professional in all interactions.

---

## Getting Started

### Prerequisites for Contributors

- **Lean 4 experience:** Familiarity with theorem proving and dependent types
- **ML background:** Understanding of neural networks and backpropagation
- **Time commitment:** Most contributions require 2-10 hours

### Finding Issues to Work On

Look for issues labeled:
- `good first issue` - Suitable for new contributors
- `help wanted` - Project maintainers need assistance
- `verification` - Proof completion work
- `enhancement` - New features or improvements
- `documentation` - Documentation improvements

### Before You Start

1. **Open an issue** describing what you want to work on (or comment on existing issue)
2. **Wait for approval** from maintainers (avoids duplicate work)
3. **Fork the repository** and create a feature branch
4. **Read relevant documentation:** ARCHITECTURE.md, GETTING_STARTED.md, module READMEs

---

## Development Setup

### 1. Clone and Build

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/LEAN_mnist.git
cd LEAN_mnist

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/LEAN_mnist.git

# Install dependencies
lake update
lake exe cache get

# Build project
lake build

# Run tests
lake exe smokeTest
```

### 2. Install Development Tools

**Lean 4 LSP (Language Server):**
- VS Code: Install "Lean 4" extension
- Emacs: Use `lean4-mode`
- Neovim: Use `lean.nvim`

**Recommended VS Code Extensions:**
- Lean 4 (official)
- Lean 4 Web Documentation
- Code Spell Checker (for doc comments)

### 3. Configure MCP (Optional but Recommended)

If using Claude Code for development:

```bash
# Add lean-lsp-mcp server
claude mcp add lean-lsp uvx lean-lsp-mcp -e LEAN_PROJECT_PATH=/path/to/LEAN_mnist

# Verify
claude mcp list
```

### 4. Manage Lean Processes

```bash
# Check running Lean servers
pgrep -af lean

# Kill hung servers (if needed)
pkill -f "lean --server"

# Restart LSP (via MCP)
# Use lean_build MCP tool instead of manual pkill
```

---

## Code Standards

### Naming Conventions

```lean
-- Structures: PascalCase
structure DenseLayer (inDim outDim : Nat) where
  weights : Float^[outDim, inDim]
  bias : Float^[outDim]

-- Functions: camelCase
def forwardPass (layer : DenseLayer m n) (x : Float^[n]) : Float^[m] := ...

-- Theorems: snake_case
theorem gradient_correct (f : Float → Float) : ... := by ...

-- Type parameters: lowercase or descriptive
{n : Nat} {inDim outDim : Nat}
```

### Code Style

**Line length:** 100 characters maximum

**Indentation:** 2 spaces (no tabs)

**Import organization:**
```lean
-- 1. Lean standard library
import Lean.Data.Json

-- 2. Mathlib
import Mathlib.Analysis.Calculus.FDeriv.Basic

-- 3. SciLean
import SciLean

-- 4. Project modules (grouped by namespace)
import VerifiedNN.Core.DataTypes
import VerifiedNN.Network.Architecture
```

**Type annotations:**
```lean
-- GOOD: Explicit type signatures for public definitions
def computeGradient (net : MLPArchitecture) (x : Float^[784]) (y : Nat)
  : Float^[nParams] := ...

-- BAD: Missing type signature
def computeGradient net x y := ...
```

**Manual Backpropagation Standard:**
```lean
-- GOOD: Use manual backprop for training code
def networkGradient (params : Float^[n]) (x : Float^[784]) (y : Nat)
  : Float^[n] :=
  -- Forward pass with caching
  let z1 := W1 * x + b1
  let h1 := relu z1
  -- Backward pass with explicit chain rule
  let dL_dz2 := y_hat - y_onehot
  ...

-- BAD: Don't use noncomputable AD in training code
def networkGradient (params : Float^[n]) (x : Float^[784]) (y : Nat)
  : Float^[n] :=
  (∇ p, loss (unflattenParams p) x y) params  -- Noncomputable!
```

### Performance Guidelines

**Use efficient array types:**
```lean
-- GOOD: SciLean's DataArrayN (efficient)
def process (x : Float^[n]) : Float^[n] := ...

-- BAD: Standard Array (inefficient for numerics)
def process (x : Array Float) : Array Float := ...
```

**Mark hot-path functions:**
```lean
@[inline]
@[specialize]
def matrixMultiply {m n k : Nat} (A : Float^[m,n]) (B : Float^[n,k])
  : Float^[m,k] := ...
```

**Avoid unnecessary allocations:**
```lean
-- GOOD: In-place style operations
let result := ⊞ i => a[i] + b[i]

-- BAD: Multiple intermediate arrays
let temp := a.map (· + 1)
let result := temp.zipWith b (· + ·)
```

---

## Verification Standards

### Proof Requirements

**All public theorems must:**
1. Have clear docstrings explaining what's proven
2. State properties on ℝ (real numbers) when applicable
3. Document the Float ↔ ℝ correspondence if relevant
4. Include references to literature or mathlib lemmas

**Example:**
```lean
/-- Gradient of ReLU activation is correct.

Proves that the manually computed derivative matches the analytical
derivative: ∂ReLU(x)/∂x = (x > 0) ? 1 : 0.

This is proven on ℝ and assumed to hold for Float via the
`float_real_correspondence` axiom.

**References:** Standard calculus, ReLU is piecewise linear.
-/
theorem relu_gradient_correct (x : Float) :
  reluDerivative x = if x > 0 then 1 else 0 := by
  unfold reluDerivative relu
  simp [ite_apply]
```

### Handling Incomplete Proofs

**Sorries are acceptable IF:**
1. Proof strategy is documented in TODO comment
2. References to relevant lemmas provided
3. Issue created to track completion

**Example:**
```lean
-- TODO: Prove flatten and unflatten are inverses
-- Strategy: Use DataArrayN.ext and unfold both definitions
-- Needs: Lemma about indexed access preservation
-- See issue #42 for full strategy
theorem flatten_unflatten_inverse (x : Float^[nParams]) :
  flatten (unflatten x) = x := by
  sorry
```

### Axiom Usage

**Axioms require:**
1. Comprehensive docstring (minimum 20 lines for non-trivial axioms)
2. Justification for why axiomatizing is acceptable
3. References to literature or specifications
4. Clear statement of what's being assumed

**Example:** See `VerifiedNN/Loss/Properties.lean` lines 121-180 for exemplary axiom documentation.

### Proof Style

**Prefer explicit tactics:**
```lean
-- GOOD: Clear tactic sequence
theorem composition_correct {f g : Float → Float} :
  derivative (g ∘ f) = (derivative g ∘ f) * derivative f := by
  unfold derivative composition
  fun_trans
  simp only [mul_comm]
  ring

-- Note: ACCEPTABLE: Auto tactics if proof is obvious
theorem trivial_lemma : 2 + 2 = 4 := by norm_num
```

**Use specialized tactics:**
- `fun_trans` for differentiation
- `fun_prop` for continuity/differentiability
- `simp` with explicit lemma lists when possible
- `norm_num` for numeric computations

---

## Testing Requirements

### Unit Tests

**All new functions should have smoke tests:**

```lean
-- In VerifiedNN/Testing/UnitTests.lean
def test_newActivation : TestSeq :=
  test "newActivation basic properties" do
    let x : Float := 1.5
    let result := newActivation x

    -- Test non-negative output
    check (result >= 0.0) "Output should be non-negative"

    -- Test derivative exists
    let deriv := newActivationDerivative x
    check (not deriv.isNaN) "Derivative should be finite"
```

### Integration Tests

**Major features need integration tests:**
- Forward pass with new layer type
- Gradient computation end-to-end
- Training loop with new optimizer

### Gradient Checking

**New differentiable operations require gradient checks:**

```lean
-- Compare manual gradient vs finite differences
def checkGradient (f : Float^[n] → Float) (manualGrad : Float^[n] → Float^[n])
  (x : Float^[n]) (ε : Float := 1e-5) : Bool :=
  let manual := manualGrad x
  let numerical := ⊞ i =>
    let x_plus := ⊞ j => if i = j then x[j] + ε else x[j]
    let x_minus := ⊞ j => if i = j then x[j] - ε else x[j]
    (f x_plus - f x_minus) / (2 * ε)

  -- Check relative error < 1%
  let relError := ‖manual - numerical‖ / max ‖manual‖ ‖numerical‖
  relError < 0.01
```

### Performance Tests

**Performance-critical changes need benchmarks:**

```bash
# Before changes
lake exe performanceTest > before.txt

# After changes
lake exe performanceTest > after.txt

# Compare
diff before.txt after.txt
```

---

## Documentation Standards

### Module-Level Documentation

**Every new `.lean` file must have:**

```lean
/-!
# Module Name

Brief overview of the module's purpose.

## Main Definitions
- `DefOne`: Description
- `DefTwo`: Description

## Main Results
- `theorem_name`: Statement

## Implementation Notes
- Design decisions
- Dependencies
- Verification status

## References
- Papers, specifications, or mathlib modules
-/
```

### Definition Docstrings

**Every public definition/theorem needs:**

```lean
/-- Brief one-line summary.

Detailed explanation with mathematical context.

**Verified properties:** List of proven properties.

**Parameters:**
- `param1`: Description with type info
- `param2`: Description

**Returns:** Description of return value

**Implementation notes:** Performance, numerical stability, etc.

**References:** Cite papers, mathlib lemmas, or specifications
-/
def myFunction ...
```

### Directory READMEs

**New directories must have README.md:**
- Purpose statement
- Module descriptions
- Usage examples
- Dependencies
- Verification status

---

## Pull Request Process

### 1. Pre-PR Checklist

Before opening a PR, ensure:

- [ ] Code compiles with zero errors (`lake build`)
- [ ] No new linter warnings (except documented sorries)
- [ ] All tests pass (`lake exe smokeTest`)
- [ ] Documentation added for all public definitions
- [ ] Sorries documented with TODO + strategy
- [ ] Git history is clean (squash work-in-progress commits)
- [ ] Branch is up to date with main

### 2. PR Description Template

```markdown
## Description
Brief summary of changes (2-3 sentences).

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update
- [ ] Verification work (proof completion)

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Smoke tests pass
- [ ] Integration tests added/updated
- [ ] Gradient checks pass (if applicable)

## Verification
- [ ] New theorems proven (list them)
- [ ] Sorries documented with strategies
- [ ] Axioms justified in docstrings

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Builds successfully

## Related Issues
Closes #42
```

### 3. PR Review Process

**Expect feedback on:**
- Proof strategy and tactics used
- Code style and naming
- Performance implications
- Documentation completeness
- Test coverage

**Review timeline:**
- Small PRs (<100 lines): 1-3 days
- Medium PRs (100-500 lines): 3-7 days
- Large PRs (>500 lines): 1-2 weeks

**Responding to feedback:**
- Address all comments (or explain why not)
- Push new commits (don't force-push during review)
- Mark resolved comments
- Request re-review when ready

### 4. Merging

PRs are merged when:
- All tests pass
- At least one maintainer approval
- All review comments addressed
- Build is green
- Documentation complete

---

## Common Contribution Types

### 1. Completing Sorries

**Current sorries (4 total):** All in `VerifiedNN/Verification/TypeSafety.lean`

**Process:**
1. Read the TODO comment for strategy
2. Attempt proof following suggested approach
3. If stuck, ask on Lean Zulip or open discussion issue
4. Submit PR when proof is complete

**Resources:**
- Mathlib documentation: https://leanprover-community.github.io/mathlib4_docs/
- Lean Zulip: https://leanprover.zulipchat.com/ (#mathlib4, #new members)

### 2. Adding New Layer Types

**Steps:**
1. Define layer structure with dependent types
2. Implement forward pass (computable)
3. Implement backward pass (manual backprop)
4. Prove gradient correctness
5. Add integration test
6. Document with examples

**Example structure:**
```
VerifiedNN/Layer/Conv2D.lean
├── Structure definition
├── Forward pass
├── Backward pass (manual gradients)
├── Initialization (He/Xavier)
└── Tests

VerifiedNN/Verification/Conv2DGradients.lean
├── Gradient correctness theorems
├── Dimension safety proofs
└── Composition properties
```

### 3. Performance Optimization

**Focus areas:**
- Matrix operations (use OpenBLAS)
- Batch processing (parallelize when safe)
- Memory allocation (reduce copies)

**Requirements:**
- Benchmark before/after
- No correctness regressions
- Document trade-offs

### 4. Documentation Improvements

**High-value contributions:**
- Tutorial examples
- API reference clarity
- Architecture diagrams
- Troubleshooting guides

**No approval needed for:**
- Typo fixes
- Broken link repairs
- Grammar improvements

---

## Getting Help

### Communication Channels

**Lean Zulip Chat:**
- #scientific-computing - SciLean and numerical computing
- #mathlib4 - Proof assistance and lemma finding
- #new members - Beginner questions

**GitHub Discussions:**
- Q&A - General questions
- Ideas - Feature proposals
- Show and tell - Share your work

**Issues:**
- Bug reports
- Feature requests
- Documentation improvements

### Asking Good Questions

**Include:**
1. What you're trying to achieve
2. What you've tried so far
3. Minimal reproducible example
4. Relevant error messages
5. Lean version (`lean --version`)

**Example:**
```
I'm trying to prove that my custom activation's gradient is correct,
but I'm getting a type mismatch error.

Goal state at line 42:
  ⊢ Float^[n] = Float^[m]

I've tried using `DataArrayN.ext` but it doesn't apply.

Minimal example:
[paste code]

Lean version: v4.20.1
```

### Mentorship

New contributors can request mentorship by:
1. Opening an issue tagged `mentorship`
2. Describing your background and goals
3. Proposing a contribution area

Maintainers will:
- Suggest appropriate issues
- Provide guidance on approach
- Review PRs with educational feedback

---

## Attribution

Contributors are credited in:
- Git commit history
- AUTHORS file (maintained by maintainers)
- Release notes (for significant contributions)

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

**Thank you for contributing to verified machine learning.**

For questions about these guidelines, open a discussion or ask in the Lean Zulip #scientific-computing channel.

---

**Last Updated:** November 21, 2025
