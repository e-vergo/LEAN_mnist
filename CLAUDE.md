# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# Verified Neural Network Training in Lean 4

## Project Overview

**Primary Goal:** Prove that automatic differentiation computes mathematically correct gradients. For every differentiable operation in the network, formally verify that `fderiv ℝ f = analytical_derivative(f)`, and prove that composition via chain rule preserves correctness through the entire network.

**Secondary Goal:** Leverage dependent types to enforce dimension consistency at compile time, proving that type-checked operations maintain correct tensor dimensions at runtime.

**Implementation:** Train an MLP on MNIST using Lean 4, SciLean, and SGD with backpropagation, where the core gradient computation is formally verified.

**Verification Philosophy:** Mathematical properties proven on ℝ (real numbers), computational implementation in Float (IEEE 754). The Float→ℝ gap is acknowledged—we verify symbolic correctness, not floating-point numerics.

**New to this project?** See [START_HERE.md](START_HERE.md) for a quick 5-minute overview or [GETTING_STARTED.md](GETTING_STARTED.md) for comprehensive onboarding with installation instructions.

**Finding documentation:** See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for a complete guide to all documentation organized by experience level and task.

## Current Implementation Status

**Build Status:** ✅ **All 40 Lean files compile successfully with ZERO errors**

**Verification Progress:**
- **Sorries:** 17 strategic placeholders (all documented with completion strategies)
  - Network/Gradient.lean: 7 (index arithmetic bounds)
  - Verification/GradientCorrectness.lean: 6 (mathlib integration)
  - Verification/TypeSafety.lean: 2 (flatten/unflatten inverses)
  - Layer/Properties.lean: 1 (affine combination)
- **Axioms:** 9 total (8 convergence theory + 1 Float bridge, all justified)

**Documentation:** 100% coverage with mathlib-quality standards
- All 10 directories have comprehensive READMEs (~103KB total)
- All sorries documented with proof strategies
- All axioms documented with justification and references
- See [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) for detailed cleanup report

**Development Philosophy:** Build working implementations first, add verification as design stabilizes. The codebase is now ready for systematic proof completion following the [verified-nn-spec.md](verified-nn-spec.md) roadmap.

## Tech Stack

```
Lean: v4.23.0 (specified in lean-toolchain)
SciLean: latest compatible version (master branch)
mathlib4: (via SciLean)
LSpec: testing framework
OpenBLAS: system package (for performance)
Platform: Linux/macOS preferred (Windows support depends on SciLean)
```

## MCP Integration (lean-lsp-mcp)

This project uses the **lean-lsp-mcp** Model Context Protocol server to enable AI-assisted development with deep Lean language awareness. The MCP server provides real-time access to LSP diagnostics, goal states, term information, and external theorem search tools.

### Setup & Configuration

The MCP server is configured in `~/.claude.json` and automatically starts when using Claude Code:

```bash
# One-time setup (already done for this project)
claude mcp add lean-lsp uvx lean-lsp-mcp -e LEAN_PROJECT_PATH=/Users/eric/LEAN_mnist

# Verify MCP server is connected
claude mcp list

# Restart Claude Code to activate the MCP server
```

**Prerequisites:**
- `uv` package manager installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Project built at least once (`lake build`) to avoid LSP timeouts
- `ripgrep` (`rg`) for local search functionality

### Available MCP Tools

The lean-lsp-mcp server provides 15 specialized tools for Lean development:

#### File & LSP Interactions
1. **lean_file_contents** - Get file contents with optional line number annotations
2. **lean_diagnostic_messages** - Get all diagnostics (infos, warnings, errors) for a file
3. **lean_goal** - Get proof goal at a specific location (line or line & column)
4. **lean_term_goal** - Get term goal at a specific position (line & column)
5. **lean_hover_info** - Retrieve hover documentation for symbols, terms, and expressions
6. **lean_declaration_file** - Get the file where a symbol/term is declared
7. **lean_completions** - Code auto-completion: find available identifiers or imports
8. **lean_run_code** - Run/compile an independent Lean code snippet and return results
9. **lean_multi_attempt** - Attempt multiple code snippets and return goal states/diagnostics

#### Search Tools
10. **lean_local_search** - Search definitions and theorems in local project and stdlib (requires `rg`)

#### External Search Tools (rate-limited: 3 requests per 30 seconds)
11. **lean_leansearch** - Natural language search via leansearch.net (mixed queries, Lean terms)
12. **lean_loogle** - Search via loogle.lean-lang.org (by name, subexpression, type, conclusion)
13. **lean_state_search** - Search applicable theorems for current goal via premise-search.com
14. **lean_hammer_premise** - Find relevant premises using Lean Hammer Premise Search

#### Project Management
15. **lean_build** - Rebuild project and **restart the Lean LSP server**

### Managing Lean Language Servers

**IMPORTANT:** The Lean LSP can spawn multiple server processes, consuming significant memory and CPU. Carefully manage server instances to avoid resource exhaustion.

#### Best Practices
```bash
# Check running Lean processes
pgrep -af lean

# Kill all Lean language servers (if unresponsive or consuming too much memory)
pkill -f "lean --server"

# Kill all Lake processes
pkill -f lake

# Restart LSP via MCP (preferred method when working through Claude Code)
# Use the lean_build MCP tool instead of manual pkill
```

#### When to Restart the LSP
- After major code changes that affect many files
- When diagnostics become stale or incorrect
- If the LSP becomes unresponsive or slow
- After `lake update` or dependency changes
- When memory usage grows excessively (check with `htop` or Activity Monitor)

#### Memory Management Tips
- Build the project manually before starting MCP (`lake build`) to cache dependencies
- Use `lake exe cache get` to download precompiled mathlib (avoids expensive rebuilds)
- Limit concurrent file analysis by working on one module at a time
- Restart the LSP periodically during long coding sessions
- Close unused Lean files in your editor to reduce LSP load

### Workflow Integration

#### During Proof Development
1. Use **lean_goal** to inspect current proof state
2. Use **lean_leansearch** or **lean_loogle** to find relevant theorems
3. Use **lean_completions** to discover available tactics/lemmas
4. Use **lean_diagnostic_messages** to check for errors
5. Use **lean_hover_info** to understand term types and documentation

#### During Implementation
1. Use **lean_file_contents** to review existing code
2. Use **lean_local_search** to find similar patterns in the codebase
3. Use **lean_run_code** to test code snippets before integration
4. Use **lean_multi_attempt** to explore multiple implementation approaches
5. Use **lean_declaration_file** to navigate to dependencies

#### Debugging & Troubleshooting
1. Check **lean_diagnostic_messages** for detailed error information
2. Use **lean_goal** to verify proof state matches expectations
3. Use **lean_term_goal** for term mode elaboration issues
4. Use **lean_build** to rebuild and restart LSP if diagnostics are stale
5. Use manual `pkill -f lean` if LSP becomes completely unresponsive

### Rate Limiting & External Services

External search tools (leansearch, loogle, state_search, hammer_premise) are rate-limited to **3 requests per 30 seconds**. Plan queries strategically and prefer local search when possible.

### Configuration Files

- **MCP Config:** `~/.claude.json` - Global MCP server configuration
- **LSP Logs:** Check logs if the MCP server fails to start (location varies by system)
- **Verify configuration:** Run `claude mcp list` to see configured servers

### Troubleshooting

**Problem:** MCP server timeout on startup
**Solution:** Build the project first (`lake build`), then restart Claude Code

**Problem:** Diagnostics not updating
**Solution:** Use `lean_build` MCP tool to restart LSP

**Problem:** Multiple Lean servers consuming memory
**Solution:** `pkill -f "lean --server"` then use `lean_build` to restart cleanly

**Problem:** Rate limit errors on external search
**Solution:** Wait 30 seconds or use `lean_local_search` instead

**Problem:** MCP tools not available in Claude Code
**Solution:** Restart Claude Code after adding MCP configuration

## Build Commands

```bash
# Setup
lake update                    # Update dependencies
lake exe cache get             # Download precompiled mathlib

# Build
lake build                     # Build entire project
lake build VerifiedNN.Core.DataTypes  # Build specific module


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

**Detailed architecture documentation:** See [ARCHITECTURE.md](ARCHITECTURE.md) for module dependency graphs, call flow diagrams, and design decisions.

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

## Repository Cleanup Standards

This repository has been cleaned to mathlib submission quality. All new code and documentation must adhere to these standards.

### Documentation Standards (Mandatory)

**Module-level docstrings** (use `/-!` format):
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
-/
```

**Definition/theorem docstrings** (use `/--` format):
```lean
/-- Brief one-line summary.

Detailed explanation with mathematical context.

**Verified properties:** List of proven properties.

**Parameters:**
- `param1`: Description with type info
- `param2`: Description

**Returns:** Description of return value

**References:** Cite papers, mathlib lemmas, or specifications
-/
def myFunction ...
```

### Sorry and Axiom Documentation (Mandatory)

**Every `sorry` must have:**
1. Immediately preceding TODO comment explaining what needs to be proven
2. Strategy notes on how to complete the proof
3. References to relevant mathlib lemmas if known

Example:
```lean
-- TODO: Prove flatten and unflatten are inverses
-- Strategy: Use DataArrayN.ext and unfold both definitions
-- Needs: Lemma about indexed access preservation
theorem flatten_unflatten_inverse : flatten (unflatten x) = x := by
  sorry
```

**Every axiom must have:**
1. Comprehensive docstring (minimum 20 lines for non-trivial axioms)
2. Explanation of what is being axiomatized and why
3. Justification for why it's acceptable (out of scope, Float/ℝ gap, etc.)
4. References to literature or specifications

See [Loss/Properties.lean](VerifiedNN/Loss/Properties.lean#L121-180) for exemplary 58-line axiom documentation.

### Code Quality Standards

**Mandatory checks before committing:**
- [ ] Zero build errors (`lake build` succeeds)
- [ ] Zero linter warnings (unused variables, imports, etc.)
- [ ] All public definitions have docstrings
- [ ] All sorries have TODO comments with strategies
- [ ] Mathematical notation uses Unicode (∀, ∃, →, ℝ, ∇)
- [ ] Imports organized: Lean stdlib, mathlib, SciLean, project modules
- [ ] No commented-out code (delete or document why it's kept)

### Directory Structure Requirements

Each `VerifiedNN/` subdirectory must have a `README.md` containing:
1. **Purpose**: What this directory provides
2. **Module Descriptions**: File-by-file breakdown with verification status
3. **Key Concepts**: Mathematical background if applicable
4. **Dependencies**: Import hierarchy
5. **Usage Examples**: Code snippets showing how to use the modules
6. **Verification Status**: Sorry count, axiom count, build status
7. **Last Updated**: Date of last significant change

See existing directory READMEs for examples (all 10 are complete).

### Refactoring Guidelines

**When to split a file:**
- File exceeds 500 lines (guideline, not hard rule)
- Natural conceptual boundaries exist (axioms vs. lemmas, core vs. derived)
- Dependencies can be cleanly separated

**When splitting files:**
1. Create subdirectory if multiple files share a namespace
2. Use re-export pattern for backward compatibility
3. Update all imports across the codebase
4. Verify build succeeds after refactoring
5. Document the split in directory README

**Example:** [Verification/Convergence/](VerifiedNN/Verification/Convergence/) split (802→112 line main file)

### Build Hygiene

**Always maintain:**
- Zero compilation errors across all files
- Only expected sorry warnings (all documented)
- Clean `lake build` output (no unexpected warnings)

**Before major changes:**
```bash
# Clean build verification
lake clean && lake build 2>&1 | tee build-log.txt

# Count sorries
grep -r "sorry" VerifiedNN/*.lean VerifiedNN/*/*.lean | wc -l

# Check for linter warnings
lake build 2>&1 | grep -i "warning" | grep -v "sorry" | grep -v "openblas"
```

**Process management:**
- Monitor Lean LSP processes: `pgrep -af lean`
- Kill if resource consumption excessive: `pkill -f "lean --server"`
- Limit parallel agent spawning to avoid LSP pile-up (max 3-4 agents)

### Quality Gate Checklist

Before considering a module "complete":
- [x] All files build with zero errors
- [x] Zero non-sorry warnings
- [x] Directory README exists and is comprehensive
- [x] All sorries documented with strategies
- [x] All axioms justified with detailed docstrings
- [x] Code follows mathlib comment standards
- [x] Cross-references to related modules documented

See [DOCUMENTATION_ENHANCEMENT_REPORT.md](DOCUMENTATION_ENHANCEMENT_REPORT.md) for documentation quality improvements.

## Verification Workflow

**Detailed proof development guide:** See [VERIFICATION_WORKFLOW.md](VERIFICATION_WORKFLOW.md) for step-by-step instructions on developing formal proofs in this codebase.

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
- **MCP Tools (Preferred):**
  - Use `lean_goal` to inspect proof state at specific locations
  - Use `lean_diagnostic_messages` for detailed error analysis
  - Use `lean_hover_info` to check types and documentation
  - Use `lean_term_goal` for term mode elaboration issues
- **In-Code Debugging:**
  - Use `#check` to inspect types
  - Use `#print` to see definitions
  - Use `trace.Meta.Tactic.simp` for simp debugging
  - Use `set_option trace.fun_trans true` for AD debugging
- **Code Search:**
  - Check `sorry` locations: Use `lean_local_search` or `grep -r "sorry" VerifiedNN/`

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
- Running 'lake clean' at any point. This causes all of mathlib to be rebuilt and is time consuming. DO NOT USE IT EVER

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

### Internal Documentation (Start Here!)

**Essential reading for this project:**
- **[START_HERE.md](START_HERE.md)** - Quick 5-minute project overview
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Comprehensive setup and onboarding
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and module dependencies
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Complete testing handbook
- **[COOKBOOK.md](COOKBOOK.md)** - Practical recipes and examples
- **[VERIFICATION_WORKFLOW.md](VERIFICATION_WORKFLOW.md)** - Proof development guide
- **[verified-nn-spec.md](verified-nn-spec.md)** - Complete technical specification

### Essential External Documentation
- Lean 4 Official: https://lean-lang.org/documentation/
- Theorem Proving in Lean 4: https://leanprover.github.io/theorem_proving_in_lean4/
- Mathlib4 docs: https://leanprover-community.github.io/mathlib4_docs/
- SciLean Repository: https://github.com/lecopivo/SciLean
- Lean Zulip chat: https://leanprover.zulipchat.com/ (channels: #scientific-computing, #mathlib4, #new members)

### Academic References
- Certigrad (ICML 2017): Prior work on verified backpropagation in Lean 3
- "Developing Bug-Free Machine Learning Systems With Formal Mathematics" (Selsam et al.)

## Critical Reminders for Claude Code

### Using MCP Tools (First Priority)
When working with Lean code in this project, **always leverage the MCP tools** as your primary interface:

- **Before editing:** Use `lean_file_contents` to review current implementation
- **During proofs:** Use `lean_goal` to inspect proof states at specific lines
- **For errors:** Use `lean_diagnostic_messages` to get detailed error information
- **Finding theorems:** Use `lean_leansearch` (natural language) or `lean_loogle` (type search) before searching manually
- **Understanding code:** Use `lean_hover_info` and `lean_declaration_file` to navigate definitions
- **Testing ideas:** Use `lean_run_code` or `lean_multi_attempt` to experiment before committing changes
- **After changes:** Use `lean_build` to rebuild and restart LSP for fresh diagnostics

**Memory management:** Monitor Lean server processes (`pgrep -af lean`) and restart when necessary to avoid resource exhaustion.

### During Active Development
- Incomplete proofs (`sorry`) are acceptable with TODO comments explaining what needs verification
- Focus on building working implementations before perfecting proofs
- Iterate on design before committing to formal verification
- Document verification scope clearly in docstrings
- Flag areas where verification is aspirational vs. complete
- Use MCP search tools (`lean_local_search`, `lean_leansearch`) to discover existing patterns before implementing from scratch

### When in Doubt
- **First:** Use `lean_leansearch` or `lean_loogle` to search for relevant theorems/definitions
- **Second:** Use `lean_local_search` to find similar code in this codebase
- **Third:** Check internal documentation guides:
  - [COOKBOOK.md](COOKBOOK.md) - Practical recipes and examples
  - [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing strategies and patterns
  - [VERIFICATION_WORKFLOW.md](VERIFICATION_WORKFLOW.md) - Proof development steps
  - [ARCHITECTURE.md](ARCHITECTURE.md) - Module structure and dependencies
  - Directory READMEs - All 10 VerifiedNN/ subdirectories have comprehensive documentation
- Consult SciLean examples and documentation
- Check mathlib for existing analysis lemmas via `lean_hover_info` and `lean_declaration_file`
- Reference [verified-nn-spec.md](verified-nn-spec.md) for detailed implementation guidance
- Review [ARCHITECTURE.md](ARCHITECTURE.md) for codebase organization and module structure
- Ask on Lean Zulip #scientific-computing (Tomáš Skřivan, SciLean author, is responsive)

---

**Last Updated:** October 22, 2025
**Maintained by:** Project contributors

**Recent Updates:**
- **2025-10-22:** Added references to new documentation guides (GETTING_STARTED, ARCHITECTURE, TESTING_GUIDE, COOKBOOK, VERIFICATION_WORKFLOW, DOCUMENTATION_INDEX)
- **2025-10-21:** Repository cleanup to mathlib submission quality
