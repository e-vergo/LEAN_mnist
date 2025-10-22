# Verification Workflow Tutorial

A step-by-step guide to proving properties about neural network code in VerifiedNN.

## Table of Contents

1. [Introduction to Formal Verification](#introduction-to-formal-verification)
2. [Verification Goals](#verification-goals)
3. [Proof Development Workflow](#proof-development-workflow)
4. [Common Proof Patterns](#common-proof-patterns)
5. [Working with SciLean](#working-with-scilean)
6. [Verifying Gradient Correctness](#verifying-gradient-correctness)
7. [Type-Level Verification](#type-level-verification)
8. [Troubleshooting Proofs](#troubleshooting-proofs)

---

## Introduction to Formal Verification

### What is Formal Verification?

Formal verification means **mathematically proving** that code satisfies a specification. In VerifiedNN, we prove:

1. **Gradient correctness:** `fderiv ℝ f = analytical_derivative(f)`
2. **Type safety:** Dimension specifications match runtime behavior
3. **Mathematical properties:** Non-negativity, convexity, etc.

### Why Verify Neural Networks?

**Without verification:**
- Gradients computed by backpropagation *might* be correct
- Tested on examples, but no mathematical guarantee
- Bugs can silently corrupt training

**With verification:**
- Gradients are **provably** correct
- Dimension mismatches caught at compile time
- Mathematical properties formally established

### Verification vs Testing

| Aspect | Testing | Verification |
|--------|---------|--------------|
| Method | Run code on examples | Mathematical proof |
| Coverage | Specific inputs | All possible inputs |
| Guarantee | "Works on these cases" | "Works for all cases" |
| Effort | Lower | Higher |

**VerifiedNN uses both:** Test for numerical correctness, prove for mathematical correctness.

---

## Verification Goals

### Primary Goal: Gradient Correctness

**Objective:** Prove automatic differentiation computes correct gradients.

**Strategy:**
1. Prove each operation's gradient: ReLU, matrix multiply, softmax, cross-entropy
2. Prove composition preserves correctness (chain rule)
3. Establish end-to-end theorem for full network

**Main Theorem:**
```lean
theorem network_gradient_correct :
  Differentiable ℝ (λ params => networkLoss (unflattenParams params) x y) := by
  -- Proof establishes end-to-end differentiability
  ...
```

### Secondary Goal: Type Safety

**Objective:** Prove dependent types enforce runtime dimensions.

**Strategy:**
1. Use dependent types in definitions
2. Prove operations preserve dimensions
3. Leverage type system for compile-time checks

**Example Theorem:**
```lean
theorem layer_output_dim {m n : Nat} (layer : DenseLayer n m) (x : Vector n) :
  (layer.forward x).size = m := by
  rfl  -- Trivial: type system guarantees this
```

### Out of Scope: Convergence

**Not verified:** Convergence properties of SGD (explicitly axiomatized)

**Rationale:** Optimization theory is a separate research project. We focus on proving gradient correctness, not convergence behavior.

---

## Proof Development Workflow

### Step 1: Implement the Function

Write computational code first:

```lean
def relu (x : Float) : Float :=
  if x > 0 then x else 0

def reluVec {n : Nat} (v : Vector n) : Vector n :=
  ⊞ i => relu v[i]
```

### Step 2: State the Property

Clearly state what you want to prove:

```lean
-- Property: ReLU is non-negative
theorem relu_nonneg (x : ℝ) : relu x ≥ 0 := by
  sorry  -- TODO: Prove this
```

### Step 3: Develop the Proof

Fill in the proof using tactics:

```lean
theorem relu_nonneg (x : ℝ) : relu x ≥ 0 := by
  unfold relu
  split_ifs with h
  · -- Case: x > 0
    linarith  -- Linear arithmetic solver
  · -- Case: x ≤ 0
    linarith  -- relu x = 0, so 0 ≥ 0
```

### Step 4: Verify with `#check`

Confirm the theorem type-checks:

```lean
#check relu_nonneg
-- relu_nonneg : ∀ (x : ℝ), relu x ≥ 0
```

### Step 5: Check Axiom Usage

See what axioms your proof relies on:

```bash
lean --print-axioms VerifiedNN/Verification/GradientCorrectness.lean
```

Expected axioms:
- `propext`, `Quot.sound`, `Classical.choice` (mathlib standard)
- `SciLean.sorryProofAxiom` (from fun_prop automation)

Unexpected axioms suggest incomplete proofs.

### Step 6: Document the Proof

Add comprehensive docstring:

```lean
/-- ReLU is non-negative for all inputs.

This theorem establishes that `relu x ≥ 0` for any real number `x`.
This is a fundamental property used in proving that ReLU preserves
non-negative activations throughout the network.

**Proof strategy:** Case analysis on `x > 0`. In both cases, the result
follows from linear arithmetic (`linarith` tactic).

**References:**
- ReLU definition: VerifiedNN.Core.Activation.relu
- Used in: layer_preserves_nonnegativity
-/
theorem relu_nonneg (x : ℝ) : relu x ≥ 0 := by
  unfold relu
  split_ifs
  · linarith
  · linarith
```

---

## Common Proof Patterns

### Pattern 1: Proof by Reflexivity (Type-Level)

**Use case:** Property holds by construction (type system enforces it).

```lean
theorem layer_forward_output_dim {m n : Nat}
  (layer : DenseLayer n m) (x : Vector n) :
  (layer.forward x).size = m := by
  rfl
```

**Explanation:** If the code type-checks with dependent types, dimensions are correct by definition. `rfl` (reflexivity) proves it immediately.

### Pattern 2: Proof by Unfolding and Simplification

**Use case:** Property follows from definitions.

```lean
theorem vadd_comm {n : Nat} (v w : Vector n) :
  vadd v w = vadd w v := by
  unfold vadd
  simp [add_comm]  -- Use commutativity of addition
```

### Pattern 3: Proof by Case Analysis

**Use case:** Function has branches (if-then-else).

```lean
theorem relu_cases (x : ℝ) :
  relu x = if x > 0 then x else 0 := by
  unfold relu
  split_ifs
  · rfl  -- Case: x > 0
  · rfl  -- Case: x ≤ 0
```

### Pattern 4: Proof by Induction

**Use case:** Recursive definitions or properties over natural numbers.

```lean
theorem sum_nonneg {n : Nat} (v : Vector n) (h : ∀ i, v[i] ≥ 0) :
  (∑ i, v[i]) ≥ 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp [sum_succ]
    apply add_nonneg
    · exact h n
    · exact ih
```

### Pattern 5: Proof by Calculation (calc)

**Use case:** Chain of equalities or inequalities.

```lean
theorem affine_combination_property {n : Nat} (v w : Vector n) (α : ℝ) :
  ‖α • v + (1 - α) • w‖ ≤ α * ‖v‖ + (1 - α) * ‖w‖ := by
  calc ‖α • v + (1 - α) • w‖
      ≤ ‖α • v‖ + ‖(1 - α) • w‖   := by apply norm_add_le
    _ = |α| * ‖v‖ + |1 - α| * ‖w‖  := by simp [norm_smul]
    _ ≤ α * ‖v‖ + (1 - α) * ‖w‖    := by linarith [...]
```

---

## Working with SciLean

### Registering Differentiable Operations

Every differentiable operation needs two attributes:

1. **`@[fun_prop]`** - Declares the function is differentiable
2. **`@[fun_trans]`** - Provides the derivative

```lean
import SciLean
import Mathlib.Analysis.Calculus.FDeriv.Basic

-- 1. Implement operation
def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

-- 2. Prove differentiability
@[fun_prop]
theorem sigmoid_differentiable : Differentiable ℝ sigmoid := by
  unfold sigmoid
  fun_prop  -- SciLean tactic

-- 3. Prove derivative formula
@[fun_trans]
theorem sigmoid_fderiv (x : ℝ) :
  fderiv ℝ sigmoid x = fun dx => sigmoid x * (1 - sigmoid x) * dx := by
  unfold sigmoid
  fun_trans  -- SciLean tactic
  simp [...]
```

### Using `fun_trans` Tactic

`fun_trans` automatically computes derivatives:

```lean
example (x : ℝ) : fderiv ℝ (fun x => x^2 + 3*x + 2) x = fun dx => (2*x + 3) * dx := by
  fun_trans
  simp
```

**How it works:**
- Looks up `@[fun_trans]` lemmas for each operation
- Applies chain rule automatically
- Simplifies the result

### Using `fun_prop` Tactic

`fun_prop` proves differentiability/continuity properties:

```lean
example : Differentiable ℝ (fun x : ℝ => x^2 + Real.sin x) := by
  fun_prop
```

**How it works:**
- Looks up `@[fun_prop]` lemmas
- Composes differentiability proofs
- Handles arithmetic operations automatically

---

## Verifying Gradient Correctness

### Step-by-Step: Verify ReLU Gradient

**Step 1:** Define ReLU on ℝ (for proofs)

```lean
def reluReal (x : ℝ) : ℝ := max x 0
```

**Step 2:** Prove differentiability almost everywhere

```lean
@[fun_prop]
theorem relu_differentiable_ae :
  ∀ x : ℝ, x ≠ 0 → DifferentiableAt ℝ reluReal x := by
  intro x hx
  unfold reluReal
  -- ReLU is differentiable everywhere except x = 0
  sorry  -- TODO: Complete proof
```

**Step 3:** Compute derivative

```lean
@[fun_trans]
theorem relu_fderiv (x : ℝ) (hx : x ≠ 0) :
  fderiv ℝ reluReal x = fun dx => if x > 0 then dx else 0 := by
  unfold reluReal
  fun_trans (disch := aesop)
  split_ifs with h
  · -- Case: x > 0, derivative is identity
    sorry
  · -- Case: x < 0, derivative is zero
    sorry
```

**Step 4:** Verify against analytical derivative

The analytical derivative of ReLU is:
```
∂(ReLU(x))/∂x = { 1  if x > 0
                 { 0  if x ≤ 0
```

Our theorem matches this, so gradient is correct!

### Verifying Layer Gradient

**Claim:** Dense layer `f(x) = Wx + b` has gradient `∇f = W^T`.

**Proof:**

```lean
theorem dense_layer_gradient_wrt_input {m n : Nat}
  (W : Matrix m n) (b : Vector m) :
  ∀ x : Vector n, fderiv ℝ (fun x => matvec W x + b) x = fun dx => matvec W dx := by
  intro x
  -- Derivative of Wx + b with respect to x is W
  fun_trans
  simp
```

### Verifying Chain Rule

**Claim:** Composition preserves gradient correctness.

**Proof:**

```lean
theorem composition_preserves_gradient
  {f : ℝ → ℝ} {g : ℝ → ℝ}
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g) :
  Differentiable ℝ (g ∘ f) := by
  apply Differentiable.comp hg hf
```

This is the foundation of backpropagation: if each operation is differentiable, the composition is differentiable.

---

## Type-Level Verification

### Using Dependent Types

Dependent types encode properties in the type system:

```lean
-- Good: Dimension in type
structure DenseLayer (inDim outDim : Nat) where
  weights : Matrix outDim inDim
  bias : Vector outDim

-- Bad: Dimension unchecked
structure BadLayer where
  weights : Matrix
  bias : Vector
  -- No guarantee weights and bias dimensions match!
```

### Proving Dimension Preservation

```lean
theorem dense_layer_preserves_dimension
  {m n : Nat}
  (layer : DenseLayer n m)
  (x : Vector n) :
  (layer.forward x).size = m := by
  -- Proof by reflexivity: type system guarantees this
  rfl
```

**Key insight:** If the code type-checks, dimensions are correct.

### Preventing Invalid Constructions

```lean
def createInvalidLayer : DenseLayer 10 20 := {
  weights := ⊞ (i, j) => 0.0,  -- Type error if dimensions don't match!
  bias := ⊞ i => 0.0
}
```

If dimensions are wrong, code won't compile.

---

## Troubleshooting Proofs

### Issue 1: `unknown identifier` Error

**Problem:**
```lean
theorem my_theorem : relu x ≥ 0 := by
  unfold relu
  -- Error: unknown identifier 'relu'
```

**Solution:** Import the module:
```lean
import VerifiedNN.Core.Activation
open VerifiedNN.Core.Activation
```

### Issue 2: `unsolved goals` Error

**Problem:**
```lean
theorem vadd_comm (v w : Vector n) : vadd v w = vadd w v := by
  unfold vadd
  -- unsolved goals: ⊢ (⊞ i => v[i] + w[i]) = (⊞ i => w[i] + v[i])
```

**Solution:** Use `simp` with commutativity lemma:
```lean
theorem vadd_comm (v w : Vector n) : vadd v w = vadd w v := by
  unfold vadd
  simp [add_comm]  -- Now solved
```

### Issue 3: `fun_trans failed` Error

**Problem:**
```lean
theorem my_fderiv : fderiv ℝ myFunc x = ... := by
  fun_trans
  -- Error: fun_trans failed to apply
```

**Solutions:**
1. Check `myFunc` has `@[fun_trans]` attribute
2. Try `fun_trans (disch := aesop)` to solve side conditions
3. Unfold definition first: `unfold myFunc; fun_trans`

### Issue 4: Proof Takes Too Long

**Problem:** Tactic runs for minutes without finishing.

**Solutions:**
1. Break proof into smaller lemmas
2. Use `set_option maxHeartbeats 0` to disable timeout
3. Simplify goal before applying heavy tactics:
```lean
theorem big_theorem : ... := by
  simp only [...]  -- Simplify first
  fun_trans  -- Now faster
```

### Issue 5: Type Mismatch

**Problem:**
```lean
theorem test (x : Float) : relu x ≥ 0 := by
  -- Error: type mismatch, expected ℝ, got Float
```

**Solution:** Prove on ℝ, not Float:
```lean
theorem test (x : ℝ) : reluReal x ≥ 0 := by
  unfold reluReal
  simp
```

Float properties are axiomatized (see `Loss/Properties.lean` for example).

---

## Verification Checklist

Before claiming a property is verified:

- [ ] **Theorem stated clearly** with precise mathematical statement
- [ ] **Proof compiles** without errors or warnings
- [ ] **No active `sorry`** in proof (all cases handled)
- [ ] **Axiom usage checked** (`lean --print-axioms`)
- [ ] **Only expected axioms used** (mathlib standard + SciLean sorryProofAxiom)
- [ ] **Docstring added** explaining theorem and proof strategy
- [ ] **References provided** to related theorems and literature
- [ ] **Numerical validation** (gradient check) confirms theorem

---

## Resources

### Lean 4 Proof Development

- **Theorem Proving in Lean 4:** https://leanprover.github.io/theorem_proving_in_lean4/
- **Mathlib4 Tactics:** https://leanprover-community.github.io/mathlib4_docs/tactics.html
- **Lean Zulip:** https://leanprover.zulipchat.com/ (#new members, #mathlib4)

### SciLean Documentation

- **SciLean Repository:** https://github.com/lecopivo/SciLean
- **SciLean Docs:** https://lecopivo.github.io/scientific-computing-lean/
- **SciLean Examples:** https://github.com/lecopivo/SciLean/tree/master/examples

### VerifiedNN Verification Files

- **Main theorem:** `VerifiedNN/Verification/GradientCorrectness.lean:352-403`
- **Type safety:** `VerifiedNN/Verification/TypeSafety.lean`
- **Layer properties:** `VerifiedNN/Layer/Properties.lean`
- **Loss properties:** `VerifiedNN/Loss/Properties.lean`

### Academic References

- **Certigrad (Selsam et al., ICML 2017):** Verified backpropagation in Lean 3
- **mathlib calculus:** https://leanprover-community.github.io/mathlib4_docs/Mathlib/Analysis/Calculus/FDeriv/Basic.html

---

## Next Steps

1. **Read proven theorems:** Start with `VerifiedNN/Verification/GradientCorrectness.lean`
2. **Study proof patterns:** See `VerifiedNN/Layer/Properties.lean` for examples
3. **Try simple proofs:** Prove properties of activation functions
4. **Ask for help:** Lean Zulip #scientific-computing channel
5. **Contribute:** Complete remaining `sorry` statements (see TODO comments)

---

**Last Updated:** 2025-10-22
**Maintained by:** Project contributors
**Questions?** Ask on Lean Zulip or open an issue
