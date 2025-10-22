# Quick Start: Registering AD Attributes

## TL;DR: The Two Attributes You Need

### @[fun_prop] - "This function is differentiable"
Register that a function is differentiable or continuous:
```lean
@[fun_prop]
theorem myFunc.Differentiable : Differentiable R myFunc := by fun_prop
```

### @[data_synth] - "Here's how to compute the derivative"
Provide explicit derivative implementations:
```lean
@[data_synth]
theorem myFunc.HasRevFDeriv :
  HasRevFDeriv R myFunc (fun x => (result, pullback_function)) := by data_synth
```

---

## Three Operation Types

### Type 1: Linear (Simplest)
- Examples: vadd, smul, transpose, matAdd
- What to do: Just add `@[fun_prop]` and let composition handle it
```lean
@[fun_prop]
theorem vadd.Differentiable {n : Nat} :
  Differentiable Float (fun xy => xy.1 + xy.2) := by fun_prop
```

### Type 2: Bilinear (Medium)
- Examples: vmul, dot, matvec, matmul
- What to do: Add `@[fun_prop]` + separate `@[data_synth]` for each argument
```lean
@[fun_prop]
theorem vmul.arg_x.Differentiable {n : Nat} (y : Vector n) :
  Differentiable Float (fun x => vmul x y) := by fun_prop

@[data_synth]
theorem vmul.arg_x.HasRevFDeriv {n : Nat} (y : Vector n) :
  HasRevFDeriv Float (fun x => vmul x y)
    (fun x => (vmul x y, fun dz => vmul dz y)) := by data_synth
```

### Type 3: Nonlinear (Complex)
- Examples: relu, sigmoid, softmax, tanh
- What to do: Add `@[fun_prop]` + full `@[data_synth]` implementation
```lean
@[fun_prop]
theorem relu.Differentiable :
  Differentiable Float relu := by fun_prop

@[data_synth]
theorem relu.HasRevFDeriv :
  HasRevFDeriv Float relu
    (fun x => (relu x, fun dy => if x > 0 then dy else 0)) := by data_synth
```

---

## Naming Conventions

Use this pattern:
```
functionName.arg_ARGUMENT.PropertyName_rule
```

- `functionName`: Name of the function being differentiated
- `arg_ARGUMENT`: Which argument (if applicable), e.g., `arg_x`, `arg_A`, `arg_xy`
  - Omit if single argument
  - Use `arg_x` and `arg_y` for two arguments
  - Use `arg_xy` for tuple of both arguments
- `PropertyName`: What property, e.g., `Differentiable`, `HasFwdFDeriv`, `HasRevFDeriv`

**Examples:**
```lean
theorem vadd.Differentiable  -- single arg
theorem vmul.arg_x.HasRevFDeriv  -- first arg of two
theorem vmul.arg_y.HasRevFDeriv  -- second arg of two
theorem matvec.arg_Ax.HasFwdFDeriv  -- both args as tuple
```

---

## The Three Derivative Types

### Forward Mode: HasFwdFDeriv
"Given a tangent vector dx, compute the output and its derivative"

```lean
@[data_synth]
theorem operation.HasFwdFDeriv :
  HasFwdFDeriv R operation
    (fun x dx => (operation x, tangent_output)) := by data_synth

-- Example: relu
HasFwdFDeriv Float relu
  (fun x dx => (relu x, if x > 0 then dx else 0))

-- Example: vmul
HasFwdFDeriv Float (fun x => vmul x y)
  (fun x dx => (vmul x y, vmul dx y))
```

### Reverse Mode: HasRevFDeriv
"Given the output, return a function that maps cotangent to input cotangent"

```lean
@[data_synth]
theorem operation.HasRevFDeriv :
  HasRevFDeriv R operation
    (fun x => (operation x, fun dy => pullback_result)) := by data_synth

-- Example: relu
HasRevFDeriv Float relu
  (fun x => (relu x, fun dy => if x > 0 then dy else 0))

-- Example: vmul
HasRevFDeriv Float (fun x => vmul x y)
  (fun x => (vmul x y, fun dz => vmul dz y))
```

### Reverse Mode with Update: HasRevFDerivUpdate
"Efficiently accumulate gradients in place"

```lean
@[data_synth]
theorem operation.HasRevFDerivUpdate :
  HasRevFDerivUpdate R operation
    (fun x => (operation x, fun dy x' => updated_x)) := by data_synth

-- Example: vmul with accumulation
HasRevFDerivUpdate Float (fun x => vmul x y)
  (fun x => (vmul x y, fun dy x' => x' + vmul dy y))
```

---

## Quick Reference: What to Register for Each Operation

### LinearAlgebra.lean

```lean
-- Vector operations
@[fun_prop] theorem vadd.Differentiable
@[fun_prop] theorem vsub.Differentiable
@[fun_prop] theorem smul.Differentiable

-- Elementwise
@[fun_prop] theorem vmul.arg_x.Differentiable
@[data_synth] theorem vmul.arg_x.HasRevFDeriv
@[data_synth] theorem vmul.arg_y.HasRevFDeriv

-- Reductions
@[fun_prop] theorem dot.arg_x.Differentiable
@[data_synth] theorem dot.arg_x.HasRevFDeriv
@[data_synth] theorem dot.arg_y.HasRevFDeriv

-- Norms
@[fun_prop] theorem normSq.Differentiable
@[data_synth] theorem normSq.HasRevFDeriv
@[fun_prop] theorem norm.Differentiable
@[data_synth] theorem norm.HasRevFDeriv

-- Matrix operations
@[fun_prop] theorem matvec.arg_A.Differentiable  -- Check if SciLean has this
@[fun_prop] theorem matmul.arg_A.Differentiable
@[data_synth] theorem matmul.arg_A.HasRevFDeriv
@[data_synth] theorem matmul.arg_B.HasRevFDeriv
@[fun_prop] theorem transpose.Differentiable

-- Matrix arithmetic
@[fun_prop] theorem matAdd.Differentiable
@[fun_prop] theorem matSub.Differentiable
@[fun_prop] theorem matSmul.Differentiable
@[fun_prop] theorem outer.arg_x.Differentiable
@[data_synth] theorem outer.arg_x.HasRevFDeriv
@[data_synth] theorem outer.arg_y.HasRevFDeriv

-- Batch operations (may inherit from simpler ops)
@[fun_prop] theorem batchMatvec.Differentiable
@[fun_prop] theorem batchAddVec.Differentiable
```

### Activation.lean

```lean
-- ReLU family
@[fun_prop] theorem relu.Differentiable
@[data_synth] theorem relu.HasRevFDeriv
@[fun_prop] theorem reluVec.Differentiable
@[fun_prop] theorem reluBatch.Differentiable
@[fun_prop] theorem leakyRelu.Differentiable
@[data_synth] theorem leakyRelu.HasRevFDeriv
@[fun_prop] theorem leakyReluVec.Differentiable

-- Softmax (check if SciLean provides this)
@[fun_prop] theorem softmax.Differentiable
@[data_synth] theorem softmax.HasRevFDeriv

-- Sigmoid family
@[fun_prop] theorem sigmoid.Differentiable
@[data_synth] theorem sigmoid.HasRevFDeriv
@[fun_prop] theorem sigmoidVec.Differentiable
@[fun_prop] theorem sigmoidBatch.Differentiable

-- Tanh family
@[fun_prop] theorem tanh.Differentiable
@[data_synth] theorem tanh.HasRevFDeriv
@[fun_prop] theorem tanhVec.Differentiable
```

---

## Common Proof Strategies

### For @[fun_prop] - Usually just:
```lean
@[fun_prop] theorem operation.Differentiable : Differentiable R operation := by
  unfold operation
  fun_prop
```

### For @[data_synth] - Usually:
```lean
@[data_synth] theorem operation.HasRevFDeriv : HasRevFDeriv R operation f' := by
  unfold operation
  data_synth
```

### If that doesn't work - Decompose:
```lean
@[data_synth] theorem operation.HasRevFDeriv : HasRevFDeriv R operation f' := by
  unfold operation
  apply hasRevFDeriv_from_hasFDerivAt_hasAdjoint
  case deriv => intros; data_synth
  case adjoint => intros; simp; data_synth
  case simp => intros; simp
```

---

## Testing Your Registrations

After implementing:

```bash
# Build and check for errors
lake build VerifiedNN.Core.LinearAlgebra
lake build VerifiedNN.Core.Activation

# Check that derivative computation works
lake env lean --run test_ad.lean  # if you create a test file

# Run gradient checking tests
lake build VerifiedNN.Testing.GradientCheck
```

---

## Key Files to Reference

- **SciLean MatVecMul rules:** `.lake/packages/SciLean/SciLean/AD/Rules/MatVecMul.lean`
- **SciLean VecMatMul rules:** `.lake/packages/SciLean/SciLean/AD/Rules/VecMatMul.lean`
- **SciLean Exp rules:** `.lake/packages/SciLean/SciLean/AD/Rules/Exp.lean` (simpler example)
- **Your project's current LinearAlgebra.lean:** `/Users/eric/LEAN_mnist/VerifiedNN/Core/LinearAlgebra.lean`
- **Your project's current Activation.lean:** `/Users/eric/LEAN_mnist/VerifiedNN/Core/Activation.lean`

---

## When to Skip Registration

Some operations **don't need explicit registration** because SciLean handles them automatically:

1. **Operations that are compositions** (if components are registered)
   - `reluBatch = element_wise relu` → Automatically differentiable if relu is
   - `batchMatvec = map matvec` → Automatically differentiable if matvec is

2. **Operations that are simple algebra**
   - `matAdd = element_wise (+)` → Inherit from vector addition
   - `matSmul = element_wise (•)` → Inherit from scalar multiplication

3. **Operations already in SciLean**
   - Check first: `grep -r "operation_name" .lake/packages/SciLean/SciLean/AD/`
   - If found, may already be registered

---

## Most Important Rule

**When in doubt, check the SciLean patterns:**

The MatVecMul.lean and VecMatMul.lean files are your templates. If your operation is similar, follow the same structure.

