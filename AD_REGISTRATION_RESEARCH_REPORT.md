# Automatic Differentiation Registration Research Report

## Executive Summary

This report documents the correct syntax and patterns for registering operations with SciLean's automatic differentiation system using `@[fun_trans]` and `@[fun_prop]` attributes. The research reveals two main registration approaches in SciLean:

1. **High-level DSL approach** (using `def_fun_prop` and `abbrev_fun_trans` macros) - suitable for scalar operations
2. **Low-level theorem approach** (using `@[fun_prop]` and `@[data_synth]` on explicit theorems) - suitable for custom operations

---

## Part 1: SciLean's AD Registration System

### Core Concepts

SciLean uses a rule-based automatic differentiation system with several key components:

#### 1. **@[fun_prop] - Differentiability/Continuity Properties**
- Marks theorems that prove differentiability, continuity, or linearity properties
- Used by the `fun_prop` tactic to automatically build proofs
- Examples: `Differentiable R f`, `Continuous f`, `IsContinuousLinearMap R f`

#### 2. **@[fun_trans] - Derivative Rules**
- Marks theorems about how to compute derivatives (fderiv, fwdFDeriv, revFDeriv)
- Used by the `fun_trans` tactic to transform function definitions into their derivatives
- Examples: `fderiv K exp = (fun x => fun dx => dx • exp x)`

#### 3. **@[data_synth] - Data-Level Derivative Information**
- Marks theorems with detailed derivative information (HasFwdFDeriv, HasRevFDeriv, HasAdjoint)
- Used by the `data_synth` tactic to synthesize concrete derivative implementations
- Provides forward-mode and reverse-mode AD rules

---

## Part 2: Existing SciLean Patterns

### Pattern 1: Scalar Operations (High-Level DSL)

**File:** `.lake/packages/SciLean/SciLean/AD/Rules/Exp.lean`

```lean
namespace SciLean.Scalar

-- Define the function
def exp (x : K) : K := ...

-- Register fun_prop rules (differentiability/continuity)
def_fun_prop exp in x with_transitive : Differentiable K by sorry_proof
def_fun_prop exp in x with_transitive : ContDiff K ⊤ by sorry_proof

-- Register fun_trans rules (derivative computation)
abbrev_fun_trans exp in x : fderiv K by
  equals (fun x => fun dx =>L[K] dx • exp x) => sorry_proof

abbrev_fun_trans exp in x : fwdFDeriv K by
  unfold fwdFDeriv; fun_trans; to_ssa

abbrev_fun_trans exp in x : revFDeriv K by
  unfold revFDeriv; fun_trans; to_ssa

-- Register data_synth rules (concrete AD implementations)
abbrev_data_synth exp in x (x₀) : (HasFDerivAt (𝕜:=K) · · x₀) by
  apply hasFDerivAt_from_fderiv
  case deriv => conv => rhs; fun_trans
  case diff => dsimp[autoParam]; fun_prop

abbrev_data_synth exp in x : HasFwdFDeriv K by
  hasFwdFDeriv_from_def => simp; to_ssa

abbrev_data_synth exp in x : HasRevFDeriv K by
  hasRevFDeriv_from_def => simp; to_ssa

abbrev_data_synth exp in x : HasRevFDerivUpdate K by
  hasRevFDerivUpdate_from_def => simp; to_ssa

end SciLean.Scalar
```

**Key Features:**
- Uses `def_fun_prop` and `abbrev_fun_trans` macros (higher-level DSL)
- The `in x` syntax specifies the differentiation variable
- The `with_transitive` keyword allows transitive rule application
- `abbrev_data_synth` generates concrete AD implementations

### Pattern 2: Matrix Operations (Low-Level Theorem Approach)

**File:** `.lake/packages/SciLean/SciLean/AD/Rules/MatVecMul.lean`

```lean
variable
  {M : Type*} [MatrixMulNotation M]
  {R : Type*} {_ : RCLike R}
  {X : Type*} {_ : NormedAddCommGroup X} {_ : AdjointSpace R X}
  {Y : Type*} {_ : NormedAddCommGroup Y} {_ : AdjointSpace R Y}
  [NormedAddCommGroup M] [AdjointSpace R M] [MatrixType R Y X M]

set_default_scalar R

-- Differentiability properties using @[fun_prop]
@[fun_prop]
theorem matVecMul.arg_A.IsContinuousLinearMap_rule (A : M) :
  IsContinuousLinearMap R (fun x : X => A * x) := sorry_proof

@[fun_prop]
theorem matVecMul.arg_x.IsContinuousLinearMap_rule (x : X) :
  IsContinuousLinearMap R (fun A : M => A * x) := sorry_proof

@[fun_prop]
theorem matVecMul.arg_Ax.ContDiff_rule :
  ContDiff R ⊤ (fun Ax : M×X => Ax.1 * Ax.2) := sorry_proof

@[fun_prop]
theorem matVecMul.arg_Ax.Differentiable_rule :
  Differentiable R (fun Ax : M×X => Ax.1 * Ax.2) := by fun_prop

-- Forward-mode AD using @[data_synth]
@[data_synth]
theorem matVecMul.arg_Ax.HasFwdFDeriv_rule :
  HasFwdFDeriv R
    (fun Ax : M×X => Ax.1 * Ax.2)
    (fun Ax dAx =>
      let' (A,x) := Ax
      let' (dA,dx) := dAx
      (A*x, matVecMulAdd (1:R) dA x (1:R) (A*dx))) := sorry_proof

@[data_synth]
theorem matVecMul.arg_A.HasFwdFDeriv_rule (x : X) :
  HasFwdFDeriv R
    (fun A : M => A * x)
    (fun A dA =>
      (A*x, dA*x)) := sorry_proof

-- Reverse-mode AD using @[data_synth]
@[data_synth]
theorem matVecMul.arg_Ax.HasRevFDeriv_rule :
  HasRevFDeriv R
    (fun Ax : M×X => Ax.1 * Ax.2)
    (fun Ax =>
      let' (A,x) := Ax
      (A*x, fun dy =>
        (dy⊗x, dy*A))) := sorry_proof

@[data_synth]
theorem matVecMul.arg_x.HasRevFDeriv_rule (A : M) :
  HasRevFDeriv R
    (fun x : X => A * x)
    (fun x => (A*x, fun dy => dy*A)) := sorry_proof

-- Reverse-mode update for efficient gradient accumulation
@[data_synth]
theorem matVecMul.arg_x.HasRevFDerivUpdate_rule (A : M) :
  HasRevFDerivUpdate R
    (fun x : X => A * x)
    (fun x => (A*x, fun dy x' => vecMatMulAdd (1:R) dy A (1:R) x')) := sorry_proof
```

**Key Features:**
- Uses explicit `@[fun_prop]` and `@[data_synth]` attributes on theorem definitions
- Naming pattern: `functionName.arg_X.PropertyName_rule` where:
  - `functionName` is the operation being differentiated (matVecMul, vecMatMul)
  - `arg_X` specifies which argument(s) are being differentiated
  - `PropertyName` is the property being proven (IsContinuousLinearMap, HasFwdFDeriv, etc.)
- Supports separate rules for different argument combinations (arg_A, arg_x, arg_Ax)
- Provides both forward-mode (HasFwdFDeriv) and reverse-mode (HasRevFDeriv) rules
- HasRevFDerivUpdate for efficient in-place gradient accumulation

---

## Part 3: Current Project Status

### Files Needing Registration

#### A. **VerifiedNN/Core/LinearAlgebra.lean**

Operations that need `@[fun_trans]` and `@[fun_prop]` registration:

1. **Vector Operations:**
   - `vadd` (vector addition) - Linear
   - `vsub` (vector subtraction) - Linear
   - `smul` (scalar-vector multiplication) - Linear
   - `vmul` (element-wise multiplication) - Bilinear
   - `dot` (dot product) - Bilinear
   - `normSq` (squared L2 norm) - Nonlinear (quadratic)
   - `norm` (L2 norm) - Nonlinear (composition of sqrt and normSq)

2. **Matrix Operations:**
   - `matvec` (matrix-vector multiply) - Linear in both arguments, bilinear combined
   - `matmul` (matrix-matrix multiply) - Bilinear
   - `transpose` (matrix transpose) - Linear
   - `matAdd` (matrix addition) - Linear
   - `matSub` (matrix subtraction) - Linear
   - `matSmul` (scalar-matrix multiplication) - Linear
   - `outer` (outer product) - Bilinear

3. **Batch Operations:**
   - `batchMatvec` (batch matrix-vector) - Linear in weights, nonlinear in batch
   - `batchAddVec` (batch add bias) - Linear

#### B. **VerifiedNN/Core/Activation.lean**

Operations that need `@[fun_trans]` and `@[fun_prop]` registration:

1. **ReLU Family:**
   - `relu` - Piecewise linear (differentiable almost everywhere)
   - `reluVec` - Element-wise ReLU
   - `reluBatch` - Batch ReLU
   - `leakyRelu` - Piecewise linear with different slope for negatives
   - `leakyReluVec` - Element-wise leaky ReLU

2. **Classification:**
   - `softmax` - Nonlinear (uses exp and sum)

3. **Sigmoid Family:**
   - `sigmoid` - Nonlinear (uses exp composition)
   - `sigmoidVec` - Element-wise sigmoid
   - `sigmoidBatch` - Batch sigmoid
   - `tanh` - Nonlinear (composition of sinh/cosh)
   - `tanhVec` - Element-wise tanh

---

## Part 4: Registration Patterns by Operation Type

### Pattern A: Linear Operations (e.g., vadd, matvec, transpose)

Linear operations are the easiest to register. They satisfy:
- `f(αx + βy) = αf(x) + βf(y)` for all scalars α, β and inputs x, y
- Derivative is the function itself (or simpler)
- Adjoint is the transpose of the Jacobian

**Recommended Approach:**

```lean
-- fun_prop: Prove it's a linear/continuous linear map
@[fun_prop]
theorem vadd.IsContinuousLinearMap :
  IsContinuousLinearMap R (fun (xy : Vector n × Vector n) => xy.1 + xy.2) := by
  fun_prop

-- Alternative: If it's differentiable but not necessarily continuous linear
@[fun_prop]
theorem vadd.Differentiable :
  Differentiable R (fun (xy : Vector n × Vector n) => xy.1 + xy.2) := by
  fun_prop

-- data_synth: Forward mode
@[data_synth]
theorem vadd.HasFwdFDeriv :
  HasFwdFDeriv R
    (fun (xy : Vector n × Vector n) => xy.1 + xy.2)
    (fun xy dxy =>
      let (x, y) := xy
      let (dx, dy) := dxy
      (x + y, dx + dy)) := by
  data_synth

-- data_synth: Reverse mode
@[data_synth]
theorem vadd.HasRevFDeriv :
  HasRevFDeriv R
    (fun (xy : Vector n × Vector n) => xy.1 + xy.2)
    (fun xy =>
      let (x, y) := xy
      (x + y, fun dz => (dz, dz))) := by
  data_synth
```

### Pattern B: Bilinear Operations (e.g., vmul, dot, matmul)

Bilinear operations satisfy:
- `f(αx, y) = αf(x, y)` and `f(x, αy) = αf(x, y)`
- `f(x1 + x2, y) = f(x1, y) + f(x2, y)` and `f(x, y1 + y2) = f(x, y1) + f(x, y2)`
- Derivative has both partial derivatives

**Recommended Approach:**

```lean
-- Separate rules for each argument combination
-- When differentiating w.r.t. x: dy[i] = y[i] * dx[i]
@[fun_prop]
theorem vmul.arg_y.Differentiable {n : Nat} (x : Vector n) :
  Differentiable R (fun y : Vector n => vmul x y) := by
  fun_prop

@[data_synth]
theorem vmul.arg_x.HasFwdFDeriv {n : Nat} (y : Vector n) :
  HasFwdFDeriv R
    (fun x : Vector n => vmul x y)
    (fun x dx =>
      (vmul x y, vmul dx y)) := by
  data_synth

@[data_synth]
theorem vmul.arg_y.HasFwdFDeriv {n : Nat} (x : Vector n) :
  HasFwdFDeriv R
    (fun y : Vector n => vmul x y)
    (fun y dy =>
      (vmul x y, vmul x dy)) := by
  data_synth

-- Reverse mode
@[data_synth]
theorem vmul.arg_x.HasRevFDeriv {n : Nat} (y : Vector n) :
  HasRevFDeriv R
    (fun x : Vector n => vmul x y)
    (fun x =>
      (vmul x y, fun dz => vmul dz y)) := by
  data_synth

@[data_synth]
theorem vmul.arg_y.HasRevFDeriv {n : Nat} (x : Vector n) :
  HasRevFDeriv R
    (fun y : Vector n => vmul x y)
    (fun y =>
      (vmul x y, fun dz => vmul x dz)) := by
  data_synth
```

### Pattern C: Nonlinear Operations (e.g., relu, softmax, sigmoid)

Nonlinear operations require:
- Proving differentiability (may be conditional)
- Providing the derivative/gradient formula
- Handling special cases (e.g., ReLU at x=0)

**Recommended Approach for ReLU:**

```lean
-- Differentiability: ReLU is differentiable almost everywhere
@[fun_prop]
theorem relu.Differentiable :
  Differentiable R relu := by
  unfold relu
  fun_prop  -- fun_prop can handle if-then-else in many cases

-- Forward mode: d/dx relu(x) = if x > 0 then 1 else 0
@[data_synth]
theorem relu.HasFwdFDeriv :
  HasFwdFDeriv R relu
    (fun x dx =>
      (relu x, if x > 0 then dx else 0)) := by
  data_synth

-- Reverse mode
@[data_synth]
theorem relu.HasRevFDeriv :
  HasRevFDeriv R relu
    (fun x =>
      (relu x, fun dy =>
        if x > 0 then dy else 0)) := by
  data_synth
```

**Recommended Approach for Softmax:**

```lean
-- Softmax has special treatment in SciLean - may already be registered
-- Check if SciLean.softmax has existing rules before creating new ones

-- For custom softmax implementation:
@[fun_prop]
theorem softmax.Differentiable {n : Nat} :
  Differentiable R (softmax (n := n)) := by
  unfold softmax
  fun_prop  -- Relies on exp, sum already being differentiable

@[data_synth]
theorem softmax.HasRevFDeriv {n : Nat} :
  HasRevFDeriv R (softmax (n := n))
    (fun x =>
      let s := softmax x
      (s, fun dy =>
        let dz := ∑ i, dy[i] * s[i]
        ⊞ i => dy[i] * s[i] - s[i] * dz)) := by
  data_synth
```

---

## Part 5: Attribute Details

### @[fun_prop] - Differentiability Property Registry

**When to use:**
- Proving `Differentiable R f`, `DifferentiableAt R f x`, `Continuous f`, `IsContinuousLinearMap R f`, `ContDiff R n f`
- Proofs that can be built compositionally using the `fun_prop` tactic

**Naming convention:**
```lean
@[fun_prop]
theorem functionName.arg_VAR.PROPERTY_rule : PROPERTY := ...
```

**Common properties:**
- `Differentiable R f` - function is R-differentiable
- `DifferentiableAt R f x` - differentiable at specific point
- `Continuous f` - function is continuous
- `ContDiff R n f` - n-times continuously differentiable
- `IsContinuousLinearMap R f` - continuous R-linear map

### @[data_synth] - Derivative Implementation Registry

**When to use:**
- Providing concrete implementations of HasFwdFDeriv, HasRevFDeriv, HasFDerivAt
- Rules that need more explicit construction

**Key types:**
```lean
-- Forward-mode automatic differentiation
HasFwdFDeriv R f f' where
  f' : X → X → Y × Y  -- (input, tangent) ↦ (output, derivative)

-- Reverse-mode automatic differentiation
HasRevFDeriv R f f' where
  f' : X → (Y × (Y → X))  -- input ↦ (output, pullback function)

-- Reverse-mode with in-place update (efficient)
HasRevFDerivUpdate R f f' where
  f' : X → (Y × (Y → X → X))  -- input ↦ (output, gradient update function)

-- Derivative at a specific point
HasFDerivAt R f f' x where
  f' : ContinuousLinearMap R X Y  -- the derivative is a linear map
```

### @[fun_trans] - Derivative Transform Registry

**When to use:**
- Rewriting function definitions to their derivative definitions
- Used less commonly in modern SciLean (mostly via macros)

**Naming convention:**
```lean
@[fun_trans]
theorem functionName_fderiv : fderiv K f = ... := ...
```

---

## Part 6: Implementation Recommendations

### Step 1: Categorize Operations

**Linear operations (simplest):**
- vadd, vsub, matAdd, matSub, transpose, matSmul, smul
- batchAddVec (batch + vector broadcast)
- These inherit differentiability from SciLean's module structure

**Bilinear operations (moderate complexity):**
- vmul, dot, matvec, matmul, outer
- Each needs separate forward/reverse rules for each argument
- Jacobian is linear in the other argument

**Nonlinear operations (most complex):**
- relu, leakyRelu, sigmoid, tanh, softmax
- May need special handling (piecewise, composed with special functions)
- Consider numerical stability

**Batch operations (mixed):**
- batchMatvec - linear in weights, vectorized element-wise in batch
- Compose from simpler operations

### Step 2: Leverage SciLean's Existing Rules

Before writing custom rules, check if SciLean already provides:
1. **exp, log** - In `.lake/packages/SciLean/SciLean/AD/Rules/`
2. **softmax** - May be in DataArrayN rules
3. **Matrix operations** - matVecMul, vecMatMul already registered
4. **Basic arithmetic** - May be auto-derived from Ring/Module structures

Search: `grep -r "softmax\|exp\|sigmoid" .lake/packages/SciLean/SciLean/AD/Rules/`

### Step 3: Choose Registration Approach

**For custom linear operations:**
```lean
-- Just prove it's continuous linear or differentiable
@[fun_prop] theorem myLinearOp.Differentiable := by fun_prop
-- Derivative composition will handle the rest
```

**For custom bilinear operations:**
```lean
-- Register forward and reverse rules separately
@[data_synth] theorem myBilinear.arg_x.HasFwdFDeriv := ...
@[data_synth] theorem myBilinear.arg_y.HasFwdFDeriv := ...
@[data_synth] theorem myBilinear.arg_x.HasRevFDeriv := ...
@[data_synth] theorem myBilinear.arg_y.HasRevFDeriv := ...
```

**For custom nonlinear operations:**
```lean
-- Prove differentiability, then provide specific rules
@[fun_prop] theorem myNonlinear.Differentiable := by ...
@[data_synth] theorem myNonlinear.HasFwdFDeriv := ...
@[data_synth] theorem myNonlinear.HasRevFDeriv := ...
```

### Step 4: Proof Strategy

Most proofs should follow this pattern:

1. **For @[fun_prop] theorems:**
   ```lean
   @[fun_prop]
   theorem operation.Differentiable : Differentiable R operation := by
     unfold operation
     fun_prop  -- Lean will try to prove this using composition
   ```

2. **For @[data_synth] theorems:**
   ```lean
   @[data_synth]
   theorem operation.HasFwdFDeriv : HasFwdFDeriv R operation f' := by
     data_synth  -- or
     hasFwdFDeriv_from_def => simp; to_ssa
   ```

3. **If automated tactics fail:**
   ```lean
   @[data_synth]
   theorem operation.HasRevFDeriv : HasRevFDeriv R operation f' := by
     unfold operation
     apply hasRevFDeriv_from_hasFDerivAt_hasAdjoint
     case deriv => intros; data_synth
     case adjoint => intros; simp; data_synth
     case simp => intros; simp
   ```

---

## Part 7: Potential Issues and Considerations

### Issue 1: ReLU at x=0
The gradient of ReLU is technically undefined at x=0. The project convention is to use 0:
```lean
-- This is a subgradient choice, matching PyTorch
def relu (x : Float) : Float := if x > 0 then x else 0
-- Derivative: if x > 0 then 1 else 0 (giving 0 at x=0)
```

### Issue 2: Float.exp Differentiability
Check if SciLean registers `Float.exp` as differentiable. If not, you may need to:
```lean
-- Register Float.exp before sigmoid/softmax
@[fun_prop]
theorem Float.exp.Differentiable : Differentiable R Float.exp := by
  sorry  -- Check SciLean docs
```

### Issue 3: Operations Already Registered in SciLean
Don't duplicate registrations. Check:
```bash
grep -r "matvec\|softmax\|exp" .lake/packages/SciLean/SciLean/AD/Rules/ --include="*.lean"
```

### Issue 4: The Float vs ℝ Gap
All registrations use type variable `K` or `R` (scalar type). When proving differentiability:
- Prove on abstract types that could be ℝ, Float, ℂ, etc.
- Use `set_default_scalar Float` at top of file if working specifically with Float
- Mathlib's `Differentiable` is about ℝ-differentiability; Float equivalents use numerical properties

### Issue 5: Batch Operations
For batch operations like `batchMatvec` or `batchAddVec`:
- These are often compositions: `⊞ (k,i) => matvec ...`
- If matvec is registered, composition rules should auto-generate derivative
- May not need explicit registration

### Issue 6: Optional Arguments and Default Scalars
If a function uses a numerical tolerance (like leaky ReLU's slope):
```lean
def leakyRelu (α : Float) (x : Float) : Float :=
  if x > 0 then x else α * x

-- Register for specific α, or generalize
@[fun_prop]
theorem leakyRelu.arg_x.Differentiable (α : Float) :
  Differentiable R (fun x => leakyRelu α x) := by
  fun_prop
```

---

## Part 8: Summary of Operations and Recommended Approach

### LinearAlgebra.lean Operations

| Operation | Type | Approach | Priority | Notes |
|-----------|------|----------|----------|-------|
| `vadd` | Linear | fun_prop only | High | Simple composition of existing rules |
| `vsub` | Linear | fun_prop only | High | Similar to vadd |
| `smul` | Linear | fun_prop only | High | Scalar multiplication |
| `vmul` | Bilinear | data_synth + fun_prop | High | Element-wise Hadamard product |
| `dot` | Bilinear | data_synth + fun_prop | High | Reduction to scalar |
| `normSq` | Nonlinear | data_synth + fun_prop | Medium | Composition: dot(x, x) |
| `norm` | Nonlinear | data_synth + fun_prop | Medium | Composition: sqrt(normSq) |
| `matvec` | Linear×Vector | data_synth + fun_prop | High | Core operation, already in SciLean |
| `matmul` | Bilinear | data_synth + fun_prop | High | Composition of matvec |
| `transpose` | Linear | fun_prop only | Medium | Adjoint operator |
| `matAdd` | Linear | fun_prop only | Low | Inherited from vector addition |
| `matSub` | Linear | fun_prop only | Low | Inherited from vector subtraction |
| `matSmul` | Linear | fun_prop only | Low | Inherited from scalar multiplication |
| `outer` | Bilinear | data_synth + fun_prop | Medium | Rank-1 matrix construction |
| `batchMatvec` | Linear×Batch | Inherit from matvec | Medium | Vectorized matvec |
| `batchAddVec` | Linear | Inherit from vadd | Low | Broadcasted addition |

### Activation.lean Operations

| Operation | Type | Approach | Priority | Notes |
|-----------|------|----------|----------|-------|
| `relu` | Piecewise Linear | data_synth + fun_prop | High | Differentiable almost everywhere |
| `reluVec` | Vectorized ReLU | Inherit from relu | High | Element-wise |
| `reluBatch` | Batched ReLU | Inherit from relu | High | Element-wise over batch |
| `leakyRelu` | Piecewise Linear | data_synth + fun_prop | Medium | Parameter α |
| `leakyReluVec` | Vectorized Leaky | Inherit from leakyRelu | Medium | Element-wise |
| `softmax` | Nonlinear | Check SciLean first | High | May already be registered |
| `sigmoid` | Nonlinear | data_synth + fun_prop | High | Composition with exp |
| `sigmoidVec` | Vectorized | Inherit from sigmoid | High | Element-wise |
| `sigmoidBatch` | Batched | Inherit from sigmoid | High | Element-wise over batch |
| `tanh` | Nonlinear | data_synth + fun_prop | High | Built-in or via exp |
| `tanhVec` | Vectorized | Inherit from tanh | High | Element-wise |

---

## Part 9: Step-by-Step Implementation Plan

### Phase 1: Foundation (Linear Operations)
1. Register `vadd`, `vsub`, `smul` with `@[fun_prop]`
2. Register `matAdd`, `matSub`, `matSmul` with `@[fun_prop]`
3. Register `transpose` with `@[fun_prop]`
4. Verify existing `matvec` rules in SciLean (may already be done)

**Expected outcome:** All linear ops automatically differentiated

### Phase 2: Bilinear Operations
1. Register `vmul` forward and reverse modes
2. Register `dot` forward and reverse modes
3. Register `matmul` rules (as composition of matvec)
4. Register `outer` rules

**Expected outcome:** Tensor operations fully differentiable

### Phase 3: Nonlinear Activations
1. Register `relu` and `reluVec`/`reluBatch`
2. Register `sigmoid` and variants
3. Register `tanh` and variants
4. Register `leakyRelu` and variants

**Expected outcome:** All activation functions fully differentiable

### Phase 4: Softmax and Special Operations
1. Check if SciLean already provides softmax
2. If not, implement custom softmax derivative
3. Verify `norm` and `normSq` from composition

**Expected outcome:** Complete coverage

---

## Part 10: Example Implementations

### Example 1: Linear Operation - vadd

```lean
-- In Core/LinearAlgebra.lean

@[fun_prop]
theorem vadd.Differentiable {n : Nat} :
  Differentiable Float (fun (xy : Vector n × Vector n) => vadd xy.1 xy.2) := by
  unfold vadd
  fun_prop

-- Optional: explicit forward mode
@[data_synth]
theorem vadd.HasFwdFDeriv {n : Nat} :
  HasFwdFDeriv Float
    (fun (xy : Vector n × Vector n) => vadd xy.1 xy.2)
    (fun xy dxy =>
      let (x, y) := xy
      let (dx, dy) := dxy
      (vadd x y, vadd dx dy)) := by
  data_synth
```

### Example 2: Bilinear Operation - vmul

```lean
-- In Core/LinearAlgebra.lean

@[fun_prop]
theorem vmul.arg_x.Differentiable {n : Nat} (y : Vector n) :
  Differentiable Float (fun x : Vector n => vmul x y) := by
  unfold vmul
  fun_prop

@[data_synth]
theorem vmul.arg_x.HasFwdFDeriv {n : Nat} (y : Vector n) :
  HasFwdFDeriv Float
    (fun x : Vector n => vmul x y)
    (fun x dx =>
      (vmul x y, vmul dx y)) := by
  unfold vmul
  data_synth

@[data_synth]
theorem vmul.arg_x.HasRevFDeriv {n : Nat} (y : Vector n) :
  HasRevFDeriv Float
    (fun x : Vector n => vmul x y)
    (fun x =>
      (vmul x y, fun dz => vmul dz y)) := by
  unfold vmul
  data_synth

-- Similar for arg_y
```

### Example 3: Nonlinear Operation - relu

```lean
-- In Core/Activation.lean

@[fun_prop]
theorem relu.Differentiable :
  Differentiable Float relu := by
  unfold relu
  fun_prop

@[data_synth]
theorem relu.HasFwdFDeriv :
  HasFwdFDeriv Float relu
    (fun x dx =>
      (relu x, if x > 0 then dx else 0)) := by
  unfold relu
  data_synth

@[data_synth]
theorem relu.HasRevFDeriv :
  HasRevFDeriv Float relu
    (fun x =>
      (relu x, fun dy =>
        if x > 0 then dy else 0)) := by
  unfold relu
  data_synth
```

---

## Part 11: Testing and Validation

After implementing registrations, validate with:

1. **Compilation:** Should compile with zero errors
   ```bash
   lake build VerifiedNN.Core.LinearAlgebra
   lake build VerifiedNN.Core.Activation
   ```

2. **Gradient checking:** Run existing gradient check tests
   ```bash
   lake build VerifiedNN.Testing.GradientCheck
   ```

3. **Symbolic verification:** Check that `fun_trans` can compute derivatives
   ```lean
   example : fderiv Float relu 1.0 1.0 = 1.0 := by fun_trans
   example : fderiv Float relu (-1.0) 1.0 = 0.0 := by fun_trans
   ```

4. **End-to-end training:** Verify MNIST training still works
   ```bash
   lake exe mnistTrain --epochs 1 --batch-size 32
   ```

---

## Conclusion

The SciLean automatic differentiation system provides two complementary registration approaches:

1. **High-level macros** (`def_fun_prop`, `abbrev_fun_trans`) for simple scalar operations
2. **Low-level theorems** (`@[fun_prop]`, `@[data_synth]`) for complex custom operations

For the LEAN_mnist project:
- **Linear operations** need only `@[fun_prop]` registration
- **Bilinear operations** need `@[fun_prop]` + `@[data_synth]` for forward/reverse modes
- **Nonlinear operations** need complete `@[data_synth]` implementations

The existing SciLean pattern (MatVecMul.lean, VecMatMul.lean) provides an excellent template for custom operations. By following these patterns, all custom operations can be registered for full automatic differentiation support.

