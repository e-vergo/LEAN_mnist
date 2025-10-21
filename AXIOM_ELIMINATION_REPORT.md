# Axiom Elimination Report for VerifiedNN

**Generated:** 2025-10-20
**Total User Axioms Found:** 44 sorry statements + 11 axiom declarations
**Eliminable Axioms:** 38 sorry statements + 1 axiom declaration
**Out of Scope (Intentional):** 5 convergence axiom declarations

---

## Executive Summary

After comprehensive analysis of all modules, the repository contains:

1. **Standard Lean/Mathlib axioms** (propext, Classical.choice, Quot.sound) - **Acceptable**, cannot eliminate
2. **SciLean library axioms** (sorryProofAxiom, Float↔Real isomorphisms) - **Acceptable per project scope**, library dependencies
3. **User-introduced axioms via sorry** - **Can be eliminated** (38 of 44)
4. **Axiom declarations** - **Mixed** (1 eliminable, 5 intentionally out of scope, 5 SciLean-dependent)

### Elimination Priority Summary

| Priority | Count | Effort | Impact |
|----------|-------|--------|--------|
| **Critical Fixes** | 2 | 5 min | Fix compilation errors |
| **Quick Wins** | 20 | 1-2 hours | Easy proofs, high confidence |
| **Medium Effort** | 18 | 1-2 weeks | Requires proof development |
| **SciLean-Dependent** | 5 | N/A | Coordinate with SciLean maintainers |
| **Out of Scope** | 5 | N/A | Intentional per project design |

---

## Module-by-Module Analysis

### Core Module (0 user axioms)

**Status:** ✅ Clean - No user-introduced axioms

**Files:**
- `Core/DataTypes.lean` - Complete ✅
- `Core/LinearAlgebra.lean` - Complete ✅
- `Core/Activation.lean` - Complete ✅

**Inherited Axioms:**
- Standard Lean: propext, Classical.choice, Quot.sound
- SciLean: sorryProofAxiom, Float↔Real isomorphisms

**Recommendation:** No action needed. Document SciLean axioms as acceptable per verification philosophy.

---

### Layer Module (2 user axioms)

**Files with Axioms:**

#### Layer/Properties.lean (2 sorry statements)

| Line | Theorem | Difficulty | Estimated Time |
|------|---------|------------|----------------|
| 122 | `forwardLinear_is_linear` | Medium | 2-4 hours |
| 165 | `stackLinear_is_linear` | Easy | 30 min (depends on above) |

**Elimination Strategy:**

**Step 1:** Add required lemmas to Core/LinearAlgebra.lean
```lean
-- In VerifiedNN/Core/LinearAlgebra.lean

theorem matvec_linear {m n : Nat} (A : Matrix m n) (x y : Vector n) (α β : Float) :
  matvec A (vadd (smul α x) (smul β y)) =
  vadd (smul α (matvec A x)) (smul β (matvec A y)) := by
  unfold matvec vadd smul
  ext i
  simp [sum_add, sum_smul, mul_add, add_mul]

theorem vadd_comm {n : Nat} (x y : Vector n) :
  vadd x y = vadd y x := by
  unfold vadd; ext i; ring

theorem vadd_assoc {n : Nat} (x y z : Vector n) :
  vadd (vadd x y) z = vadd x (vadd y z) := by
  unfold vadd; ext i; ring

theorem smul_vadd_distrib {n : Nat} (α : Float) (x y : Vector n) :
  smul α (vadd x y) = vadd (smul α x) (smul α y) := by
  unfold smul vadd; ext i; ring
```

**Step 2:** Complete Layer/Properties.lean proofs
```lean
-- Line 122
theorem forwardLinear_is_linear ... := by
  unfold DenseLayer.forwardLinear
  rw [matvec_linear]
  simp [vadd_assoc, smul_vadd_distrib]

-- Line 165
theorem stackLinear_is_linear ... := by
  unfold stackLinear
  rw [forwardLinear_is_linear, forwardLinear_is_linear]
```

**Priority:** High - Aligns with primary verification goals

---

### Loss Module (1 user axiom)

**Files with Axioms:**

#### Loss/Properties.lean (1 sorry statement)

| Line | Theorem | Difficulty | Estimated Time |
|------|---------|------------|----------------|
| 73 | `loss_nonneg` | Medium | 4-6 hours |

**Elimination Strategy:**

**Approach A (Direct):**
```lean
theorem loss_nonneg {n : Nat} (pred : Vector n) (target : Nat) :
  target < n → crossEntropyLoss pred target ≥ 0 := by
  intro h
  unfold crossEntropyLoss logSumExp
  -- Show: log(∑ exp(pred[i])) ≥ pred[target]
  -- Use Jensen's inequality or exp monotonicity
  apply Real.log_sum_exp_ge_max
  exact h
```

**Approach B (Via softmax):**
```lean
theorem loss_nonneg {n : Nat} (pred : Vector n) (target : Nat) :
  target < n → crossEntropyLoss pred target ≥ 0 := by
  intro h
  -- L = -log(softmax(pred)[target])
  -- softmax(pred)[target] ∈ (0, 1]
  have softmax_bounded : softmax pred target ∈ Set.Ioc 0 1 := by
    apply softmax_in_unit_interval
  have log_neg : Real.log (softmax pred target) ≤ 0 := by
    apply Real.log_nonpos
    exact softmax_bounded.1
    exact softmax_bounded.2
  linarith
```

**Required Mathlib lemmas:**
- `Real.log_nonpos : 0 < x → x ≤ 1 → log x ≤ 0`
- `Real.exp_pos : ∀ x, 0 < exp x`

**Priority:** High - Core mathematical property

---

### Network Module (9 user axioms)

**Files with Axioms:**

#### Network/Architecture.lean (2 sorry statements)

| Line | Function | Difficulty | Estimated Time |
|------|----------|------------|----------------|
| 85 | `argmax` | Easy | 30 min |
| 129 | `predictBatch` | Easy | 30 min |

**Elimination Strategy:**

```lean
-- Line 85
def argmax {n : Nat} (v : Vector n) : Nat :=
  Id.run do
    let mut maxIdx := 0
    let mut maxVal := v[0]
    for i in [1:n] do
      if v[i] > maxVal then
        maxIdx := i
        maxVal := v[i]
    return maxIdx

-- Line 129
def MLPArchitecture.predictBatch {b : Nat} (net : MLPArchitecture) (X : Batch b 784) : Array Nat :=
  let outputs := net.forwardBatch X
  Id.run do
    let mut predictions := Array.empty
    for i in [0:b] do
      let row : Vector 10 := ⊞ j => outputs[i, j]
      predictions := predictions.push (argmax row)
    return predictions
```

**Priority:** High - Required for inference

#### Network/Gradient.lean (7 sorry statements)

| Line | Function | Difficulty | Estimated Time |
|------|----------|------------|----------------|
| 56 | `flattenParams` | Medium | 4-6 hours |
| 72 | `unflattenParams` | Medium | 4-6 hours |
| 80 | `unflatten_flatten_id` | Medium | 2-3 hours |
| 88 | `flatten_unflatten_id` | Medium | 2-3 hours |
| 140 | `networkGradient` | Hard | 8-12 hours |
| 161 | `networkGradientBatch` | Hard | 4-6 hours |
| 176 | `computeLossBatch` | Medium | 2-3 hours |

**Elimination Strategy:**

**Phase 1: Implement flatten/unflatten (Priority)**
```lean
def flattenParams (net : MLPArchitecture) : Vector nParams :=
  let w1Size := 784 * 128
  let b1Size := 128
  let w2Size := 128 * 10
  let b2Size := 10

  ⊞ (i : Idx nParams) =>
    let idx := i.1.toNat
    if h : idx < w1Size then
      let row := idx / 128
      let col := idx % 128
      net.layer1.weights[row, col]
    else if h : idx < w1Size + b1Size then
      net.layer1.bias[idx - w1Size]
    else if h : idx < w1Size + b1Size + w2Size then
      let offset := idx - w1Size - b1Size
      let row := offset / 10
      let col := offset % 10
      net.layer2.weights[row, col]
    else
      net.layer2.bias[idx - w1Size - b1Size - w2Size]

def unflattenParams (params : Vector nParams) : MLPArchitecture :=
  -- Inverse extraction from flattened vector
  -- Extract slices and reconstruct network structure
```

**Phase 2: Prove isomorphisms**
```lean
theorem unflatten_flatten_id (net : MLPArchitecture) :
  unflattenParams (flattenParams net) = net := by
  ext
  -- Prove field-by-field equality
  -- Uses index arithmetic and DataArrayN extensionality
```

**Phase 3: Implement gradient computation**
```lean
def networkGradient (params : Vector nParams) (input : Vector 784) (target : Nat) : Vector nParams :=
  let lossFunc := fun p => computeLoss p input target
  (∇ p, lossFunc p) params
    |>.rewrite_by fun_trans (disch := aesop)
```

**Priority:** **Critical** - Required for training loop

---

### Optimizer Module (0 user axioms)

**Status:** ✅ Clean - No user-introduced axioms

**Files:**
- `Optimizer/SGD.lean` - Complete ✅
- `Optimizer/Momentum.lean` - Complete ✅
- `Optimizer/Update.lean` - Complete ✅

**Recommendation:** No action needed.

---

### Verification Module (26 user axioms + 11 axiom declarations)

#### Verification/GradientCorrectness.lean (10 sorry statements + 1 axiom declaration)

**Axiom Declaration:**

| Line | Axiom | Eliminable? | Strategy |
|------|-------|-------------|----------|
| 264 | `gradient_matches_finite_difference` | ✅ YES | Use mathlib's `has_fderiv_at_iff_is_o` |

**Sorry Statements:**

| Line | Theorem | Difficulty | Priority |
|------|---------|------------|----------|
| 63, 73 | `relu_gradient_almost_everywhere` | Medium | High |
| 87 | `sigmoid_gradient_correct` | Medium | High |
| 105 | `matvec_gradient_wrt_vector` | Easy | High |
| 120 | `matvec_gradient_wrt_matrix` | Easy | High |
| 143 | `vadd_gradient_correct` | Easy | High |
| 152 | `smul_gradient_correct` | Easy | High |
| 190 | `layer_composition_gradient_correct` | Medium | High |
| 212 | `cross_entropy_softmax_gradient_correct` | Hard | **Critical** |
| 244 | `network_gradient_correct` | Medium | **Critical** |

**Elimination Strategy:**

**Quick Wins (4 proofs, ~2 hours total):**
```lean
-- Linear algebra operations
theorem matvec_gradient_wrt_vector ... := by
  unfold matvec
  fun_trans

theorem vadd_gradient_correct ... := by
  fun_prop

theorem smul_gradient_correct ... := by
  fun_prop
```

**Medium Proofs (4 proofs, ~8-12 hours total):**
- ReLU: Use `deriv_eventually_eq` pattern for x ≠ 0
- Sigmoid: Chain rule + exp derivative
- Layer composition: Compose component proofs

**Critical Proofs (2 proofs, ~16-24 hours total):**
- Cross-entropy softmax gradient: Jacobian calculation
- Network gradient: End-to-end chain rule

**Priority:** **Highest** - Aligns with primary project goal (gradient correctness)

#### Verification/TypeSafety.lean (11 sorry statements + 4 axiom declarations)

**Axiom Declarations:**

| Line | Axiom | Eliminable? | Strategy |
|------|-------|-------------|----------|
| 59 | `dataArrayN_size_correct` | ⚠️ Maybe | Upstream to SciLean |
| 271 | `flatten_unflatten_left_inverse` | ✅ YES | Array indexing arithmetic |
| 284 | `unflatten_flatten_right_inverse` | ✅ YES | Array indexing arithmetic |
| 320 | `gradient_dimension_matches_params` | ⚠️ Maybe | May follow from SciLean's fderiv |

**Sorry Statements (9 easy, 2 hard):**

**Easy (Apply type signatures, ~2-3 hours total):**
- Lines 112, 124, 135, 150, 181, 203, 222, 240, 257

**Hard (SciLean-dependent):**
- Lines 83, 86, 99: Multi-dimensional array properties

**Priority:** Medium - Important for type safety verification

#### Verification/Convergence.lean (4 sorry statements + 5 axiom declarations)

**Axiom Declarations (Intentionally out of scope):**

| Line | Axiom | Eliminable? | Effort if pursued |
|------|-------|-------------|-------------------|
| 120 | `sgd_converges_strongly_convex` | ⚠️ YES (out of scope) | Months |
| 161 | `sgd_converges_convex` | ⚠️ YES (out of scope) | Months |
| 203 | `sgd_finds_stationary_point_nonconvex` | ⚠️ YES (out of scope) | Months |
| 321 | `batch_size_reduces_variance` | ⚠️ YES (out of scope) | Weeks |
| Implicit | Series convergence axioms | ✅ YES | Easy (search mathlib) |

**Sorry Statements (3 easy, 1 needs fix):**

| Line | Lemma | Status |
|------|-------|--------|
| 276, 282, 300 | Robbins-Monro conditions | Easy - use mathlib |
| 308 | `one_over_sqrt_t` convergence | Needs revision (proof sketch incorrect) |

**Priority:** Low - Explicitly out of scope per project design

#### Verification/Tactics.lean (3 intentional placeholders)

**Status:** Not axioms to eliminate - tactics to implement

**Priority:** Medium - Useful for proof automation

---

### Training Module (1 user axiom)

#### Training/Metrics.lean (1 sorry statement)

| Line | Function | Difficulty | Estimated Time |
|------|----------|------------|----------------|
| 31 | `getPredictedClass` | Easy | 5 min |

**Elimination Strategy:**
```lean
def getPredictedClass {n : Nat} (output : Vector n) : Nat :=
  Id.run do
    let mut maxIdx := 0
    let mut maxVal := if h : 0 < n then output[⟨0, h⟩] else 0.0
    for i in [1:n] do
      if output[i] > maxVal then
        maxIdx := i
        maxVal := output[i]
    return maxIdx
```

**Priority:** Medium - Simple utility function

---

### Data Module (3 user axioms)

#### Data/Preprocessing.lean (3 sorry statements)

| Line | Issue | Difficulty | Estimated Time |
|------|-------|------------|----------------|
| 102, 108 | Zero vector fallback | Trivial | 1 min |
| 159 | Arithmetic proof | Trivial | 1 min |

**Elimination Strategy:**
```lean
-- Lines 102, 108
return (0 : Vector 784)  -- Replace sorry

-- Line 159
let idx : Idx 784 := ⟨linearIdx.toUSize, by omega⟩  -- Replace sorry with proof
```

**Priority:** High - Quick wins

---

### Testing Module (0 user axioms)

**Status:** ✅ Clean - No user-introduced axioms

**Files:**
- All testing files complete ✅

**Note:** Some files have compilation errors (not axiom-related)

---

## Compilation Issues (Fix First)

Before axiom elimination, fix these blocking errors:

1. **Network/Initialization.lean:134** - Syntax error
   ```diff
   -def initDenseLayerXavier (inDim outDim : Nat) : IO (DenseLayer inDim outDim) := do
   +def initDenseLayerXavier (inDim : Nat) (outDim : Nat) : IO (DenseLayer inDim outDim) := do
   ```

2. **Training/Loop.lean** - Type synthesis errors (multiple locations)
3. **Testing/GradientCheck.lean** - Type errors with GetElem instances

**Estimated Time:** 1-2 hours

---

## Elimination Roadmap

### Phase 1: Quick Wins (1-2 days)

**Goal:** Eliminate 23 trivial axioms

1. ✅ Fix compilation errors (2 hours)
2. ✅ Data/Preprocessing.lean (3 axioms, 5 min)
3. ✅ Training/Metrics.lean (1 axiom, 30 min)
4. ✅ Network/Architecture.lean (2 axioms, 1 hour)
5. ✅ Verification/TypeSafety.lean easy proofs (9 axioms, 3 hours)
6. ✅ Verification/GradientCorrectness.lean linear algebra (4 axioms, 2 hours)
7. ✅ Verification/Convergence.lean series lemmas (3 axioms, 2 hours)

**Total:** 22 axioms eliminated

### Phase 2: Core Verification (1-2 weeks)

**Goal:** Prove gradient correctness for core operations

1. ✅ Layer/Properties.lean (2 axioms, 4-6 hours)
   - Add lemmas to Core/LinearAlgebra
   - Complete linearity proofs

2. ✅ Loss/Properties.lean (1 axiom, 4-6 hours)
   - Prove loss non-negativity

3. ✅ Verification/GradientCorrectness.lean activations (4 axioms, 8-12 hours)
   - ReLU gradient (subgradient handling)
   - Sigmoid gradient
   - Layer composition

4. ✅ Verification/TypeSafety.lean isomorphisms (2 axioms, 8-12 hours)
   - Flatten/unflatten inverses (coordinate with Network/Gradient)

**Total:** 9 axioms eliminated

### Phase 3: Network Gradients (2-3 weeks)

**Goal:** Complete end-to-end gradient computation

1. ✅ Network/Gradient.lean flatten/unflatten (4 axioms, 12-16 hours)
   - Implement parameter flattening
   - Prove isomorphisms

2. ✅ Network/Gradient.lean batched operations (3 axioms, 8-12 hours)
   - networkGradient via SciLean AD
   - Batched gradient computation

3. ✅ Verification/GradientCorrectness.lean end-to-end (2 axioms, 16-24 hours)
   - Cross-entropy softmax gradient
   - Network gradient correctness

**Total:** 9 axioms eliminated

### Phase 4: SciLean Coordination (Ongoing)

**Goal:** Upstream axioms to SciLean or document as acceptable

1. 📧 Contact SciLean maintainers about:
   - `dataArrayN_size_correct` proof
   - Multi-dimensional array size properties
   - `gradient_dimension_matches_params` derivation

2. 📝 Document remaining acceptable axioms:
   - Float↔Real isomorphism rules (project philosophy)
   - SciLean.sorryProofAxiom (library dependency)

**Total:** 5 axioms coordinated/documented

### Out of Scope (Preserve)

**Convergence axioms** - Keep as documented out-of-scope per project design:
- sgd_converges_strongly_convex
- sgd_converges_convex
- sgd_finds_stationary_point_nonconvex
- batch_size_reduces_variance

**Total:** 5 axioms intentionally preserved

---

## Summary Statistics

### Current State

| Category | Count | Status |
|----------|-------|--------|
| User sorry statements | 44 | 38 eliminable |
| Axiom declarations | 11 | 1 eliminable, 5 out of scope, 5 SciLean-dependent |
| Standard Lean axioms | 3 | Acceptable |
| SciLean library axioms | ~10 | Acceptable per project scope |

### After All Phases

| Category | Remaining | Justification |
|----------|-----------|---------------|
| User sorry statements | 0 | ✅ All eliminated |
| Axiom declarations | 10 | 5 out of scope, 5 SciLean-coordinated |
| Standard Lean axioms | 3 | Fundamental to Lean |
| SciLean library axioms | ~10 | Float↔Real gap (project philosophy) |

### Effort Breakdown

| Phase | Axioms Eliminated | Time Estimate |
|-------|------------------|---------------|
| Quick Wins | 22 | 1-2 days |
| Core Verification | 9 | 1-2 weeks |
| Network Gradients | 9 | 2-3 weeks |
| SciLean Coordination | 5 coordinated | Ongoing |
| **Total** | **40 eliminated** | **4-6 weeks** |

---

## Acceptable Axioms (Final State)

After elimination efforts, the following axioms will remain and are **acceptable**:

### 1. Standard Lean/Mathlib Axioms (3)
- `propext` - Propositional extensionality
- `Classical.choice` - Axiom of choice
- `Quot.sound` - Quotient soundness

**Justification:** Fundamental to classical mathematics in Lean. Nearly all non-trivial mathematics requires these.

### 2. SciLean Library Axioms (~10)
- `SciLean.sorryProofAxiom` - SciLean's internal axiom for incomplete proofs
- Float↔Real isomorphism rules - Bridge between Float and Real

**Justification:** From project's CLAUDE.md:
> "Verification Philosophy: Mathematical properties proven on ℝ (real numbers), computational implementation in Float (IEEE 754). The Float→ℝ gap is acknowledged—we verify symbolic correctness, not floating-point numerics."

### 3. Convergence Theory Axioms (5, out of scope)
- `sgd_converges_strongly_convex`
- `sgd_converges_convex`
- `sgd_finds_stationary_point_nonconvex`
- `batch_size_reduces_variance`
- Series convergence (if not in mathlib)

**Justification:** From verified-nn-spec.md Section 5.4:
> "Convergence proofs are explicitly out of scope. The primary goal is gradient correctness, not optimization theory."

### 4. SciLean Infrastructure (5, coordinate upstream)
- `dataArrayN_size_correct` - DataArrayN has correct size
- `matrix_size_correct` - Matrix dimensions correct
- `batch_size_correct` - Batch dimensions correct
- `gradient_dimension_matches_params` - Gradient dimension preservation
- Multi-dimensional array properties

**Justification:** These are properties of SciLean's array types that should be proven within SciLean itself. Coordinate with SciLean maintainers to add these proofs upstream.

---

## Next Steps

1. **Immediate (This week):**
   - Fix compilation errors
   - Execute Phase 1 (Quick Wins)
   - Create tracking issues for each axiom

2. **Short-term (Next 2 weeks):**
   - Execute Phase 2 (Core Verification)
   - Begin Phase 3 (Network Gradients)

3. **Medium-term (Next month):**
   - Complete Phase 3
   - Contact SciLean maintainers (Phase 4)

4. **Long-term (Ongoing):**
   - Monitor SciLean development
   - Update documentation
   - Maintain axiom-free status for user code

---

## Conclusion

The VerifiedNN project can achieve **0 user-introduced axioms** with focused effort over 4-6 weeks. The remaining axioms will be:
- Standard mathematical foundations (acceptable)
- SciLean library dependencies (acceptable per project scope)
- Convergence theory (intentionally out of scope)
- SciLean infrastructure (coordinate upstream)

This aligns perfectly with the project's verification philosophy and primary goal of proving gradient correctness.
