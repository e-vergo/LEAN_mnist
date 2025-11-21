# File Review: Gradient.lean

## Summary
**DEPRECATED / NONCOMPUTABLE REFERENCE IMPLEMENTATION.** Defines parameter flattening and gradient computation using SciLean's automatic differentiation (`∇` operator). Contains 2 axioms (flatten/unflatten inverses) and 4 documentation-only sorries. **Superseded by ManualGradient.lean for executable training.** Should be retained as a correctness specification but clearly marked as noncomputable reference.

## Findings

### Orphaned Code
**Partially orphaned - noncomputable functions unused in production:**

**Still referenced (as specification):**
- `flattenParams` - Used in TypeSafety.lean, ManualGradient.lean, GradientFlattening.lean
- `unflattenParams` - Used in Loop.lean, ManualGradient.lean, FiniteDifference.lean
- `nParams` - Used throughout (GradientFlattening, ManualGradient, Training, Testing)
- `unflatten_flatten_id` - Referenced in TypeSafety.lean (verification)
- `flatten_unflatten_id` - Referenced in TypeSafety.lean (verification)

**Noncomputable / unused in production:**
- **Line 430-434: `networkGradient`** - Noncomputable AD-based gradient (superseded by `networkGradientManual`)
  - Still used in: FiniteDifference.lean, InspectGradient.lean (testing/debugging only)
  - **NOT used in production training** (MNISTTrainFull, MNISTTrainMedium use manual version)
- **Line 464: `networkGradient'`** - Alias for networkGradient (also noncomputable)
- **Line 479-497: `networkGradientBatch`** - Noncomputable batch gradient
  - Not used anywhere (production uses manual backprop)
- **Line 508-523: `computeLossBatch`** - Batch loss computation
  - Used in DebugTraining.lean only (not production)

**Evidence of supersession:**
- CLAUDE.md line 441: "lake exe simpleExample # ❌ Cannot execute - noncomputable main"
- CLAUDE.md line 442: "lake exe trainManual # ❌ Cannot execute - noncomputable main"
- README.md line 84: "**Executable Training:** Use ManualGradient.networkGradientManual"

### Axioms (Total: 2)
**Line 241: `axiom unflatten_flatten_id`**
- **Category:** Type safety (parameter marshalling)
- **Documentation:** ✓ Excellent (58 lines, lines 197-241)
- **Justification:**
  - Requires DataArrayN extensionality (not available in SciLean)
  - SciLean's DataArray.ext is itself axiomatized as `sorry_proof`
  - Algorithmically true (bijection by construction)
  - Essential for gradient descent correctness
- **References:** SciLean/Data/DataArray/DataArray.lean:130, Certigrad paper

**Line 394: `axiom flatten_unflatten_id`**
- **Category:** Type safety (parameter marshalling, dual of above)
- **Documentation:** ✓ Excellent (150 lines including proof sketch, lines 244-394)
- **Justification:**
  - Same as above (dual property)
  - **Contains detailed proof sketch** (lines 270-367) showing how to prove once DataArrayN.ext available
  - Proof strategy includes case analysis on 4 parameter ranges
  - Lists needed lemmas: Nat.div_add_mod, Nat.add_sub_cancel', if_pos, if_neg, natToIdx_toNat_inverse
- **References:** Same as above

**Assessment:** Both axioms are **adequately documented and justified**. They axiomatize what is algorithmically true but unprovable without SciLean infrastructure. The proof sketch for `flatten_unflatten_id` is exemplary documentation.

### Sorries (Total: 4 - ALL DOCUMENTATION MARKERS)
**IMPORTANT:** Lines 300, 321, 345, 366 contain `sorry` **INSIDE COMMENTED PROOF SKETCH**, not executable code.

**Line 35-38 (module docstring):**
```lean
The 4 `sorry` occurrences at lines 294, 315, 339, 360 are **documentation markers**
within a proof sketch comment, NOT executable code. They serve as placeholders in
the proof roadmap showing how `flatten_unflatten_id` would be proven once DataArrayN.ext
becomes available. These do not compile into the binary.
```

**Actual sorries (in multi-line comment block within axiom docstring):**
- **Line 300:** "DOCUMENTATION MARKER: Index arithmetic automation for USize ↔ Nat ↔ Idx chain"
- **Line 321:** "DOCUMENTATION MARKER: Similar to case 1, requires branch selection automation"
- **Line 345:** "DOCUMENTATION MARKER: Constant arithmetic normalization required"
- **Line 366:** "DOCUMENTATION MARKER: Final case, constant arithmetic + branch selection"

These are **NOT compiled** - they exist purely to document the proof strategy.

**Executable sorries:** **0** (none in actual code)

### Code Correctness Issues
**Minor documentation discrepancy:**

**Line 71-73: Proven theorem `array_range_mem_bound`**
- Docstring claims "Previously eliminated" but theorem is actually **proven** (not eliminated)
- Should say "Now proven using Array.mem_def, Array.toList_range, and List.mem_range"
- Proof is complete (lines 71-73), not a TODO

**No algorithmic issues detected:**
- `flattenParams` index arithmetic verified via omega (lines 125-151)
- `unflattenParams` index arithmetic verified via omega (lines 163-195)
- `computeLoss` correctly composes unflatten → forward → crossEntropyLoss
- `networkGradient` uses standard SciLean AD pattern (even though noncomputable)

### Hacks & Deviations
**Significant: Noncomputable AD (by design, not hack)**
- **Line 430-434: `networkGradient` marked `noncomputable`**
  - Severity: Significant (blocks executable training)
  - Justification: SciLean's `∇` operator is fundamentally noncomputable
  - Workaround: ManualGradient.lean provides computable alternative
  - Documentation: Lines 440-462 explain limitation and workarounds

**Line 440-462: Extensive documentation of noncomputability**
```lean
**Current Limitation:** SciLean's automatic differentiation uses noncomputable
functions at the type level. Our complex loss pipeline (unflatten + matmul + ReLU +
softmax + cross-entropy) cannot be made computable via standard rewrite_by patterns.
```

Lists 4 workaround options and explains why this happens.

**Minor: Helper function `natToIdx` (lines 77-78)**
- Uses `Idx.finEquiv` internally to avoid USize conversion proofs
- Standard pattern in SciLean, not a hack

**Float/ℝ gap (acknowledged, not addressed):**
- `computeLoss` and `networkGradient` operate on Float
- Verification theorems (GradientCorrectness.lean) prove properties on ℝ
- Gap documented in CLAUDE.md but not mentioned in this file's docstring

## Statistics
- **Definitions:** 9 total (5 computable utilities + 4 noncomputable AD functions)
- **Theorems:** 2 proven (array_range_mem_bound, nParams_value), 0 with sorry
- **Axioms:** 2 total, 2 excellently documented
- **Sorries (executable):** 0
- **Sorries (documentation markers):** 4 (in comment block)
- **Lines of code:** 526
- **Documentation quality:** Excellent (comprehensive module docstring + detailed axiom justification)
- **Usage:** High for utilities (flattenParams, unflattenParams, nParams), Low for AD functions (superseded)

## Recommendations

### Priority 1: Clarify File Status (Moderate)
**Add prominent deprecation notice to module docstring:**
```lean
/-!
# Network Gradient Computation (NONCOMPUTABLE REFERENCE)

**⚠️ FOR EXECUTABLE TRAINING, USE `Network.ManualGradient` INSTEAD.**

This module provides the *specification* for gradient computation using SciLean's
automatic differentiation. The `networkGradient` function is **noncomputable** and
cannot be compiled to executable code.

**Recommended usage:**
- Production training: Use `ManualGradient.networkGradientManual` (computable)
- Gradient verification: Use this module as correctness specification
- Finite difference testing: Compare manual gradients against this specification

## Main Definitions (Parameter Utilities - COMPUTABLE)
- `flattenParams`: Convert network structure to parameter vector
- `unflattenParams`: Convert parameter vector to network structure
...

## Gradient Functions (NONCOMPUTABLE - REFERENCE ONLY)
- `networkGradient`: Automatic differentiation (noncomputable)
- `networkGradientBatch`: Batched AD gradient (noncomputable)
...
```

### Priority 2: Mark Deprecated Functions (Low)
Add `@[deprecated]` attribute or comments:
```lean
/-- **DEPRECATED: Use ManualGradient.networkGradientManual for executable code.**
Compute gradient of loss with respect to network parameters.
...
-/
@[inline]
noncomputable def networkGradient ...
```

### Priority 3: Update Verification Status (Low)
Module docstring line 24 claims:
```lean
- **Axioms:** 0 (no axiomatized properties)
```
Should be:
```lean
- **Axioms:** 2 (flatten/unflatten inverses, both justified - see axiom docstrings)
```

Line 46 states verification status is outdated.

## Critical Assessment

**Strengths:**
- Exemplary axiom documentation (58-line and 150-line docstrings with proof sketches)
- Clean parameter flattening implementation with verified index arithmetic
- Serves as correctness specification for ManualGradient.lean
- Proven helper theorems (array_range_mem_bound)

**Weaknesses:**
- File purpose not immediately clear (looks like production code, but isn't)
- Noncomputable functions may mislead users into thinking AD works
- No clear pointer to ManualGradient.lean for executable alternative
- Module docstring doesn't explain the noncomputability issue

**Relationship to ManualGradient.lean:**
This file should be retained as the **formal specification**. ManualGradient.lean is the **computable implementation** that should match this specification. The verification theorem would be:
```lean
theorem manual_matches_automatic :
  networkGradientManual params input target = networkGradient params input target
```

**Verdict:** Keep file, but add prominent deprecation notice and clarify role as specification.
