# Automatic Differentiation Registration - Research Summary

## Documents Delivered

This research package includes three complementary documents:

### 1. **AD_REGISTRATION_RESEARCH_REPORT.md** (Comprehensive)
- Complete explanation of SciLean's AD system
- All major patterns found in SciLean codebase
- Detailed analysis of linear, bilinear, and nonlinear operations
- Step-by-step proof strategies for each type
- Potential issues and solutions
- 600+ lines of detailed reference material

**Use when:** You need deep understanding of how/why things work

### 2. **AD_REGISTRATION_QUICKSTART.md** (Quick Reference)
- TL;DR overview of @[fun_prop] vs @[data_synth]
- Three operation types with examples
- Naming conventions and proof strategies
- Quick reference table
- Common patterns and testing approaches

**Use when:** You need syntax or pattern examples quickly

### 3. **AD_REGISTRATION_IMPLEMENTATION_CHECKLIST.md** (Actionable)
- 29 operations broken into 10 implementation phases
- Phase-by-phase dependencies and testing
- Before/during/after checklists
- Success criteria
- Common pitfalls to avoid

**Use when:** Actually implementing the registrations

---

## Key Findings

### SciLean Registration System

SciLean uses a **rule-based automatic differentiation** system with two primary attributes:

1. **@[fun_prop]** - Marks differentiability/continuity properties
   - Used by the `fun_prop` tactic for compositional proofs
   - Simplest to implement

2. **@[data_synth]** - Marks concrete derivative implementations
   - Provides forward-mode and reverse-mode rules
   - Used by the `data_synth` tactic
   - More work but necessary for custom operations

### Pattern Templates Found in SciLean

**Pattern 1: Linear Operations (MatVecMul.lean, VecMatMul.lean)**
```lean
@[fun_prop]
theorem operation.arg_VAR.IsContinuousLinearMap_rule : ... := sorry_proof

@[data_synth]
theorem operation.arg_VAR.HasRevFDeriv_rule : HasRevFDeriv R operation ... := sorry_proof
```

**Pattern 2: Nonlinear Operations (Exp.lean)**
```lean
def_fun_prop operation in x : Differentiable K by sorry_proof
abbrev_fun_trans operation in x : fderiv K by equals ... => sorry_proof
abbrev_data_synth operation in x : HasRevFDeriv K by hasRevFDeriv_from_def => simp; to_ssa
```

**Pattern 3: Special Operations (Log.lean)**
```lean
@[fun_prop]
theorem log.arg_x.Differentiable_rule (x : W → R) : Differentiable R ... := by fun_prop (disch:=aesop)

@[data_synth]
theorem log.arg_x.HasRevFDeriv_rule : HasRevFDeriv R ... := by
  apply hasRevFDeriv_from_hasFDerivAt_hasAdjoint
  ...
```

### Critical Patterns for This Project

#### For Linear Operations (vadd, smul, transpose, etc.)
**What to do:** Register with `@[fun_prop]`, let composition handle derivatives
```lean
@[fun_prop]
theorem operation.Differentiable : Differentiable Float operation := by
  unfold operation
  fun_prop
```

#### For Bilinear Operations (vmul, dot, matvec, etc.)
**What to do:** Register `@[fun_prop]` + separate `@[data_synth]` for each argument
```lean
@[fun_prop] theorem operation.arg_x.Differentiable ... := by fun_prop
@[data_synth] theorem operation.arg_x.HasRevFDeriv ... := by data_synth
@[data_synth] theorem operation.arg_y.HasRevFDeriv ... := by data_synth
```

#### For Nonlinear Operations (relu, sigmoid, softmax, etc.)
**What to do:** Full implementation with `@[fun_prop]` + explicit `@[data_synth]`
```lean
@[fun_prop]
theorem operation.Differentiable : Differentiable Float operation := by
  unfold operation
  fun_prop

@[data_synth]
theorem operation.HasRevFDeriv :
  HasRevFDeriv Float operation (fun x => (result, pullback)) := by
  unfold operation
  data_synth
```

---

## Operations Inventory

### LinearAlgebra.lean (18 operations)

**Linear Operations (9):** vadd, vsub, smul, matAdd, matSub, matSmul, transpose, batchAddVec, (and inner matrix ops)

**Bilinear Operations (5):** vmul, dot, outer, matvec, matmul

**Nonlinear Operations (2):** normSq, norm

**Batch Operations (2):** batchMatvec, batchAddVec

### Activation.lean (11 operations)

**ReLU Family (5):** relu, reluVec, reluBatch, leakyRelu, leakyReluVec

**Sigmoid Family (5):** sigmoid, sigmoidVec, sigmoidBatch, tanh, tanhVec

**Classification (1):** softmax

---

## Important Discoveries

### 1. SciLean Already Provides Some Operations
Before implementing custom registrations, check if SciLean already has:
- `matvec` and `vecMatMul` (MatVecMul.lean, VecMatMul.lean) - CONFIRMED
- `exp` (Exp.lean) - CONFIRMED
- `log` (Log.lean) - CONFIRMED
- `softmax` - NEED TO CHECK (possibly in DataArrayN rules)

### 2. Two Proof Approaches
- **Declarative:** Use `sorry_proof` placeholders (as SciLean does)
- **Constructive:** Provide actual proofs via `fun_prop` tactic or `data_synth` tactic

The project can mix both - use `sorry_proof` for complex proofs that need mathematical work, use tactic proofs for compositions that SciLean can derive.

### 3. Naming Convention is Strict
```
functionName.arg_ARGUMENT.PropertyName_rule
```
Naming must follow this pattern for SciLean's tactic system to find the rules.

### 4. Both Forward and Reverse Modes Usually Needed
While forward-mode AD (HasFwdFDeriv) is simpler, reverse-mode (HasRevFDeriv) is essential for efficient backpropagation in neural networks. Many operations should implement both:
- `HasFwdFDeriv` - For forward-mode AD
- `HasRevFDeriv` - For reverse-mode AD (backpropagation)
- `HasRevFDerivUpdate` - For in-place gradient accumulation (optimization)

### 5. Proof Difficulty Varies by Type
- **Linear:** 1-2 lines proof (just `fun_prop`)
- **Bilinear:** 5-10 lines (define pullback, apply tactics)
- **Nonlinear:** 20-50 lines (may need decomposition into steps)
- **Special Cases:** 50-100+ lines (e.g., log requires positivity side conditions)

---

## Implementation Strategy Recommended

### Parallel Work Possible

Because these operations have clear dependencies, work can be parallelized:

1. **Thread A:** Linear operations (phases 1-2 of LinearAlgebra)
   - vadd, vsub, smul, matAdd, matSub, matSmul, transpose
   - All can be worked on simultaneously
   - ~1-2 hours total

2. **Thread B:** ReLU/Sigmoid (phases A-B of Activation)
   - relu, sigmoid, tanh, leaky variants
   - Can start immediately
   - ~2-3 hours total

3. **Thread C:** Matrix operations (phases 3-5 of LinearAlgebra)
   - vmul, dot, matvec, matmul
   - Some interdependencies, but can start after checking SciLean
   - ~3-4 hours total

4. **Thread D:** Advanced operations (phases 6-7 of LinearAlgebra, phase C of Activation)
   - normSq, norm, outer, softmax, batch ops
   - Can proceed once dependencies are registered
   - ~2-3 hours total

**Total estimated work:** 8-12 hours for experienced Lean developer, 15-20 hours for less experienced

### Suggested Order for Single Developer

1. **Day 1:** Phases 1-2 (Linear operations in LinearAlgebra) - foundation
2. **Day 2:** Phases A (ReLU in Activation) - core activation
3. **Day 3:** Phases 3-4 (Bilinear + norms in LinearAlgebra) - essential operations
4. **Day 4:** Phases B (Sigmoid/tanh in Activation) + Phase 5 (matrix ops)
5. **Day 5:** Phases 6-7 (advanced) + Phase C (softmax)

With testing/debugging after each day.

---

## Risk Analysis

### Low Risk Operations
- Linear operations (vadd, smul, transpose, etc.)
  - Proof is just `fun_prop`
  - SciLean's composition rules handle everything
  - **Estimated effort:** 10 min each

### Medium Risk Operations
- Bilinear operations (vmul, dot)
  - Need explicit reverse-mode definition
  - Formula straightforward to verify
  - **Estimated effort:** 20-30 min each

- Vectorized operations (reluVec, sigmoidVec)
  - May inherit from scalar version
  - Likely just need wrapper `@[fun_prop]`
  - **Estimated effort:** 5 min each

### Higher Risk Operations
- Nonlinear with composition (sigmoid with exp, norm with sqrt)
  - Depends on Float.exp, Float.sqrt being registered
  - May need side conditions (positivity, non-zero)
  - **Estimated effort:** 30-45 min each

- Softmax
  - Complex formula with multiple cases
  - Numerical stability trick (max subtraction) may affect proof
  - May already be in SciLean
  - **Estimated effort:** 45 min to 2 hours (or 0 if SciLean has it)

### Unknown Risk
- **Float.exp differentiability:** Need to verify this exists in SciLean
- **Float.sqrt differentiability:** Need to verify
- **Softmax availability:** Need to check SciLean
- **Batch operation composition:** May work automatically or need explicit registration

**Mitigation:** Create quick `grep` script to check these before implementation

---

## Success Metrics

### Phase Completion Criteria

Each phase is successful when:
1. Code compiles without errors: `lake build VerifiedNN.Core.LinearAlgebra`
2. All relevant TODOs are removed (no "TODO.*fun_trans" remaining)
3. Gradient checking tests pass (if implemented)
4. No warnings from Lean beyond expected sorries

### Project Success Criteria

Project is complete when:
1. **All 29 operations registered** with appropriate attributes
2. **All TODOs replaced** with actual theorems or removed
3. **Gradient checking passes** - numerical gradients match symbolic derivatives
4. **Training works** - MNIST training completes without errors
5. **Tests pass** - All existing tests still pass

---

## Next Steps After Research

1. **Setup:**
   - Copy all three documents to project root (DONE)
   - Review each document to understand patterns
   - Set up SciLean template files for reference

2. **Pre-implementation checks:**
   - Verify Float.exp is differentiable in SciLean
   - Check if softmax is available
   - Check if matvec registrations already exist
   - Run baseline tests to ensure current state is good

3. **Phased implementation:**
   - Follow AD_REGISTRATION_IMPLEMENTATION_CHECKLIST.md
   - One phase at a time
   - Test after each phase
   - Document any findings or deviations

4. **During implementation:**
   - Use AD_REGISTRATION_QUICKSTART.md for syntax
   - Reference AD_REGISTRATION_RESEARCH_REPORT.md for detailed patterns
   - Follow naming conventions strictly
   - Keep proofs as simple as possible (use `sorry_proof` if needed)

5. **After completion:**
   - Update module docstrings
   - Run full test suite
   - Update verification status
   - Document lessons learned

---

## Files Referenced

### SciLean Reference Files
- `.lake/packages/SciLean/SciLean/AD/Rules/MatVecMul.lean` (bilinear template)
- `.lake/packages/SciLean/SciLean/AD/Rules/VecMatMul.lean` (variant of bilinear)
- `.lake/packages/SciLean/SciLean/AD/Rules/Exp.lean` (nonlinear template)
- `.lake/packages/SciLean/SciLean/AD/Rules/Log.lean` (with side conditions)
- `.lake/packages/SciLean/SciLean/AD/Rules/Common.lean` (macro definitions)

### Project Files to Modify
- `VerifiedNN/Core/LinearAlgebra.lean` (18 operations)
- `VerifiedNN/Core/Activation.lean` (11 operations)

### Reference Documentation Created
- `AD_REGISTRATION_RESEARCH_REPORT.md` (this folder) - CREATED
- `AD_REGISTRATION_QUICKSTART.md` (this folder) - CREATED
- `AD_REGISTRATION_IMPLEMENTATION_CHECKLIST.md` (this folder) - CREATED

---

## Conclusion

This research provides a complete roadmap for registering automatic differentiation attributes in the LEAN_mnist project. The analysis of SciLean's patterns reveals that:

1. **Registration is systematic** - Follow strict naming and attribute conventions
2. **Complexity varies** - Linear ops are trivial, nonlinear ops require more care
3. **SciLean provides templates** - MatVecMul.lean and Exp.lean are excellent references
4. **Both attributes usually needed** - @[fun_prop] for differentiability, @[data_synth] for derivative computation
5. **Implementation is feasible** - All 29 operations can be registered following documented patterns

The three accompanying documents provide:
- **Deep understanding** (Research Report)
- **Quick reference** (Quickstart)
- **Actionable checklist** (Implementation Checklist)

With these resources, implementation can proceed with confidence.

---

## Questions? Next Steps?

If spawning agents to implement:

1. **For exploration:** Use general-purpose agent to run verification searches
2. **For implementation:** Use code-focused agent with this research as context
3. **For testing:** Verify each phase compiles and gradient checks pass

All three documents should be committed to the repository for reference during implementation.

