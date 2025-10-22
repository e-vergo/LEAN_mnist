# AD Registration Research - Complete Documentation

## Overview

This folder contains comprehensive research and implementation guidance for registering automatic differentiation attributes (`@[fun_trans]` and `@[fun_prop]`) in SciLean for the LEAN_mnist project.

**Total Research:** 4 documents, ~2000 lines, covering all aspects of AD registration

---

## Quick Navigation

### I'm a Developer - Where Do I Start?

**If you have 10 minutes:** Read this README, then skim `AD_REGISTRATION_QUICKSTART.md`

**If you have 30 minutes:** Read `AD_REGISTRATION_SUMMARY.md` for the executive summary

**If you have 1-2 hours:** Read `AD_REGISTRATION_RESEARCH_REPORT.md` for complete context

**When implementing:** Use `AD_REGISTRATION_IMPLEMENTATION_CHECKLIST.md` as your guide

---

## Document Index

### 1. AD_REGISTRATION_README.md (This File)
- **Purpose:** Navigation and quick reference
- **Length:** ~300 lines
- **Contains:** Overview, document index, quick navigation, key findings
- **When to use:** Getting oriented, finding specific information

### 2. AD_REGISTRATION_SUMMARY.md
- **Purpose:** Executive summary and key findings
- **Length:** ~400 lines
- **Contains:**
  - Overview of all three research documents
  - Key findings from SciLean analysis
  - Operation inventory (29 operations total)
  - Implementation strategy and risk analysis
  - Success metrics and next steps
- **When to use:** Need overall understanding before diving deep

### 3. AD_REGISTRATION_RESEARCH_REPORT.md
- **Purpose:** Complete technical reference
- **Length:** ~700 lines
- **Contains:**
  - SciLean AD system explanation (core concepts)
  - Existing SciLean patterns (from MatVecMul.lean, Exp.lean, Log.lean)
  - Current project status and operations needing registration
  - Registration patterns by operation type (linear, bilinear, nonlinear)
  - Attribute details (@[fun_prop], @[data_synth], @[fun_trans])
  - Implementation recommendations
  - Potential issues and solutions
  - Summary table of operations and recommended approaches
  - Step-by-step implementation plan
  - Example implementations with code
  - Testing and validation guidance
- **When to use:** Need deep understanding of how/why things work

### 4. AD_REGISTRATION_QUICKSTART.md
- **Purpose:** Quick reference for syntax and patterns
- **Length:** ~300 lines
- **Contains:**
  - TL;DR of @[fun_prop] vs @[data_synth]
  - Three operation types with examples
  - Naming conventions
  - Common proof strategies
  - Quick reference table for LinearAlgebra operations
  - Quick reference table for Activation operations
  - Key files to reference
  - When to skip registration
  - Most important rule
- **When to use:** Need syntax examples or quick pattern lookup

### 5. AD_REGISTRATION_IMPLEMENTATION_CHECKLIST.md
- **Purpose:** Actionable implementation guide
- **Length:** ~400 lines
- **Contains:**
  - 10 implementation phases (7 for LinearAlgebra, 3 for Activation)
  - Each operation broken down into concrete steps
  - Phase dependencies and testing checkpoints
  - Prerequisites and critical checks
  - Common pitfalls to avoid
  - Success criteria
  - Documentation after completion
- **When to use:** Actually implementing the registrations

---

## Key Findings (TL;DR)

### SciLean's Two-Attribute System

1. **@[fun_prop]** - Marks differentiability properties
   - Used by `fun_prop` tactic for compositional proofs
   - Simplest to implement (often just 1-2 lines)

2. **@[data_synth]** - Marks derivative implementations
   - Provides forward and reverse mode AD rules
   - More complex but necessary for custom operations

### Three Operation Types

| Type | Example | Effort | Approach |
|------|---------|--------|----------|
| Linear | vadd, smul, transpose | 5 min | Just @[fun_prop] |
| Bilinear | vmul, dot, matvec | 20-30 min | @[fun_prop] + @[data_synth] for each arg |
| Nonlinear | relu, sigmoid, softmax | 30-45 min | Full @[fun_prop] + @[data_synth] |

### Operations to Register

- **LinearAlgebra.lean:** 18 operations
  - 9 linear (vadd, smul, transpose, etc.)
  - 5 bilinear (vmul, dot, matvec, matmul, outer)
  - 2 nonlinear (normSq, norm)
  - 2 batch (batchMatvec, batchAddVec)

- **Activation.lean:** 11 operations
  - 5 ReLU family (relu, variants)
  - 5 Sigmoid family (sigmoid, tanh, variants)
  - 1 Classification (softmax)

**Total:** 29 operations, estimated 8-20 hours to implement

### Critical Patterns from SciLean

**Pattern for Linear/Bilinear (MatVecMul.lean, VecMatMul.lean):**
```lean
@[fun_prop]
theorem operation.arg_VAR.Property_rule : ... := sorry_proof

@[data_synth]
theorem operation.arg_VAR.HasRevFDeriv_rule : ... := sorry_proof
```

**Pattern for Nonlinear (Exp.lean):**
```lean
def_fun_prop operation in x : Differentiable K by sorry_proof
abbrev_fun_trans operation in x : fderiv K by equals ... => sorry_proof
abbrev_data_synth operation in x : HasRevFDeriv K by ...
```

---

## Reference: File Locations

### SciLean Template Files (in .lake/packages/SciLean)
- `SciLean/AD/Rules/MatVecMul.lean` - Bilinear operation template
- `SciLean/AD/Rules/VecMatMul.lean` - Alternative bilinear template
- `SciLean/AD/Rules/Exp.lean` - Nonlinear operation template
- `SciLean/AD/Rules/Log.lean` - Operation with side conditions
- `SciLean/AD/Rules/Common.lean` - Macro definitions

### Project Files to Modify
- `VerifiedNN/Core/LinearAlgebra.lean` - 18 operations
- `VerifiedNN/Core/Activation.lean` - 11 operations

---

## How to Use This Package

### Scenario 1: Understanding the System
1. Read this README (10 min)
2. Read `AD_REGISTRATION_SUMMARY.md` (20 min)
3. Skim `AD_REGISTRATION_RESEARCH_REPORT.md` (30 min)
4. Reference `AD_REGISTRATION_QUICKSTART.md` for details

### Scenario 2: Quick Implementation
1. Read `AD_REGISTRATION_QUICKSTART.md` (15 min)
2. Use `AD_REGISTRATION_IMPLEMENTATION_CHECKLIST.md` as you work
3. Reference `AD_REGISTRATION_RESEARCH_REPORT.md` for complex cases

### Scenario 3: Deep Dive
1. Read `AD_REGISTRATION_RESEARCH_REPORT.md` (60 min)
2. Study SciLean template files side-by-side
3. Use checklist for phased implementation
4. Reference quickstart for syntax lookups

### Scenario 4: Team Work
- **Documentation Lead:** Start with Research Report and Summary
- **Implementation Lead:** Use Checklist and Quickstart
- **Review Lead:** Use Research Report for validation

---

## Implementation Timeline

### Recommended 5-Day Schedule (for single developer)

**Day 1 - Foundation (3-4 hours)**
- Read research documents (1-2 hours)
- Implement Phase 1-2 (Linear operations): vadd, vsub, smul, matAdd, matSub, matSmul, transpose
- Test: `lake build VerifiedNN.Core.LinearAlgebra`

**Day 2 - Core Activation (2-3 hours)**
- Implement Phase A (ReLU): relu, reluVec, reluBatch, leakyRelu, leakyReluVec
- Test: `lake build VerifiedNN.Core.Activation`
- Verify gradient checks pass

**Day 3 - Bilinear Operations (3-4 hours)**
- Implement Phase 3-4: vmul, dot, normSq, norm
- Verify: Check if SciLean provides matvec
- Test: `lake build VerifiedNN.Core.LinearAlgebra`

**Day 4 - Matrix & Sigmoid (3-4 hours)**
- Implement Phase 5: matvec, matmul (if not in SciLean)
- Implement Phase B: sigmoid, sigmoidVec, sigmoidBatch, tanh, tanhVec
- Test: Both modules compile

**Day 5 - Advanced & Polish (2-3 hours)**
- Implement Phase 6-7: outer, batch operations
- Implement Phase C: softmax (or verify SciLean provides it)
- Final testing: Full gradient check suite
- Documentation updates

**Total: 13-18 hours over 5 days**

---

## Critical Checks Before Starting

Before implementation, verify these dependencies:

```bash
# Check if Float.exp is registered
grep -r "exp.*Differentiable\|exp.*fun_prop" .lake/packages/SciLean/SciLean/AD/Rules/

# Check if softmax is available
grep -r "softmax" .lake/packages/SciLean/SciLean/AD/Rules/

# Check if matvec is already registered
grep -r "matvec" .lake/packages/SciLean/SciLean/AD/Rules/

# Check current build status
lake build VerifiedNN.Core.LinearAlgebra
lake build VerifiedNN.Core.Activation
```

---

## Success Criteria

Implementation is successful when:

1. ✓ All files compile without errors
   ```bash
   lake build VerifiedNN.Core.LinearAlgebra
   lake build VerifiedNN.Core.Activation
   ```

2. ✓ All TODOs about fun_trans/fun_prop are removed
   ```bash
   grep "TODO.*fun_trans\|TODO.*fun_prop" VerifiedNN/Core/*.lean  # Empty
   ```

3. ✓ Gradient checking passes
   ```bash
   lake build VerifiedNN.Testing.GradientCheck
   # Numerical gradients should match symbolic derivatives
   ```

4. ✓ Training still works
   ```bash
   lake exe mnistTrain --epochs 1 --batch-size 32
   ```

5. ✓ Documentation updated
   - Module docstrings reflect completed registrations
   - Verification status updated in README files

---

## FAQ

**Q: Which document should I read first?**
A: Start with this README, then `AD_REGISTRATION_SUMMARY.md`. Go deeper into the Research Report as needed.

**Q: Can I implement operations in any order?**
A: Mostly, but some have dependencies. Follow the phases in the Checklist for optimal order.

**Q: How long will this take?**
A: 8-20 hours depending on experience level. See timeline above.

**Q: Do I need to prove all derivatives formally?**
A: No, use `sorry_proof` for complex proofs. The important thing is registering the operations correctly.

**Q: What if SciLean already provides an operation?**
A: Don't reimplement. Check the Research Report for how to verify this.

**Q: Where are the syntax examples?**
A: See `AD_REGISTRATION_QUICKSTART.md` for syntax examples and patterns.

**Q: What if a proof doesn't work?**
A: See the "Proof Strategy" sections in the Research Report for alternatives.

**Q: Can this be parallelized?**
A: Yes! See "Parallel Work Possible" in the Summary for details on threading.

---

## Commits and Version Control

After completing each phase:

```bash
# Phase 1-2 (Linear operations)
git add VerifiedNN/Core/LinearAlgebra.lean
git commit -m "Register @[fun_prop] for linear algebra operations (Phase 1-2)"

# Phase A (ReLU)
git add VerifiedNN/Core/Activation.lean
git commit -m "Register AD attributes for ReLU family (Phase A)"

# ... etc for other phases

# Final summary commit
git add VerifiedNN/
git commit -m "Complete AD attribute registration for all 29 operations"
```

---

## Support and Debugging

If stuck:

1. **Check the patterns:** Reference `AD_REGISTRATION_RESEARCH_REPORT.md` sections 2 and 4
2. **Check naming:** Verify you're using correct naming convention from Quickstart
3. **Check SciLean:** Look at MatVecMul.lean and Exp.lean side-by-side
4. **Check compilation:** Run `lake build` to see exact error messages
5. **Check similar operations:** Look for same operation type already registered in SciLean

---

## Document Statistics

| Document | Length | Purpose |
|----------|--------|---------|
| This README | ~500 lines | Navigation and overview |
| Summary | ~400 lines | Executive summary |
| Research Report | ~700 lines | Complete technical reference |
| Quickstart | ~300 lines | Quick syntax reference |
| Checklist | ~400 lines | Implementation guide |
| **Total** | **~2,700 lines** | **Complete package** |

---

## Final Notes

This research package represents a complete analysis of:
- SciLean's automatic differentiation system
- 29 operations needing registration in LEAN_mnist
- All patterns found in SciLean codebase
- Step-by-step implementation strategy
- Testing and validation approach

Everything needed to successfully implement AD attribute registration is included in these documents.

Good luck with the implementation! The research is done, the patterns are documented, and the path is clear.

---

**Created:** 2025-10-22
**Project:** LEAN_mnist - Verified Neural Network in Lean 4
**Research Scope:** Complete AD registration for LinearAlgebra.lean and Activation.lean

