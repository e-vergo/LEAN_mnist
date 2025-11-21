# File Review: DataTypes.lean

## Summary
Defines core type aliases (Vector, Matrix, Batch) and approximate equality functions for testing. Minimal, clean file serving as foundation for entire project. Zero errors/warnings.

## Findings

### Orphaned Code
**NONE** - All definitions are fundamental and heavily used:
- `Vector`, `Matrix`, `Batch`: Used in every module (59+ files reference these)
- `epsilon`: Default tolerance used throughout testing infrastructure
- `approxEq`, `vectorApproxEq`, `matrixApproxEq`: Used extensively in Testing/ modules

### Axioms (Total: 0)
No axioms in this file.

### Sorries (Total: 0)
No sorries - file is complete.

### Code Correctness Issues
**NONE** - File is clean:
- ✅ All docstrings accurate and match implementations
- ✅ Type aliases correctly map to SciLean DataArrayN types
- ✅ Approximate equality functions correctly implement average absolute difference
- ✅ Zero LSP diagnostics

### Hacks & Deviations

**1. Lines 146-147: Average vs Maximum Metric**
- **Issue:** `vectorApproxEq` uses average absolute difference instead of maximum
- **Severity:** Moderate
- **Impact:** May pass tests with outliers that would fail max-based metric
- **Justification:** "Conservative approximation pending efficient max reduction in SciLean"
- **TODO:** Line 146-147 states this should be replaced when SciLean provides efficient reduction ops
- **Similar issue:** Lines 173-174 for `matrixApproxEq`

**2. Lines 38-40: Float vs ℝ Gap**
- **Issue:** Types use Float (IEEE 754) but verification is on ℝ (real numbers)
- **Severity:** Minor (acknowledged design decision)
- **Documentation:** Clearly stated: "The correspondence between Float operations and their ℝ counterparts is assumed but not formally proven"
- **Status:** Accepted limitation, properly documented

**3. Line 103: Epsilon Value (1e-7)**
- **Issue:** Fixed tolerance may not be appropriate for all scales
- **Severity:** Minor
- **Discussion:** 1e-7 is standard for Float32 precision, but no discussion of relative vs absolute error
- **Impact:** May give false negatives/positives for very large/small values
- **Workaround:** All approxEq functions accept custom `eps` parameter

## Statistics
- **Definitions:** 7 total
  - Type aliases: 3 (Vector, Matrix, Batch)
  - Constants: 1 (epsilon)
  - Functions: 3 (approxEq, vectorApproxEq, matrixApproxEq)
- **Theorems:** 0 (no formal proofs)
- **Unused definitions:** 0 (all heavily used across project)
- **Axioms:** 0
- **Sorries:** 0
- **Lines of code:** 183
- **TODOs:** 2 (both related to replacing average with max when SciLean supports it)
- **Build status:** ✅ Zero errors, zero warnings

## Recommendations

### High Priority
None - file serves its purpose well.

### Medium Priority
1. **Document metric choice tradeoffs:** Add note explaining why average metric is acceptable for current use cases despite being weaker than max metric
2. **Add relative error option:** Consider adding `relativeApproxEq` variants for comparing values at different scales

### Low Priority
1. **Monitor SciLean progress:** Track when efficient max reduction becomes available and update metric
2. **Consider epsilon tiers:** Different default tolerances for gradient checking (1e-5) vs loss checking (1e-7)
3. **Add unit tests:** While used extensively, direct unit tests for approxEq edge cases would improve confidence

## File Health Score: 92/100

**Deductions:**
- -5 for average vs max metric (acknowledged limitation with TODO)
- -3 for lack of relative error comparison option

**Strengths:**
- Zero errors/warnings
- Minimal, focused scope
- Excellent documentation
- All definitions heavily used
- Float/ℝ gap clearly acknowledged
- Customizable epsilon parameter
