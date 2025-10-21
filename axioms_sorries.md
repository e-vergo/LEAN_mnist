# Axiom & Sorry Elimination Progress

**Project:** Verified Neural Network Training in Lean 4
**Goal:** Reduce to ≤8 axioms (convergence theory only), 0 sorries
**Started:** 2025-10-21

---

## Initial Audit Summary

**Total Count:**
- Axioms: 24
- Sorries: ~18 (excluding comments)
- **Target:** 8 axioms, 0 sorries

---

## Axioms to KEEP (8 total)

### Verification/Convergence.lean - Optimization Theory
✅ **Status:** KEEPING (out of scope per spec)

1. `IsSmooth` - Lipschitz gradient definition
2. `IsStronglyConvex` - Strong convexity definition
3. `HasBoundedVariance` - Stochastic gradient variance bound
4. `HasBoundedGradient` - Gradient norm bound
5. `sgd_converges_strongly_convex` - SGD convergence for strongly convex functions
6. `sgd_converges_convex` - SGD convergence for convex functions
7. `sgd_finds_stationary_point_nonconvex` - SGD stationary point theorem
8. `batch_size_reduces_variance` - Batch size variance reduction

**Justification:** True optimization theory requiring stochastic analysis beyond project scope

---

## Elimination Progress by Batch

### Batch 1: Network/Gradient.lean Arithmetic (9 items)
**Status:** 🔴 NOT STARTED

| Item | Type | Description | Status |
|------|------|-------------|--------|
| \`idx_toNat_lt\` | private axiom | Idx bounds lemma | 🔴 TODO |
| Index bound 1 | sorry | i*784+j < nParams | 🔴 TODO |
| Index bound 2 | sorry | 784*128+i < nParams | 🔴 TODO |
| Index bound 3 | sorry | 784*128+128+i*128+j < nParams | 🔴 TODO |
| Index bound 4 | sorry | 784*128+128+128*10+i < nParams | 🔴 TODO |
| If-branch arith | sorry | Combining branch conditions | 🔴 TODO |
| \`unflatten_flatten_id\` | sorry | Round-trip theorem 1 | 🔴 TODO |
| \`flatten_unflatten_id\` | sorry | Round-trip theorem 2 | 🔴 TODO |
| Loop membership (2×) | sorry | For-loop range membership | 🔴 TODO |

### Batch 4: Verification/GradientCorrectness.lean - Basic Calculus (3 items)
**Status:** ✅ COMPLETE

| Item | Type | Description | Status |
|------|------|-------------|--------|
| \`deriv_id'\` | sorry | Derivative of identity is 1 | ✅ DONE (used `deriv_id`) |
| \`relu_gradient_almost_everywhere\` | sorry | ReLU derivative for x ≠ 0 | ✅ DONE (case split + neighborhood analysis) |
| \`sigmoid_gradient_correct\` | sorry | Sigmoid derivative σ(x)(1-σ(x)) | ✅ DONE (quotient rule + chain rule) |

**Summary:** All 3 basic calculus proofs completed using standard mathlib lemmas.
- `deriv_id'`: Direct application of `deriv_id`
- `relu_gradient_almost_everywhere`: Case analysis on x > 0 vs x < 0, using `deriv_congr_nhds` with neighborhood arguments
- `sigmoid_gradient_correct`: Quotient rule with `deriv_div`, chain rule for exponential composition, algebraic simplification with `field_simp` and `ring`

### Batch 2-3, 5-7: See full tracking file for details

---

## Overall Progress

```
Initial: 24 axioms + 18 sorries = 42 items to address
Target:   8 axioms +  0 sorries =  8 items remaining

Progress: 3/34 eliminated (8.8%)
  - Batch 4 complete: 3 sorries eliminated
```

**Recent Updates:**
- **2025-10-21 (Batch 4):** Completed 3 basic calculus proofs in GradientCorrectness.lean using standard mathlib lemmas
- **2025-10-21:** Initial audit and tracking setup

---

**Last Updated:** 2025-10-21 (Batch 4 complete)
