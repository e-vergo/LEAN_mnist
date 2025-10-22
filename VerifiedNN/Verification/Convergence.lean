import VerifiedNN.Verification.Convergence.Axioms
import VerifiedNN.Verification.Convergence.Lemmas

/-!
# Convergence Properties

Formal statements of convergence theorems for stochastic gradient descent.

This module provides formal specifications of SGD convergence properties on ℝ, establishing
the theoretical foundation for the training algorithm. While convergence proofs are out of
scope (per project specification), precise mathematical statements are provided for
theoretical completeness and potential future formalization.

## Module Structure

- `Convergence/Axioms.lean`: 8 axiomatized convergence theorems with detailed justifications
- `Convergence/Lemmas.lean`: Robbins-Monro learning rate schedule lemmas (1 proven)
- `Convergence.lean`: Re-exports and helper definitions (this file)

## Main Axioms

1. `IsSmooth`: L-smoothness (gradient Lipschitz continuity)
2. `IsStronglyConvex`: μ-strong convexity
3. `HasBoundedVariance`: Bounded stochastic gradient variance
4. `HasBoundedGradient`: Bounded gradient norm
5. `sgd_converges_strongly_convex`: Linear convergence (strongly convex case)
6. `sgd_converges_convex`: Sublinear convergence (convex case)
7. `sgd_finds_stationary_point_nonconvex`: **PRIMARY FOR MNIST** - Stationary point convergence
8. `batch_size_reduces_variance`: Variance reduction via larger batches

## Verification Status

**Axiomatized:** 8 convergence theorems (all in Axioms.lean)
- Explicitly out of scope per verified-nn-spec.md Section 5.4
- Well-established results from optimization literature
- Precisely stated for theoretical completeness

**Proven:** 1 lemma in Lemmas.lean
- `one_over_t_plus_one_satisfies_robbins_monro`: α_t = 1/(t+1) learning rate schedule

**Future Work:** Convergence proofs would be a separate major project (6-12 months estimated)

## Applicability to MNIST MLP

**Network:** 784 → 128 → 10 with ReLU activation (non-convex loss landscape)

**Primary Theorem:** Axiom 7 (`sgd_finds_stationary_point_nonconvex`)
- Applies to neural networks (unlike Axioms 5-6 which require convexity)
- Guarantees: SGD finds stationary points where ∇L ≈ 0
- Caveat: May be local minima, saddle points, or global minima
- Empirically: Over-parameterized networks have benign landscapes (many good local minima)

**Hyperparameter Guidance:**

1. **Learning rate:** α ∈ [0.001, 0.1] typical for MNIST
   - Theory requires: α < 1/L (smoothness constant)
   - Too large: oscillation/divergence
   - Too small: slow convergence
   - Use diminishing schedule (e.g., 1/(t+1)) for provable convergence

2. **Batch size:** b ∈ [16, 128] common for MNIST
   - Variance reduction: Var[∇_batch] = Var[∇_single] / b (Axiom 8)
   - Larger batches: more stable, better final accuracy
   - Smaller batches: faster iterations, help escape poor local minima

3. **Training duration:** 10-50 epochs typical
   - Convergence rate: O(1/T) for gradient norm (Axiom 7)
   - Monitor validation loss for early stopping

## Mathematical Context

All theorems are stated on ℝ (real numbers), not Float. They establish theoretical
soundness of SGD independent of floating-point implementation details.

## Implementation Notes

- Convergence axioms provide theoretical justification, not implementation requirements
- Numerical validation via loss curves and gradient norms complements formal theory
- Float→ℝ gap acknowledged: convergence guarantees assume exact arithmetic

## References

- Bottou, L., Curtis, F. E., & Nocedal, J. (2018). "Optimization methods for large-scale machine learning." SIAM Review.
- Allen-Zhu, Z., Li, Y., & Song, Z. (2018). "A convergence theory for deep learning via over-parameterization." arXiv:1811.03962.
- Robbins, H., & Monro, S. (1951). "A stochastic approximation method." Ann. Math. Stat.

See Convergence/Axioms.lean for detailed mathematical statements and complete references.
-/

-- Re-export key definitions for convenience
namespace VerifiedNN.Verification.Convergence

-- Helper definitions (from Axioms.lean, re-exported for backward compatibility)

/-- Helper: A loss function achieves its minimum at a point θ*.

This definition captures the notion of an optimal point.
-/
def IsMinimizer {n : ℕ} (f : (Fin n → ℝ) → ℝ) (θ_opt : (Fin n → ℝ)) : Prop :=
  ∀ θ, f θ_opt ≤ f θ

/-- Helper: The optimality gap at a point θ.

Measures how far the loss at θ is from the optimal loss.
-/
def OptimalityGap {n : ℕ} (f : (Fin n → ℝ) → ℝ) (θ θ_opt : (Fin n → ℝ)) : ℝ :=
  f θ - f θ_opt

/-- Helper: A function is convex (not necessarily strongly convex).

This is a weaker condition than strong convexity (μ = 0).
-/
def IsConvex {n : ℕ} (f : (Fin n → ℝ) → ℝ) : Prop :=
  ConvexOn ℝ Set.univ f

/-- Constant learning rate conditions for convergence.

For a constant learning rate to ensure convergence, it must satisfy:
  0 < α < 2/L (for smooth functions)

Smaller learning rates converge more slowly but more stably.
-/
def IsValidConstantLearningRate {n : ℕ} (f : (Fin n → ℝ) → ℝ) (L : ℝ) (α : ℝ) : Prop :=
  IsSmooth f L ∧ 0 < α ∧ α < 2 / L

end VerifiedNN.Verification.Convergence
