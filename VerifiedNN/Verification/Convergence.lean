/-
# Convergence Properties

Formal statements of convergence theorems for stochastic gradient descent.

This module provides formal specifications of SGD convergence properties on ℝ.
These theorems state the mathematical conditions under which SGD converges,
establishing the theoretical foundation for the training algorithm.

**Module Structure:**
- `Convergence/Axioms.lean`: 8 axiomatized convergence theorems
- `Convergence/Lemmas.lean`: Robbins-Monro learning rate schedule lemmas
- `Convergence.lean`: Re-exports both modules (this file)

**Verification Status:**
- Convergence theorem statements: Complete (8 axioms)
- Learning rate lemmas: 1 proven (one_over_t_plus_one_satisfies_robbins_monro)
- Full proofs: Not required (explicitly out of scope per project spec)

**Scope Note:**
Per the project specification (verified-nn-spec.md Section 5.4), convergence proofs
are explicitly out of scope. This module provides precise mathematical statements
that can be axiomatized or proven in future work.

**Mathematical Context:**
These theorems are on ℝ (real numbers), not Float. They establish the theoretical
soundness of SGD, separate from implementation details or floating-point numerics.

**Axiom Summary:**
1. IsSmooth - L-smoothness (gradient Lipschitz continuity)
2. IsStronglyConvex - μ-strong convexity
3. HasBoundedVariance - Bounded stochastic gradient variance
4. HasBoundedGradient - Bounded gradient norm
5. sgd_converges_strongly_convex - Linear convergence for strongly convex functions
6. sgd_converges_convex - Sublinear convergence for convex functions
7. sgd_finds_stationary_point_nonconvex - Stationary point convergence (neural networks)
8. batch_size_reduces_variance - Variance reduction with larger batches

**Practical Implications for MNIST Training:**

The MNIST MLP (784 → 128 → 10 with ReLU) has a non-convex loss landscape.
Theoretical guarantees from axiom 7 (sgd_finds_stationary_point_nonconvex):
- SGD finds stationary points (∇L ≈ 0), not necessarily global optima
- Gradient norm decreases to zero as training progresses
- Final accuracy depends on which local minimum is found

Hyperparameter guidance from theory:
1. **Learning rate:** α ∈ [0.001, 0.1] for MNIST
   - Too large: oscillation/divergence
   - Too small: slow convergence
   - Use decay for better final accuracy

2. **Batch size:** b ∈ [16, 128] for MNIST
   - Larger batches: more stable, lower variance
   - Smaller batches: help escape poor local minima
   - Variance reduction: Var[∇_batch] = Var[∇_single] / b (axiom 8)

3. **Number of epochs:** 10-50 epochs typical for MNIST
   - Monitor validation loss (early stopping)
   - Loss decreases on average (not monotonically)

**References:**
- Bottou et al. (2018): "Optimization methods for large-scale machine learning"
- Allen-Zhu et al. (2018): "A convergence theory for deep learning"
- Robbins & Monro (1951): "A stochastic approximation method"

See Convergence/Axioms.lean for detailed references and mathematical statements.
-/

-- Re-export convergence axioms
import VerifiedNN.Verification.Convergence.Axioms

-- Re-export convergence lemmas (Robbins-Monro schedules)
import VerifiedNN.Verification.Convergence.Lemmas

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
