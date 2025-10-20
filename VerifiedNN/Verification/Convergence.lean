/-
# Convergence Properties

Formal statements of convergence theorems for SGD.
-/

import VerifiedNN.Optimizer.SGD
import SciLean

namespace VerifiedNN.Verification.Convergence

open VerifiedNN.Core
open VerifiedNN.Optimizer
open SciLean

-- SGD convergence under convexity (statement only)
-- theorem sgd_converges_convex
--   (lipschitz_grad : LipschitzWith L (∇ loss))
--   (bounded_variance : ∀ batch, ‖∇ loss batch - ∇ loss‖ ≤ σ²) :
--   ∃ (N : Nat), ∀ (n ≥ N), ‖∇ loss params[n]‖ < ε := sorry

end VerifiedNN.Verification.Convergence
