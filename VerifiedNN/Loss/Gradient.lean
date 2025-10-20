/-
# Loss Gradient

Analytical gradient computation for cross-entropy loss.
-/

import VerifiedNN.Loss.CrossEntropy
import SciLean

namespace VerifiedNN.Loss.Gradient

open VerifiedNN.Core
open VerifiedNN.Loss
open SciLean

/-- Gradient of cross-entropy loss with respect to predictions -/
def lossGradient {n : Nat} (predictions : Vector n) (target : Nat) : Vector n :=
  sorry -- TODO: implement analytical gradient

end VerifiedNN.Loss.Gradient
