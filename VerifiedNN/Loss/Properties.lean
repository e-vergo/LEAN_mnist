/-
# Loss Function Properties

Formal properties of the cross-entropy loss function.
-/

import VerifiedNN.Loss.CrossEntropy

namespace VerifiedNN.Loss.Properties

open VerifiedNN.Core
open VerifiedNN.Loss

-- Loss is non-negative
-- theorem loss_nonneg {n : Nat} :
--   ∀ (pred : Vector n) (target : Nat),
--   crossEntropyLoss pred target ≥ 0 := sorry

-- Loss is differentiable
-- theorem loss_differentiable : ... := sorry

end VerifiedNN.Loss.Properties
