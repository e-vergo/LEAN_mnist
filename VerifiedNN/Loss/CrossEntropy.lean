/-
# Cross-Entropy Loss

Cross-entropy loss function with numerical stability.
-/

import VerifiedNN.Core.DataTypes

namespace VerifiedNN.Loss

open VerifiedNN.Core

/-- Cross-entropy loss for a single prediction -/
def crossEntropyLoss {n : Nat} (predictions : Vector n) (target : Nat) : Float :=
  sorry -- TODO: implement with log-sum-exp trick

/-- Batched cross-entropy loss (average over mini-batch) -/
def batchCrossEntropyLoss {b n : Nat} (predictions : Batch b n) (targets : Array Nat) : Float :=
  sorry -- TODO: implement batched loss

end VerifiedNN.Loss
