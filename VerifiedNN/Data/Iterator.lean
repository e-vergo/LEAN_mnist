/-
# Data Iterator

Memory-efficient data iteration for training.
-/

import VerifiedNN.Core.DataTypes

namespace VerifiedNN.Data.Iterator

open VerifiedNN.Core

/-- Data iterator state -/
structure DataIterator where
  data : Array (Vector 784 × Nat)
  currentIdx : Nat
  batchSize : Nat

/-- Get next batch from iterator -/
def DataIterator.nextBatch (iter : DataIterator) :
    Option (Array (Vector 784 × Nat) × DataIterator) :=
  sorry -- TODO: implement batch iteration

end VerifiedNN.Data.Iterator
