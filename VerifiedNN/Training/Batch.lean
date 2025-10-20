/-
# Mini-Batch Handling

Data batching utilities for training.
-/

import VerifiedNN.Core.DataTypes

namespace VerifiedNN.Training.Batch

open VerifiedNN.Core

/-- Create mini-batches from training data -/
def createBatches {n : Nat}
    (data : Array (Vector 784 × Nat))
    (batchSize : Nat) : Array (Array (Vector 784 × Nat)) :=
  sorry -- TODO: implement batching logic

/-- Shuffle data array -/
def shuffleData {α : Type} (data : Array α) : IO (Array α) :=
  sorry -- TODO: implement Fisher-Yates shuffle

end VerifiedNN.Training.Batch
