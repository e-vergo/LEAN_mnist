/-
# Training Metrics

Accuracy computation and evaluation metrics.
-/

import VerifiedNN.Network.Architecture

namespace VerifiedNN.Training.Metrics

open VerifiedNN.Core
open VerifiedNN.Network

/-- Compute classification accuracy on a dataset -/
def computeAccuracy
    (net : MLPArchitecture)
    (testData : Array (Vector 784 × Nat)) : Float :=
  sorry -- TODO: implement accuracy computation

/-- Compute average loss on a dataset -/
def computeAverageLoss
    (net : MLPArchitecture)
    (testData : Array (Vector 784 × Nat)) : Float :=
  sorry -- TODO: implement loss computation

end VerifiedNN.Training.Metrics
