/-
# Training Loop

Main training loop implementation.
-/

import VerifiedNN.Network.Architecture
import VerifiedNN.Training.Batch
import VerifiedNN.Training.Metrics

namespace VerifiedNN.Training.Loop

open VerifiedNN.Core
open VerifiedNN.Network

/-- Train network for multiple epochs -/
partial def trainEpochs
    (net : MLPArchitecture)
    (trainData : Array (Vector 784 × Nat))
    (epochs : Nat)
    (batchSize : Nat)
    (learningRate : Float) : IO MLPArchitecture :=
  sorry -- TODO: implement training loop

end VerifiedNN.Training.Loop
