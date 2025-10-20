/-
# Network Architecture

MLP architecture definition and forward pass implementation.
-/

import VerifiedNN.Layer.Dense
import VerifiedNN.Core.Activation

namespace VerifiedNN.Network

open VerifiedNN.Core
open VerifiedNN.Layer
open VerifiedNN.Core.Activation

/-- MLP architecture: 784 -> 128 -> 10 -/
structure MLPArchitecture where
  layer1 : DenseLayer 784 128
  layer2 : DenseLayer 128 10

/-- Forward pass through the network -/
def MLPArchitecture.forward (net : MLPArchitecture) (x : Vector 784) : Vector 10 :=
  sorry -- TODO: implement layer1 -> relu -> layer2 -> softmax

/-- Batched forward pass -/
def MLPArchitecture.forwardBatch {b : Nat} (net : MLPArchitecture) (X : Batch b 784) : Batch b 10 :=
  sorry -- TODO: implement batched forward pass

end VerifiedNN.Network
