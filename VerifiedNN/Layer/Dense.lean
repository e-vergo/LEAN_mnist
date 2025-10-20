/-
# Dense Layer

Dense (fully-connected) layer implementation with type-safe dimensions.
-/

import VerifiedNN.Core.DataTypes
import VerifiedNN.Core.LinearAlgebra
import VerifiedNN.Core.Activation

namespace VerifiedNN.Layer

open VerifiedNN.Core
open VerifiedNN.Core.LinearAlgebra
open VerifiedNN.Core.Activation

/-- Dense layer structure with weights and biases -/
structure DenseLayer (inDim outDim : Nat) where
  weights : Matrix outDim inDim
  bias : Vector outDim

/-- Forward pass through a dense layer -/
def DenseLayer.forward {m n : Nat} (layer : DenseLayer n m) (x : Vector n) : Vector m :=
  sorry -- TODO: implement Wx + b

/-- Batched forward pass -/
def DenseLayer.forwardBatch {b m n : Nat} (layer : DenseLayer n m) (X : Batch b n) : Batch b m :=
  sorry -- TODO: implement batched forward pass

end VerifiedNN.Layer
