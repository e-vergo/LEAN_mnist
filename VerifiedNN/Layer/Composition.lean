/-
# Layer Composition

Utilities for composing layers to build neural networks.
-/

import VerifiedNN.Layer.Dense

namespace VerifiedNN.Layer

open VerifiedNN.Core

/-- Compose two dense layers sequentially -/
def stack {d1 d2 d3 : Nat}
    (layer1 : DenseLayer d1 d2)
    (layer2 : DenseLayer d2 d3)
    (x : Vector d1) : Vector d3 :=
  sorry -- TODO: implement layer2.forward (layer1.forward x)

end VerifiedNN.Layer
