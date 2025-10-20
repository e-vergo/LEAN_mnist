/-
# Layer Properties

Formal properties and theorems about layer operations.
-/

import VerifiedNN.Layer.Dense
import VerifiedNN.Layer.Composition

namespace VerifiedNN.Layer.Properties

open VerifiedNN.Core
open VerifiedNN.Layer

-- Dimension consistency theorems
-- theorem forward_dimension_correct {m n : Nat} :
--   ∀ (layer : DenseLayer n m) (x : Vector n),
--   (layer.forward x).size = m := sorry

-- Linearity before activation
-- theorem forward_linear_before_activation : ... := sorry

end VerifiedNN.Layer.Properties
