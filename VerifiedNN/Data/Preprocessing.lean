/-
# Data Preprocessing

Normalization and data augmentation utilities.
-/

import VerifiedNN.Core.DataTypes

namespace VerifiedNN.Data.Preprocessing

open VerifiedNN.Core

/-- Normalize pixel values from [0, 255] to [0, 1] -/
def normalizePixels {n : Nat} (image : Vector n) : Vector n :=
  sorry -- TODO: implement normalization

/-- Flatten 28x28 image to 784-dimensional vector -/
def flattenImage (image : Array (Array Float)) : Vector 784 :=
  sorry -- TODO: implement flattening

end VerifiedNN.Data.Preprocessing
