/-
# Unit Tests

Component-level unit tests using LSpec.
-/

import VerifiedNN.Core.DataTypes
import VerifiedNN.Core.LinearAlgebra
import VerifiedNN.Core.Activation
import VerifiedNN.Layer.Dense
import LSpec

namespace VerifiedNN.Testing.UnitTests

open VerifiedNN.Core
open VerifiedNN.Core.LinearAlgebra
open VerifiedNN.Core.Activation
open VerifiedNN.Layer
open LSpec

-- def layerTests : TestSeq :=
--   test "forward pass dimensions" $ sorry

-- def activationTests : TestSeq :=
--   test "relu properties" $ sorry

-- def linearAlgebraTests : TestSeq :=
--   test "matrix operations" $ sorry

-- def allTests : TestSeq :=
--   layerTests ++ activationTests ++ linearAlgebraTests

end VerifiedNN.Testing.UnitTests
