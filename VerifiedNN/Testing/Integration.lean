/-
# Integration Tests

End-to-end integration tests for the training pipeline.
-/

import VerifiedNN.Network.Architecture
import VerifiedNN.Training.Loop
import LSpec

namespace VerifiedNN.Testing.Integration

open VerifiedNN.Network
open VerifiedNN.Training.Loop
open LSpec

-- def trainingPipelineTest : TestSeq :=
--   test "train on tiny dataset" $ sorry

-- def overfittingTest : TestSeq :=
--   test "overfit on small dataset" $ sorry

-- def allIntegrationTests : TestSeq :=
--   trainingPipelineTest ++ overfittingTest

end VerifiedNN.Testing.Integration
