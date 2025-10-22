import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Gradient
open VerifiedNN.Network
open VerifiedNN.Network.Gradient

-- Try to prove using ext tactic
example (net : MLPArchitecture) :
  unflattenParams (flattenParams net) = net := by
  ext
  sorry