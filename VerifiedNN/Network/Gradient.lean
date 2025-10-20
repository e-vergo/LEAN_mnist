/-
# Network Gradient Computation

Gradient computation using SciLean's automatic differentiation.
-/

import VerifiedNN.Network.Architecture
import SciLean

namespace VerifiedNN.Network.Gradient

open VerifiedNN.Core
open VerifiedNN.Network
open SciLean

/-- Total number of parameters in the network -/
def nParams : Nat := 784 * 128 + 128 + 128 * 10 + 10

/-- Flatten network parameters into a single vector -/
def flattenParams (net : MLPArchitecture) : Vector nParams :=
  sorry -- TODO: implement parameter flattening

/-- Unflatten parameter vector back to network structure -/
def unflattenParams (params : Vector nParams) : MLPArchitecture :=
  sorry -- TODO: implement parameter unflattening

/-- Compute gradient of loss with respect to network parameters -/
def networkGradient (params : Vector nParams)
    (input : Vector 784) (target : Nat) : Vector nParams :=
  sorry -- TODO: implement using SciLean's ∇ operator

end VerifiedNN.Network.Gradient
