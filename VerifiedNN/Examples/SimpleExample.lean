/-
# Simple Example

Minimal working example: Train a small network on a simple problem.
-/

import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Training.Loop

namespace VerifiedNN.Examples.SimpleExample

open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Training.Loop

/-- Main entry point for simple example -/
def main : IO Unit := do
  IO.println "Simple neural network example"
  -- TODO: implement simple training example
  sorry

end VerifiedNN.Examples.SimpleExample
