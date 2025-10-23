import VerifiedNN.Data.MNIST
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization

/-! Minimal test to debug where the hang occurs -/

open VerifiedNN.Data.MNIST
open VerifiedNN.Network
open VerifiedNN.Network.Initialization

unsafe def main : IO Unit := do
  IO.println "Test 1: Program started"

  IO.println "Test 2: Loading MNIST data..."
  let trainData ← loadMNISTTrain "data"
  IO.println s!"Test 3: Loaded {trainData.size} samples"

  IO.println "Test 4: Initializing network..."
  let net ← initializeNetworkHe
  IO.println "Test 5: Network initialized"

  IO.println "Test 6: Computing forward pass on first sample..."
  if trainData.size > 0 then
    let (input, _label) := trainData[0]!
    let output := MLPArchitecture.forward net input
    IO.println "Test 7: Forward pass complete"

  IO.println "Test 8: All tests passed!"
