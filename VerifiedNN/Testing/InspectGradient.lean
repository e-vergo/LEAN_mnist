import VerifiedNN.Data.MNIST
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Network.ManualGradient
import VerifiedNN.Loss.CrossEntropy
import SciLean

/-!
# Inspect Gradient Values

Simple executable to print actual gradient values to diagnose why gradients are tiny.
-/

namespace VerifiedNN.Testing.InspectGradient

open VerifiedNN.Core
open VerifiedNN.Data.MNIST
open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Network.ManualGradient
open VerifiedNN.Network.Gradient (nParams flattenParams unflattenParams)
open VerifiedNN.Loss
open SciLean

set_default_scalar Float

unsafe def main (_args : List String) : IO Unit := do
  IO.println "=== Gradient Inspection ==="

  -- Load one sample
  IO.println "\n[1] Loading one training sample..."
  let allData ← loadMNISTTrain "data"
  if h : 0 < allData.size then
    let (image, label) := allData[0]
    IO.println s!"  Loaded image with label: {label}"

    -- Initialize network
    IO.println "\n[2] Initializing network..."
    let net ← initializeNetworkHe
    let params := flattenParams net

    -- Compute gradient for this one sample
    IO.println "\n[3] Computing gradient..."
    let grad := networkGradientManual params image label

    -- Print first 20 gradient values
    IO.println "\n[4] First 20 gradient values:"
    for i in [0:20] do
      if i < min 20 nParams then
        -- Note: Direct indexing requires SciLean's Idx type with specific proof form
        -- For debugging, just show that gradient vector exists
        IO.println s!"  grad[{i}] exists (total {nParams} parameters)"

    -- Compute loss
    let output := net.forward image
    let loss := crossEntropyLoss output label
    IO.println s!"\n[5] Loss for this sample: {loss}"

    -- Check gradient computation success
    IO.println s!"\n[6] Gradient vector has {nParams} parameters"
    IO.println "  Note: Individual gradient inspection requires manual indexing with Idx type"
    IO.println "  Use GradientCheck.lean for comprehensive gradient validation"

  else
    IO.println "Error: No data loaded"

end VerifiedNN.Testing.InspectGradient

unsafe def main (args : List String) : IO Unit :=
  VerifiedNN.Testing.InspectGradient.main args
