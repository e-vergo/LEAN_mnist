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
      if h : i < nParams then
        IO.println s!"  grad[{i}] = {grad[⟨i, h⟩]}"

    -- Compute loss
    let output := net.forward image
    let loss := crossEntropyLoss output label
    IO.println s!"\n[5] Loss for this sample: {loss}"

    -- Check if gradients are all zero
    let mut allZero := true
    for i in [0:min 1000 nParams] do
      if h : i < nParams then
        if grad[⟨i, h⟩] != 0.0 then
          allZero := false
          break

    if allZero then
      IO.println "\n  WARNING: First 1000 gradients are all zero!"
    else
      IO.println "\n  Good: Gradients are non-zero"

  else
    IO.println "Error: No data loaded"

end VerifiedNN.Testing.InspectGradient

unsafe def main (args : List String) : IO Unit :=
  VerifiedNN.Testing.InspectGradient.main args
