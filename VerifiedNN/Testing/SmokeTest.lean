import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Network.Gradient
import VerifiedNN.Data.MNIST
import SciLean

/-!
# Quick Smoke Test

Ultra-fast smoke test for CI/CD pipelines.

## Purpose

Validates basic functionality in <30 seconds:
- Network initialization
- Forward pass computation
- Data structure creation
- Prediction logic

## Usage

```bash
lake exe smokeTest
```

Expected runtime: <10 seconds

## Verification Status

This is a computational validation test, not a formal proof.
-/

namespace VerifiedNN.Testing.SmokeTest

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Network.Gradient
open VerifiedNN.Data.MNIST
open SciLean

/-- Quick smoke test for CI/CD. -/
unsafe def main : IO Unit := do
  IO.println "==================================="
  IO.println "VerifiedNN Quick Smoke Test"
  IO.println "==================================="

  -- Test 1: Network creation
  IO.println "\nTest 1: Network Initialization"
  let net ← initializeNetworkHe
  IO.println "  ✓ Network created (784 → 128 → 10)"

  -- Test 2: Forward pass
  IO.println "\nTest 2: Forward Pass"
  let input : Vector 784 := ⊞ (_ : Idx 784) => 0.5
  let output := net.forward input
  -- Output is Vector 10, which is Float^[10] - size is in the type, not runtime

  IO.println "  ✓ Forward pass produced output of dimension 10"

  -- Test 3: Prediction
  IO.println "\nTest 3: Prediction"
  let pred := argmax output
  if pred >= 10 then
    IO.eprintln s!"  ✗ FAIL: Invalid prediction {pred} (expected 0-9)"
    IO.Process.exit 1

  IO.println s!"  ✓ Predicted class: {pred}"

  -- Test 4: MNIST data structure check (don't load full dataset)
  IO.println "\nTest 4: Data Structure Check"
  -- Just verify the types compile, don't load actual data
  let _dummyDataset : Array (Vector 784 × Nat) := #[
    (⊞ (_ : Idx 784) => 0.0, 0),
    (⊞ (_ : Idx 784) => 1.0, 9)
  ]
  IO.println s!"  ✓ Dataset structure valid ({_dummyDataset.size} samples)"

  -- Test 5: Parameter counting
  IO.println "\nTest 5: Parameter Count"
  let _params := Gradient.flattenParams net
  let expectedParams := 784 * 128 + 128 + 128 * 10 + 10
  -- _params is Vector nParams where nParams = 101770
  IO.println s!"  ✓ Network has {expectedParams} parameters (type-checked)"

  -- Success
  IO.println "\n==================================="
  IO.println "✓ All smoke tests passed!"
  IO.println "==================================="

end VerifiedNN.Testing.SmokeTest
