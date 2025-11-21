import VerifiedNN.Core.DataTypes
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Training.Metrics
import VerifiedNN.Data.MNIST
import SciLean

/-!
# Minimal Performance Test

Quick test to measure per-sample forward pass timing.
Uses only 10 samples to identify performance bottlenecks.
-/

namespace VerifiedNN.Testing.PerformanceTest

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Training
open VerifiedNN.Data
open SciLean

set_default_scalar Float

unsafe def main : IO UInt32 := do
  IO.println "=========================================="
  IO.println "Minimal Performance Test (10 samples)"
  IO.println "=========================================="
  IO.println ""

  -- Load minimal data
  IO.println "Loading data..."
  let dataDir : System.FilePath := "data"
  let trainDataFull ← MNIST.loadMNISTTrain dataDir
  let testDataFull ← MNIST.loadMNISTTest dataDir

  let trainData := trainDataFull.toSubarray 0 (min 10 trainDataFull.size) |>.toArray
  let testData := testDataFull.toSubarray 0 (min 10 testDataFull.size) |>.toArray

  IO.println s!"✓ Loaded {trainData.size} train samples"
  IO.println s!"✓ Loaded {testData.size} test samples"
  IO.println ""

  -- Initialize network
  IO.println "Initializing network..."
  let net ← initializeNetworkHe
  IO.println "✓ Network initialized"
  IO.println ""

  -- Test single forward pass timing
  IO.println "Testing single forward pass..."
  let (input, _) := trainData[0]!

  let startSingle ← IO.monoMsNow
  let _ := net.forward input
  let endSingle ← IO.monoMsNow
  let singleTime := (endSingle - startSingle).toFloat

  IO.println s!"  Single forward pass: {singleTime}ms"
  IO.println ""

  -- Test 10 forward passes
  IO.println "Testing 10 forward passes..."
  let start10 ← IO.monoMsNow
  for i in [0:trainData.size] do
    let (input, _) := trainData[i]!
    let _ := net.forward input
    pure ()
  let end10 ← IO.monoMsNow
  let total10 := (end10 - start10).toFloat
  let avg10 := total10 / 10.0

  IO.println s!"  Total time: {total10}ms"
  IO.println s!"  Average per sample: {avg10}ms"
  IO.println ""

  -- Test accuracy computation
  IO.println "Testing accuracy computation on 10 samples..."
  (← IO.getStdout).flush

  let startAcc ← IO.monoMsNow
  let acc := Metrics.computeAccuracy net trainData
  let endAcc ← IO.monoMsNow
  let accTime := (endAcc - startAcc).toFloat

  IO.println s!"  Accuracy: {acc * 100.0}%"
  IO.println s!"  Time: {accTime}ms"
  IO.println s!"  Average per sample: {accTime / 10.0}ms"
  IO.println ""

  -- Extrapolate to larger datasets
  IO.println "Extrapolated timing estimates:"
  IO.println s!"  100 samples: {(avg10 * 100.0) / 1000.0}s"
  IO.println s!"  200 samples: {(avg10 * 200.0) / 1000.0}s"
  IO.println s!"  500 samples: {(avg10 * 500.0) / 1000.0}s"
  IO.println s!"  5000 samples: {(avg10 * 5000.0) / 1000.0}s"
  IO.println ""

  IO.println "=========================================="
  IO.println "Performance test complete!"
  IO.println "=========================================="

  return 0

end VerifiedNN.Testing.PerformanceTest

unsafe def main : IO UInt32 := VerifiedNN.Testing.PerformanceTest.main
