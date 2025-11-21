import VerifiedNN.Data.MNIST
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Network.ManualGradient
import VerifiedNN.Network.Gradient
import VerifiedNN.Optimizer.SGD
import VerifiedNN.Loss.CrossEntropy
import SciLean

/-!
# Debug Training Smoke Test

Minimal training reproducer that trains on 100 samples for 10 steps to quickly
isolate the training bug. This test runs in under 30 seconds to enable fast
debugging iteration.

## Purpose

Previous full training tests showed loss increasing instead of decreasing.
This minimal test:
- Uses only 100 samples (vs 60,000)
- Runs only 10 training steps
- Prints detailed diagnostics at each step
- Completes in <30 seconds for fast iteration

## Expected Behavior

**If training works:**
```
Step 1: Loss: 2.15, Gradient norm: 0.045
Step 5: Loss: 1.82, Gradient norm: 0.038
Step 10: Loss: 1.54, Gradient norm: 0.032
✓ SUCCESS: Loss decreased
```

**If bug exists:**
```
Step 1: Loss: 2.35, Gradient norm: 0.001 (too small!)
Step 10: Loss: 2.36, Gradient norm: 0.0009
✗ FAILURE: Loss increased
```

## Key Diagnostics

For each training step, prints:
- Current loss value
- Gradient norm (should be O(0.01-0.1) for typical networks)
- First 5 gradient values (check for zeros or NaNs)
- First 5 parameters before/after update (verify they change)
- Whether parameters actually changed

## Usage

```bash
lake build VerifiedNN.Testing.DebugTraining
lake exe debugTraining
```

## Implementation Notes

- Uses manual backpropagation (computable)
- Batches of 10 samples each
- Learning rate: 0.01 (standard for MNIST)
- Averages gradients over each batch before updating
- Network: 784 → 128 (ReLU) → 10 (same as main training)

## References

- Main training test: VerifiedNN/Examples/MNISTTrain.lean
- Manual gradients: VerifiedNN/Network/ManualGradient.lean
- SGD optimizer: VerifiedNN/Optimizer/SGD.lean
-/

namespace VerifiedNN.Testing.DebugTraining

open VerifiedNN.Core
open VerifiedNN.Data.MNIST
open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Network.ManualGradient
open VerifiedNN.Network.Gradient
open VerifiedNN.Loss
open SciLean

set_default_scalar Float

/-- Compute squared L2 norm of a vector.

**Parameters:**
- `v`: Input vector

**Returns:** Sum of squared elements: Σᵢ v[i]²
-/
def vectorNormSquared {n : Nat} (v : Vector n) : Float :=
  ∑ i, v[i] * v[i]

/-- Compute L2 norm of a vector.

**Parameters:**
- `v`: Input vector

**Returns:** ‖v‖₂ = sqrt(Σᵢ v[i]²)
-/
def vectorNorm {n : Nat} (v : Vector n) : Float :=
  Float.sqrt (vectorNormSquared v)

/-- Compute statistics for a vector (mean, std).

**Parameters:**
- `v`: Input vector

**Returns:** Tuple of (mean, std)
-/
def vectorStats {n : Nat} (v : Vector n) : Float × Float :=
  if n = 0 then
    (0.0, 0.0)
  else
    let sum := ∑ i, v[i]
    let mean := sum / n.toFloat
    let sumSq := ∑ i, (v[i] - mean) * (v[i] - mean)
    let std := Float.sqrt (sumSq / n.toFloat)
    (mean, std)

/-- Compute average loss over a dataset.

Given flattened parameters, computes the average cross-entropy loss across
all examples in the dataset.

**Parameters:**
- `params`: Flattened parameter vector [101,770]
- `data`: Array of (image, label) pairs

**Returns:** Average loss value

**Implementation:**
For each example:
1. Unflatten params → network structure
2. Forward pass → logits
3. Compute cross-entropy loss
4. Average all losses
-/
def computeAverageLoss (params : Vector nParams) (data : Array (Vector 784 × Nat)) : Float :=
  if data.size == 0 then
    0.0
  else
    let totalLoss := data.foldl (fun sum (image, label) =>
      let net := unflattenParams params
      let logits := net.forward image
      let loss := crossEntropyLoss logits label
      sum + loss
    ) 0.0
    totalLoss / data.size.toFloat

/-- Main debug training executable.

Loads 100 training samples, initializes network, and runs 10 training steps
with detailed diagnostic output.

**Algorithm:**
1. Load 100 samples from MNIST training set
2. Initialize network with He initialization
3. Compute initial loss
4. For 10 steps:
   - Take batch of 10 samples
   - Compute average gradient over batch using manual backprop
   - Apply SGD update
   - Compute new loss
   - Print detailed diagnostics
5. Report final results and success/failure

**Expected Runtime:** <30 seconds

**Unsafe:** Uses IO operations, marked unsafe for interpreter mode
-/
unsafe def main (_args : List String) : IO Unit := do
  IO.println "=== Debug Training Smoke Test ==="

  -- [1/5] Load only 100 training samples
  IO.println "\n[1/5] Loading 100 training samples..."
  let allData ← loadMNISTTrain "data"
  let smallData := allData.toList.take 100 |>.toArray
  IO.println s!"  Loaded {smallData.size} samples"

  -- [2/5] Initialize network
  IO.println "\n[2/5] Initializing network with He initialization..."
  let initialNet ← initializeNetworkHe
  let initialParams := flattenParams initialNet
  IO.println s!"  Parameter vector size: {nParams}"

  IO.println "  Network initialized successfully"

  -- [3/5] Print initial loss
  IO.println "\n[3/5] Computing initial loss..."
  let initialLoss := computeAverageLoss initialParams smallData
  IO.println s!"  Initial average loss: {initialLoss}"

  -- [4/5] Run 10 training steps
  IO.println "\n[4/5] Running 10 training steps..."
  IO.println "  (Each step = 1 batch of 10 samples)"

  let mut params := initialParams
  let learningRate : Float := 0.001  -- Reduced from 0.01 to fix oscillation

  for step in [0:10] do
    -- Take batch of 10 samples
    let batchStart := (step * 10) % smallData.size
    let batchEnd := min (batchStart + 10) smallData.size
    let batch := smallData.toList.drop batchStart |>.take (batchEnd - batchStart) |>.toArray

    -- Compute average gradient over batch
    let mut gradSum : Vector nParams := ⊞ (_ : Idx nParams) => (0.0 : Float)
    for (image, label) in batch do
      let grad := networkGradientManual params image label
      gradSum := ⊞ i => gradSum[i] + grad[i]
    let avgGrad := ⊞ i => gradSum[i] / batch.size.toFloat

    -- SGD update (params' = params - lr * gradient)
    params := ⊞ i => params[i] - learningRate * avgGrad[i]

    -- Compute new loss
    let newLoss := computeAverageLoss params smallData

    -- Print diagnostics
    IO.println s!"\n  Step {step + 1}: Loss = {newLoss}"

  -- [5/5] Final statistics
  IO.println "\n[5/5] Final Results:"
  let finalLoss := computeAverageLoss params smallData
  IO.println s!"  Initial loss: {initialLoss}"
  IO.println s!"  Final loss: {finalLoss}"
  let lossChange := finalLoss - initialLoss
  IO.println s!"  Change: {lossChange}"
  if lossChange < 0.0 then
    IO.println "  ✓ SUCCESS: Loss decreased (training is learning!)"
  else
    IO.println "  ✗ FAILURE: Loss increased (bug confirmed)"

  IO.println "\n=== Debug Training Complete ==="

end VerifiedNN.Testing.DebugTraining

-- Top-level main for Lake executable infrastructure
unsafe def main (args : List String) : IO Unit :=
  VerifiedNN.Testing.DebugTraining.main args
