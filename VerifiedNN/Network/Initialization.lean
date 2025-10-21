/-
# Weight Initialization

Weight initialization strategies for neural networks.

This module implements Xavier/Glorot and He initialization methods,
which are designed to maintain proper gradient flow during training.
-/

import VerifiedNN.Network.Architecture
import VerifiedNN.Layer.Dense
import VerifiedNN.Core.DataTypes
import SciLean

namespace VerifiedNN.Network.Initialization

open VerifiedNN.Network
open VerifiedNN.Layer
open VerifiedNN.Core
open SciLean

/-- Generate random float in range [min, max) using uniform distribution.

**Implementation Note:** This uses Lean's IO-based random number generation.
For reproducibility in experiments, consider seeding the RNG.
-/
def randomFloat (min max : Float) : IO Float := do
  -- Generate random UInt64 and convert to [0, 1)
  let randU64 ← IO.rand 0 UInt64.size.pred
  let uniform01 := randU64.toFloat / UInt64.size.toFloat
  -- Scale to [min, max)
  return min + uniform01 * (max - min)

/-- Generate random float from standard normal distribution N(0, 1).

**Implementation Note:** Uses Box-Muller transform to convert uniform random
variables to normal distribution:
  If U1, U2 ~ Uniform(0,1), then
  Z = sqrt(-2 * ln(U1)) * cos(2π * U2) ~ N(0, 1)
-/
def randomNormal : IO Float := do
  -- Box-Muller transform
  let u1 ← randomFloat 1e-10 1.0  -- Avoid log(0)
  let u2 ← randomFloat 0.0 1.0
  -- Z = sqrt(-2 * ln(u1)) * cos(2π * u2)
  -- Note: Float.pi is not available, use approximation
  let pi := 3.141592653589793
  let r := Float.sqrt (-2.0 * Float.log u1)
  let theta := 2.0 * pi * u2
  return r * Float.cos theta

/-- Initialize a vector with random values from uniform distribution.

**Parameters:**
- `n`: Dimension of the vector
- `min`: Minimum value (inclusive)
- `max`: Maximum value (exclusive)

**Returns:** Vector of dimension n with random values
-/
def initVectorUniform (n : Nat) (min max : Float) : IO (Vector n) := do
  -- Generate array of random floats
  let mut vals : Array Float := Array.empty
  for _ in [0:n] do
    let val ← randomFloat min max
    vals := vals.push val
  -- Convert to DataArrayN
  return ⊞ (i : Idx n) => vals[i.1.toNat]!

/-- Initialize a vector with zeros.

**Parameters:**
- `n`: Dimension of the vector

**Returns:** Zero vector of dimension n
-/
def initVectorZeros (n : Nat) : IO (Vector n) := do
  -- Create zero-initialized vector
  return ⊞ (_ : Idx n) => 0.0

/-- Initialize a matrix with random values from uniform distribution.

**Parameters:**
- `m`: Number of rows
- `n`: Number of columns
- `min`: Minimum value (inclusive)
- `max`: Maximum value (exclusive)

**Returns:** Matrix of dimension m×n with random values
-/
def initMatrixUniform (m n : Nat) (min max : Float) : IO (Matrix m n) := do
  -- Generate m×n random floats in row-major order
  let mut vals : Array Float := Array.empty
  for _ in [0:m*n] do
    let val ← randomFloat min max
    vals := vals.push val
  -- Convert to DataArrayN matrix (row-major indexing)
  return ⊞ ((i, j) : Idx m × Idx n) => vals[(i.1.toNat * n + j.1.toNat)]!

/-- Initialize a matrix with random values from normal distribution.

**Parameters:**
- `m`: Number of rows
- `n`: Number of columns
- `mean`: Mean of the distribution
- `std`: Standard deviation of the distribution

**Returns:** Matrix of dimension m×n with normally distributed values
-/
def initMatrixNormal (m n : Nat) (mean std : Float) : IO (Matrix m n) := do
  -- Generate m×n normally distributed floats
  let mut vals : Array Float := Array.empty
  for _ in [0:m*n] do
    let z ← randomNormal
    let val := mean + std * z
    vals := vals.push val
  -- Convert to DataArrayN matrix
  return ⊞ ((i, j) : Idx m × Idx n) => vals[(i.1.toNat * n + j.1.toNat)]!

/-- Initialize a dense layer using Xavier/Glorot initialization.

Xavier initialization uses uniform distribution in the range:
  [-√(6/(n_in + n_out)), √(6/(n_in + n_out))]

This helps maintain the variance of activations across layers.

**Parameters:**
- `inDim`: Input dimension
- `outDim`: Output dimension

**Returns:** Initialized dense layer
-/
def initDenseLayerXavier (inDim outDim : Nat) : IO (DenseLayer inDim outDim) := do
  let limit : Float := Float.sqrt (6.0 / (inDim.toFloat + outDim.toFloat))
  let weights ← initMatrixUniform outDim inDim (- limit) limit
  let bias ← initVectorZeros outDim
  return { weights, bias }

/-- Initialize a dense layer using He initialization.

He initialization uses normal distribution with:
  N(0, √(2/n_in))

This is specifically designed for ReLU activation functions.

**Parameters:**
- `inDim`: Input dimension
- `outDim`: Output dimension

**Returns:** Initialized dense layer
-/
def initDenseLayerHe (inDim outDim : Nat) : IO (DenseLayer inDim outDim) := do
  let std := Float.sqrt (2.0 / inDim.toFloat)
  let weights ← initMatrixNormal outDim inDim 0.0 std
  let bias ← initVectorZeros outDim
  return { weights, bias }

/-- Initialize complete MLP network with Xavier/Glorot initialization.

Applies Xavier initialization to all layers in the network.
Xavier is a good general-purpose initialization strategy.

**Returns:** Initialized MLP network (784 -> 128 -> 10)
-/
def initializeNetwork : IO MLPArchitecture := do
  let layer1 ← initDenseLayerXavier 784 128
  let layer2 ← initDenseLayerXavier 128 10
  return { layer1, layer2 }

/-- Initialize complete MLP network with He initialization.

Applies He initialization to all layers in the network.
He initialization is specifically designed for ReLU activations,
making it the preferred choice for this architecture.

**Returns:** Initialized MLP network (784 -> 128 -> 10)
-/
def initializeNetworkHe : IO MLPArchitecture := do
  let layer1 ← initDenseLayerHe 784 128
  let layer2 ← initDenseLayerHe 128 10
  return { layer1, layer2 }

/-- Initialize network with custom scale factor.

Allows manual control over initialization scale, useful for experimentation.

**Parameters:**
- `scale`: Scale factor for weight initialization

**Returns:** Initialized MLP network
-/
def initializeNetworkCustom (scale : Float) : IO MLPArchitecture := do
  let layer1Weights ← initMatrixUniform 128 784 (- scale) scale
  let layer1Bias ← initVectorZeros 128
  let layer2Weights ← initMatrixUniform 10 128 (- scale) scale
  let layer2Bias ← initVectorZeros 10
  return {
    layer1 := { weights := layer1Weights, bias := layer1Bias },
    layer2 := { weights := layer2Weights, bias := layer2Bias }
  }

end VerifiedNN.Network.Initialization
