/-
# Verified Neural Network Training in Lean 4

This library implements and formally verifies a multilayer perceptron (MLP)
trained on MNIST handwritten digits using stochastic gradient descent (SGD)
with backpropagation.

## Main Components

- **Core**: Basic data types, linear algebra, and activation functions
- **Layer**: Dense layer implementation with formal properties
- **Network**: MLP architecture and gradient computation
- **Loss**: Cross-entropy loss function with proofs
- **Optimizer**: SGD and other optimization algorithms
- **Training**: Training loop and mini-batch handling
- **Data**: MNIST data loading and preprocessing
- **Verification**: Formal proofs of correctness
- **Testing**: Unit tests and gradient checking
- **Examples**: Simple examples and MNIST training script
-/

-- Core modules
import VerifiedNN.Core.DataTypes
import VerifiedNN.Core.LinearAlgebra
import VerifiedNN.Core.Activation

-- Layer modules
import VerifiedNN.Layer.Dense
import VerifiedNN.Layer.Composition
import VerifiedNN.Layer.Properties

-- Network modules
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Network.Gradient

-- Loss modules
import VerifiedNN.Loss.CrossEntropy
import VerifiedNN.Loss.Properties
import VerifiedNN.Loss.Gradient

-- Optimizer modules
import VerifiedNN.Optimizer.SGD
import VerifiedNN.Optimizer.Momentum
import VerifiedNN.Optimizer.Update

-- Training modules
import VerifiedNN.Training.Loop
import VerifiedNN.Training.Batch
import VerifiedNN.Training.Metrics

-- Data modules
import VerifiedNN.Data.MNIST
import VerifiedNN.Data.Preprocessing
import VerifiedNN.Data.Iterator

-- Verification modules
import VerifiedNN.Verification.GradientCorrectness
import VerifiedNN.Verification.TypeSafety
import VerifiedNN.Verification.Convergence
import VerifiedNN.Verification.Tactics

-- Testing modules
import VerifiedNN.Testing.GradientCheck
import VerifiedNN.Testing.UnitTests
import VerifiedNN.Testing.Integration

-- Examples
import VerifiedNN.Examples.SimpleExample
import VerifiedNN.Examples.MNISTTrain
