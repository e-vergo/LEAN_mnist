/-
# MNIST Data Loading

Load MNIST dataset from IDX or CSV format.
-/

import VerifiedNN.Core.DataTypes

namespace VerifiedNN.Data.MNIST

open VerifiedNN.Core

/-- Load MNIST images from IDX file format -/
def loadMNISTImages (path : System.FilePath) : IO (Array (Vector 784)) :=
  sorry -- TODO: implement IDX parser

/-- Load MNIST labels from IDX file format -/
def loadMNISTLabels (path : System.FilePath) : IO (Array Nat) :=
  sorry -- TODO: implement IDX parser

/-- Load full MNIST dataset -/
def loadMNIST (imagePath : System.FilePath) (labelPath : System.FilePath) :
    IO (Array (Vector 784 × Nat)) :=
  sorry -- TODO: combine images and labels

end VerifiedNN.Data.MNIST
