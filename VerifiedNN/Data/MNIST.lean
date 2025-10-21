/-
# MNIST Data Loading

Load MNIST dataset from IDX or CSV format.

The IDX file format is a binary format used for storing MNIST data:
- Magic number (4 bytes): data type indicator
- Number of dimensions (4 bytes)
- Dimension sizes (4 bytes each)
- Data values (1 byte per pixel for images, 1 byte per label)

**Verification status:** Implementation only, no formal verification.
**Float vs ℝ gap:** Acknowledged - pixel values are Float approximations.
-/

import VerifiedNN.Core.DataTypes
import SciLean

namespace VerifiedNN.Data.MNIST

open VerifiedNN.Core
open SciLean

/-- Read a big-endian 32-bit unsigned integer from byte array -/
private def readU32BE (bytes : ByteArray) (offset : Nat) : Option UInt32 := do
  if offset + 4 > bytes.size then
    none
  else
    let b0 := bytes.get! offset
    let b1 := bytes.get! (offset + 1)
    let b2 := bytes.get! (offset + 2)
    let b3 := bytes.get! (offset + 3)
    some ((b0.toUInt32 <<< 24) ||| (b1.toUInt32 <<< 16) |||
          (b2.toUInt32 <<< 8) ||| b3.toUInt32)

/-- Convert byte (0-255) to Float -/
private def byteToFloat (b : UInt8) : Float :=
  b.toNat.toFloat

/-- Load MNIST images from IDX file format.

The IDX format for MNIST images:
- Magic number: 2051 (0x00000803)
- Number of images: 4 bytes
- Number of rows: 4 bytes (28)
- Number of columns: 4 bytes (28)
- Pixel data: 1 byte per pixel (0-255)

**Parameters:**
- `path`: Path to the IDX file containing MNIST images

**Returns:** Array of 784-dimensional vectors (flattened 28x28 images)

**Error handling:** Returns empty array on parse failure (logs error to stderr)
-/
def loadMNISTImages (path : System.FilePath) : IO (Array (Vector 784)) := do
  try
    -- Read entire file as ByteArray
    let bytes ← IO.FS.readBinFile path

    -- Parse header
    let some magic := readU32BE bytes 0
      | throw (IO.userError "Failed to read magic number")
    let some numImages := readU32BE bytes 4
      | throw (IO.userError "Failed to read number of images")
    let some numRows := readU32BE bytes 8
      | throw (IO.userError "Failed to read number of rows")
    let some numCols := readU32BE bytes 12
      | throw (IO.userError "Failed to read number of columns")

    -- Validate magic number for image file
    if magic != 2051 then
      throw (IO.userError s!"Invalid magic number for images: {magic}")

    -- Validate dimensions (MNIST is 28x28)
    if numRows != 28 || numCols != 28 then
      throw (IO.userError s!"Expected 28x28 images, got {numRows}x{numCols}")

    let imageSize := numRows.toNat * numCols.toNat
    if imageSize != 784 then
      throw (IO.userError s!"Image size mismatch: expected 784, got {imageSize}")

    -- Read pixel data
    let mut images : Array (Vector 784) := #[]
    let headerSize := 16

    for i in [0:numImages.toNat] do
      let offset := headerSize + i * 784
      if offset + 784 > bytes.size then
        throw (IO.userError s!"Truncated file: cannot read image {i}")

      -- Create vector from pixel bytes
      -- Directly construct using byte indexing
      let pixels : Float^[784] := sorry  -- TODO: Fix DataArrayN construction from ByteArray indexing
      images := images.push pixels

    pure images

  catch e =>
    IO.eprintln s!"Error loading MNIST images from {path}: {e}"
    pure #[]

/-- Load MNIST labels from IDX file format.

The IDX format for MNIST labels:
- Magic number: 2049 (0x00000801)
- Number of labels: 4 bytes
- Label data: 1 byte per label (0-9)

**Parameters:**
- `path`: Path to the IDX file containing MNIST labels

**Returns:** Array of natural numbers (0-9) representing digit classes

**Error handling:** Returns empty array on parse failure (logs error to stderr)
-/
def loadMNISTLabels (path : System.FilePath) : IO (Array Nat) := do
  try
    -- Read entire file as ByteArray
    let bytes ← IO.FS.readBinFile path

    -- Parse header
    let some magic := readU32BE bytes 0
      | throw (IO.userError "Failed to read magic number")
    let some numLabels := readU32BE bytes 4
      | throw (IO.userError "Failed to read number of labels")

    -- Validate magic number for label file
    if magic != 2049 then
      throw (IO.userError s!"Invalid magic number for labels: {magic}")

    -- Read label data
    let mut labels : Array Nat := #[]
    let headerSize := 8

    for i in [0:numLabels.toNat] do
      let offset := headerSize + i
      if offset >= bytes.size then
        throw (IO.userError s!"Truncated file: cannot read label {i}")

      let label := bytes.get! offset
      if label.toNat > 9 then
        throw (IO.userError s!"Invalid label value: {label.toNat} (expected 0-9)")

      labels := labels.push label.toNat

    pure labels

  catch e =>
    IO.eprintln s!"Error loading MNIST labels from {path}: {e}"
    pure #[]

/-- Load full MNIST dataset by combining images and labels.

**Parameters:**
- `imagePath`: Path to the IDX file containing MNIST images
- `labelPath`: Path to the IDX file containing MNIST labels

**Returns:** Array of (image, label) pairs where:
  - image is a 784-dimensional Float vector
  - label is a Nat in range [0-9]

**Error handling:** If image and label counts don't match, returns only matching pairs.
                   Logs warning to stderr if mismatch detected.
-/
def loadMNIST (imagePath : System.FilePath) (labelPath : System.FilePath) :
    IO (Array (Vector 784 × Nat)) := do
  let images ← loadMNISTImages imagePath
  let labels ← loadMNISTLabels labelPath

  -- Validate matching sizes
  if images.size != labels.size then
    IO.eprintln s!"Warning: image count ({images.size}) != label count ({labels.size})"
    IO.eprintln s!"         using minimum of both"

  -- Combine into pairs
  let n := min images.size labels.size
  let mut dataset : Array (Vector 784 × Nat) := Array.mkEmpty n

  for i in [0:n] do
    dataset := dataset.push (images[i]!, labels[i]!)

  pure dataset

/-- Load standard MNIST training set (60,000 samples).

**Parameters:**
- `dataDir`: Directory containing MNIST files

**Expected files:**
- `dataDir/train-images-idx3-ubyte`
- `dataDir/train-labels-idx1-ubyte`

**Returns:** Array of (image, label) pairs for training
-/
def loadMNISTTrain (dataDir : System.FilePath) : IO (Array (Vector 784 × Nat)) := do
  let imagePath := dataDir / "train-images-idx3-ubyte"
  let labelPath := dataDir / "train-labels-idx1-ubyte"
  loadMNIST imagePath labelPath

/-- Load standard MNIST test set (10,000 samples).

**Parameters:**
- `dataDir`: Directory containing MNIST files

**Expected files:**
- `dataDir/t10k-images-idx3-ubyte`
- `dataDir/t10k-labels-idx1-ubyte`

**Returns:** Array of (image, label) pairs for testing
-/
def loadMNISTTest (dataDir : System.FilePath) : IO (Array (Vector 784 × Nat)) := do
  let imagePath := dataDir / "t10k-images-idx3-ubyte"
  let labelPath := dataDir / "t10k-labels-idx1-ubyte"
  loadMNIST imagePath labelPath

end VerifiedNN.Data.MNIST
