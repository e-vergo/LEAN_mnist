import VerifiedNN.Core.DataTypes
import SciLean

/-!
# MNIST Data Loading

Functions for loading MNIST dataset from IDX binary format.

Provides I/O utilities for parsing the MNIST handwritten digit database, which consists
of 70,000 labeled 28×28 grayscale images (60,000 training, 10,000 test).

## Main definitions

* `loadMNISTImages`: Parse IDX image file into array of 784-dimensional vectors
* `loadMNISTLabels`: Parse IDX label file into array of natural numbers (0-9)
* `loadMNIST`: Combine images and labels into dataset pairs
* `loadMNISTTrain`: Load standard training set (60,000 samples)
* `loadMNISTTest`: Load standard test set (10,000 samples)

## Implementation notes

The IDX file format is a simple binary format:
* Magic number (4 bytes): data type indicator (2051 for images, 2049 for labels)
* Dimension metadata (4 bytes each)
* Data values (1 byte per pixel/label)

All multi-byte integers are stored in big-endian format.

Pixel values are stored as UInt8 (0-255) and converted to Float without normalization
(normalization is applied separately via `VerifiedNN.Data.Preprocessing`).

**Verification status:** This is an unverified implementation operating on `Float` values.
No formal proofs of IDX parsing correctness or file I/O safety.

## References

* MNIST Database: http://yann.lecun.com/exdb/mnist/
* LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (Proc. IEEE 1998)
-/

namespace VerifiedNN.Data.MNIST

open VerifiedNN.Core
open SciLean

/-- Read big-endian 32-bit unsigned integer from byte array.

**Parameters:**
- `bytes`: Byte array to read from
- `offset`: Starting position in the array

**Returns:** `some value` if 4 bytes available at offset, `none` if out of bounds

**Implementation notes:** Decodes big-endian format (most significant byte first) -/
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

/-- Convert byte (0-255) to Float.

**Parameters:**
- `b`: UInt8 value (0-255) representing pixel intensity

**Returns:** Float representation of the byte value -/
private def byteToFloat (b : UInt8) : Float :=
  b.toNat.toFloat

/-- Load MNIST images from IDX file format.

Parses binary IDX3 format files containing MNIST digit images. The format specification:
- Magic number: 2051 (0x00000803) - identifies unsigned byte 3D tensor
- Number of images: UInt32 (big-endian)
- Number of rows: UInt32 (28 for MNIST)
- Number of columns: UInt32 (28 for MNIST)
- Pixel data: UInt8 values (0-255) in row-major order

**Parameters:**
- `path`: File path to IDX image file (e.g., `train-images-idx3-ubyte`)

**Returns:** Array of 784-dimensional Float vectors (flattened 28×28 images in row-major order),
or empty array on error (errors logged to stderr)

**Implementation notes:**
- Validates magic number, dimensions, and file size
- Converts UInt8 pixels (0-255) to Float without normalization
- Uses SciLean `⊞` notation for efficient vector construction

**Error handling:** Returns empty array and logs to stderr on file I/O errors, invalid format,
or truncated data -/
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

      -- Create vector from pixel bytes using SciLean's array constructor notation
      let pixels : Float^[784] := ⊞ (j : Idx 784) =>
        byteToFloat (bytes.get! (offset + j.1.toNat))
      images := images.push pixels

    pure images

  catch e =>
    IO.eprintln s!"Error loading MNIST images from {path}: {e}"
    pure #[]

/-- Load MNIST labels from IDX file format.

Parses binary IDX1 format files containing MNIST digit labels. The format specification:
- Magic number: 2049 (0x00000801) - identifies unsigned byte 1D tensor
- Number of labels: UInt32 (big-endian)
- Label data: UInt8 values (0-9) representing digit classes

**Parameters:**
- `path`: File path to IDX label file (e.g., `train-labels-idx1-ubyte`)

**Returns:** Array of natural numbers in range [0, 9] representing digit classes,
or empty array on error (errors logged to stderr)

**Implementation notes:**
- Validates magic number and label values (must be 0-9)
- Checks for truncated files

**Error handling:** Returns empty array and logs to stderr on file I/O errors, invalid format,
invalid label values, or truncated data -/
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

Loads and combines image and label files into paired dataset format.

**Parameters:**
- `imagePath`: File path to IDX image file
- `labelPath`: File path to IDX label file

**Returns:** Array of (image, label) pairs where:
- `image`: 784-dimensional Float vector (unnormalized, values in [0, 255])
- `label`: Nat in range [0, 9] representing digit class

**Error handling:**
- If image and label counts don't match, returns only matching pairs (min of both) and logs warning
- Propagates empty arrays from failed image/label loading -/
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

Convenience function for loading the standard MNIST training dataset.

**Parameters:**
- `dataDir`: Directory containing MNIST files

**Returns:** Array of 60,000 (image, label) pairs, or fewer on error

**Expects files:**
- `train-images-idx3-ubyte` (47 MB, 60,000 images)
- `train-labels-idx1-ubyte` (60 KB, 60,000 labels) -/
def loadMNISTTrain (dataDir : System.FilePath) : IO (Array (Vector 784 × Nat)) := do
  let imagePath := dataDir / "train-images-idx3-ubyte"
  let labelPath := dataDir / "train-labels-idx1-ubyte"
  loadMNIST imagePath labelPath

/-- Load standard MNIST test set (10,000 samples).

Convenience function for loading the standard MNIST test dataset.

**Parameters:**
- `dataDir`: Directory containing MNIST files

**Returns:** Array of 10,000 (image, label) pairs, or fewer on error

**Expects files:**
- `t10k-images-idx3-ubyte` (7.8 MB, 10,000 images)
- `t10k-labels-idx1-ubyte` (10 KB, 10,000 labels) -/
def loadMNISTTest (dataDir : System.FilePath) : IO (Array (Vector 784 × Nat)) := do
  let imagePath := dataDir / "t10k-images-idx3-ubyte"
  let labelPath := dataDir / "t10k-labels-idx1-ubyte"
  loadMNIST imagePath labelPath

end VerifiedNN.Data.MNIST
