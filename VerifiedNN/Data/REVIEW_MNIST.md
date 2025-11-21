# File Review: MNIST.lean

## Summary
IDX binary format parser for MNIST dataset loading. Critical infrastructure code that is heavily used throughout the project. Implementation is robust with comprehensive error handling, but lacks formal verification of binary parsing correctness.

## Findings

### Orphaned Code

#### Private Helper Functions (All Used)
- **Line 55-64**: `readU32BE` - Used by both `loadMNISTImages` and `loadMNISTLabels` (internal helper)
- **Line 72-73**: `byteToFloat` - Used by `loadMNISTImages` for pixel conversion (internal helper)

**Assessment:** No orphaned code - all definitions are actively used.

### Axioms (Total: 0)
None - this is a pure implementation module with IO effects.

### Sorries (Total: 0)
None - all code is complete and executable.

### Code Correctness Issues

#### Big-Endian Decoding (Line 55-64)
```lean
some ((b0.toUInt32 <<< 24) ||| (b1.toUInt32 <<< 16) |||
      (b2.toUInt32 <<< 8) ||| b3.toUInt32)
```
- **Correctness:** ✓ Properly implements big-endian byte order (most significant byte first)
- **Bit shifting:** Correct (<<< is Lean's left shift operator)
- **Bitwise OR:** Properly combines bytes using `|||`
- **Validation:** Matches IDX format specification

#### IDX Format Validation (Lines 112-122)
```lean
if magic != 2051 then
  throw (IO.userError s!"Invalid magic number for images: {magic}")
if numRows != 28 || numCols != 28 then
  throw (IO.userError s!"Expected 28x28 images, got {numRows}x{numCols}")
```
- **Magic numbers:** Correctly validates 2051 (images) and 2049 (labels)
- **Dimension checking:** Enforces MNIST-specific 28×28 constraint
- **Issue:** Hard-coded dimensions prevent reuse for other IDX datasets

#### Array Bounds Validation (Line 130-131, 184)
```lean
if offset + 784 > bytes.size then
  throw (IO.userError s!"Truncated file: cannot read image {i}")
```
- **Correctness:** ✓ Prevents buffer overflow on truncated files
- **Edge case handling:** Properly detects incomplete images/labels
- **Error reporting:** Clear error messages with indices

#### Label Value Validation (Line 188-189)
```lean
if label.toNat > 9 then
  throw (IO.userError s!"Invalid label value: {label.toNat} (expected 0-9)")
```
- **Correctness:** ✓ Enforces MNIST label range [0, 9]
- **Data integrity:** Catches corrupted or non-MNIST files

#### Size Mismatch Handling (Line 223-225)
```lean
if images.size != labels.size then
  IO.eprintln s!"Warning: image count ({images.size}) != label count ({labels.size})"
  IO.eprintln s!"         using minimum of both"
```
- **Behavior:** Takes minimum of both arrays - recovers gracefully
- **Warning:** Logs to stderr but continues execution
- **Design choice:** Pragmatic for partial datasets, but could mask errors
- **Recommendation:** Consider making this configurable (strict vs. lenient mode)

### Hacks & Deviations

#### Hard-Coded MNIST Dimensions (Lines 116-122) - Severity: Moderate
```lean
if numRows != 28 || numCols != 28 then
  throw (IO.userError s!"Expected 28x28 images, got {numRows}x{numCols}")
let imageSize := numRows.toNat * numCols.toNat
if imageSize != 784 then
  throw (IO.userError s!"Image size mismatch: expected 784, got {imageSize}")
```
- **Issue:** Function signature allows arbitrary dimensions (`loadMNISTImages : IO (Array (Vector 784))`) but implementation enforces 28×28
- **Inconsistency:** Type signature suggests genericity, but validation rejects non-MNIST sizes
- **Impact:** Cannot reuse parser for other IDX format datasets (e.g., Fashion-MNIST with different dimensions)
- **Recommendation:** Either remove dimension checks for generic IDX parser, or make dimensions type parameters

#### Unsafe Array Indexing with `!` (Lines 134-135, 187, 232) - Severity: Minor
```lean
let pixels : Float^[784] := ⊞ (j : Idx 784) =>
  byteToFloat (bytes.get! (offset + j.1.toNat))
```
- **Pattern:** Uses `get!` (unsafe indexing) instead of bounds-checked access
- **Justification:** Bounds already validated in conditional checks (lines 130, 184)
- **Safety:** Loop invariants guarantee offset + j < bytes.size
- **Assessment:** Acceptable - validation ensures safety, and `!` avoids proof overhead

#### Silent Error Recovery (Lines 140-142, 195-197) - Severity: Moderate
```lean
catch e =>
  IO.eprintln s!"Error loading MNIST images from {path}: {e}"
  pure #[]
```
- **Behavior:** Returns empty array on any IO error (file not found, permission denied, corrupted data, etc.)
- **Issue:** Caller cannot distinguish "no data" from "load failure" without checking stderr
- **Silent failure:** Training code might proceed with empty dataset unnoticed
- **Impact:** Could cause confusing downstream errors (division by zero, etc.)
- **Recommendation:** Use `IO.ofExcept` or `EIO` to propagate errors, or add explicit validation that array is non-empty

#### No Pixel Value Validation (Line 134-135) - Severity: Minor
```lean
byteToFloat (bytes.get! (offset + j.1.toNat))
```
- **Observation:** UInt8 values (0-255) are converted to Float without validation
- **Assumption:** All byte values are valid pixel intensities
- **Note:** This is correct for MNIST - pixels are inherently valid
- **Assessment:** No issue for MNIST use case

#### Missing File Path Validation (Line 97, 163, 217) - Severity: Minor
- Functions accept `System.FilePath` without checking existence before reading
- Relies on `IO.FS.readBinFile` to fail and trigger catch block
- **Assessment:** Standard IO pattern - file system checks are redundant with IO error handling

### Verification Gaps

#### No Formal Proof of Parsing Correctness (Lines 97-142, 163-196)
- **Gap:** No verification that parsed bytes correctly represent intended data
- **Examples of unverified properties:**
  - Big-endian decoding matches specification
  - Row-major pixel ordering preserved
  - No byte swapping errors
  - Correct offset arithmetic (no off-by-one errors)
- **Status:** Documented as "unverified implementation" (line 32-33)
- **Assessment:** Acceptable for research prototype - empirically validated via MNISTLoadTest

#### Float Conversion Without Numerical Properties (Line 72-73)
```lean
private def byteToFloat (b : UInt8) : Float :=
  b.toNat.toFloat
```
- **Unverified:** No proof that `b.toNat.toFloat` preserves intended value
- **Gap:** UInt8 → Nat → Float conversion chain not formally verified
- **Impact:** Falls under documented Float/ℝ verification boundary
- **Assessment:** Acceptable - empirically correct and documented limitation

## Statistics
- **Structures:** 0
- **Definitions:** 7 total
  - Public: 5 (loadMNISTImages, loadMNISTLabels, loadMNIST, loadMNISTTrain, loadMNISTTest)
  - Private: 2 (readU32BE, byteToFloat)
- **Unused definitions:** 0 (all heavily used throughout Examples/ and Testing/)
- **Theorems:** 0 (implementation only, no verification)
- **Axioms:** 0
- **Sorries:** 0
- **Lines of code:** 270
- **Documentation quality:** Excellent - comprehensive docstrings with format specifications

## Usage Analysis
**Heavily used across 20+ files:**
- Examples: MNISTTrainFull, MNISTTrainMedium, MiniTraining, TrainManual, RenderMNIST, etc.
- Testing: MNISTLoadTest, MNISTIntegration, FullIntegration, DebugTraining, etc.
- Utilities: CheckDataDistribution, ImageRenderer

**Critical dependency:** Core infrastructure for all MNIST-based workflows.

## Recommendations

### High Priority
1. **Improve error handling** - Return `IO (Except String (Array ...))` instead of silently returning empty arrays
   - Allows callers to distinguish load failures from empty datasets
   - Prevents silent failures that could cause confusing downstream errors

### Medium Priority
1. **Make dimension checking configurable** - Add parameter to allow non-MNIST IDX files (e.g., Fashion-MNIST)
   - Current hard-coding prevents reuse for other datasets
   - Type signature suggests genericity but implementation rejects non-28×28

2. **Add explicit dataset size validation** - Warn or error if loaded arrays are suspiciously small
   - Helps catch silent failures from corrupted downloads
   - Could detect truncated files earlier in pipeline

### Low Priority
1. **Consider stricter size mismatch handling** - Make configurable whether to error or take minimum
   - Current behavior (take minimum) could mask data corruption
   - Add `strict: Bool` parameter to `loadMNIST`

2. **Add checksums or magic byte validation** - Verify file integrity beyond just magic numbers
   - Could detect partial downloads or corruption
   - Standard MNIST files have known sizes (47MB, 60KB, etc.)

3. **Extract generic IDX parser** - Separate MNIST-specific validation from generic IDX format parsing
   - Would allow reuse for other IDX datasets (Fashion-MNIST, EMNIST, etc.)
   - Keep MNIST-specific wrappers for convenience

## Overall Assessment
**Status:** Production-quality implementation with room for better error handling

This is robust, well-documented code that correctly implements the IDX binary format specification. The implementation handles edge cases properly (truncated files, size mismatches, invalid values) and has been empirically validated across 20+ usage sites.

**Main weakness:** Silent failure mode (returning empty arrays) could cause confusing downstream errors. Switching to explicit error propagation would improve debuggability.

**Correctness:** While unverified formally, the implementation has been extensively tested and is correct for the MNIST use case. The hard-coded dimension checks prevent generalization but ensure MNIST-specific safety.
