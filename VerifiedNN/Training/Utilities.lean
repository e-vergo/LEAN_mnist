import SciLean

/-!
# Training Utilities

Minimal utility functions for neural network training.

## Main Definitions

### Timing Utilities
- `timeIt`: Execute an action and measure elapsed time in milliseconds

### Number Formatting
- `formatBytes`: Format byte count as human-readable size (B, KB, MB, GB)

### Helper Functions
- `replicateString`: Replicate a string n times (internal helper)

## Implementation Notes

**Timing precision:** Uses `IO.monoMsNow` for monotonic millisecond precision.
This is suitable for training progress tracking but not for high-precision benchmarking.

**Byte formatting:** Uses binary prefixes (1024-based):
- 1 KB = 1024 B
- 1 MB = 1024 KB
- 1 GB = 1024 MB

## Verification Status

This module provides pure computational utilities without formal verification.
No theorems are proven. All functions are computable and side-effect explicit.

- **Sorries:** 0
- **Axioms:** 0
- **Compilation:** ✅ Fully computable
-/

namespace VerifiedNN.Training.Utilities

open SciLean

-- ============================================================================
-- Helper Functions
-- ============================================================================

/-- Replicate a string n times.

Helper function since `String.replicate` only works with `Char`.

**Parameters:**
- `n`: Number of times to repeat the string
- `s`: String to repeat

**Returns:** Concatenated string repeated n times

**Examples:**
```lean
replicateString 3 "abc"  -- "abcabcabc"
replicateString 5 "█"    -- "█████"
```
-/
def replicateString (n : Nat) (s : String) : String :=
  String.join (List.replicate n s)

-- ============================================================================
-- Timing Utilities
-- ============================================================================

/-- Execute an action and measure elapsed time.

Runs the provided IO action and returns both the result and the elapsed
time in milliseconds. Uses monotonic clock for accurate timing independent
of system clock adjustments.

**Parameters:**
- `label`: Descriptive label for the timed operation (currently unused, for future logging)
- `action`: IO action to execute and time

**Returns:** Tuple of (action result, elapsed time in milliseconds)

**Example:**
```lean
let (trainedNet, timeMs) ← timeIt "training epoch" do
  trainOneEpoch net data config
IO.println s!"Training took {timeMs}ms"
```

**Implementation:** Uses `IO.monoMsNow` before and after the action,
computing the difference. Monotonic clock prevents timing anomalies from
system clock changes (NTP adjustments, daylight saving time, etc.).

**Precision:** Millisecond precision is adequate for training operations
(typically seconds to minutes). For microsecond-level benchmarking,
consider platform-specific high-resolution timers.
-/
def timeIt {α : Type} (_label : String) (action : IO α) : IO (α × Float) := do
  let startTime ← IO.monoMsNow
  let result ← action
  let endTime ← IO.monoMsNow
  let elapsedMs := (endTime - startTime).toFloat
  return (result, elapsedMs)

-- ============================================================================
-- Number Formatting
-- ============================================================================

/-- Format byte count as human-readable size.

Converts byte count to appropriate unit (B, KB, MB, GB) with 2 decimal places.

**Parameters:**
- `bytes`: Number of bytes to format

**Returns:** Formatted string like "1.46 MB" or "512 B"

**Examples:**
```lean
formatBytes 789           -- "789 B"
formatBytes 2048          -- "2.00 KB"
formatBytes 1536000       -- "1.46 MB"
formatBytes 5368709120    -- "5.00 GB"
```

**Units:** Uses binary prefixes (base 1024):
- 1 KB = 1024 B
- 1 MB = 1024 KB = 1,048,576 B
- 1 GB = 1024 MB = 1,073,741,824 B

**Precision:** Shows 2 decimal places for KB/MB/GB, no decimals for bytes.

**Thresholds:**
- < 1024 B: Show as bytes
- < 1 MB: Show as KB
- < 1 GB: Show as MB
- ≥ 1 GB: Show as GB

**Use case:** Display model size, dataset size, memory usage in logs.

**Note:** Uses binary (1024-based) not decimal (1000-based) units.
This matches OS file size displays but differs from storage marketing.
-/
def formatBytes (bytes : Nat) : String :=
  let b := bytes.toFloat
  if bytes < 1024 then
    s!"{bytes} B"
  else if bytes < 1024 * 1024 then
    let kb := b / 1024.0
    let rounded := Float.floor (kb * 100.0) / 100.0
    s!"{rounded} KB"
  else if bytes < 1024 * 1024 * 1024 then
    let mb := b / (1024.0 * 1024.0)
    let rounded := Float.floor (mb * 100.0) / 100.0
    s!"{rounded} MB"
  else
    let gb := b / (1024.0 * 1024.0 * 1024.0)
    let rounded := Float.floor (gb * 100.0) / 100.0
    s!"{rounded} GB"

end VerifiedNN.Training.Utilities
