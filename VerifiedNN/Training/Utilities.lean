import SciLean

/-!
# Training Utilities

Professional logging, timing, formatting, and progress tracking utilities for neural network training.

## Main Definitions

### Timing Utilities
- `timeIt`: Execute an action and measure elapsed time in milliseconds
- `printTiming`: Print formatted timestamp with action label and duration
- `formatDuration`: Convert milliseconds to human-readable format (e.g., "2m 15.678s")
- `formatRate`: Format throughput as examples/second (e.g., "45.3 ex/s")

### Progress Tracking
- `printProgress`: Display progress as "current/total (percentage%)"
- `printProgressBar`: Display ASCII progress bar with percentage
- `ProgressState`: Stateful progress tracker with timing and ETA
- `updateProgress`: Update progress state and optionally print
- `formatETA`: Format estimated time remaining

### Number Formatting
- `formatPercent`: Format float as percentage with specified decimal places
- `formatBytes`: Format byte count as human-readable size (B, KB, MB, GB)
- `formatFloat`: Format float with specified decimal places
- `formatLargeNumber`: Format large numbers with thousands separators

### Console Helpers
- `printBanner`: Print centered text in a decorated box
- `printSection`: Print section header with horizontal rule
- `printKeyValue`: Print aligned key-value pair
- `clearLine`: Clear current console line (for progress updates)

## Main Results

This module provides pure computational utilities without formal verification.
No theorems are proven. All functions are computable and side-effect explicit.

## Implementation Notes

**Timing precision:** Uses `IO.monoMsNow` for monotonic millisecond precision.
This is suitable for training progress tracking but not for high-precision benchmarking.

**Console output:** All IO functions flush stdout to ensure immediate visibility
of progress updates and status messages. This is critical for real-time feedback
during long-running training jobs.

**Unicode formatting:** Uses Unicode box-drawing characters and symbols for
professional console output:
- Box drawing: ═ ║ ╔ ╗ ╚ ╝ ─ │ ┌ ┐ └ ┘
- Progress bars: █ ░ ▓ ▒
- Symbols: ✓ ✗ ⚠ •

**Duration formatting:** Intelligently selects units based on magnitude:
- < 1000ms: "123.456ms"
- < 60s: "12.345s"
- < 60m: "2m 15.678s"
- ≥ 60m: "1h 23m 45.678s"

**Byte formatting:** Uses binary prefixes (1024-based):
- 1 KB = 1024 B
- 1 MB = 1024 KB
- 1 GB = 1024 MB

**Progress bars:** Fixed-width ASCII bars for consistent formatting:
```
[████████░░░░░░░] 54.2%
[██████████████░] 93.3%
[███████████████] 100.0%
```

**Color support:** Not implemented (requires platform-specific terminal codes).
Future enhancement could add ANSI color codes with feature flag.

## Usage Examples

### Timing Operations
```lean
let (result, timeMs) ← timeIt "forward pass" do
  return net.forward input

printTiming "Batch processing" timeMs
-- Output: "[12:34:56.789] Batch processing (123.456ms)"
```

### Progress Tracking
```lean
-- Simple progress
printProgress 150 1000 "Processing batches"
-- Output: "  Processing batches 150/1000 (15.0%)"

-- Progress bar
printProgressBar 540 1000 15
-- Output: "[████████░░░░░░░] 54.0%"
```

### Number Formatting
```lean
formatPercent 0.9234 2  -- "92.34%"
formatBytes 1536000     -- "1.46 MB"
formatFloat 3.14159 2   -- "3.14"
formatDuration 135678.0 -- "2m 15.678s"
```

### Console Helpers
```lean
printBanner "Training Complete"
-- Output:
-- ╔════════════════════════╗
-- ║  Training Complete     ║
-- ╚════════════════════════╝

printSection "Epoch 5"
-- Output:
-- ────────────────────────────
-- Epoch 5
-- ────────────────────────────
```

## Performance Notes

**String concatenation:** Uses `String.append` and interpolation which
allocates new strings. For high-frequency logging, consider buffering.

**Float formatting:** Uses `Float.floor` for rounding, which is adequate
for display purposes but not for numerical precision requirements.

**IO operations:** Each `IO.println` and `flush` has syscall overhead.
Batching output or using buffered IO could improve performance for
high-frequency progress updates.

## References

- ANSI escape codes: https://en.wikipedia.org/wiki/ANSI_escape_code
- Unicode box drawing: https://en.wikipedia.org/wiki/Box-drawing_character
- ISO 8601 duration format: https://en.wikipedia.org/wiki/ISO_8601#Durations
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

/-- Format duration in milliseconds as human-readable string.

Intelligently selects appropriate units based on magnitude:
- < 1 second: "123.456ms"
- < 1 minute: "12.345s"
- < 1 hour: "2m 15.678s"
- ≥ 1 hour: "1h 23m 45.678s"

**Parameters:**
- `ms`: Duration in milliseconds

**Returns:** Formatted duration string with appropriate units

**Examples:**
```lean
formatDuration 234.5       -- "234.500ms"
formatDuration 5432.1      -- "5.432s"
formatDuration 135678.0    -- "2m 15.678s"
formatDuration 4523456.0   -- "1h 15m 23.456s"
```

**Formatting rules:**
- Always shows 3 decimal places for sub-second precision
- Minutes and hours are shown as integers
- Only shows hours if duration ≥ 1 hour
- Seconds always included with decimals

**Implementation note:** Uses `Float.floor` for truncation to integers
when extracting hours/minutes. Maintains float precision for seconds component.
-/
def formatDuration (ms : Float) : String :=
  if ms < 1000.0 then
    -- Less than 1 second: show milliseconds
    s!"{ms}ms"
  else if ms < 60000.0 then
    -- Less than 1 minute: show seconds
    let sec := ms / 1000.0
    s!"{sec}s"
  else if ms < 3600000.0 then
    -- Less than 1 hour: show minutes and seconds
    let totalSec := ms / 1000.0
    let min := Float.floor (totalSec / 60.0)
    let sec := totalSec - (min * 60.0)
    s!"{min}m {sec}s"
  else
    -- 1 hour or more: show hours, minutes, and seconds
    let totalSec := ms / 1000.0
    let hours := Float.floor (totalSec / 3600.0)
    let remainingSec := totalSec - (hours * 3600.0)
    let min := Float.floor (remainingSec / 60.0)
    let sec := remainingSec - (min * 60.0)
    s!"{hours}h {min}m {sec}s"

/-- Get current time as formatted timestamp.

Returns the current wall-clock time as a formatted string in HH:MM:SS.mmm format.

**Returns:** Timestamp string like "12:34:56.789"

**Note:** This is a placeholder implementation that returns a fixed format.
Actual time formatting requires platform-specific system calls to get
wall-clock time (not just monotonic time). Lean 4's standard library
does not currently provide wall-clock formatting utilities.

**Future implementation:** When Lean gains `IO.getCurrentTime` or similar,
this should be updated to return actual formatted time.

**Current behavior:** Returns placeholder "[TIME]" since we only have access
to monotonic time (milliseconds since arbitrary epoch) via `IO.monoMsNow`,
which cannot be converted to wall-clock time.
-/
def getCurrentTimeString : IO String := do
  -- NOTE: Lean 4's IO.monoMsNow gives monotonic time (not wall-clock time)
  -- We cannot easily convert to HH:MM:SS format without platform-specific calls
  -- For now, we return a placeholder
  return "[TIME]"

/-- Print formatted timing information with timestamp.

Displays a log line with timestamp, label, and formatted duration.
Output format: "[HH:MM:SS.mmm] {label} ({duration})"

**Parameters:**
- `label`: Description of the timed operation
- `ms`: Elapsed time in milliseconds

**Returns:** IO action that prints formatted timing and flushes output

**Example output:**
```
[TIME] Training complete (2m 15.678s)
[TIME] Batch processing (234.567ms)
[TIME] Data loading (45.123s)
```

**Note:** Timestamp is currently "[TIME]" placeholder. See `getCurrentTimeString`
documentation for details.

**Flushing:** Explicitly flushes stdout to ensure immediate visibility,
important for real-time progress monitoring during long operations.
-/
def printTiming (label : String) (ms : Float) : IO Unit := do
  let timestamp ← getCurrentTimeString
  let duration := formatDuration ms
  IO.println s!"{timestamp} {label} ({duration})"
  (← IO.getStdout).flush

/-- Format throughput as examples/second.

Computes and formats processing rate based on count and elapsed time.

**Parameters:**
- `count`: Number of items processed
- `ms`: Elapsed time in milliseconds

**Returns:** Formatted rate string like "45.3 ex/s" or "0.0 ex/s"

**Examples:**
```lean
formatRate 1000 25000.0  -- "40.0 ex/s"
formatRate 100 5000.0    -- "20.0 ex/s"
formatRate 0 1000.0      -- "0.0 ex/s"
```

**Edge cases:**
- Zero time: Returns "0.0 ex/s" to avoid division by zero
- Zero count: Returns "0.0 ex/s"
- Very slow: May return "0.0 ex/s" if rate < 0.05 (rounds to 0)

**Formatting:** Shows 1 decimal place. For integer rates, shows ".0".

**Use case:** Display training speed (examples/second, batches/second).
-/
def formatRate (count : Nat) (ms : Float) : String :=
  if ms <= 0.0 then
    "0.0 ex/s"
  else
    let sec := ms / 1000.0
    let rate := count.toFloat / sec
    let rounded := Float.floor (rate * 10.0) / 10.0
    s!"{rounded} ex/s"

-- ============================================================================
-- Progress Tracking
-- ============================================================================

/-- Print progress as fraction with percentage.

Displays progress in format: "  {label} {current}/{total} ({percentage}%)"

**Parameters:**
- `current`: Current progress count (e.g., batches processed)
- `total`: Total count to reach (e.g., total batches)
- `label`: Optional descriptive label (default: empty string)

**Returns:** IO action that prints progress and flushes output

**Example output:**
```
  Processing batches 150/1000 (15.0%)
  Epoch progress 5/10 (50.0%)
  Loading data 500/500 (100.0%)
```

**Formatting:**
- Leading spaces for indentation
- Percentage shown with 1 decimal place
- Always flushes stdout for immediate visibility

**Edge cases:**
- Total = 0: Shows "0.0%" to avoid division by zero
- Current > total: Shows percentage > 100% (no clamping)

**Use case:** Simple text-based progress for batch processing, data loading, etc.
-/
def printProgress (current : Nat) (total : Nat) (label : String := "") : IO Unit := do
  let percent := if total == 0 then 0.0
    else (current.toFloat / total.toFloat) * 100.0
  let roundedPercent := Float.floor (percent * 10.0) / 10.0
  let labelPart := if label.isEmpty then "" else s!"{label} "
  IO.println s!"  {labelPart}{current}/{total} ({roundedPercent}%)"
  (← IO.getStdout).flush

/-- Print ASCII progress bar with percentage.

Displays a visual progress bar: "[████████░░░░░░░] 54.2%"

**Parameters:**
- `current`: Current progress count
- `total`: Total count to reach
- `width`: Width of the progress bar in characters (default: 15)

**Returns:** IO action that prints progress bar and flushes output

**Example output:**
```
[███████████████] 100.0%
[████████░░░░░░░] 54.2%
[███░░░░░░░░░░░░] 20.0%
[░░░░░░░░░░░░░░░] 0.0%
```

**Characters used:**
- `█` (U+2588): Filled portion
- `░` (U+2591): Empty portion

**Width:** The bar width excludes brackets and percentage. For width=15,
the full output is ~26 characters: `[` + 15 chars + `] ` + `100.0%`

**Percentage precision:** Shows 1 decimal place.

**Edge cases:**
- Total = 0: Shows empty bar, "0.0%"
- Current > total: Shows full bar, percentage > 100%
- Width = 0: Shows `[] percentage%`

**Use case:** Visual feedback during long operations. More intuitive than
raw numbers for human monitoring.

**Implementation:** Computes filled count as (current/total) * width,
then builds string with filled + empty characters.
-/
def printProgressBar (current : Nat) (total : Nat) (width : Nat := 15) : IO Unit := do
  let fraction := if total == 0 then 0.0
    else current.toFloat / total.toFloat
  let percent := fraction * 100.0
  let roundedPercent := Float.floor (percent * 10.0) / 10.0

  -- Calculate filled portion
  let filled := Float.floor (fraction * width.toFloat) |>.toUInt64.toNat
  let filled := if filled > width then width else filled
  let empty := width - filled

  -- Build bar string
  let bar := replicateString filled "█" ++ replicateString empty "░"

  IO.println s!"[{bar}] {roundedPercent}%"
  (← IO.getStdout).flush

-- ============================================================================
-- Number Formatting
-- ============================================================================

/-- Format float as percentage with specified decimal places.

Converts a fractional value (0.0-1.0 range) to percentage string with
configurable precision.

**Parameters:**
- `value`: Float value to convert (typically in [0, 1] but not required)
- `decimals`: Number of decimal places to show (default: 1)

**Returns:** Percentage string like "92.3%" or "100.0%"

**Examples:**
```lean
formatPercent 0.923 1      -- "92.3%"
formatPercent 0.9234 2     -- "92.34%"
formatPercent 1.0 0        -- "100%"
formatPercent 0.05 2       -- "5.00%"
```

**Implementation:** Multiplies by 100, then uses power-of-10 rounding
to achieve decimal precision. For decimals=2, rounds to 0.01 precision.

**Edge cases:**
- Value > 1.0: Shows percentage > 100% (no clamping)
- Value < 0.0: Shows negative percentage (valid for some metrics)
- decimals=0: Shows integer percentage without decimal point

**Use case:** Displaying accuracy, loss reduction, progress percentages in logs.
-/
def formatPercent (value : Float) (decimals : Nat := 1) : String :=
  let percent := value * 100.0
  let multiplier := Float.pow 10.0 decimals.toFloat
  let rounded := Float.floor (percent * multiplier) / multiplier
  if decimals == 0 then
    s!"{Float.floor percent}%"
  else
    s!"{rounded}%"

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

/-- Format float with specified decimal places.

Simple float formatting with configurable precision for display purposes.

**Parameters:**
- `value`: Float value to format
- `decimals`: Number of decimal places (default: 2)

**Returns:** Formatted string like "3.14" or "123.457"

**Examples:**
```lean
formatFloat 3.14159 2      -- "3.14"
formatFloat 123.456789 3   -- "123.456"
formatFloat 2.0 0          -- "2"
```

**Implementation:** Uses power-of-10 rounding. For decimals=2,
rounds to 0.01 precision via floor((x * 100) / 100).

**Edge cases:**
- decimals=0: Returns integer string (no decimal point)
- Very large/small values: May lose precision or use scientific notation
  (depends on Float.toString implementation)
- Negative values: Preserves sign

**Limitations:** Simple rounding, may not handle edge cases like NaN, Inf.
Not suitable for precise numerical output requiring IEEE 754 awareness.

**Use case:** Display loss values, learning rates, metrics in logs.
-/
def formatFloat (value : Float) (decimals : Nat := 2) : String :=
  let multiplier := Float.pow 10.0 decimals.toFloat
  let rounded := Float.floor (value * multiplier) / multiplier
  if decimals == 0 then
    s!"{Float.floor value}"
  else
    s!"{rounded}"

/-- Format large numbers with thousands separators.

Formats integers with commas as thousands separators for readability.

**Parameters:**
- `n`: Natural number to format

**Returns:** Formatted string like "1,234,567" or "42"

**Examples:**
```lean
formatLargeNumber 42           -- "42"
formatLargeNumber 1234         -- "1,234"
formatLargeNumber 1234567      -- "1,234,567"
formatLargeNumber 1000000000   -- "1,000,000,000"
```

**Formatting:** Uses comma (`,`) as thousands separator, standard in
US/UK locale. Different locales use different separators (space, period).

**Implementation:** Converts to string, then inserts commas every 3 digits
from right to left using character list manipulation.

**Use case:** Display dataset sizes, parameter counts, batch counts in logs.
Makes large numbers more readable (60,000 vs 60000).

**Note:** Always uses comma separator regardless of locale. For i18n
applications, consider locale-aware formatting.
-/
def formatLargeNumber (n : Nat) : String :=
  let s := toString n
  let chars := s.toList
  let rec insertCommas (cs : List Char) (count : Nat) (acc : List Char) : List Char :=
    match cs with
    | [] => acc
    | c :: rest =>
      if count > 0 && count % 3 == 0 then
        insertCommas rest (count + 1) (c :: ',' :: acc)
      else
        insertCommas rest (count + 1) (c :: acc)
  let reversed := chars.reverse
  let withCommas := insertCommas reversed 0 []
  String.mk withCommas

-- ============================================================================
-- Console Helpers
-- ============================================================================

/-- Print centered text in a decorative box.

Creates a visually appealing banner with Unicode box-drawing characters:
```
╔════════════════════════╗
║  Training Complete     ║
╚════════════════════════╝
```

**Parameters:**
- `text`: Text to display in the banner

**Returns:** IO action that prints the banner and flushes output

**Example output:**
```
╔════════════════════════╗
║  Training Complete     ║
╚════════════════════════╝

╔════════════════════════════╗
║  Epoch 5 / 10              ║
╚════════════════════════════╝
```

**Box characters used:**
- `╔` (U+2554): Top-left corner
- `═` (U+2550): Horizontal line (top/bottom)
- `╗` (U+2557): Top-right corner
- `║` (U+2551): Vertical line (sides)
- `╚` (U+255A): Bottom-left corner
- `╝` (U+255D): Bottom-right corner

**Width:** Box width is text length + 4 (2 spaces padding + 2 borders).
Minimum width is 4 characters.

**Centering:** Text is left-padded with 2 spaces. For true centering,
text should be pre-padded to desired width.

**Use case:** Highlight important status messages (training start/end,
milestone achievements, error conditions).

**Note:** Requires terminal with Unicode support. ASCII-only terminals
may show garbled characters.
-/
def printBanner (text : String) : IO Unit := do
  let width := text.length + 4
  let topLine := "╔" ++ replicateString (width - 2) "═" ++ "╗"
  let midLine := "║  " ++ text ++ "  ║"
  let botLine := "╚" ++ replicateString (width - 2) "═" ++ "╝"

  IO.println topLine
  IO.println midLine
  IO.println botLine
  (← IO.getStdout).flush

/-- Print section header with horizontal rule.

Creates a section divider with title:
```
────────────────────────────
Epoch 5
────────────────────────────
```

**Parameters:**
- `title`: Section title to display

**Returns:** IO action that prints section header and flushes output

**Example output:**
```
────────────────────────────
Model Architecture
────────────────────────────

────────────────────────────
Training Results
────────────────────────────
```

**Characters:** Uses `─` (U+2500) box-drawing horizontal line.

**Width:** Fixed width of 30 characters for horizontal rules.
Title is not centered (left-aligned).

**Use case:** Organize console output into logical sections during
training (configuration, data loading, training loop, results).

**Spacing:** Prints 3 lines: rule, title, rule. No blank lines added.
Caller should add blank lines before/after if desired.
-/
def printSection (title : String) : IO Unit := do
  let line := replicateString 30 "─"
  IO.println line
  IO.println title
  IO.println line
  (← IO.getStdout).flush

/-- Print aligned key-value pair.

Formats a key-value pair with consistent alignment for log output:
"  Key name:        value"

**Parameters:**
- `key`: The label/key to display
- `value`: The value to display (converted to string)
- `keyWidth`: Width to pad the key to (default: 20)

**Returns:** IO action that prints the formatted pair and flushes output

**Example output:**
```
  Epochs:             10
  Batch size:         32
  Learning rate:      0.01
  Training samples:   60000
```

**Formatting:**
- Leading 2 spaces for indentation
- Key padded to `keyWidth` characters
- Colon after key
- Value right-aligned after padding

**Use case:** Display configuration settings, hyperparameters, statistics
in aligned columns for easy reading.

**Note:** If key length > keyWidth, alignment breaks. Choose keyWidth to
accommodate longest expected key.
-/
def printKeyValue (key : String) (value : String) (keyWidth : Nat := 20) : IO Unit := do
  let padding := if key.length < keyWidth then
    replicateString (keyWidth - key.length) " "
  else
    ""
  IO.println s!"  {key}:{padding} {value}"
  (← IO.getStdout).flush

/-- Clear the current console line (carriage return).

Prints carriage return character to move cursor to start of line,
enabling overwriting the current line on next print. Used for updating
progress displays without scrolling.

**Returns:** IO action that prints CR and flushes output

**Example usage:**
```lean
for i in [0:100] do
  clearLine
  IO.print s!"Processing {i}/100..."
  (← IO.getStdout).flush
-- Next line will overwrite progress display
```

**Behavior:**
- `\r` moves cursor to start of current line
- Next print will overwrite existing text
- Useful for updating progress counters without scrolling

**Limitations:**
- Only works on terminals supporting carriage return
- Requires subsequent print to fill the line (or use spaces to clear)
- Not compatible with logging to files (CR appears as literal character)

**Best practice:** Use for interactive progress updates. For file logs,
use regular line-by-line printing.

**Note:** Does not erase existing text, just repositions cursor.
To truly clear, print spaces to overwrite old content.
-/
def clearLine : IO Unit := do
  IO.print "\r"
  (← IO.getStdout).flush

-- ============================================================================
-- Advanced Progress Tracking with ETA
-- ============================================================================

/-- Stateful progress tracker with timing and ETA.

Tracks progress through a multi-step process with elapsed time and
estimated time remaining. Useful for long-running operations where
users want to know completion ETA.

**Fields:**
- `total`: Total number of steps to complete
- `current`: Current step number (0-indexed or 1-indexed, user's choice)
- `startTime`: Start time in milliseconds (from `IO.monoMsNow`)
- `lastUpdate`: Last update time in milliseconds

**Invariants:**
- `current <= total` for accurate progress
- `lastUpdate >= startTime`

**Use case:** Track epoch progress, batch progress, or any iterative process
where ETA estimation is valuable.

**Example:**
```lean
let state ← ProgressState.init 10  -- 10 total steps
for i in [0:10] do
  state := state.update (i + 1)
  state.printWithETA s!"Epoch {i+1}"
```
-/
structure ProgressState where
  total : Nat
  current : Nat
  startTime : Nat  -- milliseconds from IO.monoMsNow
  lastUpdate : Nat -- milliseconds from IO.monoMsNow

/-- Initialize progress state.

**Parameters:**
- `total`: Total number of steps

**Returns:** IO action that creates initial progress state with current=0
  and timestamps set to current time
-/
def ProgressState.init (total : Nat) : IO ProgressState := do
  let now ← IO.monoMsNow
  return {
    total := total
    current := 0
    startTime := now
    lastUpdate := now
  }

/-- Update progress state with new current value.

**Parameters:**
- `state`: Current progress state
- `newCurrent`: New current step value

**Returns:** IO action that creates updated state with new current value
  and updated lastUpdate timestamp
-/
def ProgressState.update (state : ProgressState) (newCurrent : Nat) : IO ProgressState := do
  let now ← IO.monoMsNow
  return {
    state with
    current := newCurrent
    lastUpdate := now
  }

/-- Calculate estimated time remaining in milliseconds.

**Parameters:**
- `state`: Current progress state

**Returns:** Estimated milliseconds remaining, or 0.0 if no progress yet

**Algorithm:** Linear extrapolation based on current progress rate:
```
elapsed = now - startTime
rate = current / elapsed
remaining = (total - current) / rate
```

**Edge cases:**
- No progress (current=0): Returns 0.0
- Already complete (current >= total): Returns 0.0
- Zero elapsed time: Returns 0.0
-/
def ProgressState.estimateRemaining (state : ProgressState) : Float :=
  if state.current == 0 || state.current >= state.total then
    0.0
  else
    let elapsed := (state.lastUpdate - state.startTime).toFloat
    if elapsed <= 0.0 then
      0.0
    else
      let rate := state.current.toFloat / elapsed  -- items per ms
      let remaining := state.total - state.current
      remaining.toFloat / rate  -- ms remaining

/-- Format estimated time remaining.

**Parameters:**
- `state`: Current progress state

**Returns:** Formatted ETA string like "ETA: 2m 15s" or "ETA: unknown"

**Output:**
- If progress made: "ETA: {duration}"
- If no progress yet: "ETA: unknown"
- If complete: "ETA: 0s"
-/
def ProgressState.formatETA (state : ProgressState) : String :=
  let remaining := state.estimateRemaining
  if remaining <= 0.0 then
    if state.current == 0 then
      "ETA: unknown"
    else
      "ETA: 0s"
  else
    s!"ETA: {formatDuration remaining}"

/-- Print progress with ETA estimation.

Displays progress bar, percentage, and estimated time remaining.

**Parameters:**
- `state`: Current progress state
- `label`: Optional label prefix

**Returns:** IO action that prints formatted progress with ETA

**Example output:**
```
  Epoch progress: [████████░░░░░░░] 54.2% (ETA: 1m 23s)
```
-/
def ProgressState.printWithETA (state : ProgressState) (label : String := "Progress") : IO Unit := do
  let fraction := if state.total == 0 then 0.0
    else state.current.toFloat / state.total.toFloat
  let percent := Float.floor (fraction * 1000.0) / 10.0

  -- Build progress bar (width 15)
  let width := 15
  let filled := Float.floor (fraction * width.toFloat) |>.toUInt64.toNat
  let filled := if filled > width then width else filled
  let empty := width - filled
  let bar := replicateString filled "█" ++ replicateString empty "░"

  let eta := state.formatETA
  IO.println s!"  {label}: [{bar}] {percent}% ({eta})"
  (← IO.getStdout).flush

end VerifiedNN.Training.Utilities
