# File Review: Utilities.lean

## Summary
Comprehensive console utilities for training progress, timing, and formatting. Zero axioms, zero sorries, but **heavily underutilized** (only 3 external references despite 30+ utility functions). Most code is orphaned.

## Findings

### Orphaned Code
**CRITICAL: 90% of definitions are unused (27 out of 30)**

**Timing utilities (4 functions, 0 used externally):**
- `timeIt` (198-203) - ✗ Only used in MNISTTrain.lean
- `formatDuration` (235-256) - ✗ Only used internally in Utilities.lean
- `getCurrentTimeString` (276-280) - ✗ Only used internally
- `printTiming` (306-310) - ✗ Not used
- `formatRate` (338-345) - ✗ Not used

**Progress tracking (3 functions, 0 used externally):**
- `printProgress` (380-386) - ✗ Not used
- `printProgressBar` (427-442) - ✗ Not used
- `ProgressState` + 4 methods (832-954) - ✗ Not used

**Number formatting (4 functions, 0 used externally):**
- `formatPercent` (477-484) - ✗ Not used
- `formatBytes` (521-536) - ✗ Only used in Serialization.lean
- `formatFloat` (569-575) - ✗ Not used
- `formatLargeNumber` (606-619) - ✗ Not used

**Console helpers (4 functions, 0 used externally):**
- `printBanner` (670-679) - ✗ Not used
- `printSection` (717-722) - ✗ Not used
- `printKeyValue` (756-762) - ✗ Not used
- `clearLine` (797-799) - ✗ Not used

**Internal helper (1 function):**
- `replicateString` (164-165) - ✓ Used internally (progress bars)

**ONLY 3 FUNCTIONS USED EXTERNALLY:**
1. `timeIt` - MNISTTrain.lean (1 usage)
2. `formatBytes` - Serialization.lean (1 usage)
3. Internal references don't count

**Usage breakdown:**
- Utilities.lean (self-references): ~50 lines
- MNISTTrain.lean: 1 function (timeIt)
- Serialization.lean: 1 function (formatBytes)
- **Total external usage: 2 out of 30 functions (6.7%)**

### Axioms (Total: 0)
**None.** Pure computational formatting code.

### Sorries (Total: 0)
**None.** No formal verification.

### Code Correctness Issues
**Implementation appears correct but mostly untested:**

1. **timeIt (Lines 198-203):**
   - ✓ Correct timing: endTime - startTime
   - ✓ Monotonic clock usage correct
   - ✓ **TESTED:** Used in MNISTTrain.lean

2. **formatDuration (Lines 235-256):**
   - ✓ Unit selection logic correct (ms, s, m, h)
   - ✓ Math correct: floor division for minutes/hours
   - ⚠️ **UNTESTED:** Only used internally

3. **getCurrentTimeString (Lines 276-280):**
   - ✗ **PLACEHOLDER:** Always returns "[TIME]"
   - ✗ **TODO:** Needs wall-clock time implementation
   - ⚠️ **BROKEN:** Not a real timestamp

4. **printTiming (Lines 306-310):**
   - ✓ Formatting correct
   - ⚠️ **NEVER EXECUTED:** No external usage

5. **formatRate (Lines 338-345):**
   - ✓ Rate calculation correct: count / (ms / 1000)
   - ✓ Edge case: ms ≤ 0 returns "0.0 ex/s"
   - ⚠️ **NEVER EXECUTED:** No usage

6. **printProgress (Lines 380-386):**
   - ✓ Percentage calculation correct
   - ✓ Edge case: total == 0 → 0%
   - ⚠️ **NEVER EXECUTED:** No usage

7. **printProgressBar (Lines 427-442):**
   - ✓ Bar rendering logic correct
   - ✓ Edge cases handled
   - ⚠️ **NEVER EXECUTED:** No usage

8. **formatPercent (Lines 477-484):**
   - ✓ Percentage conversion correct
   - ✓ Decimal precision correct
   - ⚠️ **NEVER EXECUTED:** No usage

9. **formatBytes (Lines 521-536):**
   - ✓ Unit conversion correct (B, KB, MB, GB)
   - ✓ Thresholds correct (1024-based)
   - ✓ **TESTED:** Used in Serialization.lean

10. **formatFloat (Lines 569-575):**
    - ✓ Rounding logic correct
    - ⚠️ **NEVER EXECUTED:** No usage

11. **formatLargeNumber (Lines 606-619):**
    - ✓ Comma insertion logic correct
    - ⚠️ **NEVER EXECUTED:** No usage

12. **printBanner (Lines 670-679):**
    - ✓ Box drawing correct
    - ⚠️ **NEVER EXECUTED:** No usage

13. **printSection (Lines 717-722):**
    - ✓ Formatting correct
    - ⚠️ **NEVER EXECUTED:** No usage

14. **printKeyValue (Lines 756-762):**
    - ✓ Alignment logic correct
    - ⚠️ **NEVER EXECUTED:** No usage

15. **clearLine (Lines 797-799):**
    - ✓ Carriage return correct
    - ⚠️ **NEVER EXECUTED:** No usage

16. **ProgressState + methods (Lines 832-954):**
    - ✓ ETA calculation logic correct
    - ✓ State tracking correct
    - ⚠️ **NEVER EXECUTED:** No usage (entire subsystem orphaned)

### Hacks & Deviations

**Broken placeholder:**
- **Lines 276-280:** `getCurrentTimeString` returns "[TIME]" - **Severity: moderate**
  - Not a real timestamp
  - Comment admits limitation: "Lean 4's IO.monoMsNow gives monotonic time (not wall-clock time)"
  - Affects `printTiming` (also unused)
  - Should either implement properly or document as permanent limitation

**Over-engineering:**
- **Lines 832-954:** Entire ProgressState subsystem unused - **Severity: significant**
  - 123 lines of unused code
  - Sophisticated ETA tracking that nobody calls
  - Should be removed or integrated

**Code duplication with Loop.lean:**
- Training/Loop.lean has its own TrainingLog namespace (lines 145-189)
- Utilities.lean has parallel but unused functionality
- Indicates incomplete consolidation or abandoned refactoring

## Statistics
- **Definitions:** 30 total
  - Timing: 5 (timeIt, formatDuration, getCurrentTimeString, printTiming, formatRate)
  - Progress: 8 (printProgress, printProgressBar, ProgressState + methods)
  - Formatting: 4 (formatPercent, formatBytes, formatFloat, formatLargeNumber)
  - Console: 5 (printBanner, printSection, printKeyValue, clearLine, replicateString)
- **Unused definitions:** 28 (93% orphaned)
- **Partially broken:** 1 (getCurrentTimeString - placeholder)
- **Theorems:** 0
- **Axioms:** 0
- **Sorries:** 0
- **Lines of code:** 956 (largest file in Training/)
- **Documentation quality:** Excellent (comprehensive docstrings for all functions)
- **Usage:** Only 2 functions used externally (6.7% utilization)
- **Test coverage:** ~0% (most code never executed)

## Recommendations

### Priority 1: Massive Code Cleanup Required

**Option A: Remove orphaned code (strongly recommended)**
Delete 90% of this file:
```bash
# Keep only:
# - timeIt (used in MNISTTrain.lean)
# - formatBytes (used in Serialization.lean)
# - replicateString (internal helper)

# Delete:
# - All progress tracking (printProgress, printProgressBar, ProgressState)
# - All console helpers (printBanner, printSection, printKeyValue, clearLine)
# - Unused formatting (formatPercent, formatFloat, formatLargeNumber)
# - Broken timestamp code (getCurrentTimeString, printTiming)
```

**Estimated reduction:** 956 lines → ~100 lines (89% reduction)

**Option B: Integrate utilities into training loop**
- Replace TrainingLog in Loop.lean with Utilities functions
- Use printProgressBar instead of custom progress in Loop.lean
- Add calls to printBanner for training start/end
- Fix getCurrentTimeString to use actual wall-clock time

**Requires significant refactoring:** Not recommended unless there's a specific need for these utilities.

### Priority 2: Fix or Remove Broken Code

**getCurrentTimeString (Lines 276-280):**
- Either implement wall-clock time formatting
- Or remove printTiming (which depends on it)
- Document as permanent limitation if keeping

### Priority 3: Consolidate with Loop.lean

**Current duplication:**
- Loop.lean has TrainingLog namespace (5 functions)
- Utilities.lean has overlapping but unused functionality
- Choose one approach and delete the other

### Priority 4: Add Tests if Keeping Code

If utilities are integrated:
- Test formatDuration with various inputs
- Test formatBytes edge cases
- Test progress bar rendering
- Test ProgressState ETA calculation

## Critical Issues

**This file has the worst code health in Training/ directory:**

1. **93% dead code:** 28 out of 30 functions never called
2. **Broken placeholder:** getCurrentTimeString doesn't work
3. **Massive file size:** 956 lines, mostly unused
4. **Zero test coverage:** Untested code that never executes
5. **Duplication:** Overlaps with Loop.lean logging

## Decision Required

**Maintainer must choose:**

1. **Delete orphaned code** (recommended)
   - Keep only timeIt, formatBytes
   - Reduce to ~100 lines
   - Eliminate maintenance burden

2. **Integrate into training loop**
   - Replace TrainingLog with Utilities
   - Fix getCurrentTimeString
   - Add comprehensive tests
   - Significant refactoring required

3. **Document as "future utilities"**
   - Mark entire file as WIP/experimental
   - Explain why it's not integrated
   - Risk: code rot over time

**Current state is unacceptable:** 900+ lines of beautiful, well-documented, completely unused code.
