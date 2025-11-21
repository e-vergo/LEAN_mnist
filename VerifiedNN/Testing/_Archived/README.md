# Archived Test Files

This directory contains test files that have been archived (not deleted) for historical reference.

## Files

### FiniteDifference.lean
- **Archived:** November 21, 2025
- **Reason:** Duplicates functionality of GradientCheck.lean
- **Status:** Functional but redundant
- **Replacement:** Use GradientCheck.lean instead (superior implementation)
- **Lines:** 458 lines

**Why archived instead of deleted:**
- Code was functional and working correctly
- Represents historical approach to gradient checking
- May have value for comparison or reference
- GradientCheck.lean is the superior implementation (776 lines, 15 comprehensive tests, ALL PASS)

**Key differences from GradientCheck.lean:**
- FiniteDifference.lean: Infrastructure-focused (gradient checking primitives)
- GradientCheck.lean: Test-focused (comprehensive validation of all operations)

## Why Archive vs Delete?

Files in this directory were working code that became redundant due to:
- Better implementations existing elsewhere
- Duplicate functionality
- Evolution of testing strategy

They are kept for historical reference and comparison, but are not part of the active test suite.

## Impact on Build

Archived files are **not imported** by any active code. They will not be compiled during normal builds.

If you need to reference or restore an archived file:
1. Copy it back to `/Users/eric/LEAN_mnist/VerifiedNN/Testing/`
2. Update imports in test runners if necessary
3. Verify it still builds with current dependencies

---

**Archive Maintained By:** Phase 5 Cleanup (November 21, 2025)
