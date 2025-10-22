import VerifiedNN.Testing.UnitTests
import VerifiedNN.Testing.OptimizerTests
import VerifiedNN.Testing.Integration
import VerifiedNN.Testing.LinearAlgebraTests
import VerifiedNN.Testing.LossTests
import VerifiedNN.Testing.NumericalStabilityTests
import VerifiedNN.Testing.DataPipelineTests
import VerifiedNN.Testing.GradientCheck

/-!
# Test Runner

Entry point for running all VerifiedNN tests.

## Main Definition

- `runAllTests`: Unified test runner that executes all available test suites
  and reports comprehensive results

## Test Suites

This runner executes the following test suites in order:

1. **Unit Tests** - Component-level tests for activations, data types (✓ Working)
2. **Linear Algebra Tests** - Matrix and vector operations (✓ NEW - Comprehensive)
3. **Loss Tests** - Loss function properties and stability (✓ NEW - Comprehensive)
4. **Numerical Stability Tests** - Edge cases and extreme values (✓ NEW - Comprehensive)
5. **Data Pipeline Tests** - Preprocessing and iteration (✓ NEW - Comprehensive)
6. **Optimizer Tests** - Parameter update operations (✓ Working)
7. **Gradient Check Tests** - Finite difference validation (✓ Infrastructure ready)
8. **Integration Tests** - End-to-end pipeline tests (⚠ Partial, dataset generation ready)

## Implementation Notes

**Test Organization:** Tests are organized by dependency level:
- **Level 0**: Core functionality (activations, data types, linear algebra) - WORKING
- **Level 1**: Optimizer operations, loss functions, preprocessing - WORKING
- **Level 2**: Numerical stability, gradient checking - WORKING
- **Level 3**: Data pipeline, integration tests - PARTIAL

**Status Reporting:** The runner provides clear visual separation between test
suites and reports pass/fail status for each test.

## Current Status

- ✓ Ready to run: All test suites listed above
- ✓ NEW: 4 comprehensive test suites added (LinearAlgebra, Loss, NumericalStability, DataPipeline)
- ✓ EXPANDED: Gradient checking infrastructure available
- ⚠ Partial: Full integration tests (blocked by Training.Loop)

## Test Coverage Summary

**Total Test Suites:** 8
**New Test Suites (2025-10-22):** 4
- LinearAlgebraTests: 9 test suites covering all matrix/vector operations
- LossTests: 7 test suites covering cross-entropy and softmax properties
- NumericalStabilityTests: 7 test suites for extreme value handling
- DataPipelineTests: 8 test suites for preprocessing and iteration

## Usage

```bash
# Run all available tests
lake build VerifiedNN.Testing.RunTests
lake env lean --run VerifiedNN/Testing/RunTests.lean

# Run specific test suites
lake env lean --run VerifiedNN/Testing/UnitTests.lean
lake env lean --run VerifiedNN/Testing/LinearAlgebraTests.lean
lake env lean --run VerifiedNN/Testing/LossTests.lean
lake env lean --run VerifiedNN/Testing/NumericalStabilityTests.lean
lake env lean --run VerifiedNN/Testing/DataPipelineTests.lean
lake env lean --run VerifiedNN/Testing/OptimizerTests.lean
lake env lean --run VerifiedNN/Testing/GradientCheck.lean
lake env lean --run VerifiedNN/Testing/Integration.lean
```
-/

namespace VerifiedNN.Testing.Runner

open VerifiedNN.Testing.UnitTests
open VerifiedNN.Testing.OptimizerTests
open VerifiedNN.Testing.Integration
open VerifiedNN.Testing.LinearAlgebraTests
open VerifiedNN.Testing.LossTests
open VerifiedNN.Testing.NumericalStabilityTests
open VerifiedNN.Testing.DataPipelineTests
open VerifiedNN.Testing.GradientCheck

/-- Run all available test suites with comprehensive reporting.

Executes the following test suites in order:
1. Unit Tests: Activation functions, data types, approximate equality
2. Linear Algebra Tests: Matrix/vector operations, batch operations
3. Loss Tests: Cross-entropy properties, softmax stability
4. Numerical Stability Tests: Extreme values, NaN/Inf handling
5. Data Pipeline Tests: Preprocessing, iteration, MNIST pipeline
6. Optimizer Tests: SGD, momentum, learning rate scheduling, gradient accumulation
7. Gradient Check Tests: Finite difference validation of gradients
8. Integration Tests: Dataset generation and placeholder tests for full pipeline

**Returns:** IO Unit - Prints test results to stdout with visual separators

**Usage:**
```lean
-- From Lean code
RunAllTests.runAllTests

-- From command line
lake env lean --run VerifiedNN/Testing/RunTests.lean
```
-/
def runAllTests : IO Unit := do
  IO.println "=========================================="
  IO.println "VerifiedNN Complete Test Suite"
  IO.println "=========================================="
  IO.println ""

  -- Run unit tests
  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║          UNIT TESTS                      ║"
  IO.println "╚══════════════════════════════════════════╝"
  UnitTests.runAllTests
  IO.println ""

  -- Run linear algebra tests (NEW)
  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║      LINEAR ALGEBRA TESTS                ║"
  IO.println "╚══════════════════════════════════════════╝"
  LinearAlgebraTests.runAllTests
  IO.println ""

  -- Run loss function tests (NEW)
  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║        LOSS FUNCTION TESTS               ║"
  IO.println "╚══════════════════════════════════════════╝"
  LossTests.runAllTests
  IO.println ""

  -- Run numerical stability tests (NEW)
  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║    NUMERICAL STABILITY TESTS             ║"
  IO.println "╚══════════════════════════════════════════╝"
  NumericalStabilityTests.runAllTests
  IO.println ""

  -- Run data pipeline tests (NEW)
  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║      DATA PIPELINE TESTS                 ║"
  IO.println "╚══════════════════════════════════════════╝"
  DataPipelineTests.runAllTests
  IO.println ""

  -- Run optimizer tests
  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║       OPTIMIZER TESTS                    ║"
  IO.println "╚══════════════════════════════════════════╝"
  OptimizerTests.runTests
  IO.println ""

  -- Run gradient check tests
  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║      GRADIENT CHECK TESTS                ║"
  IO.println "╚══════════════════════════════════════════╝"
  GradientCheck.runAllGradientTests
  IO.println ""

  -- Run integration tests
  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║      INTEGRATION TESTS                   ║"
  IO.println "╚══════════════════════════════════════════╝"
  Integration.runAllIntegrationTests
  IO.println ""

  IO.println "=========================================="
  IO.println "✓ All Test Suites Complete"
  IO.println "  - 8 Test Suites Executed"
  IO.println "  - 4 NEW Comprehensive Test Suites"
  IO.println "=========================================="

/-- Quick smoke test for rapid iteration.

Tests basic functionality without running full test suites:
- ReLU activation on positive and negative inputs
- Sigmoid activation at zero (should be ~0.5)
- Tanh activation at zero (should be 0.0)
- Approximate equality function

**Expected runtime:** <1 second

**Usage:** Useful during development to quickly verify core functions work
without waiting for full test suite execution.
-/
def smokeTest : IO Unit := do
  IO.println "=========================================="
  IO.println "VerifiedNN Test Suite - Smoke Test"
  IO.println "=========================================="

  -- Test basic activations
  IO.println "\n=== Activation Functions ==="

  let r1 := VerifiedNN.Core.Activation.relu 5.0
  IO.println s!"relu(5.0) = {r1} (expected 5.0)"

  let r2 := VerifiedNN.Core.Activation.relu (-3.0)
  IO.println s!"relu(-3.0) = {r2} (expected 0.0)"

  let s := VerifiedNN.Core.Activation.sigmoid 0.0
  IO.println s!"sigmoid(0.0) = {s} (expected ~0.5)"

  let t := VerifiedNN.Core.Activation.tanh 0.0
  IO.println s!"tanh(0.0) = {t} (expected 0.0)"

  -- Test approximate equality
  IO.println "\n=== Approximate Equality ==="
  let eq1 := VerifiedNN.Core.approxEq 1.0 1.0
  IO.println s!"approxEq(1.0, 1.0) = {eq1} (expected true)"

  let eq2 := VerifiedNN.Core.approxEq 1.0 2.0
  IO.println s!"approxEq(1.0, 2.0) = {eq2} (expected false)"

  IO.println "\n=========================================="
  IO.println "✓ Smoke Test Complete"
  IO.println "=========================================="

end VerifiedNN.Testing.Runner

-- Main entry point commented out to avoid conflicts with other mains
-- Uncomment when you want to run this as a standalone test runner
-- def main : IO Unit := do
--   VerifiedNN.Testing.Runner.runAllTests
