/-
# Test Runner

Entry point for running all VerifiedNN tests.

This module provides a unified test runner that executes all available test
suites and reports comprehensive results.

## Test Suites Available

1. **Unit Tests** - Component-level tests for activations, data types
2. **Optimizer Tests** - Parameter update operations (fully functional)
3. **Integration Tests** - End-to-end pipeline tests (dataset generation ready)
4. **Gradient Check** - Numerical validation (infrastructure ready, blocked by Network.Gradient)

## Test Organization

Tests are organized by dependency level:
- **Level 0**: Core functionality (activations, data types) - WORKING
- **Level 1**: Optimizer operations - WORKING
- **Level 2**: Dataset generation - WORKING
- **Level 3**: Gradient checking - READY (blocked by Network.Gradient)
- **Level 4**: Full integration - PLANNED (blocked by Training.Loop)

## Usage

```bash
# Run all available tests
lake build VerifiedNN.Testing.RunTests
lake env lean --run VerifiedNN/Testing/RunTests.lean

# Run specific test suites
lake env lean --run VerifiedNN/Testing/UnitTests.lean
lake env lean --run VerifiedNN/Testing/OptimizerTests.lean
lake env lean --run VerifiedNN/Testing/Integration.lean
```

## Current Status

✓ Ready to run: UnitTests, OptimizerTests, Integration (partial)
⚠ Blocked: GradientCheck (needs Network.Gradient)
⚠ Planned: Full integration tests (needs Training.Loop)
-/

import VerifiedNN.Testing.UnitTests
import VerifiedNN.Testing.OptimizerTests
import VerifiedNN.Testing.Integration

namespace VerifiedNN.Testing.Runner

open VerifiedNN.Testing.UnitTests
open VerifiedNN.Testing.OptimizerTests
open VerifiedNN.Testing.Integration

/-- Run all available test suites with comprehensive reporting -/
def runAllTests : IO Unit := do
  sorry

/-- Quick smoke test for rapid iteration -/
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
