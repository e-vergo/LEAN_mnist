import Lake
open Lake DSL

package «verifiedNN» where
  version := v!"0.1.0"
  keywords := #["machine learning", "neural networks", "formal verification"]
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`pp.proofs.withType, false⟩,
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]
  -- Build with optimizations enabled
  moreServerArgs := #[
    "-Dpp.unicode.fun=true",
    "-DautoImplicit=false"
  ]
  -- Optimization flags commented out for Lean 4.23.0 compatibility
  -- moreLeanArgs := #["-O3"]
  moreLeancArgs := #["-O3", "-march=native"]

require scilean from git
  "https://github.com/lecopivo/SciLean.git" @ "master"

-- LSpec temporarily disabled due to Lean version incompatibility
-- Re-enable when writing tests or when SciLean updates to newer Lean version
-- require LSpec from git
--   "https://github.com/argumentcomputer/LSpec.git" @ "main"

@[default_target]
lean_lib «VerifiedNN» where
  globs := #[.submodules `VerifiedNN]

-- Example executables
lean_exe simpleExample where
  root := `VerifiedNN.Examples.SimpleExample
  supportInterpreter := true

lean_exe mnistTrain where
  root := `VerifiedNN.Examples.MNISTTrain
  supportInterpreter := true
