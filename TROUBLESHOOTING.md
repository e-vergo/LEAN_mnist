# Troubleshooting Guide

Common issues and solutions for the Verified Neural Network project.

## Quick Diagnosis

**Choose your symptom:**

- [Build fails with "library not found for -lblas"](#openblas-not-found)
- [Build fails with mathlib version mismatch](#mathlib-version-mismatch)
- [Executable not found](#executable-not-found)
- [MNIST data not loading](#mnist-data-issues)
- [Interpreter mode errors](#interpreter-mode-issues)
- [Memory issues during build](#memory-problems)
- [Slow build times](#slow-builds)
- [Platform-specific issues](#platform-specific)

---

## Build Issues

### OpenBLAS Not Found

**Symptoms:**
```
ld64.lld: error: library not found for -lblas
error: external command 'gcc' exited with code 1
fatal error: 'cblas.h' file not found
```

**Cause:** OpenBLAS library not installed or not in expected location.

**Solutions:**

**macOS:**
```bash
brew install openblas
```

After installation, verify paths exist:
```bash
# Apple Silicon:
ls /opt/homebrew/opt/openblas/lib/libblas.dylib

# Intel:
ls /usr/local/opt/openblas/lib/libblas.dylib
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install libopenblas-dev
```

**Arch Linux:**
```bash
sudo pacman -S openblas
```

**Verify installation:**
```bash
# Look for cblas.h header
find /usr -name "cblas.h" 2>/dev/null
```

**Still not working?**

Check lakefile.lean has correct paths (lines 38-79). For macOS Apple Silicon, should include:
```lean
moreLinkArgs := #["-L/opt/homebrew/opt/openblas/lib", "-lopenblas"]
```

---

### Mathlib Version Mismatch

**Symptoms:**
```
warning: mathlib: repository has local changes
Dependency Mathlib uses a different lean-toolchain
  Project uses leanprover/lean4:v4.20.1
  Mathlib uses leanprover/lean4:v4.24.0
```

**Cause:** Corrupted .lake directory or interrupted build.

**Solution 1 - Clean rebuild:**
```bash
rm -rf .lake
lake build
```

**Solution 2 - Reset mathlib:**
```bash
cd .lake/packages/mathlib
git reset --hard
cd ../../..
lake build
```

**Solution 3 - Nuclear option:**
```bash
rm -rf .lake
lake update
lake build
```

**Prevention:** Don't interrupt builds with Ctrl+C during dependency download phase.

---

### Executable Not Found

**Symptoms:**
```bash
$ lake exe renderMNIST
error: executable 'renderMNIST' not found
```

**Cause:** Project not built, or build failed.

**Solution:**
```bash
# Rebuild project
lake build

# Check build succeeded (should show "Build completed successfully")
echo $?  # Should print 0

# List available executables
lake build --help | grep "exe"
```

**Available executables:**
- `simpleExample` - Simple training demo
- `mnistTrain` - Full MNIST training
- `renderMNIST` - ASCII visualization
- `mnistLoadTest` - Data loading test
- `trainManual` - Manual training example
- `fullIntegration` - Integration tests
- `smokeTest` - Quick smoke test
- `gradientCheck` - Gradient validation
- `checkDataDistribution` - Data distribution analysis

---

## MNIST Data Issues

### Data Not Downloading

**Symptoms:**
```bash
$ ./scripts/download_mnist.sh
Error: failed to download file
```

**Solutions:**

**Option 1 - Manual download:**
```bash
mkdir -p data
cd data
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/test-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/test-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

**Option 2 - Use curl instead of wget:**

Edit `scripts/download_mnist.sh`, replace `wget` with `curl -O`.

**Option 3 - Alternative mirror:**

Original MNIST site (if available): http://yann.lecun.com/exdb/mnist/

---

### Data Files Exist But Not Loading

**Symptoms:**
```bash
$ lake exe mnistLoadTest
Error: failed to load MNIST data
```

**Checks:**

```bash
# Verify files exist and are uncompressed
ls -lh data/
# Should show 4 files WITHOUT .gz extension

# Check file sizes (approximate):
# train-images-idx3-ubyte: ~47MB
# train-labels-idx1-ubyte: ~60KB
# test-images-idx3-ubyte: ~7.8MB
# test-labels-idx1-ubyte: ~10KB
```

**If files are still compressed (.gz):**
```bash
cd data
gunzip *.gz
```

**If files are corrupted:**
```bash
rm data/*
./scripts/download_mnist.sh
```

---

## Interpreter Mode Issues

### "noncomputable" Error

**Symptoms:**
```
error: failed to compile executable, consider marking it `noncomputable`
```

**Cause:** Trying to compile code that uses `∇` (gradient operator).

**Solution:** Use interpreter mode:
```bash
# Wrong:
lake exe mnistTrain

# Correct:
lake env lean --run VerifiedNN/Examples/MNISTTrain.lean
```

**Why?** SciLean's automatic differentiation uses metaprogramming that only works in interpreter mode. This is normal and documented.

---

### Interpreter Hangs or Crashes

**Symptoms:**
- Process hangs indefinitely
- Segmentation fault
- Out of memory error

**Solutions:**

**1. Reduce batch size:**

Edit the example file, reduce batch size from 32 to 8 or 16.

**2. Reduce training data:**

Use a subset of MNIST (e.g., first 1000 samples).

**3. Check available memory:**
```bash
# macOS:
vm_stat

# Linux:
free -h
```

Training MNIST needs ~2-4GB RAM.

**4. Close other applications** to free memory.

---

## Memory Problems

### Build Runs Out of Memory

**Symptoms:**
```
clang: error: unable to execute command: Killed
error: build failed
```

**Cause:** Parallel compilation using too much RAM.

**Solutions:**

**1. Reduce parallel jobs:**
```bash
lake build -j 2  # Use only 2 parallel jobs
```

**2. Download precompiled mathlib:**
```bash
lake exe cache get
```

This avoids compiling mathlib (~500MB download, saves hours).

**3. Increase swap space (Linux):**
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Minimum requirements:**
- 4GB RAM + 4GB swap for building
- 8GB RAM recommended

---

### Runtime Out of Memory

**Symptoms:**
```
malloc: can't allocate region
Segmentation fault
```

**During training:**

**1. Reduce batch size** (edit example file)
**2. Reduce training set size** (use subset)
**3. Close other applications**

**During ASCII rendering:**

The renderer uses significant memory. Try:
```bash
lake exe renderMNIST --count 3  # Reduce from 5 to 3
```

---

## Slow Builds

### First Build Takes Forever

**Normal:** First build takes 5-10 minutes (compiles ~3000 modules).

**Speed it up:**

**1. Download precompiled mathlib:**
```bash
lake exe cache get
```

**2. Use multiple cores:**
```bash
lake build -j 8  # Use 8 parallel jobs (adjust for your CPU)
```

**3. Don't interrupt the build** - starting over wastes time.

---

### Incremental Builds Slow

**Symptoms:** Small changes trigger large rebuilds.

**Solutions:**

**1. Make targeted builds:**
```bash
# Build specific module instead of entire project
lake build VerifiedNN.Examples.SimpleExample
```

**2. Avoid touching core files** (Core/, Network/Gradient.lean) - these have many dependents.

**3. Restart Lean LSP servers:**
```bash
pkill -f "lean --server"
```

Multiple Lean servers can slow things down.

---

## Platform-Specific Issues

### macOS Apple Silicon (M1/M2/M3)

**OpenBLAS paths:** Use `/opt/homebrew/opt/openblas/` (not `/usr/local/`)

**Rosetta warning:** Don't use Rosetta - native ARM build is faster.

**Xcode Command Line Tools:** May be required:
```bash
xcode-select --install
```

---

### macOS Intel

**OpenBLAS paths:** Use `/usr/local/opt/openblas/`

**Homebrew location:** Ensure Homebrew is in `/usr/local/` not `/opt/homebrew/`

---

### Ubuntu/Debian

**Missing build tools:**
```bash
sudo apt install build-essential curl git
```

**OpenBLAS package:** `libopenblas-dev` (not `libopenblas-base`)

---

### Arch Linux

**OpenBLAS:** Install `openblas` not `blas`

**Lean installation:** Use elan, not system packages

---

### Windows

**Status:** Not supported.

**Reason:** SciLean doesn't support Windows.

**Workaround:** Use WSL2 (Windows Subsystem for Linux):

```bash
# Inside WSL2 Ubuntu:
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
sudo apt install libopenblas-dev
# Then follow normal Linux setup
```

---

## LSP and Editor Issues

### Multiple Lean Servers Consuming Resources

**Symptoms:**
- High CPU/memory usage
- System slowdown
- Editor becomes unresponsive

**Check running processes:**
```bash
pgrep -af lean
```

**Kill all Lean language servers:**
```bash
pkill -f "lean --server"
```

**Kill all Lake processes:**
```bash
pkill -f lake
```

**When to restart:**
- After major code changes affecting many files
- When diagnostics become stale or incorrect
- If LSP becomes unresponsive or slow
- After `lake update` or dependency changes
- When memory usage grows excessively

**Prevention:**
- Build project before starting editor (`lake build`)
- Use `lake exe cache get` to download precompiled mathlib
- Work on one module at a time
- Close unused Lean files in editor
- Restart LSP periodically during long sessions

---

## Verification and Proof Issues

### "unknown identifier" in Proofs

**Symptoms:**
```
error: unknown identifier 'Continuous.comp'
```

**Cause:** Missing import or mathlib lemma not in scope.

**Solutions:**

**1. Search for the theorem:**
```bash
# Use MCP tools (if available):
lean_leansearch "composition of continuous functions"

# Or search locally:
rg "Continuous.comp" .lake/packages/mathlib/
```

**2. Add missing import:**

Common imports for analysis proofs:
```lean
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Topology.Basic
```

**3. Check mathlib documentation:**

Visit https://leanprover-community.github.io/mathlib4_docs/

---

### Tactic Timeout

**Symptoms:**
```
error: tactic 'simp' failed, timeout
```

**Cause:** Simp trying too many lemmas, or complex goal.

**Solutions:**

**1. Use targeted simp:**
```lean
-- Instead of:
simp

-- Use:
simp only [lemma1, lemma2, lemma3]
```

**2. Increase timeout:**
```lean
set_option maxHeartbeats 400000  -- Default is 200000
```

**3. Break into smaller steps:**
```lean
-- Instead of one big simp:
simp [part1]
simp [part2]
simp [part3]
```

---

### "failed to synthesize instance"

**Symptoms:**
```
error: failed to synthesize instance
  Differentiable ℝ myFunction
```

**Cause:** Missing type class instance (Differentiable, Continuous, etc.)

**Solutions:**

**1. Register the instance:**
```lean
@[fun_prop]
theorem myFunction_differentiable : Differentiable ℝ myFunction := by
  unfold myFunction
  fun_prop
```

**2. Provide instance manually:**
```lean
have h : Differentiable ℝ f := by fun_prop
exact some_theorem h
```

**3. Check composition chain:**

If `f ∘ g`, need both `Differentiable ℝ f` and `Differentiable ℝ g`.

---

### "sorry" Won't Compile

**Symptoms:**
```
error: 'sorry' is not allowed in computable code
```

**Cause:** Using `sorry` in executable code path.

**Solutions:**

**1. Mark definition noncomputable:**
```lean
noncomputable def myFunction := ...
```

**2. Use axiom instead:**
```lean
axiom myFunction_placeholder : ...
```

**3. Complete the proof** (preferred).

---

## Data Loading Edge Cases

### Partial MNIST Dataset

**Symptoms:**

Only seeing a subset of images (e.g., 1000 instead of 60000).

**Cause:** Data loader truncating dataset.

**Check:**
```lean
-- In VerifiedNN/Data/MNIST.lean
#eval do
  let data ← loadMNIST "data"
  IO.println s!"Training samples: {data.trainImages.size}"
  -- Should print 60000
```

**Fix:** Verify data files are complete (see "Data Files Exist But Not Loading").

---

### Label Mismatch

**Symptoms:**

Training accuracy stuck at ~10% (random guessing).

**Cause:** Image/label arrays out of sync.

**Debug:**
```lean
-- Check first few labels match expected
lake exe renderMNIST --count 5
-- Manually verify digits shown match labels printed
```

**Fix:** Re-download MNIST data.

---

## Performance Issues

### Training Too Slow

**Expected performance:**
- Simple example: <1 minute for 10 epochs
- MNIST: 5-15 minutes for 10 epochs (CPU-only)

**If much slower:**

**1. Verify OpenBLAS is used:**
```bash
# Build should show OpenBLAS linking:
lake build 2>&1 | grep -i blas
```

**2. Reduce batch size** (increases batches per epoch, may be slower overall, but faster feedback).

**3. Use subset of data** for testing:
```lean
-- Edit training file to use first 1000 samples
let trainData := data.trainImages[0:1000]
```

**4. Profile the code:**
```lean
set_option profiler true
#eval timeit "forward pass" do ...
```

---

### Gradient Computation Hangs

**Symptoms:**

Process hangs when computing gradients with `∇`.

**Cause:** Complex gradient expression that SciLean can't simplify.

**Debug:**
```lean
set_option trace.Meta.Tactic.simp true
set_option trace.fun_trans true

-- Try computing gradient on simplified function
```

**Workaround:** Simplify network architecture or use manual gradient.

---

## Testing Issues

### Tests Not Running

**Symptoms:**
```bash
$ lake build VerifiedNN.Testing.UnitTests
# Builds but no output
```

**Cause:** Need to use interpreter to run tests:
```bash
# Wrong:
lake build VerifiedNN.Testing.UnitTests

# Correct:
lake env lean --run VerifiedNN/Testing/UnitTests.lean
```

---

### Gradient Check Fails

**Symptoms:**
```
Gradient check FAILED
Max relative error: 0.15
```

**Normal threshold:** Relative error < 0.01 is good, < 0.001 is excellent.

**Causes:**

**1. Numerical precision:** Float arithmetic introduces errors.
**2. Discontinuous activation:** ReLU has non-differentiable point at 0.
**3. Finite difference epsilon too large:** Try smaller epsilon.

**Not a problem if:** Relative error < 0.05 and network trains successfully.

**Fix if error > 0.1:** Check gradient implementation for bugs.

---

## Still Stuck?

### Collect Diagnostic Information

```bash
# System info
uname -a
lean --version
lake --version

# Build log
lake clean
lake build 2>&1 | tee build-log.txt

# Check Lean processes
pgrep -af lean

# Check file sizes
ls -lh data/
```

### Verify Lean Toolchain

```bash
lean --version
# Should show: Lean (version 4.20.1, ...)
```

If wrong version:
```bash
elan toolchain list
elan default leanprover/lean4:v4.20.1
```

### Clean Everything and Start Over

```bash
# Remove build artifacts
rm -rf .lake build

# Update dependencies
lake update

# Download precompiled mathlib
lake exe cache get

# Rebuild
lake build
```

### Get Help

- **Documentation:** [README.md](README.md), [CLAUDE.md](CLAUDE.md)
- **Lean Zulip:** https://leanprover.zulipchat.com/
  - #new members - General help
  - #scientific-computing - SciLean and numerical computing
  - #mathlib4 - Mathlib-related questions
- **SciLean issues:** https://github.com/lecopivo/SciLean/issues
- **This project:** File an issue at your repository

---

## Most Common Solutions

**90% of problems are solved by:**

1. **Installing OpenBLAS correctly** for your platform
2. **Using interpreter mode** (`lake env lean --run`) for training
3. **Downloading MNIST data** before running examples
4. **Using `lake exe cache get`** to avoid compiling mathlib
5. **Restarting Lean LSP servers** when diagnostics are stale

---

**Quick Reference Commands:**

```bash
# Setup
brew install openblas              # macOS
sudo apt install libopenblas-dev   # Ubuntu/Debian
lake exe cache get                 # Download mathlib

# Build
lake build                         # Full build
lake build ModuleName              # Single module

# Run
lake exe executableName            # Compiled mode
lake env lean --run File.lean      # Interpreter mode

# Debug
pkill -f "lean --server"          # Restart LSP
rm -rf .lake && lake build         # Clean rebuild
lake build 2>&1 | tee build.log    # Capture build log

# Data
./scripts/download_mnist.sh        # Get MNIST
ls -lh data/                       # Verify files
```
