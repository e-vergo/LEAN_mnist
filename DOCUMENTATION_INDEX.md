# Documentation Index

**Master reference for all documentation in the Verified Neural Network Training project.**

---

## Quick Navigation by Experience Level

### First-Time Visitors (Start Here!)

**Never used Lean or formal verification before?**

1. **[START_HERE.md](START_HERE.md)** - 5-minute project overview
   - What this project achieves
   - Why it matters
   - Quick visual examples
   - Next steps based on your interests

2. **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete setup guide
   - Prerequisites and installation
   - Building the project
   - Running your first example
   - Troubleshooting common issues
   - What to explore next

### Developers & Contributors

**Ready to contribute code or proofs?**

3. **[CLAUDE.md](CLAUDE.md)** - Development guide and standards
   - Lean 4 coding conventions
   - MCP tools for AI-assisted development
   - Repository cleanup standards
   - Development workflow
   - Build commands and testing

4. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design documentation
   - Module dependency graphs
   - Call flow diagrams
   - Design decisions and rationale
   - Where to find specific functionality

5. **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Complete testing handbook
   - Testing philosophy and strategy
   - Test types (unit, integration, gradient checks)
   - Writing and running tests
   - Debugging test failures
   - CI/CD integration

6. **[COOKBOOK.md](COOKBOOK.md)** - Practical recipes and examples
   - Common tasks with copy-paste code
   - Working with the network
   - Running training experiments
   - Visualization and debugging
   - Performance optimization

### Researchers & Verification Specialists

**Interested in the formal verification aspects?**

7. **[VERIFICATION_WORKFLOW.md](VERIFICATION_WORKFLOW.md)** - Proof development guide
   - Step-by-step proof methodology
   - Using MCP tools for proofs
   - Common proof patterns
   - Verification priorities
   - Documentation standards for proofs

8. **[verified-nn-spec.md](verified-nn-spec.md)** - Complete technical specification
   - Formal problem statement
   - Mathematical foundations
   - Verification goals and scope
   - Axiom justifications
   - Implementation details

---

## Documentation by Category

### Core Project Documentation

| Document | Purpose | Audience | Size |
|----------|---------|----------|------|
| **[README.md](README.md)** | Project overview, achievements, axiom catalog, transparency | Everyone | 30KB |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design, module structure, dependencies | Developers | 37KB |
| **[verified-nn-spec.md](verified-nn-spec.md)** | Formal technical specification | Researchers | 33KB |
| **[CLAUDE.md](CLAUDE.md)** | Development guide, MCP integration, standards | Contributors | 28KB |

### Getting Started Guides

| Document | Purpose | Audience | Size |
|----------|---------|----------|------|
| **[START_HERE.md](START_HERE.md)** | Quick 5-minute overview | First-time visitors | 6KB |
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Comprehensive onboarding with setup | New users | 14KB |

### Practical Handbooks

| Document | Purpose | Audience | Size |
|----------|---------|----------|------|
| **[TESTING_GUIDE.md](TESTING_GUIDE.md)** | Testing strategies and best practices | Developers | 22KB |
| **[COOKBOOK.md](COOKBOOK.md)** | Copy-paste recipes for common tasks | Developers | 21KB |
| **[VERIFICATION_WORKFLOW.md](VERIFICATION_WORKFLOW.md)** | Step-by-step proof development | Verification specialists | 15KB |

### Enhancement & Research Reports

| Document | Purpose | Audience | Size |
|----------|---------|----------|------|
| **[DOCUMENTATION_ENHANCEMENT_REPORT.md](DOCUMENTATION_ENHANCEMENT_REPORT.md)** | Documentation quality improvements | Maintainers | 18KB |
| **[BUILD_STATUS_REPORT.md](BUILD_STATUS_REPORT.md)** | Build status and compilation metrics | Maintainers | Varies |
| **[AD_REGISTRATION_SUMMARY.md](AD_REGISTRATION_SUMMARY.md)** | SciLean AD registration overview | Advanced developers | 12KB |
| **[AD_REGISTRATION_RESEARCH_REPORT.md](AD_REGISTRATION_RESEARCH_REPORT.md)** | Detailed AD registration research | Advanced developers | 25KB |
| **[RESEARCH_DELIVERY_REPORT.md](RESEARCH_DELIVERY_REPORT.md)** | Research milestone summary | Project stakeholders | 10KB |

### Module-Specific Documentation

Each `VerifiedNN/` subdirectory contains comprehensive documentation (~10KB each):

| Directory | README | Purpose |
|-----------|--------|---------|
| **[Core/](VerifiedNN/Core/)** | [README.md](VerifiedNN/Core/README.md) | Foundation types, linear algebra, activations |
| **[Data/](VerifiedNN/Data/)** | [README.md](VerifiedNN/Data/README.md) | MNIST loading and preprocessing |
| **[Examples/](VerifiedNN/Examples/)** | [README.md](VerifiedNN/Examples/README.md) | Minimal examples and full MNIST training |
| **[Layer/](VerifiedNN/Layer/)** | [README.md](VerifiedNN/Layer/README.md) | Dense layers with differentiability proofs |
| **[Loss/](VerifiedNN/Loss/)** | [README.md](VerifiedNN/Loss/README.md) | Cross-entropy with mathematical properties |
| **[Network/](VerifiedNN/Network/)** | [README.md](VerifiedNN/Network/README.md) | MLP architecture, initialization, gradients |
| **[Optimizer/](VerifiedNN/Optimizer/)** | [README.md](VerifiedNN/Optimizer/README.md) | SGD implementation |
| **[Testing/](VerifiedNN/Testing/)** | [README.md](VerifiedNN/Testing/README.md) | Unit tests, integration tests |
| **[Training/](VerifiedNN/Training/)** | [README.md](VerifiedNN/Training/README.md) | Training loop, batching, metrics |
| **[Verification/](VerifiedNN/Verification/)** | [README.md](VerifiedNN/Verification/README.md) | Formal proofs (gradient correctness, type safety) |

**Status:** 10/10 subdirectories have complete READMEs

---

## Documentation by Task

### "I want to understand what this project is about"

1. [START_HERE.md](START_HERE.md) - Quick overview
2. [README.md](README.md) - Detailed achievements and status
3. [ARCHITECTURE.md](ARCHITECTURE.md) - System design

### "I want to build and run the project"

1. [GETTING_STARTED.md](GETTING_STARTED.md) - Installation and setup
2. [COOKBOOK.md](COOKBOOK.md) - Running examples
3. [TESTING_GUIDE.md](TESTING_GUIDE.md) - Running tests

### "I want to contribute code"

1. [CLAUDE.md](CLAUDE.md) - Coding standards
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Where to add features
3. [TESTING_GUIDE.md](TESTING_GUIDE.md) - Writing tests

### "I want to develop formal proofs"

1. [VERIFICATION_WORKFLOW.md](VERIFICATION_WORKFLOW.md) - Proof methodology
2. [verified-nn-spec.md](verified-nn-spec.md) - What needs proving
3. [CLAUDE.md](CLAUDE.md) - MCP tools for proof development
4. [VerifiedNN/Verification/README.md](VerifiedNN/Verification/README.md) - Existing proofs

### "I want to understand a specific module"

Navigate to the relevant subdirectory README:
- **Core functionality**: [Core/README.md](VerifiedNN/Core/README.md)
- **Data loading**: [Data/README.md](VerifiedNN/Data/README.md)
- **Neural network layers**: [Layer/README.md](VerifiedNN/Layer/README.md)
- **Network architecture**: [Network/README.md](VerifiedNN/Network/README.md)
- **Loss functions**: [Loss/README.md](VerifiedNN/Loss/README.md)
- **Optimization**: [Optimizer/README.md](VerifiedNN/Optimizer/README.md)
- **Training infrastructure**: [Training/README.md](VerifiedNN/Training/README.md)
- **Verification proofs**: [Verification/README.md](VerifiedNN/Verification/README.md)
- **Testing**: [Testing/README.md](VerifiedNN/Testing/README.md)
- **Example usage**: [Examples/README.md](VerifiedNN/Examples/README.md)

---

## Documentation Quality Metrics

| Metric | Status |
|--------|--------|
| **Total Documentation Files** | 27+ markdown files |
| **Total Documentation Size** | ~350KB |
| **Module READMEs** | 10/10 complete |
| **Code Documentation** | 100% of public APIs |
| **Proof Documentation** | All sorries/axioms justified |
| **Last Major Update** | October 2025 |
| **Documentation Standard** | Mathlib submission quality |

---

## Maintenance

**Last Updated:** October 22, 2025

**Update Frequency:** Documentation is updated continuously as the codebase evolves.

**Contributing:** When adding new features or modules:
1. Update relevant module README
2. Add examples to COOKBOOK.md if applicable
3. Update ARCHITECTURE.md if structure changes
4. Add test documentation to TESTING_GUIDE.md
5. Update this index if adding major new documentation

**Issues:** Report documentation gaps or errors via GitHub issues or update directly via PR.

---

## External Resources

### Lean 4 Documentation
- Official docs: https://lean-lang.org/documentation/
- Theorem Proving in Lean 4: https://leanprover.github.io/theorem_proving_in_lean4/
- Mathlib4 docs: https://leanprover-community.github.io/mathlib4_docs/

### SciLean (Automatic Differentiation)
- Repository: https://github.com/lecopivo/SciLean
- Documentation: https://lecopivo.github.io/scientific-computing-lean/

### Community
- Lean Zulip chat: https://leanprover.zulipchat.com/
  - #scientific-computing - SciLean questions
  - #mathlib4 - Proof development
  - #new members - Getting started

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Oct 22, 2025 | Initial creation of documentation index |
| 0.9 | Oct 21, 2025 | Major documentation enhancement (GETTING_STARTED, ARCHITECTURE, etc.) |
| 0.8 | Oct 21, 2025 | Repository cleanup to mathlib submission quality |

---

**Navigation Tip:** Use your browser's search (Ctrl+F / Cmd+F) to quickly find relevant documentation by keyword.
