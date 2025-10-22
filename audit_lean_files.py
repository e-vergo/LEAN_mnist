#!/usr/bin/env python3
"""
Comprehensive audit of Lean 4 files for verification status and proof obligations.

Updated: 2025-10-22
- Path portability (auto-detects project root)
- Multi-line comment detection
- Lean 4 specific patterns (noncomputable, unsafe, Classical, extern)
- Module-level reporting
- Axiom documentation quality analysis
- Theorem completion metrics
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import argparse

# Auto-detect paths
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR / 'VerifiedNN'
OUTPUT_DIR = SCRIPT_DIR

# Keywords to search for
KEYWORDS = {
    'axiom': r'\baxiom\b',
    'sorry': r'\bsorry\b',
    'noncomputable': r'\bnoncomputable\b',
    'unsafe': r'\bunsafe\b',
    'classical': r'Classical\.',
    'extern': r'@\[extern\]',
    'implemented_by': r'@\[implemented_by\]',
}

# Patterns to detect declarations
PATTERNS = {
    'axiom_decl': r'^axiom\s+\w+',
    'theorem_decl': r'^theorem\s+\w+',
    'def_decl': r'^def\s+\w+',
    'theorem_sorry': r'^theorem.*:=.*sorry',
    'def_sorry': r'^def.*:=.*sorry',
    'noncomputable_def': r'noncomputable\s+def',
    'unsafe_def': r'unsafe\s+def',
}

class CommentTracker:
    """Track multi-line comment state across lines."""

    def __init__(self):
        self.in_multiline = False
        self.multiline_type = None  # 'doc' for /-! or 'normal' for /-

    def update(self, line):
        """Update comment state and return if current line is a comment."""
        stripped = line.strip()

        # Check for multi-line comment start
        if '/-!' in line or '/-' in line:
            self.in_multiline = True
            self.multiline_type = 'doc' if '/-!' in line else 'normal'
            # Check if it closes on same line
            if '-/' in line:
                self.in_multiline = False
            return True

        # Check for multi-line comment end
        if self.in_multiline and '-/' in line:
            self.in_multiline = False
            return True

        # If in multi-line comment
        if self.in_multiline:
            return True

        # Check for single-line comment
        if stripped.startswith('--'):
            return True

        # Check for inline comment (not at start)
        if '--' in line:
            # Find the position of --
            comment_pos = line.find('--')
            code_before = line[:comment_pos].strip()
            # If there's code before the comment, it's inline
            return len(code_before) == 0  # Only comment if no code before

        return False

    def is_in_comment(self):
        """Check if currently inside a multi-line comment."""
        return self.in_multiline

def find_lean_files(root_dir):
    """Find all .lean files in the directory tree."""
    lean_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.lean'):
                lean_files.append(Path(root) / file)
    return sorted(lean_files)

def get_module_name(filepath, root_dir):
    """Get module name from filepath."""
    rel_path = filepath.relative_to(root_dir)
    # Get the first directory name (Core, Layer, Network, etc.)
    parts = rel_path.parts
    if len(parts) > 0:
        return parts[0]
    return 'Root'

def analyze_axiom_documentation(lines, axiom_line_num):
    """Check if axiom has comprehensive documentation (20+ lines)."""
    # Look backwards from axiom line for docstring
    # axiom_line_num is 0-indexed position in lines array
    i = axiom_line_num - 1  # Start at line immediately before axiom

    # Skip the closing -/ if present
    if i >= 0 and lines[i].strip() == '-/':
        doc_end_line = i
        i -= 1
    else:
        # No closing -/, assume no docstring
        return {'has_docstring': False, 'doc_lines': 0, 'is_comprehensive': False}

    # Now search backwards for the opening /-- or /-!
    while i >= 0:
        line = lines[i].strip()
        if line.startswith('/--') or line.startswith('/-!'):
            # Found start of docstring - count lines from here to closing -/
            doc_lines = doc_end_line - i + 1  # +1 to include both start and end
            return {
                'has_docstring': True,
                'doc_lines': doc_lines,
                'is_comprehensive': doc_lines >= 20
            }
        i -= 1

    # Reached start of file without finding opening
    return {'has_docstring': False, 'doc_lines': 0, 'is_comprehensive': False}

def search_file(filepath, keywords, patterns, root_dir):
    """Search a file for keywords and patterns with proper comment tracking."""
    results = {
        'keywords': defaultdict(list),
        'patterns': defaultdict(list),
        'stats': defaultdict(int),
        'module': get_module_name(filepath, root_dir)
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    comment_tracker = CommentTracker()

    for line_num, line in enumerate(lines, 1):
        is_comment = comment_tracker.update(line)

        # Check keywords
        for keyword, regex in keywords.items():
            if re.search(regex, line):
                results['keywords'][keyword].append({
                    'line': line_num,
                    'text': line.strip(),
                    'in_comment': is_comment
                })
                results['stats'][keyword] += 1
                if is_comment:
                    results['stats'][f'{keyword}_comment'] += 1
                else:
                    results['stats'][f'{keyword}_code'] += 1

        # Check patterns (only in code, not comments)
        if not is_comment:
            for pattern_name, regex in patterns.items():
                if re.match(regex, line.strip()):
                    match_info = {
                        'line': line_num,
                        'text': line.strip()
                    }

                    # For axioms, analyze documentation
                    if pattern_name == 'axiom_decl':
                        doc_info = analyze_axiom_documentation(lines, line_num - 1)
                        match_info.update(doc_info)

                    results['patterns'][pattern_name].append(match_info)
                    results['stats'][f'pattern_{pattern_name}'] += 1

    return results

def group_by_module(all_results):
    """Group results by module (directory)."""
    modules = defaultdict(lambda: {
        'files': [],
        'axioms': 0,
        'axioms_undocumented': 0,
        'sorries_code': 0,
        'sorries_comment': 0,
        'theorems': 0,
        'theorems_proven': 0,
        'theorems_sorry': 0,
        'noncomputable': 0,
        'unsafe': 0,
    })

    for filepath, results in all_results.items():
        if not results:
            continue

        module = results['module']
        modules[module]['files'].append(filepath.name)

        # Count stats
        modules[module]['axioms'] += results['stats'].get('pattern_axiom_decl', 0)
        modules[module]['sorries_code'] += results['stats'].get('sorry_code', 0)
        modules[module]['sorries_comment'] += results['stats'].get('sorry_comment', 0)
        modules[module]['theorems'] += results['stats'].get('pattern_theorem_decl', 0)
        modules[module]['theorems_sorry'] += results['stats'].get('pattern_theorem_sorry', 0)
        modules[module]['noncomputable'] += results['stats'].get('noncomputable_code', 0)
        modules[module]['unsafe'] += results['stats'].get('unsafe_code', 0)

        # Count undocumented axioms
        for axiom in results['patterns'].get('axiom_decl', []):
            if not axiom.get('is_comprehensive', False):
                modules[module]['axioms_undocumented'] += 1

        # Calculate proven theorems
        modules[module]['theorems_proven'] = (
            modules[module]['theorems'] - modules[module]['theorems_sorry']
        )

    # Calculate completion percentages
    for module_data in modules.values():
        total = module_data['theorems']
        if total > 0:
            module_data['completion'] = 100 * module_data['theorems_proven'] / total
        else:
            module_data['completion'] = 100.0

    return dict(modules)

def print_results(all_results, module_stats, output_format='text'):
    """Print formatted results."""
    if output_format == 'json':
        print_json_results(all_results, module_stats)
        return

    print("=" * 80)
    print("LEAN 4 VERIFICATION AUDIT REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {ROOT_DIR}")
    print()

    # Overall statistics
    total_files = len([r for r in all_results.values() if r])
    total_axioms = sum(m['axioms'] for m in module_stats.values())
    total_axioms_undoc = sum(m['axioms_undocumented'] for m in module_stats.values())
    total_sorries_code = sum(m['sorries_code'] for m in module_stats.values())
    total_sorries_comment = sum(m['sorries_comment'] for m in module_stats.values())
    total_theorems = sum(m['theorems'] for m in module_stats.values())
    total_theorems_proven = sum(m['theorems_proven'] for m in module_stats.values())
    total_noncomp = sum(m['noncomputable'] for m in module_stats.values())
    total_unsafe = sum(m['unsafe'] for m in module_stats.values())

    print("📊 EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"Files scanned: {total_files}")
    print(f"Modules: {len(module_stats)}")
    print()

    print(f"Axiom Declarations: {total_axioms}")
    if total_axioms_undoc > 0:
        print(f"  ⚠️  Undocumented (<20 lines): {total_axioms_undoc}")
    else:
        print(f"  ✅ All documented (≥20 lines)")
    print()

    print(f"Sorry Statements:")
    print(f"  Executable (in code): {total_sorries_code}")
    print(f"  Documentation (in comments): {total_sorries_comment}")
    print()

    if total_theorems > 0:
        completion = 100 * total_theorems_proven / total_theorems
        print(f"Theorem Completion: {total_theorems_proven}/{total_theorems} ({completion:.1f}%)")
        if completion == 100:
            print("  ✅ All theorems proven!")
        else:
            print(f"  ⚠️  {total_theorems - total_theorems_proven} theorems with sorry")
    print()

    print(f"Verification Flags:")
    print(f"  Noncomputable: {total_noncomp}")
    print(f"  Unsafe: {total_unsafe}")
    print()

    # Module breakdown
    print("=" * 80)
    print("📁 MODULE BREAKDOWN")
    print("=" * 80)
    print()

    for module_name in sorted(module_stats.keys()):
        m = module_stats[module_name]
        status = "✅" if m['sorries_code'] == 0 and m['axioms_undocumented'] == 0 else "⚠️"

        print(f"{status} {module_name}/")
        print(f"   Files: {len(m['files'])}")
        print(f"   Axioms: {m['axioms']}", end="")
        if m['axioms_undocumented'] > 0:
            print(f" ({m['axioms_undocumented']} undocumented)")
        else:
            print()
        print(f"   Sorries: {m['sorries_code']} (code), {m['sorries_comment']} (docs)")
        if m['theorems'] > 0:
            print(f"   Theorems: {m['theorems_proven']}/{m['theorems']} proven ({m['completion']:.0f}%)")
        if m['noncomputable'] > 0:
            print(f"   Noncomputable: {m['noncomputable']}")
        if m['unsafe'] > 0:
            print(f"   Unsafe: {m['unsafe']}")
        print()

    # Detailed axiom analysis
    print("=" * 80)
    print("🔍 AXIOM ANALYSIS")
    print("=" * 80)
    print()

    axiom_count = 0
    for filepath, results in sorted(all_results.items()):
        if not results:
            continue

        axioms = results['patterns'].get('axiom_decl', [])
        if axioms:
            rel_path = filepath.relative_to(ROOT_DIR)
            print(f"\n{rel_path}:")
            for axiom in axioms:
                axiom_count += 1
                doc_status = "✅" if axiom.get('is_comprehensive') else "⚠️"
                doc_lines = axiom.get('doc_lines', 0)
                print(f"  {doc_status} Line {axiom['line']}: {axiom['text'][:70]}")
                print(f"      Documentation: {doc_lines} lines", end="")
                if not axiom.get('is_comprehensive'):
                    print(" (< 20 lines recommended)")
                else:
                    print()

    if axiom_count == 0:
        print("No axioms found.")

    # Detailed sorry analysis
    print("\n" + "=" * 80)
    print("🔍 EXECUTABLE SORRY ANALYSIS")
    print("=" * 80)
    print()

    sorry_files = []
    for filepath, results in sorted(all_results.items()):
        if not results:
            continue

        code_sorries = [s for s in results['keywords'].get('sorry', []) if not s['in_comment']]
        if code_sorries:
            sorry_files.append((filepath, code_sorries))

    if sorry_files:
        for filepath, sorries in sorry_files:
            rel_path = filepath.relative_to(ROOT_DIR)
            print(f"\n{rel_path}: {len(sorries)} executable sorries")
            for sorry in sorries[:5]:  # Show first 5
                print(f"  Line {sorry['line']}: {sorry['text'][:70]}")
            if len(sorries) > 5:
                print(f"  ... and {len(sorries) - 5} more")
    else:
        print("✅ No executable sorries found!")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

def print_json_results(all_results, module_stats):
    """Print results in JSON format."""
    # Build structured output
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(ROOT_DIR),
            "files_scanned": len([r for r in all_results.values() if r]),
            "modules": len(module_stats)
        },
        "summary": {
            "axioms": {
                "total": sum(m['axioms'] for m in module_stats.values()),
                "documented": sum(m['axioms'] for m in module_stats.values()) - sum(m['axioms_undocumented'] for m in module_stats.values()),
                "undocumented": sum(m['axioms_undocumented'] for m in module_stats.values())
            },
            "sorries": {
                "executable": sum(m['sorries_code'] for m in module_stats.values()),
                "documentation": sum(m['sorries_comment'] for m in module_stats.values())
            },
            "theorems": {
                "total": sum(m['theorems'] for m in module_stats.values()),
                "proven": sum(m['theorems_proven'] for m in module_stats.values()),
                "sorried": sum(m['theorems_sorry'] for m in module_stats.values())
            },
            "verification_flags": {
                "noncomputable": sum(m['noncomputable'] for m in module_stats.values()),
                "unsafe": sum(m['unsafe'] for m in module_stats.values())
            }
        },
        "modules": module_stats,
        "files": {}
    }

    # Add per-file details
    for filepath, results in all_results.items():
        if results:
            rel_path = str(filepath.relative_to(ROOT_DIR))
            output["files"][rel_path] = {
                "module": results['module'],
                "stats": dict(results['stats']),
                "axioms": results['patterns'].get('axiom_decl', []),
                "sorries": [s for s in results['keywords'].get('sorry', []) if not s['in_comment']]
            }

    print(json.dumps(output, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Audit Lean 4 files for verification status')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--root', type=Path, default=ROOT_DIR,
                       help=f'Root directory to scan (default: {ROOT_DIR})')
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()

    if args.format == 'text':
        print(f"Scanning for .lean files in {root_dir}...")

    lean_files = find_lean_files(root_dir)

    if args.format == 'text':
        print(f"Found {len(lean_files)} .lean files\n")

    all_results = {}
    for filepath in lean_files:
        results = search_file(filepath, KEYWORDS, PATTERNS, root_dir)
        all_results[filepath] = results

    # Group by module
    module_stats = group_by_module(all_results)

    # Print results
    print_results(all_results, module_stats, args.format)

    # Export JSON (always, for comparison)
    if args.format == 'text':
        json_output = OUTPUT_DIR / 'audit_results.json'

        # Compute summary statistics
        total_files = len([r for r in all_results.values() if r])
        total_axioms = sum(m['axioms'] for m in module_stats.values())
        total_axioms_undoc = sum(m['axioms_undocumented'] for m in module_stats.values())
        total_sorries_code = sum(m['sorries_code'] for m in module_stats.values())
        total_sorries_comment = sum(m['sorries_comment'] for m in module_stats.values())
        total_theorems = sum(m['theorems'] for m in module_stats.values())
        total_theorems_proven = sum(m['theorems_proven'] for m in module_stats.values())
        total_theorems_sorry = sum(m['theorems_sorry'] for m in module_stats.values())
        total_noncomp = sum(m['noncomputable'] for m in module_stats.values())
        total_unsafe = sum(m['unsafe'] for m in module_stats.values())

        # Build files section
        files_output = {}
        for filepath, results in all_results.items():
            if results:
                files_output[str(filepath.relative_to(root_dir.parent))] = {
                    "module": results['module'],
                    "stats": dict(results['stats']),
                    "patterns": {
                        "axiom_decl": results['patterns'].get('axiom_decl', []),
                        "theorem_decl": results['patterns'].get('theorem_decl', []),
                        "theorem_sorry": results['patterns'].get('theorem_sorry', []),
                    },
                    "keywords": {
                        "sorry": results['keywords'].get('sorry', []),
                        "axiom": results['keywords'].get('axiom', []),
                    }
                }

        with open(json_output, 'w') as f:
            json.dump({
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "project_root": str(root_dir),
                    "files_scanned": total_files,
                    "modules": len(module_stats)
                },
                "summary": {
                    "axioms": {
                        "total": total_axioms,
                        "documented": total_axioms - total_axioms_undoc,
                        "undocumented": total_axioms_undoc
                    },
                    "sorries": {
                        "executable": total_sorries_code,
                        "documentation": total_sorries_comment
                    },
                    "theorems": {
                        "total": total_theorems,
                        "proven": total_theorems_proven,
                        "sorried": total_theorems_sorry
                    },
                    "verification_flags": {
                        "noncomputable": total_noncomp,
                        "unsafe": total_unsafe
                    }
                },
                "modules": module_stats,
                "files": files_output
            }, f, indent=2)
        print(f"\n📄 Results exported to: {json_output}")

if __name__ == '__main__':
    main()
