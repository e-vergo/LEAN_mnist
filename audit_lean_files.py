#!/usr/bin/env python3
"""
Comprehensive audit of Lean files for proof obligations and axioms.

Searches for: axiom, sorry, trivial, admit, sorry_proof, and other verification keywords.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

# Keywords to search for
KEYWORDS = {
    'axiom': r'\baxiom\b',
    'sorry': r'\bsorry\b',
    'trivial': r'\btrivial\b',
    'admit': r'\badmit\b',
    'sorry_proof': r'sorry_proof'
}

# Additional patterns to detect
PATTERNS = {
    'by_sorry': r'by\s+sorry',
    'sorry_in_proof': r':=\s*by\s+sorry',
    'axiom_declaration': r'axiom\s+\w+',
    'theorem_with_sorry': r'theorem\s+\w+.*:=\s*by\s+sorry',
}

def find_lean_files(root_dir):
    """Find all .lean files in the directory tree."""
    lean_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.lean'):
                lean_files.append(os.path.join(root, file))
    return sorted(lean_files)

def search_file(filepath, keywords, patterns):
    """Search a file for keywords and patterns."""
    results = {
        'keywords': defaultdict(list),
        'patterns': defaultdict(list),
        'stats': defaultdict(int),
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            # Check keywords
            for keyword, regex in keywords.items():
                if re.search(regex, line):
                    # Check if it's in a comment
                    in_comment = False
                    stripped = line.strip()
                    if stripped.startswith('--') or stripped.startswith('/-') or stripped.startswith('*'):
                        in_comment = True

                    results['keywords'][keyword].append({
                        'line': line_num,
                        'text': line.strip(),
                        'in_comment': in_comment
                    })
                    results['stats'][keyword] += 1
                    if in_comment:
                        results['stats'][f'{keyword}_in_comment'] += 1

            # Check patterns
            for pattern_name, regex in patterns.items():
                if re.search(regex, line, re.IGNORECASE):
                    results['patterns'][pattern_name].append({
                        'line': line_num,
                        'text': line.strip()
                    })
                    results['stats'][f'pattern_{pattern_name}'] += 1

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    return results

def print_results(all_results):
    """Print formatted results."""
    print("=" * 80)
    print("LEAN FILE AUDIT REPORT")
    print("=" * 80)
    print()

    # Overall statistics
    print("📊 OVERALL STATISTICS")
    print("-" * 80)

    total_files = len(all_results)
    total_stats = defaultdict(int)
    files_with_keyword = defaultdict(set)

    for filepath, results in all_results.items():
        if results:
            for key, count in results['stats'].items():
                total_stats[key] += count
                if count > 0:
                    files_with_keyword[key].add(filepath)

    print(f"Total .lean files scanned: {total_files}")
    print()

    # Print keyword statistics
    for keyword in ['axiom', 'sorry', 'trivial', 'admit', 'sorry_proof']:
        count = total_stats[keyword]
        in_comment = total_stats.get(f'{keyword}_in_comment', 0)
        actual = count - in_comment
        files = len(files_with_keyword[keyword])

        if count > 0:
            print(f"{keyword.upper()}:")
            print(f"  Total occurrences: {count}")
            print(f"  In comments/docs: {in_comment}")
            print(f"  In actual code: {actual}")
            print(f"  Files affected: {files}")
            print()

    # Print pattern statistics
    print("PATTERN MATCHES:")
    for pattern_name in PATTERNS.keys():
        key = f'pattern_{pattern_name}'
        count = total_stats[key]
        if count > 0:
            print(f"  {pattern_name}: {count}")
    print()

    # Detailed file breakdown
    print("=" * 80)
    print("📁 DETAILED FILE BREAKDOWN")
    print("=" * 80)
    print()

    for filepath, results in sorted(all_results.items()):
        if not results:
            continue

        # Check if file has any actual (non-comment) issues
        has_issues = False
        for keyword in ['axiom', 'sorry', 'trivial']:
            if results['keywords'].get(keyword):
                for match in results['keywords'][keyword]:
                    if not match['in_comment']:
                        has_issues = True
                        break

        if not has_issues:
            continue

        rel_path = os.path.relpath(filepath, '/Users/eric/LEAN_mnist')
        print(f"\n{'=' * 80}")
        print(f"FILE: {rel_path}")
        print(f"{'=' * 80}")

        # Print keywords found
        for keyword in ['axiom', 'sorry', 'trivial', 'admit']:
            matches = results['keywords'].get(keyword, [])
            code_matches = [m for m in matches if not m['in_comment']]
            comment_matches = [m for m in matches if m['in_comment']]

            if code_matches:
                print(f"\n🔴 {keyword.upper()} (in code): {len(code_matches)} occurrences")
                for match in code_matches[:10]:  # Show first 10
                    print(f"  Line {match['line']}: {match['text'][:100]}")
                if len(code_matches) > 10:
                    print(f"  ... and {len(code_matches) - 10} more")

            if comment_matches and keyword in ['sorry', 'axiom']:
                print(f"\n💬 {keyword.upper()} (in comments): {len(comment_matches)} occurrences")
                for match in comment_matches[:3]:  # Show first 3
                    print(f"  Line {match['line']}: {match['text'][:100]}")
                if len(comment_matches) > 3:
                    print(f"  ... and {len(comment_matches) - 3} more")

        # Print patterns
        for pattern_name, matches in results['patterns'].items():
            if matches:
                print(f"\n⚠️  PATTERN '{pattern_name}': {len(matches)} matches")
                for match in matches[:5]:
                    print(f"  Line {match['line']}: {match['text'][:100]}")
                if len(matches) > 5:
                    print(f"  ... and {len(matches) - 5} more")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

    # Summary
    print("\n📋 SUMMARY:")
    actual_sorries = total_stats['sorry'] - total_stats.get('sorry_in_comment', 0)
    actual_axioms = total_stats['axiom'] - total_stats.get('axiom_in_comment', 0)
    actual_trivials = total_stats['trivial'] - total_stats.get('trivial_in_comment', 0)

    print(f"  • Axioms in code: {actual_axioms}")
    print(f"  • Sorries in code: {actual_sorries}")
    print(f"  • Trivials in code: {actual_trivials}")
    print(f"  • Total files with issues: {len([f for f, r in all_results.items() if r and any(not m['in_comment'] for matches in r['keywords'].values() for m in matches)])}")

def main():
    root_dir = '/Users/eric/LEAN_mnist/VerifiedNN'

    print("Scanning for .lean files...")
    lean_files = find_lean_files(root_dir)
    print(f"Found {len(lean_files)} .lean files")
    print()

    all_results = {}
    for filepath in lean_files:
        results = search_file(filepath, KEYWORDS, PATTERNS)
        all_results[filepath] = results

    print_results(all_results)

    # Export to JSON for further analysis
    json_output = '/Users/eric/LEAN_mnist/audit_results.json'
    with open(json_output, 'w') as f:
        json.dump({
            'files': lean_files,
            'results': {k: {
                'keywords': {kw: matches for kw, matches in v['keywords'].items()},
                'patterns': {p: matches for p, matches in v['patterns'].items()},
                'stats': dict(v['stats'])
            } for k, v in all_results.items() if v}
        }, f, indent=2)
    print(f"\n📄 Detailed results exported to: {json_output}")

if __name__ == '__main__':
    main()
