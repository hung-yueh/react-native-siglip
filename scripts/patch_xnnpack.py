#!/usr/bin/env python3
"""
Patch ExecuTorch's XNNCompiler.cpp to remove duplicate case labels for Sin/Cos.

ExecuTorch v1.1.0 has a bug where Sin and Cos share the same enum value,
causing a "duplicate case label" compile error. This script comments out
the duplicate case block.

Usage:
    python3 patch_xnnpack.py <path/to/XNNCompiler.cpp>
"""

import sys
import re


def patch_file(filepath: str) -> None:
    with open(filepath, "r") as f:
        content = f.read()

    lines = content.split("\n")
    patched_lines = []
    skip_until_break = False
    duplicate_found = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for duplicate case labels for Sin or Cos
        # The pattern: a second `case xnn_node_type_*_sin:` or `case xnn_node_type_*_cos:`
        # that causes a duplicate case value error.
        #
        # Strategy: find the second occurrence of xnn_unary_sin or xnn_unary_cos
        # case in an xnn_node_type switch and comment it out.
        if not duplicate_found and re.search(
            r"case\s+xnn_node_type_static_unary.*cos", line, re.IGNORECASE
        ):
            # Check if there's already a sin case above with the same enum value
            # Look back for a sin case
            has_sin = any(
                re.search(r"case\s+xnn_node_type_static_unary.*sin", prev, re.IGNORECASE)
                for prev in patched_lines[-20:]
            )
            if has_sin:
                # Comment out this case and its body until the next break/case/}
                patched_lines.append(f"// PATCHED: duplicate case label removed: {line.strip()}")
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    patched_lines.append(f"// PATCHED: {next_line}")
                    if "break;" in next_line or (
                        re.match(r"\s*(case |default:|})", lines[min(i + 1, len(lines) - 1)])
                    ):
                        i += 1
                        break
                    i += 1
                duplicate_found = True
                continue

        patched_lines.append(line)
        i += 1

    with open(filepath, "w") as f:
        f.write("\n".join(patched_lines))

    if duplicate_found:
        print(f"[patch_xnnpack] Patched duplicate Sin/Cos case in {filepath}")
    else:
        print(f"[patch_xnnpack] No duplicate case found in {filepath} (may not need patching)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <XNNCompiler.cpp>", file=sys.stderr)
        sys.exit(1)
    patch_file(sys.argv[1])
