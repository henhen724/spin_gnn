"""Check a deck builder's page literals against the slides library's own caps,
without running it.

A render that violates a cap dies inside slides.py, which costs a whole job to
find out. This reads the builder with ast instead, so the check runs on a login
node in under a second.

Written for the login node's python3, which is 3.6.8: string literals there are
ast.Str, NOT ast.Constant, and an isinstance(node, ast.Constant) lint silently
matches nothing and reports a clean bill. That exact mistake let nine
over-length literals through on 2026-11-12.

A literal that is the left side of a % is counted as-is: the format string has
one token per placeholder, and every substitution these builders make is a
number or a single word, so the word count carries over.

    python3 scripts/deck_lint.py notebooks/make_phases_web.py ...
"""

import ast
import re
import sys

# slides.py: CLAIM_MAX_WORDS, NOTE_MAX_WORDS, BULLET_MAX_WORDS.
CAPS = {"claim": 12, "note": 25, "bullet": 14}
PAGE_CALLS = ("claim_page", "table_page", "points_page", "title_page", "expect")


def words(s):
    return [w for w in re.split(r"\s+", s.strip()) if w]


def literal(node):
    """The string a node contributes, or None."""
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return literal(node.left)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = literal(node.left), literal(node.right)
        return None if a is None or b is None else a + b
    return None


def check(path):
    with open(path) as fh:
        tree = ast.parse(fh.read(), path)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name not in PAGE_CALLS:
            continue
        # arg 0 is the title, except on expect(), whose arg 0 is the figure slug
        # and arg 1 the title. A list arg is bullets.
        first = 1 if name == "expect" else 0
        items = []
        if len(node.args) > first:
            items.append(("claim", node.args[first]))
        for a in node.args[first + 1:]:
            if isinstance(a, (ast.List, ast.Tuple)):
                items += [("bullet", e) for e in a.elts]
        for kw in node.keywords:
            if kw.arg == "note":
                items.append(("note", kw.value))
        for kind, arg in items:
            s = literal(arg)
            if s is None:
                continue
            n = len(words(s))
            if n > CAPS[kind]:
                bad.append((node.lineno, kind, n, CAPS[kind], s))
            if kind == "claim" and n < 3:
                bad.append((node.lineno, "claim-too-short", n, 3, s))
    for lineno, kind, n, cap, s in bad:
        print("%s:%d  %s %d words (cap %d): %s" % (path, lineno, kind, n, cap, s))
    return len(bad)


if __name__ == "__main__":
    total = sum(check(p) for p in sys.argv[1:])
    print("%d over-length page literals" % total)
    sys.exit(1 if total else 0)
