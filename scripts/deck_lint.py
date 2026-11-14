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

A title or bullet that is COMPUTED -- a verdict-branched conditional, or a
lookup into a dict of alternatives -- used to be invisible here, so the lint
reported a clean bill and the render died anyway. Job 52268621 was lost that
way on 2026-11-13 to a 13-word conditional title. Conditionals, dict lookups and
names bound to either are now followed, and EVERY branch is checked, because
which branch renders depends on data the lint cannot see.

    python3 scripts/deck_lint.py notebooks/make_phases_web.py ...
"""

import ast
import re
import sys

# slides.py: CLAIM_MAX_WORDS, NOTE_MAX_WORDS, BULLET_MAX_WORDS.
CAPS = {"claim": 12, "note": 25, "bullet": 14}
PAGE_CALLS = ("claim_page", "table_page", "points_page", "title_page", "expect")
# slides.py _check_points: a points_page carries at most this many bullets. The
# word caps above are per literal, so a page of three short bullets passed the
# lint and still died in the render -- that cost jobs 52183401 and 52189773 on
# 2026-11-13. Counted here because the count is visible in the ast even when
# the bullets themselves are computed strings.
POINTS_MAX = 3


def words(s):
    return [w for w in re.split(r"\s+", s.strip()) if w]


def env_of(tree):
    """Every `name = <expr>` in the file, so a page call that passes a variable
    can still be followed. These builders bind each name once; if one were
    rebound, the lint would check both bindings, which errs toward noise rather
    than toward a clean bill."""
    env = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    env.setdefault(t.id, []).append(node.value)
    return env


def resolve(node, env, seen=None):
    """Every node a page argument could actually be at render time.

    A conditional contributes both branches and a lookup into a dict literal
    contributes all of its values, because the lint cannot know which one the
    data selects -- so all of them have to fit.
    """
    seen = seen or set()
    if node is None or id(node) in seen:
        return []
    seen = seen | {id(node)}
    if isinstance(node, ast.IfExp):
        return (resolve(node.body, env, seen) + resolve(node.orelse, env, seen))
    if isinstance(node, ast.Name):
        out = []
        for v in env.get(node.id, []):
            out += resolve(v, env, seen)
        return out
    if isinstance(node, ast.Subscript):
        inner = resolve(node.value, env, seen)
        out = []
        for d in inner:
            if isinstance(d, ast.Dict):
                for v in d.values:
                    out += resolve(v, env, seen)
        return out
    return [node]


def literals(node, env):
    """The strings a node could contribute."""
    for cand in resolve(node, env):
        if isinstance(cand, ast.Str):
            yield cand.s
        elif isinstance(cand, ast.BinOp) and isinstance(cand.op, ast.Mod):
            for s in literals(cand.left, env):
                yield s
        elif isinstance(cand, ast.BinOp) and isinstance(cand.op, ast.Add):
            for a in literals(cand.left, env):
                for b in literals(cand.right, env):
                    yield a + b


def check(path):
    with open(path) as fh:
        tree = ast.parse(fh.read(), path)
    env = env_of(tree)
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
            for lst in resolve(a, env):
                if isinstance(lst, (ast.List, ast.Tuple)):
                    items += [("bullet", e) for e in lst.elts]
                    if name == "points_page" and len(lst.elts) > POINTS_MAX:
                        bad.append((node.lineno, "points-per-page",
                                    len(lst.elts), POINTS_MAX,
                                    "split this page"))
        for kw in node.keywords:
            if kw.arg == "note":
                items.append(("note", kw.value))
        for kind, arg in items:
            for s in literals(arg, env):
                n = len(words(s))
                if n > CAPS[kind]:
                    bad.append((node.lineno, kind, n, CAPS[kind], s))
                if kind == "claim" and n < 3:
                    bad.append((node.lineno, "claim-too-short", n, 3, s))
    for lineno, kind, n, cap, s in bad:
        unit = "bullets" if kind == "points-per-page" else "words"
        print("%s:%d  %s %d %s (cap %d): %s"
              % (path, lineno, kind, n, unit, cap, s))
    return len(bad)


if __name__ == "__main__":
    total = sum(check(p) for p in sys.argv[1:])
    print("%d page-cap violations" % total)
    sys.exit(1 if total else 0)
