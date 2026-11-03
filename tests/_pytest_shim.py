"""Enough of pytest to run this suite without it, plus a plain runner.

Sherlock has no pytest for this project's interpreter: `ml spider py-pytest`
offers only py-pytest/7.1.3_py39, and everything here runs on the py312 stack.
So a test file whose first lines are `import pytest` cannot be collected on
this machine AT ALL -- which is how tests/test_sudoku_dd.py, the only check
that the incremental `remove_hint` update agrees with from-scratch DD, came to
have never run on the machine where every result is produced.

The fix in tests/test_sudoku_fall.py was to write that file without pytest.
That does not work here: test_sudoku_dd.py is built out of `parametrize`, and
hand-expanding 32 cases into loops would lose the per-case reporting that
makes a failure readable. So instead a test file imports the real pytest when
there is one and falls back to this, and stays collectable by a real pytest
elsewhere either way.

Only `mark.parametrize` is provided, because it is the only thing these files
use. Fixtures, `raises`, `approx` and custom marks are deliberately absent: a
test that reaches for one should fail loudly here rather than quietly not run,
which is the exact failure mode this module exists to end.
"""
import functools
import itertools
import traceback


class _Parametrize:
    """Records cases on the function; `collect` expands them at run time.

    Stacked decorators multiply the way pytest's do -- layers of 3 and 6 give
    18 cases. Decorators apply bottom-up, so layers arrive innermost-first and
    are prepended to recover the order they appear in the source.
    """

    def __call__(self, argnames, argvalues):
        names = ([a.strip() for a in argnames.split(",")]
                 if isinstance(argnames, str) else [str(a) for a in argnames])

        def decorate(fn):
            layer = []
            for value in argvalues:
                values = (value,) if len(names) == 1 else tuple(value)
                if len(values) != len(names):
                    raise ValueError(
                        f"{fn.__name__}: {len(values)} values for {len(names)} names {names}")
                layer.append(dict(zip(names, values)))
            fn.shim_params = [layer] + list(getattr(fn, "shim_params", []))
            return fn

        return decorate


class _Mark:
    parametrize = _Parametrize()


mark = _Mark()


def collect(namespace):
    """[(id, zero-arg callable)] for every case in a module's globals.

    Sorted by line number so the report reads in source order; `namespace`
    dict order would also work today but depends on nothing in particular.
    """
    fns = sorted(
        (v for k, v in namespace.items() if k.startswith("test_") and callable(v)),
        key=lambda f: f.__code__.co_firstlineno,
    )
    cases = []
    for fn in fns:
        layers = getattr(fn, "shim_params", None)
        if not layers:
            cases.append((fn.__name__, fn))
            continue
        for combo in itertools.product(*layers):
            kwargs = {}
            for one in combo:
                kwargs.update(one)
            ident = ",".join(f"{k}={v}" for k, v in kwargs.items())
            cases.append((f"{fn.__name__}[{ident}]", functools.partial(fn, **kwargs)))
    return cases


def run(namespace):
    """Run every case, print a report, return a process exit code.

    Catches Exception, not just AssertionError: a TypeError in one case must
    not hide the verdict on the other 31. Tracebacks are held back to the end
    so the pass/fail column stays scannable.
    """
    cases = collect(namespace)
    failures = []
    for ident, case in cases:
        try:
            case()
        except Exception:
            failures.append((ident, traceback.format_exc()))
            print(f"FAIL {ident}", flush=True)
        else:
            print(f"pass {ident}", flush=True)
    for ident, tb in failures:
        print(f"\n--- {ident}\n{tb}", end="")
    print(f"\n{len(cases) - len(failures)}/{len(cases)} passed")
    return 1 if failures else 0
