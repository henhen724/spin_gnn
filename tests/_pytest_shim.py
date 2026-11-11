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

This module first provided only `mark.parametrize`, on the argument that
`raises`, `approx` and fixtures should be absent so that a test reaching for
one fails loudly rather than quietly not running. Running the whole suite in
one job (scripts/run_tests.py, 2026-11-10) showed what that cost: TEN of the 27
files still began with a bare `import pytest` and so had never been collected
on this machine at all -- silence of exactly the kind the paragraph above was
written against. A missing `approx` does not make a test fail loudly; it makes
the file holding it disappear.

So `approx`, `raises`, `fail` and the `tmp_path` fixture are provided now,
being everything those files actually use. Loudness is kept where it belongs:
the driver counts an import error as a failure rather than a skip, and
anything NOT provided here still raises AttributeError inside the case that
reaches for it, which the runner reports as a failure with its traceback.
"""
import functools
import inspect
import itertools
import os
import pathlib
import re
import tempfile
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


class _Approx:
    """`expected == approx(x)` with pytest's tolerance rule and defaults.

    A value matches when |actual - expected| <= max(rel * |expected|, abs),
    rel=1e-6 and abs=1e-12 as in pytest. Sequences compare elementwise and
    must have the same length; that covers every use in this suite (scalars,
    lists of floats, and 1-D numpy arrays, which iterate elementwise).

    Deliberately NOT a numpy-aware comparison: `np.allclose` is already
    imported where an array needs one, and quietly accepting a 2-D array here
    would compare the wrong thing.
    """

    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = 1e-6 if rel is None else rel
        self.abs = 1e-12 if abs is None else abs

    def _close(self, actual, expected):
        diff = abs(float(actual) - float(expected))
        return diff <= max(self.rel * abs(float(expected)), self.abs)

    def __eq__(self, actual):
        if hasattr(self.expected, "__len__") or hasattr(self.expected, "__iter__"):
            try:
                pairs = list(zip(actual, self.expected))
            except TypeError:
                return NotImplemented
            if len(pairs) != len(list(self.expected)):
                return False
            return all(self._close(a, e) for a, e in pairs)
        return self._close(actual, self.expected)

    def __ne__(self, actual):
        eq = self.__eq__(actual)
        return eq if eq is NotImplemented else not eq

    def __repr__(self):
        return "approx(%r, rel=%g, abs=%g)" % (self.expected, self.rel, self.abs)


def approx(expected, rel=None, abs=None):
    return _Approx(expected, rel=rel, abs=abs)


class _Raises:
    """`with raises(E, match=...) as e:` -- E (or a subclass) must be raised.

    `e.value` is the exception, so a test can assert on its message beyond the
    regex. `match` is `re.search`, not `fullmatch`, as in pytest.
    """

    def __init__(self, expected, match=None):
        self.expected = expected
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError("DID NOT RAISE %r" % (self.expected,))
        if not issubclass(exc_type, self.expected):
            return False  # a different exception propagates, as it should
        if self.match is not None and not re.search(self.match, str(exc)):
            raise AssertionError(
                "%r raised, but its message %r does not match %r"
                % (exc_type.__name__, str(exc), self.match))
        self.value = exc
        return True


def raises(expected, match=None):
    return _Raises(expected, match=match)


def fail(msg="failed"):
    raise AssertionError(msg)


def _tmp_path():
    """A fresh directory for one case, as a pathlib.Path.

    Prefers $L_SCRATCH, which is node-local and wiped when the job ends. These
    are NOT removed afterwards: deleting paths on a user's behalf is the one
    thing this suite should never learn to do, and the directories hold a few
    kilobytes of checkpoint and config written by the training smoke tests.
    """
    parent = os.environ.get("L_SCRATCH") or None
    return pathlib.Path(tempfile.mkdtemp(prefix="shim_tmp_", dir=parent))


_FIXTURES = {"tmp_path": _tmp_path}


def _bind(fn, kwargs):
    """Zero-arg callable for one case, with fixtures built at CALL time.

    Built at call time and not at collection, so a run that stops early has
    not already made a directory for every case after it. A parameter this
    module does not know stays unbound and the case fails with a TypeError
    naming it, which is the intended way to find out.
    """
    wanted = [name for name in inspect.signature(fn).parameters
              if name in _FIXTURES and name not in kwargs]
    if not wanted:
        return functools.partial(fn, **kwargs) if kwargs else fn

    def call():
        full = dict(kwargs)
        for name in wanted:
            full[name] = _FIXTURES[name]()
        return fn(**full)

    return call


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
            cases.append((fn.__name__, _bind(fn, {})))
            continue
        for combo in itertools.product(*layers):
            kwargs = {}
            for one in combo:
                kwargs.update(one)
            ident = ",".join(f"{k}={v}" for k, v in kwargs.items())
            cases.append((f"{fn.__name__}[{ident}]", _bind(fn, kwargs)))
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
