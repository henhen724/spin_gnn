"""Run every tests/test_*.py file in one process and return one exit code.

There is no pytest for this project's interpreter on Sherlock (see
tests/_pytest_shim.py), and the files are split between ones that end in
`sys.exit(_pytest_shim.run(globals()))` and ones that are plain collections of
test_* functions. Running the suite therefore meant either `python3
tests/test_x.py` file by file -- which silently skips every file without a
__main__ block -- or a hand-written loop per session.

This driver reuses the suite's OWN collector, _pytest_shim.collect, so
parametrized cases expand here exactly as they do when a file is run directly,
and a file that reaches for a pytest feature the shim does not provide fails
loudly instead of quietly not running.

    python3 scripts/run_tests.py                 # the whole suite
    python3 scripts/run_tests.py tied_depth      # substring filter on filenames

Modules only -- never activate a venv or set PYTHONPATH here; see
scripts/slurm/run_tests.sbatch for the module list.
"""
import importlib
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))  # for `import _pytest_shim`

from tests import _pytest_shim  # noqa: E402


def main(argv):
    want = argv[1:]
    names = sorted(
        f[:-3] for f in os.listdir(os.path.join(_ROOT, "tests"))
        if f.startswith("test_") and f.endswith(".py")
    )
    if want:
        names = [n for n in names if any(w in n for w in want)]
        if not names:
            print("no test files matched %r" % (want,))
            return 1

    total = failed = 0
    broken = []
    for name in names:
        print("\n=== %s" % name, flush=True)
        try:
            mod = importlib.import_module("tests.%s" % name)
        except Exception:
            # An import error is a failure, not a skip: it is how a whole file
            # stops running without anything in the log saying so.
            broken.append((name, traceback.format_exc()))
            print("IMPORT FAILED", flush=True)
            continue
        total += len(_pytest_shim.collect(vars(mod)))
        if _pytest_shim.run(vars(mod)):  # 1 if any case in the file failed
            failed += 1

    for name, tb in broken:
        print("\n--- import %s\n%s" % (name, tb), end="")
    print("\n%d file(s), %d case(s); %d file(s) with failures, %d import failure(s)"
          % (len(names), total, failed, len(broken)))
    return 1 if (failed or broken) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
