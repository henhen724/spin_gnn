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

Exit code 0 means every file that CAN run did run and passed. Files blocked by a
third-party package Sherlock does not have are listed in ABSENT below, reported
as SKIPPED by name, and excluded from the exit code -- otherwise the code would
be permanently 1 and a new import failure would not show up in it.

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

# Third-party packages that do not exist for this interpreter on Sherlock, each
# with the check that established it. A file whose import fails for EXACTLY one
# of these is reported SKIPPED and does not set the exit code; every other
# import error is still a failure.
#
# Why this list exists rather than just letting the suite exit 1: with two files
# permanently unimportable, the exit code was permanently 1, so it carried no
# signal and a NEW import failure would have been invisible in it. An explicit
# list is the opposite of a silent skip -- it names what is not running, and
# `stale` below fails the suite if one of these ever becomes importable and the
# entry is left behind.
ABSENT = {
    "torch_geometric": "`ml spider torch-geometric` finds nothing on Sherlock. "
                       "Only spin_gnn/model.py (the original GNN) imports it; "
                       "nothing on the sudoku line does.",
}


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
    broken, skipped = [], []
    for name in names:
        print("\n=== %s" % name, flush=True)
        try:
            mod = importlib.import_module("tests.%s" % name)
        except ImportError as exc:
            # ModuleNotFoundError.name is the module that was actually missing,
            # however deep the chain was -- here it is reached through
            # spin_gnn.model, two imports down from the test file.
            missing = getattr(exc, "name", None)
            if missing in ABSENT:
                skipped.append((name, missing))
                print("SKIPPED -- no %s for this interpreter" % missing, flush=True)
                continue
            broken.append((name, traceback.format_exc()))
            print("IMPORT FAILED", flush=True)
            continue
        except Exception:
            # An import error is a failure, not a skip: it is how a whole file
            # stops running without anything in the log saying so.
            broken.append((name, traceback.format_exc()))
            print("IMPORT FAILED", flush=True)
            continue
        total += len(_pytest_shim.collect(vars(mod)))
        if _pytest_shim.run(vars(mod)):  # 1 if any case in the file failed
            failed += 1

    # A skip reason that has stopped being true is worse than no skip list: it
    # hides a file that could be running again.
    stale = []
    for pkg in sorted(ABSENT):
        try:
            importlib.import_module(pkg)
        except ImportError:
            continue
        stale.append(pkg)

    for name, tb in broken:
        print("\n--- import %s\n%s" % (name, tb), end="")
    if skipped:
        print("\n--- skipped, not run:")
        for name, missing in skipped:
            print("    %-28s needs %s\n        %s" % (name, missing, ABSENT[missing]))
    for pkg in stale:
        print("\n--- STALE SKIP: %s imports fine now. Remove it from ABSENT so "
              "the files that need it run again." % pkg)
    print("\n%d file(s), %d case(s); %d file(s) with failures, %d import "
          "failure(s), %d skipped" % (len(names), total, failed, len(broken),
                                      len(skipped)))
    return 1 if (failed or broken or stale) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
