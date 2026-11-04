"""Execute a notebook's code cells in order and SAVE the outputs back into it.

    python3 -m notebooks.exec_nb <notebook.ipynb>

`run_nb_cells.py` proves the cells run but throws the output away, which leaves a
committed .ipynb that renders blank on GitHub and is unreadable without a kernel
-- and there is no jupyter on this cluster to re-run it with. A results notebook
has to carry its own numbers, so this one captures stdout and any figures each
cell produced and writes them in as real nbformat outputs.

Cells run in a shared namespace, in order, exactly as a kernel would. A cell that
raises is recorded as an `error` output and execution stops there rather than
continuing into cells that would silently read stale state.
"""
import base64
import contextlib
import io
import json
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg switch)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _png_outputs(seen):
    """Any figure opened since the last call, as base64 PNG display outputs."""
    out = []
    for n in plt.get_fignums():
        if n in seen:
            continue
        seen.add(n)
        buf = io.BytesIO()
        plt.figure(n).savefig(buf, format="png", dpi=110, bbox_inches="tight")
        out.append({"output_type": "display_data",
                    "data": {"image/png": base64.b64encode(buf.getvalue()).decode()},
                    "metadata": {}})
    return out


def main(path):
    nb = json.load(io.open(path, encoding="utf-8"))
    g = {"__name__": "__main__"}
    seen, count, failed = set(), 0, None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        count += 1
        buf = io.StringIO()
        outs = []
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile("".join(c["source"]), f"cell{i}", "exec"), g)
        except Exception:
            # Record the traceback in the cell rather than only on the terminal:
            # a notebook that fails should say so where it is read.
            outs.append({"output_type": "error", "ename": "Error", "evalue": "see traceback",
                         "traceback": traceback.format_exc().splitlines()})
            failed = i
        text = buf.getvalue()
        if text:
            outs.insert(0, {"output_type": "stream", "name": "stdout",
                            "text": text.splitlines(keepends=True)})
        c["outputs"] = outs + _png_outputs(seen)
        c["execution_count"] = count
        print(f"cell {i:2d}: {len(text.splitlines()):4d} lines"
              + (", FAILED" if failed == i else ""))
        if failed is not None:
            break
    io.open(path, "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(("FAILED at cell %d; " % failed if failed is not None else "all cells OK; ")
          + f"wrote outputs into {path}")
    return 1 if failed is not None else 0


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    sys.exit(main(sys.argv[1]))
