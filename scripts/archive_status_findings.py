"""One-off: move STATUS.md's historical findings block into FINDINGS.md.

STATUS.md had grown to 60 lines but one of them was 26,513 characters, so the
60-line brief was being met in letter only. This moves findings 62-114 and the
accumulated method lessons into FINDINGS.md, which already archives 1-54, and
leaves STATUS.md to be rewritten as current best / running jobs / open problems
/ next steps.

Nothing is deleted: every archived line is appended verbatim under a dated
header before STATUS.md is rewritten by hand.
"""
import datetime
import os

REPO = "/home/users/hshunt/spin_gnn"
# 1-indexed, inclusive. Everything else stays in STATUS.md and is rewritten.
BLOCKS = [(16, 35), (37, 46), (51, 54), (58, 58)]


def main():
    with open(os.path.join(REPO, "STATUS.md")) as f:
        lines = f.readlines()

    out = ["\n\n", "## Archived from STATUS.md on %s\n"
           % datetime.date.today().isoformat(),
           "\nFindings 62-114, the stage LOGIC fit table, the per-stage artefact\n"
           "locations and the accumulated method lessons, moved here verbatim so\n"
           "STATUS.md could go back to being a status file. Grep this, never read\n"
           "it whole.\n\n"]
    kept = 0
    for lo, hi in BLOCKS:
        for i in range(lo - 1, hi):
            out.append(lines[i])
            kept += 1
        out.append("\n")

    with open(os.path.join(REPO, "FINDINGS.md"), "a") as f:
        f.writelines(out)
    print("appended %d lines (%d chars) to FINDINGS.md"
          % (kept, sum(len(x) for x in out)))


if __name__ == "__main__":
    main()
