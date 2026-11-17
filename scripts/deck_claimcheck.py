"""Cross-check every number in the talk deck against audited sources.

Zero compute, read-only. Flags numbers that appear nowhere in the source decks,
STATUS.md, FINDINGS.md or LOG.md. False positives are expected (derived values,
equation constants, counts of things); the point is to produce a short candidate
list to check by hand, not a verdict.
"""
import re, os, glob

TALK = "/home/users/hshunt/slides/decks/transition-and-minimal-models/deck.md"
SRC = sorted(glob.glob("/home/users/hshunt/slides/decks/spin-gnn/*/deck.md"))
PROJ = ["/scratch/users/hshunt/spin_gnn/STATUS.md",
        "/scratch/users/hshunt/spin_gnn/FINDINGS.md",
        "/scratch/users/hshunt/spin_gnn/LOG.md"]

corpus = []
for p in SRC + PROJ:
    try:
        corpus.append(open(p, errors="replace").read())
    except OSError:
        pass
corpus = "\n".join(corpus)
# a comma-free view so 9,638,458 matches 9638458 and vice versa
corpus_nc = corpus.replace(",", "")

talk = open(TALK).read()
body = talk.split("---\n", 2)[2]
pages = body.split("\n---\n")

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:[,{}]*\d{3})+|\d+\.\d+|\d{3,})(?![\w])")

# numbers that are structural rather than measured: grid sizes, counts of
# slides, years, and the small integers that define the problem
SKIP = {
    "2026", "1117", "1116", "0830", "512", "414", "256", "192", "176", "320",
    "128", "160", "224", "384", "1024", "4096", "240", "100", "400", "300",
    "350", "0.5", "1.0", "0.0", "48", "96", "252", "996", "4140", "17196",
    "49992", "120", "0.95", "0.33", "0.46", "0.50", "0.30", "0.85", "0.65",
    "0.60", "0.40", "0.90", "0.10", "0.25", "0.250",
}

flagged = []
for p in pages:
    if not p.strip():
        continue
    lines = p.strip().split("\n")
    title = next((l for l in lines if re.match(r"^#{1,2} ", l)), lines[0])[:80]
    for m in NUM.finditer(p):
        raw = m.group(1)
        norm = raw.replace(",", "").replace("{", "").replace("}", "")
        if norm in SKIP or raw in SKIP:
            continue
        if len(norm) < 3:
            continue
        if raw in corpus or norm in corpus_nc:
            continue
        s = max(0, m.start() - 60)
        ctx = " ".join(p[s:m.end() + 40].split())
        flagged.append((title, raw, ctx))

print(f"pages={len([p for p in pages if p.strip()])}  flagged={len(flagged)}\n")
seen = set()
for title, raw, ctx in flagged:
    key = (title, raw)
    if key in seen:
        continue
    seen.add(key)
    print(f"[{raw}]  {title}")
    print(f"      ...{ctx}...")
