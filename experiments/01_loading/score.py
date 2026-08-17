"""Step C: grade a candidate source against the golden set.

One script grades all three candidates, because they all hand back the same
`Loaded` shape (see shape.py). The four checks are fixed in docs/SPRINT.md and
are not adjusted after seeing a result.

  1. found     -- share of the 8 probe sentences present in the output
  2. order     -- share of adjacent probe pairs that appear in the right order
  3. sections  -- share of the paper's real headings present as headings
  4. junk      -- running headers, hyphen-splits, duplicated maths per 1k words

Why the matcher is fuzzy, and how far
-------------------------------------
Golden probes come from LaTeX with \\cite{...} deleted; the rendered paper keeps
the citation ("...from data [47]."). Exact substring matching therefore scores
near zero for reasons that have nothing to do with the extractor's quality.

So both sides are reduced to lowercase word tokens (punctuation and pure-number
tokens dropped -- that alone removes numeric citations), and a probe counts as
found when its tokens appear *in order, contiguously enough*: at least 90% of
them inside a window only 1.6x its own length. Extra tokens the renderer added
are tolerated; reordered or missing text is not.

Both failure modes are checked before any score is reported -- see `verify()`:
a real probe must be found, and a shuffled or foreign probe must not be.

Run:  uv run python experiments/01_loading/score.py html
      uv run python experiments/01_loading/score.py html 2005.11401v4   (one paper)
"""

# ruff: noqa: RUF001 -- this file exists to normalise the ambiguous characters
# below, so it must be allowed to name them.
from __future__ import annotations

import difflib
import itertools
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_html
from shape import Loaded

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "golden" / "loading.json"
OUT = Path(__file__).resolve().parent / "scores.json"

MATCH_THRESHOLD = 0.90      # share of probe tokens that must line up, in order
WINDOW_SLACK = 1.6          # how much longer than the probe the window may be
ANCHOR = 5                  # tokens used to find where to look

QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "´": "'"}
DASHES = {"‐": "-", "‑": "-", "‒": "-", "–": "-",
          "—": "-", "―": "-", "−": "-"}


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for table in (QUOTES, DASHES):
        for bad, good in table.items():
            s = s.replace(bad, good)
    return re.sub(r"\s+", " ", s).strip().casefold()


def tokens(s: str) -> list[str]:
    """Word tokens only. Pure-number tokens go too -- they are usually [47]."""
    return [t for t in re.findall(r"[a-z0-9]+", normalise(s)) if not t.isdigit()]


def find_probe(probe: list[str], hay: list[str]) -> tuple[int, float]:
    """Return (token position, score) of the best in-order match, or (-1, best)."""
    if len(probe) < ANCHOR:
        return -1, 0.0
    span = int(len(probe) * WINDOW_SLACK) + ANCHOR
    best_pos, best_score = -1, 0.0

    # Anchor on the probe's opening tokens; if the renderer mangled those, slide
    # the anchor along the probe rather than giving up.
    for start in range(0, min(len(probe) - ANCHOR, 12) + 1, ANCHOR):
        key = probe[start : start + ANCHOR]
        for i in range(len(hay) - ANCHOR + 1):
            if hay[i : i + ANCHOR] != key:
                continue
            window_start = max(0, i - start)
            window = hay[window_start : window_start + span]
            matcher = difflib.SequenceMatcher(None, probe, window, autojunk=False)
            score = sum(b.size for b in matcher.get_matching_blocks()) / len(probe)
            if score > best_score:
                best_pos, best_score = window_start, score
            if best_score >= 1.0:
                return best_pos, best_score
        if best_score >= MATCH_THRESHOLD:
            break
    return (best_pos, best_score) if best_score >= MATCH_THRESHOLD else (-1, best_score)


# --------------------------------------------------------------------------
# junk


def junk_counts(text: str) -> dict[str, int]:
    """Three kinds of damage a rendered-page extractor leaves behind."""
    hyphen = len(re.findall(r"[a-z]{2,}-\s+[a-z]{2,}", text.casefold()))
    # the same non-trivial string repeated back-to-back: duplicated maths
    duped = len(re.findall(r"(\S{8,}(?:\s\S+){0,3})\s+\1\b", text))
    return {"hyphen_split": hyphen, "duplicated": duped}


def running_headers(doc: Loaded) -> int:
    """Short passages repeated three or more times are page furniture."""
    counts: dict[str, int] = {}
    for p in doc.passages:
        key = normalise(p.text)
        if len(key.split()) <= 15:
            counts[key] = counts.get(key, 0) + 1
    return sum(n for n in counts.values() if n >= 3)


# --------------------------------------------------------------------------
# scoring


def heading_key(s: str) -> str:
    """Compare headings without their numbering: '2 Background' == 'Background'."""
    s = normalise(s)
    s = re.sub(r"^(appendix\s+)?[0-9ivxlca]+([.)][0-9]*)*\s+", "", s)
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def score_paper(golden: dict, doc: Loaded) -> dict:
    hay_text = "\n".join(p.text for p in doc.passages)
    hay = tokens(hay_text)

    positions: list[tuple[int, int]] = []      # (probe order, token position)
    misses: list[tuple[str, float]] = []
    for probe in golden["probes"]:
        pos, score = find_probe(tokens(probe["text"]), hay)
        if pos >= 0:
            positions.append((probe["order"], pos))
        else:
            misses.append((probe["text"], round(score, 2)))

    n_probes = len(golden["probes"])
    found = len(positions) / n_probes if n_probes else 0.0

    pairs = list(itertools.pairwise(positions))
    ordered = sum(1 for a, b in pairs if b[1] > a[1])
    order = ordered / len(pairs) if pairs else 1.0

    want = {heading_key(s) for s in golden["sections"] if heading_key(s)}
    got = {heading_key(s) for s in doc.sections if heading_key(s)}
    sections = len(want & got) / len(want) if want else 0.0

    words = max(1, len(hay_text.split()))
    counts = junk_counts(hay_text)
    counts["running_header"] = running_headers(doc)
    junk = 1000 * sum(counts.values()) / words

    return {
        "arxiv_id": golden["arxiv_id"],
        "found": round(found, 3),
        "order": round(order, 3),
        "sections": round(sections, 3),
        "junk_per_1k": round(junk, 2),
        "passages": len(doc.passages),
        "words": words,
        "no_location": sum(1 for p in doc.passages if not p.location),
        "junk_detail": counts,
        "missed": misses,
        "missing_sections": sorted(want - got),
        "note": doc.note,
    }


# --------------------------------------------------------------------------
# matcher verification -- runs before any score is printed


def verify(golden: list[dict], loader, client) -> None:
    """Prove the matcher is neither too strict nor too loose, on real data."""
    paper = next(g for g in golden if g["probes"])
    doc = loader(paper["arxiv_id"], client)
    hay = tokens("\n".join(p.text for p in doc.passages))

    probe = paper["probes"][0]
    pos, score = find_probe(tokens(probe["text"]), hay)
    print("VERIFY  positive case")
    print(f"  probe : {probe['text'][:110]}")
    if pos < 0:
        print(f"  FOUND?: no (best {score:.2f}) -- matcher is too strict, stop here")
        raise SystemExit(1)
    print(f"  window: {' '.join(hay[pos : pos + len(tokens(probe['text'])) + 4])[:110]}")
    print(f"  FOUND?: yes at token {pos}, score {score:.2f}")

    rng = random.Random(0)
    scrambled = tokens(probe["text"])[:]
    rng.shuffle(scrambled)
    _, s_score = find_probe(scrambled, hay)
    other = next(g for g in golden if g["probes"] and g["arxiv_id"] != paper["arxiv_id"])
    _, f_score = find_probe(tokens(other["probes"][0]["text"]), hay)
    print("VERIFY  negative cases (both must be 'no')")
    print(f"  shuffled words of the same probe -> {'no' if s_score < MATCH_THRESHOLD else 'YES'}"
          f" ({s_score:.2f})")
    print(f"  a probe from {other['arxiv_id']}      -> "
          f"{'no' if f_score < MATCH_THRESHOLD else 'YES'} ({f_score:.2f})")
    if s_score >= MATCH_THRESHOLD or f_score >= MATCH_THRESHOLD:
        print("  matcher is too loose, stop here")
        raise SystemExit(1)
    print()


LOADERS = {"html": load_html.load}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "html"
    only = sys.argv[2:]
    loader = LOADERS[which]
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    with httpx.Client(follow_redirects=True) as client:
        verify(golden, loader, client)

        rows = []
        for g in golden:
            if only and g["arxiv_id"] not in only:
                continue
            if not g["probes"]:
                continue
            rows.append(score_paper(g, loader(g["arxiv_id"], client)))

    header = f"{'paper':<14}{'found':>7}{'order':>7}{'sect':>7}{'junk/1k':>9}{'pass':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['arxiv_id']:<14}{r['found']:>7.2f}{r['order']:>7.2f}"
              f"{r['sections']:>7.2f}{r['junk_per_1k']:>9.2f}{r['passages']:>6}")
    if len(rows) > 1:
        n = len(rows)
        print("-" * len(header))
        print(f"{'MEAN':<14}{sum(r['found'] for r in rows) / n:>7.2f}"
              f"{sum(r['order'] for r in rows) / n:>7.2f}"
              f"{sum(r['sections'] for r in rows) / n:>7.2f}"
              f"{sum(r['junk_per_1k'] for r in rows) / n:>9.2f}")

    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    existing[which] = rows
    OUT.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
