"""Build papers_300.json: a hand-curated, text-only GenAI canon.

Matches each curated title against the existing 3k pool for a verified arXiv id.
Anything not in the pool is fetched directly from the arXiv API.
"""
import json, re, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
from corpus_canon import CANON

ROOT = Path(__file__).parents[1]
POOL_PATH = ROOT / "data" / "corpus" / "papers_3k.json"
OUT_PATH = ROOT / "data" / "corpus" / "papers_300.json"

# Curated papers that are not in the 3k pool. Every id below was verified
# against the live arXiv API on 2026-09-12.
MANUAL = {
    "KTO: Model Alignment as Prospect": "2402.01306",
    "WizardLM: Empowering Large Language Models to Follow Complex": "2304.12244",
    "Back to Basics: Revisiting REINFORCE": "2402.14740",
    "Chain of Thought Empowers Transformers": "2402.12875",
    "Quiet-STaR": "2403.09629",
    "Chain-of-Thought Reasoning Without Prompting": "2402.10200",
    "ColBERT: Efficient and Effective Passage Search": "2004.12832",
    "ColBERTv2": "2112.01488",
    "Precise Zero-Shot Dense Retrieval without Relevance Labels": "2212.10496",
    "RAPTOR": "2401.18059",
    "Dense X Retrieval": "2312.06648",
    "tau-bench": "2406.12045",
    "H2O: Heavy-Hitter Oracle": "2306.14048",
    "Alignment faking in large language models": "2412.14093",
    "Weak-to-Strong Generalization": "2312.09390",
    "Ring Attention": "2310.01889",
    "Textbooks Are All You Need II": "2309.05463",
    "Deduplicating Training Data Makes Language Models Better": "2107.06499",
    "Rephrasing the Web": "2401.16380",
    "The Curse of Recursion": "2305.17493",
    "Are Emergent Abilities of Large Language Models a Mirage": "2304.15004",
    "SmolLM2": "2502.02737",
    "An Explanation of In-context Learning as Implicit Bayesian": "2111.02080",
    "Model soups": "2203.05482",
}


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def fetch_arxiv(ids):
    """Return {bare_id: (versioned_id, title, submitted_date)} from the arXiv API."""
    url = "http://export.arxiv.org/api/query?id_list=" + ",".join(ids) + "&max_results=100"
    tree = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = {}
    for entry in tree.findall("a:entry", ns):
        versioned = entry.find("a:id", ns).text.split("/abs/")[-1]
        title = " ".join(entry.find("a:title", ns).text.split())
        published = entry.find("a:published", ns).text[:10]
        out[versioned.split("v")[0]] = (versioned, title, published)
    return out


pool = json.load(open(POOL_PATH, encoding="utf-8"))
pool_norm = [(norm(p["title"]), p) for p in pool]
fetched = fetch_arxiv(sorted(set(MANUAL.values())))

papers, used, problems = [], set(), []
for family, tier, query, why in CANON:
    if query in MANUAL:
        bare = MANUAL[query]
        if bare not in fetched:
            problems.append(f"arXiv API returned nothing for {bare} ({query})")
            continue
        versioned, title, submitted = fetched[bare]
        source, referenced_by, built_on_by = "arxiv_api", None, None
    else:
        q = norm(query)
        hits = [p for t, p in pool_norm if q in t and p["arxiv_id"] not in used]
        if not hits:
            problems.append(f"no pool match for {query!r}")
            continue
        hits.sort(key=lambda p: (not norm(p["title"]).startswith(q), len(p["title"])))
        best = hits[0]
        versioned, title, submitted = best["arxiv_id"], best["title"], best["submitted"]
        source = "papers_3k"
        referenced_by, built_on_by = best["referenced_by"], best["built_on_by"]

    if versioned in used:
        problems.append(f"duplicate arxiv_id {versioned} for {query!r}")
        continue
    used.add(versioned)
    papers.append({
        "arxiv_id": versioned,
        "title": title,
        "family": family,
        "tier": tier,
        "submitted": submitted,
        "why": why,
        "citations": referenced_by,
        "influential_citations": built_on_by,
        "source": source,
    })

papers.sort(key=lambda p: (p["family"], p["tier"], p["submitted"]))
json.dump(papers, open(OUT_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(f"wrote {len(papers)} papers to {OUT_PATH}")
if problems:
    print("\nPROBLEMS:")
    for p in problems:
        print("  " + p)
