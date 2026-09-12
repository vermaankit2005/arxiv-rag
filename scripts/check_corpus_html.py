"""Probe https://arxiv.org/html/<id> for every paper in papers_300.json."""
import json, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

CORPUS = Path(__file__).parents[1] / "data" / "corpus" / "papers_300.json"
PAPERS = json.load(open(CORPUS, encoding="utf-8"))


def probe(paper):
    bare = paper["arxiv_id"].split("v")[0]
    url = f"https://arxiv.org/html/{bare}"
    req = urllib.request.Request(url, headers={"User-Agent": "arxiv-rag-corpus-check/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            body = response.read()
        ok = response.status == 200 and b"ltx_" in body
        return paper, ok, len(body)
    except urllib.error.HTTPError as error:
        return paper, False, f"HTTP {error.code}"
    except Exception as error:
        return paper, False, type(error).__name__


with ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(probe, PAPERS))

failures = [(p, info) for p, ok, info in results if not ok]
print(f"html ok: {len(results) - len(failures)}/{len(results)}")
if failures:
    print("\nNO HTML:")
    for paper, info in failures:
        print(f"  {paper['arxiv_id']:14s} {info}  {paper['title'][:65]}")
