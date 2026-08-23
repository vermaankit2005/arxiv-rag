import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).parents[2]
BENCHMARK_PAPERS_PATH = ROOT / "evals" / "dataset" / "papers.json"
RECALL_PROBES_PATH = ROOT / "evals" / "dataset" / "recall_probes_dataset.json"
CACHED_HTML_DIRECTORY = ROOT / "data" / "raw" / "sampled_html"
LANGSMITH_DATASET_NAME = "sampled_probe_recall_dataset"


def cached_html_path(arxiv_id: str) -> Path:
    """Return the cache path without allowing an id to escape the cache folder."""
    safe_filename = f"{arxiv_id.replace('/', '_')}.html"
    path = (CACHED_HTML_DIRECTORY / safe_filename).resolve()
    if path.parent != CACHED_HTML_DIRECTORY.resolve():
        raise ValueError(f"Invalid arXiv id: {arxiv_id}")
    return path


def load_local_dataset() -> tuple[list[dict], list[dict]]:
    """Read the benchmark-paper list and sampled recall probes."""
    try:
        papers = json.loads(BENCHMARK_PAPERS_PATH.read_text(encoding="utf-8"))
        probe_sets = json.loads(RECALL_PROBES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Could not read the local evaluation dataset") from error
    return papers, probe_sets


def build_examples() -> list[dict]:
    """Build one serializable LangSmith example per benchmark paper."""
    papers, probe_sets = load_local_dataset()
    probes_by_arxiv_id = {probe_set["arxiv_id"]: probe_set for probe_set in probe_sets}
    examples: list[dict] = []

    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        probe_set = probes_by_arxiv_id.get(arxiv_id)
        if probe_set is None:
            raise RuntimeError(
                f"No recall probes found for benchmark paper: {arxiv_id}"
            )

        html_content = cached_html_path(arxiv_id).read_bytes()
        examples.append(
            {
                "inputs": {"arxiv_id": arxiv_id},
                "outputs": {
                    "html_sha256": hashlib.sha256(html_content).hexdigest(),
                    "probes": [
                        {"text": probe["text"], "section": probe["section"]}
                        for probe in probe_set["probes"]
                    ],
                },
            }
        )

    return examples


def create_passage_anchor_and_probe_recall_dataset() -> None:
    """Upload the local answer key without silently replacing an existing one."""
    examples = build_examples()
    load_dotenv()
    langsmith_client = Client()

    if langsmith_client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
        existing = {
            existing_example.inputs["arxiv_id"]: existing_example.outputs
            for existing_example in langsmith_client.list_examples(dataset_name=LANGSMITH_DATASET_NAME)
        }
        expected = {
            example["inputs"]["arxiv_id"]: example["outputs"] for example in examples
        }
        if existing == expected:
            print(f"LangSmith dataset is up to date: {LANGSMITH_DATASET_NAME}")
            return
        raise RuntimeError(
            f"LangSmith dataset is stale: {LANGSMITH_DATASET_NAME}. "
            "Delete it before rebuilding so old experiments are not changed silently."
        )

    langsmith_client.create_dataset(LANGSMITH_DATASET_NAME)
    for example in examples:
        langsmith_client.create_example(
            dataset_name=LANGSMITH_DATASET_NAME,
            inputs=example["inputs"],
            outputs=example["outputs"],
        )

    print(f"Created LangSmith dataset: {LANGSMITH_DATASET_NAME}")


if __name__ == "__main__":
    create_passage_anchor_and_probe_recall_dataset()
