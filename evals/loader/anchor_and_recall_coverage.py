from pathlib import Path

import httpx
from dotenv import load_dotenv
from langsmith import Client

from loader.load import load

load_dotenv()

client = Client()
HTML_PATH = Path(__file__).parents[2] / "data" / "raw" / "html"
TEST_HTML_DATA_PATH = Path(__file__).parents[1] / "data" / "papers.json"


def target(inputs: dict) -> dict:
    arxiv_id = inputs["arxiv_id"]
    with httpx.Client() as http_client:
        loaded = load(arxiv_id, http_client)
        return {
            "anchors_from_loaded_passages": [p.location.lstrip("#") for p in loaded.passages],
            "passages": loaded.passages
        }


def evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    arxiv_id = inputs["arxiv_id"]
    html_content = (HTML_PATH / f"{arxiv_id}.html").read_text(encoding="utf-8")

    loaded_anchor_found_in_html = 0
    loaded_passage_count = 0
    anchors_from_loaded_passages = outputs["anchors_from_loaded_passages"]

    # Calculate the coverage score.
    for anchor in anchors_from_loaded_passages:
        loaded_passage_count += 1
        if f'id="{anchor}"' in html_content:
            loaded_anchor_found_in_html += 1

    coverage_score = loaded_anchor_found_in_html / loaded_passage_count if loaded_passage_count > 0 else 0.0

    # Calculate the recall score.
    probes_found_in_loaded_passages = 0

    for reference_output in reference_outputs["probes"]:
        probe_text = reference_output["text"]
        print(f"Checking probe: {probe_text}")
        for passage in outputs["passages"]:
            if probe_text in passage.text:
                print(f"Probe found in passage: {passage.text}")
                probes_found_in_loaded_passages += 1
                break
        print("\n \n --------------------------------- \n \n")



    recall_score = probes_found_in_loaded_passages / len(reference_outputs["probes"]) if reference_outputs[
        "probes"] else 0.0

    return [
        {
            "key": "anchor_coverage",
            "score": coverage_score,
        },
        {
            "key": "text_recall",
            "score": recall_score,
        }
    ]


if __name__ == "__main__":
    client.evaluate(
        target,
        data="anchor_and_recall_dataset",
        evaluators=[evaluator],
        experiment_prefix="anchor_and_recall_dataset"
    )
