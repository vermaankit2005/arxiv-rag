import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()
PAPERS_JSON_PATH = Path(__file__).parents[1] / "data" / "papers.json"
LOADED_PROBES_DATASET_PATH = Path(__file__).parents[1] / "data" / "recall_probes_dataset.json"


def create_dataset_anchor_and_recall():
    papers = json.loads(PAPERS_JSON_PATH.read_text(encoding="utf-8"))
    loaded_probes_dataset = json.loads(LOADED_PROBES_DATASET_PATH.read_text(encoding="utf-8"))

    if client.has_dataset(dataset_name="anchor_and_recall_dataset"):
        return
    client.create_dataset("anchor_and_recall_dataset")

    for paper in papers:

        # for the paper["arxiv_id"], find the probe in probes that matches the arxiv_id
        matching_dataset = next(
            (dataset for dataset in loaded_probes_dataset if dataset["arxiv_id"] == paper["arxiv_id"]), None)

        if matching_dataset is None:
            print(f"No matching probe found for paper: {paper['arxiv_id']}")
            continue

        outputs = []

        for probe in matching_dataset["probes"]:
            outputs.append({
                "text": probe["text"],
                "section": probe["section"],
            })

        client.create_example(
            dataset_name="anchor_and_recall_dataset",
            inputs={"arxiv_id": paper["arxiv_id"]},
            outputs={"probes": outputs},
        )


if __name__ == "__main__":
    create_dataset_anchor_and_recall()
