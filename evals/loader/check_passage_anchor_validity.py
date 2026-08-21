import json
from pathlib import Path

from loader.load import load

HTML_PATH = Path(__file__).parents[2] / "data" / "raw" / "html"
TEST_HTML_DATA_PATH = Path(__file__).parents[1] / "data" / "papers.json"


def check_passage_anchor_validity():
    # Load the HTML content from the HTML file
    # Convert it to the Loaded object
    # Comapre the anchors in the HTML with the locations in the Loaded object

    json_data = json.loads(TEST_HTML_DATA_PATH.read_text(encoding="utf-8"))

    loaded_anchor_found_in_html = 0
    loaded_passage_count = 0

    for paper in json_data:
        html_content = (HTML_PATH / f"{paper['arxiv_id']}.html").read_text(
            encoding="utf-8"
        )
        loaded = load(
            paper["arxiv_id"],
            None,  # pyright: ignore[reportArgumentType]
        )  # Cached benchmark HTML means the client is never used.

        for passage in loaded.passages:
            loaded_passage_count += 1
            if passage.location:
                anchor = passage.location.lstrip("#")

                if f'id="{anchor}"' in html_content:
                    loaded_anchor_found_in_html += 1
                else:
                    print(
                        f"Anchor not found in HTML: {passage.location} for passage: {passage.text[:60]}..."
                    )

    print(f"Total anchors found in HTML: {loaded_anchor_found_in_html}")
    print(
        f"Metrics Anchor Coverage: {loaded_anchor_found_in_html / loaded_passage_count * 100:.2f}%"
    )


if __name__ == "__main__":
    check_passage_anchor_validity()
