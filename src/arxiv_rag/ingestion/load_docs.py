import json
from abc import abstractmethod, ABC
from pathlib import Path

from arxiv_rag.util import application_config


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def get_docs_name(self) -> list[str]:
        """Load documents from a source."""
        pass


# For now, I am reading the sample HTML files from a local directory.
# Later the documents can be loaded from a database or a cloud storage bucket.
# Temporarily, I am using the sample HTML files from the `data/raw/sampled_html` directory.
class ArxivSampleHTMLLoader(DocumentLoader):
    """Load documents from a local directory containing arXiv HTML files."""
    ROOT = Path(__file__).parents[3]
    CACHED_HTML_DIRECTORY = ROOT / "data" / "raw" / "sampled_html"

    def get_docs_name(self) -> list[str]:
        """Load documents from the specified directory."""
        documents_name = []

        for file_path in Path(self.CACHED_HTML_DIRECTORY).glob("*.html"):
            documents_name.append(file_path.name.removesuffix(".html"))
        return documents_name


class ArxivCorpusHtmlLoader(DocumentLoader):
    """Load documents from a local directory containing arXiv HTML files."""
    ROOT = Path(__file__).parents[3]
    CORPUS_JSON_PATH = ROOT / "data" / "corpus" / "papers_300.json"

    def get_docs_name(self) -> list[str]:
        documents_name = []

        with open(self.CORPUS_JSON_PATH, "r") as file:
            data = json.load(file)
            for item in data:
                documents_name.append(item["arxiv_id"])

        return documents_name


def get_loader() -> DocumentLoader:
    """Get the appropriate document loader based on the environment."""
    if application_config()["loading"]["active"]["html_corpus"] != "PROD":
        return ArxivSampleHTMLLoader()

    return ArxivCorpusHtmlLoader()


if __name__ == "__main__":
    loader = get_loader()
    docs_name = loader.get_docs_name()
    print(f"Loaded {len(docs_name)} documents:")
    for name in docs_name:
        print(name)
