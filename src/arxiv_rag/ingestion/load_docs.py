from abc import abstractmethod, ABC
from pathlib import Path


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
            documents_name.append(file_path.name)
        return documents_name


if __name__ == "__main__":
    loader = ArxivSampleHTMLLoader()
    docs_name = loader.get_docs_name()
    print(f"Loaded {len(docs_name)} documents:")
    for name in docs_name:
        print(name)
