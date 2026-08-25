import json
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document

from arxiv_rag.loading.models import LoadedPaper


def _document_id(arxiv_id: str, location: str) -> str:
    source = f"https://arxiv.org/html/{arxiv_id}{location}"
    return str(uuid5(NAMESPACE_URL, source))


# This is convert a given loaded paper into a list of Documents.
def convert_loaded_paper_to_documents(loaded_paper: LoadedPaper) -> list[Document]:
    """Convert a loaded paper into retrieval documents."""
    documents = []
    for passage in loaded_paper.passages:
        doc = Document(
            id=_document_id(loaded_paper.arxiv_id, passage.location),
            page_content=passage.text,
            metadata={
                "arxiv_id": loaded_paper.arxiv_id,
                "location": passage.location,
                "order": passage.order,
                "section_path": json.dumps(passage.section_path),
                "kind": passage.kind,
                "images": json.dumps(
                    [
                        {"url": image.url, "location": image.location}
                        for image in passage.images
                    ]
                ),
            }
        )
        documents.append(doc)
    return documents
