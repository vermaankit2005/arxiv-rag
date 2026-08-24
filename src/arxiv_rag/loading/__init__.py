"""Load papers into citation-aware application models."""

from arxiv_rag.loading.arxiv import load_paper
from arxiv_rag.loading.models import FigureImage, LoadedPaper, Passage

__all__ = ["FigureImage", "LoadedPaper", "Passage", "load_paper"]
