from arxiv_rag.logging import get_logger


def test_get_logger_does_not_duplicate_the_project_prefix():
    assert get_logger("arxiv_rag.loading.arxiv").name == "arxiv_rag.loading.arxiv"
