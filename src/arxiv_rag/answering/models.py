from dataclasses import dataclass
from typing import Literal

from arxiv_rag.answering import AnswerMode
from arxiv_rag.graph.workflow_graph import WorkflowGraphState
from arxiv_rag.retrieval import RetrievalContext


@dataclass(frozen=True)
class AnsweredQuestion:
    thread_id: str
    answer: str
    context: RetrievalContext
    passages_by_id: dict[str, str]
    answer_type: Literal["chat", "rag"]
    answer_mode: AnswerMode


@dataclass(frozen=True)
class AnswerChunk:
    text: str


@dataclass(frozen=True)
class AnswerComplete:
    result: AnsweredQuestion


AnswerEvent = AnswerChunk | AnswerComplete
