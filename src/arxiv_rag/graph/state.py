# pyright: reportMissingImports=false
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from arxiv_rag.retrieval import BuiltContext


class WorkflowGraphState(TypedDict):
    original_question: str
    messages: Annotated[list[BaseMessage], add_messages]
    route: Literal["chat", "rag"] | None
    answer_request: str | None
    retrieval_query: str | None
    answer_mode: Literal["standard", "easy"]
    current_context: BuiltContext | None
    answer: str
