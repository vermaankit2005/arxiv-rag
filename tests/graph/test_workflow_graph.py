from typing import Literal

import pytest  # pyright: ignore[reportMissingImports]

from arxiv_rag.answering import AnswerMode
from arxiv_rag.graph import workflow_graph


class FakeRouterModel:
    def __init__(self, output: workflow_graph.RouterNodeOutput):
        self.output = output

    def with_structured_output(self, schema):
        assert schema is workflow_graph.RouterNodeOutput
        return self

    def invoke(self, messages):
        return self.output


def _state(answer_mode: AnswerMode) -> workflow_graph.WorkflowGraphState:
    return {
        "original_question": "Explain this in simple words.",
        "messages": [],
        "route": "rag",
        "rewritten_question": None,
        "retrieval_query": None,
        "answer_mode": answer_mode,
        "current_evidence": None,
        "current_built_context": None,
        "answer": "",
    }


@pytest.mark.parametrize(
    ("selected_mode", "style_override", "expected_mode"),
    [
        ("standard", "easy", "easy"),
        ("easy", None, "easy"),
        ("standard", None, "standard"),
    ],
)
def test_router_applies_only_an_easy_style_override(
    monkeypatch,
    selected_mode: AnswerMode,
    style_override: Literal["easy"] | None,
    expected_mode: AnswerMode,
):
    model = FakeRouterModel(
        workflow_graph.RouterNodeOutput(
            route="rag",
            rewritten_question="Explain multi-head attention in simple terms.",
            retrieval_query="What is multi-head attention?",
            style_override=style_override,
        )
    )
    monkeypatch.setattr(workflow_graph, "get_chat_model", lambda: model)

    result = workflow_graph.route_node(_state(selected_mode))

    assert result["answer_mode"] == expected_mode
    assert result["retrieval_query"] == "What is multi-head attention?"
