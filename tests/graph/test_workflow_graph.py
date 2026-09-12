from typing import Literal
from uuid import uuid4

import pytest  # pyright: ignore[reportMissingImports]
from langchain_core.messages import (  # pyright: ignore[reportMissingImports]
    AIMessage,
    HumanMessage,
    SystemMessage,
)

import arxiv_rag.graph.chat_node as chat_node_module
import arxiv_rag.graph.rag_node as rag_node_module
import arxiv_rag.graph.route_node as route_node_module
from arxiv_rag.answering import AnswerMode
from arxiv_rag.graph import workflow_graph
from arxiv_rag.retrieval import BuiltContext, Citation, RetrievalContext


class FakeRouterModel:
    def __init__(self, output: object):
        self.output = output
        self.messages = None

    def with_structured_output(self, schema):
        assert schema is route_node_module.RouterNodeOutput
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.output


class FakeConversationModel:
    def __init__(
        self,
        router_outputs: list[object],
        chat_replies: list[str] | None = None,
    ):
        self.router_outputs = list(router_outputs)
        self.chat_replies = list(chat_replies or [])
        self.router_prompts = []
        self.chat_prompts = []

    def with_structured_output(self, schema):
        assert schema is route_node_module.RouterNodeOutput
        return FakeStructuredRouter(self)

    def invoke(self, messages):
        self.chat_prompts.append(messages)
        return AIMessage(content=self.chat_replies.pop(0))


class FakeStructuredRouter:
    def __init__(self, model: FakeConversationModel):
        self.model = model

    def invoke(self, messages):
        self.model.router_prompts.append(messages)
        return self.model.router_outputs.pop(0)


class FailingModel:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        raise OSError("model unavailable")


class InvalidRouterOutputModel:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        raise ValueError("invalid structured output")


def _stub_chat_model(monkeypatch, model):
    monkeypatch.setattr(route_node_module, "get_chat_model", lambda: model)
    monkeypatch.setattr(chat_node_module, "get_chat_model", lambda: model)


def _router_output(
    route: Literal["chat", "rag"],
    answer_request: str | None = None,
    retrieval_query: str | None = None,
    style_override: Literal["easy"] | None = None,
) -> object:
    return route_node_module.RouterNodeOutput(
        route=route,
        answer_request=answer_request if route == "rag" else "",
        retrieval_query=retrieval_query if route == "rag" else "",
        style_override=style_override,
    )


def _state(
    answer_mode: AnswerMode = "standard",
    question: str = "Explain this in simple words.",
    messages=None,
) -> workflow_graph.WorkflowGraphState:
    return {
        "original_question": question,
        "messages": list(messages or []),
        "route": "rag",
        "answer_request": None,
        "retrieval_query": None,
        "answer_mode": answer_mode,
        "current_context": None,
        "answer": "",
    }


def _built_context(text: str = "Transformers use attention.") -> BuiltContext:
    return BuiltContext(
        context=RetrievalContext(
            text=f"[P1]\nSection: Introduction\nText: {text}",
            citations={
                "P1": Citation(
                    label="paper — Introduction",
                    url="https://arxiv.org/html/paper#S1",
                )
            },
        ),
        passages_by_id={"P1": text},
    )


def _stub_rag_dependencies(monkeypatch, answers: list[str] | None = None):
    built = _built_context()
    retrieval_queries = []
    generation_calls = []
    generated_answers = list(answers or ["Grounded answer [P1]."])

    class FakeRetriever:
        def retrieve(self, question: str) -> BuiltContext:
            retrieval_queries.append(question)
            return built

        def close(self) -> None:
            pass

    def fake_generate_answer(answer_request, context, answer_mode="standard"):
        generation_calls.append(
            {
                "answer_request": answer_request,
                "context": context,
                "answer_mode": answer_mode,
            }
        )
        if len(generated_answers) > 1:
            return generated_answers.pop(0)
        return generated_answers[0]

    monkeypatch.setattr(rag_node_module, "PaperRetriever", FakeRetriever)
    monkeypatch.setattr(rag_node_module, "generate_answer", fake_generate_answer)
    return built, retrieval_queries, generation_calls


@pytest.mark.parametrize(
    ("route", "answer_request", "retrieval_query"),
    [
        ("chat", "", ""),
        ("rag", "Explain multi-head attention.", "What is multi-head attention?"),
    ],
)
def test_router_returns_both_supported_routes(
    monkeypatch,
    route: Literal["chat", "rag"],
    answer_request: str,
    retrieval_query: str,
):
    model = FakeRouterModel(_router_output(route, answer_request, retrieval_query))
    _stub_chat_model(monkeypatch, model)

    result = route_node_module.route_node(_state(question="Current question"))

    assert result["route"] == route
    assert result["answer_request"] == answer_request
    assert result["retrieval_query"] == retrieval_query
    assert model.messages is not None
    assert isinstance(model.messages[0], SystemMessage)
    assert model.messages[0].content == route_node_module.ROUTER_SYSTEM_PROMPT
    assert "Current question" in model.messages[1].content


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
        _router_output(
            "rag",
            "Explain multi-head attention in simple terms.",
            "What is multi-head attention?",
            style_override,
        )
    )
    _stub_chat_model(monkeypatch, model)

    result = route_node_module.route_node(_state(selected_mode))

    assert result["answer_mode"] == expected_mode
    assert result["answer_request"] == "Explain multi-head attention in simple terms."
    assert result["retrieval_query"] == "What is multi-head attention?"


def test_router_receives_conversation_history_separately_from_the_current_message(monkeypatch):
    model = FakeRouterModel(
        _router_output("rag", "Explain the encoder.", "Transformer encoder")
    )
    _stub_chat_model(monkeypatch, model)
    messages = [
        HumanMessage(content="What is attention?"),
        AIMessage(content="Attention weighs relevant tokens [P1]."),
    ]

    route_node_module.route_node(
        _state(question="What about the encoder?", messages=messages)
    )

    assert model.messages is not None
    prompt = model.messages[1].content
    assert "What is attention?" in prompt
    assert "Attention weighs relevant tokens [P1]." in prompt
    assert "What about the encoder?" in prompt
    assert prompt.index("Conversation history:") < prompt.index("Current user message:")


def test_router_wraps_model_failures(monkeypatch):
    _stub_chat_model(monkeypatch, FailingModel())

    with pytest.raises(RuntimeError, match="Could not generate an answer") as raised:
        route_node_module.route_node(_state())

    assert isinstance(raised.value.__cause__, OSError)


@pytest.mark.parametrize(
    "output",
    [
        object(),
        _router_output("rag"),
    ],
)
def test_invalid_router_response_falls_back_to_rag_with_raw_question(
    monkeypatch, output: object
):
    _stub_chat_model(monkeypatch, FakeRouterModel(output))
    state = _state(answer_mode="easy", question="What is attention?")

    result = route_node_module.route_node(state)

    assert result == {
        "route": "rag",
        "answer_request": "What is attention?",
        "retrieval_query": "What is attention?",
        "answer_mode": "easy",
    }


def test_invalid_structured_router_output_falls_back_to_rag(monkeypatch):
    _stub_chat_model(monkeypatch, InvalidRouterOutputModel())

    result = route_node_module.route_node(_state(question="What is attention?"))

    assert result["route"] == "rag"
    assert result["answer_request"] == "What is attention?"
    assert result["retrieval_query"] == "What is attention?"


@pytest.mark.parametrize(
    ("route", "expected_node"),
    [("chat", "chat_node"), ("rag", "rag_node")],
)
def test_route_edge_selects_the_expected_node(route, expected_node):
    state = _state()
    state["route"] = route

    assert workflow_graph.route_edge(state) == expected_node


def test_route_edge_rejects_an_unknown_route():
    state = _state()
    state["route"] = "unknown"  # pyright: ignore[reportGeneralTypeIssues]

    with pytest.raises(ValueError, match="Invalid route: unknown"):
        workflow_graph.route_edge(state)


def test_chat_node_uses_history_and_returns_no_evidence(monkeypatch):
    model = FakeConversationModel([], ["Hi! I can help with the ingested papers."])
    _stub_chat_model(monkeypatch, model)
    state = _state(
        question="What can you do?",
        messages=[HumanMessage(content="Hi"), AIMessage(content="Hello!")],
    )
    state["current_context"] = _built_context()

    result = chat_node_module.chat_node(state)

    assert result["answer"] == "Hi! I can help with the ingested papers."
    assert result["current_context"] is None
    assert [message.content for message in result["messages"]] == [
        "What can you do?",
        "Hi! I can help with the ingested papers.",
    ]
    assert model.chat_prompts[0][0].content == chat_node_module.CHAT_SYSTEM_PROMPT
    assert "Hi" in model.chat_prompts[0][1].content
    assert "What can you do?" in model.chat_prompts[0][1].content


def test_chat_node_wraps_model_failures(monkeypatch):
    _stub_chat_model(monkeypatch, FailingModel())

    with pytest.raises(RuntimeError, match="Could not generate an answer") as raised:
        chat_node_module.chat_node(_state(question="Hi"))

    assert isinstance(raised.value.__cause__, OSError)


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ("Read more at https://example.com.", "model-written URLs"),
        ("The answer is in [P1].", "passage markers"),
        ("The answer is in [P1, P2].", "passage markers"),
    ],
)
def test_chat_node_rejects_urls_and_passage_markers(monkeypatch, reply, error):
    model = FakeConversationModel([], [reply])
    _stub_chat_model(monkeypatch, model)

    with pytest.raises(RuntimeError, match=error):
        chat_node_module.chat_node(_state(question="Hi"))


def test_rag_node_retrieves_and_answers(monkeypatch):
    built, retrieval_queries, generation_calls = _stub_rag_dependencies(monkeypatch)
    state = _state(answer_mode="easy", question="Explain it simply")
    state["answer_request"] = "Explain multi-head attention simply."
    state["retrieval_query"] = "What is multi-head attention?"

    result = rag_node_module.rag_node(state)

    assert retrieval_queries == ["What is multi-head attention?"]
    assert generation_calls == [{
        "answer_request": "Explain multi-head attention simply.",
        "context": built.context,
        "answer_mode": "easy",
    }]
    assert result["answer"] == "Grounded answer [P1]."
    assert result["current_context"] is built


def test_rag_node_closes_the_retriever_when_retrieval_fails(monkeypatch):
    closed = []

    class FailingRetriever:
        def retrieve(self, question):
            raise RuntimeError("retrieval failed")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(rag_node_module, "PaperRetriever", FailingRetriever)
    state = _state()
    state["answer_request"] = "Explain attention."
    state["retrieval_query"] = "attention"

    with pytest.raises(RuntimeError, match="retrieval failed"):
        rag_node_module.rag_node(state)

    assert closed == [True]


def test_rag_node_requires_query():
    state = _state()
    state["answer_request"] = "Explain attention."

    with pytest.raises(ValueError, match="retrieval_query must not be None"):
        rag_node_module.rag_node(state)


def test_rag_node_requires_answer_request():
    state = _state()
    state["retrieval_query"] = "attention"

    with pytest.raises(ValueError, match="answer_request must not be None"):
        rag_node_module.rag_node(state)


def test_invoke_resets_turn_only_state(monkeypatch):
    captured = {}

    class FakeCompiledGraph:
        def invoke(self, input, config):
            captured.update(input=input, config=config)
            return input

    monkeypatch.setattr(workflow_graph, "workflow_graph", FakeCompiledGraph())

    result = workflow_graph.invoke_workflow_graph(
        "What is attention?",
        "conversation-1",
        answer_mode="easy",
    )

    assert result["original_question"] == "What is attention?"
    assert result["route"] is None
    assert result["answer_request"] is None
    assert result["retrieval_query"] is None
    assert result["current_context"] is None
    assert result["answer"] == ""
    assert result["answer_mode"] == "easy"
    assert captured["config"] == {"configurable": {"thread_id": "conversation-1"}}


def test_default_retriever_is_not_constructed_for_a_chat_turn(monkeypatch):
    model = FakeConversationModel([_router_output("chat")], ["Hello!"])
    _stub_chat_model(monkeypatch, model)

    class UnexpectedRetriever:
        def __init__(self):
            raise AssertionError("A chat turn must not construct the paper retriever")

    monkeypatch.setattr(rag_node_module, "PaperRetriever", UnexpectedRetriever)

    result = workflow_graph.invoke_workflow_graph("Hi", f"test-lazy-{uuid4()}")

    assert result["route"] == "chat"
    assert result["answer"] == "Hello!"


def test_compiled_graph_chat_route_never_retrieves(monkeypatch):
    model = FakeConversationModel(
        [_router_output("chat")],
        ["Hello! Ask me about the ingested papers."],
    )
    _stub_chat_model(monkeypatch, model)
    _, retrieval_queries, generation_calls = _stub_rag_dependencies(monkeypatch)

    result = workflow_graph.invoke_workflow_graph("Hi", f"test-chat-{uuid4()}")

    assert result["route"] == "chat"
    assert result["answer"] == "Hello! Ask me about the ingested papers."
    assert result["current_context"] is None
    assert retrieval_queries == []
    assert generation_calls == []


def test_compiled_graph_rag_route_returns_current_context(monkeypatch):
    model = FakeConversationModel(
        [
            _router_output(
                "rag",
                "Explain multi-head attention.",
                "What is multi-head attention?",
            )
        ]
    )
    _stub_chat_model(monkeypatch, model)
    built, retrieval_queries, generation_calls = _stub_rag_dependencies(monkeypatch)

    result = workflow_graph.invoke_workflow_graph(
        "What is multi-head attention?", f"test-rag-{uuid4()}"
    )

    assert result["route"] == "rag"
    assert result["answer"] == "Grounded answer [P1]."
    current_context = result["current_context"]
    assert current_context is not None
    assert current_context is built
    assert current_context.passages_by_id == {
        "P1": "Transformers use attention."
    }
    assert retrieval_queries == ["What is multi-head attention?"]
    assert generation_calls[0]["answer_request"] == "Explain multi-head attention."


def test_same_thread_retains_messages_for_a_follow_up(monkeypatch):
    model = FakeConversationModel(
        [
            _router_output(
                "rag", "Explain attention.", "What is attention?"
            ),
            _router_output(
                "rag", "Explain the Transformer encoder.", "Transformer encoder"
            ),
        ]
    )
    _stub_chat_model(monkeypatch, model)
    _stub_rag_dependencies(
        monkeypatch,
        ["Attention answer [P1].", "Encoder answer [P1]."],
    )
    thread_id = f"test-same-thread-{uuid4()}"

    workflow_graph.invoke_workflow_graph("What is attention?", thread_id)
    result = workflow_graph.invoke_workflow_graph("What about the encoder?", thread_id)

    assert [message.content for message in result["messages"]] == [
        "What is attention?",
        "Attention answer [P1].",
        "What about the encoder?",
        "Encoder answer [P1].",
    ]
    second_router_prompt = model.router_prompts[1][1].content
    assert "What is attention?" in second_router_prompt
    assert "Attention answer [P1]." in second_router_prompt


def test_different_thread_starts_without_previous_messages(monkeypatch):
    model = FakeConversationModel(
        [
            _router_output("chat"),
            _router_output("chat"),
        ],
        ["First reply", "Second reply"],
    )
    _stub_chat_model(monkeypatch, model)
    _stub_rag_dependencies(monkeypatch)

    workflow_graph.invoke_workflow_graph("Hi from the first thread", f"thread-a-{uuid4()}")
    result = workflow_graph.invoke_workflow_graph(
        "Hi from the second thread", f"thread-b-{uuid4()}"
    )

    assert [message.content for message in result["messages"]] == [
        "Hi from the second thread",
        "Second reply",
    ]
    second_history = model.router_prompts[1][1].content.split("Current user message:")[0]
    assert "Hi from the first thread" not in second_history
    assert "First reply" not in second_history


def test_each_rag_follow_up_retrieves_fresh_evidence(monkeypatch):
    model = FakeConversationModel(
        [
            _router_output("rag", "Explain attention.", "attention"),
            _router_output(
                "rag",
                "Explain attention in simple words.",
                "attention",
                "easy",
            ),
        ]
    )
    _stub_chat_model(monkeypatch, model)
    _, retrieval_queries, generation_calls = _stub_rag_dependencies(
        monkeypatch,
        ["Attention answer [P1].", "Simple attention answer [P1]."],
    )
    thread_id = f"test-fresh-retrieval-{uuid4()}"

    workflow_graph.invoke_workflow_graph("Explain attention", thread_id)
    result = workflow_graph.invoke_workflow_graph("Explain that simply", thread_id)

    assert retrieval_queries == ["attention", "attention"]
    assert [call["answer_request"] for call in generation_calls] == [
        "Explain attention.",
        "Explain attention in simple words.",
    ]
    assert generation_calls[1]["answer_mode"] == "easy"
    assert result["answer"] == "Simple attention answer [P1]."


def test_chat_turn_clears_the_previous_turns_exposed_evidence(monkeypatch):
    model = FakeConversationModel(
        [
            _router_output("rag", "Explain attention.", "attention"),
            _router_output("chat"),
        ],
        ["You're welcome!"],
    )
    _stub_chat_model(monkeypatch, model)
    _stub_rag_dependencies(monkeypatch)
    thread_id = f"test-clear-evidence-{uuid4()}"

    first = workflow_graph.invoke_workflow_graph("Explain attention", thread_id)
    second = workflow_graph.invoke_workflow_graph("Thanks", thread_id)

    assert first["current_context"] is not None
    assert second["route"] == "chat"
    assert second["current_context"] is None
