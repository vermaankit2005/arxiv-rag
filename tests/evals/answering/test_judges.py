from evals import judges


def test_build_judge_model_uses_configured_chat_model(monkeypatch):
    expected_model = object()
    captured_model_name = []
    monkeypatch.setattr(
        judges,
        "get_chat_model",
        lambda model_name=None: captured_model_name.append(model_name) or expected_model,
    )

    model = judges.build_judge_model()

    assert model is expected_model
    assert captured_model_name == [None]


def test_build_judge_model_passes_a_caller_model_override(monkeypatch):
    captured_model_name = []
    monkeypatch.setattr(
        judges,
        "get_chat_model",
        lambda model_name=None: captured_model_name.append(model_name) or object(),
    )

    judges.build_judge_model("custom-model")

    assert captured_model_name == ["custom-model"]
