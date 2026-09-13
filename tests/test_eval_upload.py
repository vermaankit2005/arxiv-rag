from types import SimpleNamespace

import pytest


def _load_eval_modules(monkeypatch):
    monkeypatch.setenv("ARXIV_RAG_RUNTIME", "")
    from evals import utils
    from evals.regression import runner
    return utils, runner


def _disable_dotenv(monkeypatch, utils) -> None:
    monkeypatch.setattr(utils, "load_dotenv", lambda: None)


def test_eval_upload_is_disabled_when_flag_is_absent(monkeypatch):
    utils, _ = _load_eval_modules(monkeypatch)
    _disable_dotenv(monkeypatch, utils)
    monkeypatch.delenv(utils.EVAL_UPLOAD_ENV_NAME, raising=False)

    assert utils.eval_upload_enabled() is False


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_eval_upload_uses_the_shared_flag(monkeypatch, value, expected):
    utils, _ = _load_eval_modules(monkeypatch)
    _disable_dotenv(monkeypatch, utils)
    monkeypatch.setenv(utils.EVAL_UPLOAD_ENV_NAME, value)

    assert utils.eval_upload_enabled() is expected


def test_eval_upload_rejects_an_invalid_flag(monkeypatch):
    utils, _ = _load_eval_modules(monkeypatch)
    _disable_dotenv(monkeypatch, utils)
    monkeypatch.setenv(utils.EVAL_UPLOAD_ENV_NAME, "sometimes")

    with pytest.raises(RuntimeError, match="EVAL_UPLOAD_TO_LANGSMITH must be true or false"):
        utils.eval_upload_enabled()


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_regression_suite_passes_the_shared_flag_to_langsmith(monkeypatch, value, expected):
    utils, runner = _load_eval_modules(monkeypatch)
    _disable_dotenv(monkeypatch, utils)
    monkeypatch.setenv(utils.EVAL_UPLOAD_ENV_NAME, value)

    class FakeClient:
        def __init__(self):
            self.upload_results = None

        def evaluate(self, target, **kwargs):
            self.upload_results = kwargs["upload_results"]
            return [{"example": SimpleNamespace(metadata={"example_id": "case-1"}, id="case-1"), "evaluation_results": {"results": [SimpleNamespace(key="score", score=1.0)]}}]

    client = FakeClient()
    monkeypatch.setattr(runner, "Client", lambda: client)
    monkeypatch.setattr(runner, "build_results", lambda suite, started_at, uploaded, passed, records: {})
    spec = {"name": "test_metric", "dataset": "test_dataset", "target": object(), "evaluator": object(), "feedback_keys": ("score",), "metadata": {}, "prefix": "test", "description": "test", "expected": 1}

    status = runner.run_suite("test", [spec])

    assert status == 0
    assert client.upload_results is expected
