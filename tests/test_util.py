import os

import pytest

from arxiv_rag import util


def _write_config(path, eval_html_corpus="SAMPLE", eval_vector_store="CHROMA"):
    path.write_text(
        f"""
evals:
  upload_results: false
  runtime:
    html_corpus: {eval_html_corpus}
    vector_store: {eval_vector_store}
runtime:
  html_corpus: PROD
  vector_store: WEAVIATE
""".strip(),
        encoding="utf-8",
    )


def test_eval_runtime_overrides_application_runtime(monkeypatch, tmp_path):
    application_file = tmp_path / "application.yaml"
    _write_config(application_file)
    monkeypatch.setattr(util, "APPLICATION_FILE", application_file)
    monkeypatch.setenv(util.RUNTIME_ENV_NAME, util.EVAL_RUNTIME_NAME)

    config = util.application_config()

    assert config["runtime"] == {
        "html_corpus": "SAMPLE",
        "vector_store": "CHROMA",
    }


def test_activate_eval_runtime_selects_eval_profile(monkeypatch, tmp_path):
    application_file = tmp_path / "application.yaml"
    _write_config(application_file)
    monkeypatch.setattr(util, "APPLICATION_FILE", application_file)
    monkeypatch.delenv(util.RUNTIME_ENV_NAME, raising=False)

    config = util.activate_eval_runtime()

    assert config["runtime"]["html_corpus"] == "SAMPLE"
    assert config["runtime"]["vector_store"] == "CHROMA"
    assert os.environ[util.RUNTIME_ENV_NAME] == util.EVAL_RUNTIME_NAME


def test_eval_runtime_rejects_non_sample_corpus(monkeypatch, tmp_path):
    application_file = tmp_path / "application.yaml"
    _write_config(application_file, eval_html_corpus="PROD")
    monkeypatch.setattr(util, "APPLICATION_FILE", application_file)
    monkeypatch.setenv(util.RUNTIME_ENV_NAME, util.EVAL_RUNTIME_NAME)

    with pytest.raises(RuntimeError, match="html_corpus=SAMPLE"):
        util.application_config()


def test_application_config_rejects_unknown_runtime(monkeypatch, tmp_path):
    application_file = tmp_path / "application.yaml"
    _write_config(application_file)
    monkeypatch.setattr(util, "APPLICATION_FILE", application_file)
    monkeypatch.setenv(util.RUNTIME_ENV_NAME, "production")

    with pytest.raises(RuntimeError, match="Unknown ARXIV_RAG_RUNTIME value"):
        util.application_config()
