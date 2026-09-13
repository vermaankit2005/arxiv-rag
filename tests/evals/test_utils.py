from types import SimpleNamespace

from evals import utils


def test_local_evals_enabled_reads_application_config(monkeypatch) -> None:
    monkeypatch.setattr(utils, "application_config", lambda: {"evals": {"local": True}})

    assert utils.local_evals_enabled()


def test_local_evals_enabled_defaults_to_false(monkeypatch) -> None:
    monkeypatch.setattr(utils, "application_config", dict)

    assert not utils.local_evals_enabled()


def test_print_local_score_prints_each_score_and_average(capsys) -> None:
    results = [
        {
            "evaluation_results": {
                "results": [SimpleNamespace(key="fact_citation", score=1)]
            }
        },
        {
            "evaluation_results": {
                "results": [SimpleNamespace(key="fact_citation", score=0.5)]
            }
        },
    ]

    utils.print_local_score(results, "fact_citation")

    assert capsys.readouterr().out == (
        "Example 1: 1.0000\n"
        "Example 2: 0.5000\n"
        "fact_citation completed: 2\n"
        "fact_citation average: 0.7500\n"
    )
