from streamlit.testing.v1 import AppTest

APP_PATH = "../../ui/streamlit_app.py"


def _run_app():
    """Render the app with no question asked, so nothing calls the model."""
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


def test_app_renders_its_controls_without_error():
    app = _run_app()

    assert not app.exception
    assert not app.error
    assert app.title[0].value == "arXiv reading assistant"
    assert app.chat_input[0].placeholder == "Ask about the papers"


def test_app_starts_with_an_empty_conversation_and_offers_suggestions():
    app = _run_app()

    assert app.session_state["messages"] == []
    assert len(app.pills[0].options) == 3


def test_app_hides_the_suggestions_once_the_conversation_has_started():
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.session_state["messages"] = [{"question": "Q", "answer": "A", "sources": []}]
    app.run()

    assert not app.exception
    assert not app.pills
