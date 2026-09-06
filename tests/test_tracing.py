import os

from langsmith.utils import test_tracking_is_disabled as is_test_tracking_disabled
from langsmith.utils import tracing_is_enabled


def test_langsmith_uploads_are_disabled_during_tests(pytestconfig):
    tracing_variables = (
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2",
    )

    assert all(os.environ[variable] == "false" for variable in tracing_variables)
    assert os.environ["LANGSMITH_API_KEY"] == ""
    assert os.environ["LANGCHAIN_API_KEY"] == ""
    assert pytestconfig.pluginmanager.get_plugin("langsmith_plugin") is None
    assert tracing_is_enabled() is False
    assert is_test_tracking_disabled()
