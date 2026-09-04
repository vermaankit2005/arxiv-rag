import os

import langsmith

# Keep every pytest run local, even when tracing was enabled by the shell or an
# older LangChain variable. Evals upload explicitly and do not run under pytest.
for variable in (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TEST_TRACKING",
):
    os.environ[variable] = "false"

# This process-wide override also wins if LangSmith cached its configuration
# before pytest loaded this conftest (for example, while loading its plugin).
langsmith.configure(enabled=False)
