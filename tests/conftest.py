import os

# pytest loads this before any test module, so the flag is already off when
# arxiv_rag calls load_dotenv() (which never overrides what is already set).
# Without it, every unit test would ship a fake run to LangSmith.
os.environ["LANGSMITH_TRACING"] = "false"
