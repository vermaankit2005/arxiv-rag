"""Evidence-grounded tools for reading arXiv papers."""

from dotenv import load_dotenv

# LangSmith reads its settings from the environment the moment the first traced
# function runs, so .env has to be loaded before any of them can be called.
load_dotenv()
