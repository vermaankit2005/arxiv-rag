"""Evaluation package with a fixed, validated runtime profile."""

from arxiv_rag.util import activate_eval_runtime

# Importing any eval module selects the benchmark runtime before targets open a
# corpus or vector store. This also covers direct local eval commands.
activate_eval_runtime()
