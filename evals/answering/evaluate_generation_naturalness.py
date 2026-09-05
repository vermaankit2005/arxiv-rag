"""This checks how natural the generated answer feels to a user.

It looks at whether the writing is direct and easy to read, not whether the
facts are correct. Each answer gets 0, 0.25, 0.5, 0.75, or 1. The final score
is the average across all questions.
"""

from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from evals.answering import context as evaluation_context
from arxiv_rag.ollama_config import get_generator_model, get_judge_model
from evals.judges import build_judge_model

DESCRIPTION = __doc__
LANGSMITH_DATASET_NAME = "generation_quality_dataset"
EXPERIMENT_PREFIX = "generation_naturalness"
NATURALNESS_RUBRIC_VERSION = "user-facing-v2"
ALLOWED_SCORES = {0, 0.25, 0.5, 0.75, 1}
EXPERIMENT_METADATA = {
    "metric": "naturalness",
    "rubric_version": NATURALNESS_RUBRIC_VERSION,
    "evaluation_focus": "user-facing readability and interaction",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": get_generator_model(),
    "judge_model": get_judge_model(),
    "judge_thinking": "disabled",
    "generator_thinking": "disabled",
}

NATURALNESS_PROMPT = """
You are evaluating how natural a generated research answer feels to a user who
is reading it and interacting with an assistant.

Judge whether the answer is direct, smoothly written, and pleasant to read. It
should synthesize the requested information instead of looking like facts copied
into a standard response template. Technical language is appropriate when the
question requires it, but a polished report is not automatically a natural
assistant response.

Reduce the score for unnecessary headings, excessive bullet lists, canned
introductions such as "the following results were reported," repetitive sentence
patterns, fragmented facts, or formatting that makes a short answer feel like a
report. Bullets are acceptable when they genuinely improve a complex answer, but
a list question does not automatically make a rigid list feel natural. Do not
reward verbosity, chattiness, jokes, enthusiasm, or extra detail.

Ignore citation markers such as [P1] when judging the prose. Do not score factual
correctness, completeness, groundedness, or citation support; those are evaluated
separately.

Question:
{inputs}

Generated answer:
{outputs}

Return one of these scores:
- 1: effortless, direct, and pleasant to read as a user-facing assistant response.
- 0.75: natural overall, but somewhat formal or structured in a way a user may notice.
- 0.5: clear and understandable, but obviously generated, templated, or report-like.
- 0.25: strongly robotic, fragmented, repetitive, or overloaded with formatting.
- 0: highly unnatural and difficult for a user to read or interact with.
"""

judge_model = build_judge_model()

naturalness_judge = create_llm_as_judge(
    prompt=NATURALNESS_PROMPT,
    feedback_key="naturalness",
    judge=judge_model,
    choices=[0, 0.25, 0.5, 0.75, 1],
)

def evaluate_naturalness(inputs: dict, outputs: dict) -> dict:
    """Judge human-like phrasing without mixing in factual answer quality."""
    result = naturalness_judge(
        inputs={"question": inputs.get("question", "")},
        outputs={"answer": outputs.get("answer", "")},
    )

    score = result.get("score")
    if score not in ALLOWED_SCORES:
        raise ValueError(f"Naturalness judge returned an invalid score: {score!r}")

    return {
        "key": "naturalness",
        "score": score,
        "comment": result.get("comment", "The answer's naturalness was judged."),
    }

def run_naturalness() -> None:
    """Run naturalness evaluation against the frozen generation dataset."""
    client = Client()
    client.evaluate(
        evaluation_context.generate_answer_for_evaluation,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_naturalness],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=DESCRIPTION,
        max_concurrency=1,
    )

if __name__ == "__main__":
    run_naturalness()
