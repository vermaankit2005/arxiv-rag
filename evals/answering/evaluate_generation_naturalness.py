from collections.abc import Callable

from langsmith import Client
from openevals.llm import create_llm_as_judge  # pyright: ignore[reportMissingImports]

from evals.answering import evaluate_generation_citation_support as citation_support
from evals.answering.judges import build_judge_model

LANGSMITH_DATASET_NAME = "generation_quality_dataset"
EXPERIMENT_PREFIX = "generation_naturalness"
JUDGE_MODEL_NAME = "gemma4:26b"
ALLOWED_SCORES = {0, 0.25, 0.5, 0.75, 1}
EXPERIMENT_METADATA = {
    "metric": "naturalness",
    "dataset": LANGSMITH_DATASET_NAME,
    "generator_model": "gemma4:26b",
    "judge_model": JUDGE_MODEL_NAME,
}

NATURALNESS_PROMPT = """
You are evaluating only how natural and human-written a generated research
answer sounds in response to a question.

Consider phrasing, sentence rhythm, transitions, and whether the answer reads as
a coherent explanation rather than a template or a sequence of extracted facts.
A technical answer can be formal and still sound natural. Do not reward
chattiness, jokes, enthusiasm, or extra detail. Ignore citation markers such as
[P1]. Do not score factual correctness, completeness, groundedness, citation
quality, or answer length; those are evaluated separately.

Question:
{inputs}

Generated answer:
{outputs}

Return one of these scores:
- 1: natural and effortless; it reads like an articulate person wrote it.
- 0.75: mostly natural, with minor stiffness that does not disrupt the flow.
- 0.5: understandable but noticeably mechanical, repetitive, or template-like.
- 0.25: strongly robotic or choppy, with awkward phrasing or poor transitions.
- 0: highly unnatural and difficult to read as a human-written response.
"""

Judge = Callable[..., dict]

judge_model = build_judge_model(JUDGE_MODEL_NAME)

naturalness_judge = create_llm_as_judge(
    prompt=NATURALNESS_PROMPT,
    feedback_key="naturalness",
    judge=judge_model,
    choices=[0, 0.25, 0.5, 0.75, 1],
)


def evaluate_naturalness(inputs: dict, outputs: dict, judge: Judge = naturalness_judge) -> dict:
    """Judge human-like phrasing without mixing in factual answer quality."""
    result = judge(
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
        citation_support.generate_answer_for_citation_support,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[evaluate_naturalness],
        metadata=EXPERIMENT_METADATA,
        experiment_prefix=EXPERIMENT_PREFIX,
        description=(
            "Judge whether each generated answer sounds natural and human-written. "
            "The score measures phrasing, rhythm, transitions, and freedom from "
            "robotic or template-like language on a restricted 0 to 1 scale."
        ),
        max_concurrency=4,
    )


if __name__ == "__main__":
    run_naturalness()
