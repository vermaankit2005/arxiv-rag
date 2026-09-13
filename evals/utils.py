import os

from dotenv import load_dotenv

EVAL_UPLOAD_ENV_NAME = "EVAL_UPLOAD_TO_LANGSMITH"
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def eval_upload_enabled() -> bool:
    """Return the one upload policy shared by every evaluation entry point."""
    load_dotenv()
    value = os.environ.get(EVAL_UPLOAD_ENV_NAME, "false").strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise RuntimeError(f"{EVAL_UPLOAD_ENV_NAME} must be true or false.")


def print_local_score(results, feedback_key: str) -> None:
    scores = []

    for item in results:
        evaluations = item["evaluation_results"]["results"]

        for evaluation in evaluations:
            if evaluation.key != feedback_key:
                continue

            score = evaluation.score
            if not isinstance(score, (int, float, bool)):
                continue

            numeric_score = score * 1.0
            scores.append(numeric_score)
            print(f"Example {len(scores)}: {numeric_score:.4f}")

    if not scores:
        print(f"{feedback_key}: no scores returned")
        return

    total = sum(scores)
    average = total / len(scores)

    print(f"{feedback_key} completed: {len(scores)}")
    print(f"{feedback_key} average: {average:.4f}")
