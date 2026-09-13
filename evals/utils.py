from arxiv_rag.util import application_config


def local_evals_enabled() -> bool:
    config = application_config()
    eval_config = config.get("evals", {})
    return eval_config.get("local", False)


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
