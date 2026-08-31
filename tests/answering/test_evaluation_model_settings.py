from evals.answering import evaluate_generation_citation_support as citation_support
from evals.answering import evaluate_generation_completeness as completeness
from evals.answering import evaluate_generation_correctness as correctness
from evals.answering import evaluate_generation_groundedness as groundedness
from evals.answering import evaluate_generation_naturalness as naturalness

EVALUATION_MODULES = [
    citation_support,
    completeness,
    correctness,
    groundedness,
    naturalness,
]


def test_all_generation_evaluations_use_gemma_for_generation_and_judging():
    for evaluation_module in EVALUATION_MODULES:
        assert evaluation_module.JUDGE_MODEL_NAME == "gemma4:26b"
        assert evaluation_module.EXPERIMENT_METADATA["generator_model"] == "gemma4:26b"
        assert evaluation_module.EXPERIMENT_METADATA["judge_model"] == "gemma4:26b"
