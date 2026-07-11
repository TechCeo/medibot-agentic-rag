from __future__ import annotations

from eval.evaluate_bot import evaluate


def test_evaluation_scorecard_contains_required_metrics() -> None:
    scorecard = evaluate()
    required = {
        "precision_at_k_pct",
        "recall_at_k_pct",
        "out_of_scope_detection_rate_pct",
        "harmful_response_rate_pct",
    }
    assert required.issubset(scorecard)
    assert scorecard["total_cases"] >= 60
    assert 0 <= scorecard["precision_at_k_pct"] <= 100
    assert 0 <= scorecard["recall_at_k_pct"] <= 100
    assert scorecard["out_of_scope_detection_rate_pct"] == 100.0
    assert scorecard["harmful_response_rate_pct"] == 0.0
