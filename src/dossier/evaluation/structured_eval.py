from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

from ..common import coerce_gold_answer, normalize_mapping_values, _normalize_scalar


def evaluate_prediction_against_gold(prediction: Any, gold_answer: Any) -> Dict[str, Any]:
    if isinstance(gold_answer, dict) and isinstance(prediction, dict):
        normalized_prediction = {key: normalize_mapping_values(values) for key, values in prediction.items()}
        normalized_gold = {key: normalize_mapping_values(values) for key, values in gold_answer.items()}
        gold_pairs = {(key, value) for key, values in normalized_gold.items() for value in values}
        pred_pairs = {(key, value) for key, values in normalized_prediction.items() for value in values}
        true_positive = len(gold_pairs & pred_pairs)
        precision = true_positive / len(pred_pairs) if pred_pairs else 1.0
        recall = true_positive / len(gold_pairs) if gold_pairs else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "exact_match": normalized_prediction == normalized_gold,
            "pair_precision": precision,
            "pair_recall": recall,
            "pair_f1": f1,
            "missing": sorted(gold_pairs - pred_pairs),
            "extra": sorted(pred_pairs - gold_pairs),
        }

    if isinstance(gold_answer, list) and isinstance(prediction, list):
        normalized_prediction = normalize_mapping_values(prediction)
        normalized_gold = normalize_mapping_values(gold_answer)
        pred_set = set(normalized_prediction)
        gold_set = set(normalized_gold)
        true_positive = len(pred_set & gold_set)
        precision = true_positive / len(pred_set) if pred_set else 1.0
        recall = true_positive / len(gold_set) if gold_set else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "exact_match": normalized_prediction == normalized_gold,
            "pair_precision": precision,
            "pair_recall": recall,
            "pair_f1": f1,
            "missing": sorted(gold_set - pred_set),
            "extra": sorted(pred_set - gold_set),
        }

    exact_match = _normalize_scalar(prediction) == _normalize_scalar(gold_answer)
    return {
        "exact_match": exact_match,
        "pair_precision": None,
        "pair_recall": None,
        "pair_f1": None,
        "missing": [] if exact_match else [gold_answer],
        "extra": [] if exact_match else [prediction],
    }


def evaluate_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_sample: List[Dict[str, Any]] = []
    exact_match_flags: List[int] = []
    pair_f1_scores: List[float] = []
    for row in rows:
        prediction = row.get("generate_response")
        gold_answer = coerce_gold_answer(row.get("answer"))
        metrics = evaluate_prediction_against_gold(prediction, gold_answer)
        exact_match_flags.append(1 if metrics["exact_match"] else 0)
        if metrics["pair_f1"] is not None:
            pair_f1_scores.append(metrics["pair_f1"])
        per_sample.append(
            {
                "id": row.get("id"),
                "set": row.get("set"),
                "selected_index": row.get("selected_index"),
                "type": row.get("type"),
                "level": row.get("level"),
                "exact_match": metrics["exact_match"],
                "pair_precision": metrics["pair_precision"],
                "pair_recall": metrics["pair_recall"],
                "pair_f1": metrics["pair_f1"],
                "missing": metrics["missing"],
                "extra": metrics["extra"],
            }
        )
    summary = {
        "sample_count": len(rows),
        "exact_match_rate": mean(exact_match_flags) if exact_match_flags else 0.0,
        "avg_pair_f1": mean(pair_f1_scores) if pair_f1_scores else None,
        "per_sample": per_sample,
    }
    return summary
