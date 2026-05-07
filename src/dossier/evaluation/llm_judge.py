"""Loong-style LLM judge reusing our local Qwen client."""

from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any, Dict, List, Optional, Union

from ..backend import QwenLocalClient, GeminiClient, OpenAIClient


JUDGE_PROMPT = """[Question]
{question}

[Gold Answer]
{gold}

[The Start of Assistant's Predicted Answer]
{prediction}
[The End of Assistant's Predicted Answer]

[System]
We would like to request your feedback on the performance of the AI assistant in response to the user question displayed above according to the gold answer. Please use the following listed aspects and their descriptions as evaluation criteria:
    - Accuracy and Hallucinations: The assistant's answer is semantically consistent with the gold answer; The numerical value and order need to be accurate, and there should be no hallucinations.
    - Completeness: Referring to the reference answers, the assistant's answer should contain all the key points needed to answer the user's question; further elaboration on these key points can be omitted.
Please rate whether this answer is suitable for the question. Please note that the gold answer can be considered as a correct answer to the question.

The assistant receives an overall score on a scale of 1 to 100, where a higher score indicates better overall performance.
Please note that if the assistant's answer and the gold answer fully meet the above criteria, its overall rating should be the full marks (100).
Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias.
Then, output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS A SCALE OF 1 TO 100 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[100]]":
<start output>
Evaluation evidence: your evluation explanation here, no more than 100 words
Rating: [[score]]
<end output>

Now, start your evaluation:"""


def _extract_score(text: str) -> Optional[float]:
    m = re.search(r"\[\[([0-9]*\.?[0-9]+)\]\]", text)
    if m:
        return float(m.group(1))
    m = re.search(r"\[([0-9]*\.?[0-9]+)\]", text)
    if m:
        return float(m.group(1))
    return None


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def run_llm_judge(
    *,
    llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
    prediction_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    verdicts: List[Dict[str, Any]] = []
    scores: List[float] = []
    for row in prediction_rows:
        question = str(row.get("question", ""))
        instruction = str(row.get("instruction", ""))
        gold = _stringify(row.get("answer", ""))
        prediction = _stringify(row.get("generate_response", ""))
        full_question = question if not instruction else f"{question}\n\n[Instruction]\n{instruction}"
        prompt = JUDGE_PROMPT.format(
            question=full_question,
            gold=gold,
            prediction=prediction,
        )
        raw = llm.generate_text(
            system_prompt="You are a strict, impartial evaluator.",
            user_prompt=prompt,
            max_output_tokens=400,
            metadata={"module": "llm_judge", "sample_id": row.get("sample_id", "")},
        )
        score = _extract_score(raw)
        if score is not None:
            scores.append(score)
        verdicts.append(
            {
                "sample_id": row.get("sample_id"),
                "id": row.get("id"),
                "set": row.get("set"),
                "type": row.get("type"),
                "level": row.get("level"),
                "score": score,
                "gold": gold,
                "prediction": prediction,
                "raw": raw,
            }
        )

    total = len(prediction_rows)
    effective = len(scores)
    perfect = sum(1 for s in scores if s == 100)
    summary = {
        "total": total,
        "scoring_success_rate": (effective / total) if total else 0.0,
        "avg_score": mean(scores) if scores else 0.0,
        "perfect_count": perfect,
        "perfect_rate": (perfect / effective) if effective else 0.0,
    }
    return {"summary": summary, "verdicts": verdicts}
