"""Formatter — final answer serialization.

Consumes the composed cross-document analysis `{projection_map, records,
structure_description}` and emits the user-facing answer in the exact format
requested by the question/instruction. All internal DOC ids are resolved to real
entity names through the projection_map before the answer is returned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..backend import QwenLocalClient, GeminiClient, OpenAIClient
from ..common import extract_json_payload, read_json, write_json


class Formatter:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        pdir = prompt_dir or Path(__file__).resolve().parents[3] / "prompts" / "formatter"
        self.system_prompt = (pdir / "system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (pdir / "user.txt").read_text(encoding="utf-8").strip()

    def run(
        self,
        *,
        question: str,
        instruction: str,
        composed: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
        doc_title_list: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / "formatter.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        projection_map = composed.get("projection_map", {})
        records = composed.get("records", [])
        structure_description = composed.get("structure_description", "")

        composed_compact = {
            "structure_description": structure_description,
            "records": records,
        }

        title_list = doc_title_list or {}
        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            doc_title_list=json.dumps(title_list, ensure_ascii=False, indent=2),
            projection_map=json.dumps(projection_map, ensure_ascii=False, indent=2),
            composed_json=json.dumps(composed_compact, ensure_ascii=False, indent=2),
        )

        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=8192,
            metadata={"module": "formatter"},
        )

        # Try to parse as JSON first; fall back to the raw string for unstructured answers
        final_answer: Any = raw_text.strip()
        parsed = extract_json_payload(raw_text)
        if parsed is not None:
            final_answer = parsed

        result = {
            "final_answer": final_answer,
            "raw_text": raw_text,
            "projection_map": projection_map,
        }
        write_json(cache_path, result)
        return result
