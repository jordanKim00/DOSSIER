"""Composer — cross-document evidence synthesis and entity resolution.

Takes per-document evidence sheets from DocRefineAgent and:
1. Builds `projection_map` (DOC id → real entity name)
2. Cross-document comparison / category assignment / relation extraction
3. Outputs a structured `records` list plus a one-line `structure_description` for
   the Generator.

No two-layer envelope, no `analysis_schema`/`artifacts`. The LLM designs the inner
shape of each record based on what the question/instruction asks for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..backend import QwenLocalClient, GeminiClient, OpenAIClient
from ..common import read_json, write_json


class Composer:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        pdir = prompt_dir or Path(__file__).resolve().parents[3] / "prompts" / "composer"
        self.system_prompt = (pdir / "system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (pdir / "user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _build_doc_title_map(doc_sheets: List[Dict[str, Any]]) -> Dict[str, str]:
        return {
            sheet["doc_id"]: sheet.get("doc_title", sheet["doc_id"])
            for sheet in doc_sheets
        }

    @staticmethod
    def _build_prompt_sheet(sheet: Dict[str, Any]) -> Dict[str, Any]:
        records = sheet.get("evidence_records") or []
        evidence = (sheet.get("evidence", "") or "").strip()
        prompt_sheet: Dict[str, Any] = {
            "doc_id": sheet["doc_id"],
            "doc_title": sheet.get("doc_title", sheet["doc_id"]),
            "scan_result": sheet.get("scan_result", "no_evidence"),
            "evidence_records": records if isinstance(records, list) else [],
        }
        if evidence and not prompt_sheet["evidence_records"]:
            prompt_sheet["evidence"] = evidence
        return prompt_sheet

    @staticmethod
    def _prepare_sheets_for_prompt(
        doc_sheets: List[Dict[str, Any]],
        anchor_docs_by_id: Dict[str, Dict[str, Any]],
    ) -> str:
        """Format per-document evidence for the LLM.

        Each block contains:
          - a compact JSON sheet with doc_id, doc_title, scan_result, and
            structured 5W1H evidence_records from DocRefineAgent
          - legacy verbatim evidence as a fallback when no records exist
          - top entity candidates aggregated from the anchor agent's `key_entities`
            (this helps the Composer build a faithful projection_map even when the
            refined evidence doesn't explicitly name the issuing company/author/party)
        """
        parts: List[str] = []
        for sheet in doc_sheets:
            doc_id = sheet["doc_id"]
            title = sheet.get("doc_title", "")
            status = sheet.get("scan_result", "no_evidence")

            block = f"### {doc_id}: {title}\n"
            block += f"Status: {status}\n"

            anchor_doc = anchor_docs_by_id.get(doc_id, {})
            candidate_entities: List[str] = []
            seen = set()
            for anc in anchor_doc.get("anchors", [])[:4]:
                for ent in anc.get("key_entities", []) or []:
                    ent = str(ent).strip()
                    if ent and ent not in seen:
                        seen.add(ent)
                        candidate_entities.append(ent)
                    if len(candidate_entities) >= 10:
                        break
                if len(candidate_entities) >= 10:
                    break
            if candidate_entities:
                block += f"Anchor entity candidates: {', '.join(candidate_entities)}\n"

            prompt_sheet = Composer._build_prompt_sheet(sheet)
            block += "Evidence sheet JSON:\n"
            block += json.dumps(prompt_sheet, ensure_ascii=False, indent=2) + "\n"
            parts.append(block)
        return "\n".join(parts)

    def run(
        self,
        *,
        question: str,
        instruction: str,
        doc_sheets: List[Dict[str, Any]],
        anchor_docs: Optional[List[Dict[str, Any]]] = None,
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / "composed.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        anchor_docs_by_id: Dict[str, Dict[str, Any]] = {}
        if anchor_docs:
            for d in anchor_docs:
                if isinstance(d, dict) and "doc_id" in d:
                    anchor_docs_by_id[d["doc_id"]] = d

        doc_title_map = self._build_doc_title_map(doc_sheets)
        sheets_text = self._prepare_sheets_for_prompt(doc_sheets, anchor_docs_by_id)

        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            doc_title_map=json.dumps(doc_title_map, ensure_ascii=False, indent=2),
            doc_sheets_text=sheets_text,
        )

        payload, raw_text = self.llm.generate_json(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=16384,
            metadata={"module": "composer"},
        )

        projection_map: Dict[str, str] = {}
        records: List[Any] = []
        structure_description = ""

        if isinstance(payload, dict):
            raw_projection = payload.get("projection_map", {})
            if isinstance(raw_projection, dict):
                projection_map = {str(k): str(v) for k, v in raw_projection.items()}
            raw_records = payload.get("records", [])
            if isinstance(raw_records, list):
                records = raw_records
            elif isinstance(raw_records, dict):
                records = [
                    {"category": str(k), "items": v if isinstance(v, list) else [v]}
                    for k, v in raw_records.items()
                ]
            structure_description = str(payload.get("structure_description", ""))

        # Fallback: build projection_map from doc_title if the LLM missed it
        if not projection_map:
            projection_map = {
                sheet["doc_id"]: sheet.get("doc_title", sheet["doc_id"])
                for sheet in doc_sheets
            }

        result = {
            "projection_map": projection_map,
            "structure_description": structure_description,
            "records": records,
            "raw_text": raw_text,
            "doc_sheet_count": len(doc_sheets),
            "evidence_found_count": sum(
                1 for s in doc_sheets if s.get("scan_result") == "evidence_found"
            ),
        }
        write_json(cache_path, result)
        return result
