"""SearchAgent — per-document evidence extraction compatible with TOCBuilder.

Identical loop to DocRefineAgent (plan→search→info) but consumes the TOC-based
doc_payload produced by TOCBuilder instead of the block-based anchor payload.

Inputs per doc: doc_title + toc (list of {number, title, level, char_start,
char_end, text}). The LLM references sections by their dotted number (e.g.
"3.2"). The agent now expects the LLM's <answer> to be a single JSON object
with `scan_result` and `records[]`, where each record carries 5W1H slots, a
verbatim source span, and an optional `found_relation` triple. The raw verbatim
text is preserved alongside as `evidence` for backward compatibility.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ..backend import GeminiClient, OpenAIClient, QwenLocalClient
from ..common import (
    extract_json_payload,
    extract_tag_content,
    normalize_ws,
    read_json,
    write_json,
)


VALID_SCAN_RESULTS = {
    "evidence_found",
    "negative_evidence_found",
    "irrelevant_document",
    "insufficient_evidence",
}

VALID_HOW_TAGS = {
    "lookup_value",
    "rank_input",
    "classification_cue",
    "constraint_check",
    "relation_evidence",
}


class SearchAgent:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
        max_rounds: int = 6,
    ) -> None:
        self.llm = llm
        self.max_rounds = max_rounds
        pdir = prompt_dir or Path(__file__).resolve().parents[3] / "prompts" / "search_agent"
        self.system_prompt = (pdir / "system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (pdir / "user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _format_doc_map(
        toc: List[Dict[str, Any]],
        categories: Optional[List[Any]] = None,
    ) -> str:
        """Render TOC as an indented section list for the LLM."""
        lines: List[str] = []
        for entry in toc:
            number = str(entry.get("number", ""))
            title = str(entry.get("title", ""))
            level = int(entry.get("level", 1))
            indent = "  " * (level - 1)
            lines.append(f"{indent}[{number}] {title}")
        doc_map_text = "\n".join(lines) if lines else "(no sections available)"

        clean_categories: List[str] = []
        if isinstance(categories, list):
            for category in categories:
                label = normalize_ws(str(category))
                if label:
                    clean_categories.append(label)
        if clean_categories:
            doc_map_text += "\nCategories: " + " | ".join(clean_categories)
        return doc_map_text

    def _call_llm(
        self,
        *,
        question: str,
        instruction: str,
        doc_id: str,
        doc_title: str,
        doc_map_text: str,
        all_docs_list: str,
        accumulated_trace: str,
    ) -> str:
        trace_section = ""
        if accumulated_trace.strip():
            trace_section = f"Previous rounds:\n{accumulated_trace}"
        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            doc_title=doc_title,
            doc_id=doc_id,
            doc_map=doc_map_text,
            all_docs_list=all_docs_list,
            accumulated_trace=trace_section,
        )
        return self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=12288,
            metadata={"module": "search_agent", "phase": "plan_search", "doc_id": doc_id},
        )

    @staticmethod
    def _normalise_record(
        rec: Any,
        section_by_id: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Return a cleaned record dict, or None if it lacks a verbatim
        source span. Computes char_start/char_end relative to the opened
        section text whenever the verbatim string is found there."""
        if not isinstance(rec, dict):
            return None
        src = rec.get("source")
        if not isinstance(src, dict):
            return None
        verbatim = str(src.get("verbatim", "")).strip()
        if not verbatim:
            return None
        section_id = str(src.get("section", "")).strip()
        section_title = str(src.get("section_title", "")).strip()
        if not section_title and section_id in section_by_id:
            section_title = str(section_by_id[section_id].get("title", ""))

        # Post-hoc char-span lookup (best effort)
        char_start, char_end = -1, -1
        section_text = ""
        if section_id in section_by_id:
            section_text = section_by_id[section_id].get("text", "") or ""
            idx = section_text.find(verbatim)
            if idx >= 0:
                char_start, char_end = idx, idx + len(verbatim)

        cleaned_source: Dict[str, Any] = {
            "section": section_id,
            "section_title": section_title,
            "verbatim": verbatim,
            "char_start": char_start,
            "char_end": char_end,
        }

        # Optional generic relation triple (subject, predicate, object).
        # Polarity is intentionally NOT separately tracked — the
        # predicate carries any negation, and document-level absence is
        # already represented by scan_result="negative_evidence_found".
        rel_raw = rec.get("found_relation")
        cleaned_rel: Optional[Dict[str, Any]] = None
        if isinstance(rel_raw, dict):
            subj = str(rel_raw.get("subject", "")).strip()
            pred = str(rel_raw.get("predicate", "")).strip().lower()
            obj = str(rel_raw.get("object", "")).strip()
            if subj or pred or obj:
                cleaned_rel = {
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                }

        how = str(rec.get("how", "")).strip().lower()
        if how not in VALID_HOW_TAGS:
            # Default to lookup_value if missing/unknown
            how = "lookup_value"

        return {
            "who":   str(rec.get("who", "")).strip(),
            "what":  str(rec.get("what", "")).strip(),
            "when":  str(rec.get("when", "—")).strip() or "—",
            "where": str(rec.get("where", "")).strip(),
            "why":   str(rec.get("why", "")).strip(),
            "how":   how,
            "source": cleaned_source,
            "found_relation": cleaned_rel,
        }

    @staticmethod
    def _iter_json_payloads(text: str) -> Iterator[Tuple[Any, str]]:
        """Yield parseable top-level JSON snippets from text in order.

        This is used for answer extraction instead of taking the first JSON
        object, because model thoughts may contain a <search> JSON before the
        real <answer> JSON.
        """
        stack: List[str] = []
        start_idx: Optional[int] = None
        in_string = False
        escape = False

        for idx, char in enumerate(str(text or "")):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue
            if char in "{[":
                if start_idx is None:
                    start_idx = idx
                stack.append("}" if char == "{" else "]")
            elif char in "}]" and stack and char == stack[-1]:
                stack.pop()
                if start_idx is not None and not stack:
                    raw = str(text or "")[start_idx: idx + 1]
                    try:
                        yield json.loads(raw), raw
                    except Exception:
                        pass
                    start_idx = None

    @staticmethod
    def _is_answer_payload(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        scan = str(payload.get("scan_result", "")).strip()
        if scan not in VALID_SCAN_RESULTS:
            return False
        return isinstance(payload.get("records", []), list)

    @classmethod
    def _extract_answer_text(cls, raw: str) -> str:
        """Return the last schema-valid answer JSON, tolerating tag leakage.

        Some models write literal strings like "<answer> block" inside their
        thought text. A regex span can then start at that leaked tag and include
        unrelated <search> JSON. We therefore select by SearchAgent answer schema
        instead of trusting the first tagged span.
        """
        raw_text = str(raw or "")
        answer_starts = [
            match.start()
            for match in re.finditer(r"<answer\b[^>]*>", raw_text, flags=re.IGNORECASE)
        ]
        regions = [raw_text[pos:] for pos in reversed(answer_starts)]
        regions.append(raw_text)

        for region in regions:
            candidates = [
                raw_json
                for payload, raw_json in cls._iter_json_payloads(region)
                if cls._is_answer_payload(payload)
            ]
            if candidates:
                return candidates[-1].strip()

        return extract_tag_content(raw_text, "answer") or ""

    @staticmethod
    def _parse_answer_block(
        answer_text: str,
        section_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Parse the LLM's <answer> JSON; tolerate missing fields and
        legacy verbatim-only output. Always returns a dict with
        scan_result, records, and a flat verbatim-string fallback."""
        if not answer_text or not answer_text.strip():
            return {
                "scan_result": "insufficient_evidence",
                "records": [],
                "verbatim_blob": "",
            }

        payload = extract_json_payload(answer_text)
        if isinstance(payload, dict):
            scan = str(payload.get("scan_result", "")).strip()
            if scan not in VALID_SCAN_RESULTS:
                scan = "evidence_found" if payload.get("records") else "insufficient_evidence"
            raw_records = payload.get("records", [])
            records: List[Dict[str, Any]] = []
            if isinstance(raw_records, list):
                for rec in raw_records:
                    cleaned = SearchAgent._normalise_record(rec, section_by_id)
                    if cleaned is not None:
                        records.append(cleaned)
            verbatim_blob = "\n".join(
                r["source"]["verbatim"] for r in records if r.get("source", {}).get("verbatim")
            )
            if scan == "evidence_found" and not records:
                # Promised evidence but no usable record — downgrade
                scan = "insufficient_evidence"
            return {
                "scan_result": scan,
                "records": records,
                "verbatim_blob": verbatim_blob,
            }

        # Legacy fallback: payload was a list of strings or a single string.
        if isinstance(payload, list):
            blob = "\n".join(str(x) for x in payload if x)
        else:
            blob = answer_text.strip()
        return {
            "scan_result": "evidence_found" if blob else "insufficient_evidence",
            "records": [],
            "verbatim_blob": blob,
        }

    def run(
        self,
        *,
        question: str,
        instruction: str,
        doc_payload: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
        other_docs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        doc_id = doc_payload["doc_id"]
        doc_title = doc_payload["doc_title"]
        cache_path = sample_dir / f"{doc_id}_search.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        toc = doc_payload.get("toc", [])
        doc_map_text = self._format_doc_map(toc, doc_payload.get("categories", []))

        # Build lookup: section number → toc entry (for text injection)
        section_by_id: Dict[str, Dict[str, Any]] = {
            str(entry.get("number", "")): entry for entry in toc
        }

        docs_for_list = other_docs or [{"doc_id": doc_id, "doc_title": doc_title}]
        all_docs_list = "\n".join(
            f"- {d['doc_id']}: {d.get('doc_title', '')}" for d in docs_for_list
        )

        opened_sections: List[str] = []
        accumulated_trace = ""

        scan_result = "insufficient_evidence"
        evidence_records: List[Dict[str, Any]] = []
        evidence_text = ""

        all_numbers = [str(e.get("number", "")) for e in toc]

        for round_idx in range(1, self.max_rounds + 1):
            raw = self._call_llm(
                question=question,
                instruction=instruction,
                doc_id=doc_id,
                doc_title=doc_title,
                doc_map_text=doc_map_text,
                all_docs_list=all_docs_list,
                accumulated_trace=accumulated_trace,
            )

            plan_text = extract_tag_content(raw, "plan") or ""
            search_text = extract_tag_content(raw, "search") or ""
            answer_text = self._extract_answer_text(raw)
            search_payload = extract_json_payload(search_text)

            accumulated_trace += f"\n--- Round {round_idx} ---\n"
            accumulated_trace += f"<plan>{plan_text}</plan>\n"
            accumulated_trace += f"<search>{search_text}</search>\n"

            action = "stop"
            anchor_id = ""
            if isinstance(search_payload, dict):
                action = str(search_payload.get("action", "stop")).lower()
                anchor_id = normalize_ws(str(search_payload.get("anchor_id", "")))

            if action == "stop":
                if not opened_sections and toc:
                    anchor_id = all_numbers[0]
                    accumulated_trace += (
                        "<note>Stop overridden: must open at least one section before "
                        "judging irrelevance. Forcing open of first section.</note>\n"
                    )
                    action = "open"
                else:
                    if answer_text:
                        accumulated_trace += f"<answer>{answer_text}</answer>\n"
                        parsed = self._parse_answer_block(answer_text, section_by_id)
                        scan_result = parsed["scan_result"]
                        evidence_records = parsed["records"]
                        evidence_text = parsed["verbatim_blob"]
                    else:
                        scan_result = "insufficient_evidence"
                    break

            if not anchor_id:
                if not opened_sections and toc:
                    anchor_id = all_numbers[0]
                else:
                    break

            if anchor_id not in section_by_id or anchor_id in opened_sections:
                fallback_id = ""
                for num in all_numbers:
                    if num not in opened_sections:
                        fallback_id = num
                        break
                if not fallback_id:
                    break
                anchor_id = fallback_id

            opened_sections.append(anchor_id)
            section_text = section_by_id[anchor_id].get("text", "")
            accumulated_trace += f"<info anchor_id=\"{anchor_id}\">\n{section_text}\n</info>\n"

        evidence_sheet = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "scan_result": scan_result,
            "evidence_records": evidence_records,
            "evidence": evidence_text,           # backward-compat: verbatim blob
            "opened_anchors": opened_sections,
            "rounds_used": len(opened_sections),
            "trace": accumulated_trace,
        }
        write_json(cache_path, evidence_sheet)
        return evidence_sheet
