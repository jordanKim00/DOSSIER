"""TOCBuilder — hierarchical TOC builder over raw documents.

Reads one document at a time, calls an LLM to produce a hierarchical
table of contents with LINE numbers, then converts line numbers to exact
character offsets by looking up a pre-computed line-start table.

This avoids asking the LLM to count characters (unreliable), replacing it
with line numbers (L0001, L0002, ...) which are visible in the prompt.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..backend import GeminiClient, OpenAIClient, QwenLocalClient
from ..common import (
    ensure_dir,
    extract_json_payload,
    normalize_ws,
    parse_docs_bundle,
    read_json,
    safe_filename,
    write_json,
)


DEFAULT_SYSTEM_PROMPT = """
You are the TOC (Table of Contents) Builder.

Your job is to read ONE document and produce a clean, logical, hierarchical
table of contents that captures the document's true structure. A downstream
agent will use this TOC to extract any single section verbatim from the
document, so each entry MUST point to exact line numbers.

The document is presented to you with line numbers prefixed as:
  L0001: <line content>
  L0002: <line content>
  ...

For each TOC entry output exactly these fields:
- `number`    : dotted section number, e.g. "1", "1.1", "10.2"
- `title`     : short faithful label (2-8 words)
- `level`     : integer 1, 2, or 3 — must equal the dot-depth of `number`
- `start_line`: L-number where this section begins (inclusive), e.g. "L0003"
- `end_line`  : L-number where this section ends (inclusive), e.g. "L0041"

Rules for a good TOC:

1. Hierarchy is real.
   Use dotted numbering "1, 1.1, 1.1.1". A parent's line range must exactly
   enclose all its children: parent.start_line == first_child.start_line,
   parent.end_line == last_child.end_line.

2. Titles are faithful.
   Copy the document's own heading when visible. Invent a short label only
   when no heading exists. Do not fabricate content.

3. Leaves tile the full document with no gaps and no overlaps.
   The first leaf must start at L0001. The last leaf must end at the final
   line. Consecutive leaves must meet exactly (prev.end_line + 1 == next.start_line).
   Front matter, back matter, references, footers — include everything.

4. Granularity follows the document's own structure.
   Short docs: 4-8 sections. Long structured docs: up to 60 entries, 2-3 levels.

After building the TOC, generate Wikipedia-style categories for the whole
document. Categories are flat bottom-of-page style labels, e.g. "1726 births",
"Royal Navy captains", "Members of Parliament for Portsmouth". They describe
what the document's main subject belongs to, not generic keywords. Prefer
specific categories supported by the document. Do not prefix labels with
"Category:" unless the document itself explicitly uses that prefix.

Output strict JSON only — a single object with exactly these top-level fields:
- `toc`       : list of TOC entries in document order
- `categories`: list of 5-15 source-language, Wikipedia-style category labels (Do NOT translate named entities into another language; preserve the language/script used in the document.)

No prose, no markdown fences, no extra fields.
""".strip()


DEFAULT_USER_PROMPT = """
Document title: {doc_title}
Total lines: {total_lines}

Document text (line-numbered):
{numbered_text}

Build the hierarchical TOC for this document.

Reminders:
- `start_line` and `end_line` must be valid L-numbers from L0001 to L{total_lines:04d}.
- The first leaf's start_line must be L0001.
- The last leaf's end_line must be L{total_lines:04d}.
- Consecutive leaves must meet: prev end_line + 1 == next start_line.
- Parent ranges must exactly enclose their children.
- Output a JSON object only with `toc` and `categories`. No other text.
""".strip()


class TOCBuilder:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parents[3] / "prompts" / "toc_builder"
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = DEFAULT_USER_PROMPT
        system_path = self.prompt_dir / "system.txt"
        user_path = self.prompt_dir / "user.txt"
        if system_path.exists():
            self.system_prompt = system_path.read_text(encoding="utf-8").strip()
        if user_path.exists():
            self.user_prompt_template = user_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _number_lines(doc_text: str) -> Tuple[List[str], List[int], str]:
        """Split text into lines, compute char offsets, return numbered text.

        Returns:
            lines        : raw line strings (without newline)
            line_starts  : char offset of the start of each line in doc_text
            numbered_text: the prompt-ready string with L0001: ... prefixes
        """
        lines = doc_text.split("\n")
        line_starts: List[int] = []
        cursor = 0
        for line in lines:
            line_starts.append(cursor)
            cursor += len(line) + 1  # +1 for the '\n'

        numbered_lines = [
            f"L{i + 1:04d}: {line}" for i, line in enumerate(lines)
        ]
        numbered_text = "\n".join(numbered_lines)
        return lines, line_starts, numbered_text

    @staticmethod
    def _parse_lnum(lnum_str: str) -> int:
        """Convert 'L0003' → 2 (0-based line index). Returns -1 on failure."""
        s = str(lnum_str).strip().lstrip("Ll")
        try:
            return int(s) - 1  # 1-based → 0-based
        except ValueError:
            return -1

    def _lines_to_char_span(
        self,
        start_line_str: str,
        end_line_str: str,
        line_starts: List[int],
        lines: List[str],
        doc_text: str,
    ) -> Tuple[int, int]:
        """Convert L-number strings to (char_start, char_end) in doc_text."""
        n = len(lines)
        si = self._parse_lnum(start_line_str)
        ei = self._parse_lnum(end_line_str)

        si = max(0, min(si, n - 1))
        ei = max(si, min(ei, n - 1))

        char_start = line_starts[si]
        # char_end = start of the line AFTER end_line (exclusive)
        if ei + 1 < n:
            char_end = line_starts[ei + 1]
        else:
            char_end = len(doc_text)

        return char_start, char_end

    def _build_toc_entry(
        self,
        *,
        doc_id: str,
        doc_title: str,
        doc_text: str,
        lines: List[str],
        line_starts: List[int],
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        char_start, char_end = self._lines_to_char_span(
            item["start_line"],
            item["end_line"],
            line_starts,
            lines,
            doc_text,
        )
        return {
            "number": str(item.get("number", "")),
            "title": normalize_ws(str(item.get("title", ""))),
            "level": int(item.get("level", 1)),
            "start_line": str(item["start_line"]),
            "end_line": str(item["end_line"]),
            "char_start": char_start,
            "char_end": char_end,
            "text": doc_text[char_start:char_end],
            "doc_id": doc_id,
            "doc_title": doc_title,
        }

    def _fallback_toc(self, *, doc_id: str, doc_title: str, doc_text: str, total_lines: int) -> List[Dict[str, Any]]:
        return [
            {
                "number": "1",
                "title": doc_title or "Document",
                "level": 1,
                "start_line": "L0001",
                "end_line": f"L{total_lines:04d}",
                "char_start": 0,
                "char_end": len(doc_text),
                "text": doc_text,
                "doc_id": doc_id,
                "doc_title": doc_title,
            }
        ]

    @staticmethod
    def _normalize_categories(categories: Any) -> List[str]:
        if not isinstance(categories, list):
            return []
        normalized: List[str] = []
        for category in categories:
            label = normalize_ws(str(category))
            if label.startswith("Category:"):
                label = normalize_ws(label[len("Category:"):])
            if label:
                normalized.append(label)
        return normalized[:15]

    @classmethod
    def _extract_toc_items_and_categories(
        cls,
        payload: Any,
    ) -> Tuple[Optional[List[Any]], List[str]]:
        if isinstance(payload, list):
            return payload, []
        if not isinstance(payload, dict) or not isinstance(payload.get("toc"), list):
            return None, []

        categories = cls._normalize_categories(payload.get("categories", []))
        toc_items: List[Any] = []
        for item in payload["toc"]:
            if (
                isinstance(item, dict)
                and "categories" in item
                and (
                    "number" not in item
                    or "start_line" not in item
                    or "end_line" not in item
                )
            ):
                if not categories:
                    categories = cls._normalize_categories(item.get("categories", []))
                continue
            toc_items.append(item)
        return toc_items, categories

    def _build_toc_with_llm(
        self,
        *,
        doc: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], str, List[str]]:
        doc_text: str = doc.get("content", "")
        lines, line_starts, numbered_text = self._number_lines(doc_text)
        total_lines = len(lines)

        user_prompt = self.user_prompt_template.format(
            doc_title=doc["doc_title"],
            total_lines=total_lines,
            numbered_text=numbered_text,
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=16384,
            metadata={"module": "toc_builder", "doc_title": doc["doc_title"]},
        )

        payload = extract_json_payload(raw_text)
        toc_items, categories = self._extract_toc_items_and_categories(payload)
        if toc_items is None:
            return self._fallback_toc(
                doc_id=doc["doc_id"], doc_title=doc["doc_title"],
                doc_text=doc_text, total_lines=total_lines,
            ), raw_text, []

        toc: List[Dict[str, Any]] = []
        for item in toc_items:
            if not isinstance(item, dict):
                continue
            if "start_line" not in item or "end_line" not in item:
                continue
            try:
                entry = self._build_toc_entry(
                    doc_id=doc["doc_id"],
                    doc_title=doc["doc_title"],
                    doc_text=doc_text,
                    lines=lines,
                    line_starts=line_starts,
                    item=item,
                )
            except (KeyError, TypeError, ValueError):
                continue
            toc.append(entry)

        if not toc:
            return self._fallback_toc(
                doc_id=doc["doc_id"], doc_title=doc["doc_title"],
                doc_text=doc_text, total_lines=total_lines,
            ), raw_text, []

        return toc, raw_text, categories

    def run(
        self,
        *,
        record: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        ensure_dir(sample_dir)
        cache_path = sample_dir / "toc.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        docs = parse_docs_bundle(record.get("docs", ""))
        doc_payloads: List[Dict[str, Any]] = []
        doc_timings: List[Dict[str, Any]] = []
        total_start = time.perf_counter()
        for doc in docs:
            doc_start = time.perf_counter()
            try:
                toc, raw_response, categories = self._build_toc_with_llm(doc=doc)
            except Exception:
                elapsed = time.perf_counter() - doc_start
                doc_timings.append(
                    {
                        "doc_id": doc.get("doc_id"),
                        "doc_title": doc.get("doc_title", ""),
                        "elapsed_seconds": round(elapsed, 6),
                        "status": "error",
                    }
                )
                write_json(
                    sample_dir / "toc.partial.json",
                    {
                        "record_id": record.get("id"),
                        "progress": {
                            "completed_docs": len(doc_payloads),
                            "total_docs": len(docs),
                        },
                        "timing": {
                            "module": "toc_builder",
                            "partial_total_seconds": round(time.perf_counter() - total_start, 6),
                            "docs": doc_timings,
                        },
                        "docs": doc_payloads,
                    },
                )
                raise
            elapsed = time.perf_counter() - doc_start
            doc_timing = {
                "doc_id": doc["doc_id"],
                "doc_title": doc["doc_title"],
                "elapsed_seconds": round(elapsed, 6),
                "toc_count": len(toc),
                "status": "ok",
            }
            doc_timings.append(doc_timing)
            doc_payloads.append(
                {
                    "doc_id": doc["doc_id"],
                    "doc_title": doc["doc_title"],
                    "toc_count": len(toc),
                    "toc": toc,
                    "categories": categories,
                    "toc_generation_raw": raw_response,
                    "timing": {
                        "module": "toc_builder",
                        "elapsed_seconds": doc_timing["elapsed_seconds"],
                    },
                }
            )
            write_json(
                sample_dir / "toc.partial.json",
                {
                    "record_id": record.get("id"),
                    "progress": {"completed_docs": len(doc_payloads), "total_docs": len(docs)},
                    "timing": {
                        "module": "toc_builder",
                        "partial_total_seconds": round(time.perf_counter() - total_start, 6),
                        "docs": doc_timings,
                    },
                    "docs": doc_payloads,
                },
            )

        total_elapsed = time.perf_counter() - total_start
        payload = {
            "record_id": record.get("id"),
            "sample_id": safe_filename(
                f"{record.get('type', '?')}_level{record.get('level', '?')}_{record.get('id', '?')}"
            ),
            "segmentation_mode": "hierarchical_toc_line_anchored",
            "timing": {
                "module": "toc_builder",
                "total_seconds": round(total_elapsed, 6),
                "doc_count": len(docs),
                "docs": doc_timings,
            },
            "docs": doc_payloads,
        }
        write_json(cache_path, payload)
        return payload
