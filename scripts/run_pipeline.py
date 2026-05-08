"""Run the DOSSIER pipeline on Loong samples.

Pipeline:
    TOCBuilder -> SearchAgent -> Composer -> Formatter

Evaluation: structured_eval (EM, F1) + LLM judge (1-100)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dossier.agents.composer import Composer
from dossier.agents.formatter import Formatter
from dossier.agents.search_agent import SearchAgent
from dossier.agents.toc_builder import TOCBuilder
from dossier.backend import get_default_client
from dossier.common import (
    current_query_from_record,
    ensure_dir,
    instruction_from_record,
    json_dumps_pretty,
    safe_filename,
    write_json,
    write_jsonl,
)
from dossier.evaluation.structured_eval import evaluate_predictions
from dossier.evaluation.llm_judge import run_llm_judge
from dossier.manifest import build_manifest, load_records, save_manifest


DEFAULT_INPUT_PATH = PROJECT_ROOT / "Loong" / "data" / "loong_process.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "runs" / "dossier_v2_v2_v1_v1"

MAX_OUTPUT_TOKENS = 50000
MAX_INPUT_TOKENS = 50000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DOSSIER pipeline.")
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Run only the first N records. Values <= 0 run the full input.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_refine_rounds", type=int, default=6)
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["local", "gemini", "openai"],
        help="LLM backend: 'openai' for vLLM server (default), 'gemini', 'local'",
    )
    return parser.parse_args()


def ensure_layout(output_dir: Path) -> Dict[str, Path]:
    layout = {
        "root": output_dir,
        "samples": output_dir / "samples",
        "reports": output_dir / "reports",
    }
    for p in layout.values():
        ensure_dir(p)
    return layout


def _elapsed(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 6)


def _sample_metadata(item: Any, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": item.sample_id,
        "selected_index": item.selected_index,
        "set": record.get("set", item.set_id),
        "type": record.get("type"),
        "level": record.get("level"),
        "record_id": record.get("id"),
    }


def _timing_row(
    *,
    item: Any,
    record: Dict[str, Any],
    module: str,
    scope: str,
    elapsed_seconds: float,
    status: str = "ok",
    doc_id: str = "",
    doc_title: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        **_sample_metadata(item, record),
        "module": module,
        "scope": scope,
        "doc_id": doc_id,
        "doc_title": doc_title,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "status": status,
    }
    if extra:
        row.update(extra)
    return row


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "total_seconds": 0.0,
            "avg_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
        }
    return {
        "count": len(values),
        "total_seconds": round(sum(values), 6),
        "avg_seconds": round(mean(values), 6),
        "min_seconds": round(min(values), 6),
        "max_seconds": round(max(values), 6),
    }


def summarize_timings(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = {}
    grouped_by_set: Dict[str, List[float]] = {}
    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        elapsed = float(row.get("elapsed_seconds", 0.0) or 0.0)
        module = str(row.get("module", "unknown"))
        scope = str(row.get("scope", "unknown"))
        key = f"{module}:{scope}"
        grouped.setdefault(key, []).append(elapsed)
        set_key = f"set={row.get('set')}:{module}:{scope}"
        grouped_by_set.setdefault(set_key, []).append(elapsed)
    return {
        "row_count": len(rows),
        "status_counts": status_counts,
        "by_module_scope": {key: _stats(values) for key, values in sorted(grouped.items())},
        "by_set_module_scope": {
            key: _stats(values) for key, values in sorted(grouped_by_set.items())
        },
    }


def write_sample_timings(sample_dir: Path, sample_timing: Dict[str, Any]) -> None:
    write_json(sample_dir / "timings.json", sample_timing)


def flush_timing_logs(layout: Dict[str, Path], rows: List[Dict[str, Any]]) -> None:
    write_jsonl(layout["root"] / "dossier_timings.partial.jsonl", rows)
    write_json(layout["reports"] / "timing_summary.partial.json", summarize_timings(rows))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    layout = ensure_layout(output_dir)

    records = load_records(input_path, limit=args.limit if args.limit > 0 else None)
    manifest = build_manifest(records)
    save_manifest(manifest, layout["root"] / "manifest.json")

    print(f"Backend: {args.backend}", flush=True)
    llm = get_default_client(backend=args.backend)

    # Patch token limits to 50k
    if hasattr(llm, '_default_max_output_tokens'):
        llm._default_max_output_tokens = MAX_OUTPUT_TOKENS

    toc_builder = TOCBuilder(llm=llm)
    search_agent = SearchAgent(
        llm=llm,
        max_rounds=args.max_refine_rounds,
    )
    composer = Composer(llm=llm)
    formatter = Formatter(llm=llm)

    prediction_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []

    for item in manifest:
        record = records[item.selected_index]
        sample_dir = layout["samples"] / safe_filename(item.sample_id)
        ensure_dir(sample_dir)
        print(
            f"\n==== sample {item.sample_id} "
            f"(idx={item.selected_index}, set={item.set_id}) ====",
            flush=True,
        )
        sample_start = time.perf_counter()
        sample_timing: Dict[str, Any] = {
            **_sample_metadata(item, record),
            "status": "running",
            "modules": {},
        }

        try:
            question = current_query_from_record(
                str(record.get("question", "")).strip(),
                str(record.get("instruction", "")).strip(),
            )
            instruction = instruction_from_record(
                str(record.get("instruction", "")).strip(),
            )

            # Stage 1: TOCBuilder — build hierarchical TOC per doc
            toc_start = time.perf_counter()
            try:
                toc_payload = toc_builder.run(
                    record=record, sample_dir=sample_dir, force=args.force,
                )
            except Exception:
                toc_elapsed = _elapsed(toc_start)
                sample_timing["modules"]["toc_builder"] = {
                    "total_seconds": toc_elapsed,
                    "status": "error",
                }
                timing_rows.append(
                    _timing_row(
                        item=item,
                        record=record,
                        module="toc_builder",
                        scope="module_total",
                        elapsed_seconds=toc_elapsed,
                        status="error",
                    )
                )
                sample_timing["total_seconds"] = _elapsed(sample_start)
                sample_timing["status"] = "error"
                write_sample_timings(sample_dir, sample_timing)
                raise

            toc_elapsed = _elapsed(toc_start)
            toc_timing = toc_payload.get("timing", {}) if isinstance(toc_payload, dict) else {}
            toc_total = float(toc_timing.get("total_seconds", toc_elapsed) or toc_elapsed)
            toc_doc_timings = toc_timing.get("docs", []) if isinstance(toc_timing, dict) else []
            sample_timing["modules"]["toc_builder"] = {
                "total_seconds": round(toc_total, 6),
                "wall_seconds": toc_elapsed,
                "doc_count": len(toc_payload.get("docs", [])),
                "docs": toc_doc_timings,
                "status": "ok",
            }
            timing_rows.append(
                _timing_row(
                    item=item,
                    record=record,
                    module="toc_builder",
                    scope="module_total",
                    elapsed_seconds=toc_total,
                    status="ok",
                    extra={"doc_count": len(toc_payload.get("docs", []))},
                )
            )
            for doc_timing in toc_doc_timings:
                timing_rows.append(
                    _timing_row(
                        item=item,
                        record=record,
                        module="toc_builder",
                        scope="doc",
                        elapsed_seconds=float(doc_timing.get("elapsed_seconds", 0.0) or 0.0),
                        status=str(doc_timing.get("status", "ok")),
                        doc_id=str(doc_timing.get("doc_id", "")),
                        doc_title=str(doc_timing.get("doc_title", "")),
                        extra={"toc_count": doc_timing.get("toc_count")},
                    )
                )
            write_sample_timings(sample_dir, sample_timing)
            print(
                f"  toc_builder -> docs={len(toc_payload.get('docs', []))} "
                f"| seconds={toc_total:.3f}",
                flush=True,
            )

            # Stage 2: SearchAgent per document
            doc_sheets: List[Dict[str, Any]] = []
            other_docs_list = [
                {"doc_id": d["doc_id"], "doc_title": d.get("doc_title", "")}
                for d in toc_payload["docs"]
            ]
            search_total_start = time.perf_counter()
            search_doc_timings: List[Dict[str, Any]] = []
            for doc_payload in toc_payload["docs"]:
                toc_count = doc_payload.get("toc_count", len(doc_payload.get("toc", [])))
                print(
                    f"  doc {doc_payload['doc_id']} '{doc_payload['doc_title'][:40]}' "
                    f"toc_sections={toc_count}",
                    flush=True,
                )
                search_agent.max_rounds = min(
                    args.max_refine_rounds,
                    max(1, toc_count + 1),
                )
                search_doc_start = time.perf_counter()
                try:
                    sheet = search_agent.run(
                        question=question,
                        instruction=instruction,
                        doc_payload=doc_payload,
                        sample_dir=sample_dir,
                        force=args.force,
                        other_docs=other_docs_list,
                    )
                except Exception:
                    search_elapsed = _elapsed(search_doc_start)
                    doc_timing = {
                        "doc_id": doc_payload["doc_id"],
                        "doc_title": doc_payload.get("doc_title", ""),
                        "elapsed_seconds": search_elapsed,
                        "toc_count": toc_count,
                        "status": "error",
                    }
                    search_doc_timings.append(doc_timing)
                    timing_rows.append(
                        _timing_row(
                            item=item,
                            record=record,
                            module="search_agent",
                            scope="doc",
                            elapsed_seconds=search_elapsed,
                            status="error",
                            doc_id=doc_payload["doc_id"],
                            doc_title=doc_payload.get("doc_title", ""),
                            extra={"toc_count": toc_count},
                        )
                    )
                    sample_timing["modules"]["search_agent"] = {
                        "total_seconds": _elapsed(search_total_start),
                        "doc_count": len(toc_payload.get("docs", [])),
                        "docs": search_doc_timings,
                        "status": "error",
                    }
                    sample_timing["total_seconds"] = _elapsed(sample_start)
                    sample_timing["status"] = "error"
                    write_sample_timings(sample_dir, sample_timing)
                    raise
                search_elapsed = _elapsed(search_doc_start)
                doc_timing = {
                    "doc_id": doc_payload["doc_id"],
                    "doc_title": doc_payload.get("doc_title", ""),
                    "elapsed_seconds": search_elapsed,
                    "toc_count": toc_count,
                    "rounds_used": sheet.get("rounds_used", 0),
                    "scan_result": sheet.get("scan_result"),
                    "status": "ok",
                }
                search_doc_timings.append(doc_timing)
                sheet["timing"] = {
                    "module": "search_agent",
                    "elapsed_seconds": search_elapsed,
                    "status": "ok",
                }
                write_json(sample_dir / f"{doc_payload['doc_id']}_search.json", sheet)
                timing_rows.append(
                    _timing_row(
                        item=item,
                        record=record,
                        module="search_agent",
                        scope="doc",
                        elapsed_seconds=search_elapsed,
                        status="ok",
                        doc_id=doc_payload["doc_id"],
                        doc_title=doc_payload.get("doc_title", ""),
                        extra={
                            "toc_count": toc_count,
                            "rounds_used": sheet.get("rounds_used", 0),
                            "scan_result": sheet.get("scan_result"),
                        },
                    )
                )
                doc_sheets.append(sheet)
                evidence_preview = (sheet.get("evidence", "") or "")[:80]
                print(
                    f"    -> {sheet['scan_result']} "
                    f"| rounds={sheet.get('rounds_used', 0)} "
                    f"| seconds={search_elapsed:.3f} "
                    f"| evidence={evidence_preview!r}",
                    flush=True,
                )
            search_total_elapsed = _elapsed(search_total_start)
            sample_timing["modules"]["search_agent"] = {
                "total_seconds": search_total_elapsed,
                "doc_count": len(toc_payload.get("docs", [])),
                "docs": search_doc_timings,
                "status": "ok",
            }
            timing_rows.append(
                _timing_row(
                    item=item,
                    record=record,
                    module="search_agent",
                    scope="module_total",
                    elapsed_seconds=search_total_elapsed,
                    status="ok",
                    extra={"doc_count": len(toc_payload.get("docs", []))},
                )
            )
            write_sample_timings(sample_dir, sample_timing)
            print(
                f"  search_agent -> docs={len(search_doc_timings)} "
                f"| seconds={search_total_elapsed:.3f}",
                flush=True,
            )

            # Stage 3: Composer
            composer_start = time.perf_counter()
            try:
                composed = composer.run(
                    question=question,
                    instruction=instruction,
                    doc_sheets=doc_sheets,
                    sample_dir=sample_dir,
                    force=args.force,
                )
            except Exception:
                composer_elapsed = _elapsed(composer_start)
                sample_timing["modules"]["composer"] = {
                    "total_seconds": composer_elapsed,
                    "status": "error",
                }
                timing_rows.append(
                    _timing_row(
                        item=item,
                        record=record,
                        module="composer",
                        scope="module_total",
                        elapsed_seconds=composer_elapsed,
                        status="error",
                    )
                )
                sample_timing["total_seconds"] = _elapsed(sample_start)
                sample_timing["status"] = "error"
                write_sample_timings(sample_dir, sample_timing)
                raise
            composer_elapsed = _elapsed(composer_start)
            composed["timing"] = {
                "module": "composer",
                "elapsed_seconds": composer_elapsed,
                "status": "ok",
            }
            write_json(sample_dir / "composed.json", composed)
            sample_timing["modules"]["composer"] = {
                "total_seconds": composer_elapsed,
                "status": "ok",
            }
            timing_rows.append(
                _timing_row(
                    item=item,
                    record=record,
                    module="composer",
                    scope="module_total",
                    elapsed_seconds=composer_elapsed,
                    status="ok",
                )
            )
            print(
                f"  composer -> projection_map keys={list(composed.get('projection_map', {}).keys())} "
                f"| records={len(composed.get('records', []))} "
                f"| seconds={composer_elapsed:.3f}",
                flush=True,
            )
            write_sample_timings(sample_dir, sample_timing)

            # Stage 4: Formatter
            doc_title_list = {
                d["doc_id"]: d.get("doc_title", d["doc_id"])
                for d in toc_payload["docs"]
            }
            formatter_start = time.perf_counter()
            try:
                gen_out = formatter.run(
                    question=question,
                    instruction=instruction,
                    composed=composed,
                    sample_dir=sample_dir,
                    force=args.force,
                    doc_title_list=doc_title_list,
                )
            except Exception:
                formatter_elapsed = _elapsed(formatter_start)
                sample_timing["modules"]["formatter"] = {
                    "total_seconds": formatter_elapsed,
                    "status": "error",
                }
                timing_rows.append(
                    _timing_row(
                        item=item,
                        record=record,
                        module="formatter",
                        scope="module_total",
                        elapsed_seconds=formatter_elapsed,
                        status="error",
                    )
                )
                sample_timing["total_seconds"] = _elapsed(sample_start)
                sample_timing["status"] = "error"
                write_sample_timings(sample_dir, sample_timing)
                raise
            formatter_elapsed = _elapsed(formatter_start)
            gen_out["timing"] = {
                "module": "formatter",
                "elapsed_seconds": formatter_elapsed,
                "status": "ok",
            }
            write_json(sample_dir / "formatter.json", gen_out)
            sample_timing["modules"]["formatter"] = {
                "total_seconds": formatter_elapsed,
                "status": "ok",
            }
            timing_rows.append(
                _timing_row(
                    item=item,
                    record=record,
                    module="formatter",
                    scope="module_total",
                    elapsed_seconds=formatter_elapsed,
                    status="ok",
                )
            )
            final_answer = gen_out["final_answer"]

            keep = ("id", "set", "type", "level", "question", "instruction", "answer")
            pred_row = {k: record.get(k) for k in keep if k in record}
            pred_row["set"] = record.get("set", item.set_id)
            pred_row["selected_index"] = item.selected_index
            pred_row["sample_id"] = item.sample_id
            pred_row["timing"] = sample_timing.get("modules", {})
            if isinstance(final_answer, (dict, list)):
                pred_row["generate_response"] = json.dumps(final_answer, ensure_ascii=False)
            else:
                pred_row["generate_response"] = str(final_answer)
            pred_row["dossier_trace_dir"] = str(sample_dir)
            prediction_rows.append(pred_row)
            sample_total_elapsed = _elapsed(sample_start)
            sample_timing["total_seconds"] = sample_total_elapsed
            sample_timing["status"] = "ok"
            timing_rows.append(
                _timing_row(
                    item=item,
                    record=record,
                    module="sample",
                    scope="sample_total",
                    elapsed_seconds=sample_total_elapsed,
                    status="ok",
                    extra={"doc_count": len(toc_payload.get("docs", []))},
                )
            )
            write_sample_timings(sample_dir, sample_timing)
            flush_timing_logs(layout, timing_rows)
            print(
                f"  -> answer: {str(final_answer)[:200]} "
                f"| formatter_seconds={formatter_elapsed:.3f} "
                f"| sample_seconds={sample_total_elapsed:.3f}",
                flush=True,
            )

        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"  !! error on {item.sample_id}: {exc}\n{tb}", flush=True)
            sample_total_elapsed = _elapsed(sample_start)
            sample_timing["total_seconds"] = sample_total_elapsed
            sample_timing["status"] = "error"
            timing_rows.append(
                _timing_row(
                    item=item,
                    record=record,
                    module="sample",
                    scope="sample_total",
                    elapsed_seconds=sample_total_elapsed,
                    status="error",
                )
            )
            write_sample_timings(sample_dir, sample_timing)
            flush_timing_logs(layout, timing_rows)
            errors.append(
                {
                    "sample_id": item.sample_id,
                    "selected_index": item.selected_index,
                    "set": record.get("set", item.set_id),
                    "type": record.get("type"),
                    "level": record.get("level"),
                    "error": str(exc),
                    "traceback": tb,
                    "timing": sample_timing,
                }
            )
            write_json(sample_dir / "error.json", errors[-1])

    prediction_path = write_jsonl(layout["root"] / "dossier_predictions.jsonl", prediction_rows)
    timing_path = write_jsonl(layout["root"] / "dossier_timings.jsonl", timing_rows)
    timing_summary = summarize_timings(timing_rows)
    write_json(layout["reports"] / "timing_summary.json", timing_summary)

    # Structured evaluation
    try:
        structured_summary = evaluate_predictions(
            [
                {
                    **row,
                    "generate_response": (
                        json.loads(row["generate_response"])
                        if isinstance(row["generate_response"], str)
                        and row["generate_response"].startswith(("{", "["))
                        else row["generate_response"]
                    ),
                }
                for row in prediction_rows
            ]
        )
    except Exception as exc:  # noqa: BLE001
        structured_summary = {"error": str(exc)}
    write_json(layout["reports"] / "structured_eval.json", structured_summary)
    write_json(layout["reports"] / "errors.json", {"count": len(errors), "errors": errors})

    # LLM judge
    try:
        judge_out = run_llm_judge(llm=llm, prediction_rows=prediction_rows)
    except Exception as exc:  # noqa: BLE001
        judge_out = {"summary": {"error": str(exc)}, "verdicts": []}
    write_json(layout["reports"] / "llm_judge.json", judge_out)

    print(
        json_dumps_pretty(
            {
                "prediction_path": str(prediction_path),
                "timing_path": str(timing_path),
                "timing_summary": timing_summary,
                "structured_summary": {
                    k: v
                    for k, v in (structured_summary or {}).items()
                    if k != "per_sample"
                },
                "llm_judge_summary": judge_out.get("summary", {}),
                "num_errors": len(errors),
            }
        )
    )


if __name__ == "__main__":
    main()
