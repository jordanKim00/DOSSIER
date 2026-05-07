# Pipeline

DOSSIER runs four stages:

1. `TOCBuilder`: reads each raw document, builds a line-numbered hierarchical
   table of contents, and attaches document-level Wikipedia-style categories.
2. `SearchAgent`: navigates TOC sections and emits per-document 5W1H evidence
   records with verbatim source spans.
3. `Composer`: resolves document IDs to entities and turns all evidence sheets
   into a task-specific structured relation view.
4. `Formatter`: renders the final answer from the composed structure.

Timing artifacts are written for timing-sensitive analysis:

- `samples/<sample_id>/timings.json`: per-sample timing tree with module totals
  and per-document TOC/Search timings.
- `dossier_timings.jsonl`: row-level timing log for every sample/module/doc.
- `reports/timing_summary.json`: aggregate timing statistics by module/scope and
  by Loong `set`.
- `*.partial` timing logs are flushed after each sample so interrupted runs still
  preserve timing information.

The prompt text used by `SearchAgent`, `Composer`, and `Formatter` is copied
byte-for-byte from the current experiment prompts. `TOCBuilder` uses the same
TOC prompt that was embedded in the original v2 code, now externalized under
`prompts/toc_builder/`.
