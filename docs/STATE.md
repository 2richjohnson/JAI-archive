# JAI Archive — Project State

## Last Session Summary (2026-05-21)

### What We Accomplished
- **Diagnosed VM crash loop**: `ollama-gpu1.service` + `ollama.service` both auto-started on boot, both loaded llama3.1:8b (4.58 GiB) onto GPU1 simultaneously, exceeding 8 GiB VRAM → hard VM reset with no OOM warning. Confirmed via `journalctl -b -1`.
- **Disabled gpu1 service**: `systemctl disable --now ollama-gpu1.service`. Primary Ollama with `OLLAMA_NUM_PARALLEL=2` handles both GPUs internally.
- **Completed EAV extraction**: 177 markdown files → ~549 parquet files → 6,396 facts rows in DuckDB. Extraction ran ~4.5 hours in tmux, survived two crash/resume cycles.
- **Ingested two new documents** (2026-05-21) via the full ingest pipeline.
- **Fixed DuckDB schema errors**:
  - Explicit `CAST()` on all columns in facts view to prevent NULL-type inference errors
  - Fixed `cask_summary`: replaced `FIRST(value_raw ORDER BY _extracted_at DESC)` with `ANY_VALUE(value_raw)` (column doesn't exist)
  - Fixed `cost_summary`: removed missing `_description`/`_table_index` columns
  - Added `facts_timeline`, `facts_other` views
- **Fixed SQL generation quality** in `07_query.py`: tightened SQL_RULES with explicit column names (`country` not `entity`), EAV row semantics explanation, example queries.
- **Added `ingest.sh`**: new 4-step resumable pipeline (OCR → markdown → EAV extraction → DuckDB rebuild).
- **Added `08_export_excel.py`** and `attribute_registry.json` to repo.
- **Set up git credential store** (`~/.git-credentials` + `credential.helper=store`) so pushes work without prompting.
- **Committed and pushed all changes** to GitHub (5 commits, `d7a5ceb`).
- **CLAUDE.md updated**: pipeline diagram, Ollama crash details, Claude Code SSH permissions, AWS scaling plan, known data quality issues, extraction completion status, correct DuckDB view schema.

### What's Broken or Incomplete
- **Data quality issues** (known, not yet fixed):
  - HTML entities in entity names (e.g. `&#124;` instead of `|`)
  - Junk `cask_model` entities (stray numbers, file paths, table footers)
  - NULL units on some capacity rows
- **Query system tested structurally but not validated end-to-end** with real questions against live data.
- **ChromaDB / semantic search** not re-indexed after new documents were added — unclear if new docs are in the vector store.

---

## Current Focus (2026-05-21)

Pipeline is fully operational end-to-end. Immediate priorities before scaling:

1. **Validate the query system** — run `python 07_query.py --interactive` on VM and test 3–5 real questions covering capacity, cask specs, and cost data. Confirm routing, SQL generation, and synthesis are working correctly.
2. **Check ChromaDB** — verify new docs are indexed; re-run embedding ingest if not.
3. **Data quality triage** — decide whether to fix HTML entity / junk-entity issues now (small script) or defer until 70B model reprocessing on AWS.

---

## Open Questions

- Is ChromaDB current after the two new docs were ingested? (02_convert.py produces markdown; a separate step embeds into ChromaDB — was that step run?)
- At what point does data quality degrade enough to block useful queries? Current 8B extraction may be sufficient for most capacity/cask queries but unreliable for cost data.
- NeatDesk scanning strategy for 15 banker boxes — batching plan, folder organization.

## Blockers

- None currently. VM is stable, pipeline is running, git is set up.

## Known Debt

- HTML entity decoding in entity names
- Junk entity filtering for `cask_model`
- NULL unit population for capacity rows
- All three improve significantly with 70B model on AWS (see DECISIONS.md)
