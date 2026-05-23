# JAI Archive — Project State

## Last Session Summary (2026-05-23)

### What We Accomplished
- **Diagnosed VM crash loop**: `ollama-gpu1.service` + `ollama.service` both auto-started on boot, both loaded llama3.1:8b (4.58 GiB) onto GPU1 simultaneously, exceeding 8 GiB VRAM → hard VM reset with no OOM warning. Confirmed via `journalctl -b -1`.
- **Disabled gpu1 service**: `systemctl disable --now ollama-gpu1.service`. Primary Ollama with `OLLAMA_NUM_PARALLEL=2` handles both GPUs internally.
- **Completed EAV extraction**: 177 markdown files → ~549 parquet files → 6,396 facts rows in DuckDB. Extraction ran ~4.5 hours in tmux, survived two crash/resume cycles.
- **Ingested two new documents** (2026-05-21) via the full ingest pipeline.
- **Fixed DuckDB schema errors**, added `ingest.sh`, `08_export_excel.py`, `attribute_registry.json`. Pushed to GitHub (`d7a5ceb`).

## Last Session Summary (2026-05-23)

### What We Accomplished
- **Fixed ChromaDB silent failure**: `07_query.py` was using `query_texts` which invoked ChromaDB's default 384-dim embedder, but the index was built with `nomic-embed-text` (768-dim) via Ollama. Every semantic search silently threw a dimension mismatch exception and returned `[]`. Fixed by adding `_embed()` using Ollama and switching to `query_embeddings`. Matches how `03_ingest.py` built the collection.
- **Fixed LLM router misclassification**: Descriptive questions ("tell me about X") were being routed to STRUCTURED by the LLM, skipping semantic search entirely. Added `_SEMANTIC_KEYWORDS` shortcut list (mirrors existing `_CAPACITY_KEYWORDS` / `_SPECS_KEYWORDS` pattern) so narrative questions bypass the LLM router.
- **Fixed SQL vendor matching**: Added SQL rule clarifying `cask_model` values are specific model names (e.g. `TN-68`), not vendor names — vendor queries should use `ILIKE 'TN-%'` etc.
- **Validated all three routing paths** interactively: semantic (rotary dissolvers ✅), structured (wet storage > 5000 MTU ✅), hybrid (transnuclear cask assemblies ✅).
- **Committed and pushed** (`0e461e4`).

### What's Broken or Incomplete
- **Data quality issues** (known, not yet fixed):
  - HTML entities in entity names (e.g. `&#124;` instead of `|`)
  - Junk `cask_model` entities (stray numbers, file paths, table footers)
  - NULL units on some capacity rows
- **ChromaDB** — unclear if the two new docs ingested 2026-05-21 were embedded into the vector store (03_ingest.py may not have been re-run after ingest).
- **SQL structured results sparse for cask queries** — `fuel_assembly_capacity` attribute mostly missing from `cask_summary` due to dirty 8B extraction; semantic path picks up the slack but structured data is thin.

---

## Current Focus (2026-05-23)

Query system is validated and working. Next priorities:

1. **Check ChromaDB indexing for new docs** — verify the two docs added 2026-05-21 are in the vector store; re-run `python 03_ingest.py` if not.
2. **Data quality triage** — decide whether to fix HTML entity / junk-entity issues now (small script) or defer until 70B model reprocessing on AWS.
3. **AWS scaling** — when ready, spin up g5.12xlarge spot, pull llama3.1:70B, re-run extraction with `--workers 4`.

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
