# JAI Archive — Project State

## Last Session Summary (2026-05-21)

### What We Accomplished
- **Diagnosed and fixed VM crash loop**: `ollama-gpu1.service` auto-start caused VRAM contention on GPU1 → disabled permanently. Primary Ollama with `OLLAMA_NUM_PARALLEL=2` handles both GPUs.
- **Completed EAV extraction**: 177 markdown files → ~549 parquet files → 6,396 facts rows in DuckDB.
- **Ingested two new documents** via full ingest pipeline.
- **Fixed DuckDB schema errors**, added `ingest.sh`, `08_export_excel.py`, `attribute_registry.json`.

---

## Last Session Summary (2026-05-23)

### What We Accomplished
- **Fixed ChromaDB silent failure** (root cause of all "no results" queries): `07_query.py` used `query_texts` → ChromaDB's default 384-dim embedder, but index was built with `nomic-embed-text` (768-dim). Every query threw a silent dimension mismatch exception. Fixed: added `_embed()` via Ollama, switched to `query_embeddings`.
- **Fixed LLM router misclassification**: Descriptive questions routed to STRUCTURED, skipping semantic search. Added `_SEMANTIC_KEYWORDS` shortcut list — "tell me about", "explain", "summarize", etc. bypass the LLM router.
- **Fixed SQL vendor matching**: Added rule that `cask_model` values are specific model names; vendor queries need `ILIKE 'TN-%'` not exact match.
- **Validated all three routing paths** interactively: semantic (rotary dissolvers ✅), structured (wet storage > 5000 MTU ✅), hybrid (transnuclear cask assemblies ✅).
- **Confirmed ChromaDB is current**: 170/177 files indexed; 7 excluded are junk (MS Office temp files, image-only pages) — no action needed.
- **Added `.claude/settings.json`**: SSH/remote commands (sshpass, ssh, find, grep, etc.) now auto-allowed without per-call permission prompts.
- **Fixed Claude Code auto-update**: Reinstalled to `~/.npm-global` (user-writable); version bumped 2.1.138 → 2.1.150. Auto-updates will now work.
- **Improved session continuity**: Created `/home/bbbb/CLAUDE.md` (root-level project index) and updated memory files with project path so future sessions start without needing to search for project files.

### What's Broken or Incomplete
- **Data quality issues** (known, deferred to AWS 70B run):
  - HTML entities in entity names (e.g. `&#124;` instead of `|`)
  - Junk `cask_model` entities (stray numbers, file paths, table footers)
  - NULL units on some capacity rows
- **SQL structured results sparse for cask queries** — `fuel_assembly_capacity` mostly missing from `cask_summary` due to dirty 8B extraction; semantic path compensates but structured data is thin.

---

## Current Focus

Pipeline and query system are fully operational. Next priorities:

1. **Data quality triage** — decide whether to fix HTML entity / junk-entity issues now (small cleanup script) or defer entirely until 70B model reprocessing on AWS.
2. **AWS scaling** — when ready: spin up g5.12xlarge spot, install Ollama, pull llama3.1:70B Q4_K_M, re-run `05_extract_tables.py --workers 4`, rebuild DuckDB, tear down instance.

---

## Open Questions

- At what point does data quality degrade enough to block useful queries? Current 8B extraction is sufficient for semantic queries but unreliable for structured cask/cost data.
- NeatDesk scanning strategy for 15 banker boxes — batching plan, folder organization.

## Blockers

- None. VM stable, pipeline running, query system validated, git set up, auto-updates fixed.

## Known Debt

- HTML entity decoding in entity names
- Junk entity filtering for `cask_model`
- NULL unit population for capacity rows
- All three improve significantly with 70B model on AWS (see DECISIONS.md)
