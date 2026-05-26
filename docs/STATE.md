# JAI Archive — Project State

## Last Session Summary (2026-05-21)

### What We Accomplished
- **Diagnosed and fixed VM crash loop**: `ollama-gpu1.service` auto-start caused VRAM contention on GPU1 → disabled permanently. Primary Ollama with `OLLAMA_NUM_PARALLEL=2` handles both GPUs.
- **Completed EAV extraction**: 177 markdown files → ~549 parquet files → 6,396 facts rows in DuckDB.
- **Ingested two new documents** via full ingest pipeline.
- **Fixed DuckDB schema errors**, added `ingest.sh`, `08_export_excel.py`, `attribute_registry.json`.

---

## Last Session Summary (2026-05-25)

### What We Accomplished
- **Implemented document injection feature in `07_query.py`**: `--doc`, `--doc2`, `--pages`, `--deep`, `--model` CLI flags; `load_document_context()`, `parse_page_range()`, `two_stage_query()` — all implemented per spec.
- Feature adds: inject full markdown or PDF pages into query context, two-stage deep dive (ChromaDB retrieval → full document analysis), per-invocation model override.
- **GPU1 experiment concluded** — `NUM_PARALLEL=1` with qwen2.5:14b crashed Ollama (GPU1 hit 100%, GPU0 barely loaded). Hardware confirmed unreliable under any sustained load.
- **Permanent GPU1 fix applied**: `CUDA_VISIBLE_DEVICES=0` added to Ollama override; Ollama restarted. GPU1 invisible to Ollama permanently.
- **`07_query.py` reverted**: `LLM_MODEL` → `llama3.1:8b`, `SQL_MODEL` → `qwen2.5-coder:7b`. Synced to both VM locations.

### What's Broken / Pending
- Nothing blocking. System is operational.

### Immediate Next Steps
1. **Data quality triage** — fix HTML entity / junk-entity issues now, or defer to AWS 70B run
2. **AWS scaling** — g5.12xlarge spot, llama3.1:70B Q4_K_M, `--workers 4`, rebuild DuckDB, tear down

---

## Last Session Summary (2026-05-24)

### What We Accomplished
- **Fixed docling CUDA crash in `02_convert.py`**: GTX 1070 is Pascal (sm_61); PyTorch on Python 3.14 only provides CUDA 12.x builds which dropped sm_61 support, and no cu118 wheels exist for Python 3.14. Fixed by forcing `AcceleratorDevice.CPU` via `PdfPipelineOptions` / `PdfFormatOption`. Confirmed clean startup.
- **Added `02_convert.py` to git repo**: Was previously VM-only; now tracked in `projects/JAI-archive/` and synced to both VM locations.
- Pushed `f63e8c5`.

### What's Broken or Incomplete
- Nothing new. Ingest pipeline is operational again.

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

## Last Session Summary (2026-05-26)

### What We Accomplished
- **GPU1 experiment concluded**: `NUM_PARALLEL=1` with `qwen2.5:14b` crashed Ollama — GPU1 hit 100%, GPU0 barely loaded. Hardware confirmed unreliable. Permanent fix applied: `CUDA_VISIBLE_DEVICES=0` in Ollama override; models reverted to `llama3.1:8b` / `qwen2.5-coder:7b`.
- **Diagnosed thin query results**: synthesis was truncating semantic text at 3,000 chars, discarding most retrieved chunks. Raised to 8,000 chars; `num_ctx` 4,096 → 12,288 (no-inject) / 16,384 (with inject).
- **Rewrote `03_ingest.py`** with heading-aware chunking:
  - Splits by `##` sections instead of blind 300-word word-count cuts
  - Prepends `[doc_id] title\n## section\n` to every chunk — LLM always has structural context
  - Stores `doc_id`, `doc_family`, `title`, `section` metadata on every chunk
  - `--rebuild` flag wipes and rebuilds; `--file X.md` re-indexes one file
- **Full ChromaDB rebuild completed**: 6,314 chunks from 177 markdown files
- **Updated `07_query.py`**:
  - `_source_where_filter`: uses `doc_family`/`doc_id` metadata for clean ChromaDB `where` filtering when query names a JAI document — no more filename prefix-scanning
  - Fixed `_DOC_ID_RE` regex to match IDs with trailing letter (e.g. JAI-N006a)
  - Auto-inject matching markdown files when doc ID detected and `--doc` not specified
- **Tested**: content queries (shootaring canyon) and doc-name queries (JAI-N006 family) both significantly improved. Committed `9cab54c` and pushed.

### What's Broken or Incomplete
- Nothing blocking. Query system operational.

### Immediate Next Steps
1. **Data quality triage** — decide: fix HTML entity / junk-entity / NULL unit issues now, or defer entirely to the AWS 70B run
2. **AWS scaling** — g5.12xlarge spot, llama3.1:70B Q4_K_M, `--workers 4`, rebuild DuckDB, tear down

## Current Focus

1. **Data quality triage** — defer to AWS 70B run, or fix select issues now
2. **AWS scaling** — g5.12xlarge spot, llama3.1:70B Q4_K_M, `--workers 4`, rebuild DuckDB, tear down

---

## Open Questions

- At what point does data quality degrade enough to block useful queries? Current 8B extraction is sufficient for semantic queries but unreliable for structured cask/cost data.
- NeatDesk scanning strategy for 15 banker boxes — batching plan, folder organization.

## Blockers

- None. VM stable, GPU0-only Ollama running, query system operational.

## Known Debt

- HTML entity decoding in entity names
- Junk entity filtering for `cask_model`
- NULL unit population for capacity rows
- All three improve significantly with 70B model on AWS (see DECISIONS.md)
