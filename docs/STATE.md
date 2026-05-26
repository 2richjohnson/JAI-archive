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

### Immediate Next Steps (continued same session)
- **DuckDB data quality cleanup completed** — see below.

---

## Last Session Summary (2026-05-26 continued)

### What We Accomplished
- **DuckDB data quality cleanup** in `06_setup_duckdb.py` view definitions (no re-extraction needed):
  - `facts` view: REGEXP_REPLACE decodes HTML entities (`&#124;` → `|`, `&#38;` → `&`) — 0 remaining
  - `capacity_summary`: COALESCE fills NULL units by attribute name (MTU/MTHM/year/%/MW) — 0 NULL units remaining
  - `cask_summary`: filter chain removes junk entities — pure numbers, comma-formatted numbers, scientific notation, numbered list items, path-like strings, column headers with parentheses. Distinct cask_models: 491 → 278
  - Fixed stale `entity` → `country` column reference in `print_info()`
- Committed `9c181c9` and pushed.

### What's Broken or Incomplete
- Nothing blocking. Query system and DuckDB both clean.
- Some borderline cask_model entries remain ("CANISTERED STORAGE/TRANSPORT SYSTEMS", "Type Cask/Canister") — not worth further filtering at 8B quality; will be clean in AWS 70B run.

### Immediate Next Steps
1. **Fix query routing gaps** (two known issues):
   - `_SEMANTIC_KEYWORDS` matches "summary of" but not "summary on" — add bare "summary" as trigger
   - UK/United Kingdom mismatch: SQL queries for `country = 'United Kingdom'` but data may be stored as `'UK'`; verbose output needed to confirm routing path
2. **Pipeline for new documents** — wire `03_ingest.py` into `ingest.sh` so new docs get ChromaDB-indexed automatically
3. **Continue query testing** — exercise more query types and document edge cases

---

## Last Session Summary (2026-05-26 continued — wiki layer)

### What We Accomplished
- **Implemented wiki generation layer** (`02b_generate_wiki.py`) — new pipeline step between `02_convert.py` and `03_ingest.py`:
  - Classifies each markdown file by entity type (CASK/COUNTRY/VENDOR/REGULATORY/FACILITY/TOPIC)
  - Generates structured wiki articles per entity to `~/jai-archive/wiki/{casks,countries,vendors,...}/`
  - Survey documents (JAI-490) generate multiple articles — one per country/entity detected
  - Merge logic: new documents enrich existing articles without overwriting
  - Registry (`wiki/processed_docs.json`) for resumability
  - CLI: `--doc`, `--force`, `--index-only`, `--validate`, `--stats`
- **Fixed classification for long survey documents**: initial run missed UK, Korea, Sweden etc. because classification only used first 500 words. Fixed by extracting all `##` headings from full document and passing those alongside the 500-word excerpt.
- **Fixed entity retrieval in `07_query.py`**: UK article ranked 29th by cosine similarity even though it existed — topic-heavy chunks from other countries outscored it. Added `_wiki_entity_filter()` that loads known entity names from `wiki/` at startup and applies `where={"entity_name": ...}` ChromaDB metadata filter when the query names a known entity. UK query now returns all 13 UK chunks exclusively.
- **Fixed `_SEMANTIC_KEYWORDS`**: added `"summary on"`, `"give me a summary"`, and bare `"summary"` — "give me a summary on X" now routes to SEMANTIC correctly.
- **Updated `03_ingest.py`**: auto-detects wiki/ vs markdown/ source; wiki articles get enhanced metadata (`entity_type`, `entity_name`, `source_documents`, `article_section`); `--source wiki/markdown/auto` flag.
- **Updated `ingest.sh`**: now 6-step pipeline; wiki generation (step 3) and ChromaDB rebuild (step 6) wired in.
- **Tested JAI-490.md**: generated 24 articles (13 countries, 5 facilities, 3 regulatory, 3 vendors) from one survey document. UK query now returns accurate answer.
- **Full corpus wiki generation running** in tmux on VM — 177 markdown files, ~5-6 hours total. Started 2026-05-26 18:30.

### What's Broken or Incomplete
- Wiki generation still running (~157 files remaining as of 19:12).
- After wiki run completes: need to rebuild ChromaDB (`python 03_ingest.py --rebuild`) to pick up all new articles.
- Duplicate log lines in `wiki_generation.log` — stdout also redirected to log file in tmux command. Cosmetic only, no functional impact.

### Immediate Next Steps (after wiki run completes)
1. `python 03_ingest.py --rebuild` — rebuild ChromaDB from all wiki articles
2. Test queries across multiple entity types (casks, countries, vendors)
3. `python 02b_generate_wiki.py --validate` — check for broken wikilinks
4. `python 02b_generate_wiki.py --stats` — review article counts by category

## Current Focus

1. Wiki generation running on full corpus (background, VM tmux session `wiki`)
2. After completion: ChromaDB rebuild, then query testing across entity types

---

## Open Questions

- NeatDesk scanning strategy for 15 banker boxes — batching plan, folder organization.

## Blockers

- None. Wiki generation running unattended in tmux.

## Known Debt

- Some borderline junk cask_model entities in DuckDB remain (section headers misclassified) — tolerable at 8B, clean in future 70B run
- Duplicate log lines in `wiki_generation.log` (cosmetic — stdout + FileHandler both writing to log)
- Wiki articles only cover entities found in the first classification pass — some entities in very long documents may be missed if not in headings (unlikely now that heading scan is implemented)
