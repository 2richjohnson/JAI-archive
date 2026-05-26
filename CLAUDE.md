# JAI Archive Intelligence System — CLAUDE.md

## What This Is
Hybrid RAG pipeline for the JAI Corporation nuclear consulting archive. Combines
semantic search (ChromaDB) with structured table queries (DuckDB) to answer
natural language questions over a document set that is heavily tabular.

## Infrastructure
- **Ollama VM**: 192.168.1.198, user `cccc`, password in `~/.ssh_pass`
- **GPU**: NVIDIA GTX 1070 × 2 (16GB VRAM total) — both cards visible to primary Ollama
- **Models**: `llama3.1:8b` (routing + synthesis + **extraction**), `qwen2.5-coder:7b` (SQL generation),
  `nomic-embed-text` (embeddings)
- **WARNING**: `qwen2.5:14b` was briefly set as default in `07_query.py` on 2026-05-25 and caused a hard VM crash (spread across both GPUs → VRAM contention). Safe models are 8b-class only on this VM. See DECISIONS.md.
- **Python**: 3.14, venv at `~/jai-rag/` on BOTH local and remote
- **Scripts live in two places on remote** — always sync both:
  - `~/projects/JAI-archive/` (canonical)
  - `~/jai-archive/` (user runs scripts from here)
- **Local dev machine** (`/home/bbbb/projects/JAI-archive/`): scripts + git only — no markdown/parquet data here
- **ChromaDB**: `~/jai-archive/db/` — collection `jai_archive`, **6,314 chunks** from 177 markdown files (heading-aware chunking as of 2026-05-26; metadata: doc_id, doc_family, title, section)
- **Markdown source**: `~/jai-archive/markdown/` (177 files)
- **DuckDB**: `~/jai-archive/duckdb/jai.db`
- **Canonical attribute registry**: `~/jai-archive/attribute_registry.json` (also in projects/)

## Claude Code Setup (Dev Machine)
Add these to `~/.claude/settings.json` to avoid permission prompts every turn:
```json
"permissions": {
  "allow": [
    "Bash(sshpass*)",
    "Bash(ssh *)",
    "Bash(nvidia-smi*)",
    "Bash(systemctl*)",
    "Bash(journalctl*)",
    "Bash(curl*)",
    "Bash(find *)",
    "Bash(grep *)",
    "Bash(tail*)",
    "Bash(wc *)",
    "Bash(ls*)",
    "Bash(ps *)",
    "Bash(ss *)",
    "Bash(tmux*)",
    "Bash(sleep*)"
  ]
}
```

## SSH / Remote Execution
```bash
sshpass -p "$(cat ~/.ssh_pass)" ssh -o StrictHostKeyChecking=no cccc@192.168.1.198 "<cmd>"
sshpass -p "$(cat ~/.ssh_pass)" scp -o StrictHostKeyChecking=no <local> cccc@192.168.1.198:<remote>
```
After copying scripts, always sync both remote locations:
```bash
sshpass -p "$(cat ~/.ssh_pass)" scp -o StrictHostKeyChecking=no 05_extract_tables.py cccc@192.168.1.198:~/projects/JAI-archive/
sshpass -p "$(cat ~/.ssh_pass)" ssh -o StrictHostKeyChecking=no cccc@192.168.1.198 "cp ~/projects/JAI-archive/05_extract_tables.py ~/jai-archive/"
# repeat for each changed file
```

## Ollama Service Configuration (VM) — CRITICAL
Two systemd services exist. **Only the primary should be used and enabled.**

### Primary (USE THIS)
- Service: `ollama.service`, port **11434**
- Config override: `/etc/systemd/system/ollama.service.d/override.conf`

**Current config (permanent as of 2026-05-25):**
```
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="CUDA_VISIBLE_DEVICES=0"
```

- `CUDA_VISIBLE_DEVICES=0`: GPU1 permanently excluded — hardware unreliable under sustained load
- `OLLAMA_NUM_PARALLEL=1`: no concurrent requests; single inference at a time
- GPU0 only (~8 GiB usable VRAM) — safe for `llama3.1:8b` + `nomic-embed-text`
- **Do not revert** `CUDA_VISIBLE_DEVICES=0`. Do not attempt 14B+ models on this VM.

**Current status**: Stable. Ollama running on GPU0 only. Query system operational.

### Secondary (DISABLED — DO NOT RE-ENABLE)
- Service: `ollama-gpu1.service`, port **11435**
- Config: `CUDA_VISIBLE_DEVICES=1`, `OLLAMA_HOST=0.0.0.0:11435`
- **Disabled 2026-05-20** after causing two consecutive VM hard crashes
- **Crash mechanism**: Primary Ollama uses GPU1 as part of its multi-GPU spread. When gpu1 service also loads onto GPU1, combined VRAM exceeds 8 GiB → immediate hard VM reset, no OOM warning. Crash loop on reboot.
- If it somehow gets re-enabled: `echo "$(cat ~/.ssh_pass)" | sudo -S systemctl disable --now ollama-gpu1.service`

### GPU1 — PERMANENTLY EXCLUDED FROM OLLAMA
- **2026-05-25**: Second crash caused by `qwen2.5:14b` spreading across both GPUs from the *primary* service (no secondary service running). All VMs on the Proxmox host went down; Proxmox survived.
- **Fix**: Add `Environment="CUDA_VISIBLE_DEVICES=0"` to primary service override → GPU1 invisible to Ollama entirely.
- GPU1 is unreliable under any high-VRAM load. Do not attempt to re-include it without diagnosing the hardware.

### Safe Extraction Command
```bash
# On the VM, from ~/jai-archive/, with venv active:
source ~/jai-rag/bin/activate
nohup python 05_extract_tables.py --workers 1 > logs/extraction_run.log 2>&1 &
# OR in a tmux session:
tmux new -s extract
python 05_extract_tables.py --workers 1
```
**Do NOT use `--hosts` flag** — activates dual-Ollama mode, caused crashes.
**Use `--workers 1`** (not 2) — GPU1 is now disabled; two concurrent requests no longer benefit from parallelism and may queue unnecessarily.

## Current Pipeline (EAV Architecture + Wiki Layer)
```
NeatDesk scanner
  └── raw/          ← drop new PDFs here
        └── 01_ocr.sh              — ocrmypdf: add searchable text layer (skips existing)
              └── ocr/
                    └── 02_convert.py          — docling: PDF → markdown (skips existing)
                          └── markdown/
                                └── 02b_generate_wiki.py  — LLM: markdown → wiki articles  ← NEW
                                      └── wiki/{casks,countries,...}/
                                            └── 03_ingest.py  — ChromaDB indexer (uses wiki/ if present)
                                            └── 05_extract_tables.py  — LLM: tables → EAV Parquet
                                                  └── tables/parquet/
                                                        └── 06_setup_duckdb.py  — Parquet → DuckDB
                                                              └── duckdb/jai.db
                                                                    └── 07_query.py  — NL query interface
```

Scripts:
- `ingest.sh` — runs all 6 steps in sequence, every step resumable
- `02b_generate_wiki.py` — wiki generator (LLM classifies + generates entity articles; `--doc X.md`, `--force`, `--index-only`, `--validate`, `--stats`)
- `03_ingest.py` — ChromaDB indexer; auto-detects wiki/ vs markdown/ source; `--source wiki/markdown/auto`, `--rebuild`, `--file X.md`
- `05_extract_tables.py` — LLM 2-pass extraction (llama3.1:8b)
- `06_setup_duckdb.py` — builds DuckDB views from all parquet
- `07_query.py` — hybrid NL query (DuckDB + ChromaDB)
- `08_export_excel.py` — cost study → Excel (openpyxl)

## Ingest Workflow (for new documents)
```bash
# 1. Copy new PDF(s) to VM
sshpass -p "$(cat ~/.ssh_pass)" scp -o StrictHostKeyChecking=no \
  MyDoc.pdf cccc@192.168.1.198:~/jai-archive/raw/

# 2. SSH in and run the pipeline (all steps resumable)
sshpass -p "$(cat ~/.ssh_pass)" ssh -o StrictHostKeyChecking=no cccc@192.168.1.198
cd ~/jai-archive && ./ingest.sh
```

Individual steps if needed:
```bash
source ~/jai-rag/bin/activate
bash 01_ocr.sh                                # raw/ → ocr/
python 02_convert.py                          # ocr/ → markdown/
python 02b_generate_wiki.py                   # markdown/ → wiki/ (skips already-processed)
python 02b_generate_wiki.py --doc X.md        # single file
python 02b_generate_wiki.py --force           # regenerate all articles
python 02b_generate_wiki.py --stats           # article count by category
python 02b_generate_wiki.py --validate        # check broken wikilinks
python 03_ingest.py --rebuild                 # wiki/ → ChromaDB (auto-detects source)
python 03_ingest.py --source markdown --rebuild  # force markdown mode (bypass wiki)
python 05_extract_tables.py --workers 2       # markdown/ → parquet/
python 05_extract_tables.py --workers 1       # single worker (safer)
python 05_extract_tables.py --file X.md       # single file test
python 06_setup_duckdb.py --rebuild           # parquet/ → DuckDB

# Query (basic)
python 07_query.py "question"
python 07_query.py --interactive
python 07_query.py --verbose "question"           # shows routing + SQL

# Query with document injection (bypasses ChromaDB retrieval for that doc)
python 07_query.py --doc ~/jai-archive/markdown/JAI-490.md "question"
python 07_query.py --doc ~/jai-archive/ocr/JAI-490.pdf --pages 89-95 "question"
python 07_query.py --doc ... --doc2 ... "question"

# Two-stage deep dive (ChromaDB → identify top doc → re-query with full doc)
python 07_query.py --deep "question"

# Override model for one session (default is llama3.1:8b — do NOT use 14b on this VM)
python 07_query.py --model llama3.1:8b "question"

# Export cost studies to Excel
python 08_export_excel.py
python 08_export_excel.py --source JAI-497
```

## EAV Schema (facts table in DuckDB)
Every extracted data point is one row:
```
entity          TEXT    -- country name, cask model, program, facility
entity_type     TEXT    -- country | cask_model | facility | cost_study | other
row_label       TEXT    -- original row label (for cost line items)
attribute       TEXT    -- canonical snake_case name (e.g. wet_storage_mtu)
value_raw       TEXT    -- original string value from document
value_numeric   DOUBLE  -- numeric value (null for text-only)
unit            TEXT    -- MTU | lb | in | kW | M$ | year | % | etc.
line_item_type  TEXT    -- input | calculated | subtotal | total (cost rows only)
_source_doc     TEXT
_table_type     TEXT    -- capacity | cost | specifications | timeline | other
_year           INT
_currency_year  INT
_confidence     TEXT
```

DuckDB views:
- `facts` — all EAV rows
- `facts_capacity`, `facts_specifications`, `facts_cost`, `facts_timeline`, `facts_other` — filtered by _table_type
- `capacity_summary(country, attribute, value_numeric, unit, source_docs)` — MAX per (country, attribute)
- `cask_summary(cask_model, attribute, value_raw, value_numeric, unit, source_doc)` — per (cask_model, attribute)
- `cost_summary(source_doc, study_year, currency_year, entity, row_label, attribute, value_raw, value_numeric, unit, line_item_type)` — cost line items

## Two-Pass Extraction Design (05_extract_tables.py)
**Problem solved**: LLM stopped after first entity in multi-row tables.
- **Pass 1 (LLM)**: headers + 3 sample rows → schema: `{format, entity_column, column_map, ...}`
- **Pass 2 (Python)**: iterates ALL data rows using the schema — guaranteed complete

Table formats:
- `wide` — rows are different entities (e.g. each row is a country/cask model)
- `tall` — rows are attributes of one entity (key-value structure)
- `cost` — rows are line items with one or more value columns

## Query Routing (07_query.py)
Three keyword shortcut lists bypass the LLM router for reliability:
- `_SEMANTIC_KEYWORDS` — "tell me about", "explain", "summarize", "what is/are", "how does/do", etc. → always SEMANTIC
- `_SPECS_KEYWORDS` — cask/canister model names, vendor names, spec terms → always HYBRID
- `_CAPACITY_KEYWORDS` — MTU, wet/dry storage, inventory, saturation → always STRUCTURED
- LLM router handles anything not matched by the above; defaults to HYBRID on uncertain output.

**ChromaDB embedding**: The collection was indexed with `nomic-embed-text` (768-dim) via Ollama.
`semantic_search()` must use `query_embeddings` (pre-embedded via Ollama) — **never `query_texts`**,
which invokes ChromaDB's default 384-dim embedder and silently returns empty results.

## Known Issues / Behavior
- `entity_column` returned by LLM may be canonical name (e.g. "cask_model") not original header ("Cask Designation"); `_col_index()` normalizes both sides to alphanumeric for matching
- `entity_type` varies slightly in LLM output ("cask_design" vs "cask_model"); acceptable
- Occasional "no EAV records" for very simple/degenerate tables — normal
- `openpyxl` required for 08_export_excel.py: `pip install openpyxl`
- **HTML entities in entity names**: LLM sometimes outputs `&#124;` (for `|`) in cask model names; needs cleanup query or preprocessing fix
- **Junk cask_model entities**: LLM occasionally classifies stray numbers, file paths, or table footers as `entity_type=cask_model`; filter with `WHERE LENGTH(cask_model) > 3 AND cask_model NOT REGEXP '^[0-9.]+$'`
- **NULL units in capacity rows**: Some extracted capacity values have `unit IS NULL` despite being MTU; likely a prompting gap in 05_extract_tables.py
- **Data quality**: All known issues above become less severe with 70B model (see AWS plan below)

## Query System Status (as of 2026-05-26)
- **Operational** — wiki generation in progress on full corpus
- **All routing paths validated**: semantic ✅, structured ✅, hybrid ✅
- **Flags**: `--doc`, `--doc2`, `--pages`, `--deep`, `--model`, `--verbose`, `--interactive`
- **Doc-ID queries**: naming a JAI document (e.g. "JAI-N006") triggers metadata filtering + auto-inject of matching markdown files
- **Wiki entity queries**: naming a known entity (e.g. "United Kingdom", "TN-40") triggers `entity_name` metadata filter — returns that entity's article chunks exclusively
- ChromaDB: currently 264 chunks from 24 wiki articles (JAI-490 only); will be rebuilt from full wiki after generation completes
- Wiki articles in `~/jai-archive/wiki/` — rebuild ChromaDB after generation: `python 03_ingest.py --rebuild`

## Extraction Progress (as of 2026-05-21)
- Extraction **completed** 2026-05-20 after two crash/resume cycles
- **177 markdown files** processed; two new documents added and ingested 2026-05-21
- Parquet output in `~/jai-archive/tables/parquet/` (EAV):
  - `capacity/`: 40 files
  - `cost/`: 1 file
  - `specifications/`: 195 files
  - `other/`: 311 files
  - `timeline/`: 2 files
  - Total: ~549 parquet files → **6,396 facts rows** in DuckDB
- Extraction is **resumable** — script skips files whose parquet already exists (hash-based)
- VM crashed twice (see Ollama section above); completed on single Ollama with `--workers 2`
- Monitor progress: `tail -f ~/jai-archive/logs/extraction_run.log`
- Check parquet count: `find ~/jai-archive/tables/parquet -name '*.parquet' | wc -l`

## Scaling to Full Corpus — AWS Plan
Current extraction (llama3.1:8b on GTX 1070 × 2) is too slow and produces dirty data
(junk entities, missing units, HTML artifacts) for the full 15-banker-box corpus.

**Privacy note**: Data is sensitive — do NOT use Claude API (Anthropic) for extraction.
All processing must stay within user-controlled infrastructure (local VM or AWS).

**Recommended: AWS g5.12xlarge spot instance (one-time extraction run)**
- 4× NVIDIA A10G GPUs, 96GB VRAM — runs llama3.1:70B Q4_K_M (~40GB)
- Spot price: ~$2–2.50/hr
- Same Ollama + 05_extract_tables.py setup — no code changes, just set `--workers 4`
- Estimated time for ~3,000 pages / ~5,000 tables: **10–15 hours, ~$25–40 total**
- 70B model eliminates most entity classification and unit extraction errors

**Smaller option**: g5.2xlarge spot (~$0.35/hr, 1× A10G 24GB)
- Run llama3.1:8b very fast OR llama3.3:70B at aggressive quantization (~20GB)
- ~20–30 hours, ~$10 — lower quality than 70B Q4 but much faster than current setup

After extraction, tear down the instance — query stays on local VM forever.

## Next Phase — Scaling to 15 Banker Boxes
- Attribute registry stabilizes after ~50 new documents
- New attributes emerge as novel: LLM uses descriptive snake_case fallback
- Periodically review `SELECT attribute, COUNT(*) FROM facts GROUP BY attribute` to add new canonical names
- After extraction completes, run `python 06_setup_duckdb.py` to rebuild DuckDB views

## Scale Context
- Current: 7 PDFs → 177 markdown files → 629 tables to extract
- Planned: ~15 banker boxes (1,000–3,000 pages, thousands of tables)
- Same topical area: nuclear fuel storage, transport cask specs, cost studies, country surveys
