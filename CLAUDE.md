# JAI Archive Intelligence System — CLAUDE.md

## What This Is
Hybrid RAG pipeline for the JAI Corporation nuclear consulting archive. Combines
semantic search (ChromaDB) with structured table queries (DuckDB) to answer
natural language questions over a document set that is heavily tabular.

## Infrastructure
- **Ollama VM**: 192.168.1.198, user `cccc`, password in `~/.ssh_pass`
- **GPU**: NVIDIA GTX 1070 × 2 (16GB VRAM total) — both cards active after reboot
- **Models**: `llama3.1:8b` (routing + synthesis), `qwen2.5-coder:7b` (SQL generation),
  `llama3.2:3b` (table extraction — currently, upgrading to 8b in next phase),
  `nomic-embed-text` (embeddings)
- **Python**: 3.14, venv at `~/jai-rag/` on BOTH local and remote
- **Scripts live in two places on remote** — always sync both:
  - `~/projects/JAI-archive/` (canonical)
  - `~/jai-archive/` (user runs from here — keep in sync with scp + cp)
- **ChromaDB**: `~/jai-archive/db/` — collection `jai_archive`, 1367 documents
- **Markdown source**: `~/jai-archive/markdown/` (177 files)
- **DuckDB**: `~/jai-archive/duckdb/jai.db`

## SSH / Remote Execution
```bash
sshpass -p "$(cat ~/.ssh_pass)" ssh -o StrictHostKeyChecking=no cccc@192.168.1.198 "<cmd>"
sshpass -p "$(cat ~/.ssh_pass)" scp -o StrictHostKeyChecking=no <local> cccc@192.168.1.198:<remote>
```
After copying scripts, always sync: `cp ~/projects/JAI-archive/07_query.py ~/jai-archive/07_query.py`

## Current Pipeline
```
PDFs (~/home/*.pdf)
  └── markdown/ (already converted, 177 files)
        ├── 05_extract_tables.py  — markdown tables → Parquet (llama3.2:3b via Ollama)
        ├── 06_setup_duckdb.py    — Parquet → DuckDB views
        └── 07_query.py           — unified NL query interface
```
Steps 01–04 (OCR, conversion, ChromaDB ingest) were done in a prior session.
ChromaDB is already populated. DuckDB is populated.

## Running the Pipeline
```bash
source ~/jai-rag/bin/activate

# Extract tables (resumable — skips already-processed files)
python 05_extract_tables.py

# Rebuild DuckDB views from Parquet
python 06_setup_duckdb.py

# Query
python 07_query.py "Which countries have wet storage capacity over 5,000 MTU?"
python 07_query.py --interactive
python 07_query.py --verbose "question"   # shows SQL
```

## DuckDB Schema (current — being replaced in next phase)
- `tables_all` — union of all extracted Parquet (wide schema, 250+ columns, unwieldy)
- `capacity_normalized` — hand-crafted view unifying wet/dry MTU column variants
- `tables_capacity/cost/specifications/timeline/other` — views by type
- **Known limitation**: wide schema makes SQL generation unreliable for non-capacity queries

## 07_query.py Architecture
- **Dual model**: `llama3.1:8b` for routing + synthesis, `qwen2.5-coder:7b` for SQL only
- **Keyword router** (bypasses LLM): cask/equipment keywords → HYBRID, capacity comparison → STRUCTURED
- **Direct equipment query** (`_equipment_query`): bypasses LLM SQL for vendor/model lookups
- **ChromaDB filters**: vendor/model terms trigger `where_document` content filtering
- **Capacity fallback**: countries not in `capacity_normalized` fall back to `tables_all` + ChromaDB

## Known Issues / Workarounds in Current Code
- `column_name` field used as row-value by 3b model — patched in `capacity_normalized` UNION
- Finland data classified as `other` — fixed by fallback UNION in `capacity_normalized`
- DuckDB SQL errors on equipment queries — bypassed by `_equipment_query` direct pattern
- Two copies of scripts on remote must be kept in sync manually

## Next Phase — EAV Refactor (PLANNED, NOT STARTED)
**Goal**: Replace wide Parquet schema with Entity-Attribute-Value canonical facts table.

### Canonical facts schema:
```
source_doc | entity | entity_type | attribute | value_raw | value_numeric | unit |
table_type | line_item_type | study_year | currency_year | scenario | confidence
```

### Canonical attribute registry (~150-200 names):
Maps all column name variants to canonical names, e.g.:
- "Installed Wet Storage Capacity (MTU)", "Wet Storage (MTU)", "Wet Storage" → `wet_storage_mtu`
- "No. Fuel Assemblies", "Capacity (assys)", "Capacity (intact assemblies)" → `fuel_assembly_capacity`
- "Net Capacity (MW)", "Design Capacity (MTU)" → disambiguated by entity_type

### Cost table extraction:
- Separate extraction path for cost/calculation tables
- Captures: line_item_type (input/calculated/subtotal/total), implied_formula, assumptions
- Step 08 export: reconstruct Excel workbook from cost facts (openpyxl)
- Reconstructed Excel carries provenance note — verify formulas before financial use

### Parallel extraction with dual GPU:
- Two worker processes, one model instance per GPU (Ollama auto-detects both 1070s)
- Use `llama3.1:8b` for extraction (better classification, fewer misclassification errors)
- Expected: ~2x throughput vs. single GPU

### Steps to implement:
1. Define canonical attribute registry (curated list in a JSON file)
2. Rewrite `05_extract_tables.py` — new prompt outputs EAV rows + cost metadata
3. Rewrite `06_setup_duckdb.py` — single `facts` table + standard views
4. Simplify `07_query.py` — remove all the workaround code, simpler SQL prompt
5. New `08_export_excel.py` — cost table → Excel reconstruction

## Scale Context
- Current: 7 PDFs → 177 markdown files → 553 tables extracted
- Planned: ~15 banker boxes of JAI documents (est. 1,000–3,000 pages, thousands of tables)
- Same topical area: nuclear fuel storage, transport cask specs, cost studies, country surveys
- Attribute registry stabilizes after first ~50 new documents; mostly reuses existing attributes
