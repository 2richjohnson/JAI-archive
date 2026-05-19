# JAI Archive Intelligence System

Hybrid RAG pipeline for the JAI Corporation nuclear consulting archive. Combines
semantic search (ChromaDB) with structured table queries (DuckDB) to answer
natural language questions over a document set that is heavily tabular in nature.

## Infrastructure

- **Ollama VM**: Ubuntu, NVIDIA GTX 1070 8GB, Ollama at `http://localhost:11434`
- **Models**: `llama3.1:8b` (generation/routing), `nomic-embed-text` (embeddings)
- **Python**: 3.14, venv at `~/jai-rag/`
- **ChromaDB**: `~/jai-archive/db/`
- **Markdown source**: `~/jai-archive/markdown/`

## Pipeline

```
JAI Documents (markdown/)
        │
        ├── 01_ocr.sh          — PDF → raw text
        ├── 02_convert.py      — raw text → markdown
        ├── 03_ingest.py       — markdown → ChromaDB (prose/narrative)
        ├── 05_extract_tables.py — markdown tables → normalized Parquet
        ├── 06_setup_duckdb.py — Parquet → DuckDB views
        └── 07_query.py        — unified natural language query interface
```

## Storage Layers

| Layer    | Backend  | Path                        | Content              |
|----------|----------|-----------------------------|----------------------|
| Semantic | ChromaDB | `~/jai-archive/db/`         | Prose and narrative  |
| Structured | DuckDB | `~/jai-archive/duckdb/jai.db` | Normalized tables  |

## Running the Pipeline

```bash
source ~/jai-rag/bin/activate

# Extract and normalize tables (resumable — skips already-processed files)
python 05_extract_tables.py

# Test against one file first
python 05_extract_tables.py --file JAI-490.md

# Dry-run: parse tables without calling Ollama
python 05_extract_tables.py --dry-run

# Initialize DuckDB from extracted Parquet files
python 06_setup_duckdb.py

# Query
python 07_query.py "Which countries have wet storage capacity over 5,000 MTU?"
python 07_query.py --interactive
```

## Directory Structure

```
jai-archive/
├── markdown/          — source documents
├── db/                — ChromaDB vector store
├── tables/
│   ├── manifest.json  — tracks processed files for resumability
│   └── parquet/
│       ├── capacity/
│       ├── cost/
│       ├── specifications/
│       ├── timeline/
│       └── other/
├── duckdb/
│   └── jai.db
└── logs/
    └── extraction.log
```

## Test Queries (JAI-490.md)

| Query | Expected backend |
|-------|-----------------|
| "What is Finland's wet storage capacity?" | DuckDB |
| "Which countries have dry storage over 5,000 MTU?" | DuckDB |
| "Compare UK and Germany spent fuel storage" | Hybrid |
| "What did JAI recommend for Korean utility storage?" | ChromaDB |
| "Total European wet storage capacity" | DuckDB |
