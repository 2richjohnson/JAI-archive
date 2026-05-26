#!/bin/bash
# JAI Archive — Full ingest pipeline
# Usage: ./ingest.sh
#   Drop new PDFs into ~/jai-archive/raw/ first, then run this script.
#   Every step is resumable — already-processed files are skipped automatically.

set -e
cd ~/jai-archive
source ~/jai-rag/bin/activate

echo "========================================"
echo " JAI Archive Ingest Pipeline"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Step 1 — OCR: raw/ → ocr/
echo ""
echo "[1/6] OCR  (raw/ → ocr/)"
bash ~/jai-archive/01_ocr.sh

# Step 2 — Markdown conversion: ocr/ → markdown/
echo ""
echo "[2/6] Markdown conversion  (ocr/ → markdown/)"
python3 02_convert.py

# Step 3 — Wiki generation: markdown/ → wiki/
echo ""
echo "[3/6] Wiki generation  (markdown/ → wiki/)"
python3 02b_generate_wiki.py

# Step 4 — EAV extraction: markdown/ → tables/parquet/
echo ""
echo "[4/6] EAV extraction  (markdown/ → tables/parquet/)"
python3 05_extract_tables.py --workers 2

# Step 5 — Rebuild DuckDB
echo ""
echo "[5/6] Rebuilding DuckDB"
python3 06_setup_duckdb.py --rebuild

# Step 6 — Rebuild ChromaDB from wiki articles
echo ""
echo "[6/6] Rebuilding ChromaDB  (wiki/ → db/)"
python3 03_ingest.py --rebuild

echo ""
echo "========================================"
echo " Ingest complete  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
