#!/usr/bin/env python3
"""
05_extract_tables.py — Extract and normalize tables from JAI archive markdown files.

Scans ~/jai-archive/markdown/ for markdown tables, sends each to Ollama for
normalization, and stores results as Parquet files organized by table type.
Resumable: skips files whose content hasn't changed since last run.

Usage:
    python 05_extract_tables.py [--file FILE] [--reprocess] [--dry-run]

    --file FILE    Process only this file (name relative to markdown/)
    --reprocess    Ignore manifest and reprocess everything
    --dry-run      Parse and log tables without calling Ollama
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ollama

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path.home() / "jai-archive"
MARKDOWN_DIR = BASE_DIR / "markdown"
TABLES_DIR = BASE_DIR / "tables" / "parquet"
LOGS_DIR = BASE_DIR / "logs"
MANIFEST_PATH = BASE_DIR / "tables" / "manifest.json"

TABLE_TYPES = ["capacity", "cost", "specifications", "timeline", "other"]
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

MIN_DATA_ROWS = 1
GARBAGE_THRESHOLD = 0.30


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "extraction.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


log = setup_logging()


# ── Manifest ───────────────────────────────────────────────────────────────────
def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime}"


# ── OCR Garbage Detection ──────────────────────────────────────────────────────
_NOISE_CHARS = re.compile(r"[^\w\s.,;:()\-/%'\"]")


def _cell_garbage_score(cell: str) -> float:
    """0.0 = clean, 1.0 = pure noise. Checks for random short tokens and noise chars."""
    cell = cell.strip()
    if not cell or re.fullmatch(r"[-:]+", cell):
        return 0.0

    tokens = cell.split()
    if not tokens:
        return 0.0

    # Tokens with ≤2 alphabetic characters that aren't standalone numbers
    def is_noise_token(t):
        alpha = re.sub(r"[^a-zA-Z]", "", t)
        return len(alpha) <= 2 and not re.fullmatch(r"[\d.,]+", t)

    token_noise = sum(1 for t in tokens if is_noise_token(t)) / len(tokens)
    char_noise = len(_NOISE_CHARS.findall(cell)) / max(len(cell), 1)

    # Short average token length is also suspicious
    avg_token_len = sum(len(t) for t in tokens) / len(tokens)
    length_penalty = max(0.0, (3.0 - avg_token_len) / 3.0) * 0.4

    return max(token_noise * 0.6, char_noise, length_penalty)


def row_is_garbled(cells: list[str]) -> bool:
    if not cells:
        return False
    scores = [_cell_garbage_score(c) for c in cells]
    return (sum(scores) / len(scores)) > GARBAGE_THRESHOLD


def is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r"[-:| ]+", c) for c in cells if c.strip())


# ── Table Extraction ───────────────────────────────────────────────────────────
def extract_tables(content: str, source_doc: str) -> list[dict]:
    """
    Walk markdown and collect all pipe-table blocks with their section context.
    Returns list of {raw, section_header, source_doc, table_index}.
    """
    lines = content.split("\n")
    tables = []
    current_section = "Introduction"
    i = 0

    while i < len(lines):
        line = lines[i]

        if re.match(r"^#{1,6}\s", line):
            current_section = line.lstrip("#").strip()

        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            tables.append(
                {
                    "raw": "\n".join(block),
                    "section_header": current_section,
                    "source_doc": source_doc,
                    "table_index": len(tables),
                }
            )
            continue

        i += 1

    return tables


def parse_table_rows(raw: str) -> tuple[list[str], list[list[str]], int]:
    """
    Parse raw markdown table into (headers, clean_data_rows, skipped_count).
    Filters out separator rows and garbled rows.
    """
    lines = [l for l in raw.strip().split("\n") if l.strip().startswith("|")]
    if not lines:
        return [], [], 0

    def split_row(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    all_rows = [split_row(l) for l in lines]
    headers = all_rows[0] if all_rows else []
    skipped = 0
    data_rows = []

    for row in all_rows[1:]:
        if is_separator_row(row):
            continue
        if row_is_garbled(row):
            log.debug("    Skipping garbled row: %s", row)
            skipped += 1
            continue
        data_rows.append(row)

    return headers, data_rows, skipped


# ── LLM Normalization ──────────────────────────────────────────────────────────
NORMALIZATION_PROMPT = """\
You are a data extraction assistant for a nuclear consulting archive.
Analyze this table from a JAI Corporation document and return ONLY valid JSON.

Document: {source_document}
Section context: {section_header}

Table:
{raw_table_markdown}

Return JSON with this structure:
{{
  "table_type": "capacity|cost|specification|timeline|other",
  "description": "brief description of what this table shows",
  "year": null or integer,
  "entity": null or string (country/utility/site if applicable),
  "columns": [
    {{"name": "normalized column name", "unit": "MTU|$/MTU|MW|etc or null"}}
  ],
  "rows": [
    {{"column_name": value}}
  ],
  "confidence": "high|medium|low",
  "notes": "any issues with OCR quality or data interpretation"
}}

Return ONLY the JSON object, no explanation, no markdown code blocks.\
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def normalize_table(table: dict) -> dict | None:
    """Send table to Ollama for normalization. Returns parsed dict or None on failure."""
    prompt = NORMALIZATION_PROMPT.format(
        source_document=table["source_doc"],
        section_header=table["section_header"],
        raw_table_markdown=table["raw"],
    )

    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0, "num_ctx": 4096},
        )
        raw_text = response.get("response", "").strip()
    except Exception as exc:
        log.warning(
            "    Ollama error for table %d in %s: %s",
            table["table_index"],
            table["source_doc"],
            exc,
        )
        return None

    # Strip accidental markdown code fences
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    m = _JSON_BLOCK.search(raw_text)
    if not m:
        log.warning(
            "    No JSON in LLM response for table %d of %s",
            table["table_index"],
            table["source_doc"],
        )
        return None

    try:
        normalized = json.loads(m.group())
    except json.JSONDecodeError as exc:
        log.warning(
            "    JSON parse error for table %d: %s", table["table_index"], exc
        )
        return None

    if not {"table_type", "columns", "rows"}.issubset(normalized):
        log.warning(
            "    Missing required keys in LLM output for table %d",
            table["table_index"],
        )
        return None

    # Normalize table_type to known set; map "specification" → "specifications"
    raw_type = normalized.get("table_type", "other").lower()
    type_map = {"specification": "specifications", "specs": "specifications"}
    normalized["table_type"] = type_map.get(raw_type, raw_type)
    if normalized["table_type"] not in TABLE_TYPES:
        normalized["table_type"] = "other"

    return normalized


# ── Parquet Storage ────────────────────────────────────────────────────────────
def save_to_parquet(normalized: dict, table: dict) -> Path | None:
    """
    Save normalized table rows to Parquet.
    Path: tables/parquet/<type>/<source_stem>_t<index>.parquet
    """
    table_type = normalized.get("table_type", "other")
    out_dir = TABLES_DIR / table_type
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = normalized.get("rows", [])
    if not rows:
        log.info("    Table %d: no rows to store", table["table_index"])
        return None

    source_stem = Path(table["source_doc"]).stem
    out_path = out_dir / f"{source_stem}_t{table['table_index']:03d}.parquet"

    df = pd.DataFrame(rows)

    # Metadata columns prefixed with _ to distinguish from data
    df["_source_doc"] = table["source_doc"]
    df["_section_header"] = table["section_header"]
    df["_table_index"] = table["table_index"]
    df["_table_type"] = table_type
    df["_description"] = normalized.get("description", "")
    df["_year"] = normalized.get("year")
    df["_entity"] = normalized.get("entity")
    df["_confidence"] = normalized.get("confidence", "low")
    df["_notes"] = normalized.get("notes", "")
    df["_extracted_at"] = datetime.utcnow().isoformat()

    df.to_parquet(out_path, index=False)
    log.info(
        "    Saved %d rows → %s [%s confidence]",
        len(df),
        out_path.name,
        normalized.get("confidence", "?"),
    )
    return out_path


# ── Per-file Processing ────────────────────────────────────────────────────────
def process_file(md_path: Path, dry_run: bool) -> dict:
    source_doc = md_path.name
    log.info("Processing: %s", source_doc)

    content = md_path.read_text(encoding="utf-8", errors="replace")
    raw_tables = extract_tables(content, source_doc)
    log.info("  Found %d table block(s)", len(raw_tables))

    stats = {
        "tables_found": len(raw_tables),
        "normalized": 0,
        "skipped": 0,
        "failed": 0,
    }

    for table in raw_tables:
        headers, data_rows, garbled_count = parse_table_rows(table["raw"])
        log.info(
            "  Table %d: %d header cols, %d data rows, %d garbled rows skipped",
            table["table_index"],
            len(headers),
            len(data_rows),
            garbled_count,
        )

        if len(data_rows) < MIN_DATA_ROWS:
            log.info("  Table %d: insufficient data rows, skipping", table["table_index"])
            stats["skipped"] += 1
            continue

        if dry_run:
            stats["normalized"] += 1
            continue

        normalized = normalize_table(table)
        if normalized is None:
            stats["failed"] += 1
            continue

        out = save_to_parquet(normalized, table)
        if out:
            stats["normalized"] += 1
        else:
            stats["skipped"] += 1

    return stats


# ── Entry Point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract and normalize tables from JAI archive markdown files."
    )
    parser.add_argument("--file", help="Process only this file (relative to markdown/)")
    parser.add_argument(
        "--reprocess", action="store_true", help="Ignore manifest and reprocess all files"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse tables only, skip LLM normalization"
    )
    args = parser.parse_args()

    for t in TABLE_TYPES:
        (TABLES_DIR / t).mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    if args.file:
        md_files = [MARKDOWN_DIR / args.file]
    else:
        md_files = sorted(MARKDOWN_DIR.glob("*.md"))

    if not md_files:
        log.error("No markdown files found in %s", MARKDOWN_DIR)
        sys.exit(1)

    totals = {
        "processed": 0,
        "skipped_cached": 0,
        "tables_found": 0,
        "normalized": 0,
        "failed": 0,
    }

    for md_path in md_files:
        if not md_path.exists():
            log.error("File not found: %s", md_path)
            continue

        fp = file_fingerprint(md_path)
        key = str(md_path)

        if not args.reprocess and manifest.get(key) == fp:
            log.info("Skipping (unchanged): %s", md_path.name)
            totals["skipped_cached"] += 1
            continue

        stats = process_file(md_path, dry_run=args.dry_run)
        totals["processed"] += 1
        totals["tables_found"] += stats["tables_found"]
        totals["normalized"] += stats["normalized"]
        totals["failed"] += stats["failed"]

        if not args.dry_run:
            manifest[key] = fp
            save_manifest(manifest)

    log.info(
        "\nDone. Files: %d processed, %d skipped (cached)"
        " | Tables: %d found, %d normalized, %d failed",
        totals["processed"],
        totals["skipped_cached"],
        totals["tables_found"],
        totals["normalized"],
        totals["failed"],
    )


if __name__ == "__main__":
    main()
