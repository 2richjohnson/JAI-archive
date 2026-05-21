#!/usr/bin/env python3
"""
05_extract_tables.py — Extract tables from JAI archive markdown files into EAV format.

Two-pass design:
  Pass 1 (LLM): classify table, identify entity column, map column headers to canonical
                attribute names. Sends only headers + 2 sample rows — small context.
  Pass 2 (Python): iterate ALL data rows using the LLM-provided mapping. No truncation.

Output: one Parquet file per table, flat EAV schema, directory tables/eav/

Usage:
    python 05_extract_tables.py [--file FILE] [--reprocess] [--dry-run] [--workers N]

    --file FILE    Process only this file (name relative to markdown/)
    --reprocess    Ignore manifest and reprocess everything
    --dry-run      Parse and log tables without calling Ollama
    --workers N    Parallel Ollama workers (default: 2)
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import ollama

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path.home() / "jai-archive"
MARKDOWN_DIR = BASE_DIR / "markdown"
TABLES_DIR = BASE_DIR / "tables" / "eav"
LOGS_DIR = BASE_DIR / "logs"
MANIFEST_PATH = BASE_DIR / "tables" / "eav_manifest.json"
REGISTRY_PATH = Path(__file__).parent / "attribute_registry.json"

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# When running dual-GPU, populate this list with one host per GPU.
# Workers round-robin across hosts. Falls back to OLLAMA_HOST if empty.
OLLAMA_HOSTS: list[str] = []

MIN_DATA_ROWS = 1
GARBAGE_THRESHOLD = 0.30

# Fixed EAV output schema — every Parquet file has exactly these columns
EAV_COLUMNS = [
    "entity", "entity_type", "row_label", "attribute",
    "value_raw", "value_numeric", "unit", "line_item_type",
    "_source_doc", "_section_header", "_table_index", "_table_type",
    "_description", "_year", "_currency_year", "_table_entity",
    "_confidence", "_notes", "_extracted_at",
]

# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "eav_extraction.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


log = setup_logging()


# ── Attribute Registry ─────────────────────────────────────────────────────────
def load_registry() -> tuple[str, set[str]]:
    if not REGISTRY_PATH.exists():
        log.warning("attribute_registry.json not found at %s", REGISTRY_PATH)
        return "", set()

    reg = json.loads(REGISTRY_PATH.read_text())
    lines = ["CANONICAL ATTRIBUTE NAMES (map every column to the closest one):"]
    valid = set()
    for domain in reg.get("domains", []):
        attr_names = [a["name"] for a in domain["attributes"]]
        lines.append(f"  {domain['name'].upper()}: {', '.join(attr_names)}")
        valid.update(attr_names)
    lines.append("  If no canonical name fits, use descriptive snake_case (e.g. supplier_name).")
    return "\n".join(lines), valid


ATTRIBUTE_PROMPT_BLOCK, VALID_ATTRIBUTES = load_registry()


# ── Manifest ───────────────────────────────────────────────────────────────────
_manifest_lock = threading.Lock()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime}"


# ── OCR Garbage Detection ──────────────────────────────────────────────────────
_NOISE_CHARS = re.compile(r"[^\w\s.,;:()\-/%'\"]")


def _cell_garbage_score(cell: str) -> float:
    cell = cell.strip()
    if not cell or re.fullmatch(r"[-:]+", cell):
        return 0.0
    tokens = cell.split()
    if not tokens:
        return 0.0

    def is_noise_token(t):
        alpha = re.sub(r"[^a-zA-Z]", "", t)
        return len(alpha) <= 2 and not re.fullmatch(r"[\d.,]+", t)

    token_noise = sum(1 for t in tokens if is_noise_token(t)) / len(tokens)
    char_noise = len(_NOISE_CHARS.findall(cell)) / max(len(cell), 1)
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
            tables.append({
                "raw": "\n".join(block),
                "section_header": current_section,
                "source_doc": source_doc,
                "table_index": len(tables),
            })
            continue
        i += 1

    return tables


def parse_table_rows(raw: str) -> tuple[list[str], list[list[str]], int]:
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


# ── Pass 1: LLM Column Classification ─────────────────────────────────────────
#
# We send only the header row + up to 3 sample rows.
# The LLM returns a schema describing how to interpret this table.
# Python then applies the schema to ALL rows (no LLM truncation).

COLUMN_MAP_PROMPT = """\
You are a data extraction assistant for a nuclear consulting archive.
Analyze this table's STRUCTURE and return a schema for programmatic extraction.

Document: {source_document}
Section: {section_header}

Table headers and sample rows:
{sample_markdown}

{attribute_block}

DETERMINE:
1. table_type: "capacity" | "cost" | "specifications" | "timeline" | "other"
2. entity_type: "country" | "cask_model" | "facility" | "utility" | "cost_study" | "other"
3. format:
   - "wide": rows are different entities (e.g. each row is a country or cask model)
   - "tall": rows are attributes of ONE entity (e.g. "Parameter | Value" structure)
   - "cost": rows are line items with one or more cost columns
4. For "wide": entity_column is the header of the column whose values are entity names
5. For "tall": value_column is the header of the data column; entity comes from table context
6. column_map: maps EVERY non-entity column header to a canonical attribute name
   - Strip footnote markers (a, b, c, *, etc.) from column names when mapping
   - Include units in the canonical name where known (e.g. "weight_loaded_lb")
7. For "cost": also populate line_item_rules — map row label patterns to line_item_type
   ("input", "calculated", "subtotal", "total")

Return ONLY valid JSON:
{{
  "table_type": "...",
  "entity_type": "...",
  "table_entity": null or "entity name if same for all rows (for tall/cost format)",
  "format": "wide|tall|cost",
  "entity_column": null or "exact original column header string",
  "entity_column_index": null or integer (0-based column position, use as fallback),
  "value_column": null or "exact original column header string (for tall format)",
  "year": null or integer,
  "currency_year": null or integer,
  "description": "one-line table description",
  "confidence": "high|medium|low",
  "notes": "OCR issues or interpretation notes",
  "column_map": {{
    "Original Column Header": "canonical_attribute_name"
  }},
  "line_item_rules": {{
    "Total": "total",
    "Subtotal": "subtotal"
  }}
}}\
"""


def _extract_first_json(text: str) -> str | None:
    depth = 0
    start = None
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def _sample_markdown(headers: list[str], data_rows: list[list[str]], max_rows: int = 3) -> str:
    """Build a compact markdown snippet: header + separator + up to max_rows data rows."""
    sep = ["-" * max(len(h), 3) for h in headers]
    rows_md = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in data_rows[:max_rows]:
        # Pad or trim row to match header count
        padded = list(row) + [""] * (len(headers) - len(row))
        rows_md.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return "\n".join(rows_md)


def _worker_host(worker_id: int = 0) -> str:
    if OLLAMA_HOSTS:
        return OLLAMA_HOSTS[worker_id % len(OLLAMA_HOSTS)]
    return OLLAMA_HOST


def classify_table(table: dict, headers: list[str], data_rows: list[list[str]],
                   worker_id: int = 0) -> dict | None:
    """
    Pass 1: LLM classifies the table structure and maps column headers to canonical attributes.
    Returns a schema dict or None on failure.
    """
    sample = _sample_markdown(headers, data_rows, max_rows=3)
    prompt = COLUMN_MAP_PROMPT.format(
        source_document=table["source_doc"],
        section_header=table["section_header"],
        sample_markdown=sample,
        attribute_block=ATTRIBUTE_PROMPT_BLOCK,
    )

    try:
        client = ollama.Client(host=_worker_host(worker_id))
        response = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0, "num_ctx": 4096},
        )
        raw_text = response.get("response", "").strip()
    except Exception as exc:
        log.warning("    Ollama error for table %d in %s: %s",
                    table["table_index"], table["source_doc"], exc)
        return None

    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    json_str = _extract_first_json(raw_text)
    if not json_str:
        log.warning("    No JSON in LLM response for table %d of %s",
                    table["table_index"], table["source_doc"])
        return None

    try:
        schema = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.warning("    JSON parse error for table %d: %s", table["table_index"], exc)
        return None

    if "column_map" not in schema or "format" not in schema:
        log.warning("    Missing required keys in schema for table %d", table["table_index"])
        return None

    # Normalize table_type
    raw_type = schema.get("table_type", "other").lower()
    type_map = {"specification": "specifications", "specs": "specifications"}
    schema["table_type"] = type_map.get(raw_type, raw_type)
    if schema["table_type"] not in {"capacity", "cost", "specifications", "timeline", "other"}:
        schema["table_type"] = "other"

    return schema


# ── Pass 2: Programmatic Row Extraction ────────────────────────────────────────

def _col_index(headers: list[str], header_name: str | None) -> int | None:
    if not header_name:
        return None
    # Direct integer (LLM may provide index directly)
    if isinstance(header_name, int):
        return header_name if 0 <= header_name < len(headers) else None
    try:
        return int(header_name)  # "0", "1", etc.
    except (ValueError, TypeError):
        pass
    # Exact match
    try:
        return headers.index(header_name)
    except ValueError:
        pass
    # Case-insensitive exact
    hn = header_name.lower()
    for i, h in enumerate(headers):
        if h.lower() == hn:
            return i
    # Normalized: strip everything except alphanumeric
    def _norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    hn_norm = _norm(header_name)
    if hn_norm:
        for i, h in enumerate(headers):
            if _norm(h) == hn_norm:
                return i
    # Substring (last resort)
    for i, h in enumerate(headers):
        if hn in h.lower() or h.lower() in hn:
            return i
    return None


def _to_float(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    # Take first number if range like "24-32" or "27.7-28.7"
    m = re.search(r"[\d.]+", s.replace(",", ""))
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


def _line_item_type(row_label: str | None, line_item_rules: dict) -> str | None:
    if not row_label or not line_item_rules:
        return None
    rl = row_label.strip().lower()
    for pattern, lit_type in line_item_rules.items():
        if pattern.lower() in rl or rl.startswith(pattern.lower()):
            return lit_type
    # Heuristics
    if re.match(r"total", rl):
        return "total"
    if re.match(r"sub.?total", rl):
        return "subtotal"
    return None


def build_eav_rows(
    schema: dict,
    headers: list[str],
    data_rows: list[list[str]],
    table: dict,
) -> list[dict]:
    """
    Pass 2: convert all data rows to EAV records using LLM-provided schema.
    Pure Python — no LLM calls, no truncation.
    """
    fmt = schema.get("format", "wide")
    column_map: dict[str, str] = schema.get("column_map", {})
    line_item_rules: dict[str, str] = schema.get("line_item_rules", {})
    table_entity = schema.get("table_entity")
    entity_type = schema.get("entity_type", "other")
    table_type = schema.get("table_type", "other")
    year = schema.get("year")
    currency_year = schema.get("currency_year")
    description = schema.get("description", "")
    confidence = schema.get("confidence", "low")
    notes = schema.get("notes", "")
    extracted_at = datetime.utcnow().isoformat()

    records = []

    def base_meta():
        return {
            "_source_doc":    table["source_doc"],
            "_section_header":table["section_header"],
            "_table_index":   table["table_index"],
            "_table_type":    table_type,
            "_description":   description,
            "_year":          _to_int(year),
            "_currency_year": _to_int(currency_year),
            "_table_entity":  table_entity,
            "_confidence":    confidence,
            "_notes":         notes,
            "_extracted_at":  extracted_at,
        }

    if fmt == "wide":
        # Each row is a different entity; entity_column holds the entity name
        entity_col_idx = _col_index(headers, schema.get("entity_column"))
        # Fall back to explicit index if name lookup failed
        if entity_col_idx is None and schema.get("entity_column_index") is not None:
            entity_col_idx = int(schema["entity_column_index"])
        # Last resort: assume first column is the entity identifier
        if entity_col_idx is None and headers:
            entity_col_idx = 0

        for row in data_rows:
            padded = list(row) + [""] * (len(headers) - len(row))
            entity = padded[entity_col_idx].strip() if entity_col_idx is not None else table_entity
            if not entity:
                entity = table_entity

            for col_header, canonical_attr in column_map.items():
                col_idx = _col_index(headers, col_header)
                if col_idx is None or col_idx == entity_col_idx:
                    continue
                val_raw = padded[col_idx].strip() if col_idx < len(padded) else ""
                if not val_raw:
                    continue

                rec = {
                    "entity":         entity,
                    "entity_type":    entity_type,
                    "row_label":      None,
                    "attribute":      canonical_attr,
                    "value_raw":      val_raw,
                    "value_numeric":  _to_float(val_raw),
                    "unit":           None,
                    "line_item_type": None,
                }
                rec.update(base_meta())
                records.append(rec)

    elif fmt == "tall":
        # Each row is an attribute of one entity; first col = attribute label, second = value
        value_col_idx = _col_index(headers, schema.get("value_column"))
        if value_col_idx is None and len(headers) >= 2:
            value_col_idx = 1
        label_col_idx = 0 if value_col_idx != 0 else 1

        for row in data_rows:
            padded = list(row) + [""] * (len(headers) - len(row))
            row_label = padded[label_col_idx].strip() if label_col_idx < len(padded) else ""
            val_raw = padded[value_col_idx].strip() if value_col_idx is not None and value_col_idx < len(padded) else ""
            if not row_label and not val_raw:
                continue

            # Look up canonical attribute for this row label
            canonical_attr = column_map.get(row_label) or column_map.get(row_label.lower())
            if not canonical_attr:
                # Fuzzy: try to find a match
                for k, v in column_map.items():
                    if k.lower() in row_label.lower() or row_label.lower() in k.lower():
                        canonical_attr = v
                        break
            if not canonical_attr:
                canonical_attr = re.sub(r"[^a-z0-9]+", "_", row_label.lower()).strip("_") or "unknown"

            rec = {
                "entity":         table_entity,
                "entity_type":    entity_type,
                "row_label":      row_label,
                "attribute":      canonical_attr,
                "value_raw":      val_raw,
                "value_numeric":  _to_float(val_raw),
                "unit":           None,
                "line_item_type": None,
            }
            rec.update(base_meta())
            records.append(rec)

    elif fmt == "cost":
        # Rows are cost line items; label_col + one or more value columns
        label_col = schema.get("entity_column") or (headers[0] if headers else None)
        label_col_idx = _col_index(headers, label_col) or 0

        # Value columns are all mapped entries
        value_cols = {h: a for h, a in column_map.items() if h != label_col}

        for row in data_rows:
            padded = list(row) + [""] * (len(headers) - len(row))
            row_label = padded[label_col_idx].strip() if label_col_idx < len(padded) else ""
            lit = _line_item_type(row_label, line_item_rules)

            if value_cols:
                for col_header, canonical_attr in value_cols.items():
                    col_idx = _col_index(headers, col_header)
                    if col_idx is None:
                        continue
                    val_raw = padded[col_idx].strip() if col_idx < len(padded) else ""
                    if not val_raw:
                        continue
                    rec = {
                        "entity":         table_entity,
                        "entity_type":    entity_type,
                        "row_label":      row_label,
                        "attribute":      canonical_attr,
                        "value_raw":      val_raw,
                        "value_numeric":  _to_float(val_raw),
                        "unit":           None,
                        "line_item_type": lit,
                    }
                    rec.update(base_meta())
                    records.append(rec)
            else:
                # Single-value cost table: map row label to cost_line_item
                val_raw = " | ".join(c for c in padded[1:] if c.strip())
                rec = {
                    "entity":         table_entity,
                    "entity_type":    entity_type,
                    "row_label":      row_label,
                    "attribute":      "cost_line_item",
                    "value_raw":      val_raw,
                    "value_numeric":  _to_float(padded[1].strip() if len(padded) > 1 else ""),
                    "unit":           None,
                    "line_item_type": lit,
                }
                rec.update(base_meta())
                records.append(rec)

    return records


def _to_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ── Parquet Storage ────────────────────────────────────────────────────────────
def save_to_parquet(records: list[dict], table: dict) -> Path | None:
    if not records:
        log.info("    Table %d: no EAV records to store", table["table_index"])
        return None

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    source_stem = Path(table["source_doc"]).stem
    out_path = TABLES_DIR / f"{source_stem}_t{table['table_index']:03d}.parquet"

    df = pd.DataFrame(records, columns=EAV_COLUMNS)
    df.to_parquet(out_path, index=False)
    log.info(
        "    Saved %d EAV rows → %s",
        len(df), out_path.name,
    )
    return out_path


# ── Per-file Processing ────────────────────────────────────────────────────────
def process_file(md_path: Path, dry_run: bool, worker_id: int = 0) -> dict:
    source_doc = md_path.name
    log.info("Processing: %s", source_doc)

    content = md_path.read_text(encoding="utf-8", errors="replace")
    raw_tables = extract_tables(content, source_doc)
    log.info("  Found %d table block(s)", len(raw_tables))

    stats = {"tables_found": len(raw_tables), "normalized": 0, "skipped": 0, "failed": 0}

    for table in raw_tables:
        headers, data_rows, garbled_count = parse_table_rows(table["raw"])
        log.info(
            "  Table %d: %d cols, %d data rows, %d garbled skipped",
            table["table_index"], len(headers), len(data_rows), garbled_count,
        )

        if len(data_rows) < MIN_DATA_ROWS:
            log.info("  Table %d: insufficient data rows, skipping", table["table_index"])
            stats["skipped"] += 1
            continue

        if dry_run:
            stats["normalized"] += 1
            continue

        # Pass 1: LLM classifies structure and maps column headers
        schema = classify_table(table, headers, data_rows, worker_id=worker_id)
        if schema is None:
            stats["failed"] += 1
            continue

        log.info(
            "  Table %d: format=%s, entity_type=%s, %d col mappings, confidence=%s",
            table["table_index"],
            schema.get("format"),
            schema.get("entity_type"),
            len(schema.get("column_map", {})),
            schema.get("confidence"),
        )

        # Pass 2: Python extracts ALL rows using the schema
        try:
            records = build_eav_rows(schema, headers, data_rows, table)
            out = save_to_parquet(records, table)
        except Exception as exc:
            log.warning(
                "    Failed to process table %d of %s: %s",
                table["table_index"], source_doc, exc,
            )
            stats["failed"] += 1
            continue

        if out:
            stats["normalized"] += 1
        else:
            stats["skipped"] += 1

    return stats


# ── Entry Point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract JAI archive tables to EAV Parquet format."
    )
    parser.add_argument("--file", help="Process only this file (relative to markdown/)")
    parser.add_argument("--reprocess", action="store_true", help="Ignore manifest and reprocess all")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, skip LLM")
    parser.add_argument("--workers", type=int, default=2, help="Parallel Ollama workers (default: 2)")
    parser.add_argument("--hosts", help="Comma-separated Ollama hosts for dual-GPU, e.g. http://localhost:11434,http://localhost:11435")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    if args.file:
        md_files = [MARKDOWN_DIR / args.file]
    else:
        md_files = sorted(MARKDOWN_DIR.glob("*.md"))

    if not md_files:
        log.error("No markdown files found in %s", MARKDOWN_DIR)
        sys.exit(1)

    to_process = []
    skipped_cached = 0
    for md_path in md_files:
        if not md_path.exists():
            log.error("File not found: %s", md_path)
            continue
        fp = file_fingerprint(md_path)
        if not args.reprocess and manifest.get(str(md_path)) == fp:
            log.info("Skipping (unchanged): %s", md_path.name)
            skipped_cached += 1
        else:
            to_process.append((md_path, fp))

    totals = {
        "processed": 0, "skipped_cached": skipped_cached,
        "tables_found": 0, "normalized": 0, "failed": 0,
    }

    workers = max(1, args.workers)

    # Dual-GPU: if --hosts provided, route each worker to its own Ollama instance
    if args.hosts:
        OLLAMA_HOSTS.clear()
        OLLAMA_HOSTS.extend(h.strip() for h in args.hosts.split(","))
        log.info("Dual-GPU mode: %s", OLLAMA_HOSTS)

    log.info(
        "Processing %d file(s) with %d worker(s) using %s",
        len(to_process), workers, OLLAMA_MODEL,
    )

    def _run(item_and_wid):
        item, wid = item_and_wid
        md_path, fp = item
        stats = process_file(md_path, dry_run=args.dry_run, worker_id=wid)
        return md_path, fp, stats

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run, (item, i % workers)): item
            for i, item in enumerate(to_process)
        }
        for future in as_completed(futures):
            try:
                md_path, fp, stats = future.result()
            except Exception as exc:
                log.error("Worker exception: %s", exc)
                continue
            totals["processed"] += 1
            totals["tables_found"] += stats["tables_found"]
            totals["normalized"] += stats["normalized"]
            totals["failed"] += stats["failed"]
            if not args.dry_run:
                manifest[str(md_path)] = fp
                save_manifest(manifest)

    log.info(
        "\nDone. Files: %d processed, %d skipped (cached)"
        " | Tables: %d found, %d normalized, %d failed",
        totals["processed"], totals["skipped_cached"],
        totals["tables_found"], totals["normalized"], totals["failed"],
    )


if __name__ == "__main__":
    main()
