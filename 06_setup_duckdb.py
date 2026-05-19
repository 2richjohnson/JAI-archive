#!/usr/bin/env python3
"""
06_setup_duckdb.py — Initialize DuckDB and create views over extracted Parquet files.

Run after 05_extract_tables.py has produced Parquet output.
Creates ~/jai-archive/duckdb/jai.db with:
  - tables_<type>   — one view per table type
  - tables_all      — union across all types (union_by_name handles schema variation)
  - tables_summary  — one row per extracted table (no row-level data)

Usage:
    python 06_setup_duckdb.py [--rebuild]

    --rebuild   Delete and recreate the database from scratch
"""

import argparse
import logging
import sys
from pathlib import Path

import duckdb

BASE_DIR = Path.home() / "jai-archive"
TABLES_DIR = BASE_DIR / "tables" / "parquet"
DUCKDB_DIR = BASE_DIR / "duckdb"
DUCKDB_PATH = DUCKDB_DIR / "jai.db"
TABLE_TYPES = ["capacity", "cost", "specifications", "timeline", "other"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def parquet_glob(table_type: str) -> str:
    return str(TABLES_DIR / table_type / "*.parquet")


def has_parquet(table_type: str) -> bool:
    return bool(list((TABLES_DIR / table_type).glob("*.parquet")))


def setup_database(rebuild: bool = False):
    DUCKDB_DIR.mkdir(parents=True, exist_ok=True)

    if rebuild and DUCKDB_PATH.exists():
        log.info("Removing existing database for rebuild")
        DUCKDB_PATH.unlink()

    con = duckdb.connect(str(DUCKDB_PATH))
    log.info("Connected to %s", DUCKDB_PATH)

    available_types = [t for t in TABLE_TYPES if has_parquet(t)]
    if not available_types:
        log.warning(
            "No Parquet files found under %s — run 05_extract_tables.py first",
            TABLES_DIR,
        )
        con.close()
        return

    # Per-type views
    for t in available_types:
        glob = parquet_glob(t)
        view = f"tables_{t}"
        con.execute(
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
        )
        count = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        log.info("View %-26s → %d rows", repr(view), count)

    # Unified view across all types — single read_parquet with glob so
    # union_by_name reconciles differing schemas without a column-count mismatch.
    all_glob = str(TABLES_DIR / "**" / "*.parquet")
    con.execute(
        f"CREATE OR REPLACE VIEW tables_all AS "
        f"SELECT * FROM read_parquet('{all_glob}', union_by_name=true, filename=true)"
    )
    total = con.execute("SELECT COUNT(*) FROM tables_all").fetchone()[0]
    log.info("View %-26s → %d rows (all types)", repr("tables_all"), total)

    # Summary view: one row per extracted table, no row-level data
    con.execute("""
        CREATE OR REPLACE VIEW tables_summary AS
        SELECT
            _source_doc,
            _table_type,
            _table_index,
            _description,
            _year,
            _entity,
            _confidence,
            _section_header,
            _extracted_at,
            COUNT(*) AS row_count
        FROM tables_all
        GROUP BY ALL
        ORDER BY _source_doc, _table_index
    """)
    summary_count = con.execute("SELECT COUNT(*) FROM tables_summary").fetchone()[0]
    log.info("View %-26s → %d tables", repr("tables_summary"), summary_count)

    # Normalized capacity view — unifies column name variants across documents.
    # Different source documents used different headers for the same concept;
    # COALESCE picks the first non-null value across known aliases.
    con.execute("""
        CREATE OR REPLACE VIEW capacity_normalized AS
        SELECT
            _source_doc,
            _table_index,
            _entity                                              AS country,
            _description,
            _confidence,
            COALESCE(
                "Installed Wet Storage Capacity (MTU)",
                "Wet Storage (MTU)",
                "Wet Storage"
            )                                                    AS wet_storage_raw,
            COALESCE(
                "Installed Dry Storage Capacity (MTU)",
                "Dry Storage (MTU)",
                "Dry Storage"
            )                                                    AS dry_storage_raw,
            TRY_CAST(
                REPLACE(REPLACE(REPLACE(COALESCE(
                    "Installed Wet Storage Capacity (MTU)",
                    "Wet Storage (MTU)",
                    "Wet Storage"
                ), ',', ''), '~', ''), '-', '')
            AS DOUBLE)                                           AS wet_mtu,
            TRY_CAST(
                REPLACE(REPLACE(REPLACE(COALESCE(
                    "Installed Dry Storage Capacity (MTU)",
                    "Dry Storage (MTU)",
                    "Dry Storage"
                ), ',', ''), '~', ''), '-', '')
            AS DOUBLE)                                           AS dry_mtu
        FROM tables_all
        WHERE _entity IS NOT NULL
          AND (
            "Installed Wet Storage Capacity (MTU)" IS NOT NULL OR
            "Wet Storage (MTU)"                    IS NOT NULL OR
            "Wet Storage"                          IS NOT NULL OR
            "Installed Dry Storage Capacity (MTU)" IS NOT NULL OR
            "Dry Storage (MTU)"                    IS NOT NULL OR
            "Dry Storage"                          IS NOT NULL
          )
    """)
    cap_count = con.execute("SELECT COUNT(*) FROM capacity_normalized").fetchone()[0]
    log.info("View %-26s → %d rows", repr("capacity_normalized"), cap_count)

    con.close()
    log.info("Database ready: %s", DUCKDB_PATH)


def print_info():
    if not DUCKDB_PATH.exists():
        return
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    print("\n── Contents ──────────────────────────────────────")
    try:
        rows = con.execute(
            "SELECT _table_type, COUNT(*) AS rows, COUNT(DISTINCT _source_doc) AS docs "
            "FROM tables_all GROUP BY _table_type ORDER BY rows DESC"
        ).fetchall()
        for r in rows:
            print(f"  {r[0]:<20} {r[1]:>6} rows  ({r[2]} source doc(s))")
        print()

        docs = con.execute(
            "SELECT DISTINCT _source_doc FROM tables_all ORDER BY _source_doc"
        ).fetchall()
        print("── Source documents ──────────────────────────────")
        for (d,) in docs:
            print(f"  {d}")
    except Exception as exc:
        print(f"  (could not read summary: {exc})")
    print()
    con.close()


def main():
    parser = argparse.ArgumentParser(
        description="Initialize DuckDB for JAI archive structured tables."
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Drop and recreate the database"
    )
    args = parser.parse_args()

    setup_database(rebuild=args.rebuild)
    print_info()


if __name__ == "__main__":
    main()
