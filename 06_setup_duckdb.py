#!/usr/bin/env python3
"""
06_setup_duckdb.py — Build DuckDB from EAV Parquet files produced by 05_extract_tables.py.

Creates ~/jai-archive/duckdb/jai.db with:
  - facts            — all EAV rows, single unified schema
  - facts_capacity   — capacity-type rows
  - facts_specs      — specifications-type rows
  - facts_cost       — cost-type rows
  - capacity_summary — one row per (country, attribute) with max value_numeric
  - cask_summary     — one row per (cask_model, attribute) with value

Usage:
    python 06_setup_duckdb.py [--rebuild]
"""

import argparse
import logging
import sys
from pathlib import Path

import duckdb

BASE_DIR = Path.home() / "jai-archive"
TABLES_DIR = BASE_DIR / "tables" / "eav"
DUCKDB_DIR = BASE_DIR / "duckdb"
DUCKDB_PATH = DUCKDB_DIR / "jai.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def setup_database(rebuild: bool = False):
    DUCKDB_DIR.mkdir(parents=True, exist_ok=True)

    if rebuild and DUCKDB_PATH.exists():
        log.info("Removing existing database for rebuild")
        DUCKDB_PATH.unlink()

    con = duckdb.connect(str(DUCKDB_PATH))
    log.info("Connected to %s", DUCKDB_PATH)

    parquet_files = list(TABLES_DIR.glob("*.parquet"))
    if not parquet_files:
        log.warning("No EAV Parquet files found under %s — run 05_extract_tables.py first", TABLES_DIR)
        con.close()
        return

    eav_glob = str(TABLES_DIR / "*.parquet")

    # Main facts table — EAV, single schema.
    # Explicit CASTs guard against parquet files where a column is all-NULL
    # (DuckDB infers NULL type, which then fails in aggregations like MAX()).
    con.execute(f"""
        CREATE OR REPLACE VIEW facts AS
        SELECT
            -- Decode HTML numeric entities left by LLM extraction (&#124; → |)
            REGEXP_REPLACE(
                REGEXP_REPLACE(CAST(entity AS VARCHAR), '&#124;', '|', 'g'),
                '&#38;', '&', 'g')         AS entity,
            CAST(entity_type     AS VARCHAR) AS entity_type,
            CAST(row_label       AS VARCHAR) AS row_label,
            CAST(attribute       AS VARCHAR) AS attribute,
            CAST(value_raw       AS VARCHAR) AS value_raw,
            TRY_CAST(value_numeric AS DOUBLE)  AS value_numeric,
            CAST(unit            AS VARCHAR) AS unit,
            CAST(line_item_type  AS VARCHAR) AS line_item_type,
            CAST(_source_doc     AS VARCHAR) AS _source_doc,
            CAST(_table_type     AS VARCHAR) AS _table_type,
            TRY_CAST(_year         AS INTEGER) AS _year,
            TRY_CAST(_currency_year AS INTEGER) AS _currency_year,
            CAST(_confidence     AS VARCHAR) AS _confidence
        FROM read_parquet('{eav_glob}', union_by_name=true)
    """)
    total = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    log.info("View %-22s → %d rows", repr("facts"), total)

    # Per-type filtered views
    for table_type in ("capacity", "specifications", "cost", "timeline", "other"):
        view = f"facts_{table_type}"
        con.execute(f"""
            CREATE OR REPLACE VIEW {view} AS
            SELECT * FROM facts WHERE _table_type = '{table_type}'
        """)
        count = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        log.info("View %-22s → %d rows", repr(view), count)

    # capacity_summary: best numeric value per (country, attribute)
    # Useful for "which countries have wet storage > X MTU" queries
    con.execute("""
        CREATE OR REPLACE VIEW capacity_summary AS
        SELECT
            entity                           AS country,
            attribute,
            MAX(value_numeric)               AS value_numeric,
            -- Fill NULL units for well-known capacity attributes
            COALESCE(ANY_VALUE(unit), CASE attribute
                WHEN 'wet_storage_mtu'           THEN 'MTU'
                WHEN 'dry_storage_mtu'           THEN 'MTU'
                WHEN 'total_storage_mtu'         THEN 'MTU'
                WHEN 'wet_storage_pwr_mtu'       THEN 'MTU'
                WHEN 'wet_storage_candu_mtu'     THEN 'MTU'
                WHEN 'current_inventory_mtu'     THEN 'MTU'
                WHEN 'stored_fuel_mtu'           THEN 'MTU'
                WHEN 'design_capacity_mtu'       THEN 'MTU'
                WHEN 'current_inventory_mthm'    THEN 'MTHM'
                WHEN 'nominal_pond_capacity_mthm' THEN 'MTHM'
                WHEN 'design_capacity_mthm'      THEN 'MTHM'
                WHEN 'capacity_mthm'             THEN 'MTHM'
                WHEN 'expected_fuel_40yr_mthm'   THEN 'MTHM'
                WHEN 'committed_reprocessing_mthm' THEN 'MTHM'
                WHEN 'direct_disposal_mthm'      THEN 'MTHM'
                WHEN 'year_of_saturation'        THEN 'year'
                WHEN 'percent_occupied'          THEN '%'
                WHEN 'net_capacity_mw'           THEN 'MW'
            END)                             AS unit,
            COUNT(DISTINCT _source_doc)      AS source_docs,
            MAX(_confidence)                 AS best_confidence
        FROM facts
        WHERE entity_type IN ('country', 'facility')
          AND value_numeric IS NOT NULL
          AND attribute IN (
            'wet_storage_mtu', 'dry_storage_mtu', 'total_storage_mtu',
            'wet_storage_pwr_mtu', 'wet_storage_candu_mtu',
            'current_inventory_mtu', 'current_inventory_mthm',
            'nominal_pond_capacity_mthm', 'design_capacity_mthm',
            'design_capacity_mtu', 'net_capacity_mw', 'year_of_saturation',
            'expected_fuel_40yr_mthm', 'committed_reprocessing_mthm',
            'direct_disposal_mthm', 'percent_occupied'
          )
        GROUP BY entity, attribute
        ORDER BY entity, attribute
    """)
    cap_count = con.execute("SELECT COUNT(*) FROM capacity_summary").fetchone()[0]
    log.info("View %-22s → %d rows", repr("capacity_summary"), cap_count)

    # cask_summary: latest value per (cask_model, attribute)
    con.execute("""
        CREATE OR REPLACE VIEW cask_summary AS
        SELECT
            entity                           AS cask_model,
            attribute,
            ANY_VALUE(value_raw)                             AS value_raw,
            MAX(value_numeric)               AS value_numeric,
            ANY_VALUE(unit)                  AS unit,
            ANY_VALUE(_source_doc)           AS source_doc
        FROM facts
        WHERE entity_type = 'cask_model'
          AND entity IS NOT NULL
          AND TRY_CAST(entity AS DOUBLE) IS NULL  -- remove pure numbers ("186", "0.764", "6.")
          AND LENGTH(entity) > 3                   -- remove single-char list items ("c.", "j.")
          AND entity NOT LIKE '%(%'               -- remove column headers ("Max Burnup (GWD/MTU)")
          AND entity NOT LIKE '/%'               -- remove path-like junk ("/G27/G18/...")
          -- remove comma-formatted numbers ("108,267"), quantity+type ("24 PWR"), scientific notation ("1.0 x 10 -7")
          AND NOT regexp_matches(entity, '^\d[\d.]*[, ]')
          -- remove numbered list items ("5. Thermal", "11. Cavity Atmosphere")
          AND NOT regexp_matches(entity, '^\d+\. ')
        GROUP BY entity, attribute
        ORDER BY entity, attribute
    """)
    cask_count = con.execute("SELECT COUNT(*) FROM cask_summary").fetchone()[0]
    log.info("View %-22s → %d rows", repr("cask_summary"), cask_count)

    # cost_summary: line items per cost study
    con.execute("""
        CREATE OR REPLACE VIEW cost_summary AS
        SELECT
            _source_doc,
            _year                            AS study_year,
            _currency_year                   AS currency_year,
            entity,
            row_label,
            attribute,
            value_raw,
            value_numeric,
            unit,
            line_item_type,
            _confidence
        FROM facts
        WHERE _table_type = 'cost'
        ORDER BY _source_doc, entity, row_label
    """)
    cost_count = con.execute("SELECT COUNT(*) FROM cost_summary").fetchone()[0]
    log.info("View %-22s → %d rows", repr("cost_summary"), cost_count)

    con.close()
    log.info("Database ready: %s", DUCKDB_PATH)


def print_info():
    if not DUCKDB_PATH.exists():
        return
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    print("\n── Facts breakdown ──────────────────────────────────")
    try:
        rows = con.execute("""
            SELECT _table_type, entity_type, COUNT(*) AS rows,
                   COUNT(DISTINCT _source_doc) AS docs
            FROM facts
            GROUP BY _table_type, entity_type
            ORDER BY rows DESC
        """).fetchall()
        for r in rows:
            print(f"  {(r[0] or 'NULL'):<16} {(r[1] or 'NULL'):<14} {r[2]:>6} rows  ({r[3]} source doc(s))")

        print()
        print("── Top capacity attributes ──────────────────────────")
        rows = con.execute("""
            SELECT attribute, COUNT(*) AS n, COUNT(DISTINCT country) AS entities
            FROM capacity_summary GROUP BY attribute ORDER BY n DESC LIMIT 10
        """).fetchall()
        for r in rows:
            print(f"  {r[0]:<35} {r[1]:>4} rows  {r[2]:>3} entities")

        print()
        print("── Cask models found ────────────────────────────────")
        rows = con.execute("""
            SELECT cask_model, COUNT(DISTINCT attribute) AS attrs
            FROM cask_summary GROUP BY cask_model ORDER BY cask_model LIMIT 20
        """).fetchall()
        for r in rows:
            print(f"  {r[0]:<30} {r[1]:>3} attributes")
    except Exception as exc:
        print(f"  (could not read summary: {exc})")
    print()
    con.close()


def main():
    parser = argparse.ArgumentParser(description="Build EAV DuckDB for JAI archive.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate the database")
    args = parser.parse_args()
    setup_database(rebuild=args.rebuild)
    print_info()


if __name__ == "__main__":
    main()
