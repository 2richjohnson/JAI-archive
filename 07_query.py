#!/usr/bin/env python3
"""
07_query.py — Natural language query interface for JAI archive.

Uses hybrid RAG: DuckDB (structured EAV facts) + ChromaDB (semantic chunks).

Usage:
    python 07_query.py "question"
    python 07_query.py --interactive
    python 07_query.py --verbose "question"    # show SQL and routing
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import chromadb
import duckdb
import ollama

BASE_DIR = Path.home() / "jai-archive"
DUCKDB_PATH = BASE_DIR / "duckdb" / "jai.db"
CHROMA_PATH = BASE_DIR / "db"
CHROMA_COLLECTION = "jai_archive"

LLM_MODEL = "llama3.1:8b"      # routing + synthesis
SQL_MODEL = "qwen2.5-coder:7b"  # SQL generation only
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TOP_K = 10


# ── Model helpers ──────────────────────────────────────────────────────────────
def _llm(prompt: str, model: str = LLM_MODEL) -> str:
    client = ollama.Client(host=OLLAMA_HOST)
    r = client.generate(model=model, prompt=prompt, options={"temperature": 0, "num_ctx": 4096})
    return r.get("response", "").strip()


# ── Routing ────────────────────────────────────────────────────────────────────
# Keyword shortcuts that bypass the LLM router for reliability
_SEMANTIC_KEYWORDS = [
    "tell me about", "what is", "what are", "describe", "explain",
    "how does", "how do", "how are", "how is", "why ", "background on",
    "overview of", "history of", "summarize", "summary of",
]
_CAPACITY_KEYWORDS = [
    "which countries", "wet storage", "dry storage", "storage capacity",
    "mtu by country", "mtu by", "capacity over", "capacity greater",
    "inventory", "year of saturation", "pond capacity",
]
_SPECS_KEYWORDS = [
    "cask", "canister", "container", "tn-", "castor", "constor", "nuhoms",
    "holtec", "nac-", "magnastor", "transnuclear", "transport cask",
    "storage cask", "specification", "spec ", "assembly", "assemblies",
    "heat load", "burnup", "weight", "length", "diameter", "dimensions",
]

ROUTER_PROMPT = """You are routing a question about nuclear spent fuel storage.
Return exactly one word: STRUCTURED, SEMANTIC, or HYBRID.

STRUCTURED — answerable from numeric/tabular data (capacities, cask specs, costs, counts)
SEMANTIC   — needs document text (policy, history, descriptions, explanations)
HYBRID     — needs both structured data and document context

Question: {question}
Route:"""


def route(question: str, verbose: bool = False) -> str:
    q = question.lower()
    if any(kw in q for kw in _SPECS_KEYWORDS):
        if verbose:
            print(f"[router] keyword → HYBRID (spec/cask query)")
        return "hybrid"
    if any(kw in q for kw in _CAPACITY_KEYWORDS):
        if verbose:
            print(f"[router] keyword → STRUCTURED (capacity query)")
        return "structured"
    if any(kw in q for kw in _SEMANTIC_KEYWORDS):
        if verbose:
            print(f"[router] keyword → SEMANTIC (descriptive query)")
        return "semantic"
    decision = _llm(ROUTER_PROMPT.format(question=question)).split()[0].upper()
    if decision not in ("STRUCTURED", "SEMANTIC", "HYBRID"):
        decision = "HYBRID"
    if verbose:
        print(f"[router] LLM → {decision}")
    return decision.lower()


# ── SQL Generation ─────────────────────────────────────────────────────────────
SQL_SCHEMA = """
EAV facts table — every row is one entity+attribute+value triple:
  facts(
    entity TEXT,           -- country name, cask model, facility name
    entity_type TEXT,      -- 'country' | 'cask_model' | 'facility' | 'cost_study' | ...
    row_label TEXT,        -- original row label (useful for cost line items)
    attribute TEXT,        -- canonical snake_case name (e.g. wet_storage_mtu)
    value_raw TEXT,        -- original string value from document
    value_numeric DOUBLE,  -- numeric value (null for text-only values)
    unit TEXT,             -- MTU | lb | in | kW | M$ | year | % | ...
    line_item_type TEXT,   -- for cost rows: input|calculated|subtotal|total; else null
    _source_doc TEXT,
    _table_type TEXT,      -- capacity|specifications|cost|timeline|other
    _description TEXT,
    _year INT,
    _confidence TEXT
  )

Convenience views:
  capacity_summary(country, attribute, value_numeric, unit, source_docs)
    -- pre-aggregated: MAX value_numeric per (country, attribute)
    -- Attributes: wet_storage_mtu, dry_storage_mtu, total_storage_mtu,
    --   current_inventory_mtu, year_of_saturation, net_capacity_mw, ...

  cask_summary(cask_model, attribute, value_raw, value_numeric, unit, source_doc)
    -- one row per (cask_model, attribute)
    -- Attributes: fuel_assembly_capacity, thermal_heat_rejection_kw,
    --   overall_length_in, weight_empty_lb, weight_loaded_lb,
    --   max_burnup_gwdmtu, cavity_length_in, shape, fuel_type, ...

  cost_summary(source_doc, study_year, currency_year, entity, row_label,
               attribute, value_raw, value_numeric, unit, line_item_type)
"""

SQL_RULES = """
Rules:
- Use capacity_summary for country/facility storage capacity questions. Its key column is 'country' (not 'entity').
- Use cask_summary for cask specifications. Its key column is 'cask_model' (not 'entity').
- cask_model values are specific model names like 'TN-68', 'HI-STORM 100', 'VSC-24' — never vendor names. For vendor queries use ILIKE: e.g. transnuclear → ILIKE 'TN-%', holtec → ILIKE 'HI-%', nuhoms → ILIKE 'NUHOMS%'.
- Use cost_summary for cost study data. Use facts only if a view lacks a needed column.
- EAV structure: each attribute is a SEPARATE ROW — never try to pivot two attributes into one row.
  For multiple attributes, use: WHERE attribute IN ('a','b') and accept multiple rows, or use two subqueries.
- ALWAYS SELECT the name column (country/cask_model) AND attribute AND value_numeric AND unit — never names alone.
- Example for capacity: SELECT country, attribute, value_numeric, unit FROM capacity_summary WHERE country = 'X' AND attribute IN ('year_of_saturation','percent_occupied')
- Example for threshold: SELECT country, value_numeric, unit FROM capacity_summary WHERE attribute = 'wet_storage_mtu' AND value_numeric > 5000 AND country NOT ILIKE '%total%' ORDER BY value_numeric DESC
- Return only valid DuckDB SQL, no explanation, no markdown.
"""

SQL_PROMPT = """{schema}
{rules}
Write a single DuckDB SQL query to answer: {question}

SQL:"""


def generate_sql(question: str, verbose: bool = False) -> str | None:
    prompt = SQL_PROMPT.format(schema=SQL_SCHEMA, rules=SQL_RULES, question=question)
    sql = _llm(prompt, model=SQL_MODEL)
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    sql = sql.strip()
    if verbose:
        print(f"[sql]\n{sql}\n")
    return sql or None


def run_sql(sql: str, con: duckdb.DuckDBPyConnection) -> list[dict]:
    try:
        df = con.execute(sql).fetchdf()
        return df.to_dict(orient="records")
    except Exception as exc:
        return [{"sql_error": str(exc)}]


# ── Semantic Search ────────────────────────────────────────────────────────────
_MODEL_RE = re.compile(
    r"\b(TN-\d+\w*|HI-(?:STAR|STORM)\s*\d*\w*|VSC-\d+\w*|NAC-\w+|"
    r"NUHOMS-\d+\w*|CASTOR\s*\w*|CONSTOR\s*\w*|MAGNASTOR\s*\w*)",
    re.IGNORECASE,
)

_VENDOR_TERMS = {
    "transnuclear": "transnuclear",
    "tn-": "transnuclear",
    "holtec": "holtec",
    "hi-": "holtec",
    "nac": "nac",
    "nuhoms": "nuhoms",
    "framatome": "framatome",
    "gns": "gns",
    "castor": "castor",
}


def _doc_filter(question: str) -> dict | None:
    m = _MODEL_RE.search(question)
    if m:
        return {"$contains": m.group().upper()}
    q = question.lower()
    for term, vendor in _VENDOR_TERMS.items():
        if term in q:
            return {"$contains": vendor}
    return None


EMBED_MODEL = "nomic-embed-text"


def _embed(text: str) -> list[float] | None:
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        r = client.embeddings(model=EMBED_MODEL, prompt=text)
        return r.get("embedding")
    except Exception:
        return None


def semantic_search(question: str, force_filter: dict | None = None) -> list[dict]:
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        col = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        return []

    embedding = _embed(question)
    if embedding is None:
        return []

    kwargs = {"query_embeddings": [embedding], "n_results": TOP_K}
    doc_filter = force_filter or _doc_filter(question)
    if doc_filter:
        kwargs["where_document"] = doc_filter

    try:
        results = col.query(**kwargs)
    except Exception:
        # Retry without filter if it fails (e.g., no matching docs)
        try:
            results = col.query(query_embeddings=[embedding], n_results=TOP_K)
        except Exception:
            return []

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [{"text": d, "source": m.get("source", ""), "page": m.get("page")}
            for d, m in zip(docs, metas)]


# ── Synthesis ──────────────────────────────────────────────────────────────────
SYNTHESIS_PROMPT = """You are an assistant helping analyze nuclear fuel storage documents.
Answer the question using ONLY the provided data. Be specific and cite values where available.
If the data is insufficient, say so — do not guess.

Question: {question}

Structured data:
{structured}

Document excerpts:
{semantic}

Answer:"""


def synthesize(question: str, structured: list[dict], semantic: list[dict]) -> str:
    structured_text = json.dumps(structured, indent=2) if structured else "(none)"
    semantic_text = "\n---\n".join(
        f"[{s['source']}]\n{s['text']}" for s in semantic
    ) if semantic else "(none)"

    prompt = SYNTHESIS_PROMPT.format(
        question=question,
        structured=structured_text[:3000],
        semantic=semantic_text[:3000],
    )
    return _llm(prompt)


# ── Main Query Pipeline ────────────────────────────────────────────────────────
def ask(question: str, verbose: bool = False) -> str:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True) if DUCKDB_PATH.exists() else None

    decision = route(question, verbose=verbose)
    structured_rows = []
    semantic_chunks = []

    if decision in ("structured", "hybrid") and con:
        sql = generate_sql(question, verbose=verbose)
        if sql:
            structured_rows = run_sql(sql, con)
            if verbose:
                print(f"[sql results] {len(structured_rows)} row(s)")

    if decision in ("semantic", "hybrid"):
        semantic_chunks = semantic_search(question)
        if verbose:
            print(f"[semantic] {len(semantic_chunks)} chunk(s)")

    if con:
        con.close()

    if not structured_rows and not semantic_chunks:
        return "No relevant data found in the archive for that question."

    return synthesize(question, structured_rows, semantic_chunks)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Query the JAI archive.")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not DUCKDB_PATH.exists():
        print(f"Warning: DuckDB not found at {DUCKDB_PATH} — structured queries disabled")

    if args.interactive:
        print("JAI Archive Query System  (type 'quit' to exit)\n")
        while True:
            try:
                q = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            print(ask(q, verbose=args.verbose))
            print()
    elif args.question:
        print(ask(args.question, verbose=args.verbose))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
