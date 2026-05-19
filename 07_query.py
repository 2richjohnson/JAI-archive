#!/usr/bin/env python3
"""
07_query.py — Unified natural language query interface for the JAI archive.

Routes questions to ChromaDB (semantic), DuckDB (structured), or both.
Replaces 04_query.py.

Usage:
    python 07_query.py "your question"
    python 07_query.py --interactive
    python 07_query.py --verbose "Which countries have dry storage over 5,000 MTU?"
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

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path.home() / "jai-archive"
CHROMA_DIR = BASE_DIR / "db"
DUCKDB_PATH = BASE_DIR / "duckdb" / "jai.db"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "jai_archive"
TOP_K = 5

# ── Prompts ────────────────────────────────────────────────────────────────────
ROUTER_PROMPT = """\
You are a query router for a nuclear archive database. Classify the question.

Return ONLY a JSON object:
{{"route": "semantic|structured|hybrid", "reason": "one sentence"}}

semantic  — asks for narrative explanation, recommendations, context, background, opinions
structured — asks for specific numbers, comparisons, rankings, totals, or filtered table data
hybrid    — needs both narrative context and specific numbers

Question: {question}
"""

SQL_GEN_PROMPT = """\
You are a SQL expert for a nuclear consulting archive database.
Write a DuckDB SQL query that answers the question.

PREFER the specialized views below over tables_all for capacity/storage questions:

  capacity_normalized -- best for wet/dry storage capacity questions
    country       TEXT   -- country or entity name
    wet_mtu       DOUBLE -- wet storage capacity in MTU (already numeric, NULL if unknown)
    dry_mtu       DOUBLE -- dry storage capacity in MTU (already numeric, NULL if unknown)
    wet_storage_raw TEXT -- original string value
    dry_storage_raw TEXT -- original string value
    _source_doc, _description, _confidence, _table_index

  tables_all -- raw union of all extracted tables (use when capacity_normalized lacks the data)
    _source_doc TEXT, _table_type TEXT, _entity TEXT, _year INTEGER,
    _description TEXT, _section_header TEXT, _confidence TEXT
    (data columns vary -- always double-quote names with spaces)

{schema_hint}

Rules:
- For storage/capacity questions, query capacity_normalized ONLY — never UNION it with tables_all
- Use ILIKE for case-insensitive text matching
- Country names in capacity_normalized are full names (e.g. "United Kingdom" not "UK")
- For geographic aggregations (e.g. "European total"), return all rows — the answer layer will filter by region
- Always include _source_doc or country in SELECT so results can be cited
- For aggregations (SUM, total), filter out NULL values with WHERE col IS NOT NULL

Question: {question}

Return ONLY the SQL query, no explanation, no markdown fences.\
"""

ANSWER_PROMPT = """\
You are an analyst for a nuclear consulting archive (JAI Corporation).
Answer the question concisely and factually based on the evidence below.

Question: {question}

Evidence:
{evidence}

Instructions:
- Use specific numbers where the data provides them
- Cite source document filenames
- If evidence is incomplete or uncertain, say so explicitly
- End your answer with: Sources: [comma-separated document names]
"""

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


# ── Ollama helpers ─────────────────────────────────────────────────────────────
def _llm(prompt: str, temperature: float = 0.0, num_ctx: int = 2048) -> str:
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"temperature": temperature, "num_ctx": num_ctx},
    )
    return resp.get("response", "").strip()


def _embed(text: str) -> list[float]:
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.embeddings(model=EMBED_MODEL, prompt=text)
    return resp.get("embedding", [])


# ── Query Router ───────────────────────────────────────────────────────────────
def route(question: str) -> tuple[str, str]:
    """Returns (route, reason). Route is 'semantic' | 'structured' | 'hybrid'."""
    raw = _llm(ROUTER_PROMPT.format(question=question))
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    m = _JSON_RE.search(raw)
    if m:
        try:
            parsed = json.loads(m.group())
            r = parsed.get("route", "hybrid").lower()
            if r not in ("semantic", "structured", "hybrid"):
                r = "hybrid"
            return r, parsed.get("reason", "")
        except json.JSONDecodeError:
            pass
    return "hybrid", "(router parse failed, defaulting to hybrid)"


# ── Semantic (ChromaDB) ────────────────────────────────────────────────────────
def semantic_search(question: str) -> list[dict]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embedding = _embed(question)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {"text": doc, "source": meta.get("source", "unknown"), "distance": dist}
        )
    return hits


# ── Structured (DuckDB) ────────────────────────────────────────────────────────
_INTERNAL_COLS = {"filename", "_source_doc", "_section_header", "_table_index",
                  "_table_type", "_description", "_year", "_entity", "_confidence",
                  "_notes", "_extracted_at"}

def _schema_hint(con) -> str:
    """
    Concise schema hint for SQL generation — keeps prompt short enough for llama3.1:8b.
    Focuses on capacity_normalized (most useful view) and available countries.
    """
    _EUROPEAN = {"France", "Germany", "United Kingdom", "Sweden", "Finland",
                 "Switzerland", "Belgium", "Spain", "Netherlands", "Czech Republic",
                 "Slovakia", "Hungary", "Bulgaria", "Romania", "Slovenia"}
    try:
        countries = con.execute(
            "SELECT DISTINCT country FROM capacity_normalized "
            "WHERE country IS NOT NULL ORDER BY country"
        ).fetchall()
        all_c = [r[0] for r in countries]
        eu_c = [c for c in all_c if c in _EUROPEAN]
        non_eu_c = [c for c in all_c if c not in _EUROPEAN]
        parts = [f"All countries in capacity_normalized: {', '.join(all_c)}"]
        if eu_c:
            parts.append(f"European countries in data: {', '.join(eu_c)}")
        if non_eu_c:
            parts.append(f"Non-European: {', '.join(non_eu_c)}")
        parts.append("For raw table queries, double-quote column names with spaces.")
        return "\n".join(parts)
    except Exception:
        pass
    return ""


def structured_query(question: str, verbose: bool = False) -> dict:
    if not DUCKDB_PATH.exists():
        return {
            "error": "DuckDB not found — run 06_setup_duckdb.py first",
            "rows": [],
            "sql": "",
        }

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    hint = _schema_hint(con)

    sql_raw = _llm(
        SQL_GEN_PROMPT.format(question=question, schema_hint=hint),
        num_ctx=4096,
    )
    sql = re.sub(r"^```(?:sql)?\s*", "", sql_raw)
    sql = re.sub(r"\s*```$", "", sql).strip()

    if not sql:
        con.close()
        return {"error": "LLM returned empty SQL", "rows": [], "sql": ""}

    try:
        df = con.execute(sql).fetchdf()
        con.close()
        return {"sql": sql, "rows": df.to_dict(orient="records"), "error": None}
    except Exception as exc:
        con.close()
        return {"error": str(exc), "rows": [], "sql": sql}


# ── Answer Synthesis ───────────────────────────────────────────────────────────
def synthesize(question: str, evidence_parts: list[str]) -> str:
    evidence = "\n\n---\n\n".join(evidence_parts)
    return _llm(
        ANSWER_PROMPT.format(question=question, evidence=evidence),
        temperature=0.1,
        num_ctx=4096,
    )


# ── Main Query Flow ────────────────────────────────────────────────────────────
def ask(question: str, verbose: bool = False) -> None:
    print(f"\nQ: {question}")
    print("─" * 64)

    backend, reason = route(question)
    print(f"Route: {backend.upper()}  — {reason}")

    evidence_parts = []
    sources = set()
    had_error = False

    if backend in ("semantic", "hybrid"):
        print("\n[ChromaDB — semantic search]")
        try:
            hits = semantic_search(question)
            for h in hits:
                evidence_parts.append(
                    f"[SEMANTIC | {h['source']} | distance={h['distance']:.3f}]\n{h['text']}"
                )
                sources.add(h["source"])
            print(f"  {len(hits)} chunk(s) retrieved")
        except Exception as exc:
            print(f"  Error: {exc}")
            had_error = True

    if backend in ("structured", "hybrid"):
        print("\n[DuckDB — structured query]")
        result = structured_query(question, verbose=verbose)
        if verbose:
            print(f"  SQL: {result['sql']}")
        if result["error"]:
            print(f"  Error: {result['error']}")
            had_error = True
        elif result["rows"]:
            rows = result["rows"]
            print(f"  {len(rows)} row(s) returned")
            if verbose:
                for r in rows[:5]:
                    print(f"    {r}")

            # Cap at 20 rows for LLM context; include all columns
            row_lines = []
            for r in rows[:20]:
                parts = [
                    f"{k}={v}"
                    for k, v in r.items()
                    # Always include source and key metadata; skip noisy internal fields
                    if k in ("_source_doc", "_entity", "_year", "_description", "_confidence")
                    or not k.startswith("_")
                ]
                row_lines.append(", ".join(parts))
            evidence_parts.append("[STRUCTURED DATA]\n" + "\n".join(row_lines))
            sources.update(
                str(r.get("_source_doc", "")) for r in rows if r.get("_source_doc")
            )
        else:
            print("  No matching rows")

    if not evidence_parts:
        msg = "No evidence found."
        if had_error:
            msg += " (check errors above)"
        print(f"\n{msg}")
        return

    print("\n[Synthesizing answer…]")
    answer = synthesize(question, evidence_parts)
    print(f"\n{answer}")
    print(f"\n[Backend: {backend.upper()}]")


# ── Entry Point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Query the JAI archive via ChromaDB and/or DuckDB."
    )
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show SQL and extra debug info")
    args = parser.parse_args()

    if args.interactive:
        print("JAI Archive Query  (ctrl-D or 'quit' to exit)")
        print("=" * 64)
        while True:
            try:
                q = input("\nQ: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                ask(q, verbose=args.verbose)
    elif args.question:
        ask(args.question, verbose=args.verbose)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
