#!/usr/bin/env python3
"""
07_query.py — Natural language query interface for JAI archive.

Uses hybrid RAG: DuckDB (structured EAV facts) + ChromaDB (semantic chunks).

Usage:
    python 07_query.py "question"
    python 07_query.py --interactive
    python 07_query.py --verbose "question"
    python 07_query.py --doc ~/jai-archive/markdown/JAI-490.md
    python 07_query.py --doc ~/jai-archive/ocr/JAI-490.pdf --pages 89-95
    python 07_query.py --doc ... --doc2 ...
    python 07_query.py --deep
    python 07_query.py --model llama3.1:8b "question"
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

LLM_MODEL = "llama3.1:8b"        # routing + synthesis (overridable via --model)
SQL_MODEL = "qwen2.5-coder:7b"  # SQL generation
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TOP_K = 10


# ── Model helpers ──────────────────────────────────────────────────────────────
def _llm(prompt: str, model: str = None, num_ctx: int = 4096) -> str:
    client = ollama.Client(host=OLLAMA_HOST)
    r = client.generate(
        model=model or LLM_MODEL,
        prompt=prompt,
        options={"temperature": 0, "num_ctx": num_ctx},
    )
    return r.get("response", "").strip()


# ── Document Injection ─────────────────────────────────────────────────────────
def parse_page_range(pages_str: str) -> list:
    """Parse "89-95", "89,90,91", or "89" into a list of page numbers."""
    pages = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return pages


def load_document_context(doc_path: str, pages: str = None, max_words: int = 4000) -> str:
    """
    Load document content for injection into query context.

    Supports markdown (.md) and PDF (.pdf via pymupdf).
    pages: range string like "89-95" or "89,90,91" (PDF only).
    Truncates to max_words to avoid context overflow.
    """
    path = Path(doc_path).expanduser()

    if not path.exists():
        return f"[Document not found: {doc_path}]"

    if path.suffix == ".md":
        text = path.read_text()

    elif path.suffix == ".pdf":
        try:
            import fitz  # pymupdf
        except ImportError:
            return "[pymupdf not installed — cannot read PDF directly; run: pip install pymupdf]"
        doc = fitz.open(str(path))
        if pages:
            page_nums = parse_page_range(pages)
            text = ""
            for page_num in page_nums:
                if 0 < page_num <= len(doc):
                    text += f"\n--- Page {page_num} ---\n"
                    text += doc[page_num - 1].get_text()
        else:
            text = "\n".join(page.get_text() for page in doc)

    else:
        return f"[Unsupported file type: {path.suffix}]"

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
        text += f"\n[... truncated at {max_words} words ...]"

    return f"\n=== Injected Document: {path.name} ===\n{text}\n=== End Document ===\n"


# ── Routing ────────────────────────────────────────────────────────────────────
_SEMANTIC_KEYWORDS = [
    "tell me about", "what is", "what are", "describe", "explain",
    "how does", "how do", "how are", "how is", "why ", "background on",
    "overview of", "history of", "summarize", "summary of", "summary on",
    "give me a summary", "summary",
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

# Matches JAI document IDs like "JAI-185", "JAI-N006", "JAI-N006a"
_DOC_ID_RE = re.compile(r'\b(JAI-[A-Za-z]*\d+[a-z]?)\b', re.IGNORECASE)


def _doc_filter(question: str) -> dict | None:
    m = _MODEL_RE.search(question)
    if m:
        return {"$contains": m.group().upper()}
    q = question.lower()
    for term, vendor in _VENDOR_TERMS.items():
        if term in q:
            return {"$contains": vendor}
    return None


def _source_where_filter(question: str) -> dict | None:
    """
    If the query names a JAI document, return a ChromaDB `where` metadata filter.
    - "JAI-N006"  → doc_family = "JAI-N006"  (matches all JAI-N006a/b/c/...)
    - "JAI-N006a" → doc_id    = "JAI-N006a"  (exact file only)
    Normalizes case to match stored metadata (uppercase base, lowercase suffix).
    """
    m = _DOC_ID_RE.search(question)
    if not m:
        return None
    raw = m.group(1)
    suffix_m = re.search(r'([a-z]+)$', raw)
    suffix = suffix_m.group(1) if suffix_m else ""
    base = raw[:len(raw) - len(suffix)].upper()
    if suffix:
        return {"doc_id": {"$eq": base + suffix}}
    return {"doc_family": {"$eq": base}}


def _find_matching_docs(prefix: str) -> list[Path]:
    """Return sorted markdown files whose names start with the given JAI doc ID prefix."""
    md_dir = BASE_DIR / "markdown"
    return sorted(p for p in md_dir.glob("*.md") if p.name.upper().startswith(prefix.upper()))


EMBED_MODEL = "nomic-embed-text"


def _embed(text: str) -> list[float] | None:
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        r = client.embeddings(model=EMBED_MODEL, prompt=text)
        return r.get("embedding")
    except Exception:
        return None


def semantic_search(question: str, force_filter: dict | None = None,
                    verbose: bool = False) -> list[dict]:
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        col = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        return []

    embedding = _embed(question)
    if embedding is None:
        return []

    kwargs = {"query_embeddings": [embedding], "n_results": TOP_K}

    # Doc ID filter (metadata) takes precedence over content filter.
    # Use more results when scoped to a specific document family.
    source_filter = _source_where_filter(question)
    if source_filter:
        kwargs["where"] = source_filter
        kwargs["n_results"] = TOP_K * 3
        if verbose:
            print(f"[semantic] doc-id filter → {source_filter}")
    else:
        doc_filter = force_filter or _doc_filter(question)
        if doc_filter:
            kwargs["where_document"] = doc_filter

    try:
        results = col.query(**kwargs)
    except Exception:
        # Retry without filter (e.g. no docs match the filter)
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
{injected}
Answer:"""


def synthesize(question: str, structured: list[dict], semantic: list[dict],
               injected_context: str = "") -> str:
    structured_text = json.dumps(structured, indent=2) if structured else "(none)"
    semantic_text = "\n---\n".join(
        f"[{s['source']}]\n{s['text']}" for s in semantic
    ) if semantic else "(none)"

    injected_section = (
        f"\nInjected documents:\n{injected_context}" if injected_context else ""
    )

    prompt = SYNTHESIS_PROMPT.format(
        question=question,
        structured=structured_text[:6000],
        semantic=semantic_text[:8000],
        injected=injected_section,
    )
    num_ctx = 16384 if injected_context else 12288
    return _llm(prompt, num_ctx=num_ctx)


# ── Main Query Pipeline ────────────────────────────────────────────────────────
def ask(question: str, verbose: bool = False,
        extra_doc: str = None, extra_doc2: str = None,
        pages: str = None) -> dict:
    """
    Run the full RAG pipeline and return a result dict.

    Keys: answer, sources, chunks_retrieved, backend, injected_docs
    """
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
        semantic_chunks = semantic_search(question, verbose=verbose)
        if verbose:
            print(f"[semantic] {len(semantic_chunks)} chunk(s)")

    if con:
        con.close()

    # Load injected documents
    injected_context = ""
    injected_docs = []
    if extra_doc:
        injected_context += load_document_context(extra_doc, pages=pages)
        injected_docs.append(Path(extra_doc).expanduser().name)
    if extra_doc2:
        injected_context += load_document_context(extra_doc2)
        injected_docs.append(Path(extra_doc2).expanduser().name)

    # Auto-inject matching markdown files when query names a specific JAI document
    # and the user didn't already provide --doc. Scales word budget across files.
    if not extra_doc:
        doc_id_m = _DOC_ID_RE.search(question)
        if doc_id_m:
            prefix = doc_id_m.group(1)
            matched = _find_matching_docs(prefix)
            if matched:
                per_file_words = max(500, 4000 // len(matched))
                for doc_path in matched:
                    injected_context += load_document_context(str(doc_path),
                                                              max_words=per_file_words)
                    injected_docs.append(doc_path.name)
                if verbose:
                    print(f"[auto-inject] {len(matched)} file(s): {[p.name for p in matched]}")

    sources = list(dict.fromkeys(c["source"] for c in semantic_chunks if c["source"]))

    if not structured_rows and not semantic_chunks and not injected_context:
        return {
            "answer": "No relevant data found in the archive for that question.",
            "sources": sources,
            "chunks_retrieved": 0,
            "backend": decision,
            "injected_docs": injected_docs,
        }

    backend = decision + ("+injected" if injected_docs else "")
    answer = synthesize(question, structured_rows, semantic_chunks, injected_context)

    return {
        "answer": answer,
        "sources": sources,
        "chunks_retrieved": len(semantic_chunks),
        "backend": backend,
        "injected_docs": injected_docs,
    }


# ── Two-Stage Deep Dive ────────────────────────────────────────────────────────
DEEP_PROMPT = """You are an expert nuclear spent fuel management consultant.

Initial analysis from archive search:
{stage1_answer}

Full document for detailed analysis:
{doc_context}

Question (provide a comprehensive detailed answer drawing on both sources):
{question}

Requirements:
- Be specific with numbers, dates, and technical details
- Note which source (archive search vs document) supports each point
- Use domain-appropriate technical terminology
- If the document contains tables with relevant data, extract and present the key figures

Answer:"""


def two_stage_query(question: str, verbose: bool = False) -> dict:
    """
    Stage 1: standard RAG query to find relevant documents.
    Stage 2: deep dive with full top-document content injected.

    Returns: stage1_answer, stage2_answer, sources, top_document
    """
    print("Stage 1: Searching archive...")
    stage1 = ask(question, verbose=verbose)

    if not stage1["sources"]:
        return {
            "stage1_answer": stage1["answer"],
            "stage2_answer": None,
            "sources": [],
            "top_document": None,
        }

    top_source = stage1["sources"][0]
    md_path = BASE_DIR / "markdown" / top_source

    if not md_path.exists():
        return {
            "stage1_answer": stage1["answer"],
            "stage2_answer": "Could not locate source document for deep dive.",
            "sources": stage1["sources"],
            "top_document": str(md_path),
        }

    print(f"Stage 2: Deep dive into {top_source}...")
    doc_context = load_document_context(str(md_path), max_words=5000)

    prompt = DEEP_PROMPT.format(
        stage1_answer=stage1["answer"],
        doc_context=doc_context,
        question=question,
    )
    stage2_answer = _llm(prompt, num_ctx=8192)

    return {
        "stage1_answer": stage1["answer"],
        "stage2_answer": stage2_answer,
        "sources": stage1["sources"],
        "top_document": str(md_path),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
def _print_result(result: dict, deep: bool = False):
    if deep:
        print(f"\nStage 1 Answer:\n{result['stage1_answer']}")
        if result["stage2_answer"]:
            print(f"\nStage 2 Deep Dive:\n{result['stage2_answer']}")
        if result["top_document"]:
            print(f"\nTop document: {result['top_document']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
    else:
        print(f"\n{result['answer']}")
        src = ", ".join(result["sources"]) if result["sources"] else "(none)"
        print(f"\nSources: {src}")
        print(f"Backend: {result['backend']}  |  Chunks: {result['chunks_retrieved']}")
        if result["injected_docs"]:
            print(f"Injected: {', '.join(result['injected_docs'])}")


def main():
    global LLM_MODEL

    parser = argparse.ArgumentParser(description="Query the JAI archive.")
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Force interactive mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show routing, SQL, and chunk counts")
    parser.add_argument("--doc", type=str,
                        help="Document to inject as additional context (.md or .pdf)")
    parser.add_argument("--doc2", type=str,
                        help="Second document to inject")
    parser.add_argument("--pages", type=str,
                        help='Page range for PDF injection (e.g. "89-95" or "89,90,91")')
    parser.add_argument("--deep", action="store_true",
                        help="Two-stage deep dive: ChromaDB retrieval then full document analysis")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model for routing/synthesis (default: llama3.1:8b)")
    args = parser.parse_args()

    if args.model:
        LLM_MODEL = args.model

    if not DUCKDB_PATH.exists():
        print(f"Warning: DuckDB not found at {DUCKDB_PATH} — structured queries disabled")

    def _run(q: str):
        if args.deep:
            result = two_stage_query(q, verbose=args.verbose)
        else:
            result = ask(q, verbose=args.verbose,
                         extra_doc=args.doc, extra_doc2=args.doc2, pages=args.pages)
        _print_result(result, deep=args.deep)

    # Single-shot mode when a question is given on the command line
    if args.question and not args.interactive:
        _run(args.question)
        return

    # Interactive mode
    mode = "Deep Dive" if args.deep else "Standard"
    print(f"JAI Archive Query System  [{mode}]  (type 'quit' to exit)")
    if args.model:
        print(f"Model: {LLM_MODEL}")
    if args.doc:
        label = f"{args.doc}" + (f"  pages {args.pages}" if args.pages else "")
        print(f"Injected: {label}")
    if args.doc2:
        print(f"Injected: {args.doc2}")
    print()

    while True:
        try:
            q = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        _run(q)
        print()


if __name__ == "__main__":
    main()
