#!/usr/bin/env python3
"""
03_ingest.py — Index markdown files into ChromaDB with heading-aware chunking.

Each chunk is prefixed with [doc_id], document title, and section heading so
the LLM always has structural context. Metadata includes doc_id, doc_family,
title, and section for reliable filtering.

Usage:
    python 03_ingest.py              # skip already-indexed files
    python 03_ingest.py --rebuild    # wipe collection and re-index everything
    python 03_ingest.py --file X.md  # re-index one file only
"""

import argparse
import re
import sys
from pathlib import Path

import chromadb
import ollama

MARKDOWN_DIR = Path.home() / "jai-archive/markdown"
DB_PATH = str(Path.home() / "jai-archive/db")
COLLECTION_NAME = "jai_archive"
EMBED_MODEL = "nomic-embed-text"
CHUNK_WORDS = 400
OVERLAP_WORDS = 50
MIN_CHUNK_CHARS = 60


def get_embedding(text: str) -> list[float] | None:
    try:
        r = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return r["embedding"]
    except Exception as e:
        print(f"    Embedding error: {e}")
        return None


def extract_title(text: str) -> str:
    """Extract document title from the text block before the first ## heading."""
    parts = re.split(r'^##\s', text, maxsplit=1, flags=re.MULTILINE)
    header = parts[0].strip()
    lines = [l.strip() for l in header.splitlines() if l.strip()]
    # Reject OCR garbage: keep lines where >40% of characters are alphabetic
    good = [l for l in lines
            if len(l) > 4 and sum(c.isalpha() for c in l) / len(l) > 0.4]
    return good[0][:120] if good else ""


def extract_doc_family(doc_id: str) -> str:
    """Strip trailing lowercase letter(s): JAI-N006a → JAI-N006, JAI-185 → JAI-185."""
    return re.sub(r'[a-z]+$', '', doc_id)


def split_into_chunks(text: str, prefix: str) -> list[str]:
    """Split text into word-count chunks, each with prefix prepended."""
    words = text.split()
    if not words:
        return []
    if len(words) <= CHUNK_WORDS:
        return [prefix + text.strip()]
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(prefix + " ".join(words[i:i + CHUNK_WORDS]))
        i += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


def chunk_document(text: str, doc_id: str) -> list[tuple[str, str]]:
    """
    Split a markdown document into chunks by ## heading sections.
    Returns list of (chunk_text, section_heading).
    Every chunk is prefixed with [doc_id], title, and section heading.
    """
    title = extract_title(text)
    tag = f"[{doc_id}]" + (f" {title}" if title else "")

    # Split on ## headings, keeping the heading text
    parts = re.split(r'^(##[^#\n][^\n]*)', text, flags=re.MULTILINE)
    results = []

    # parts[0] = text before first heading (title block)
    # then alternating: heading, content, heading, content, ...
    pre = parts[0].strip()
    if len(pre.split()) > 30:
        prefix = f"{tag}\n## {title or 'Introduction'}\n\n"
        for chunk in split_into_chunks(pre, prefix):
            results.append((chunk, title or "Introduction"))

    i = 1
    while i < len(parts):
        heading_line = parts[i].strip()
        heading_text = re.sub(r'^#+\s*', '', heading_line)
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        i += 2

        if not heading_text and not content:
            continue

        prefix = f"{tag}\n{heading_line}\n\n"
        section_body = content if content else heading_text
        for chunk in split_into_chunks(section_body, prefix):
            results.append((chunk, heading_text))

    # Fallback: no headings at all — chunk the whole file
    if not results:
        prefix = f"{tag}\n\n"
        for chunk in split_into_chunks(text.strip(), prefix):
            results.append((chunk, ""))

    return results


def ingest_file(md_file: Path, collection, rebuild: bool = False) -> int:
    if not rebuild:
        existing = collection.get(where={"source": md_file.name})
        if existing and existing["ids"]:
            return -1  # skipped

    doc_id = md_file.stem
    doc_family = extract_doc_family(doc_id)
    text = md_file.read_text()
    title = extract_title(text)
    chunks = chunk_document(text, doc_id)

    # Remove stale chunks for this file before upserting
    try:
        old = collection.get(where={"source": md_file.name})
        if old and old["ids"]:
            collection.delete(ids=old["ids"])
    except Exception:
        pass

    count = 0
    for i, (chunk_text, section) in enumerate(chunks):
        if len(chunk_text.strip()) < MIN_CHUNK_CHARS:
            continue
        embedding = get_embedding(chunk_text)
        if embedding is None:
            continue
        collection.upsert(
            ids=[f"{doc_id}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk_text],
            metadatas=[{
                "source": md_file.name,
                "doc_id": doc_id,
                "doc_family": doc_family,
                "title": title,
                "section": section,
                "chunk": i,
            }],
        )
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest markdown into ChromaDB.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe existing collection and re-index everything")
    parser.add_argument("--file", type=str, default=None,
                        help="Re-index a single file (by name, e.g. JAI-N006a.md)")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=DB_PATH)

    if args.rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted existing collection.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if args.file:
        md_file = MARKDOWN_DIR / args.file
        if not md_file.exists():
            print(f"File not found: {md_file}")
            sys.exit(1)
        n = ingest_file(md_file, collection, rebuild=True)
        print(f"{md_file.name}: {n} chunks indexed.")
        return

    md_files = sorted(MARKDOWN_DIR.glob("*.md"))
    if not md_files:
        print("No markdown files found.")
        sys.exit(1)

    print(f"{'Re-indexing' if args.rebuild else 'Indexing'} {len(md_files)} files...")
    total, skipped = 0, 0
    for md_file in md_files:
        n = ingest_file(md_file, collection, rebuild=args.rebuild)
        if n == -1:
            skipped += 1
        else:
            total += n
            print(f"  {md_file.name}: {n} chunks")

    print(f"\nDone. {total} chunks indexed, {skipped} files skipped.")


if __name__ == "__main__":
    main()
