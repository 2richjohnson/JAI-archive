#!/usr/bin/env python3
"""
02b_generate_wiki.py — Generate structured wiki articles from JAI archive markdown files.

Reads from ~/jai-archive/markdown/, generates entity-structured wiki articles per
country/cask/vendor/facility/regulatory item/topic, and saves to ~/jai-archive/wiki/.
These articles replace raw chunks as the ChromaDB ingestion source (03_ingest.py).

Benefits over raw chunking:
- Consistent structure regardless of source document quality
- Entity-based retrieval ("UK storage" hits one UK article, not scattered chunks)
- Compounding knowledge — new documents enrich existing articles
- Handles OCR artifacts via LLM interpretation

Usage:
    python 02b_generate_wiki.py                  # process all new markdown files
    python 02b_generate_wiki.py --doc JAI-490.md # process specific document
    python 02b_generate_wiki.py --force          # regenerate all articles
    python 02b_generate_wiki.py --index-only     # generate INDEX.md only
    python 02b_generate_wiki.py --validate       # check for broken wikilinks
    python 02b_generate_wiki.py --stats          # show wiki statistics
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import ollama

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_ENDPOINT = "http://localhost:11434"

MARKDOWN_DIR = Path.home() / "jai-archive/markdown"
WIKI_DIR = Path.home() / "jai-archive/wiki"
LOG_FILE = Path.home() / "jai-archive/logs/wiki_generation.log"
REGISTRY_FILE = WIKI_DIR / "processed_docs.json"

WIKI_SUBDIRS = ["casks", "countries", "vendors", "regulatory", "facilities", "topics"]

MAX_DOCUMENT_CONTEXT = 3000  # words of source document fed to LLM per generation call

ENTITY_TYPE_TO_DIR: dict[str, str] = {
    "CASK":       "casks",
    "COUNTRY":    "countries",
    "VENDOR":     "vendors",
    "REGULATORY": "regulatory",
    "FACILITY":   "facilities",
    "TOPIC":      "topics",
    "MIXED":      "topics",
}

TODAY = datetime.now().strftime("%Y-%m-%d")
log = logging.getLogger(__name__)


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ── LLM helper ────────────────────────────────────────────────────────────────
def _llm(prompt: str, num_ctx: int = 8192) -> str:
    client = ollama.Client(host=OLLAMA_ENDPOINT)
    r = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"temperature": 0, "num_ctx": num_ctx},
    )
    return r.get("response", "").strip()


# ── Article templates (per entity type) ───────────────────────────────────────
_TEMPLATES: dict[str, str] = {
    "CASK": """\
# {entity_name}

## Summary
[2-3 sentence overview of this cask/canister system]

## Key Facts
- [Most important quantitative facts with units; cite source document]

## Specifications
| Attribute | Value |
|-----------|-------|
| Capacity | |
| Weight empty | |
| Weight loaded | |
| Overall length | |
| Overall diameter | |
| Wall thickness | |
| Cavity atmosphere | |
| Maximum burnup | |
| Design heat rejection | |
| Surface dose | |

## Description
[Narrative description of cask design and construction]

## Licensing Status
[NRC license or Certificate of Compliance (CoC) number, approval date]

## Commercial Use
[Units deployed, locations, utilities using this cask]

## Vendor Contact
[Manufacturer name and contact information]

## Source Documents
- [JAI document references]

## Related Articles
- [[related entity]] — brief relationship description

## Last Updated
- {today}
- Documents processed: {source_doc}""",

    "COUNTRY": """\
# {entity_name}

## Summary
[2-3 sentence overview of this country's nuclear fuel management situation]

## Key Facts
- [Most important quantitative facts with units; cite source document]

## Nuclear Program Overview
[Brief description of country's nuclear power program]

## Reactor Fleet
| Station | Type | Capacity (MWe) | Operator |
|---------|------|----------------|----------|

## Spent Fuel Storage
| Storage Type | Capacity (MTU) | Inventory (MTU) | Year |
|--------------|----------------|-----------------|------|
| Wet storage | | | |
| Dry storage | | | |

## Storage Policy
[Government and utility policy on spent fuel management]

## Reprocessing
[Reprocessing contracts, facilities, volumes if applicable]

## Repository Program
[Status of geological repository development]

## Key Organizations
- [Utilities, regulatory bodies, research organizations]

## JAI Consulting Relationship
[JAI work performed for this country's utilities, if any]

## Source Documents
- [JAI document references]

## Related Articles
- [[related entity]] — brief relationship description

## Last Updated
- {today}
- Documents processed: {source_doc}""",

    "VENDOR": """\
# {entity_name}

## Summary
[2-3 sentence overview of company and product focus]

## Key Facts
- [Most important facts about company and product line]

## Company Overview
[Company history, size, and nuclear cask product focus]

## Product Portfolio
[List of cask systems offered with brief description of each]

## Certifications
[NRC and international certifications held]

## Commercial Deployments
[Summary of deployed systems by utility/country]

## Affiliates and Subsidiaries
[Related companies and their roles]

## Source Documents
- [JAI document references]

## Related Articles
- [[related entity]] — brief relationship description

## Last Updated
- {today}
- Documents processed: {source_doc}""",

    "REGULATORY": """\
# {entity_name}

## Summary
[2-3 sentence overview of this license, certificate, or regulation]

## Key Facts
- [Most important regulatory facts and key dates]

## License Details
[NRC license or CoC number, approval date, renewal history]

## Scope and Applicability
[What cask systems or facilities this covers]

## Key Requirements
[Critical technical or operational requirements]

## Current Status
[Active, expired, under review]

## Source Documents
- [JAI document references]

## Related Articles
- [[related entity]] — brief relationship description

## Last Updated
- {today}
- Documents processed: {source_doc}""",

    "FACILITY": """\
# {entity_name}

## Summary
[2-3 sentence overview of this facility]

## Key Facts
- [Most important quantitative facts with units; cite source document]

## Facility Overview
[Facility type, location, operator]

## Storage Capacity
| Storage Type | Capacity | Inventory | Year |
|--------------|----------|-----------|------|

## Operational History
[Key dates, status changes, significant events]

## Cask Systems Used
[Cask models deployed at this facility]

## Regulatory Status
[Operating license, regulatory oversight]

## Source Documents
- [JAI document references]

## Related Articles
- [[related entity]] — brief relationship description

## Last Updated
- {today}
- Documents processed: {source_doc}""",

    "TOPIC": """\
# {entity_name}

## Summary
[2-3 sentence overview of this technical topic]

## Key Facts
- [Most important technical facts]

## Overview
[Detailed description of this topic in the nuclear fuel management context]

## Current Status and Trends
[Industry status, regulatory developments]

## Key Technical Parameters
[Important specifications or standards]

## Related Technologies
[Related systems, processes, or approaches]

## Source Documents
- [JAI document references]

## Related Articles
- [[related entity]] — brief relationship description

## Last Updated
- {today}
- Documents processed: {source_doc}""",
}
_TEMPLATES["MIXED"] = _TEMPLATES["TOPIC"]


# ── Prompts ────────────────────────────────────────────────────────────────────
_CLASSIFICATION_PROMPT = """\
Analyze this nuclear industry document and identify all significant entities it describes.

Document filename: {filename}

Document opening (~300 words):
{excerpt}

Document section headings (full list):
{headings}

Entity type definitions:
- CASK: a specific storage/transport cask or canister model (e.g. TN-40, HI-STORM 100)
- COUNTRY: a nation covered in a survey (e.g. Finland, United Kingdom, Germany)
- VENDOR: a cask manufacturer or company (e.g. Transnuclear, Holtec International)
- REGULATORY: an NRC license, CoC, or regulation (e.g. CoC 72-1002, 10 CFR 72)
- FACILITY: a storage site or nuclear plant (e.g. Sellafield, Olkiluoto ISFSI)
- TOPIC: a cross-cutting technical subject (e.g. Dry Cask Storage, Fuel Reprocessing)
- MIXED: document covers multiple entity types without a clear primary

Return ONLY valid JSON, no other text:
{{
  "primary_entity_type": "CASK|COUNTRY|VENDOR|REGULATORY|FACILITY|TOPIC|MIXED",
  "primary_entity_name": "name of the most prominent entity (use Survey if truly mixed)",
  "all_entities": [
    {{"type": "CASK|COUNTRY|VENDOR|REGULATORY|FACILITY|TOPIC", "name": "entity name"}}
  ],
  "confidence": "high|medium|low"
}}

Rules:
- For country survey documents, list each country separately in all_entities
- For cask catalog sections, list each cask model separately
- Only include entities with SUBSTANTIAL coverage, not brief mentions
- Use full proper names: "United Kingdom" not "UK", "Transnuclear Inc." not "TN\""""

_GENERATION_PROMPT = """\
You are building a wiki for JAI Corporation's nuclear spent fuel management consulting archive.

Generate a wiki article for: {entity_name} (type: {entity_type})
Source document: {source_doc}

Relevant content from source document:
{document_content}

{existing_section}\
Article template to fill in:
{template}

Requirements:
- Fill in every section of the template with data from the source document
- Include ALL quantitative data with units and dates (exact figures, not approximations)
- Interpret OCR garbled text using nuclear domain knowledge; add [OCR unclear] only when genuinely ambiguous
- Create [[wikilinks]] for: country names, cask model names, vendor names, facility names
- For sections with no data in the source, write exactly: [No data in source documents]
- Cite the source document for major data points: e.g. "(JAI-490)"
- Do NOT add any text before the # heading or after the last line
- Return the complete markdown article ONLY, no explanation or preamble

Article:"""

_MERGE_PROMPT = """\
Update this nuclear wiki article by merging in new information from an additional source.

Entity: {entity_name}
New source: {new_source}

Existing article:
{existing_article}

New content from {new_source}:
{new_content}

Instructions:
- Merge new information without removing verified existing data
- Update figures only if new data is more recent; note both the old and new source
- Add {new_source} to the Source Documents section
- Flag conflicts as: [CONFLICT: existing says X, {new_source} says Y]
- Preserve all [[wikilinks]] and add new ones for newly mentioned entities
- Update "Last Updated" date to {today} and append {new_source} to "Documents processed"
- Return the complete updated article ONLY

Updated article:"""


# ── Entity content extraction ──────────────────────────────────────────────────
def extract_entity_content(text: str, entity_name: str,
                            max_words: int = MAX_DOCUMENT_CONTEXT) -> str:
    """
    Extract the sections of a document most relevant to a specific entity.

    Search order:
    1. Sections whose heading names the entity
    2. Paragraphs that mention the entity
    3. Beginning of document (fallback)
    """
    entity_lower = entity_name.lower()

    # Split on markdown headings
    sections = re.split(r'^(#{1,3}[^\n]+)', text, flags=re.MULTILINE)

    relevant_parts: list[str] = []
    total_words = 0

    # Pass 1: sections whose heading contains the entity name
    i = 1
    while i + 1 < len(sections):
        heading = sections[i]
        content = sections[i + 1]
        if entity_lower in heading.lower():
            part = f"{heading}\n{content}"
            relevant_parts.append(part)
            total_words += len(part.split())
        i += 2

    # Pass 2: paragraphs that mention the entity (if no dedicated section found)
    if not relevant_parts:
        for para in re.split(r'\n{2,}', text):
            if entity_lower in para.lower():
                relevant_parts.append(para)
                total_words += len(para.split())
                if total_words >= max_words:
                    break

    # Fallback: use beginning of document
    if not relevant_parts:
        words = text.split()
        return " ".join(words[:max_words])

    combined = "\n\n".join(relevant_parts)
    words = combined.split()
    return " ".join(words[:max_words])


# ── Article path helpers ───────────────────────────────────────────────────────
def entity_to_filename(entity_name: str) -> str:
    """Convert entity name to safe filename: 'United Kingdom' → 'United_Kingdom.md'"""
    name = entity_name.strip().replace(" ", "_")
    name = re.sub(r'[^\w\-.]', '', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return (name or "Unknown") + ".md"


def get_wiki_path(entity_type: str, entity_name: str) -> Path:
    """Return the full path where this entity's wiki article should be saved."""
    subdir = ENTITY_TYPE_TO_DIR.get(entity_type.upper(), "topics")
    return WIKI_DIR / subdir / entity_to_filename(entity_name)


# ── Registry (tracks processed markdown files for resumability) ────────────────
def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_registry(registry: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


# ── Document classification ────────────────────────────────────────────────────
def _classify_from_filename(filename: str) -> dict:
    """Fallback classification derived from filename patterns alone."""
    name = filename.lower()
    if any(x in name for x in ["cask", " tn-", "_tn-", "hi-storm", "nuhoms",
                                "castor", "holtec", "magnastor", "nac-"]):
        return {
            "primary_entity_type": "CASK",
            "primary_entity_name": Path(filename).stem,
            "all_entities": [{"type": "CASK", "name": Path(filename).stem}],
            "confidence": "low",
        }
    if any(x in name for x in ["survey", "jai-490", "jai-497"]):
        return {
            "primary_entity_type": "MIXED",
            "primary_entity_name": "Survey",
            "all_entities": [],
            "confidence": "low",
        }
    stem = Path(filename).stem.replace("-", " ").replace("_", " ")
    return {
        "primary_entity_type": "TOPIC",
        "primary_entity_name": stem,
        "all_entities": [{"type": "TOPIC", "name": stem}],
        "confidence": "low",
    }


def _extract_headings(text: str, max_headings: int = 60) -> str:
    """Extract ## headings from document to give LLM a full structural overview."""
    headings = re.findall(r'^#{1,3}[^\n]+', text, re.MULTILINE)
    # Filter out pure table-header lines (all caps + spaces, no letters > 1 word)
    meaningful = [h for h in headings if len(re.sub(r'[^a-zA-Z]', '', h)) > 3]
    return "\n".join(meaningful[:max_headings]) if meaningful else "(no headings found)"


def classify_document(md_file: Path) -> dict:
    """
    Use LLM to classify document entity type and enumerate all significant entities.
    Uses opening excerpt + full heading list so survey documents covering many
    countries/casks all get detected regardless of document length.
    Falls back to filename-based classification if LLM output cannot be parsed.
    """
    text = md_file.read_text()
    excerpt = " ".join(text.split()[:300])
    headings = _extract_headings(text)

    prompt = _CLASSIFICATION_PROMPT.format(
        filename=md_file.name,
        excerpt=excerpt,
        headings=headings,
    )
    response = _llm(prompt, num_ctx=4096)

    # Try progressively looser JSON extraction patterns
    for pattern in [
        r'\{[^{}]*"primary_entity_type"[^{}]*\}',
        r'\{.*?"primary_entity_type".*?\}',
        r'\{.*\}',
    ]:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if "primary_entity_type" not in data:
                    continue
                # Normalize all_entities
                if not isinstance(data.get("all_entities"), list):
                    data["all_entities"] = [{
                        "type": data["primary_entity_type"],
                        "name": data.get("primary_entity_name", md_file.stem),
                    }]
                return data
            except json.JSONDecodeError:
                continue

    log.warning("JSON parse failed for %s (response: %.80r) — using filename fallback",
                md_file.name, response)
    return _classify_from_filename(md_file.name)


# ── Article generation and merging ────────────────────────────────────────────
def generate_article(
    entity_name: str,
    entity_type: str,
    document_content: str,
    source_doc: str,
    existing_article: str = "",
) -> str:
    """
    Generate a new wiki article, or merge new content into an existing one.
    Returns the complete markdown article text.
    """
    template = _TEMPLATES.get(entity_type.upper(), _TEMPLATES["TOPIC"]).format(
        entity_name=entity_name,
        today=TODAY,
        source_doc=source_doc,
    )

    existing_section = ""
    if existing_article.strip():
        # Truncate existing article to keep prompt within context limits
        ex_words = existing_article.split()
        truncated = (
            " ".join(ex_words[:800]) + "\n[... truncated ...]"
            if len(ex_words) > 800
            else existing_article
        )
        existing_section = (
            "Existing article to update (merge new info without losing existing data):\n"
            f"{truncated}\n\n"
        )

    prompt = _GENERATION_PROMPT.format(
        entity_name=entity_name,
        entity_type=entity_type,
        source_doc=source_doc,
        document_content=document_content,
        existing_section=existing_section,
        template=template,
    )
    return _llm(prompt, num_ctx=12288)


# ── Wikilink utilities ─────────────────────────────────────────────────────────
def extract_wikilinks(article_text: str) -> list[str]:
    """Return all [[wikilink]] references found in an article."""
    return re.findall(r'\[\[([^\]]+)\]\]', article_text)


def validate_wikilinks(wiki_dir: Path) -> dict[str, list[str]]:
    """
    Find wikilinks that point to non-existent articles.
    Returns a dict of {broken_link: [articles_referencing_it]}.
    """
    known: set[str] = set()
    for subdir in WIKI_SUBDIRS:
        d = wiki_dir / subdir
        if d.exists():
            for article in d.glob("*.md"):
                stem = article.stem
                known.add(stem.lower())
                known.add(stem.replace("_", " ").lower())

    broken: dict[str, list[str]] = {}
    for subdir in WIKI_SUBDIRS:
        d = wiki_dir / subdir
        if not d.exists():
            continue
        for article in d.glob("*.md"):
            for link in extract_wikilinks(article.read_text()):
                norm = link.strip().lower()
                if norm not in known and norm.replace(" ", "_") not in known:
                    broken.setdefault(link, []).append(
                        str(article.relative_to(wiki_dir))
                    )

    if broken:
        log.info("Broken wikilinks (candidates for future document scanning):")
        for link, sources in sorted(broken.items()):
            log.info("  [[%s]] — referenced in: %s", link, ", ".join(sources[:3]))
    else:
        log.info("All wikilinks resolve to existing articles.")

    return broken


# ── Index generation ───────────────────────────────────────────────────────────
def generate_index(wiki_dir: Path) -> None:
    """Generate wiki/INDEX.md listing all articles by category."""
    lines: list[str] = [f"# JAI Archive Wiki — Index\n\nGenerated: {TODAY}\n\n"]

    total = 0
    for subdir in WIKI_SUBDIRS:
        d = wiki_dir / subdir
        if not d.exists():
            continue
        articles = sorted(d.glob("*.md"))
        if not articles:
            continue
        total += len(articles)
        category = subdir.replace("_", " ").title()
        lines.append(f"## {category} ({len(articles)})\n\n")
        for article in articles:
            name = article.stem.replace("_", " ")
            lines.append(f"- [{name}]({subdir}/{article.name})\n")
        lines.append("\n")

    (wiki_dir / "INDEX.md").write_text("".join(lines))
    log.info("Generated INDEX.md — %d articles", total)


# ── Main document processing ───────────────────────────────────────────────────
def process_document(
    md_file: Path,
    wiki_dir: Path,
    registry: dict,
    force: bool = False,
) -> dict:
    """
    Process a single markdown file:
    1. Classify entity type(s) and name(s)
    2. For each detected entity, generate new or merge into existing wiki article
    3. Save to appropriate wiki subdirectory
    4. Update registry

    Returns processing summary dict.
    """
    result = {
        "file": md_file.name,
        "entities_processed": 0,
        "articles_created": 0,
        "articles_updated": 0,
        "errors": [],
    }

    if not force and md_file.name in registry:
        log.info("SKIP %s (already processed; --force to regenerate)", md_file.name)
        return result

    log.info("Processing: %s", md_file.name)
    t_start = time.time()

    # Step 1: Classify
    try:
        classification = classify_document(md_file)
    except Exception as exc:
        log.error("Classification error for %s: %s", md_file.name, exc)
        result["errors"].append(f"classification: {exc}")
        return result

    log.info("  Type: %s / %s (confidence: %s)",
             classification["primary_entity_type"],
             classification["primary_entity_name"],
             classification.get("confidence", "?"))

    # Build deduplicated entity list
    all_entities = classification.get("all_entities") or []
    seen_keys: set[tuple[str, str]] = set()
    entities_to_process: list[dict] = []
    for ent in all_entities:
        etype = ent.get("type", "TOPIC").upper()
        ename = ent.get("name", "").strip()
        if etype == "MIXED" or not ename:
            continue
        key = (etype, ename.lower())
        if key not in seen_keys:
            seen_keys.add(key)
            entities_to_process.append({"type": etype, "name": ename})

    # Fallback to primary entity
    if not entities_to_process:
        etype = classification["primary_entity_type"]
        ename = classification.get("primary_entity_name", md_file.stem)
        if etype != "MIXED" and ename:
            entities_to_process = [{"type": etype, "name": ename}]
        else:
            entities_to_process = [{"type": "TOPIC", "name": md_file.stem}]

    text = md_file.read_text()
    articles_generated: list[str] = []

    for entity in entities_to_process:
        entity_type = entity["type"]
        entity_name = entity["name"]
        result["entities_processed"] += 1

        # Extract document sections most relevant to this entity
        doc_content = extract_entity_content(text, entity_name)

        wiki_path = get_wiki_path(entity_type, entity_name)
        wiki_path.parent.mkdir(parents=True, exist_ok=True)

        existing_article = ""
        is_update = wiki_path.exists()
        if is_update:
            existing_article = wiki_path.read_text()
            log.info("  Updating: %s", wiki_path.relative_to(wiki_dir))
        else:
            log.info("  Creating: %s", wiki_path.relative_to(wiki_dir))

        try:
            t0 = time.time()
            article_text = generate_article(
                entity_name=entity_name,
                entity_type=entity_type,
                document_content=doc_content,
                source_doc=md_file.name,
                existing_article=existing_article,
            )
            elapsed = time.time() - t0
        except Exception as exc:
            log.error("  Generation failed for %s: %s", entity_name, exc)
            result["errors"].append(f"{entity_name}: {exc}")
            continue

        # Ensure article starts with a heading
        if not article_text.lstrip().startswith("#"):
            article_text = f"# {entity_name}\n\n{article_text}"

        wiki_path.write_text(article_text)
        articles_generated.append(str(wiki_path.relative_to(wiki_dir)))

        wikilinks = extract_wikilinks(article_text)
        action = "Updated" if is_update else "Created"
        log.info("  ✓ %s %s (%.1fs, %d wikilinks)",
                 action, entity_name, elapsed, len(wikilinks))

        if is_update:
            result["articles_updated"] += 1
        else:
            result["articles_created"] += 1

    # Only register success if at least one article was generated
    if articles_generated:
        registry[md_file.name] = {
            "processed_at": datetime.now().isoformat(),
            "primary_type": classification["primary_entity_type"],
            "primary_name": classification["primary_entity_name"],
            "confidence": classification.get("confidence", "?"),
            "articles": articles_generated,
            "elapsed_s": round(time.time() - t_start, 1),
        }

    total_articles = result["articles_created"] + result["articles_updated"]
    log.info("Done %s: %d article(s) in %.1fs",
             md_file.name, total_articles, time.time() - t_start)
    return result


# ── Statistics ─────────────────────────────────────────────────────────────────
def print_stats(wiki_dir: Path) -> None:
    print("\n── Wiki Statistics ───────────────────────────────────")
    total = 0
    for subdir in WIKI_SUBDIRS:
        d = wiki_dir / subdir
        if d.exists():
            count = len(list(d.glob("*.md")))
            if count:
                print(f"  {subdir:<15} {count:>4} articles")
                total += count
    print(f"  {'TOTAL':<15} {total:>4} articles")

    if REGISTRY_FILE.exists():
        registry = load_registry()
        print(f"\n  Markdown files processed: {len(registry)}")

    all_links: list[str] = []
    for subdir in WIKI_SUBDIRS:
        d = wiki_dir / subdir
        if d.exists():
            for article in d.glob("*.md"):
                all_links.extend(extract_wikilinks(article.read_text()))
    if all_links:
        print(f"\n  Total wikilinks: {len(all_links)}")
        print(f"  Unique linked entities: {len(set(all_links))}")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate structured wiki articles from JAI archive markdown files."
    )
    parser.add_argument("--doc", type=str, default=None,
                        help="Process a specific file (filename or full path)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate articles even if already processed")
    parser.add_argument("--index-only", dest="index_only", action="store_true",
                        help="Only regenerate INDEX.md (no article generation)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate wikilinks and report broken references")
    parser.add_argument("--stats", action="store_true",
                        help="Show current wiki statistics")
    args = parser.parse_args()

    setup_logging()

    for subdir in WIKI_SUBDIRS:
        (WIKI_DIR / subdir).mkdir(parents=True, exist_ok=True)

    if args.stats:
        print_stats(WIKI_DIR)
        return

    if args.validate:
        validate_wikilinks(WIKI_DIR)
        return

    if args.index_only:
        generate_index(WIKI_DIR)
        return

    # Determine file list
    if args.doc:
        doc_path = Path(args.doc)
        if not doc_path.is_absolute():
            doc_path = MARKDOWN_DIR / args.doc
        if not doc_path.exists():
            log.error("File not found: %s", doc_path)
            sys.exit(1)
        md_files = [doc_path]
    else:
        if not MARKDOWN_DIR.exists():
            log.error("Markdown directory not found: %s", MARKDOWN_DIR)
            sys.exit(1)
        md_files = sorted(MARKDOWN_DIR.glob("*.md"))
        if not md_files:
            log.warning("No markdown files found in %s", MARKDOWN_DIR)
            return
        log.info("Found %d markdown files", len(md_files))

    registry = load_registry()
    total_created = total_updated = total_errors = 0

    for md_file in md_files:
        result = process_document(md_file, WIKI_DIR, registry, force=args.force)
        total_created += result["articles_created"]
        total_updated += result["articles_updated"]
        total_errors += len(result["errors"])
        save_registry(registry)  # save after each file for resumability

    generate_index(WIKI_DIR)

    log.info("=" * 50)
    log.info("Wiki generation complete: %d created, %d updated%s",
             total_created, total_updated,
             f", {total_errors} error(s)" if total_errors else "")
    log.info("Wiki directory: %s", WIKI_DIR)


if __name__ == "__main__":
    main()
