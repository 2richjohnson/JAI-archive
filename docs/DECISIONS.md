# JAI Archive — Architecture & Design Decisions

---

## 2026-05-20: Disable ollama-gpu1.service permanently

**Decision**: Disable the secondary Ollama service (`ollama-gpu1.service`, port 11435) and use only the primary (`ollama.service`, port 11434) with `OLLAMA_NUM_PARALLEL=2`.

**Context**: A secondary Ollama instance was set up to expose GPU1 as a separate endpoint for parallel extraction. On boot, both services auto-started. Both attempted to load llama3.1:8b (~4.58 GiB) onto GPU1. Combined demand exceeded 8 GiB → hard VM reset with no OOM warning. Happened twice (original crash + crash loop on reboot).

**Rationale**: The primary Ollama already uses both GPUs internally (~4.3 GiB spread across each card, 5.86 GiB total). `OLLAMA_NUM_PARALLEL=2` allows it to handle two concurrent requests, queued across both cards. Adding a second service targeting the same physical GPU causes VRAM contention with no safe fallback.

**Trade-off**: Lose the ability to point workers at separate endpoints. Accept: two workers → one Ollama → internal GPU parallelism is sufficient and safe.

**Do not re-enable** `ollama-gpu1.service`. If more throughput is needed, use AWS (see below).

---

## 2026-05-20: EAV (Entity-Attribute-Value) schema for structured extraction

**Decision**: Store every extracted data point as a single row `(entity, attribute, value_raw, value_numeric, unit)` rather than wide/pivot tables.

**Context**: Source documents contain heterogeneous tables — some wide (countries as rows, attributes as columns), some tall (key-value pairs), some cost line items. No fixed schema works across all of them.

**Rationale**: EAV handles arbitrary attributes without schema migrations. New document types add new attribute names without breaking existing queries. DuckDB views (`capacity_summary`, `cask_summary`, `cost_summary`) pivot the most common query patterns for performance.

**Trade-off**: Queries require explicit `WHERE attribute = '...'` predicates. Multi-attribute queries are verbose (subqueries or `WHERE attribute IN (...)`). Acceptable given query volume and the SQL-generation layer in `07_query.py`.

**Revisit if**: Attribute set stabilizes and query patterns become repetitive — at that point a denormalized wide table per entity type may be faster to query and easier to maintain.

---

## 2026-05-20: AWS g5.12xlarge spot for full-corpus extraction

**Decision**: Use AWS g5.12xlarge spot instance for the one-time bulk extraction of the full 15-banker-box corpus. Do not use Claude API (Anthropic).

**Context**: Current extraction (llama3.1:8b on GTX 1070 × 2) produces dirty data — HTML entity artifacts, junk entity classification, missing units. The 7-PDF pilot corpus took ~4.5 hours with two VM crashes. Full corpus is ~1,000–3,000 pages.

**Rationale**: llama3.1:70B Q4_K_M (~40 GiB) on 4× A10G (96 GiB VRAM) eliminates most entity classification and unit extraction errors. Spot price ~$2–2.50/hr; estimated $25–40 total. Same Ollama + `05_extract_tables.py` setup — no code changes, just `--workers 4`. Data is sensitive; must stay within user-controlled infrastructure (not Anthropic's API).

**Smaller option**: g5.2xlarge spot (~$0.35/hr, 1× A10G 24 GiB) — runs llama3.1:8b fast or llama3.3:70B aggressively quantized; ~$10 but lower quality.

**After extraction**: Tear down the instance. Query system stays on local VM permanently.

---

## 2026-05-23: Fix semantic search — use Ollama embeddings for ChromaDB queries

**Decision**: Replace `query_texts` with `query_embeddings` in `semantic_search()`, embedding via `ollama.embeddings(model="nomic-embed-text")`.

**Context**: Every semantic search was silently returning empty. Root cause: `query_texts` invokes ChromaDB's default all-MiniLM-L6-v2 embedder (384-dim), but the collection was built by `03_ingest.py` using `nomic-embed-text` (768-dim). ChromaDB throws a dimension mismatch exception which `semantic_search()` catches and swallows, returning `[]` every time.

**Rationale**: Query embedding must match index embedding. The fix mirrors exactly what `03_ingest.py` does. Added `_embed()` helper using `ollama.Client` so the same model/host config applies.

**Do not use `query_texts`** with this collection unless the collection is rebuilt with ChromaDB's default embedder.

---

## 2026-05-23: Keyword shortcut for semantic routing

**Decision**: Add `_SEMANTIC_KEYWORDS` list to bypass LLM router for descriptive questions.

**Context**: The LLM router misclassified narrative questions ("tell me about rotary dissolvers") as STRUCTURED, so semantic search was never called despite relevant content existing in ChromaDB. This mirrors the existing `_CAPACITY_KEYWORDS` and `_SPECS_KEYWORDS` shortcuts.

**Rationale**: LLM routing is unreliable for open-ended descriptive questions. Keyword shortcuts are deterministic and fast. Phrases like "tell me about", "explain", "summarize", "what is/are", "how does/do" reliably signal semantic intent.

**Trade-off**: Keyword list needs manual maintenance as new question patterns emerge. Acceptable — the LLM router handles anything not matched.

---

## 2026-05-23: 7 markdown files excluded from ChromaDB — confirmed junk

**Decision**: Do not index the 7 markdown files missing from ChromaDB. No action needed.

**Context**: Audit found 170/177 markdown files indexed. The 7 missing: four `~$`-prefixed MS Office temp/lock files (no real content), two "Photos and Such" image-only pages, one Index file (table of contents only).

**Rationale**: None contain queryable text. Indexing them would add noise to semantic search results.

---

## 2026-05-21: Two-pass extraction design in 05_extract_tables.py

**Decision**: Split LLM involvement into Pass 1 (schema inference) and Pass 2 (Python row iteration).

**Context**: Initial single-pass approach asked the LLM to extract all rows. LLM consistently stopped after the first entity in multi-row tables.

**Rationale**: Pass 1 sends only headers + 3 sample rows → LLM returns `{format, entity_column, column_map}`. Pass 2 uses that schema to iterate all data rows in Python. Guarantees complete extraction regardless of table length. LLM token budget is predictable and small.

**Trade-off**: Two LLM calls per table instead of one. Acceptable — Pass 1 is fast (small prompt); Pass 2 is zero LLM cost.
