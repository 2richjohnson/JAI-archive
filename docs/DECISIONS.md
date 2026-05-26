# JAI Archive — Architecture & Design Decisions

---

## 2026-05-25: Heading-aware ChromaDB ingestion (03_ingest.py rewrite)

**Decision**: Rewrite `03_ingest.py` to chunk by `##` heading sections instead of blind 300-word word-count cuts. Prepend `[doc_id] title\n## section\n` to every chunk. Add `doc_id`, `doc_family`, `title`, `section` metadata to every chunk.

**Context**: Query results were consistently thin despite retrieving 10 chunks. Root causes:
1. Blind word-count chunking produces mid-sentence, mid-section fragments with no structural context — the LLM can't use them well.
2. Synthesis was truncating at 3,000 chars, discarding most retrieved chunks. Raised to 8,000 chars / 12,288 `num_ctx`.
3. Document-name queries (e.g. "summary of JAI-N006") were failing because doc names don't appear in chunk content and only filename was stored in metadata.

**What the rewrite provides**:
- Each chunk is a coherent section with its heading intact
- `[doc_id]` prefix in every chunk means semantic search for a document name naturally finds its chunks
- `doc_family` metadata enables clean `where` filtering without regex prefix-scanning
- `--rebuild` flag wipes and rebuilds; `--file X.md` re-indexes one file; normal run skips existing

**Trade-off**: Re-ingestion required (~15–20 min with nomic-embed-text on GPU0). Chunk count increased from 1,367 → 6,314 (more sections, each more meaningful than old 300-word cuts).

**Rebuild completed 2026-05-26**: 6,314 chunks indexed. `_source_where_filter` updated to use `doc_family`/`doc_id` metadata directly. Both content queries and doc-name queries confirmed improved.

---

## 2026-05-25: GPU1 permanently excluded — qwen2.5:14b unsafe even with NUM_PARALLEL=1

**Outcome**: Experiment failed. `OLLAMA_NUM_PARALLEL=1` with qwen2.5:14b on both GPUs → Ollama crashed (only Ollama this time, not the full host — improvement over NUM_PARALLEL=2, but still a crash). GPU1 hit 100% utilization; GPU0 barely loaded. The model was not spreading evenly — GPU1 was carrying inference load and saturating.

**Conclusion**: GPU1 has a hardware reliability issue (thermal defect or VRAM fault). The NUM_PARALLEL fix addresses the parallelism amplifier but cannot fix a bad card. Any sustained GPU1 load remains unsafe regardless of model size or config.

**Permanent fix applied (2026-05-25)**:
- Added `CUDA_VISIBLE_DEVICES=0` to `/etc/systemd/system/ollama.service.d/override.conf`
- Ollama restarted; GPU1 now permanently invisible to Ollama
- `07_query.py` reverted: `LLM_MODEL = "llama3.1:8b"`, `SQL_MODEL = "qwen2.5-coder:7b"`
- 14B+ models on this VM: **do not attempt**. Use AWS for anything requiring larger models.

**Do not revert** `CUDA_VISIBLE_DEVICES=0` unless GPU1 is physically diagnosed and replaced.

---

## 2026-05-25: Pipeline parallelism root cause — n_copies=4 is the real VRAM killer

**Finding**: The qwen2.5:14b crash was NOT caused by the model being too large to load.
Ollama logs confirm it loaded cleanly at 22:14 (5.4 GiB per GPU, within limits).
The crash happened ~8 minutes later, during inference. Root cause from logs:

```
llama_context: n_seq_max = 2          ← OLLAMA_NUM_PARALLEL=2
pipeline parallelism enabled (n_copies=4)  ← 2 GPUs × 2 sequences = 4 activation copies
```

With n_copies=4, inference creates 4 sets of activation buffers and intermediate tensors
spread across both GPUs simultaneously. That pushes peak VRAM over the limit during generation.

**VRAM breakdown (NUM_PARALLEL=2, observed from logs):**
- Model weights: CUDA0=4,065 MiB, CUDA1=4,083 MiB
- KV cache:      CUDA0=  832 MiB, CUDA1=  704 MiB
- Compute graph: CUDA0=  676 MiB, CUDA1=  676 MiB
- Per-GPU total: ~5.5 GiB used, ~1.9 GiB headroom (out of ~7.5 GiB available)

That 1.9 GiB headroom is not enough for n_copies=4 activation buffers during inference.

**Fix to try: OLLAMA_NUM_PARALLEL=1**
With NUM_PARALLEL=1: n_seq_max=1, n_copies drops from 4 → 2.
KV cache and compute overhead roughly halve:
- Per-GPU total drops to ~4.8 GiB, headroom grows to ~2.7 GiB
- Loses concurrent request handling; gains safe 14B inference

**Experiment plan (2026-05-25):**
1. Set `OLLAMA_NUM_PARALLEL=1` in `/etc/systemd/system/ollama.service.d/override.conf`
2. `systemctl daemon-reload && systemctl restart ollama`
3. `watch -n1 nvidia-smi` in a second terminal to monitor VRAM during inference
4. Run a test query: `python 07_query.py --model qwen2.5:14b "What is Finland's wet storage capacity?"`
5. If stable: keep NUM_PARALLEL=1, use qwen2.5:14b as default
6. If crashes again: GPU1 is hardware-suspect; fall back to CUDA_VISIBLE_DEVICES=0

**Fallback options (in order if experiment fails):**
1. `CUDA_VISIBLE_DEVICES=0` + pull `qwen2.5:14b-q3_k_m` (~6.5 GiB, fits GPU0 alone)
2. `CUDA_VISIBLE_DEVICES=0` + `llama3.1:8b` (fully safe, proven stable)
3. AWS for anything requiring 14B+ quality (already planned)

**Caveat**: GPU1 has failed twice. NUM_PARALLEL=1 fixes the parallelism issue but cannot
fix a hardware fault. If the card has a thermal or VRAM defect, any sustained GPU1 load
may still crash. Watch nvidia-smi temps and VRAM closely during the test.

---

## 2026-05-25: Restrict primary Ollama to GPU0 only (CUDA_VISIBLE_DEVICES=0)

**Decision**: Add `Environment="CUDA_VISIBLE_DEVICES=0"` to the primary Ollama service override (`/etc/systemd/system/ollama.service.d/override.conf`). Remove `OLLAMA_NUM_PARALLEL=2` since parallel benefit requires two GPUs; set to `OLLAMA_NUM_PARALLEL=1`.

**Context**: Two hard VM crashes in five days (2026-05-20, 2026-05-25). Both caused by GPU1 VRAM contention — first from `ollama-gpu1.service` loading onto GPU1 simultaneously with the primary; second from `qwen2.5:14b` spreading across both GPUs from the primary service. GPU1 is the consistent failure point. On 2026-05-25, the crash brought down all VMs on the Proxmox host (Proxmox itself survived).

**Rationale**: GPU0 alone provides ~8 GB usable VRAM — sufficient for `llama3.1:8b` (4.58 GB) + `nomic-embed-text`. Eliminating GPU1 from Ollama's view removes the failure domain entirely. No code changes needed; Ollama automatically uses all visible GPUs, so hiding GPU1 via env var is the simplest safe fix.

**How to apply**:
```bash
# On VM — edit the override file
sudo nano /etc/systemd/system/ollama.service.d/override.conf
# Add/update these lines in [Service]:
#   Environment="CUDA_VISIBLE_DEVICES=0"
#   Environment="OLLAMA_NUM_PARALLEL=1"
# Then:
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**Trade-off**: Single-GPU inference only (~8 GB VRAM). `llama3.1:8b` still fits; 14B+ models can no longer load on this VM. Full corpus extraction must use AWS (already planned — no change to that plan).

**Do not revert** unless GPU1 hardware is diagnosed and confirmed stable.

---

## 2026-05-25: Revert default models in 07_query.py from qwen2.5:14b to safe 8b models

**Decision**: Revert `LLM_MODEL` to `"llama3.1:8b"` and `SQL_MODEL` to `"qwen2.5-coder:7b"` in `07_query.py` (lines 34–35).

**Context**: `qwen2.5:14b` was set as default in an attempt to leverage both GPUs for better synthesis quality. The 14B model at Q4 (~8–9 GB) spreads across both GPUs. This caused a hard VM reset — same failure mode as the 2026-05-20 crash. All VMs went down; Proxmox survived.

**Rationale**: Any model requiring cross-GPU VRAM spread is unsafe on this hardware. The `--model` flag allows on-demand testing of larger models without baking them in as the default. The doc injection feature (`--doc`, `--deep`) compensates for thin retrieval results independently of model size.

**Trade-off**: Synthesis reverts to 8B level. Acceptable — all three routing paths work well at 8B; doc injection adds quality for targeted queries.

---

## 2026-05-25: Document injection and two-stage deep dive added to 07_query.py

**Decision**: Implement `--doc`, `--doc2`, `--pages`, `--deep`, and `--model` flags in `07_query.py`.

**Context**: ChromaDB retrieval for some documents (e.g., JAI-490 UK content) was unreliable — chunks weren't surfacing even though the document was indexed. Direct document injection bypasses retrieval entirely for targeted queries.

**Rationale**: `load_document_context()` reads a markdown or PDF file (with optional page range) and injects it as additional context alongside the ChromaDB results. `two_stage_query()` first does a standard RAG query to identify the top source document, then re-queries with that full document injected. The `--model` flag allows model selection per invocation without modifying code.

**New usage**:
```bash
python 07_query.py --doc ~/jai-archive/markdown/JAI-490.md "UK storage situation"
python 07_query.py --doc ~/jai-archive/ocr/JAI-490.pdf --pages 89-95 "UK figures"
python 07_query.py --deep "Compare Finland and Germany storage"
python 07_query.py --model llama3.1:8b "What is the TN-40 capacity?"
```

---

## 2026-05-24: Force CPU mode for docling in 02_convert.py

**Decision**: Set `AcceleratorDevice.CPU` via `PdfPipelineOptions` in `DocumentConverter` rather than letting docling auto-detect the GPU.

**Context**: After a PyTorch update in the venv, `02_convert.py` started crashing with `CUDA error: no kernel image is available for execution on the device`. GTX 1070 is Pascal architecture (sm_61). PyTorch on Python 3.14 only provides CUDA 12.x builds, which dropped sm_61 support. No cu118 (CUDA 11.8) wheels exist for Python 3.14, so downgrading PyTorch to restore GPU support is not possible.

**Rationale**: CPU mode is fully functional for docling's layout detection. Docling is only used for PDF→markdown conversion when new documents are ingested — it is not a throughput bottleneck (Ollama extraction is). CPU is slower per page but acceptable for occasional ingest runs.

**Trade-off**: Loss of GPU acceleration for PDF layout detection. Acceptable given ingest frequency and that the extraction step (Ollama) dominates total ingest time regardless.

**Revisit if**: Python 3.14 PyTorch cu118 wheels become available, or if a future Python/venv upgrade allows installing a PyTorch build that supports sm_61.

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
