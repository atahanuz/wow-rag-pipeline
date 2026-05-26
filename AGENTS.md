# Repository Guide

A complete map of this repository, written for an AI agent (or developer) joining cold. Read top-to-bottom; everything you need to operate or extend the system is here.

---

## 1. What this project is

A **Retrieval-Augmented Generation (RAG) system for knowledge-grounded dialogue** over the **Wizard of Wikipedia (WoW)** dataset. Course project for **CMPE 58T — Advanced Natural Language Processing**, by Fatma Gizem YILMAZ & Atahan UZ.

Given a user's dialogue turn plus optional prior turns, the system:
1. Rewrites the turn to resolve coreferences and generates paraphrases.
2. Retrieves relevant Wikipedia passages from a proposition-indexed dense store.
3. Reranks them with a cross-encoder.
4. Generates a grounded answer with `gpt-oss-120b` (placeholder for GenKS + KEDiT).

The full proposed pipeline (`Application Project Proposal.pdf`) has 5 stages and 4 extensions. **Currently implemented: query rewriter (Ext 4), dense retrieval (Stage 1), propositional chunking (Ext 1), cross-encoder reranker (Ext 2), and a placeholder grounded-answer generator.** GenKS, KEDiT, FLARE, Q²/BEGIN and the per-dialogue cache are not started.

---

## 2. TL;DR for an agent

| If you need to... | Look at |
|---|---|
| Run the system | `./start.sh` (boots backend + UI) |
| Add/modify a pipeline stage | `server.py:run` is the single orchestration point |
| Change the LLM/embedding endpoint | `api_clients.py` (`_BASE_URL`, `_LLM_MODEL`, `_EMBED_MODEL`) |
| Change the reranker endpoint | `reranker_call.py:RERANKER_URL` (ngrok URL — rotates often) |
| Build the dense index | call `DenseRetriever.abuild(parents, …)`; persistent files in `data/dense_index/` |
| Rebuild the WoW passage corpus | `python3 build_corpus.py` (reads WoW JSONs in `data/`, writes `data/corpus.jsonl`) |
| Add a new example to the UI | `ui/src/examples.ts` |
| Add a new sidebar setting | `ui/src/components/Sidebar.tsx` + `ui/src/api.ts:Settings` + `server.py:Settings` |
| Regenerate the progress report | `python3 build_progress_report.py` → `progress_report.docx` |
| Read the BM25 baseline | `retrieve.py` (CLI, independent of the rest) |

---

## 3. Pipeline architecture

```
                ┌────────────────────────────────────────────────────────┐
                │ user_turn + history (role/text pairs)                  │
                └────────────────────────────────────────────────────────┘
                                       │
            ┌──────────────────────────▼──────────────────────────┐
            │  Ext 4 — Query rewriter   (query_rewriter.py)       │
            │  in : user_turn, history                            │
            │  out: rewrite + N paraphrases                       │
            │  llm: gpt-oss-120b (achat)                          │
            └──────────────────────────┬──────────────────────────┘
                                       │ all_queries = [rewrite, *paraphrases]
            ┌──────────────────────────▼──────────────────────────┐
            │  Stage 1 + Ext 1 — Dense retrieval                  │
            │  (dense_retriever.py)                               │
            │  index unit:   propositions (1 fact / sentence)     │
            │  scoring:      cosine over normalized embeddings    │
            │  multi-query:  per-prop max-pool across variants    │
            │  output unit:  parent paragraph (collapse by best   │
            │                proposition per parent)              │
            │  embed: qwen3-embedding-8b, 4096-dim                │
            └──────────────────────────┬──────────────────────────┘
                                       │ top-N parent paragraphs (with dense score, best prop)
            ┌──────────────────────────▼──────────────────────────┐
            │  Ext 2 — Cross-encoder reranker (reranker.py)       │
            │  query:  rewrite (canonical, NOT paraphrases)       │
            │  docs:   parent paragraph texts                     │
            │  model:  Qwen3-Reranker via HTTP (reranker_call.py) │
            │  output: top-K with rerank_score in [0,1]           │
            └──────────────────────────┬──────────────────────────┘
                                       │ top-K reranked parents
            ┌──────────────────────────▼──────────────────────────┐
            │  Stage 4 — Grounded answer (generator.py)           │
            │  PLACEHOLDER for GenKS (Stage 2) + KEDiT (Stage 3). │
            │  Plain RAG: feed top-K + history + turn into LLM    │
            │  with a "use only these passages" system prompt.    │
            │  llm: gpt-oss-120b                                  │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │ grounded answer string  │
                          └─────────────────────────┘
```

**Orchestration point**: [`server.py`](server.py) `run()` runs all four stages sequentially and returns timings + per-stage outputs.

---

## 4. Repository layout

```
NLP_AppProj1/
├── Application Project Proposal.pdf   reference: original proposal
├── paper_references.md                map of papers/*.pdf -> citations
├── papers/                            the 5 cited papers (PDFs)
├── mt_progress_report.docx            stylistic reference (another class)
├── progress_report.docx               this project's progress report
├── build_progress_report.py           regenerates progress_report.docx
│
├── .env                               KKB_API=<key>   (LLM/embedding API key)
├── .streamlit/config.toml             Streamlit theme (legacy UI)
│
├── api_clients.py        ──── shared sync+async OpenAI clients (LLM + embed)
├── build_corpus.py       ──── WoW JSON dumps -> data/corpus.jsonl (185,503 unique passages)
├── retrieve.py           ──── BM25 baseline retriever (CLI, standalone)
├── query_rewriter.py     ──── Ext 4: coref rewrite + paraphrases + RRF helper
├── prop_chunker.py       ──── Ext 1: LLM-driven proposition splitter (cached)
├── dense_retriever.py    ──── Stage 1 + Ext 1: build/query proposition index
├── reranker_call.py      ──── raw HTTP wrapper for hosted Qwen3-Reranker
├── reranker.py           ──── Ext 2: adapts Hit -> RerankedHit (sync + async)
├── generator.py          ──── placeholder grounded-answer generator
├── server.py             ──── FastAPI backend: /api/info, /api/run
├── start.sh              ──── boots backend (:8000) + Vite frontend (:5173)
│
├── app.py                ──── Streamlit UI (legacy; no longer updated)
├── test_reranker.py      ──── ad-hoc reranker probing script (NOT pytest)
│
├── ui/                              ──── canonical UI (Vite + React + TS + Tailwind)
│   ├── index.html
│   ├── package.json, tsconfig.json
│   ├── vite.config.ts                  (proxies /api/* -> :8000)
│   ├── tailwind.config.js, postcss.config.js
│   └── src/
│       ├── main.tsx                    React entry
│       ├── App.tsx                     top-level layout + state
│       ├── index.css                   Tailwind layers + custom components
│       ├── api.ts                      Settings/Turn types + fetch wrappers
│       ├── examples.ts                 clickable example presets
│       └── components/
│           ├── Sidebar.tsx
│           ├── Examples.tsx
│           ├── HistoryEditor.tsx
│           └── Trace.tsx               renders the 4-stage pipeline trace
│
├── data/                            ──── generated + downloaded data (gitignore-worthy)
│   ├── wizard_of_wikipedia.tgz         WoW dataset archive (~927 MB)
│   ├── train.json / valid_*.json / test_*.json / data.json   raw WoW dumps
│   ├── topic_splits.json
│   ├── corpus.jsonl                    185,503 unique (title, text) pairs
│   ├── prop_cache.jsonl                LLM chunker cache, keyed by sha1(title,text)
│   └── dense_index/                    in-memory index, persisted
│       ├── props.npy                   (N, 4096) float32, L2-normalized
│       ├── props.jsonl                 one row per proposition: {text, parent}
│       └── parents.jsonl               one row per parent: {title, text}
│
└── nlp/, nlp.zip                  ──── snapshot/export. IGNORE this directory.
                                         It mirrors the top-level repo (minus data/)
                                         and is the source of nlp.zip. Not part of
                                         the live codebase.
```

> **Note on `nlp/`**: it's an identical mirror of the top level files for export purposes. Don't edit files inside it; they will not affect the running system. If you find a discrepancy between top level and `nlp/`, trust the top level.

---

## 5. Python modules in detail

All modules expose **both sync and async** entry points where it matters (the async ones use `AsyncOpenAI` and `asyncio.gather` with a bounded semaphore for I/O concurrency). Use the sync ones for scripts; the async ones for batch work or future server concurrency.

### `api_clients.py`

Thin wrapper over the OpenAI Python SDK pointed at the class's Kloudeks endpoint.

```python
chat(prompt, system=None, temperature=0.0) -> str
embed(text) -> list[float]                          # 4096-dim
embed_batch(texts) -> list[list[float]]             # single request

achat / aembed / aembed_batch                       # async variants

gather_bounded(awaitables, concurrency=16, on_done=None) -> list
EMBED_DIM = 4096
DEFAULT_CONCURRENCY = 16
```

- **Base URL**: `https://mia.csp.kloudeks.com/v1`
- **LLM**: `gpt-oss-120b`
- **Embedding**: `qwen3-embedding-8b` (4096-dim)
- **Key**: read from `KKB_API` env var (loaded from `.env` via python-dotenv)

### `query_rewriter.py` (Extension 4)

```python
@dataclass
class RewriteResult:
    rewrite: str
    paraphrases: list[str]
    @property
    def all_queries(self) -> list[str]   # [rewrite, *paraphrases]

rewrite(user_turn, history=(), num_paraphrases=2) -> RewriteResult
arewrite(...)                                                # async

rrf_fuse(ranked_lists, k=60) -> list                         # Reciprocal Rank Fusion helper
```

History is `Sequence[tuple[str, str]]` where role ∈ `{"user", "bot"}`. The LLM is prompted to emit strict JSON; output is parsed with a JSON regex fallback. Temperature is 0.0.

### `prop_chunker.py` (Extension 1)

```python
chunk(title, text) -> list[str]
achunk(title, text) -> list[str]
chunk_many(items, concurrency=16, progress=True) -> list[list[str]]   # sync wrapper around asyncio.run
achunk_many(items, concurrency=16, progress=True) -> list[list[str]]
```

- **Cache**: `data/prop_cache.jsonl`. Key = `sha1(title + "\x00" + text)`. Cache hits skip the LLM call. Cache is append-only and thread-safe via a module-level lock.
- LLM is prompted: *"Substitute the title for every pronoun/implicit subject."*
- If JSON parsing returns empty, the chunker falls back to `[text]` (one "proposition" = the full paragraph) so the pipeline never loses content.

### `dense_retriever.py` (Stage 1 + Ext 1 collapse)

```python
@dataclass
class Hit:
    parent_idx: int
    title: str
    text: str           # parent paragraph (NOT proposition)
    best_prop: str      # the proposition that won for this parent
    score: float        # cosine, [0, 1]

class DenseRetriever:
    parents:      list[dict]      # [{title, text}]
    prop_texts:   list[str]       # one per proposition row
    prop_parent:  list[int]       # parent index for each proposition
    embeddings:   np.ndarray      # (N, 4096) float32, L2-normalized

    abuild(parents: Sequence[tuple[title, text]], concurrency=16) -> None
    build(parents, concurrency=16) -> None              # asyncio.run wrapper

    save(index_dir=INDEX_DIR) -> None
    load(index_dir=INDEX_DIR) -> None

    aretrieve(queries, top_k_parents=5, top_n_props=50) -> list[Hit]
    retrieve(queries,  top_k_parents=5, top_n_props=50) -> list[Hit]

index_exists(index_dir=INDEX_DIR) -> bool
INDEX_DIR = data/dense_index
```

**Retrieval algorithm** (the Ext 1 trick):
1. Embed all query variants in one batched call → `(Q, 4096)`.
2. Compute `Q × N` cosine scores via `query_vecs @ embeddings.T`.
3. Per query, take top `top_n_props` propositions.
4. **Max-pool**: for each proposition id, keep its best score across queries.
5. **Collapse**: for each parent paragraph, keep the highest-scoring proposition.
6. Return top `top_k_parents` parents.

This is intentionally NOT a vector DB. For ~500 props the numpy matmul is faster than any external store. When scaling to 185k parents (≈ 1–2M props) the proposal says to switch to Chroma.

### `reranker_call.py` (raw HTTP)

```python
RERANKER_URL = "https://e72a-34-6-49-208.ngrok-free.app/v1/rerank"  # ROTATES — see §11
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

rerank(query, documents, instruction=DEFAULT_INSTRUCTION, top_k=None, url=RERANKER_URL)
    -> [{"index": int, "document": str, "score": float}, ...]   # sorted desc
```

The payload wraps the inputs in Qwen3-Reranker's `<|im_start|>`/`<|im_end|>` template (prefix/suffix module-level constants). The endpoint expects the *whole* prompt as `query` and a list of `<Document>: …` strings as `documents`. **Do not call this directly from pipeline code — use `reranker.py`.**

### `reranker.py` (Extension 2)

```python
@dataclass
class RerankedHit:
    parent_idx: int
    title: str
    text: str
    best_prop: str
    dense_score: float
    rerank_score: float

rerank_hits(query: str, hits: Sequence[Hit], top_k=5,
            instruction=DEFAULT_INSTRUCTION) -> list[RerankedHit]
arerank_hits(...)        # uses asyncio.to_thread to avoid blocking the loop
```

- **Single canonical query**: paraphrases are used by the *dense* stage for recall; the reranker uses *only the rewrite* for precision.
- `dense_score` is preserved on the output so downstream code (and the UI) can show both signals.

### `generator.py` (placeholder for GenKS + KEDiT)

```python
@dataclass
class GenerationResult:
    answer: str
    used_passages: list[RerankedHit]
    prompt: str               # full LLM prompt, exposed for UI transparency

generate(user_turn, history, passages, temperature=0.2) -> GenerationResult
agenerate(...)
```

System prompt enforces: (1) use only provided passages, (2) say so plainly if missing, (3) 1–3 sentence conversational tone, (4) don't mention passage numbers in the answer. **Replace this entire module when GenKS+KEDiT land.**

### `server.py` (FastAPI)

```
GET  /api/info               -> InfoResponse  { parents, propositions, index_dir }
POST /api/run                -> RunResponse
     body: { user_turn, history: [{role,text}], settings }
```

Settings schema (`pydantic.BaseModel`, all bounded):

| field | type | default | range | meaning |
|---|---|---|---|---|
| `dense_pool` | int | 10 | 1–50 | top-N parents fed to reranker |
| `top_n_props` | int | 20 | 1–100 | propositions per query variant |
| `rerank_k` | int | 3 | 1–10 | top-K kept after reranking |
| `n_paraphrases` | int | 2 | 0–4 | extra query variants from the rewriter |
| `gen_temperature` | float | 0.2 | 0.0–1.0 | LLM temperature for the answer |

`RunResponse` fields:

```ts
{
  rewrite:     { rewrite: string, paraphrases: string[] }
  dense:       { rank, title, text, best_prop, dense_score }[]
  reranked:    { rank, title, text, best_prop, dense_score, rerank_score }[]
  answer:      string
  used_titles: string[]
  prompt:      string                  // full LLM prompt for the answer
  timings_ms:  { rewrite, dense, rerank, generate }
}
```

The retriever is loaded **once** on first request and held in a module-level singleton (`_retriever`).

### `build_corpus.py`

One-shot script. Reads the five WoW JSON dumps in `data/`, walks every dialogue + every turn, extracts `(title, paragraph)` from both `chosen_topic_passage` and each turn's `retrieved_passages`, deduplicates exact `(title, text)` matches, writes `data/corpus.jsonl` (185,503 lines after dedup; 1,438,872 raw passages before dedup).

### `retrieve.py`

**Independent BM25 baseline**, not used by the main pipeline. Reads `data/corpus.jsonl`, tokenizes with a simple `[A-Za-z0-9]+` regex, builds a `BM25Okapi` index in RAM (≈ 8 s for 185k passages), and serves an interactive prompt or one-shot CLI query. Useful for sanity-checking retrieval quality independent of embeddings.

```
python3 retrieve.py                   # interactive
python3 retrieve.py "your question"   # one-shot
```

### `app.py` (legacy Streamlit UI)

Mirrors `ui/` but in Streamlit. **Do not add features here** — keep parity-or-deprecate. Run with `streamlit run app.py` on port 8501.

### `test_reranker.py`

Ad-hoc probe script for the reranker endpoint. Not a pytest suite. Tests a few canned (query, docs) pairs and prints scores.

### `build_progress_report.py`

Generates `progress_report.docx` using python-docx. Self-contained; rerun whenever there's new progress to capture.

---

## 6. External services

| Service | Model | Where | Auth |
|---|---|---|---|
| LLM | `gpt-oss-120b` | `https://mia.csp.kloudeks.com/v1` (Kloudeks, class-provided) | `KKB_API` |
| Embedding | `qwen3-embedding-8b` (4096-dim) | same URL, same key | `KKB_API` |
| Reranker | Qwen3-Reranker | ngrok URL in `reranker_call.py` | none (rotates) |

The first two share an OpenAI-compatible API. The reranker is a custom HTTP service.

---

## 7. Data layout (`data/`)

| File | Size | Schema | Origin |
|---|---|---|---|
| `wizard_of_wikipedia.tgz` | 927 MB | tar.gz archive | Downloaded from `https://dl.fbaipublicfiles.com/parlai/wizard_of_wikipedia/wizard_of_wikipedia.tgz` |
| `train.json` | 951 MB | WoW dialogues (list) | extracted from tgz |
| `valid_random_split.json` | 130 MB | same | extracted |
| `test_random_split.json` | 126 MB | same | extracted |
| `valid_topic_split.json` | 128 MB | same | extracted |
| `test_topic_split.json` | 126 MB | same | extracted |
| `data.json` | 1.4 GB | combined | extracted |
| `topic_splits.json` | 22 KB | seen/unseen topic lists | extracted |
| `corpus.jsonl` | 138 MB | `{"title": str, "text": str}` per line | `build_corpus.py` |
| `prop_cache.jsonl` | grows | `{"key": sha1, "props": list[str]}` per line | `prop_chunker.py` |
| `dense_index/props.npy` | (N, 4096) float32 | L2-normalized embeddings | `DenseRetriever.save` |
| `dense_index/props.jsonl` | `{"text": str, "parent": int}` per line | propositions | same |
| `dense_index/parents.jsonl` | `{"title": str, "text": str}` per line | parent paragraphs | same |

### WoW dialogue JSON shape

```json
{
  "chosen_topic": "Blue",
  "persona": "my favorite color is blue.",
  "chosen_topic_passage": ["sentence 1", "sentence 2", ...],
  "wizard_eval": ...,
  "dialog": [
    {
      "speaker": "1_Apprentice" | "0_Wizard",
      "text": "...",
      "candidate_responses": [...],
      "retrieved_passages": [
        { "Title A": ["sent 1", "sent 2", ...] },
        { "Title B": [...] }
      ],
      "retrieved_topics": [...]
    },
    ...
  ]
}
```

`build_corpus.py` extracts `(title, " ".join(sentences))` from `chosen_topic_passage` (using `chosen_topic` as the title) and from each dict inside every turn's `retrieved_passages`.

### Current dense index scale

**Important**: the persisted `dense_index/` covers **40 parent paragraphs / 457 propositions** — just enough to demo the pipeline. The full 185k-paragraph build is not done yet (the proposal scales to that before evaluation). Scaling up needs:
1. More LLM chunker calls (≈ 185k requests; the cache makes this idempotent across runs).
2. More embedding calls (≈ 1–2M propositions).
3. Probably moving from a single `(N, 4096)` numpy array to Chroma.

---

## 8. Frontend (`ui/`)

Vite + React 18 + TypeScript + TailwindCSS. No router, single page.

### Key files

| File | Role |
|---|---|
| `index.html` | imports Inter font from `rsms.me`; mounts `<App>` |
| `src/main.tsx` | React entry |
| `src/index.css` | Tailwind layers + custom components (`.pill`, `.card`, `.stage-h`, `.kbd`) |
| `src/api.ts` | `Settings`, `Turn`, `RunResponse` types + `getInfo()`, `runPipeline()` |
| `src/examples.ts` | clickable example presets (curated to land in the 40-paragraph subset) |
| `src/App.tsx` | layout, state (`history`, `userTurn`, `settings`, `result`), submit handler |
| `src/components/Sidebar.tsx` | index stats + sliders |
| `src/components/Examples.tsx` | 4-card grid of clickable presets |
| `src/components/HistoryEditor.tsx` | collapsible turn-by-turn editor with role dropdown |
| `src/components/Trace.tsx` | 4-stage pipeline trace (rewrite / dense table / reranked cards / answer card) |

### Dev server

```
vite     :5173          (proxies /api/*  ->  http://127.0.0.1:8000)
fastapi  :8000
```

Defined in `ui/vite.config.ts`. The browser only sees `:5173`; there are no CORS concerns.

### Theme

Light. Custom Tailwind palette in `tailwind.config.js`:

- `ink` 50–900 (neutral grays, body text)
- `accent` = `#4F46E5` (indigo), with `soft` (`#EEF2FF`) and `ring` (`#C7D2FE`) variants

Fonts: Inter (loaded from CDN), Consolas/JetBrains Mono for code.

### Keyboard shortcut

`Enter` or `Cmd/Ctrl+Enter` in the composer triggers the pipeline (only when the input is non-empty).

---

## 9. Running the stack

### Environment

```bash
# .env
KKB_API=<class-provided key>
```

### Python deps (already installed in the user's interpreter)

```
openai
python-dotenv
numpy
rank_bm25            # only for retrieve.py BM25 baseline
fastapi
uvicorn
pydantic
python-docx          # only for build_progress_report.py
streamlit            # only for legacy app.py
pandas               # only for legacy app.py
```

### Frontend deps

```bash
cd ui && npm install
```

(react, react-dom, vite, @vitejs/plugin-react, tailwindcss, postcss, autoprefixer, typescript, @types/react, @types/react-dom)

### One-shot launcher

```bash
./start.sh
```

Boots `uvicorn server:app --port 8000` and `vite` (port 5173) in parallel, kills both on Ctrl-C. Open **http://localhost:5173**.

### Run individually

```bash
# backend
uvicorn server:app --port 8000 --reload

# frontend (in another shell)
cd ui && npm run dev
```

### Legacy Streamlit

```bash
streamlit run app.py            # http://localhost:8501
```

---

## 10. Sidebar settings — full reference

What each slider in the UI sidebar actually controls:

| Slider | Default | Range | What changes |
|---|---|---|---|
| Dense pool size | 10 | 5–30 | parents handed from dense retriever to reranker (`top_k_parents`) |
| Props per query variant | 20 | 5–50 | propositions pulled per query variant before parent collapse (`top_n_props`) |
| Reranked top-K | 3 | 1–10 | final passages fed to the LLM (`top_k` for reranker, `len(passages)` for generator) |
| Paraphrases | 2 | 0–4 | extra query variants generated by the rewriter (in addition to the canonical rewrite) |
| Answer temperature | 0.20 | 0.0–1.0 | LLM sampling temperature for Stage 4 only |

Composition: `n_total_queries = 1 + n_paraphrases`. Total proposition lookups ≈ `n_total_queries × top_n_props`. Final LLM context ≈ `rerank_k` parent paragraphs + history.

---

## 11. Things to be careful about

1. **The reranker URL rotates often.** It's an ngrok tunnel, replaced by the user every few hours/days. If the reranker stage 500s, check `reranker_call.py:RERANKER_URL` and update it. After updating, **restart the FastAPI backend** (the module is imported once; you can't hot-swap the constant via uvicorn `--reload` alone in some edge cases — kill and re-run to be safe).
2. **The dense index is currently tiny (40 parents).** Queries about topics outside science-fiction / Harry Potter will return nothing useful. The example queries in `ui/src/examples.ts` are deliberately chosen to land in the loaded subset. When scaling up, expect the chunker cache + embeddings to grow `data/` substantially.
3. **`prop_cache.jsonl` is append-only.** If the chunking prompt changes, **don't trust cached entries** — delete the cache file or version the key.
4. **`gpt-oss-120b` JSON parsing is best-effort.** Both `query_rewriter` and `prop_chunker` accept JSON wrapped in code fences and fall back gracefully (rewriter raises; chunker uses `[text]`). If you change prompts, re-test the JSON extraction.
5. **The placeholder generator is NOT GenKS.** It does plain RAG. Selection accuracy (the proposal's metric) is meaningless against this. Real GenKS implementation comes later.
6. **`nlp/`, `nlp.zip`, `~$ogress_report.docx`** are export artifacts / Word lock files. Ignore them.
7. **The Streamlit app is frozen.** Only edit `ui/` for new features. Streamlit stays as a fallback.
8. **OpenAI client `temperature=0.0`** is used everywhere except the answer stage. Do not crank it up for the rewriter or chunker — JSON parsing breaks.

---

## 12. Conventions used in the code

- **Dual sync/async API**: every module exposes `foo(...)` and `afoo(...)`. The async one is the canonical implementation; the sync one is usually `asyncio.run(afoo(...))`.
- **Concurrency**: `api_clients.gather_bounded(awaitables, concurrency=16, on_done=cb)` is the shared semaphore-gated `asyncio.gather`. Optional `on_done(n_completed)` callback for progress.
- **Caching**: only the proposition chunker caches (the embedding API is fast enough to not bother). Cache lives in `data/prop_cache.jsonl`. Pure function from `(title, text)` to `list[str]`.
- **Data classes**: `RewriteResult`, `Hit`, `RerankedHit`, `GenerationResult`. All immutable in spirit; mutation happens at composition boundaries.
- **Single orchestration point**: `server.py:run` is the only function that runs the full pipeline. The pipeline shape lives there; modules don't reach across each other.
- **Type hints**: PEP 604 unions (`int | None`), `Sequence[...]`, etc. Python 3.11+ required.

---

## 13. Status vs proposal

| Proposal item | Status | Owning module |
|---|---|---|
| **Ext 4** Query rewriter | Done | `query_rewriter.py` |
| **Stage 1** Dense retrieval | Done (over 40 parents) | `dense_retriever.py` |
| **Ext 1** Propositional chunking | Done | `prop_chunker.py` (+ collapse in `dense_retriever.py`) |
| **Ext 2** Cross-encoder reranker | Done | `reranker.py` (+ `reranker_call.py`) |
| **Stage 2** GenKS | Not started | — (placeholder: `generator.py`) |
| **Stage 3** KEDiT (Q-Former + KA-Adapter) | Not started | — (placeholder: `generator.py`) |
| **Stage 4** FLARE | Not started | — |
| **Stage 5** Q² + BEGIN | Not started | — |
| **Ext 3** Per-dialogue cache | Not started | — |
| **Ext 5** Turkish evaluation | Not started (time-permitting) | — |
| Streamlit UI | Done, frozen | `app.py` |
| Node UI | Done, canonical | `ui/` + `server.py` |
| Progress report | Done | `progress_report.docx` |

---

## 14. Where to extend

When adding a new pipeline stage, the typical edits are:

1. **New module** (e.g., `genks.py`) with sync + async entry points and a `dataclass` result.
2. **Add a call to it** in `server.py:run`, between the existing stages.
3. **Extend the response schema** in `server.py` (`RunResponse` and any new `*Out` class).
4. **Extend the frontend types** in `ui/src/api.ts:RunResponse`.
5. **Render the stage** in `ui/src/components/Trace.tsx` (one `<section>` mirroring the existing pattern).
6. **Add any new setting** to `Settings` in both `server.py` and `ui/src/api.ts`, and to `ui/src/components/Sidebar.tsx`.

For replacing the placeholder generator with GenKS + KEDiT, the simplest path is to leave `generator.py`'s signature in place and have `generate()` internally call GenKS first, then KEDiT. That keeps `server.py` and the UI unchanged.

---

## 15. Cited papers (in `papers/`)

| File | Citation | Used for |
|---|---|---|
| `papers/1.pdf` | Dziri et al., 2022 | BEGIN (Stage 5) |
| `papers/3.pdf` | Jiang et al., 2023 | FLARE (Stage 4) |
| `papers/4.pdf` | Sun et al., 2023 | GenKS (Stage 2) |
| `papers/5.pdf` | Zhang et al., 2025 | KEDiT (Stage 3) |
| `papers/6.pdf` | Honovich et al., 2021 | Q² (Stage 5) |

See `paper_references.md` for the full mapping.
