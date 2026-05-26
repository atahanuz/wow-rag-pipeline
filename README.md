# WoW RAG Pipeline

A retrieval-augmented generation system for knowledge-grounded dialogue over the
Wizard of Wikipedia (WoW) dataset. Course project for **CMPE 58T — Advanced NLP** at
Bogazici University, by **Fatma Gizem Yılmaz** and **Atahan Uz**.

Given a user's dialogue turn (plus optional prior turns), the system retrieves the
most relevant Wikipedia passage, picks the best part of it, generates a response
grounded in that part, and verifies the result is actually faithful to the source.

The final system paper is in [sys_paper/system_paper.pdf](sys_paper/system_paper.pdf).

---

## Pipeline overview

```
user turn + history
    -> query rewriter (coreference + paraphrases)
    -> per-dialogue cache (skip retrieval on same-topic follow-ups)
    -> dense retrieval over proposition-level index
    -> cross-encoder reranker
    -> GenKS (LLM picks one passage)
    -> KEDiT (LLM distills passage into a query-conditioned brief)
    -> LLM generator (grounded answer)
    -> FLARE grounding check (re-retrieve + regenerate if unsupported claims)
    -> Q² + BEGIN faithfulness gate (regenerate with runner-up if either fails)
    -> final grounded answer
```

Where the original papers (GenKS, KEDiT, FLARE, Q², BEGIN) train task-specific
small classifiers (BART, T5, Albert-XL, RoBERTa), this implementation uses zero-shot
prompts to a modern instruction-tuned LLM, so the full pipeline runs without any
local GPU training.

---

## Repository layout

```
api_clients.py        sync + async OpenAI-compatible clients (LLM + embeddings)
build_corpus.py       extract a deduplicated passage corpus from raw WoW JSONs
retrieve.py           BM25 baseline retriever (standalone CLI)
query_rewriter.py     coreference rewrite + paraphrases
prop_chunker.py       LLM-driven proposition splitter (cached)
dense_retriever.py    proposition-level dense index + parent-paragraph collapse
reranker_call.py      thin HTTP wrapper for the Qwen3-Reranker endpoint
reranker.py           reranker adapter over the dense retriever's Hit objects
genks.py              LLM-based generative knowledge selection
kedit.py              LLM-based query-conditioned passage distillation
flare.py              LLM-based grounding check for active re-retrieval
faithfulness.py       Q² + BEGIN evaluator (two parallel LLM calls)
generator.py          grounded-answer generator (single LLM call)
dialogue_cache.py     per-dialogue LRU cache of chosen passages
server.py             FastAPI backend exposing /api/run and /api/dialogue/reset
app.py                Streamlit fallback UI
start.sh              one-shot launcher (FastAPI :8000 + Vite :5173)
evaluate_recall.py    4-condition retrieval ablation, writes eval_results.json
eval_results.json     ablation results reported in the paper
ui/                   Vite + React + TS + Tailwind frontend
sys_paper/            final paper PDF
paper_references.md   citation map for the system paper
```

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 20+ (only for the React UI)
- An OpenAI-compatible API endpoint with:
  - a chat model (any reasonable instruction-tuned LLM, e.g. `gpt-oss-120b`)
  - an embedding model (we use `qwen3-embedding-8b`, 4096-dim)
- A cross-encoder reranker reachable over HTTP (we use a hosted Qwen3-Reranker)

### Environment variables

Create a `.env` file in the repo root:

```bash
KKB_API=<your-api-key-for-the-llm-and-embedding-endpoint>
```

The base URL, model names, and embedding dimension are set in
[api_clients.py](api_clients.py). The reranker URL lives in
[reranker_call.py](reranker_call.py) (`RERANKER_URL`). Swap any of these to point
at a different provider.

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Install UI dependencies

```bash
cd ui
npm install
cd ..
```

---

## Data setup

The WoW dataset and the derived index are not in the repo (too large). You build
them locally.

### 1. Download the WoW dataset

```bash
mkdir -p data
cd data
curl -L -O https://dl.fbaipublicfiles.com/parlai/wizard_of_wikipedia/wizard_of_wikipedia.tgz
tar -xzf wizard_of_wikipedia.tgz
cd ..
```

This gives you `data/train.json`, `data/valid_random_split.json`,
`data/valid_topic_split.json`, `data/test_random_split.json`,
`data/test_topic_split.json`, and `data/data.json`.

### 2. Build the deduplicated passage corpus

```bash
python3 build_corpus.py
```

Produces `data/corpus.jsonl` with 185,503 unique (title, paragraph) records.

### 3. Build a small demo dense index (~40 paragraphs)

The repo's first-run path embeds 40 representative parent paragraphs as a
demo index so the UI works end-to-end without waiting hours. You can change
the slice in `evaluate_recall.py` or in your own driver script.

A minimal script to build the demo index:

```python
import json
from dense_retriever import DenseRetriever

with open("data/corpus.jsonl") as f:
    parents = [(rec := json.loads(l))["title"] and (rec["title"], rec["text"]) for l in list(f)[:40]]

r = DenseRetriever()
r.build(parents)
r.save()
```

(Scaling to the full 185,503 paragraphs uses the exact same code path; only
the chunking and embedding loops take longer.)

---

## Running the system

### One command (backend + frontend)

```bash
./start.sh
```

Boots `uvicorn server:app` on `:8000` and the Vite dev server on `:5173`. Open
**http://localhost:5173** in your browser. Hit Ctrl-C to stop both.

### Individually

Backend:

```bash
uvicorn server:app --port 8000 --host 127.0.0.1 --reload
```

Frontend:

```bash
cd ui && npm run dev
```

The Vite config proxies `/api/*` to the FastAPI backend so the browser only
talks to `:5173`.

### Streamlit fallback (legacy)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### CLI: BM25 baseline retriever

```bash
python3 retrieve.py "your question here"
python3 retrieve.py                          # interactive mode
```

### Reproduce the ablation results

```bash
python3 evaluate_recall.py
```

Walks the 21 (apprentice-query, gold-passage) pairs whose gold title is in
the local index and reports Recall@1/@3/@5 under four conditions (paragraph
baseline, paragraph + rerank, proposition (ours), proposition + rerank).
Writes per-query details to `eval_results.json`.

---

## Settings (UI sidebar)

| Slider | Default | Meaning |
|---|---|---|
| Dense pool size | 10 | top-N parents the dense retriever passes to the reranker |
| Props per query variant | 20 | propositions pulled per query variant before parent collapse |
| Reranked top-K | 3 | passages the LLM ultimately sees |
| Paraphrases | 2 | extra query variants generated by the rewriter |
| Answer temperature | 0.2 | LLM sampling temperature for Stage 4 |
| Cache threshold τ | 0.70 | minimum cosine similarity for a cache hit |

---

## Result

A 4-condition retrieval ablation (Section 9 of the paper) on 21 WoW pairs:

| Condition | R@1 | R@3 | R@5 |
|---|---|---|---|
| paragraph              | 0.476 | 0.762 | 0.762 |
| paragraph + rerank     | 0.524 | 0.762 | 0.857 |
| **proposition (ours)** | **0.619** | **0.810** | **0.905** |
| proposition + rerank   | 0.524 | 0.810 | 0.857 |

Proposition-level indexing improves Recall@1 by **14.3 points** over the
paragraph baseline. The reranker helps the weaker paragraph baseline but
slightly hurts on top of the already-strong proposition retriever. See
[sys_paper/system_paper.pdf](sys_paper/system_paper.pdf) for the full
discussion.

---

## License

MIT.
