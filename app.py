"""Streamlit GUI for the RAG pipeline.

Pipeline:
    user turn + history
        -> [Ext 4]  query rewriter        : rewrite + paraphrases
        -> [Stage 1 + Ext 1] dense_retriever : top-N parents (proposition index)
        -> [Ext 2]  cross-encoder reranker  : top-K parents
        -> generator                        : gpt-oss-120b grounded answer
                                              (placeholder for GenKS+KEDiT)

Run:
    streamlit run app.py
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from dense_retriever import DenseRetriever, INDEX_DIR, index_exists
from generator import generate as generate_answer
from query_rewriter import rewrite as rewrite_sync
from reranker import rerank_hits

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WoW RAG Pipeline",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light CSS polish — tighter cards, monospace for prop ids, hide footer.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }
      footer { visibility: hidden; }
      .stButton > button {
          border-radius: 8px;
          font-weight: 500;
          padding: 0.45rem 0.9rem;
      }
      .example-btn .stButton > button {
          width: 100%;
          text-align: left;
          white-space: normal;
          line-height: 1.3;
          background: #F4F5F7;
          border: 1px solid #E1E4E8;
          color: #1A1A1A;
      }
      .example-btn .stButton > button:hover {
          background: #ECEEF1;
          border-color: #C8CCD1;
      }
      .pill {
          display: inline-block;
          background: #EEF2FF;
          color: #1E3A8A;
          border: 1px solid #C7D2FE;
          padding: 2px 10px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 500;
          margin-right: 6px;
      }
      .stage-h {
          font-size: 0.95rem;
          font-weight: 600;
          color: #4B5563;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          margin: 0 0 0.4rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Examples (chosen to land in the 40-paragraph subset: sci-fi + Harry Potter)
# ---------------------------------------------------------------------------

@dataclass
class Example:
    label: str
    blurb: str
    history: list[tuple[str, str]]
    turn: str


EXAMPLES: list[Example] = [
    Example(
        label="Coreference",
        blurb='Resolve "he" using prior turns.',
        history=[
            ("user", "I just finished a book by Isaac Asimov."),
            ("bot", "He was a prolific science fiction author."),
        ],
        turn="What is he famous for?",
    ),
    Example(
        label="Direct lookup",
        blurb="Specific entity, no history.",
        history=[],
        turn="Tell me about the Ministry of Magic in Harry Potter.",
    ),
    Example(
        label="Topic question",
        blurb="Broad genre overview.",
        history=[],
        turn="What is the history of science fiction?",
    ),
    Example(
        label="Theme question",
        blurb="Concept across multiple articles.",
        history=[],
        turn="How is time travel depicted in fiction?",
    ),
]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading dense index...")
def get_retriever() -> DenseRetriever | None:
    if not index_exists():
        return None
    r = DenseRetriever()
    r.load()
    return r


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### WoW RAG Pipeline")
    st.caption("Knowledge-grounded dialogue over Wizard of Wikipedia.")
    st.markdown(
        '<span class="pill">Ext 4 rewrite</span>'
        '<span class="pill">Stage 1 dense</span>'
        '<span class="pill">Ext 1 props</span>'
        '<span class="pill">Ext 2 rerank</span>'
        '<span class="pill">LLM answer</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    retriever = get_retriever()
    if retriever is None:
        st.error(
            "No dense index found.\n\n"
            f"Expected at `{INDEX_DIR}`.\n\n"
            "Run `python3 demo.py` once to build it."
        )
    else:
        st.markdown("**Index**")
        c1, c2 = st.columns(2)
        c1.metric("Parents", f"{len(retriever.parents):,}")
        c2.metric("Propositions", f"{len(retriever.prop_texts):,}")

    st.divider()
    st.markdown("**Settings**")
    dense_pool = st.slider("Dense pool size", 5, 30, 10, step=1,
                           help="How many parents the dense retriever passes to the reranker.")
    top_n_props = st.slider("Props per query variant", 5, 50, 20, step=5)
    rerank_k = st.slider("Reranked top-K", 1, 10, 3, step=1,
                         help="Final number of passages shown after reranking.")
    n_paraphrases = st.slider("Paraphrases", 0, 4, 2, step=1,
                              help="Extra query variants generated by the rewriter.")
    gen_temperature = st.slider("Answer temperature", 0.0, 1.0, 0.2, step=0.05,
                                help="LLM sampling temperature for the final grounded answer.")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "user_turn" not in st.session_state:
    st.session_state.user_turn = ""
if "history_text" not in st.session_state:
    st.session_state.history_text = ""
if "auto_run" not in st.session_state:
    st.session_state.auto_run = False


def _format_history(h: list[tuple[str, str]]) -> str:
    return "\n".join(f"{role}: {text}" for role, text in h)


def _parse_history(raw: str) -> list[tuple[str, str]]:
    out = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            role, text = line.split(":", 1)
            role = role.strip().lower()
            if role not in {"user", "bot"}:
                role = "user"
            out.append((role, text.strip()))
        else:
            out.append(("user", line))
    return out


def apply_example(ex: Example) -> None:
    st.session_state.user_turn = ex.turn
    st.session_state.history_text = _format_history(ex.history)
    st.session_state.auto_run = True


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("RAG Pipeline Tester")
st.caption(
    "Type a question or pick an example, then watch each stage of the pipeline. "
    "All retrieval is over a 40-paragraph slice of Wizard of Wikipedia."
)

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

st.markdown('<p class="stage-h">Try an example</p>', unsafe_allow_html=True)
ex_cols = st.columns(len(EXAMPLES))
for col, ex in zip(ex_cols, EXAMPLES):
    with col:
        st.markdown('<div class="example-btn">', unsafe_allow_html=True)
        clicked = st.button(
            f"**{ex.label}**\n\n{ex.blurb}\n\n_{ex.turn}_",
            key=f"ex_{ex.label}",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        if clicked:
            apply_example(ex)
            st.rerun()

st.write("")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

with st.expander("Dialogue history (optional)", expanded=bool(st.session_state.history_text)):
    st.text_area(
        "One turn per line. Prefix with `user:` or `bot:` to label speakers.",
        key="history_text",
        height=110,
        placeholder="user: I just finished a book by Isaac Asimov.\nbot: He was a prolific science fiction author.",
    )

st.text_input(
    "Your turn",
    key="user_turn",
    placeholder="Ask anything grounded in the loaded passages...",
)

run_clicked = st.button("Run pipeline", type="primary")

if st.session_state.auto_run:
    run_clicked = True
    st.session_state.auto_run = False


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(user_turn: str, history: list[tuple[str, str]]) -> None:
    if retriever is None:
        st.error("Dense index is not loaded. Build it first via `python3 demo.py`.")
        return

    # --- Stage: Query rewrite -------------------------------------------------
    st.markdown('<p class="stage-h">Stage 1 · Query rewrite</p>', unsafe_allow_html=True)
    with st.spinner("Rewriting..."):
        rw = rewrite_sync(user_turn, history=history, num_paraphrases=n_paraphrases)
    with st.container(border=True):
        st.markdown(f"**Rewrite** &nbsp; `{rw.rewrite}`")
        if rw.paraphrases:
            for i, p in enumerate(rw.paraphrases, 1):
                st.markdown(f"_Paraphrase {i}_ &nbsp; `{p}`")
        else:
            st.caption("No paraphrases requested.")

    # --- Stage: Dense retrieval ----------------------------------------------
    st.markdown(
        f'<p class="stage-h">Stage 2 · Dense retrieval (pool top-{dense_pool})</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Embedding queries and searching..."):
        hits = retriever.retrieve(rw.all_queries, top_k_parents=dense_pool, top_n_props=top_n_props)
    if not hits:
        st.warning("Dense retriever returned no hits.")
        return
    df = pd.DataFrame(
        [{"rank": i + 1, "dense": round(h.score, 3), "title": h.title, "best proposition": h.best_prop}
         for i, h in enumerate(hits)]
    )
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "rank": st.column_config.NumberColumn(width="small"),
            "dense": st.column_config.NumberColumn(width="small", format="%.3f"),
            "title": st.column_config.TextColumn(width="medium"),
            "best proposition": st.column_config.TextColumn(width="large"),
        },
    )

    # --- Stage: Reranker ------------------------------------------------------
    st.markdown(
        f'<p class="stage-h">Stage 3 · Cross-encoder reranker (top-{rerank_k})</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Reranking..."):
        reranked = rerank_hits(rw.rewrite, hits, top_k=rerank_k)
    if not reranked:
        st.warning("Reranker returned nothing.")
        return
    for i, h in enumerate(reranked, 1):
        with st.container(border=True):
            top, _ = st.columns([0.7, 0.3])
            with top:
                st.markdown(f"**[{i}] {h.title}**")
            cols = st.columns(2)
            cols[0].metric("Reranker", f"{h.rerank_score:.4f}")
            cols[1].metric("Dense (cosine)", f"{h.dense_score:.3f}")
            st.caption(f"best proposition: {h.best_prop}")
            st.markdown(h.text)

    # --- Stage: Grounded answer ----------------------------------------------
    st.markdown(
        '<p class="stage-h">Stage 4 · Grounded answer (gpt-oss-120b)</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Generating answer..."):
        result = generate_answer(user_turn, history, reranked, temperature=gen_temperature)
    with st.container(border=True):
        st.markdown(result.answer)
        st.caption(
            f"Grounded in top-{len(result.used_passages)} reranked passage(s): "
            + ", ".join(h.title for h in result.used_passages)
        )
    with st.expander("Show full prompt sent to the LLM"):
        st.code(result.prompt, language="text")


if run_clicked:
    turn = st.session_state.user_turn.strip()
    history = _parse_history(st.session_state.history_text)
    if not turn:
        st.warning("Type a user turn or pick an example.")
    else:
        st.divider()
        run_pipeline(turn, history)
