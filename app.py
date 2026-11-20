"""
app.py  -  GUI (Streamlit).  All IR logic lives in ir_backend.py.
Run locally:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

import ir_backend as ir

st.set_page_config(page_title="Boolean Retrieval Lab", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner="Pre-processing documents...")
def load_corpus(file_items, mode):
    """file_items: tuple of (name, content) so Streamlit can cache on the content."""
    return ir.build_corpus(dict(file_items), mode)


def snippet(text, n=220):
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🔎 Boolean Retrieval Lab")
st.sidebar.caption("Upload documents, build the indexes, run Boolean queries.")

uploads = st.sidebar.file_uploader(
    "Upload your corpus (PDF, Word, TXT)", type=[e.lstrip(".") for e in ir.SUPPORTED],
    accept_multiple_files=True,
)
use_sample = st.sidebar.checkbox("Use sample AI/ML documents", value=not uploads,
                                 help="Tick to try the app without uploading anything.")
mode_label = st.sidebar.radio("Normalization", ["Lemmatization", "Stemming"],
                              help="Lemmatization keeps real words (models -> model). "
                                   "Stemming chops suffixes (machine -> machin).")
mode = "lemma" if mode_label == "Lemmatization" else "stem"

if not ir.NLTK_OK:
    st.sidebar.warning("NLTK data could not be downloaded - using built-in fallback "
                       "tokenizer/stop-words and stemming.")

files = {u.name: u.getvalue() for u in uploads} if uploads else {}
if use_sample:
    files.update(ir.SAMPLE_DOCS)

if not files:
    st.title("Boolean Retrieval Lab")
    st.info("👈 Upload two or more documents in the sidebar (or tick *Use sample documents*) to begin.")
    st.stop()

raw, tokens, skipped = load_corpus(tuple(files.items()), mode)
for name, why in skipped:
    st.sidebar.error(f"Skipped {name}: {why}")
if not tokens:
    st.error("None of the documents contained readable text.")
    st.stop()

matrix = ir.IncidenceMatrix(tokens, mode)
index = ir.InvertedIndex(tokens, mode)

st.sidebar.success(f"{len(tokens)} documents · {len(matrix.terms)} unique terms")

# ---------------------------------------------------------------- tabs
st.title("Boolean Retrieval Lab")
t_pre, t_mat, t_inv, t_query = st.tabs(
    ["1 · Pre-processing", "2 · Incidence matrix", "3 · Inverted index", "4 · Query"]
)

# ---- Q1: pre-processing
with t_pre:
    st.subheader("Text pre-processing")
    doc = st.selectbox("Document", list(raw))
    steps = ir.preprocess_steps(raw[doc], mode)
    st.caption("Original text")
    st.text(snippet(raw[doc], 600))
    for title, val in steps.items():
        with st.expander(f"{title}  ({len(val) if isinstance(val, list) else len(val.split())} items)",
                         expanded=title.startswith("5")):
            st.write(val if isinstance(val, list) else snippet(val, 600))

# ---- Q2a: incidence matrix
with t_mat:
    st.subheader("Term-document incidence matrix")
    df = pd.DataFrame(matrix.M, index=matrix.terms, columns=matrix.doc_ids)
    find = st.text_input("Filter terms (optional)", key="mat_filter")
    if find:
        df = df[df.index.str.contains(find.lower().strip())]
    st.dataframe(df, width="stretch", height=420)
    st.download_button("Download matrix (CSV)", df.to_csv().encode(), "incidence_matrix.csv")

# ---- Q2b: inverted index
with t_inv:
    st.subheader("Inverted index (term → postings list)")
    inv = pd.DataFrame(
        [(t, len(p), ", ".join(p)) for t, p in index.postings.items()],
        columns=["term", "doc freq", "postings list"],
    )
    find = st.text_input("Filter terms (optional)", key="inv_filter")
    if find:
        inv = inv[inv["term"].str.contains(find.lower().strip())]
    st.dataframe(inv, width="stretch", height=420, hide_index=True)
    st.download_button("Download index (CSV)", inv.to_csv(index=False).encode(), "inverted_index.csv")

# ---- Query processing
with t_query:
    st.subheader("Boolean query processing")
    st.caption("Operators: **AND**, **OR**, **NOT** (capitals or lowercase) and ( ) brackets. "
               "Precedence: NOT, then AND, then OR.  Terms are normalized the same way as the documents.")
    query = st.text_input("Enter your query", placeholder="e.g.  learning AND NOT unsupervised")

    if query.strip():
        try:
            vec = matrix.query_vector(query)
            hits_matrix = [d for d, b in zip(matrix.doc_ids, vec) if b]
            hits_index = index.query(query)
        except ir.QueryError as e:
            st.error(f"Invalid query: {e}")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Incidence matrix → matches", len(hits_matrix))
            c2.metric("Inverted index → matches", len(hits_index))
            if hits_matrix == hits_index:
                st.success("Both methods return the same documents ✔")
            else:
                st.error("Methods disagree - please report this as a bug.")

            unknown = [t for t in ir.query_terms(query) if ir.normalize_term(t, mode) not in matrix.row]
            if unknown:
                st.warning("Not found in any document: " + ", ".join(f"`{t}`" for t in unknown))

            with st.expander("How the incidence matrix computed it (bit vectors)", expanded=True):
                rows = {t: matrix.vector(t) for t in dict.fromkeys(ir.query_terms(query))}
                rows["RESULT"] = vec
                st.dataframe(pd.DataFrame(rows, index=matrix.doc_ids).T, width="stretch")
            with st.expander("Postings lists used by the inverted index"):
                for t in dict.fromkeys(ir.query_terms(query)):
                    st.write(f"**{ir.normalize_term(t, mode)}** → {index.get(t)}")

            st.markdown("#### Matching documents")
            if not hits_index:
                st.info("No document matches this query.")
            for d in hits_index:
                with st.container(border=True):
                    st.markdown(f"**{d}**")
                    st.caption(snippet(raw[d]))
