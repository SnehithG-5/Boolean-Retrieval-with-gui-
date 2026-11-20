# Boolean Retrieval Lab

Upload documents (PDF / Word / TXT), see NLTK pre-processing, the term-document incidence matrix,
the inverted index, and run Boolean (AND / OR / NOT, brackets) queries.

| File | Role |
|---|---|
| `ir_backend.py` | Backend: pre-processing, incidence matrix, inverted index, query parser (no GUI code) |
| `app.py` | GUI: Streamlit front end that calls the backend |
| `requirements.txt` | Libraries to install |
| `sample_docs/` | Small AI/ML corpus to try (the app also has a "use sample documents" tickbox) |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Host it free so you can open it anywhere
GitHub Pages cannot run Python, so use Streamlit Community Cloud (free, deploys straight from GitHub):
1. Create a GitHub repo and upload these files (keep `app.py` and `ir_backend.py` in the repo root).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repo, branch `main`, main file `app.py`, then **Deploy**.
4. You get a public URL like `https://your-app-name.streamlit.app`.

NLTK data (punkt, stopwords, wordnet) downloads automatically on first start.
The free app sleeps after a period of inactivity; opening the link wakes it in about a minute.

## Query syntax
`learning AND NOT unsupervised` · `(neural OR language) AND learning` · `NOT vision`
Precedence: NOT, then AND, then OR. Query words are normalized the same way as the documents.
