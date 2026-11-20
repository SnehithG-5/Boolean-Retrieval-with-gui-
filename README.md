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

## Query syntax
`learning AND NOT unsupervised` · `(neural OR language) AND learning` · `NOT vision`
Precedence: NOT, then AND, then OR. Query words are normalized the same way as the documents.
