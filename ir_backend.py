"""
ir_backend.py  -  BACKEND (no GUI code in here)

Text pre-processing with NLTK, Term-Document Incidence Matrix, Inverted Index
and Boolean Retrieval.  Imported by app.py (Streamlit GUI) and usable on its own.

Libraries: nltk, pypdf, python-docx, numpy, re, io, string, collections
"""

import contextlib
import io
import re
import string
from collections import defaultdict

import nltk
import numpy as np
from docx import Document
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from pypdf import PdfReader

SUPPORTED = (".pdf", ".docx", ".txt")


class QueryError(ValueError):
    """Raised for a malformed Boolean query (shown to the user by the GUI)."""


# ------------------------------------------------------------------ NLTK setup
def _setup_nltk():
    """Download NLTK resources quietly; return True only if they all load."""
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                nltk.download(pkg, quiet=True)
        except Exception:
            pass
    try:
        word_tokenize("test sentence")
        stopwords.words("english")
        WordNetLemmatizer().lemmatize("tests")
        return True
    except Exception:
        return False


NLTK_OK = _setup_nltk()
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer() if NLTK_OK else None
_FALLBACK_TOKENIZER = RegexpTokenizer(r"[A-Za-z]+")
_FALLBACK_STOPWORDS = set(
    """a an the and or but if of at by for with about against between into through during before
    after above below to from up down in out on off over under again further then once here there
    when where why how all any both each few more most other some such no nor not only own same so
    than too very s t can will just don should now is are was were be been being have has had having
    do does did doing i me my we our you your he him his she her it its they them their what which
    who whom this that these those am as until while""".split()
)


# ------------------------------------------------------ reading PDF / Word / txt
def extract_text(filename, data: bytes) -> str:
    """Raw text from an uploaded file's bytes (.pdf, .docx or .txt)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:                      # include text inside tables too
            for row in table.rows:
                parts.extend(c.text for c in row.cells)
        return "\n".join(parts)
    return data.decode("utf-8", errors="ignore")


# ------------------------------------------------------------- pre-processing
def preprocess_steps(text, mode="lemma"):
    """Return every stage of the pipeline (for display).  mode: 'lemma' or 'stem'."""
    steps = {}
    lowered = text.lower()
    steps["1. Lowercased text"] = lowered
    if NLTK_OK:
        tokens = word_tokenize(lowered)
        sw = set(stopwords.words("english"))
    else:
        tokens = _FALLBACK_TOKENIZER.tokenize(lowered)
        sw = _FALLBACK_STOPWORDS
    steps["2. Tokens"] = tokens
    tokens = [t for t in tokens if t not in string.punctuation and t.isalpha()]
    steps["3. Punctuation / numbers removed"] = tokens
    tokens = [t for t in tokens if t not in sw]
    steps["4. Stop-words removed"] = tokens
    if mode == "lemma" and _LEMMATIZER:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    else:
        tokens = [_STEMMER.stem(t) for t in tokens]
    steps["5. Normalized tokens"] = tokens
    return steps


def preprocess(text, mode="lemma"):
    return preprocess_steps(text, mode)["5. Normalized tokens"]


def build_corpus(files, mode="lemma"):
    """files: {filename: bytes or str}.  Returns (raw_text, tokens, skipped) dicts/list."""
    raw, tokens, skipped = {}, {}, []
    for name, content in files.items():
        try:
            text = content if isinstance(content, str) else extract_text(name, content)
        except Exception as e:                        # corrupt / encrypted file
            skipped.append((name, f"could not read ({type(e).__name__})"))
            continue
        toks = preprocess(text, mode)
        if toks:
            raw[name], tokens[name] = text, toks
        else:
            skipped.append((name, "no extractable text (scanned PDF?)"))
    return raw, tokens, skipped


# ------------------------------------------------------------ query handling
def normalize_term(term, mode="lemma"):
    out = preprocess(term, mode)
    return out[0] if out else term.lower()


def evaluate(query, get_term, AND, OR, NOT):
    """Recursive-descent Boolean parser.  Precedence: NOT > AND > OR; parentheses allowed."""
    toks = re.findall(r"\(|\)|[A-Za-z0-9_]+", query)
    if not toks:
        raise QueryError("Query is empty.")
    pos = 0

    def peek():
        return toks[pos].upper() if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def parse_or():
        left = parse_and()
        while peek() == "OR":
            take()
            left = OR(left, parse_and())
        return left

    def parse_and():
        left = parse_not()
        while peek() == "AND":
            take()
            left = AND(left, parse_not())
        return left

    def parse_not():
        if peek() == "NOT":
            take()
            return NOT(parse_not())
        return parse_atom()

    def parse_atom():
        p = peek()
        if p is None:
            raise QueryError("Query ends unexpectedly - a term is missing after an operator.")
        if p == "(":
            take()
            v = parse_or()
            if peek() != ")":
                raise QueryError("Missing closing parenthesis ')'.")
            take()
            return v
        if p in (")", "AND", "OR"):
            raise QueryError(f"Unexpected '{toks[pos]}' - a search term was expected here.")
        return get_term(take())

    result = parse_or()
    if pos != len(toks):
        raise QueryError(f"Unexpected '{toks[pos]}' - did you forget AND / OR between terms?")
    return result


def query_terms(query):
    """The search words in a query (operators and brackets removed)."""
    return [t for t in re.findall(r"[A-Za-z0-9_]+", query) if t.upper() not in ("AND", "OR", "NOT")]


# ---------------------------------------------- Term-Document Incidence Matrix
class IncidenceMatrix:
    def __init__(self, tokens, mode="lemma"):
        self.mode = mode
        self.doc_ids = list(tokens)
        self.terms = sorted({t for toks in tokens.values() for t in toks})
        self.row = {t: i for i, t in enumerate(self.terms)}
        self.M = np.zeros((len(self.terms), len(self.doc_ids)), dtype=int)
        for j, d in enumerate(self.doc_ids):
            for t in tokens[d]:
                self.M[self.row[t], j] = 1

    def vector(self, term):
        t = normalize_term(term, self.mode)
        return self.M[self.row[t]] if t in self.row else np.zeros(len(self.doc_ids), dtype=int)

    def query_vector(self, q):
        ones = np.ones(len(self.doc_ids), dtype=int)
        return evaluate(q, self.vector, lambda a, b: a & b, lambda a, b: a | b, lambda a: ones - a)

    def query(self, q):
        v = self.query_vector(q)
        return [d for d, bit in zip(self.doc_ids, v) if bit]


# ------------------------------------------------------------ Inverted Index
class InvertedIndex:
    def __init__(self, tokens, mode="lemma"):
        self.mode = mode
        self.doc_ids = list(tokens)
        index = defaultdict(set)
        for d, toks in tokens.items():
            for t in toks:
                index[t].add(d)
        order = {d: i for i, d in enumerate(self.doc_ids)}
        # postings sorted in upload order so the merge algorithm works
        self.postings = {t: sorted(s, key=order.get) for t, s in sorted(index.items())}
        self._order = order

    def get(self, term):
        return self.postings.get(normalize_term(term, self.mode), [])

    def intersect(self, p1, p2):                      # AND  (two-pointer merge)
        i = j = 0
        out = []
        while i < len(p1) and j < len(p2):
            a, b = self._order[p1[i]], self._order[p2[j]]
            if a == b:
                out.append(p1[i]); i += 1; j += 1
            elif a < b:
                i += 1
            else:
                j += 1
        return out

    def union(self, p1, p2):                          # OR
        return sorted(set(p1) | set(p2), key=self._order.get)

    def complement(self, p):                          # NOT
        s = set(p)
        return [d for d in self.doc_ids if d not in s]

    def query(self, q):
        return evaluate(q, self.get, self.intersect, self.union, self.complement)


# ---------------------------------------------------------------- sample data
SAMPLE_DOCS = {
    "doc1_machine_learning.txt": "Machine learning is a branch of artificial intelligence that learns patterns from data. Supervised learning uses labeled data to train models.",
    "doc2_deep_learning.txt": "Deep learning uses neural networks with many layers. Neural networks are trained with gradient descent and backpropagation on large datasets.",
    "doc3_nlp.txt": "Natural language processing allows computers to understand text. Tokenization, stemming and lemmatization are common preprocessing steps in language processing.",
    "doc4_unsupervised.txt": "Unsupervised learning finds hidden structure in unlabeled data. Clustering and dimensionality reduction are popular unsupervised techniques.",
    "doc5_reinforcement.txt": "Reinforcement learning trains an agent using rewards and penalties. The agent learns a policy by interacting with an environment.",
    "doc6_vision.txt": "Computer vision applies deep learning and convolutional neural networks to images. Image classification and object detection are common tasks.",
}
