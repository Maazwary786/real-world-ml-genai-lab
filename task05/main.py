from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent / "data"


def read_documents(folder):
    docs = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("Install requirements-optional.txt to ingest PDF files") from exc
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            continue
        docs.append((path.name, text))
    return docs


def fixed_chunks(text, size=180, overlap=40):
    return [text[i:i + size] for i in range(0, len(text), size - overlap) if text[i:i + size].strip()]


def header_chunks(text):
    return [part.strip() for part in re.split(r"(?=^# )", text, flags=re.M) if part.strip()]


docs = read_documents(BASE)
chunks = [{"source": name, "text": part} for name, text in docs for part in fixed_chunks(text) + header_chunks(text)]
query = "patient has severe breathing difficulty and low oxygen"
try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.encode([chunk["text"] for chunk in chunks], normalize_embeddings=True)
    query_vector = model.encode([query], normalize_embeddings=True)
    scores = (query_vector @ vectors.T)[0]
    backend = "dense sentence-transformer embeddings"
except ImportError:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    scores = cosine_similarity(vectorizer.transform([query]), vectors)[0]
    backend = "offline TF-IDF cosine fallback"

print("Chunking: fixed overlap and heading chunks; chunks:", len(chunks), "backend:", backend)
for index in sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)[:3]:
    print(round(float(scores[index]), 3), chunks[index]["source"], chunks[index]["text"][:150].replace("\n", " "))
