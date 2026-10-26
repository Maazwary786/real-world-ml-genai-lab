from collections import Counter
from pathlib import Path
import math
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS = [line.strip() for line in (Path(__file__).parent / "data/contracts.txt").read_text().splitlines() if line.strip()]


def terms(text):
    return re.findall(r"[a-z0-9.]+", text.lower())


def run(query, k=2):
    token_docs = [terms(doc) for doc in DOCS]
    query_terms = terms(query)
    count = len(DOCS)
    document_frequency = Counter(token for doc in token_docs for token in set(doc))
    bm25_scores = []
    for doc in token_docs:
        score = 0.0
        for token in query_terms:
            frequency = doc.count(token)
            score += math.log(1 + (count - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5)) * (frequency * 2.2 / (frequency + 1.2))
        bm25_scores.append(score)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(DOCS)
    dense_scores = cosine_similarity(vectorizer.transform([query]), matrix)[0].tolist()
    rankings = []
    for scores in (bm25_scores, dense_scores):
        rankings.append({idx: rank + 1 for rank, idx in enumerate(sorted(range(count), key=lambda i: scores[i], reverse=True))})
    fused = sorted(range(count), key=lambda idx: sum(1 / (60 + ranking[idx]) for ranking in rankings), reverse=True)[:max(k, 3)]
    try:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = reranker.predict([(query, DOCS[idx]) for idx in fused])
        ordered = [fused[j] for j in sorted(range(len(fused)), key=lambda j: float(scores[j]), reverse=True)]
        backend = "CrossEncoder reranker"
    except ImportError:
        ordered = sorted(fused, key=lambda idx: sum(token in DOCS[idx].lower() for token in query_terms), reverse=True)
        backend = "lexical rerank fallback"
    return ordered[:k], backend


query = "What is the deadline to report a security incident?"
results, backend = run(query)
print("Query:", query, "|", backend)
for index in results:
    print(f"[Agreement citation] {DOCS[index]}")
print("RRF hybrid retrieval; measure context precision/recall against labeled queries.")
