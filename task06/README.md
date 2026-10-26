# task06: Legal contract hybrid RAG

Combines BM25-style keyword ranking with TF-IDF cosine retrieval using reciprocal rank fusion, then re-ranks with an optional sentence-transformers CrossEncoder (lexical fallback otherwise). Answers display contract text as citations and remain extractive/context-grounded.

Run `python main.py`. Optional reranker: `pip install -r requirements-optional.txt`; model weights download only when explicitly run. This sample is not legal advice.
