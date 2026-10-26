# task05: Healthcare clinical manual RAG

Parses TXT/Markdown and, with optional `pypdf`, PDF files; compares fixed-overlap and heading chunks; embeds with sentence-transformers when installed and uses TF-IDF cosine as an offline fallback. The local retrieval index is built in memory for this demo.

Run `python main.py`. Optional integrations: `pip install -r requirements-optional.txt`. Model weights download only if you install the optional package and execute the embedding path. This is not clinical advice.
