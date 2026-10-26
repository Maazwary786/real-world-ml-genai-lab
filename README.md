# Real-World ML & GenAI Lab

Ten lightweight, runnable projects based on the attached laboratory handbook. Each task is independent, uses synthetic/sample data, and avoids network calls and model downloads by default. These are educational prototypes, not production medical, legal, financial, or security decisions.

## Setup

Python 3.10+ recommended. From this directory: `python -m venv .venv`; activate it; install a task dependency with `pip install -r taskNN/requirements.txt`. The ML tasks use the pinned-compatible scikit-learn, NumPy, and pandas packages; task 03 also offers an optional PyTorch model. Copy `.env.example` to `.env` only if you choose to configure a remote provider. Never commit `.env`.

## Projects

### task01: Industrial IoT predictive maintenance

Linear and logistic regression, polynomial features, scaling, threshold tuning.

Technologies: scikit-learn, pandas. Run: `cd task01 && python main.py`.

### task02: Fintech fraud scoring and customer segments

Interpretable decision tree, random forest ROC-AUC, K-Means elbow and silhouette metrics.

Technologies: scikit-learn, pandas. Run: `cd task02 && python main.py`.

### task03: Supply chain demand forecasting

PyTorch MLP with batch normalization and dropout, random search, random forest baseline.

Technologies: scikit-learn, pandas, torch. Run: `cd task03 && python main.py`.

### task04: Customer support intent router

Validated structured JSON routing, prompt variants and benchmark; deterministic mock LLM wrapper with retry/backoff seam.

Technologies: Python standard library. Run: `cd task04 && python main.py`.

### task05: Healthcare clinical manual RAG

Text/Markdown ingestion, fixed overlap and heading chunks, TF-IDF dense-style local retrieval.

Technologies: scikit-learn. Run: `cd task05 && python main.py`.

### task06: Legal hybrid RAG

BM25-like lexical rank plus TF-IDF ranks fused by RRF; reranking and citation-grounded answer template.

Technologies: scikit-learn. Run: `cd task06 && python main.py`.

### task07: Financial analyst tool agent

ReAct-style loop with safe SQLite query, market document retrieval, and bounded calculator tools.

Technologies: Python standard library, scikit-learn. Run: `cd task07 && python main.py`.

### task08: Collaborative DevSecOps audit

Security scanner, remediation and QA agents with structured handoffs, safe demo patch and syntax validation.

Technologies: Python standard library. Run: `cd task08 && python main.py`.

### task09: Claims PEFT / QLoRA preparation

Anonymized instruction JSONL, lightweight baseline report, optional Transformers/PEFT 4-bit training entrypoint.

Technologies: Python standard library; optional torch, transformers, peft, trl, bitsandbytes. Run: `cd task09 && python prepare.py`.

### task10: Guarded agentic logistics RAG capstone

Hybrid retrieval, tool execution, input/output guardrails, synthetic 50-scenario evaluation scorecard.

Technologies: Python standard library, scikit-learn. Run: `cd task10 && python main.py`.

## Verification

Run the self-contained smoke suite from the root with `python -m unittest discover -s tests -v`. Each task also has its own `README.md`, sample data, and evaluation output behavior. Synthetic results do not establish real-world model performance. For the heavy tasks, optional integrations are explicitly labeled and do not download models automatically.
