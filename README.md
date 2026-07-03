# 🔬 BioRAG · Biomedical Literature QA Assistant

A RAG-based (Retrieval-Augmented Generation) question-answering tool for biomedical literature, built to solve a real research pain point: **scientists reading multiple papers can't retain which paper said what, and need to flip between articles repeatedly.**

**Live Demo → [biorag.streamlit.app](https://your-link-here.streamlit.app)**

---

## What It Does

Upload your own PDF papers, ask questions in natural language, and get answers grounded strictly in your uploaded literature — with source citations shown for every response.

Designed around the PTPN22–14-3-3τ–JAK-STAT signaling axis (T-cell receptor research), but works with any biomedical PDF knowledge base.

---

## Key Features

| Feature | Detail |
|---|---|
| 📄 PDF Upload | Batch upload, auto-parsed into text chunks |
| 🧠 Semantic Search | BAAI/bge-large-zh-v1.5 real embedding (not keyword matching) |
| 🚫 Refusal Mechanism | Cosine similarity threshold filters irrelevant chunks; model refuses to fabricate |
| 📎 Source Attribution | Every answer shows which paper and which passage it came from |
| 💬 Conversation History | Multi-turn Q&A within a session |

---

## Technical Architecture

```
User uploads PDFs
      ↓
PDF parsing (pypdf) → Text chunking (sentence-boundary + overlap window)
      ↓
Semantic embedding (BAAI/bge-large-zh-v1.5 via SiliconFlow API)
      ↓
Vector similarity search (cosine similarity, threshold = 0.4)
      ↓
Prompt assembly → LLM generation (Qwen2.5-72B-Instruct)
      ↓
Answer + cited sources returned to user
```

---

## Evaluation

Self-built 24-question evaluation set across 4 categories, tested on 12 real domain papers:

| Category | Score |
|---|---|
| Single-document factual | 6/6 (100%) |
| Cross-document synthesis | 5.5/8 (69%) |
| Experimental evidence | 4.5/6 (75%) |
| Refusal (out-of-scope) | 3/4 (75%) |
| **Overall** | **19/24 (80%)** |

Key finding: the refusal mechanism works well for truly out-of-scope questions. The main gap is in complex cross-document reasoning, where the model tends toward conservative refusal rather than inference from partial evidence — a clear direction for next iteration.

---

## Development Journey

| Version | What Changed | Why |
|---|---|---|
| v1 | TF-IDF retrieval | Get the pipeline working end-to-end |
| Bug fix | Replaced default token_pattern with char-level n-gram tokenizer | Chinese text has no spaces → TF-IDF similarity scores were all 0 |
| v2 | Switched to real semantic embedding | TF-IDF can't distinguish "phosphorylation" from "dephosphorylation" semantically |
| Model switch | Free small model → Qwen2.5-72B | Small models showed output collapse (token repetition, instruction-following failure) — diagnosed by comparing output patterns, not just accuracy scores |

---

## Tech Stack

- **Backend**: Python, pypdf, openai SDK
- **Embedding**: BAAI/bge-large-zh-v1.5 (SiliconFlow)
- **Generation**: Qwen2.5-72B-Instruct (SiliconFlow)
- **Frontend**: Streamlit
- **Dev environment**: Google Colab

---

## Background

Built as part of a personal AI learning project (Stanford ML/DL courses + Li Hung-yi Generative AI course), grounded in real prior wet-lab research experience: T-cell receptor signaling pathway protein interaction study (undergraduate thesis, 2020–2022).

The core use case — cross-paper mechanism synthesis — came from a real frustration during that research period.