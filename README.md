# RAG From Scratch (Without LangChain)

A beginner-friendly Retrieval-Augmented Generation (RAG) pipeline built completely from scratch using Python — without LangChain.

This project helped me understand how modern AI assistants work internally by implementing every core component manually.

<img width="902" height="478" alt="image" src="https://github.com/user-attachments/assets/d6642cc8-c376-4ad1-87e1-1d653217f1f0" />

---

# Features

- PDF text extraction using `pypdf`
- Text cleaning and chunking
- Sentence-based chunk splitting
- Embedding generation using Sentence Transformers
- Vector storage using ChromaDB
- Semantic similarity search
- Retrieval pipeline
- Response generation using Hugging Face models
- Metadata storage for chunks

---

# Tech Stack

- Python
- ChromaDB
- Sentence Transformers
- Hugging Face Transformers
- pypdf

---

# Project Flow

```text
PDF
↓
Extract Text
↓
Clean Text
↓
Split Into Chunks
↓
Generate Embeddings
↓
Store In ChromaDB
↓
User Query
↓
Semantic Search
↓
Retrieve Relevant Chunks
↓
Send Context To LLM
↓
Generate Final Response
