# Production-Grade RAG Platform with On-Device SLM Fine-Tuning

A portfolio project demonstrating an end-to-end Retrieval-Augmented Generation
pipeline built on industry-standard tooling - **LangChain**, **FAISS**,
**Hugging Face PEFT** - plus on-device fine-tuning (LoRA / QLoRA), GGUF
quantization benchmarking, and full observability. Everything runs on free
tooling: **Ollama** for local inference, Hugging Face's free model hub, and
Google Colab / Kaggle's free GPU tiers for the fine-tuning steps.

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| RAG Framework | LangChain + LangChain-Community |
| Vector Store | FAISS |
| Embeddings | sentence-transformers (local) |
| LLM Runtime | Ollama (llama3.2:3b) |
| Fine-Tuning | Hugging Face PEFT (LoRA + QLoRA) |
| Quantization | llama.cpp / GGUF |
| Evaluation | sacrebleu, rouge-score, HF `evaluate` (perplexity) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Observability | Prometheus + Grafana |
| API | FastAPI |
| Config | Pydantic Settings + `.env` |
| Testing | pytest |
| Containers | Docker + Docker Compose |

## Project status

Built phase by phase. See each phase's own explanation for how it works and why.

- [x] **Phase 1** - Project foundation: config, logging, exceptions, Ollama integration, FastAPI `/health`, Docker skeleton, tests
- [ ] Phase 2 - LangChain document ingestion
- [ ] Phase 3 - FAISS vector store
- [ ] Phase 4 - Full RAG chain + hybrid search + reranking
- [ ] Phase 5 - GGUF quantization benchmarking
- [ ] Phase 6 - LoRA fine-tuning (PEFT)
- [ ] Phase 7 - QLoRA fine-tuning
- [ ] Phase 8 - Evaluation metrics (BLEU / ROUGE / Perplexity)
- [ ] Phase 9 - FastAPI + Prometheus + Grafana observability

## Getting started

### 1. Prerequisites

- Python 3.12
- [Ollama](https://ollama.com) installed and running (`ollama serve`), with at least one model pulled:
  ```bash
  ollama pull llama3.2:3b
  ```
- Docker + Docker Compose (optional for local dev, required for the full stack in later phases)

### 2. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # adjust values if needed
```

### 3. Run the API

```bash
make run
# or: uvicorn app.api.main:app --reload
```

Then check `http://localhost:8000/health`.

### 4. Run tests

```bash
make test
# or: pytest -v
```

### 5. Run with Docker

```bash
make docker-up
# or: docker compose up --build
```

## Project structure

```
app/
  core/          # config, structured logging, exception hierarchy
  ingestion/      # (Phase 2) LangChain document loaders + splitters
  vectorstore/    # (Phase 3) FAISS wrapper
  retrieval/      # (Phase 4) dense / sparse / hybrid retrieval + reranking
  llm/           # LangChain <-> Ollama integration
  rag/           # (Phase 4) RAG chain + prompt templates
  finetuning/     # (Phase 5-7) dataset prep, LoRA/QLoRA trainers, GGUF benchmarking
  evaluation/     # (Phase 8) BLEU / ROUGE / Perplexity
  api/           # FastAPI app + routes
tests/           # pytest suite
notebooks/       # Colab/Kaggle notebooks for GPU-heavy fine-tuning steps
docker/          # Prometheus + Grafana config (Phase 9)
```

## Why these tools (not built from scratch)

This project deliberately leans on ready-made libraries - LangChain,
FAISS, Hugging Face PEFT - to demonstrate familiarity with the tooling
real AI teams use day to day, in contrast to a from-scratch implementation
of the same concepts.
