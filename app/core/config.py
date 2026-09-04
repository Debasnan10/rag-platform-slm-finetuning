"""Centralized, typed application configuration.

Every setting the app needs lives here, in one place, with a type and a
default. Values are populated from (in priority order): real environment
variables > a local `.env` file > the defaults below. This is the same
idea as `IConfiguration` / strongly-typed `IOptions<T>` in ASP.NET Core -
one object graph you inject wherever you need config, instead of calling
`os.environ["SOME_VAR"]` scattered across the codebase.

Usage:
    from app.core.config import get_settings

    settings = get_settings()
    settings.ollama_base_url
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for the RAG platform, grouped by concern.

    Field names map to environment variables of the same name
    (case-insensitive), e.g. `OLLAMA_BASE_URL` -> `ollama_base_url`.
    See `.env.example` for the full list with explanations.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App -----------------------------------------------------------
    app_name: str = "RAG Platform"
    app_env: Literal["development", "testing", "production"] = "development"
    app_debug: bool = True
    app_version: str = "0.1.0"

    # ---- Logging ---------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    # ---- API server --------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---- Ollama (local LLM runtime) -----------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_rag_model: str = "llama3.2:3b"
    ollama_finetune_small_model: str = "llama3.2:1b"
    ollama_finetune_large_model: str = "mistral:7b"
    ollama_request_timeout_seconds: int = 120
    ollama_temperature: float = 0.1

    # ---- Embeddings (sentence-transformers) ----------------------------------
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"

    # ---- Document ingestion & chunking ------------------------------------------
    chunk_size: int = 500
    chunk_overlap: int = 50
    ingestion_source_dir: str = "./data/raw"
    ingestion_supported_extensions: str = ".pdf,.docx,.html,.md"

    # ---- FAISS vector store -----------------------------------------------------
    faiss_index_dir: str = "./data/faiss_index"
    faiss_top_k: int = 5

    # ---- Hybrid retrieval / reranking ----------------------------------------------
    retrieval_dense_weight: float = 0.5
    retrieval_sparse_weight: float = 0.5
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 3

    # ---- Fine-tuning (Colab / Kaggle GPU) --------------------------------------------
    finetune_base_model: str = "meta-llama/Llama-3.2-1B"
    finetune_output_dir: str = "./data/adapters"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ---- Evaluation ---------------------------------------------------------------
    eval_test_set_path: str = "./data/eval/test_questions.json"
    eval_report_dir: str = "./data/eval/reports"

    # ---- Observability --------------------------------------------------------------
    prometheus_metrics_path: str = "/metrics"

    @property
    def ingestion_supported_extensions_list(self) -> list[str]:
        """`.pdf,.docx,.html,.md` -> `[".pdf", ".docx", ".html", ".md"]`."""
        return [ext.strip() for ext in self.ingestion_supported_extensions.split(",") if ext.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    `lru_cache` means the *first* call builds the object (reading env vars
    and `.env`); every later call returns the same cached instance rather
    than re-parsing. This is the equivalent of registering `IOptions<T>`
    as a singleton in the .NET DI container - construct once, reuse
    everywhere, including in FastAPI route handlers via `Depends(get_settings)`.
    """

    return Settings()
