"""Конфигурация приложения (pydantic-settings).

Все настраиваемые параметры вынесены в переменные окружения / .env.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- База данных ---
    postgres_user: str = "assistant"
    postgres_password: str = "assistant"
    postgres_db: str = "assistant"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # --- Ollama (локальная, нативная) ---
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b-instruct"
    embed_model: str = "nomic-embed-text"
    embed_dimensions: int = 768
    llm_timeout: float = 120.0
    embed_timeout: float = 60.0

    # --- RAG-параметры ---
    top_k: int = Field(default=4, description="Сколько чанков доставать из поиска")
    top_k_broad: int = Field(
        default=6,
        description="Чанков для обзорных вопросов («расскажи всё/подробнее»)",
    )
    sim_threshold: float = Field(
        default=0.35,
        description="Порог косинусной СХОЖЕСТИ (1 - distance). Ниже = заглушка",
    )
    sim_drop: float = Field(
        default=0.15,
        description=(
            "Относительный разрыв: чанк отбрасывается, если его sim ниже "
            "(max_sim - sim_drop). Дополняет абсолютный порог на однородных корпусах"
        ),
    )
    hybrid_alpha: float = Field(
        default=0.6,
        description="Вес векторного скоринга в гибридной формуле (1-alpha = лексический)",
    )
    retrieval_candidates: int = Field(
        default=20,
        description="Сколько кандидатов достаёт векторный поиск до реранка",
    )
    link_expansion: bool = Field(
        default=True,
        description="Расширять ли поиск по графу wikilinks (Obsidian-заметки)",
    )
    link_expansion_chunks: int = Field(
        default=8,
        description="Максимум доп. чанков из связанных заметок при расширении",
    )
    max_concurrent_llm: int = Field(
        default=1,
        description="Лимит параллельных запросов к Ollama (бережно к RAM/CPU)",
    )
    grounding_max_retries: int = Field(
        default=2,
        description="Число повторных генераций при провале grounding",
    )
    grounding_min_overlap: float = Field(
        default=0.35,
        description="Мин. доля 3-грамм ответа, покрытая контекстом (анти-галлюцинация)",
    )
    source_overlap: float = Field(
        default=0.3,
        description="Мин. пересечение ответа с чанком, чтобы считать чанк источником",
    )

    # --- Чанкование ---
    chunk_size: int = 512
    chunk_overlap: int = 50

    # --- Telegram ---
    telegram_bot_token: str = ""
    allowed_chat_ids: str = ""  # CSV, напр. "123,456"
    open_access: bool = Field(
        default=False,
        description="Разрешить всем при пустом ALLOWED_CHAT_IDS (только dev!)",
    )

    @property
    def allowed_chats(self) -> set[int]:
        return {int(x) for x in self.allowed_chat_ids.split(",") if x.strip()}

    # --- Приложение ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    uploads_dir: str = "data/uploads"
    max_upload_mb: int = Field(default=50, description="Макс. размер загружаемого файла, МБ")

    # --- Источники / синхронизация ---
    source_dirs: str = Field(
        default="",
        description="CSV абсолютных путей к источникам (Obsidian vault / папки)",
    )
    ignore_dirs: str = Field(
        default=".obsidian,.trash",
        description="CSV имён директорий, которые игнорируются при сканировании",
    )
    gc_retention_days: int = Field(
        default=30,
        description="Дней до физического удаления деактивированных документов",
    )

    @property
    def source_list(self) -> list[str]:
        return [x.strip() for x in self.source_dirs.split(",") if x.strip()]

    @property
    def ignore_set(self) -> set[str]:
        return {x.strip() for x in self.ignore_dirs.split(",") if x.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
