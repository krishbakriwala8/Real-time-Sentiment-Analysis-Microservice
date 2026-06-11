"""
app/config.py
─────────────
Central configuration loaded from environment variables (or .env file).
All other modules import `settings` from here — no scattered os.getenv() calls.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Model
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    fine_tuned_model_path: str = ""
    max_sequence_length: int = 512
    inference_batch_size: int = 16
    confidence_threshold: float = 0.6

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "sentiment_user"
    postgres_password: str = "sentiment_pass"
    postgres_db: str = "sentiment_db"
    database_url: str = (
        "postgresql+asyncpg://sentiment_user:sentiment_pass@localhost:5432/sentiment_db"
    )

    # Benchmarking
    latency_target_ms: float = 200.0
    benchmark_sample_size: int = 500
    benchmark_results_dir: str = "./benchmark_results"

    @property
    def active_model_path(self) -> str:
        """Return fine-tuned checkpoint path if set, else HuggingFace hub name."""
        return self.fine_tuned_model_path or self.model_name


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
