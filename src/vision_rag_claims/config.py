from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", protected_namespaces=(), extra="ignore")

    # MMDetection
    mmdet_config_path: Path = Path("models/dcn_plus_cfg.py")
    mmdet_checkpoint_path: Path = Path("models/best.pth")
    score_threshold: float = 0.55
    device: str = "cuda:0"

    # OpenAI
    openai_api_key: SecretStr = SecretStr("")
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # RAG / Chroma
    chroma_persist_dir: Path = Path(".chroma")
    policies_dir: Path = Path("data/policies")
    retrieval_k: int = 4


settings = Settings()
