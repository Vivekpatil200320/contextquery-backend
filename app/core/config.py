from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    nvidia_api_key: str
    chroma_persist_dir: str = "./chroma_db"
    ollama_base_url: str = "http://localhost:11434"
    llm_provider: str = "ollama"
    nvidia_llm_model: str = "meta/llama-3.1-8b-instruct"

    # Chroma Cloud
    chroma_api_key: str
    chroma_tenant: str
    chroma_database: str = "contextquery"
    chroma_collection: str = "documents"

    # Retrieval mode: "semantic" (default) or "hybrid" (BM25 + semantic via RRF)
    retrieval_mode: str = "semantic"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings() # type: ignore