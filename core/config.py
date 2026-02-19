import os
from pathlib import Path
from pydantic_settings import BaseSettings

docker = os.getenv("DOCKER", "False").lower() in ("true", "1", "t")
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / (".env.docker" if docker else ".env")

class Settings(BaseSettings):
    PROJECT_NAME: str = "Design and Engineering API Services"
    VERSION: str = "1.1.0"
    DEBUG: bool = True

    MAX_WORKERS: int
    DOCKER: bool
    FLOWER_USER: str
    FLOWER_PWD: str
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    MAINDATA_DIR: str
    DATA_DIR: str = f"{BASE_DIR}/data"
    LOG_DIR: str = f"{BASE_DIR}/logs"
    EXPORT_DIR: str = f"{BASE_DIR}/exports"
    UPLOAD_DIR: str = f"{BASE_DIR}/uploads"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    class Config:
        env_file = str(ENV_FILE)
        env_file_encoding = "utf-8"

settings = Settings()
