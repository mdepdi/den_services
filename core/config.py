import sys
import os
from pathlib import Path
from pydantic_settings import BaseSettings

sys.path.append(r"D:\JACOBS\SERVICE\API")

docker = os.getenv("DOCKER", "False").lower() in ("true", "1", "t")
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / (".env.docker" if docker else ".env")

class Settings(BaseSettings):
    PROJECT_NAME: str = "Design and Engineering API Services"
    VERSION: str = "0.1.0"
    DEBUG: bool = True

    MAX_WORKERS: int = 4
    DOCKER: bool
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    DATA_DIR: str
    MAINDATA_DIR: str
    LOG_DIR: str
    EXPORT_DIR: str
    UPLOAD_DIR: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    class Config:
        env_file = str(ENV_FILE)
        env_file_encoding = "utf-8"

settings = Settings()
