from pydantic_settings import BaseSettings
import os
import sys

sys.path.append(r"D:\Data Analytical\SERVICE\API")
docker = os.getenv("DOCKER", "False").lower() in ("true", "1", "t")

class Settings(BaseSettings):
    PROJECT_NAME: str = "DEN - Fiberization API"
    VERSION: str = "0.1.0"
    DEBUG: bool = True
    # SYNC_DATABASE_URL: str 
    # ASYNC_DATABASE_URL: str 
    # SECRET_KEY: str
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
        env_file = ".env" if not docker else ".env.docker"
    
settings = Settings()