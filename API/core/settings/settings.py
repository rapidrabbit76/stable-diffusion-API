import sys
import typing as T
from pydantic import BaseSettings


class CelerySettings(BaseSettings):
    CELERY_BROKER: str = "pyamqp://admin:admin@localhost:5672"
    CELERY_BACKEND: str = "redis://localhost:6379"


class Settings(
    CelerySettings,
):
    IMAGESERVER_URL: str = "http://localhost:3000/images"
    SAVE_DIR: str = "static"

    CORS_ALLOW_ORIGINS: T.List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: T.List[str] = ["*"]
    CORS_ALLOW_HEADERS: T.List[str] = ["*"]
