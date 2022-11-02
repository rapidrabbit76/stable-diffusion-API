import typing as T

from pydantic import BaseSettings


class ModelSetting(BaseSettings):
    MODEL_PATH: str = "./ckpt"


class DeviceSettings(BaseSettings):
    CUDA_DEVICE = "cuda"


class MicroBatchSettings(BaseSettings):
    MB_BATCH_SIZE = 2
    MB_TIMEOUT = 600


class CelerySettings(BaseSettings):
    CELERY_BROKER: str = "pyamqp://admin:admin@localhost:5672"
    CELERY_BACKEND: str = "redis://localhost:6379"


class Settings(
    ModelSetting,
    DeviceSettings,
    MicroBatchSettings,
    CelerySettings,
):
    IMAGESERVER_URL: str = "localhost:300"
    SAVE_DIR: str = "../API/static"

    CORS_ALLOW_ORIGINS: T.List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: T.List[str] = ["*"]
    CORS_ALLOW_HEADERS: T.List[str] = ["*"]
