import typing as T

from pydantic import BaseSettings


class ModelSetting(BaseSettings):
    MODEL_ID: str = "CompVis/stable-diffusion-v1-4"


class DeviceSettings(BaseSettings):
    CUDA_DEVICE = "cuda"
    CUDA_DEVICES = [0]


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
    HUGGINGFACE_TOKEN: str
    IMAGESERVER_URL: str
    SAVE_DIR: str = "static"

    CORS_ALLOW_ORIGINS: T.List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: T.List[str] = ["*"]
    CORS_ALLOW_HEADERS: T.List[str] = ["*"]
