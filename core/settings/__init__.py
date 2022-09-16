from functools import lru_cache
from .settings import Settings


@lru_cache()
def get_settings() -> Settings:
    setting = Settings("env/dev.env")
    return setting
