import os
import streamlit as st
from functools import lru_cache
from pydantic import BaseSettings

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def load_config(title=None, icone=None):
    st.set_page_config(
        page_title=title if title is not None else os.getenv("ST_TITLE"),
        layout="wide" if os.getenv("ST_WIDE") == "True" else "centered",
        menu_items={},
    )


class Settings(BaseSettings):
    API_URL: str = "http://localhost:3000"
    ST_TITLE: str = "Stable-diffusion"
    ST_WIDE: str = True
    IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp"]


@lru_cache()
def get_settings() -> Settings:
    setting = Settings()
    return setting
