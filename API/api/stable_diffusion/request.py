import typing as T
from pydantic import BaseModel, Field
from fastapi import Form, UploadFile, File
import sys
from random import randint
from PIL import Image


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="text prompt")
    num_images: int = Field(1, description="num images", ge=1, le=2)


def random_seed(seed: T.Optional[int] = Form(None)):
    seed = seed if seed is not None else randint(1, sys.maxsize)
    return seed


def read_image(image: UploadFile) -> Image.Image:
    image = Image.open(image.file).convert("RGB")
    return image
