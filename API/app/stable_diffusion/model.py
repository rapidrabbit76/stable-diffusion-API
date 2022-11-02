from pydantic import BaseModel, Field, validator
from PIL import Image
import typing as T


class Text2ImageTask(BaseModel):
    prompt: T.Union[str, T.List[str]] = Field(...)
    negative_prompt: T.Union[str, T.List[str]] = Field(...)
    num_images: int
    num_inference_steps: int = Field(..., gt=0)
    guidance_scale: float = Field(..., ge=0.0)
    height: int
    width: int
    seed: T.Optional[int]

    @validator("height", "width")
    def size_constraint(cls, size):
        cond = size % 64
        if cond != 0:
            raise ValueError("height and width must multiple of 64")
        return size


class Image2ImageTask(BaseModel):
    prompt: T.Union[str, T.List[str]] = Field(...)
    negative_prompt: T.Union[str, T.List[str]] = Field(...)
    num_images: int
    init_image: T.Any
    strength: float = Field(..., ge=0.0, le=1.0)
    num_inference_steps: int = Field(..., gt=0)
    guidance_scale: float = Field(..., ge=0.0)
    seed: T.Optional[int]


class InpaintTask(BaseModel):
    prompt: T.Union[str, T.List[str]] = Field(...)
    negative_prompt: T.Union[str, T.List[str]] = Field(...)
    num_images: int
    init_image: T.Any
    mask_image: T.Any
    strength: float = Field(..., ge=0.0, le=1.0)
    num_inference_steps: int = Field(..., gt=0)
    guidance_scale: float = Field(..., ge=0.0)
    seed: T.Optional[int]


StableDiffusionTaskSchema = T.Union[
    Text2ImageTask,
    Image2ImageTask,
    InpaintTask,
]
