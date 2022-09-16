import typing as T
from pydantic import BaseModel, Field, HttpUrl


class StableDiffussionResponse(BaseModel):
    prompt: str = Field(..., description="input prompt")
    task_id: str = Field(..., description="task id")
    image_urls: T.List[T.Union[str, HttpUrl]] = Field(
        ..., description="image url"
    )
