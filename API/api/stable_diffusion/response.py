import typing as T
from pydantic import BaseModel, Field, HttpUrl


class StableDiffussionResultResponse(BaseModel):
    task_id: str = Field(..., description="task id")
    prompt: T.Optional[str] = Field(None, description="input prompt")
    state: T.Optional[str] = Field(None, description="celery state")
    image_urls: T.Optional[T.List[T.Union[str, HttpUrl]]] = Field(
        [], description="image url"
    )


class StableDiffussionResponse(BaseModel):
    prompt: T.Optional[str] = Field(None, description="input prompt")
    task_id: str = Field(..., description="task id")
