import typing as T
from pydantic import BaseModel, Field, HttpUrl


class StableDiffussionResultResponse(BaseModel):
    task_id: str = Field(..., description="task id")
    state: str = Field(None, description="celery state")
    results: T.Any


class StableDiffussionResponse(BaseModel):
    prompt: T.Optional[str] = Field(None, description="input prompt")
    task_id: str = Field(..., description="task id")
