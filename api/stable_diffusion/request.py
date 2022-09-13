from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="text prompt")
    num_images: int = Field(1, description='num images', ge=1, le=2)
