import typing as T
import os
from uuid import uuid4, UUID

from fastapi import Form, Depends, UploadFile, File, Path, Response, status
from fastapi_restful.cbv import cbv
from fastapi_restful.inferring_router import InferringRouter

from .request import random_seed, read_image
from .response import StableDiffussionResponse, StableDiffussionResultResponse
from app.stable_diffusion.service import StableDiffusionService
from core.settings import get_settings

router = InferringRouter()
env = get_settings()


@cbv(router)
class StableDiffusion:
    svc: StableDiffusionService = Depends(StableDiffusionService)

    @router.get(
        "/task/{task_id}",
    )
    async def get_task(
        self,
        task_id: str = Path(),
    ):
        response = await self.svc.fetch_task_stats(task_id=task_id)
        return response

    @router.post(
        "/text2image",
        response_model=StableDiffussionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def text2image(
        self,
        prompt: str = Form(),
        negative_prompt: str = Form(default=""),
        num_images: int = Form(1, description="num images", ge=1, le=8),
        steps: int = Form(25, ge=1),
        guidance_scale: float = Form(7.5, gt=0, le=20),
        height: int = Form(512, ge=64),
        width: int = Form(512, ge=64),
        seed: T.Optional[int] = Form(None, ge=1),
    ):

        task_id = await self.svc.text2image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
        )

        response = StableDiffussionResponse(
            prompt=prompt,
            task_id=task_id,
        )
        return response
