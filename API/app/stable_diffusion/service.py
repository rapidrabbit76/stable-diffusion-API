import os
import typing as T
from itertools import chain, islice

import torch
from core.settings import get_settings
from fastapi import Depends, Response, status
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from celery import states

from core.celery_app import get_celery_app, Celery


env = get_settings()


def data_to_batch(datasets: T.List[T.Any], batch_size: int):
    iterator = iter(datasets)
    for first in iterator:
        yield list(chain([first], islice(iterator, batch_size - 1)))


class StableDiffusionService:
    def __init__(
        self,
        celery_app: Celery = Depends(get_celery_app),
    ) -> None:
        logger.info(f"DI:{self.__class__.__name__}")
        self.celery_app = celery_app
        self.task = self.celery_app.signature("tasks.predict")

    async def fetch_task_stats(self, task_id: str):
        res = self.celery_app.AsyncResult(task_id)
        state = res.state

        if state != states.SUCCESS:
            result = dict(task_id=task_id, state=state)
            return JSONResponse(
                content=result,
                status_code=status.HTTP_202_ACCEPTED,
            )

        result = res.get()
        image_urls = result.get("image_uris")
        image_urls = list(
            map(lambda x: os.path.join(env.IMAGESERVER_URL, x), image_urls)
        )
        result = dict(
            task_id=task_id,
            state=state,
            prompt=result.get("prompt"),
            image_urls=image_urls,
        )
        return result

    async def text2image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.5,
        height=512,
        width=512,
        seed: T.Optional[int] = None,
    ) -> str:
        task = self.task.delay(
            task="text2image",
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images=num_images,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            seed=seed,
        )
        return task.id

    def image2image(
        self,
        prompt: str,
        negative_prompt: str,
        init_image: Image.Image,
        num_images: int = 1,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.5,
        seed: int = 203,
    ) -> T.List[Image.Image]:
        ...

    @torch.inference_mode()
    def inpaint(
        self,
        prompt: str,
        negative_prompt: str,
        init_image: Image.Image,
        mask_image: Image.Image,
        strength: float,
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.5,
        seed: int = 203,
    ) -> T.List[Image.Image]:
        ...
