import os
import typing as T
from itertools import chain, islice
from service_streamer import ThreadedStreamer

import torch
from core.settings import get_settings
from fastapi import Depends
from loguru import logger
from PIL import Image
import json


from .manager import (
    Text2ImageTask,
    Image2ImageTask,
    InpaintTask,
)

from .manager import (
    build_streamer,
)


env = get_settings()


def data_to_batch(datasets: T.List[T.Any], batch_size: int):
    iterator = iter(datasets)
    for first in iterator:
        yield list(chain([first], islice(iterator, batch_size - 1)))


class StableDiffusionService:
    def __init__(
        self,
        streamer: ThreadedStreamer = Depends(build_streamer),
    ) -> None:
        logger.info(f"DI:{self.__class__.__name__}")
        self.streamer = streamer

    @torch.inference_mode()
    def text2image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.5,
        height=512,
        width=512,
        seed: T.Optional[int] = None,
    ) -> T.List[Image.Image]:
        prompts = [prompt] * num_images
        negative_prompt = [negative_prompt] * num_images

        tasks = [
            Text2ImageTask(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                seed=seed,
            )
            for prompt in data_to_batch(prompts, batch_size=env.MB_BATCH_SIZE)
        ]
        future = self.streamer.submit(tasks)
        images = future.result(timeout=env.MB_TIMEOUT)
        results = []
        for image in images:
            results += image
        return results

    @torch.inference_mode()
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
        origin_size = init_image.size
        w, h = origin_size
        w, h = map(lambda x: x - x % 64, (w, h))
        if origin_size != (w, h):
            init_image = init_image.resize((w, h), resample=Image.LANCZOS)

        prompt = [prompt] * num_images
        task = Image2ImageTask(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        future = self.streamer.submit([task])
        images = future.result(timeout=env.MB_TIMEOUT)[0]
        images = self.postprocess(images, origin_size=origin_size)
        return images

    @classmethod
    def postprocess(
        cls, images: T.List[Image.Image], origin_size: T.Tuple[int, int]
    ):
        if origin_size == images[0].size:
            return images
        for i, image in enumerate(images):
            images[i] = image.resize(origin_size)
        return images

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
        origin_size = init_image.size
        w, h = origin_size
        w, h = map(lambda x: x - x % 64, (w, h))
        if origin_size != (w, h):
            init_image = init_image.resize((w, h), resample=Image.LANCZOS)
            mask_image = mask_image.resize((w, h), resample=Image.NEAREST)

        prompt = [prompt] * num_images
        task = InpaintTask(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        future = self.streamer.submit([task])
        images = future.result(timeout=env.MB_TIMEOUT)[0]
        images = self.postprocess(images, origin_size=origin_size)
        return images

    @staticmethod
    def image_save(images: T.List[Image.Image], task_id: str, info: dict):
        save_dir = os.path.join(env.SAVE_DIR, task_id)
        os.makedirs(save_dir)
        image_urls = []

        with open(os.path.join(save_dir, "info.json"), "w") as f:
            json.dump(info, f)

        for i, image in enumerate(images):
            filename = f"{str(i).zfill(2)}.webp"
            save_path = os.path.join(env.SAVE_DIR, task_id, filename)
            image_url = os.path.join(task_id, filename)
            image.save(save_path)
            image_urls.append(image_url)
        return image_urls
