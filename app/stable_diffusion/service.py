import os
import typing as T
from itertools import chain, islice

import torch
from core.settings import get_settings
from fastapi import Depends
from loguru import logger
from PIL import Image
import json

from .model import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionText2ImagePipeline,
    build_image2image_pipeline,
    build_inpaint_pipeline,
    build_text2image_pipeline,
)

env = get_settings()


def data_to_batch(datasets: T.List[T.Any], batch_size: int):
    iterator = iter(datasets)
    for first in iterator:
        yield list(chain([first], islice(iterator, batch_size - 1)))


class StableDiffusionService:
    def __init__(
        self,
        text2image: StableDiffusionText2ImagePipeline = Depends(
            build_text2image_pipeline
        ),
        image2image: StableDiffusionImg2ImgPipeline = Depends(
            build_image2image_pipeline
        ),
        inpaint: StableDiffusionInpaintPipeline = Depends(
            build_inpaint_pipeline
        ),
    ) -> None:
        logger.info(f"DI:{self.__class__.__name__}")
        self.text2image_pipeline = text2image
        self.image2image_pipeline = image2image
        self.inpaint_pipeline = inpaint
        self.generator = torch.Generator(device=env.CUDA_DEVICE)

    @torch.inference_mode()
    def text2image(
        self,
        prompt: str,
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.5,
        height=512,
        width=512,
        seed: int = 203,
    ) -> T.List[Image.Image]:
        # [128, 256, 512, 768, 1024]
        device = env.CUDA_DEVICE
        generator = self.generator.manual_seed(seed)
        prompt = [prompt] * num_images
        prompt = data_to_batch(prompt, batch_size=env.MB_BATCH_SIZE)
        images = []
        for inputs in prompt:
            with torch.autocast("cuda" if device != "cpu" else "cpu"):
                output = self.text2image_pipeline(
                    inputs,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                )

            if device != "cpu":
                torch.cuda.empty_cache()
            images += output
        return images

    @torch.inference_mode()
    def image2image(
        self,
        prompt: str,
        init_image: Image.Image,
        num_images: int = 1,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.5,
        seed: int = 203,
    ) -> T.List[Image.Image]:
        device = env.CUDA_DEVICE
        origin_size = init_image.size
        w, h = origin_size
        w, h = map(lambda x: x - x % 64, (w, h))

        init_image = init_image.resize((w, h), resample=Image.LANCZOS)

        generator = self.generator.manual_seed(seed)
        prompt = [prompt] * num_images
        prompt = data_to_batch(prompt, batch_size=env.MB_BATCH_SIZE)
        images = []
        for inputs in prompt:
            with torch.autocast("cuda" if device != "cpu" else "cpu"):
                output = self.image2image_pipeline(
                    inputs,
                    init_image=init_image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

            if device != "cpu":
                torch.cuda.empty_cache()
            images += output

        if origin_size == images[0].size:
            return images

        for i, image in enumerate(images):
            images[i] = image.resize(origin_size)

        return images

    @torch.inference_mode()
    def inpaint(
        self,
        prompt: str,
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
        mask_image = mask_image.resize((w, h), resample=Image.NEAREST)
        init_image = init_image.resize((w, h), resample=Image.LANCZOS)

        device = env.CUDA_DEVICE
        generator = self.generator.manual_seed(seed)
        prompt = [prompt] * num_images
        prompt = data_to_batch(prompt, batch_size=env.MB_BATCH_SIZE)
        images = []
        for inputs in prompt:
            with torch.autocast("cuda" if device != "cpu" else "cpu"):
                output = self.inpaint_pipeline(
                    inputs,
                    init_image=init_image,
                    mask_image=mask_image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            if device != "cpu":
                torch.cuda.empty_cache()
            images += output

        if origin_size == images[0].size:
            return images

        for i, image in enumerate(images):
            images[i] = image.resize(origin_size)

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
