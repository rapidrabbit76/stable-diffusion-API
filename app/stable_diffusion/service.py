import os
import typing as T
from fastapi import Depends
import torch
from itertools import chain, islice
from PIL import Image
from core.settings import get_settings
from loguru import logger
from .model import build_pipeline, StableDiffusionPipeline

env = get_settings()


def data_to_batch(datasets: T.List[T.Any], batch_size: int):
    iterator = iter(datasets)
    for first in iterator:
        yield list(chain([first], islice(iterator, batch_size - 1)))


class StableDiffusionService:
    def __init__(
            self,
            model: StableDiffusionPipeline = Depends(build_pipeline),
    ) -> None:
        logger.info(f'DI:{self.__class__.__name__}')
        self.model = model

    @torch.inference_mode()
    def predict(
        self,
        prompt: str,
        num_images: int = 1,
        num_inference_steps: int = 40,
        guidance_scale: float = 8.5,
        height=512,
        width=512,
    ) -> T.List[Image.Image]:
        device = env.CUDA_DEVICE
        prompt = [prompt] * num_images
        prompt = data_to_batch(prompt, batch_size=env.MB_BATCH_SIZE)
        images = []
        for inputs in prompt:
            images += self.inference(
                inputs,
                num_inference_steps,
                guidance_scale,
                height,
                width,
                device,
            )

        return images

    @torch.inference_mode()
    def inference(
        self,
        prompt: T.List[str],
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        device,
    ) -> T.List[Image.Image]:
        with torch.autocast('cuda' if device != 'cpu' else 'cpu'):
            output = self.model(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            )

        if device != "cpu":
            torch.cuda.empty_cache()

        images = output.images
        return images

    @staticmethod
    def image_save(prompt: str, images: T.List[Image.Image], task_id: str):
        save_dir = os.path.join(env.SAVE_DIR, task_id)
        os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt)
        image_urls = []

        for i, image in enumerate(images):
            filename = f'{str(i).zfill(2)}.webp'
            save_path = os.path.join(env.SAVE_DIR, task_id, filename)
            image_url = os.path.join(task_id, filename)
            image.save(save_path)
            image_urls.append(image_url)

        return image_urls