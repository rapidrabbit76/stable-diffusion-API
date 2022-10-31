import typing as T
from functools import lru_cache
import torch
from fastapi import Depends
from service_streamer import ThreadedStreamer
from app.stable_diffusion.manager.manager import StableDiffusionManager
from app.stable_diffusion.manager.schema import (
    InpaintTask,
    Text2ImageTask,
    Image2ImageTask,
)
from app.stable_diffusion.service import StableDiffusionService
from PIL import Image

_StableDiffusionTask = T.Union[
    Text2ImageTask,
    Image2ImageTask,
    InpaintTask,
]


class ManagerMock(StableDiffusionManager):
    def __init__(self):
        pass

    @torch.inference_mode()
    def predict(self, batch: T.List[_StableDiffusionTask]):
        batch_size = len(batch)

        task = batch[0]
        if isinstance(task, Text2ImageTask):
            images = [Image.new("RGB", (task.width, task.height), (0, 0, 0))]
        elif isinstance(task, Image2ImageTask):
            size = task.init_image.size
            images = [Image.new("RGB", size, (0, 0, 0))]
        elif isinstance(task, InpaintTask):
            size = task.init_image.size
            images = [Image.new("RGB", size, (0, 0, 0))]

        if isinstance(task.prompt, list):
            images *= len(task.prompt)

        return [images]


def build_mock_streamer() -> ThreadedStreamer:
    manager = ManagerMock()
    streamer = ThreadedStreamer(
        manager.predict,
        batch_size=1,
        max_latency=0,
    )
    return streamer


class StableDiffusionServiceMock(StableDiffusionService):
    def __init__(self) -> None:
        super().__init__(build_mock_streamer())

    @staticmethod
    def image_save(images: T.List[Image.Image], task_id: str, info: dict):
        return [f"{task_id}/{str(i).zfill(2)}.webp" for i, image in enumerate(images)]
