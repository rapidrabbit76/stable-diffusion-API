from functools import lru_cache
import typing as T
import torch

torch.backends.cudnn.benchmark = True
import sys
from random import randint
from service_streamer import ThreadedStreamer
from app.stable_diffusion.model import (
    build_text2image_pipeline,
    build_image2image_pipeline,
    build_inpaint_pipeline,
)
from app.stable_diffusion.manager.schema import (
    InpaintTask,
    Text2ImageTask,
    Image2ImageTask,
)
from core.settings import get_settings

env = get_settings()

_StableDiffusionTask = T.Union[
    Text2ImageTask,
    Image2ImageTask,
    InpaintTask,
]


class StableDiffusionManager:
    def __init__(self):
        self.text2image = build_text2image_pipeline()
        self.image2image = build_image2image_pipeline()
        self.inpaint = build_inpaint_pipeline()

    def predict(
        self,
        batch: T.List[_StableDiffusionTask],
    ):
        task = batch[0]
        pipeline = self.text2image
        if isinstance(task, Text2ImageTask):
            pipeline = self.text2image
        elif isinstance(task, Image2ImageTask):
            pipeline = self.image2image
        elif isinstance(task, InpaintTask):
            pipeline = self.inpaint

        device = env.CUDA_DEVICE

        generator = self._get_generator(task, device)
        with torch.autocast("cuda" if device != "cpu" else "cpu"):
            images = pipeline(**task.dict(), generator=generator)
            if device != "cpu":
                torch.cuda.empty_cache()

        return [images]

    def _get_generator(self, task: _StableDiffusionTask, device: str):
        generator = torch.Generator(device=device)
        seed = task.seed
        seed = seed if seed else randint(1, sys.maxsize)
        seed = seed if seed > 0 else randint(1, sys.maxsize)
        generator.manual_seed(seed)
        return generator


@lru_cache(maxsize=1)
def build_streamer() -> ThreadedStreamer:
    manager = StableDiffusionManager()
    streamer = ThreadedStreamer(
        manager.predict,
        batch_size=1,
        max_latency=0,
    )
    return streamer
