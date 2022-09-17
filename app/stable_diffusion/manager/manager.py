from functools import lru_cache
import typing as T
import torch
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
from core.decorator.singleton import singleton

env = get_settings()

_StableDiffusionTask = T.Union[
    Text2ImageTask,
    Image2ImageTask,
    InpaintTask,
]


@singleton
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
        if isinstance(task, Text2ImageTask):
            images = self.predict_text2image(task)
        elif isinstance(task, Image2ImageTask):
            images = self.predict_image2image(task)
        elif isinstance(task, InpaintTask):
            images = self.predict_inpaint(task)
        return [images]

    def _get_generator(self, task, device):
        generator = torch.Generator(device=device)
        generator.manual_seed(task.seed)
        return generator

    def predict_text2image(self, task: Text2ImageTask):
        device = env.CUDA_DEVICE
        generator = self._get_generator(task, device)
        with torch.autocast("cuda" if device != "cpu" else "cpu"):
            images = self.text2image(**task.dict(), generator=generator)
            if device != "cpu":
                torch.cuda.empty_cache()
        return images

    def predict_image2image(self, task: Image2ImageTask):
        device = env.CUDA_DEVICE
        generator = self._get_generator(task, device)
        with torch.autocast("cuda" if device != "cpu" else "cpu"):
            images = self.image2image(**task.dict(), generator=generator)
            if device != "cpu":
                torch.cuda.empty_cache()
        return images

    def predict_inpaint(self, task: InpaintTask):
        device = env.CUDA_DEVICE
        generator = self._get_generator(task, device)
        with torch.autocast("cuda" if device != "cpu" else "cpu"):
            images = self.inpaint(**task.dict(), generator=generator)

            if device != "cpu":
                torch.cuda.empty_cache()
        return images


@lru_cache(maxsize=1)
def build_streamer() -> ThreadedStreamer:
    manager = StableDiffusionManager()
    streamer = ThreadedStreamer(
        manager.predict,
        batch_size=1,
        max_latency=0,
    )
    return streamer
