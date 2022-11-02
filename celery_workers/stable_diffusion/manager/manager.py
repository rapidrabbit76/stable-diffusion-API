import typing as T
import torch
import sys
from random import randint
from .models import (
    build_text2image_pipeline,
    build_image2image_pipeline,
    build_inpaint_pipeline,
)

from .schema import (
    InpaintTask,
    Text2ImageTask,
    Image2ImageTask,
)
from core.settings import env


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

    @torch.inference_mode()
    def predict(
        self,
        task: _StableDiffusionTask,
    ):
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

        return images

    def _get_generator(self, task: _StableDiffusionTask, device: str):
        generator = torch.Generator(device=device)
        seed = task.seed
        seed = seed if seed else randint(1, sys.maxsize)
        seed = seed if seed > 0 else randint(1, sys.maxsize)
        generator.manual_seed(seed)
        return generator
