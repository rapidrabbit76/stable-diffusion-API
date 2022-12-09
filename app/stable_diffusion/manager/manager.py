import os
from functools import lru_cache
import typing as T
import torch

torch.backends.cudnn.benchmark = True
import sys
from random import randint
from service_streamer import ThreadedStreamer
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from app.stable_diffusion.manager.schema import (
    InpaintTask,
    Text2ImageTask,
    Image2ImageTask,
)
from core.settings import get_settings

from core.utils.convert_script import conver_ckpt_to_diff

from functools import lru_cache

env = get_settings()

_StableDiffusionTask = T.Union[
    Text2ImageTask,
    Image2ImageTask,
    InpaintTask,
]


@lru_cache()
def build_pipeline(repo: str, device: str, enable_attention_slicing: bool):

    # convert ckpt to diffusers
    if repo.lower().endswith(".ckpt") and os.path.exists(repo):
        dump_path = repo[:-5]
        repo = conver_ckpt_to_diff(ckpt_path=repo, dump_path=dump_path)

    pipe = DiffusionPipeline.from_pretrained(
        repo,
        torch_dtype=torch.float16,
        revision="fp16",
        custom_pipeline="lpw_stable_diffusion",
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, clip_input: (images, False)

    if enable_attention_slicing:
        pipe.enable_attention_slicing()

    pipe = pipe.to(device)
    return pipe


build_pipeline(
    repo=env.MODEL_ID,
    device=env.CUDA_DEVICE,
    enable_attention_slicing=env.ENABLE_ATTENTION_SLICING,
)


class StableDiffusionManager:
    def __init__(self):
        self.pipe = build_pipeline(
            repo=env.MODEL_ID,
            device=env.CUDA_DEVICE,
            enable_attention_slicing=env.ENABLE_ATTENTION_SLICING,
        )

    @torch.inference_mode()
    def predict(
        self,
        batch: T.List[_StableDiffusionTask],
    ):
        task = batch[0]
        pipeline = self.pipe
        if isinstance(task, Text2ImageTask):
            pipeline = self.pipe.text2img
        elif isinstance(task, Image2ImageTask):
            pipeline = self.pipe.img2img
        elif isinstance(task, InpaintTask):
            pipeline = self.pipe.inpaint
        else:
            raise NotImplementedError

        device = env.CUDA_DEVICE

        generator = self._get_generator(task, device)
        with torch.autocast("cuda" if device != "cpu" else "cpu"):
            task = task.dict()
            del task["seed"]
            images = pipeline(**task, generator=generator).images
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
