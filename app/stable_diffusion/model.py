from fastapi import Depends
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from functools import lru_cache
from service_streamer import ThreadedStreamer
from core.settings import get_settings
from loguru import logger

env = get_settings()


@lru_cache(maxsize=1)
def build_pipeline() -> StableDiffusionPipeline:
    model_id = env.MODEL_ID
    device = env.CUDA_DEVICE
    safety_check: bool = False

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    logger.info('stable diffusion model loading...')
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=env.HUGGINGFACE_TOKEN,
        scheduler=scheduler,
    )
    logger.info('init stable diffusion pipeline')
    pipe = pipe.to(device)
    logger.info(f'model to device({device})')
    pipe.safety_checker = lambda images, clip_input: (images, safety_check)
    return pipe
