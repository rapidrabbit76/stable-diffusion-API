from fastapi import Depends
import torch
from functools import lru_cache
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPFeatureExtractor,
    AutoFeatureExtractor,
)
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    LMSDiscreteScheduler,
)
from diffusers import StableDiffusionPipeline
import torch
from core.settings import get_settings
from .pipeline import (
    StableDiffusionText2ImagePipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from loguru import logger

env = get_settings()


@lru_cache(maxsize=1)
def build_text2image_pipeline() -> StableDiffusionText2ImagePipeline:
    vae = build_vae()
    unet = build_unet()
    tokenizer = build_tokenizer()
    text_encoder = build_text_encoder()
    scheduler = build_scheduler()

    logger.info("stable diffusion text2image pipeline loading...")
    pipe = StableDiffusionText2ImagePipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    pipe.enable_attention_slicing()
    return pipe


@lru_cache(maxsize=1)
def build_image2image_pipeline() -> StableDiffusionImg2ImgPipeline:
    vae = build_vae()
    unet = build_unet()
    tokenizer = build_tokenizer()
    text_encoder = build_text_encoder()
    scheduler = build_scheduler()

    logger.info("stable diffusion image2image pipeline loading...")
    pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    return pipe


@lru_cache(maxsize=1)
def build_inpaint_pipeline() -> StableDiffusionInpaintPipeline:
    vae = build_vae()
    unet = build_unet()
    tokenizer = build_tokenizer()
    text_encoder = build_text_encoder()
    scheduler = build_scheduler()

    logger.info("stable diffusion inpaint pipeline loading...")
    pipe = StableDiffusionInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    return pipe


@lru_cache(maxsize=1)
def build_inpaint_pipeline() -> StableDiffusionPipeline:
    vae = build_vae()
    unet = build_unet()
    tokenizer = build_tokenizer()
    text_encoder = build_text_encoder()
    scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )

    logger.info("stable diffusion text2image pipeline loading...")
    pipe = StableDiffusionInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    return pipe


@lru_cache(maxsize=1)
def build_vae() -> AutoencoderKL:
    logger.info("stable diffusion model VAE loading...")
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=env.HUGGINGFACE_TOKEN,
    )
    vae = vae.to(env.CUDA_DEVICE)
    return vae


@lru_cache(maxsize=1)
def build_unet():
    model_id = env.MODEL_ID
    device = env.CUDA_DEVICE

    logger.info("stable diffusion model Unet loading...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=env.HUGGINGFACE_TOKEN,
    )
    unet = unet.to(device)
    return unet


@lru_cache(maxsize=1)
def build_tokenizer():
    logger.info("stable diffusion model tokenizer loading...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    return tokenizer


@lru_cache(maxsize=1)
def build_text_encoder():
    model_id = env.MODEL_ID
    device = env.CUDA_DEVICE

    logger.info("stable diffusion model text encoder loading...")
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    text_encoder = text_encoder.to(device)
    return text_encoder


@lru_cache(maxsize=1)
def build_scheduler():
    logger.info("stable diffusion model text scheduler loading...")
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    return scheduler
