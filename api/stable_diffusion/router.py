import typing as T
import os
from uuid import uuid4

from fastapi import Form, Depends, UploadFile, File
from fastapi_restful.cbv import cbv
from fastapi_restful.inferring_router import InferringRouter
from PIL import Image

from .request import random_seed, read_image
from .response import StableDiffussionResponse
from app.stable_diffusion.service import StableDiffusionService
from core.settings import get_settings

router = InferringRouter()
env = get_settings()


@cbv(router)
class StableDiffusion:
    svc: StableDiffusionService = Depends(StableDiffusionService)

    @router.post("/text2image", response_model=StableDiffussionResponse)
    def text2image(
        self,
        prompt: str = Form(),
        negative_prompt: str = Form(default=""),
        num_images: int = Form(1, description="num images", ge=1, le=8),
        guidance_scale: float = Form(7.5, description="guidance_scale", gt=0, le=20),
        height: int = Form(512, description="result height"),
        width: int = Form(512, description="result width"),
        seed: int = Depends(random_seed),
    ):
        task_id = str(uuid4())

        images = self.svc.text2image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
        )

        info = {
            "task": "text2image",
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "seed": seed,
        }
        image_paths = self.svc.image_save(images, task_id, info=info)
        urls = [os.path.join(env.IMAGESERVER_URL, path) for path in image_paths]

        response = StableDiffussionResponse(
            prompt=prompt,
            task_id=task_id,
            image_urls=urls,
        )
        return response

    @router.post("/image2image", response_model=StableDiffussionResponse)
    def image2image(
        self,
        prompt: str = Form(),
        negative_prompt: str = Form(default=""),
        init_image: UploadFile = File(...),
        num_images: int = Form(1, description="num images", ge=1, le=8),
        strength: float = Form(0.8, ge=0, le=1.0),
        guidance_scale: float = Form(7.5, description="guidance_scale", gt=0, le=20),
        seed: int = Depends(random_seed),
    ):
        init_image = read_image(init_image)
        task_id = str(uuid4())

        images = self.svc.image2image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            num_images=num_images,
            strength=strength,
            num_inference_steps=60,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        info = {
            "task": "image2image",
            "prompt": prompt,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        image_paths = self.svc.image_save(images, task_id, info=info)
        init_image.save(os.path.join(env.SAVE_DIR, task_id, "init_image.webp"))
        urls = [os.path.join(env.IMAGESERVER_URL, path) for path in image_paths]

        response = StableDiffussionResponse(
            prompt=prompt,
            task_id=task_id,
            image_urls=urls,
        )
        return response

    @router.post("/inpaint", response_model=StableDiffussionResponse)
    def inpaint(
        self,
        prompt: str = Form(),
        negative_prompt: str = Form(default=""),
        init_image: UploadFile = File(...),
        mask_image: UploadFile = File(...),
        num_images: int = Form(1, description="num images", ge=1, le=8),
        strength: float = Form(0.8, ge=0, le=1.0),
        guidance_scale: float = Form(7.5, description="guidance_scale", gt=0, le=20),
        seed: int = Depends(random_seed),
    ):
        init_image = read_image(init_image)
        mask_image = read_image(mask_image)

        task_id = str(uuid4())
        images = self.svc.inpaint(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            mask_image=mask_image,
            strength=strength,
            num_images=num_images,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        info = {
            "task": "inpaint",
            "prompt": prompt,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        image_paths = self.svc.image_save(images, task_id, info=info)
        init_image.save(os.path.join(env.SAVE_DIR, task_id, "init_image.webp"))
        mask_image.save(os.path.join(env.SAVE_DIR, task_id, "mask_image.webp"))
        urls = [os.path.join(env.IMAGESERVER_URL, path) for path in image_paths]

        response = StableDiffussionResponse(
            prompt=prompt,
            task_id=task_id,
            image_urls=urls,
        )
        return response
