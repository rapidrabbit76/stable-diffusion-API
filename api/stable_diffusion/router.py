import os
from uuid import uuid4

from fastapi import Form, Depends
from fastapi_restful.cbv import cbv
from fastapi_restful.inferring_router import InferringRouter

from .request import PromptRequest
from .response import StableDiffussionResponse
from app.stable_diffusion.service import StableDiffusionService
from core.settings import get_settings

router = InferringRouter()
env = get_settings()


@cbv(router)
class StableDiffusion:
    svc: StableDiffusionService = Depends(StableDiffusionService)

    @router.post('/', response_model=StableDiffussionResponse)
    def predict(
            self,
            prompt: str = Form(),
            num_images: int = Form(1, description='num images', ge=1, le=8),
            guidance_scale: float = Form(7.5,
                                         description='guidance_scale',
                                         gt=0),
            num_inference_steps: int = Form(40,ge=4,le=200),
            height: int = Form(512, description='result height'),
            width: int = Form(512, description='result width'),
    ):
        images = self.svc.predict(
            prompt=prompt,
            num_images=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
        )

        task_id = uuid4().hex
        image_paths = self.svc.image_save(prompt, images, task_id)
        urls = [os.path.join(env.IMAGESERVER_URL, path) for path in image_paths]

        response = StableDiffussionResponse(
            prompt=prompt,
            task_id=task_id,
            image_urls=urls,
        )
        return response
