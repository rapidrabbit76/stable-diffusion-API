import typing as T
from stable_diffusion import (
    StableDiffusionManager,
    Text2ImageTask,
    Image2ImageTask,
    InpaintTask,
)


manager = StableDiffusionManager()


def task_text2image(
    prompt: T.Union[str, T.List[str]],
    negative_prompt: T.Union[str, T.List[str]],
    num_images: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: T.Optional[int],
):
    pass


def task_image2image():
    pass


def task_inpaints():
    pass
