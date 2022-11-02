import typing as T
import os
from PIL import Image
import json
from itertools import chain, islice
from stable_diffusion import (
    StableDiffusionManager,
)
from stable_diffusion.manager.schema import Text2ImageTask


# from core.celery_app import app
from core.settings import env
from celery import Celery, states
from random import randint


app = Celery(
    "tasks",
    broker=env.CELERY_BROKER,
    backend=env.CELERY_BACKEND,
)
app.conf.task_serializer = "pickle"
app.conf.result_serializer = "pickle"
app.conf.accept_content = [
    "application/json",
    "application/x-python-serialize",
]

from core.settings import env


manager = StableDiffusionManager()


def data_to_batch(datasets: T.List[T.Any], batch_size: int):
    iterator = iter(datasets)
    for first in iterator:
        yield list(chain([first], islice(iterator, batch_size - 1)))


@app.task(bind=True)
def predict(self, **kwargs):
    task_id = self.request.id
    self.update_state(state=states.RECEIVED)
    task = kwargs.pop("task")
    self.update_state(state="PROGRESS")
    if task == "text2image":
        images = task_text2image(**kwargs)
    print(images, type(images))
    image_uris = image_save_local(images=images, task_id=task_id, info=kwargs)
    return {
        "prompt": kwargs.get("prompt"),
        "image_uris": image_uris,
    }


def task_text2image(
    prompt: T.Union[str, T.List[str]],
    negative_prompt: T.Union[str, T.List[str]],
    num_images: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: T.Optional[int],
    **kwargs,
):
    prompts = [prompt] * num_images
    tasks = [
        Text2ImageTask(
            prompt=prompt,
            negative_prompt=[negative_prompt] * len(prompt),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
        )
        for prompt in data_to_batch(prompts, batch_size=env.MB_BATCH_SIZE)
    ]

    results = []

    for task in tasks:
        result = manager.predict(task)
        results += result
    return results


def task_image2image():
    pass


def task_inpaints():
    pass


def image_save_local(
    images: T.List[Image.Image],
    task_id: str,
    info: dict,
):
    save_dir = os.path.join(env.SAVE_DIR, task_id)
    os.makedirs(save_dir)
    image_uris = []

    with open(os.path.join(save_dir, "info.json"), "w") as f:
        json.dump(info, f)

    for i, image in enumerate(images):
        filename = f"{str(i).zfill(2)}.webp"
        save_path = os.path.join(env.SAVE_DIR, task_id, filename)
        image_path = os.path.join(task_id, filename)
        image.save(save_path)
        image_uris.append(image_path)
    return image_uris
