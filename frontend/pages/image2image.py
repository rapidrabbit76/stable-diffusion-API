import typing as T
import os
import sys
import streamlit as st
import requests
from settings import load_config, get_settings
from task import Task

load_config("Stable-Diffusion: image2image task")

import requests

env = get_settings()

URL = os.path.join(env.API_URL, Task.IMAGE2IMAGE)


def main():
    st.title(f"{env.ST_TITLE}: Image to Image")
    st.sidebar.markdown("#Image to Image task")
    prompt = st.text_area(
        label="Text Prompt",
        value="A fantasy landscape, trending on artstation",
        key="image2image-prompt",
    )
    
    negative_prompt = st.text_area(
        label="Negative Text Prompt",
        placeholder="Text Prompt",
        key="image2image-nega-prompt",
    )
    init_image = st.file_uploader(
        "Init image",
        env.IMAGE_TYPES,
    )
    if init_image:
        st.image(init_image)

    st.markdown("---")

    with st.sidebar as bar, st.form("key") as form:
        guidance_scale = st.slider(
            "Guidance scale",
            min_value=0.1,
            max_value=20.0,
            value=7.5,
            step=0.01,
        )
        strength = st.slider(
            "strength",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01,
        )
        seed = st.number_input(
            "Seed",
            value=203,
        )
        summit = st.form_submit_button("Predict")

    if summit:
        image_urls = predict(
            prompt=prompt,
            init_image=init_image.getvalue(),
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            seed=seed,
        )
        st.image(image_urls)


def predict(
    prompt: str,
    init_image: bytes,
    guidance_scale: float,
    strength: float,
    seed: int,
) -> T.List[str]:
    files = [("init_image", ("image.jpg", init_image, "image/*"))]
    res = requests.post(
        URL,
        data={
            "prompt": prompt,
            "num_images": 1,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "seed": seed,
        },
        files=files,
        headers={},
    )
    output = res.json()
    task_id = output["task_id"]
    image_urls = output["image_urls"]
    return image_urls


main()
