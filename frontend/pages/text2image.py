import typing as T
import os
import sys
import streamlit as st
import requests
from settings import load_config, get_settings
from utils import image_grid
from task import Task

load_config("Stable-Diffusion: text2image")

import requests

env = get_settings()

URL = os.path.join(env.API_URL, Task.TEXT2IMAGE)


def main():
    st.title(f"{env.ST_TITLE}: text to image")
    st.sidebar.markdown("# Text to Image task")
    prompt = st.text_area(
        label="Text Prompt",
        value="A fantasy landscape, trending on artstation",
        key="prompt",
    )
    negative_prompt = st.text_area(
        label="Negative Text Prompt",
        placeholder="Text Prompt",
        key="nega-prompt",
    )
    st.markdown("---")

    with st.sidebar as bar, st.form("key") as form:
        num_images = st.slider(
            "Number of Image",
            min_value=1,
            max_value=8,
            step=1,
        )
        steps = st.slider(
            "Number of Steps",
            min_value=1,
            max_value=50,
            step=1,
            value=25,
        )
        guidance_scale = st.slider(
            "Guidance scale",
            min_value=0.1,
            max_value=20.0,
            value=7.5,
            step=0.01,
        )
        height = st.select_slider(
            "Height", options=[size for size in range(128, 1025, 64)], value=512
        )
        width = st.select_slider(
            "Width", options=[size for size in range(128, 1025, 64)], value=512
        )
        seed = st.number_input("Seed", value=-1)

        summit = st.form_submit_button("Predict")

    if summit:
        image_urls = predict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=int(steps),
            num_images=int(num_images),
            guidance_scale=float(guidance_scale),
            height=height,
            width=width,
            seed=seed,
        )
        image_grid(image_urls)


def predict(
    prompt: str,
    negative_prompt: str,
    steps: int,
    num_images: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
) -> T.List[str]:
    prompt = " " if prompt is None else prompt
    negative_prompt = "" if negative_prompt is None else negative_prompt
    res = requests.post(
        URL,
        data={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "seed": seed,
        },
        headers={},
    )
    if not res.ok:
        st.error(res.json())
    output = res.json()
    image_urls = output["image_urls"]
    return image_urls


main()
