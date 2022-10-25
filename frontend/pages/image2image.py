import typing as T
import os
import streamlit as st
import requests
from PIL import Image
from streamlit_image_comparison import image_comparison
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
        key="prompt",
    )

    negative_prompt = st.text_area(
        label="Negative Text Prompt",
        placeholder="Text Prompt",
        key="nega-prompt",
    )
    init_image = st.file_uploader(
        "Init image",
        env.IMAGE_TYPES,
    )

    st.markdown("---")

    with st.sidebar as bar, st.form("key") as form:
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
        strength = st.slider(
            "strength",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01,
        )
        seed = st.number_input("Seed", value=-1)
        summit = st.form_submit_button("Predict")

    if summit:
        image_urls = predict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=int(steps),
            init_image=init_image.getvalue(),
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            seed=seed,
        )

        c1, c2 = st.columns([1, 1])
        c1.title("Origin")
        c1.image(init_image)

        c2.title("Result")
        c2.image(image_urls)

        image_comparison(
            img1=Image.open(init_image),
            img2=image_urls[-1],
            label1="origin",
            label2="diffusion",
        )


def predict(
    prompt: str,
    negative_prompt: str,
    init_image: bytes,
    steps: int,
    guidance_scale: float,
    strength: float,
    seed: int,
) -> T.List[str]:
    prompt = " " if prompt is None else prompt
    negative_prompt = "" if negative_prompt is None else negative_prompt
    files = [("init_image", ("image.jpg", init_image, "image/*"))]
    res = requests.post(
        URL,
        data={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "num_images": 1,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "seed": seed,
        },
        files=files,
        headers={},
    )
    if not res.ok:
        st.error(res.text)
    output = res.json()
    image_urls = output["image_urls"]
    return image_urls


main()
