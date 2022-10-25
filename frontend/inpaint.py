import typing as T
import os
from io import BytesIO
import streamlit as st
import requests
from PIL import Image
from streamlit_image_comparison import image_comparison
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from settings import load_config, get_settings
from task import Task

load_config("Stable-Diffusion: inpaint task")

import requests

env = get_settings()

URL = os.path.join(env.API_URL, Task.INPAINT)


def main():
    with st.sidebar:
        stroke_width = st.slider("Stroke width: ", 1, 50, 25)

        with st.form("metadata"):
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

    st.title(f"{env.ST_TITLE}: inpaint")
    prompt = st.text_area(
        label="Text Prompt",
        placeholder="Text Prompt",
        key="prompt",
    )

    negative_prompt = st.text_area(
        label="Negative Text Prompt",
        placeholder="Text Prompt",
        key="nega-prompt",
    )

    init_image = st.file_uploader("Init image", env.IMAGE_TYPES)

    if init_image:
        background_image = Image.open(init_image)
        w, h = background_image.size
        canvas = st_canvas(
            background_image=background_image,
            stroke_width=stroke_width,
            background_color="#FFFFFFFF",
            width=w,
            height=h,
            key="inpaint-canvas",
        )

    st.markdown("---")
    if summit and not init_image:
        st.warning("Input init Image")
        return -1

    if summit:
        mask_image = get_mask_image(canvas)
        mask_image_bytes = BytesIO()
        mask_image.save(mask_image_bytes, format="WEBP")

        image_urls = predict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=int(steps),
            init_image=init_image.getvalue(),
            mask_image=mask_image_bytes.getvalue(),
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


def get_mask_image(canvas) -> Image.Image:
    mask_image = Image.fromarray(canvas.image_data)
    new_mask_image = Image.new("RGBA", mask_image.size, "WHITE")
    new_mask_image.paste(mask_image, mask=mask_image)
    new_mask_image = new_mask_image.convert("RGB")
    new_mask_image = ImageOps.invert(new_mask_image)
    return new_mask_image


def predict(
    prompt: str,
    negative_prompt: str,
    steps: int,
    init_image: bytes,
    mask_image: bytes,
    guidance_scale: float,
    strength: float,
    seed: int,
) -> T.List[str]:
    print(prompt)
    prompt = " " if prompt is None else prompt
    negative_prompt = "" if negative_prompt is None else negative_prompt
    files = [
        ("init_image", ("image.webp", init_image, "image/*")),
        ("mask_image", ("image.webp", mask_image, "image/*")),
    ]
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
