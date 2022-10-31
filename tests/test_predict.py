import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image


@pytest.mark.parametrize("num_images", [i for i in range(1, 4)])
def test_text2image(client: TestClient, num_images: int):
    res = client.post(
        "/text2image",
        data={
            "prompt": "ABC",
            "num_images": num_images,
        },
        headers={},
    )

    assert res.ok
    result = res.json()

    assert len(result["image_urls"]) == num_images


@pytest.mark.parametrize("num_images", [i for i in range(1, 4)])
def test_image2image(
    client: TestClient,
    num_images: int,
):
    image = Image.new("RGB", (512, 512), (0, 0, 0))
    buf = io.BytesIO()
    image.save(buf, "WEBP")
    image_bytes = buf.getvalue()

    res = client.post(
        "/image2image",
        files=[
            ("init_image", ("image.jpg", image_bytes, "image/*")),
        ],
        data={
            "prompt": "ABC",
            "num_images": num_images,
        },
        headers={},
    )

    assert res.ok
    result = res.json()

    assert len(result["image_urls"]) == num_images


# @pytest.mark.parametrize("num_images", [i for i in range(1, 4)])
# def test_inpaints(client: TestClient, image_bytes: bytes, num_images: int):
#     res = client.post(
#         "/text2image",
#         files=[
#             ("init_image", ("image.jpg", image_bytes, "image/*")),
#         ],
#         data={
#             "prompt": "ABC",
#             "num_images": num_images,
#         },
#         headers={},
#     )

#     assert res.ok
#     result = res.json()

#     assert len(result["image_urls"]) == num_images
