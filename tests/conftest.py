import pytest

from fastapi.testclient import TestClient
from app.server import create_app

from .mocks import StableDiffusionServiceMock
from app.stable_diffusion.service import StableDiffusionService
from PIL import Image
import io


@pytest.fixture(scope="session")
def client():
    app = create_app()

    app.dependency_overrides[StableDiffusionService] = StableDiffusionServiceMock

    with TestClient(app=app) as client:
        yield client


@pytest.fixture(scope="function")
def image_bytes():
    image = Image.new("RGB", (512, 512), (0, 0, 0))
    buf = io.BytesIO()
    image.save(buf, "WEBP")
    return buf.getvalue()
