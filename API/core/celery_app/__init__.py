from celery import Celery
from PIL import Image
from core.settings import env


def get_celery_app() -> Celery:
    celery_app = Celery(
        "stable-diffusion",
        broker=env.CELERY_BROKER,
        backend=env.CELERY_BACKEND,
    )
    celery_app.conf.task_serializer = "pickle"
    celery_app.conf.result_serializer = "pickle"
    celery_app.conf.accept_content = [
        "application/json",
        "application/x-python-serialize",
    ]
    return celery_app
