from celery import Celery
from PIL import Image
from core.settings import env


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
