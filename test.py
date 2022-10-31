from celery import Celery
from PIL import Image

__BROKER__ = "pyamqp://guest@localhost:5672"
__BACKEND__ = "redis://localhost:6379"

app = Celery(
    "tasks",
    broker=__BROKER__,
    backend=__BACKEND__,
)
app.conf.task_serializer = "pickle"
app.conf.result_serializer = "pickle"
app.conf.accept_content = [
    "application/json",
    "application/x-python-serialize",
]


@app.task(name="add")
def add(x, y):
    result = x + y
    return result


@app.task
def imageprocessing(
    image: Image.Image,
    size,
) -> Image.Image:
    print(size)

    image = image.resize(size)
    return image
