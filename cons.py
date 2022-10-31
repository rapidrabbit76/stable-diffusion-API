from time import sleep, time
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
app.conf.accept_content = ["application/json", "application/x-python-serialize"]

task = app.signature("add")

result = task.delay(1, 7)
print(result.id, result.get(10))


image = Image.new("RGB", size=(1024, 1024), color=(0, 0, 0))
task = app.signature("test.imageprocessing")
result = task.delay(image, (1024, 512))
image_ = result.get(10)
print(image_.size)


# # result = task.apply_async((1, 7))
# # result = task.delay(1,7)
# print(result.id, result.get(10))
