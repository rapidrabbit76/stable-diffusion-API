FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
WORKDIR /app

RUN apt-get update && apt install -y \ 
    libgl1-mesa-glx \
    curl  \
    libglib2.0-0 && apt-get clean 

RUN pip install --no-cache-dir \
    fastapi[all]==0.80.0 \
    service-streamer==0.1.2 \
    fastapi-restful==0.4.3 \
    fastapi-health==0.4.0 \
    loguru==0.6.0  \
    gunicorn==20.1.0


RUN pip install --no-cache-dir \
    diffusers==0.3.0 \
    scipy \
    ftfy \
    transformers==4.19.2

ADD . /app
RUN chmod +x /app/docker/api/start.sh
ENTRYPOINT /app/docker/api/start.sh
