FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
WORKDIR /app

RUN apt-get update && apt install -y \ 
    libgl1-mesa-glx \
    curl  \
    libglib2.0-0 && apt-get clean 


COPY  requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


ADD . /app
RUN chmod +x /app/docker/api/start.sh
ENTRYPOINT /app/docker/api/start.sh
