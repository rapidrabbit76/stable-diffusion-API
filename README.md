# Unofficial Fastapi implementation Stable-Diffusion API

UNOFFICIAL, [Stable-Diffusion](https://github.com/CompVis/stable-diffusion) api using FastAPI

# Samples
TODO


# Requirements


# API

## /stable-diffusion

inputs:

    - prompt(str): text prompt
    - num_images(int): number of images
    - guidance_scale(float): guidance scale for stable-diffusion
    - num_inference_steps(int): diffusion itr count
    - height(int): image height
    - width(int): image width

outputs:

    - prompt(str): input text prompt
    - task_id(str): uuid4 hex string
    - image_urls(str): generated images url



# Environment variable


```bash
# env setting is in 
>> ./core/settings/settings.py
```

| Name               | Default                       | Desc                                              |
| ------------------ | ----------------------------- | ------------------------------------------------- |
| MODEL_ID           | CompVis/stable-diffusion-v1-4 | tagger embedding model part                       |
| CUDA_DEVICE        | "cpu"                         | target cuda device                                |
| CUDA_DEVICES       | [0]                           | visible cuda device                               |
| MB_BATCH_SIZE      | 64                            | Micro Batch: MAX Batch size                       |
| HUGGINGFACE_TOKEN  | None                          | huggingface access token                          |
| IMAGESERVER_URL    | None                          | result image base url                             |
| SAVE_DIR           | static                        | result image save dir                             |
| CORS_ALLOW_ORIGINS | [*]                           | cross origin resource sharing setting for FastAPI |

# RUN from code 

## 1. install python Requirements
```bash
pip install -r requirements.txt
```

## 2. downlaod and caching huggingface model
```bash
python huggingface_model_download.py
# check stable-diffusion model in huggingface cache dir 
[[ -d ~/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4 ]] && echo "exist"
>> exist
```

## 3. update settings.py in ./core/settings/settings.py
```python
# example
...
class Settings(
    ...
):
    HUGGINGFACE_TOKEN: str = "YOUR HUGGINGFACE ACCESS TOKEN"
    IMAGESERVER_URL: str  = "http://localhost:3000/images"
    SAVE_DIR: str = 'static'
    ...
```

## 4. RUN API by uvicorn
```bash
cd /REPO/ROOT/DIR/PATH
python3 -m uvicorn app.server:app \
    --host 0.0.0.0 \
    --port 3000 \
    --workers 1 
```


# RUN using Docker (docker-compose)

## 1. Image Build or pull
```bash
docker-compose build
# or image pull (docker-compose)
docker-compose pull
# or image pull from docker hub
docker pull ys2lee/stable-difussion-api:latest
```

## 2. downlaod and caching huggingface model
```bash
python huggingface_model_download.py
# check stable-diffusion model in huggingface cache dir 
[[ -d ~/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4 ]] && echo "exist"
>> exist
```

## 3. update docker-compose.yaml file in repo root
```yaml
version: "3.7"

services:
  stable-diffusion:
    ...
    volumes:
      # mount huggingface model cache dir path to container root user home dir
      - /home/{USER NAME}/.cache/huggingface:/root/.cache/huggingface
      - ...
    environment:
      ...
      HUGGINGFACE_TOKEN: {YOUR HUGGINGFACE ACCESS TOKEN}
      ...

    deploy:
      ...
```

## 4. Container RUN
```bash
docker-compose up -d
```






## Curl CMD
```java
TODO
```


## Streamlit demo
```java
TODO
```


## References
- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [huggingface, stable-diffusion](https://huggingface.co/CompVis)
- [teamhide/fastapi-boilerplate](https://github.com/teamhide/fastapi-boilerplate)