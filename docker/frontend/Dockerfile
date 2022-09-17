FROM python:3.8.12-slim-buster
WORKDIR /app

RUN apt-get update && apt install -y \ 
    libgl1-mesa-glx \
    curl \
    libglib2.0-0 && apt-get clean \
    python -m pip install --upgrade pip 

RUN pip install --no-cache-dir \
    streamlit==1.12.2 \
    requests==2.27.1 \ 
    requests-toolbelt==0.9.1 

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY . /app

EXPOSE 8501

CMD ["sh", "-c", "streamlit run --server.address=0.0.0.0 /app/inpaint.py"]