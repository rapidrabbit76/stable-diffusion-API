gunicorn app.server:app -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:3000 \
    --workers 1 