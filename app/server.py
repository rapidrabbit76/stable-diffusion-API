from datetime import datetime
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

import api
from core.settings import get_settings
from core.dependencies import models

env = get_settings()


def init_router(app: FastAPI):
    app.mount(
        '/images',
        StaticFiles(directory=env.SAVE_DIR),
        name='result image',
    )
    app.include_router(
        api.StableDiffusionRouter,
        prefix="/stable-diffusion",
    )
    app.router.redirect_slashes = False


def create_app() -> FastAPI:
    app = FastAPI(redoc_url=None)

    init_cors(app)
    init_middleware(app)
    init_router(app)
    init_settings(app)
    return app


def init_cors(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=env.CORS_ALLOW_ORIGINS,
    )


def init_middleware(app: FastAPI):
    pass


def init_settings(app: FastAPI):
    @app.on_event("startup")
    def startup_event():
        pass

    @app.on_event("shutdown")
    def shutdown_event():
        pass

    @app.get("/")
    async def index():
        """ELB check"""
        current_time = datetime.utcnow()
        msg = f"Notification API (UTC: {current_time.strftime('%Y.%m.%d %H:%M:%S')})"
        return Response(msg)


app = create_app()