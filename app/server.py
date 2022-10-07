from fastapi.staticfiles import StaticFiles
import typing as T

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware

import api
from core.settings import env


def init_router(app: FastAPI):
    app.mount(
        "/images",
        StaticFiles(directory=env.SAVE_DIR),
        name="result image",
    )
    app.include_router(api.StableDiffusionRouter)
    app.include_router(api.HomeRouter)
    app.router.redirect_slashes = False


def create_app() -> FastAPI:
    app = FastAPI(
        redoc_url=None,
        middleware=init_middleware(),
    )
    init_router(app)
    return app


def init_middleware() -> T.List[Middleware]:
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=env.CORS_ALLOW_ORIGINS,
            allow_credentials=env.CORS_CREDENTIALS,
            allow_methods=env.CORS_ALLOW_METHODS,
            allow_headers=env.CORS_ALLOW_HEADERS,
        ),
    ]
    return middleware


def init_settings(app: FastAPI):
    @app.on_event("startup")
    def startup_event():
        from core.dependencies import models

    @app.on_event("shutdown")
    def shutdown_event():
        pass


app = create_app()
init_settings(app)
