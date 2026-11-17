import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .config import Settings
from .exceptions import APIError, log_exception, respond
from .image_utils import prune_images
from .logging_config import configure_logging, prune_logs
from .middleware import CorsMiddleware, RequestMiddleware
from .model_service import ModelService
from .routes import health, images, models

log = logging.getLogger("qwen")


async def prune(settings):
    try:
        await run_in_threadpool(prune_logs, settings)
    except OSError as exc:
        log_exception("log_retention_failed", exc)
    roots = {settings.output_dir} if settings.save_outputs else set()
    if settings.save_uploads:
        roots.add(settings.upload_dir)
    for root in roots:
        try:
            await run_in_threadpool(prune_images, root, settings.output_retention_days)
        except OSError as exc:
            log_exception("image_retention_failed", exc)


async def retention_loop(settings):
    while True:
        await asyncio.sleep(3600)
        await prune(settings)


def create_app(settings=None, model_factory=ModelService):
    settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app):
        handler = configure_logging(settings)
        cleanup = None
        try:
            await prune(settings)
            await run_in_threadpool(app.state.model.load)
            cleanup = asyncio.create_task(retention_loop(settings))
            yield
        finally:
            if cleanup is not None:
                cleanup.cancel()
                with suppress(asyncio.CancelledError):
                    await cleanup
            if app.state.model.jobs:
                await asyncio.gather(*app.state.model.jobs, return_exceptions=True)
            log.removeHandler(handler)
            handler.close()

    app = FastAPI(
        title="Qwen Image API", lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None, redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        swagger_ui_oauth2_redirect_url=None,
    )
    app.state.settings = settings
    app.state.model = model_factory(settings)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(images.router)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return respond(request, APIError(422, "validation_error", "Invalid request fields"))

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        code, message = {
            400: ("invalid_request", "Invalid request"),
            404: ("not_found", "Endpoint not found"),
            405: ("method_not_allowed", "Method not allowed"),
        }.get(exc.status_code, ("http_error", "HTTP request failed"))
        return respond(request, APIError(exc.status_code, code, message, exc.headers))

    origins = [item.strip() for item in settings.cors_allowed_origins.split(",") if item.strip()]
    app.add_middleware(RequestMiddleware, settings=settings)
    if origins:
        app.add_middleware(
            CorsMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"], expose_headers=["X-Request-ID"],
        )

    original_openapi = app.openapi

    def openapi():
        schema = original_openapi()
        schema["components"]["securitySchemes"] = {"BearerAuth": {"type": "http", "scheme": "bearer"}}
        for path, operations in schema["paths"].items():
            if path.startswith("/v1/"):
                for operation in operations.values():
                    operation["security"] = [{"BearerAuth": []}]
        return schema

    app.openapi = openapi
    return app
