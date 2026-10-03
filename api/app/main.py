"""FastAPI app factory (PLAN §2, §5). ``make dev-api`` serves ``app.main:app``.

Lifespan: create the data dirs + SQLite schema, start the job runner, run startup recovery
(``processing`` -> ``failed/interrupted``, ``queued`` -> resubmitted), and on shutdown stop the
runner (a running job ends as ``interrupted``; queued jobs stay queued for the next start).

Every error response uses the contract body ``{"error": {"code", "message"}}``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api import routes_health, routes_interpreter, routes_jobs
from app.api.deps import Services
from app.config import Settings, get_settings
from app.core.errors import DEFAULT_MESSAGES, ErrorCode, PipelineError
from app.core.jobstore import SqliteJobStore
from app.core.runner import JobRunner, PipelineFn
from app.core.storage import LocalJobStorage

log = logging.getLogger("app")

#: HTTP status -> error code for errors raised by Starlette/FastAPI themselves.
_HTTP_STATUS_CODES: dict[int, ErrorCode] = {
    404: ErrorCode.NOT_FOUND,
    409: ErrorCode.NOT_READY,
    413: ErrorCode.FILE_TOO_LARGE,
    415: ErrorCode.UNSUPPORTED_FORMAT,
}


def _error(status: int, code: ErrorCode, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"code": code.value, "message": message}}
    )


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(PipelineError)
    async def _pipeline_error(_: Request, exc: PipelineError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_body())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = f"{where}: {first.get('msg', 'invalid value')}" if where else "Invalid request."
        return _error(422, ErrorCode.INVALID_REQUEST, message)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_STATUS_CODES.get(
            exc.status_code,
            ErrorCode.INTERNAL_ERROR if exc.status_code >= 500 else ErrorCode.INVALID_REQUEST,
        )
        message = exc.detail if isinstance(exc.detail, str) else DEFAULT_MESSAGES[code]
        response = _error(exc.status_code, code, message)
        for key, value in (exc.headers or {}).items():
            response.headers[key] = value
        return response

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return _error(500, ErrorCode.INTERNAL_ERROR, DEFAULT_MESSAGES[ErrorCode.INTERNAL_ERROR])


def _configure_logging() -> None:
    """Make ``app.*`` INFO logs visible under uvicorn without touching its own loggers."""
    if not log.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
        log.addHandler(handler)
    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)


def _default_pipeline() -> PipelineFn:
    from app.pipeline.run import run  # lazy: keeps app import light and cycle-free

    return run


def _default_reinterpret() -> PipelineFn:
    from app.pipeline.run import reinterpret

    return reinterpret


def create_app(
    settings: Settings | None = None,
    pipeline: PipelineFn | None = None,
    reinterpret: PipelineFn | None = None,
) -> FastAPI:
    """Build the app. ``pipeline`` defaults to :func:`app.pipeline.run.run` and ``reinterpret``
    to :func:`app.pipeline.run.reinterpret` (tests inject fakes)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        storage = LocalJobStorage(settings.jobs_dir)
        storage.init()
        store = SqliteJobStore(settings.db_path)
        store.init()
        runner = JobRunner(
            store=store,
            storage=storage,
            settings=settings,
            pipeline=pipeline if pipeline is not None else _default_pipeline(),
            reinterpret=reinterpret if reinterpret is not None else _default_reinterpret(),
        )
        app.state.services = Services(
            settings=settings, storage=storage, store=store, runner=runner
        )
        runner.recover()
        log.info("mimic api %s ready (data dir %s)", __version__, settings.data_dir)
        try:
            yield
        finally:
            await run_in_threadpool(runner.shutdown)
            app.state.services = None

    _configure_logging()
    app = FastAPI(
        title="Mimic API",
        version=__version__,
        description="UI motion reverse engineering: screen recording -> motion spec.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )
    _install_error_handlers(app)
    app.include_router(routes_health.router)
    app.include_router(routes_jobs.router)
    app.include_router(routes_interpreter.router)
    return app


app = create_app()
