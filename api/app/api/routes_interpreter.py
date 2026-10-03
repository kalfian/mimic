"""``POST /api/interpreter/check``: live connectivity check of the configured interpreter.

Unlike ``/api/health`` this may make real network calls / spawn the CLI, so it is an explicit
POST the UI triggers on demand. The work lives in :func:`app.interpret.check.check_interpreter`.

Secrets: the configured base URL may embed credentials and the API key is secret, so nothing
from the exception (whose text can contain URLs) is logged or returned, only its type name.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.api.deps import ServicesDep
from app.api.schemas import InterpreterCheck
from app.core.errors import DEFAULT_MESSAGES, ErrorCode
from app.interpret.check import check_interpreter

log = logging.getLogger(__name__)

router = APIRouter(tags=["interpreter"])


@router.post("/api/interpreter/check", response_model=InterpreterCheck)
async def check_interpreter_route(services: ServicesDep) -> JSONResponse:
    try:
        result = await run_in_threadpool(check_interpreter, services.settings)
    except Exception as exc:  # never log/return exc text: it may contain the URL or key
        log.error("interpreter check raised %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": ErrorCode.INTERNAL_ERROR.value,
                    "message": DEFAULT_MESSAGES[ErrorCode.INTERNAL_ERROR],
                }
            },
        )
    if hasattr(result, "model_dump"):
        return JSONResponse(content=result.model_dump(mode="json"))
    return JSONResponse(content=result)
