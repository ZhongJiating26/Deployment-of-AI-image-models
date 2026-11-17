import logging
import traceback

from fastapi.responses import JSONResponse

log = logging.getLogger("qwen")


class APIError(Exception):
    def __init__(self, status, code, message, headers=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


def respond(request, error):
    request.state.error_code = error.code
    error_type = {401: "authentication_error", 429: "rate_limit_error"}.get(
        error.status, "server_error" if error.status >= 500 else "invalid_request_error"
    )
    return JSONResponse(status_code=error.status, headers=error.headers, content={"error": {
        "message": error.message, "type": error_type, "code": error.code,
        "request_id": request.state.request_id,
    }})


def log_exception(event, exc, **fields):
    # Exception messages and source text can contain request data or credentials.
    frames = [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
              for frame in traceback.extract_tb(exc.__traceback__)]
    log.error(event, extra={"details": {
        **fields, "exception": type(exc).__name__, "stack": frames,
    }})
