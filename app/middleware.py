import logging
from secrets import compare_digest
from time import perf_counter
from uuid import uuid4

from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import Request

from .exceptions import APIError, log_exception, respond

log = logging.getLogger("qwen")


class CorsMiddleware(CORSMiddleware):
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "OPTIONS":
            headers = Headers(scope=scope)
            if "origin" in headers and "access-control-request-method" in headers:
                request = Request(scope)
                request.state.request_id = "req_" + uuid4().hex
                response = self.preflight_response(headers)
                if response.status_code >= 400:
                    cors_headers = {key: value for key, value in response.headers.items()
                                    if key not in {"content-type", "content-length"}}
                    response = respond(request, APIError(
                        400, "invalid_request", "CORS preflight rejected", cors_headers,
                    ))
                response.headers["X-Request-ID"] = request.state.request_id
                await response(scope, receive, send)
                log.info("http_request", extra={"details": {
                    "request_id": request.state.request_id, "method": "OPTIONS",
                    "path": "<preflight>", "status": response.status_code,
                    "error_code": getattr(request.state, "error_code", None),
                }})
                return
        await super().__call__(scope, receive, send)


class RequestMiddleware:
    def __init__(self, app, settings):
        self.app = app
        self.key = settings.qwen_api_key.get_secret_value().encode("ascii")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        request.state.request_id = "req_" + uuid4().hex
        request.state.metrics = {}
        started = perf_counter()
        status = 500
        sent = False

        async def send_response(message):
            nonlocal status, sent
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-Request-ID"] = request.state.request_id
                status = message["status"]
                sent = True
            await send(message)

        try:
            if scope["path"] == "/v1" or scope["path"].startswith("/v1/"):
                headers = Headers(scope=scope).getlist("authorization")
                scheme, _, token = headers[0].partition(" ") if len(headers) == 1 else ("", "", "")
                if scheme.lower() != "bearer" or not compare_digest(token.encode("utf-8"), self.key):
                    raise APIError(401, "invalid_api_key", "Invalid or missing API key",
                                   {"WWW-Authenticate": "Bearer"})
            await self.app(scope, receive, send_response)
        except APIError as exc:
            await respond(request, exc)(scope, receive, send_response)
        except Exception as exc:
            log_exception("unhandled_error", exc, request_id=request.state.request_id)
            if sent:
                raise
            error = APIError(500, "internal_error", "Internal server error")
            await respond(request, error)(scope, receive, send_response)
        finally:
            route = scope.get("route")
            log.info("http_request", extra={"details": {
                "request_id": request.state.request_id, "path": getattr(route, "path", "<unmatched>"),
                "method": scope["method"], "status": status,
                "elapsed_seconds": round(perf_counter() - started, 3),
                "error_code": getattr(request.state, "error_code", None), **request.state.metrics,
            }})
