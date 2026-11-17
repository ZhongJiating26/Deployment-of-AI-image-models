import json
from contextlib import asynccontextmanager
from inspect import signature

from fastapi import APIRouter, Request
from pydantic import ValidationError
from python_multipart.exceptions import MultipartParseError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from ..config import MINIMAX_ENTROPACK
from ..exceptions import APIError
from ..image_utils import decode_image
from ..schemas import ImageRequest, ImageResponse

router = APIRouter(prefix="/v1/images")
EDIT_SCHEMA = ImageRequest.model_json_schema()
EDIT_SCHEMA["properties"]["image"] = {"type": "string", "format": "binary"}
EDIT_SCHEMA["required"].append("image")
SUPPORTS_PART_LIMIT = "max_part_size" in signature(Request.form).parameters


async def bounded_body(request, maximum):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > maximum:
            raise APIError(413, "file_too_large", "Request body exceeds the configured limit")
        body.extend(chunk)
    return bytes(body)


def validate_params(values, settings):
    if set(values) - ImageRequest.model_fields.keys():
        raise APIError(400, "invalid_request", "Unknown or unsupported request parameters")
    try:
        params = ImageRequest.model_validate(values)
    except ValidationError as exc:
        raise APIError(422, "validation_error", "Invalid or missing request fields") from exc
    params.model = settings.model_name if params.model is None else params.model
    fmt = settings.default_response_format if params.response_format is None else params.response_format
    if params.model not in settings.models:
        raise APIError(400, "unsupported_model", "Unsupported model")
    if params.size is None:
        params.size = settings.default_size_for(params.model)
    if params.size not in settings.sizes_for(params.model):
        raise APIError(400, "unsupported_size", "Unsupported image size")
    if params.n != settings.max_images_per_request:
        raise APIError(400, "invalid_request", "Only n=1 is supported")
    if fmt != settings.default_response_format:
        raise APIError(400, "invalid_request", "Only b64_json is supported")
    if not params.prompt.strip() or len(params.prompt) > settings.max_prompt_length:
        raise APIError(400, "invalid_request", "Prompt is blank or exceeds the length limit")
    return params


def content_type(request):
    return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()


@asynccontextmanager
async def parse_form(request, field_limit):
    try:
        options = {"max_part_size": field_limit} if SUPPORTS_PART_LIMIT else {}
        form = await request.form(max_files=8, max_fields=16, **options)
    except MultipartParseError as exc:
        raise APIError(400, "invalid_request", "Malformed multipart body") from exc
    try:
        yield form
    finally:
        await form.close()


@router.post("/generations", response_model=ImageResponse, openapi_extra={
    "requestBody": {"required": True, "content": {
        "application/json": {"schema": ImageRequest.model_json_schema()},
    }},
})
async def generations(request: Request):
    settings = request.app.state.settings
    if content_type(request) != "application/json":
        raise APIError(415, "unsupported_media_type", "Expected application/json")
    raw = await bounded_body(request, 65536 + settings.max_prompt_length * 12)
    try:
        values = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise APIError(422, "validation_error", "Invalid JSON body") from exc
    if not isinstance(values, dict):
        raise APIError(422, "validation_error", "Expected a JSON object")
    return await request.app.state.model.infer(request, validate_params(values, settings))


@router.post("/edits", response_model=ImageResponse, openapi_extra={
    "requestBody": {"required": True, "content": {
        "multipart/form-data": {"schema": EDIT_SCHEMA},
    }},
})
async def edits(request: Request):
    settings = request.app.state.settings
    if content_type(request) != "multipart/form-data":
        raise APIError(415, "unsupported_media_type", "Expected multipart/form-data")
    field_limit = max(65536, settings.max_prompt_length * 4)
    body = await bounded_body(request, settings.max_upload_mb * 1024**2 + field_limit + 65536)

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    multipart = Request(request.scope, receive)
    async with parse_form(multipart, field_limit) as form:
        values = {}
        upload = None
        for name, value in form.multi_items():
            if isinstance(value, str) and len(value.encode("utf-8")) > field_limit:
                raise APIError(400, "invalid_request", "Multipart field exceeds the length limit")
            if name in ("image", "image[]"):
                if upload is not None:
                    raise APIError(400, "invalid_request", "Only one reference image is supported")
                if not isinstance(value, UploadFile):
                    raise APIError(422, "validation_error", "Expected an image file")
                upload = value
            elif name in values or name not in ImageRequest.model_fields or isinstance(value, UploadFile):
                raise APIError(400, "invalid_request", "Duplicate or unsupported multipart field")
            else:
                if name in ("n", "seed"):
                    try:
                        value = int(value, 10)
                    except ValueError as exc:
                        raise APIError(422, "validation_error", "Expected integer n or seed") from exc
                values[name] = value
        if upload is None:
            raise APIError(422, "validation_error", "Missing image file")
        params = validate_params(values, settings)
        request.app.state.model.require_ready()
        image = await run_in_threadpool(decode_image, upload, settings, request.state.request_id)
        if params.model == MINIMAX_ENTROPACK and min(image.size) < 32:
            raise APIError(400, "invalid_request", "MiniMax reference image sides must be at least 32 pixels")
    return await request.app.state.model.infer(request, params, image)
