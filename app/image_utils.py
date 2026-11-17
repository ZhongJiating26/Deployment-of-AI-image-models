import io
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import time

from PIL import Image, ImageOps, UnidentifiedImageError

from .exceptions import APIError

MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
IMAGE_NAME = re.compile(r"req_[0-9a-f]{32}\.(png|jpeg|webp)")


def decode_image(upload, settings, request_id):
    media_type = (upload.content_type or "").lower()
    if media_type not in {MEDIA_TYPES[fmt] for fmt in settings.upload_formats}:
        raise APIError(415, "unsupported_media_type", "Unsupported image media type")
    raw = upload.file.read(settings.max_upload_mb * 1024**2 + 1)
    if len(raw) > settings.max_upload_mb * 1024**2:
        raise APIError(413, "file_too_large", "Image upload exceeds the configured limit")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            fmt = source.format
            if fmt not in settings.upload_formats or MEDIA_TYPES[fmt] != media_type:
                raise APIError(415, "unsupported_media_type", "Uploaded image format does not match content type")
            if source.width * source.height > settings.max_image_pixels:
                raise APIError(400, "invalid_request", "Image exceeds the pixel limit")
            if getattr(source, "n_frames", 1) != 1:
                raise APIError(400, "invalid_request", "Animated images are unsupported")
            with ImageOps.exif_transpose(source) as oriented:
                result = oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise APIError(400, "invalid_request", "Image could not be decoded or exceeds the pixel limit") from exc

    if settings.save_uploads:
        save_image(settings.upload_dir, request_id, fmt.lower(), raw)
    return result


def save_image(root, request_id, extension, data):
    folder = root / datetime.now(timezone.utc).date().isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{request_id}.{extension}"
    temporary = target.with_suffix(".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def prune_images(root: Path, days: int):
    cutoff = time() - days * 86400
    for folder in root.glob("????-??-??"):
        if folder.is_symlink() or not folder.is_dir() or folder.resolve().parent != root:
            continue
        for item in folder.iterdir():
            if IMAGE_NAME.fullmatch(item.name) and not item.is_symlink() and item.is_file():
                if item.stat().st_mtime < cutoff:
                    item.unlink(missing_ok=True)
