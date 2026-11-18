import os
import re
from shutil import copyfileobj
from datetime import datetime, timezone
from pathlib import Path
from time import time

from PIL import Image, ImageOps, UnidentifiedImageError

from .exceptions import APIError

MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
IMAGE_NAME = re.compile(r"req_[0-9a-f]{32}(?:_(?:[1-9]|1[0-6]))?\.(png|jpeg|webp)")


def decode_image(upload, settings, request_id, pixel_limit):
    media_type = (upload.content_type or "").lower()
    if media_type not in {MEDIA_TYPES[fmt] for fmt in settings.upload_formats}:
        raise APIError(415, "unsupported_media_type", "Unsupported image media type")
    try:
        with Image.open(upload.file) as source:
            fmt = source.format
            if fmt not in settings.upload_formats or MEDIA_TYPES[fmt] != media_type:
                raise APIError(415, "unsupported_media_type", "Uploaded image format does not match content type")
            width, height = capped_image_size(source.size, settings.max_reference_edge)
            if width * height > pixel_limit:
                raise APIError(400, "invalid_request", "Image exceeds the pixel limit")
            if getattr(source, "n_frames", 1) != 1:
                raise APIError(400, "invalid_request", "Animated images are unsupported")
            with ImageOps.exif_transpose(source) as oriented:
                result = oriented.convert("RGB")
                size = capped_image_size(result.size, settings.max_reference_edge)
                if result.size != size:
                    result = result.resize(size, Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise APIError(400, "invalid_request", "Image could not be decoded or exceeds the pixel limit") from exc

    if settings.save_uploads:
        upload.file.seek(0)
        save_image(settings.upload_dir, request_id, fmt.lower(), upload.file)
    return result


def decode_references(uploads, settings, request_id, minimum_side):
    remaining_pixels = settings.max_image_pixels
    images = []
    for index, upload in enumerate(uploads, start=1):
        upload_id = request_id if len(uploads) == 1 else f"{request_id}_{index}"
        image = decode_image(upload, settings, upload_id, remaining_pixels)
        if min(image.size) < minimum_side:
            raise APIError(400, "invalid_request", "MiniMax reference image sides must be at least 32 pixels")
        remaining_pixels -= image.width * image.height
        images.append(image)
    return images[0] if len(images) == 1 else images


def capped_image_size(size, max_edge):
    longest = max(size)
    if longest <= max_edge:
        return size
    return tuple(max(1, round(side * max_edge / longest)) for side in size)


def save_image(root, request_id, extension, data):
    folder = root / datetime.now(timezone.utc).date().isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{request_id}.{extension}"
    temporary = target.with_suffix(".tmp")
    try:
        if isinstance(data, bytes):
            temporary.write_bytes(data)
        else:
            with temporary.open("wb") as output:
                copyfileobj(data, output)
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
