"""Run against the Windows service using the local model and OpenAI SDK."""
import argparse
import base64
import io
import mimetypes
from pathlib import Path
from urllib.request import urlopen

from openai import OpenAI
from PIL import Image

from app.config import Settings


def write_png(response, path, size, transparent=False):
    png = base64.b64decode(response.data[0].b64_json, validate=True)
    with Image.open(io.BytesIO(png)) as image:
        image.load()
        if image.format != "PNG" or image.size != size:
            raise RuntimeError("Unexpected image format or dimensions")
        if transparent and (image.mode != "RGBA" or image.getchannel("A").getextrema()[0] == 255):
            raise RuntimeError("Expected a PNG with transparent pixels")
    path.write_bytes(png)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--url", help="Service origin, without /v1")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--model", help="Model ID returned by /v1/models")
    parser.add_argument("--transparent", action="store_true", help="Request and verify an alpha channel")
    parser.add_argument("--results", type=Path, default=Path("smoke-results"))
    args = parser.parse_args()
    settings = Settings()
    model_id = args.model or settings.model_name
    origin = (args.url or f"http://127.0.0.1:{settings.port}").rstrip("/")
    size = tuple(map(int, args.size.split("x")))
    media_type, _ = mimetypes.guess_type(args.reference.name)
    for endpoint in ("/health", "/ready"):
        with urlopen(origin + endpoint, timeout=10) as response:
            print(endpoint, response.status)
    args.results.mkdir(parents=True, exist_ok=True)
    with OpenAI(base_url=origin + "/v1", api_key=settings.qwen_api_key.get_secret_value(),
                timeout=settings.queue_wait_timeout_seconds + settings.request_timeout_seconds + 30,
                max_retries=0) as client:
        if model_id not in {model.id for model in client.models.list()}:
            raise RuntimeError("Configured model missing from /v1/models")
        extra = {"seed": 42}
        if args.transparent:
            extra["background"] = "transparent"
        options = {"model": model_id, "size": args.size,
                   "n": 1, "response_format": "b64_json", "extra_body": extra}
        generated = client.images.generate(prompt="白底产品摄影，一把蓝色露营椅", **options)
        write_png(generated, args.results / "generated.png", size, args.transparent)
        with args.reference.open("rb") as reference:
            edited = client.images.edit(image=(args.reference.name, reference, media_type),
                                        prompt="将背景改为浅蓝色，保留主体", **options)
        write_png(edited, args.results / "edited.png", size, args.transparent)


if __name__ == "__main__":
    main()
