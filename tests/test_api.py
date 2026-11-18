import base64
import io
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from openai import OpenAI
from PIL import Image

from app.config import MINIMAX_ENTROPACK, QWEN_STANDARD, ROOT, Settings
from app.exceptions import APIError, log_exception
from app.main import create_app
from app.model_service import ModelService
from app.schemas import ImageData, ImageMeta, ImageResponse


def make_png(size=(32, 32)):
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, "PNG")
    return buffer.getvalue()


class FakeModel(ModelService):
    loads = 0

    def load(self):
        self.loads += 1
        self.pipe = object()
        self.unavailable_code = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.reference_sizes = []

    def run(self, params, image, seed, request_id):
        self.calls.append(params.prompt)
        if params.prompt == "block":
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError("Test failed to release worker")
        if params.prompt == "oom":
            try:
                raise RuntimeError("private path and integration-secret")
            except RuntimeError as exc:
                log_exception("cuda_out_of_memory", exc, request_id=request_id)
            raise APIError(500, "cuda_out_of_memory", "GPU ran out of memory")
        if params.prompt == "unexpected":
            raise RuntimeError("private path and integration-secret")
        if image is not None:
            references = image if isinstance(image, list) else [image]
            assert all(reference.mode == "RGB" for reference in references)
            self.reference_sizes = [reference.size for reference in references]
        width, height = map(int, params.size.split("x"))
        return ImageResponse(created=int(time.time()), data=[
            ImageData(b64_json=base64.b64encode(make_png((width, height))).decode())
        ], meta=ImageMeta(request_id=request_id, model=params.model, elapsed_seconds=0, seed=seed))


class APIContractTests(unittest.TestCase):
    def setUp(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        self.temp = Path(temp)
        self.settings = Settings(_env_file=None, qwen_api_key="integration-secret",
                                 model_path=self.temp, log_dir=self.temp,
                                 save_outputs=False, max_upload_mb=1, max_image_pixels=4096,
                                 max_queue_length=1, queue_wait_timeout_seconds=0.15,
                                 request_timeout_seconds=0.5)
        self.app = create_app(self.settings, FakeModel)
        self.client = self.enterContext(TestClient(self.app))
        self.auth = {"Authorization": "Bearer integration-secret"}

    def error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        error = response.json()["error"]
        self.assertEqual(error["code"], code)
        self.assertEqual(error["request_id"], response.headers["X-Request-ID"])
        self.assertNotIn("integration-secret", response.text)
        return error

    def generate(self, **params):
        return self.client.post("/v1/images/generations", headers=self.auth,
                                json={"prompt": "test", **params})

    def edit(self, data, image=None, media="image/png"):
        return self.client.post("/v1/images/edits", headers=self.auth, data=data,
                                files={"image": ("reference.png", make_png() if image is None else image, media)})

    def test_template_configuration(self):
        with patch.dict("os.environ", {"QWEN_API_KEY": "integration-secret"}, clear=True):
            settings = Settings(_env_file=ROOT / ".env.example")
        self.assertTrue(settings.model_local_files_only)
        self.assertEqual(settings.max_concurrent_inferences, 1)
        self.assertFalse(settings.enable_url_response)

    def test_multiple_references_keep_multipart_order(self):
        self.settings.minimax_entropack_path = self.temp
        self.settings.save_uploads = True
        self.settings.upload_dir = self.temp
        files = [
            ("image[]", ("a.png", make_png((32, 48)), "image/png")),
            ("image[]", ("b.png", make_png((48, 32)), "image/png")),
        ]
        response = self.client.post("/v1/images/edits", headers=self.auth,
                                    data={"prompt": "Combine the scenes", "model": MINIMAX_ENTROPACK},
                                    files=files)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.app.state.model.reference_sizes, [(32, 48), (48, 32)])
        request_id = response.headers["X-Request-ID"]
        self.assertEqual({path.name for path in self.temp.glob("*/*.png")},
                         {f"{request_id}_1.png", f"{request_id}_2.png"})
        too_many_pixels = self.client.post("/v1/images/edits", headers=self.auth,
                                           data={"prompt": "test"}, files=files + files[:1])
        self.error(too_many_pixels, 400, "invalid_request")

    def test_reference_count_limits_by_model(self):
        self.settings.minimax_entropack_path = self.temp
        self.settings.max_image_pixels = 16 * 32 * 32
        file = ("image[]", ("reference.png", make_png(), "image/png"))
        for model, maximum in [(self.settings.model_name, 10), (MINIMAX_ENTROPACK, 16)]:
            data = {"prompt": "test", "model": model}
            response = self.client.post("/v1/images/edits", headers=self.auth,
                                        data=data, files=[file] * maximum)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(self.app.state.model.reference_sizes), maximum)
            response = self.client.post("/v1/images/edits", headers=self.auth,
                                        data=data, files=[file] * (maximum + 1))
            self.error(response, 400, "invalid_request")

    def test_resize_large_reference_without_resizing_output(self):
        self.settings.max_reference_edge = 64
        response = self.edit({"prompt": "test", "size": "1024x1024"}, make_png((256, 128)))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.app.state.model.reference_sizes, [(64, 32)])
        with Image.open(io.BytesIO(base64.b64decode(response.json()["data"][0]["b64_json"]))) as png:
            self.assertEqual(png.size, (1024, 1024))

    def test_custom_sizes_and_2048_gate(self):
        settings = Settings(_env_file=None, qwen_api_key="integration-secret",
                            model_path=self.temp, allowed_image_sizes="1024x1024,2048x1152,1280x800")
        self.assertEqual(settings.sizes, {"1024x1024", "1280x800"})
        settings.enable_2048 = True
        self.assertEqual(settings.sizes, {"1024x1024", "1280x800", "2048x1152", "2048x2048"})

    def test_health_models_and_auth(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/ready").status_code, 200)
        self.assertEqual(self.app.state.model.loads, 1)
        self.error(self.client.get("/v1/models"), 401, "invalid_api_key")
        self.error(self.client.post("/v1/images/generations", headers={"Authorization": "Bearer wrong"}),
                   401, "invalid_api_key")
        self.assertEqual(self.client.get("/v1/models", headers=self.auth).json()["data"][0]["id"],
                         self.settings.model_name)
        self.error(self.client.get("/docs"), 404, "not_found")

    def test_openai_sdk(self):
        sdk = OpenAI(base_url=str(self.client.base_url) + "/v1", api_key="integration-secret",
                     http_client=self.client, max_retries=0)
        for response in (
            sdk.images.generate(model=self.settings.model_name, prompt="test", size="512x512",
                                extra_body={"seed": 42}),
            sdk.images.edit(model=self.settings.model_name, prompt="edit", size="512x512",
                            image=[("reference.png", make_png(), "image/png")],
                            extra_body={"seed": 42}),
            sdk.images.edit(model=self.settings.model_name, prompt="edit", size="512x512",
                            image=("reference.png", make_png(), "image/png"),
                            extra_body={"seed": 42}),
        ):
            self.assertEqual(response.meta["seed"], 42)
            with Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json))) as image:
                self.assertEqual((image.format, image.size), ("PNG", (512, 512)))

    def test_parameter_validation(self):
        for params, status, code in [
            ({"model": "wrong"}, 400, "unsupported_model"),
            ({"size": "2048x2048"}, 400, "unsupported_size"),
            ({"n": 2}, 400, "invalid_request"),
            ({"response_format": "url"}, 400, "invalid_request"),
            ({"quality": "high"}, 400, "invalid_request"),
            ({"n": "1"}, 422, "validation_error"),
            ({"prompt": ""}, 400, "invalid_request"),
        ]:
            with self.subTest(params=params):
                self.error(self.generate(**params), status, code)

    def test_enabled_model_ids_work_for_both_routes(self):
        self.settings.qwen_standard_path = self.temp
        self.settings.minimax_entropack_path = self.temp
        listed = self.client.get("/v1/models", headers=self.auth).json()["data"]
        self.assertEqual({model["id"] for model in listed},
                         {self.settings.model_name, QWEN_STANDARD, MINIMAX_ENTROPACK})
        for entry in listed:
            sizes = {
                "512x512", "768x768", "1024x1024", "1024x768", "768x1024",
                "1152x768", "768x1152", "1024x576", "576x1024",
            }
            if entry["id"] == MINIMAX_ENTROPACK:
                sizes |= {"2048x2048", "4096x4096", "6272x2688", "2688x6272"}
            self.assertEqual(set(entry["supported_sizes"]), sizes)
            self.assertEqual(entry["default_size"], "1024x1024")
        self.settings.enable_2048 = True
        listed = self.client.get("/v1/models", headers=self.auth).json()["data"]
        self.assertTrue(all("2048x2048" in entry["supported_sizes"] for entry in listed))
        for model in (self.settings.model_name, QWEN_STANDARD, MINIMAX_ENTROPACK):
            for response, expected_size in (
                (self.generate(model=model, size="1024x576"), (1024, 576)),
                (self.edit({"prompt": "edit", "model": model, "size": "768x1152"}), (768, 1152)),
            ):
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["meta"]["model"], model)
                with Image.open(io.BytesIO(base64.b64decode(response.json()["data"][0]["b64_json"]))) as png:
                    self.assertEqual(png.size, expected_size)

    def test_per_model_size_options_and_defaults(self):
        self.settings.qwen_standard_path = self.temp
        self.settings.minimax_entropack_path = self.temp
        self.settings.qwen_standard_allowed_image_sizes = "768x1024"
        self.settings.qwen_standard_default_image_size = "768x1024"
        self.settings.minimax_allowed_image_sizes = "1024x576,4096x4096"
        self.settings.minimax_default_image_size = "1024x576"
        listed = {entry["id"]: entry for entry in self.client.get(
            "/v1/models", headers=self.auth,
        ).json()["data"]}
        self.assertEqual(listed[QWEN_STANDARD]["supported_sizes"], ["768x1024"])
        self.assertEqual(listed[MINIMAX_ENTROPACK]["supported_sizes"], ["1024x576", "4096x4096"])
        self.error(self.generate(model=QWEN_STANDARD, size="4096x4096"), 400, "unsupported_size")
        mini = self.generate(model=MINIMAX_ENTROPACK, size="4096x4096")
        self.assertEqual(mini.status_code, 200, mini.text[:300])
        for response, size in (
            (self.generate(model=QWEN_STANDARD), (768, 1024)),
            (self.edit({"prompt": "edit", "model": MINIMAX_ENTROPACK}), (1024, 576)),
        ):
            self.assertEqual(response.status_code, 200)
            with Image.open(io.BytesIO(base64.b64decode(response.json()["data"][0]["b64_json"]))) as png:
                self.assertEqual(png.size, size)

    def test_transparent_option_requires_minimax_and_configured_model(self):
        self.settings.minimax_entropack_path = self.temp
        options = {"model": MINIMAX_ENTROPACK, "background": "transparent"}
        self.error(self.generate(background="transparent"), 400, "invalid_request")
        self.error(self.generate(**options), 400, "invalid_request")
        self.settings.minimax_background_model_path = self.temp / "u2net.onnx"
        self.assertEqual(self.generate(**options).status_code, 200)
        supplied = self.edit({"prompt": "edit", **options})
        omitted = self.edit({"prompt": "edit", "model": MINIMAX_ENTROPACK})
        entries = [json.loads(line) for line in (self.temp / "service.log").read_text(encoding="utf-8").splitlines()]
        for response, background in ((supplied, "transparent"), (omitted, "missing")):
            self.assertEqual(response.status_code, 200)
            request_log = next(entry for entry in entries if entry["event"] == "http_request"
                               and entry["request_id"] == response.headers["X-Request-ID"])
            self.assertEqual(request_log["background_received"], background)
            self.assertEqual(request_log["reference_count"], 1)

    def test_upload_validation(self):
        self.error(self.edit({"prompt": "edit", "mask": "unsupported"}), 400, "invalid_request")
        self.error(self.edit({"prompt": "edit"}, b"invalid"), 400, "invalid_request")
        self.error(self.edit({"prompt": "edit"}, make_png((65, 65))), 400, "invalid_request")
        large_file = make_png() + b"\x00" * (1024**2 + 1)
        self.assertEqual(self.edit({"prompt": "edit"}, large_file).status_code, 200)
        self.error(self.edit({"prompt": "edit"}, media="image/gif"), 415, "unsupported_media_type")

    def test_errors_are_sanitized(self):
        for prompt, code in [("oom", "cuda_out_of_memory"), ("unexpected", "internal_error")]:
            self.error(self.generate(prompt=prompt), 500, code)
        entries = [json.loads(line) for line in (self.temp / "service.log").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(row.get("stack") for row in entries))
        self.assertNotIn("private path", str(entries))
        self.assertNotIn("integration-secret", str(entries))

    def test_queue_full_and_wait_timeout(self):
        model = self.app.state.model
        with ThreadPoolExecutor(max_workers=2) as pool:
            active = pool.submit(self.generate, prompt="block")
            try:
                self.assertTrue(model.started.wait(2))
                waiting = pool.submit(self.generate, prompt="waiting")
                deadline = time.monotonic() + 0.1
                while model.admitted < 2 and time.monotonic() < deadline:
                    time.sleep(0.001)
                self.error(self.generate(), 429, "server_busy")
                self.error(waiting.result(2), 429, "server_busy")
                self.assertEqual(model.calls, ["block"])
            finally:
                model.release.set()
            self.assertEqual(active.result(2).status_code, 200)

    def test_response_timeout_holds_inference_slot(self):
        model = self.app.state.model
        try:
            self.error(self.generate(prompt="block"), 504, "request_timeout")
            self.error(self.generate(prompt="waiting"), 429, "server_busy")
            self.assertEqual(model.calls, ["block"])
        finally:
            model.release.set()

    def test_missing_local_model(self):
        with TestClient(create_app(self.settings)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.error(client.get("/ready"), 503, "model_load_failed")
