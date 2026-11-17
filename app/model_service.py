import asyncio
import base64
import gc
import io
import json
import logging
import secrets
from math import isqrt
from time import perf_counter, time

from starlette.concurrency import run_in_threadpool

from .config import MINIMAX_ENTROPACK, QWEN_STANDARD
from .exceptions import APIError, log_exception
from .image_utils import save_image
from .minimax import load_entropack
from .schemas import ImageData, ImageMeta, ImageResponse

log = logging.getLogger("qwen")


class ModelService:
    def __init__(self, settings):
        self.settings = settings
        self.pipe = None
        self.active_model = None
        self.torch = None
        self.unavailable_code = "model_not_ready"
        self.lock = asyncio.Lock()
        self.admitted = 0
        self.jobs = set()

    @property
    def ready(self):
        return self.unavailable_code is None

    def load(self):
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            self.torch = torch
            self.select_model(self.settings.model_name)
            self.unavailable_code = None
        except Exception as exc:
            self.unavailable_code = "model_load_failed"
            log_exception("model_load_failed", exc, model=self.settings.model_name)

    def release_pipeline(self):
        self.pipe = None
        self.active_model = None
        gc.collect()
        self.torch.cuda.empty_cache()

    def select_model(self, model):
        if self.active_model == model:
            return
        started = perf_counter()
        if self.pipe is not None:
            self.torch.cuda.synchronize()
            self.release_pipeline()
        if model == MINIMAX_ENTROPACK:
            pipeline = load_entropack(self.settings, self.torch)
        else:
            pipeline = self.load_qwen(model)
        self.pipe = pipeline
        self.active_model = model
        log.info("model_loaded", extra={"details": {
            "model": model, "load_seconds": round(perf_counter() - started, 3),
        }})

    def load_qwen(self, model):
        model_path = self.settings.models[model]
        with (model_path / "model_index.json").open(encoding="utf-8-sig") as file:
            checkpoint = json.load(file)
        sigmas = checkpoint.get("sample_sigmas")
        if checkpoint.get("_class_name") != "QwenImage21Pipeline":
            raise ValueError("Expected a QwenImage21Pipeline checkpoint")
        if model == QWEN_STANDARD and sigmas:
            raise ValueError("Expected the Qwen standard checkpoint")
        if model != QWEN_STANDARD and len(sigmas or []) != 8:
            raise ValueError("Expected the Qwen Turbo eight-step schedule")

        from diffusers import QwenImage21Pipeline

        pipe = QwenImage21Pipeline.from_pretrained(
            str(model_path), local_files_only=self.settings.model_local_files_only,
            dtype=getattr(self.torch, self.settings.model_dtype),
        )
        if self.settings.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        return pipe

    def render_image(self, params, image, seed, width, height):
        if params.model == MINIMAX_ENTROPACK:
            return self.pipe(prompt=params.prompt, width=width, height=height,
                             seed=seed, edit_image=image)
        kwargs = {"image": image} if image is not None else {}
        if params.model == QWEN_STANDARD:
            kwargs["num_inference_steps"] = 40
        reference_limit = self.settings.qwen_edit_reference_resolution
        resolution = isqrt(width * height)
        if image is not None and reference_limit:
            resolution = min(resolution, reference_limit)
        return self.pipe(
            prompt=params.prompt, width=width, height=height, output_resolution=resolution,
            generator=self.torch.Generator(device="cpu").manual_seed(seed),
            use_kv_cache=self.settings.use_kv_cache, **kwargs,
        ).images[0]

    def require_ready(self):
        if not self.ready:
            message = ("Model failed to load" if self.unavailable_code == "model_load_failed"
                       else "Model is not ready")
            raise APIError(503, self.unavailable_code, message)

    async def infer(self, request, params, image=None):
        self.require_ready()
        started = perf_counter()
        request_id = request.state.request_id
        seed = params.seed if params.seed is not None else secrets.randbits(64)
        metrics = {"model": params.model, "size": params.size, "seed": seed, "queue_wait_seconds": 0.0}
        request.state.metrics = metrics
        if self.settings.log_prompts:
            metrics["prompt"] = params.prompt

        if self.admitted >= self.settings.max_concurrent_inferences + self.settings.max_queue_length:
            raise APIError(429, "server_busy", "Inference queue is full")
        self.admitted += 1
        acquired = False
        job = None
        try:
            # Take the uncontended lock without yielding admission to another request.
            if not self.lock.locked() and self.admitted == 1:
                await self.lock.acquire()
            else:
                try:
                    async with asyncio.timeout(self.settings.queue_wait_timeout_seconds):
                        await self.lock.acquire()
                except TimeoutError as exc:
                    raise APIError(429, "server_busy", "Inference queue wait expired") from exc
            acquired = True
            self.require_ready()
            job = asyncio.create_task(run_in_threadpool(self.run, params, image, seed, request_id))
            self.jobs.add(job)
            job.add_done_callback(self.finish_job)
            acquired = False  # The running job now owns the slot.
        finally:
            metrics["queue_wait_seconds"] = round(perf_counter() - started, 3)
            if job is None:
                if acquired:
                    self.lock.release()
                self.admitted -= 1

        try:
            async with asyncio.timeout(self.settings.request_timeout_seconds):
                return await asyncio.shield(job)
        except TimeoutError as exc:
            raise APIError(504, "request_timeout", "Inference response deadline exceeded") from exc

    def finish_job(self, job):
        self.jobs.discard(job)
        self.lock.release()
        self.admitted -= 1
        if not job.cancelled():
            job.exception()

    def run(self, params, image, seed, request_id):
        cuda = self.torch.cuda
        started = perf_counter()
        peak = None
        code = None
        stage = "loading"
        try:
            cuda.reset_peak_memory_stats()
            self.select_model(params.model)
            stage = "inference"
            width, height = map(int, params.size.split("x"))
            with self.torch.inference_mode():
                result = self.render_image(params, image, seed, width, height)
            peak = cuda.max_memory_allocated()
            if result.size != (width, height):
                raise RuntimeError("Pipeline returned unexpected image dimensions")
            stage = "encoding"
            with io.BytesIO() as buffer:
                result.save(buffer, format="PNG")
                png = buffer.getvalue()
            if self.settings.save_outputs:
                save_image(self.settings.output_dir, request_id, "png", png)
            return ImageResponse(
                created=int(time()), data=[ImageData(b64_json=base64.b64encode(png).decode("ascii"))],
                meta=ImageMeta(request_id=request_id, model=params.model,
                               elapsed_seconds=round(perf_counter() - started, 3), seed=seed),
            )
        except Exception as exc:
            if isinstance(exc, self.torch.OutOfMemoryError):
                code = "cuda_out_of_memory"
                message = "GPU ran out of memory"
            elif stage == "loading":
                code = "model_load_failed"
                message = "Requested model failed to load"
            else:
                code = "image_edit_failed" if image is not None else "image_generation_failed"
                message = "Image editing failed" if image is not None else "Image generation failed"
            log_exception(code, exc, request_id=request_id, stage=stage)

        finally:
            try:
                if peak is None:
                    peak = cuda.max_memory_allocated()
            except Exception as exc:
                self.unavailable_code = "model_not_ready"
                log_exception("cuda_metrics_failed", exc, request_id=request_id)
            log.info("inference_finished", extra={"details": {
                "request_id": request_id, "model": params.model, "size": params.size,
                "seed": seed, "inference_seconds": round(perf_counter() - started, 3),
                "peak_allocated_bytes": peak, "error_code": code,
            }})

        if code == "cuda_out_of_memory":
            try:
                gc.collect()
                if self.active_model == MINIMAX_ENTROPACK:
                    self.release_pipeline()
                elif self.pipe is not None and self.settings.cpu_offload:
                    self.pipe.maybe_free_model_hooks()
                cuda.empty_cache()
                cuda.synchronize()
            except Exception as exc:
                self.unavailable_code = "model_not_ready"
                log_exception("cuda_recovery_failed", exc, request_id=request_id)
        elif stage in {"loading", "inference"}:
            try:
                self.torch.cuda.synchronize()
                self.release_pipeline()
            except Exception as exc:
                self.unavailable_code = "model_not_ready"
                log_exception("model_cleanup_failed", exc, request_id=request_id)
        raise APIError(503 if code == "model_load_failed" else 500, code, message)
