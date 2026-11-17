import io
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from app.config import MINIMAX_ENTROPACK, QWEN_STANDARD, Settings
from app.exceptions import APIError
from app.model_service import ModelService


class FakeOOM(Exception):
    pass


class PipelineAdapterTests(unittest.TestCase):
    def setUp(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.directory = directory
        (directory / "model_index.json").write_text(json.dumps({
            "_class_name": "QwenImage21Pipeline", "sample_sigmas": [1.0] * 8,
        }), encoding="utf-8")
        settings = Settings(_env_file=None, qwen_api_key="integration-secret",
                            model_path=directory, save_outputs=False)
        self.service = ModelService(settings)
        self.pipe = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (512, 512))]))
        self.loader = Mock(return_value=self.pipe)
        self.cuda = Mock()
        self.cuda.max_memory_allocated.return_value = 100
        self.torch = SimpleNamespace(cuda=self.cuda, bfloat16=object(), Generator=Mock(),
                                     inference_mode=nullcontext, OutOfMemoryError=FakeOOM)
        with patch.dict("sys.modules", {
            "torch": self.torch,
            "diffusers": SimpleNamespace(QwenImage21Pipeline=SimpleNamespace(from_pretrained=self.loader)),
        }):
            self.service.load()
        self.loader.assert_called_once_with(str(directory), local_files_only=True, dtype=self.torch.bfloat16)
        self.pipe.enable_model_cpu_offload.assert_called_once()
        self.pipe.to.assert_not_called()
        self.params = SimpleNamespace(prompt="test", model=settings.model_name, size="512x512")

    def test_single_pipeline_for_both_operations(self):
        import base64

        with Image.new("RGB", (16, 16)) as reference:
            for image in (None, reference):
                result = self.service.run(self.params, image, 42, "req_test")
                options = self.pipe.call_args.kwargs
                self.assertEqual((options["width"], options["height"]), (512, 512))
                self.assertTrue(options["use_kv_cache"])
                self.assertNotIn("num_inference_steps", options)
                self.assertNotIn("sigmas", options)
                self.assertIs(options.get("image"), image)
                with Image.open(io.BytesIO(base64.b64decode(result.data[0].b64_json))) as png:
                    self.assertEqual(png.size, (512, 512))
        self.assertEqual(self.cuda.reset_peak_memory_stats.call_count, 2)

    def test_qwen_reference_limit_does_not_change_output_dimensions(self):
        self.service.settings.qwen_edit_reference_resolution = 1024
        with Image.new("RGB", (32, 32)) as reference:
            self.service.render_image(self.params, reference, 42, 2048, 2048)
            options = self.pipe.call_args.kwargs
            self.assertEqual((options["width"], options["height"], options["output_resolution"]),
                             (2048, 2048, 1024))
            self.service.render_image(self.params, None, 42, 2048, 2048)
            self.assertEqual(self.pipe.call_args.kwargs["output_resolution"], 2048)

    def test_oom_recovery_and_unavailable_state(self):
        self.pipe.side_effect = FakeOOM("redacted")
        with self.assertRaises(APIError) as first:
            self.service.run(self.params, None, 42, "req_test")
        self.assertEqual(first.exception.code, "cuda_out_of_memory")
        self.cuda.empty_cache.assert_called_once()
        self.assertTrue(self.service.ready)
        self.cuda.synchronize.side_effect = RuntimeError("redacted")
        with self.assertRaises(APIError), self.assertLogs("qwen", level="ERROR"):
            self.service.run(self.params, None, 42, "req_test")
        self.assertFalse(self.service.ready)

    def test_switching_models_reuses_only_the_active_pipeline(self):
        standard = self.directory / "standard"
        standard.mkdir()
        (standard / "model_index.json").write_text(
            '{"_class_name":"QwenImage21Pipeline"}', encoding="utf-8",
        )
        settings = self.service.settings
        settings.qwen_standard_path = standard
        settings.qwen_edit_reference_resolution = 512
        minimax = Mock(return_value=Image.new("RGB", (512, 512)))
        fake_diffusers = SimpleNamespace(QwenImage21Pipeline=SimpleNamespace(from_pretrained=self.loader))
        with patch.dict(sys.modules, {"diffusers": fake_diffusers}), patch(
            "app.model_service.load_entropack", return_value=minimax,
        ) as load_mini, Image.new("RGB", (32, 32)) as reference:
            self.params.model = QWEN_STANDARD
            for _ in range(2):
                self.service.run(self.params, reference, 42, "req_test")
            self.assertEqual(self.loader.call_count, 2)
            self.assertEqual(self.pipe.call_args.kwargs["num_inference_steps"], 40)
            self.params.model = MINIMAX_ENTROPACK
            self.service.run(self.params, reference, 42, "req_test")
            minimax.assert_called_once_with(prompt="test", width=512, height=512,
                                            seed=42, edit_image=reference)
            load_mini.assert_called_once()
            self.params.model = settings.model_name
            self.service.run(self.params, None, 42, "req_test")
        self.assertEqual(self.loader.call_count, 3)
        self.assertEqual(self.service.active_model, settings.model_name)

    def test_failed_switch_leaves_service_available_for_another_request(self):
        self.params.model = MINIMAX_ENTROPACK
        with patch("app.model_service.load_entropack", side_effect=FileNotFoundError), \
                self.assertLogs("qwen", level="ERROR"), self.assertRaises(APIError) as error:
            self.service.run(self.params, None, 42, "req_test")
        self.assertEqual((error.exception.status, error.exception.code), (503, "model_load_failed"))
        self.assertTrue(self.service.ready)
        self.assertIsNone(self.service.active_model)
        self.params.model = self.service.settings.model_name
        fake_diffusers = SimpleNamespace(QwenImage21Pipeline=SimpleNamespace(from_pretrained=self.loader))
        with patch.dict(sys.modules, {"diffusers": fake_diffusers}):
            self.service.run(self.params, None, 42, "req_test")
        self.assertEqual(self.service.active_model, self.params.model)
