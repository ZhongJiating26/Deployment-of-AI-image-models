import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.config import Settings
from app.minimax import load_entropack


class EntroPackAdapterTests(unittest.TestCase):
    def test_profiles_use_local_component_paths(self):
        for profile, dit, text_encoder in (
            ("int8", "dit_8bit_int8", "text_encoder_int8_convrot"),
            ("7bit-int8", "dit_7bit_int8", "text_encoder_3bpp"),
            ("3bpp", "dit_extreme_3bpp", "text_encoder_3bpp"),
        ):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                settings = Settings(
                    _env_file=None, qwen_api_key="integration-secret", model_path=root,
                    minimax_entropack_path=root / "pack", minimax_text_encoder_path=root / "encoder",
                    minimax_image_assets_path=root / "assets", minimax_profile=profile,
                )
                model_paths = [
                    (root / "pack" / "models" / dit).with_suffix(".safetensors"),
                    (root / "encoder" / "models" / "MiniMax-H3" / text_encoder).with_suffix(".safetensors"),
                    root / "assets" / "pdd_fused_heads.safetensors",
                    root / "pack" / "models" / "vae_int8.safetensors",
                ]
                sidecars = [path.with_suffix(".json") for path in model_paths[:2]]
                vae_sidecar = model_paths[-1].with_suffix(".json")
                if profile == "7bit-int8":
                    sidecars.append(vae_sidecar)
                for path in [*model_paths, *sidecars]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                processor = root / "assets" / "processor"
                processor.mkdir()
                loader = Mock()
                module = SimpleNamespace(
                    MiniMaxH3ImagePipeline=SimpleNamespace(from_pretrained=loader),
                    ModelConfig=SimpleNamespace,
                )
                with patch.dict(sys.modules, {"diffsynth.pipelines.minimax_h3_image": module}):
                    load_entropack(settings, SimpleNamespace(bfloat16="bf16"))
                options = loader.call_args.kwargs
                self.assertEqual(options["vram_limit"], None if profile == "3bpp" else 20)
                models = options["model_configs"]
                self.assertEqual([item.path for item in models], [str(path) for path in model_paths])
                self.assertTrue(all(item.skip_download for item in models))
                self.assertEqual([item.config_file for item in models[:2]],
                                 [str(path.with_suffix(".json")) for path in model_paths[:2]])
                self.assertEqual(options["processor_config"].path, str(processor))
                self.assertTrue(options["processor_config"].skip_download)
                self.assertEqual(models[-1].config_file, str(vae_sidecar) if profile == "7bit-int8" else None)
