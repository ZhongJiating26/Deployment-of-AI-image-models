from .exceptions import log


FILES = {
    "int8": ("dit_8bit_int8", "text_encoder_int8_convrot"),
    "7bit-int8": ("dit_7bit_int8", "text_encoder_3bpp"),
    "3bpp": ("dit_extreme_3bpp", "text_encoder_3bpp"),
}


def load_entropack(settings, torch):
    dit_name, encoder_name = FILES[settings.minimax_profile]
    root = settings.minimax_entropack_path / "models"
    encoder = settings.minimax_text_encoder_path / "models" / "MiniMax-H3"
    assets = settings.minimax_image_assets_path
    vae_config = root / "vae_int8.json"
    components = (
        ("DiT", root / (dit_name + ".safetensors"), root / (dit_name + ".json")),
        ("text encoder", encoder / (encoder_name + ".safetensors"), encoder / (encoder_name + ".json")),
        ("PDD heads", assets / "pdd_fused_heads.safetensors", None),
        ("VAE", root / "vae_int8.safetensors",
         vae_config if settings.minimax_profile == "7bit-int8" or vae_config.is_file() else None),
    )
    missing = [str(path) for _, weights, config in components for path in (weights, config)
               if path is not None and not path.is_file()]
    processor = assets / "processor"
    if not processor.is_dir():
        missing.append(str(processor))
    if missing:
        log.error("minimax_assets_missing", extra={"details": {"paths": missing}})
        raise FileNotFoundError("Missing local MiniMax components")

    from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig

    dtype = torch.bfloat16
    # DiffSynth's consumer-GPU recipe keeps INT8 weights in host memory.
    offload = {
        "offload_dtype": dtype, "offload_device": "cpu",
        "onload_dtype": dtype, "onload_device": "cpu",
        "preparing_dtype": dtype, "preparing_device": "cuda",
        "computation_dtype": dtype, "computation_device": "cuda",
    } if settings.minimax_profile != "3bpp" else {}
    configs = [
        ModelConfig(path=str(weights), config_file=str(config) if config else None,
                    skip_download=True, **offload)
        for _, weights, config in components
    ]
    return MiniMaxH3ImagePipeline.from_pretrained(
        torch_dtype=dtype, device="cuda", model_configs=configs,
        processor_config=ModelConfig(path=str(processor), skip_download=True),
        vram_limit=20 if offload else None,
    )
