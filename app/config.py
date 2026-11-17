import re
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
QWEN_STANDARD = "Qwen/Qwen-Image-2.1"
MINIMAX_ENTROPACK = "DiffSynth-Studio/MiniMax-H3-Image-EntroPack"
DEFAULT_SIZES = "512x512,768x768,1024x1024,1024x768,768x1024,1152x768,768x1152,1024x576,576x1024"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8-sig",
                                      extra="forbid", hide_input_in_errors=True)

    qwen_api_key: SecretStr
    model_name: str = Field(default="Qwen/Qwen-Image-2.1-Turbo", min_length=1)
    model_path: Path
    qwen_standard_path: Path | None = None
    minimax_entropack_path: Path | None = None
    minimax_text_encoder_path: Path | None = None
    minimax_image_assets_path: Path | None = None
    minimax_profile: Literal["int8", "7bit-int8", "3bpp"] = "int8"
    qwen_edit_reference_resolution: Literal[512, 768, 1024, 2048] | None = None
    model_local_files_only: Literal[True] = True
    model_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    cpu_offload: bool = True
    use_kv_cache: bool = True
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    enable_docs: bool = False
    cors_allowed_origins: str = ""
    default_image_size: str = "1024x1024"
    allowed_image_sizes: str = DEFAULT_SIZES
    enable_2048: bool = False
    max_images_per_request: Literal[1] = 1
    max_concurrent_inferences: Literal[1] = 1
    max_queue_length: int = Field(3, ge=0)
    queue_wait_timeout_seconds: float = Field(120, gt=0, allow_inf_nan=False)
    request_timeout_seconds: float = Field(600, gt=0, allow_inf_nan=False)
    max_prompt_length: int = Field(4000, gt=0)
    max_upload_mb: int = Field(20, gt=0)
    max_image_pixels: int = Field(25000000, gt=0)
    allowed_upload_formats: str = "PNG,JPEG,WEBP"
    save_uploads: bool = False
    upload_dir: Path = ROOT / "uploads"
    default_response_format: Literal["b64_json"] = "b64_json"
    save_outputs: bool = True
    output_dir: Path = ROOT / "outputs"
    output_retention_days: int = Field(7, ge=1)
    enable_url_response: Literal[False] = False
    log_dir: Path = ROOT / "logs"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_retention_days: int = Field(30, ge=1)
    log_prompts: bool = False
    log_json: bool = True

    @field_validator("model_local_files_only", "enable_url_response", mode="before")
    @classmethod
    def parse_bool(cls, value):
        if isinstance(value, str):
            return {"true": True, "false": False}.get(value.lower(), value)
        return value

    @field_validator("max_images_per_request", "max_concurrent_inferences", mode="before")
    @classmethod
    def parse_one(cls, value):
        return 1 if value == "1" else value

    @field_validator("qwen_api_key")
    @classmethod
    def validate_key(cls, value):
        key = value.get_secret_value()
        if (not key or not key.isascii() or not key.isprintable()
                or any(char.isspace() for char in key)
                or key.lower().startswith(("replace", "changeme", "<", "your_"))):
            raise ValueError("Set a non-placeholder Bearer API key")
        return value

    @field_validator("qwen_standard_path", "minimax_entropack_path", "minimax_text_encoder_path",
                     "minimax_image_assets_path", "qwen_edit_reference_resolution", mode="before")
    @classmethod
    def optional_settings(cls, value):
        return None if value == "" else value

    @field_validator("qwen_edit_reference_resolution", mode="before")
    @classmethod
    def parse_resolution(cls, value):
        return int(value) if isinstance(value, str) and value.isdecimal() else value

    @field_validator("model_path", "log_dir", "upload_dir", "output_dir", "qwen_standard_path",
                     "minimax_entropack_path", "minimax_text_encoder_path", "minimax_image_assets_path")
    @classmethod
    def resolve_path(cls, value):
        return (ROOT / value).resolve() if value is not None else None

    @property
    def models(self):
        models = {self.model_name: self.model_path}
        if self.qwen_standard_path:
            models[QWEN_STANDARD] = self.qwen_standard_path
        if self.minimax_entropack_path:
            models[MINIMAX_ENTROPACK] = self.minimax_entropack_path
        return models

    @property
    def sizes(self):
        sizes = self.configured_sizes
        if self.enable_2048:
            return sizes | {"2048x2048"}
        return {size for size in sizes if "2048" not in size.split("x")}

    @property
    def configured_sizes(self):
        return {item.strip() for item in self.allowed_image_sizes.split(",")}

    @property
    def upload_formats(self):
        return {item.strip().upper() for item in self.allowed_upload_formats.split(",")}

    @model_validator(mode="after")
    def validate_limits(self):
        if self.model_name in {QWEN_STANDARD, MINIMAX_ENTROPACK}:
            raise ValueError("MODEL_NAME is reserved for the default Turbo model")
        companions = (self.minimax_text_encoder_path, self.minimax_image_assets_path)
        if any(companions) or self.minimax_entropack_path:
            if not self.minimax_entropack_path or not all(companions):
                raise ValueError("Set all three MiniMax local directories together")
        for size in self.configured_sizes:
            if not re.fullmatch(r"[1-9][0-9]{2,3}x[1-9][0-9]{2,3}", size):
                raise ValueError("Use WIDTHxHEIGHT for each allowed image size")
            if any(side < 512 or side > 2048 or side % 32 for side in map(int, size.split("x"))):
                raise ValueError("Image sides must be multiples of 32 between 512 and 2048")
        if self.default_image_size not in self.sizes:
            raise ValueError("Default image size must be enabled")
        if not self.upload_formats <= {"PNG", "JPEG", "WEBP"}:
            raise ValueError("Invalid upload format configuration")
        return self
