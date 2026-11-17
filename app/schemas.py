from pydantic import BaseModel, ConfigDict, Field


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str
    model: str | None = None
    size: str | None = None
    n: int = 1
    seed: int | None = Field(None, ge=0, le=2**64 - 1)
    response_format: str | None = None


class ImageData(BaseModel):
    b64_json: str


class ImageMeta(BaseModel):
    request_id: str
    model: str
    elapsed_seconds: float
    seed: int


class ImageResponse(BaseModel):
    created: int
    data: list[ImageData]
    meta: ImageMeta
