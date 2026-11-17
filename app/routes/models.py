from fastapi import APIRouter, Request

from ..config import MINIMAX_ENTROPACK

router = APIRouter()


@router.get("/v1/models")
async def models(request: Request):
    settings = request.app.state.settings
    sizes = sorted(settings.sizes, key=lambda size: tuple(map(int, size.split("x"))))
    return {"object": "list", "data": [{
        "id": model, "object": "model", "created": 0,
        "owned_by": "DiffSynth-Studio" if model == MINIMAX_ENTROPACK else "Qwen",
        "supported_sizes": sizes, "default_size": settings.default_image_size,
    } for model in settings.models]}
