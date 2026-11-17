from fastapi import APIRouter, Request

from ..config import MINIMAX_ENTROPACK

router = APIRouter()


@router.get("/v1/models")
async def models(request: Request):
    settings = request.app.state.settings
    return {"object": "list", "data": [{
        "id": model, "object": "model", "created": 0,
        "owned_by": "DiffSynth-Studio" if model == MINIMAX_ENTROPACK else "Qwen",
        "supported_sizes": sorted(settings.sizes_for(model), key=lambda size: tuple(map(int, size.split("x")))),
        "default_size": settings.default_size_for(model),
    } for model in settings.models]}
