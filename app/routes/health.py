from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    request.app.state.model.require_ready()
    return {"status": "ready"}
