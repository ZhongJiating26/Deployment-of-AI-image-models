import uvicorn

from .config import Settings
from .main import create_app


def main():
    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port,
                workers=1, access_log=False)


if __name__ == "__main__":
    main()
