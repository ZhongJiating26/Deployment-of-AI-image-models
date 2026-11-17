import json
import logging
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler


class Formatter(logging.Formatter):
    def __init__(self, settings):
        super().__init__()
        self.json_output = settings.log_json
        self.secret = settings.qwen_api_key.get_secret_value()

    def redact(self, value):
        if isinstance(value, str):
            return value.replace(self.secret, "[REDACTED]")
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, dict):
            return {key: self.redact(item) for key, item in value.items()}
        return value

    def format(self, record):
        fields = self.redact({
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname, "event": record.getMessage(),
            **getattr(record, "details", {}),
        })
        if self.json_output:
            return json.dumps(fields, ensure_ascii=False)
        return " ".join(f"{key}={value!r}" for key, value in fields.items())


def configure_logging(settings):
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("qwen")
    log.setLevel(settings.log_level)
    log.propagate = False
    handler = TimedRotatingFileHandler(
        settings.log_dir / "service.log", when="midnight", utc=True,
        backupCount=settings.log_retention_days, encoding="utf-8",
    )
    handler.setFormatter(Formatter(settings))
    log.addHandler(handler)
    return handler


def prune_logs(settings):
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=settings.log_retention_days)
    for path in settings.log_dir.glob("service.log.????-??-??"):
        try:
            date = datetime.strptime(path.name.removeprefix("service.log."), "%Y-%m-%d").date()
        except ValueError:
            continue
        if date < cutoff and not path.is_symlink() and path.is_file():
            path.unlink(missing_ok=True)
