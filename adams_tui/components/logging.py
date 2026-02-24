import logging
from datetime import datetime
from .messages import LogMessage

class TextualLogHandler(logging.Handler):
    """Hooks into Python logging to send updates to the TUI."""
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    def emit(self, record):
        try:
            msg = self.format(record)
            # Filter out noisy libraries if needed
            if "httpx" in record.name or "httpcore" in record.name:
                return
            ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            self.app_instance.post_message(LogMessage(
                text=msg,
                level=record.levelname,
                logger_name=record.name,
                timestamp=ts,
            ))
            # Write log records to the trace file for historical viewing
            if hasattr(self.app_instance, 'trace_processor') and self.app_instance.trace_processor:
                self.app_instance.trace_processor._write_event({
                    "event": "log_record",
                    "timestamp": datetime.now().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": msg,
                })
        except Exception:
            self.handleError(record)
