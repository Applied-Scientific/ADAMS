from textual.message import Message


class AgentResponse(Message):
    """Message sent when the agent provides a response."""
    def __init__(self, content: str, is_error: bool = False, stream: bool = False, token_usage: dict = None) -> None:
        self.content = content
        self.is_error = is_error
        self.stream = stream
        self.token_usage = token_usage
        super().__init__()


class APIKeyProvided(Message):
    """Message sent when API key is successfully provided."""
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__()


class LogMessage(Message):
    """Message sent when a log record is captured."""
    def __init__(self, text: str, level: str = "INFO",
                 logger_name: str = "", timestamp: str = "") -> None:
        self.text = text
        self.level = level
        self.logger_name = logger_name
        self.timestamp = timestamp
        super().__init__()


class TraceEvent(Message):
    """Message sent when a new trace event is detected from the JSONL file."""
    def __init__(self, event_type: str, timestamp: str, details: dict) -> None:
        self.event_type = event_type
        self.timestamp = timestamp
        self.details = details
        super().__init__()


class StageChanged(Message):
    """Message sent when the agent's execution stage changes."""
    def __init__(self, stage: str, detail: str = "") -> None:
        self.stage = stage
        self.detail = detail
        super().__init__()
