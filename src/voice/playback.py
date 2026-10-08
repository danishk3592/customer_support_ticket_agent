from __future__ import annotations


class VoicePlaybackStore:
    """Session-local cache keyed by chat-message ID."""

    def __init__(self) -> None:
        self.audio: dict[str, bytes] = {}
        self.errors: dict[str, str] = {}

    def get(self, message_id: str) -> bytes | None:
        return self.audio.get(message_id)

    def put(self, message_id: str, audio: bytes) -> None:
        self.audio[message_id] = audio
        self.errors.pop(message_id, None)

    def set_error(self, message_id: str, error: str) -> None:
        self.errors[message_id] = error

    def get_error(self, message_id: str) -> str | None:
        return self.errors.get(message_id)
