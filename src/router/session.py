from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_id: str
    locked_backend: str | None = None
    backend_session_ids: dict[str, str] = field(default_factory=dict)
    turn_count: int = 0
    last_prompt: str | None = None

    def is_first_turn(self) -> bool:
        return self.turn_count == 0

    def lock(self, backend: str) -> None:
        self.locked_backend = backend

    def unlock(self) -> None:
        self.locked_backend = None

    def remember_vendor_session(self, backend: str, vendor_id: str) -> None:
        if vendor_id:
            self.backend_session_ids[backend] = vendor_id

    def vendor_session_for(self, backend: str) -> str | None:
        return self.backend_session_ids.get(backend)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        st = self._sessions.get(session_id)
        if st is None:
            st = SessionState(session_id=session_id)
            self._sessions[session_id] = st
        return st

    def unlock(self, session_id: str) -> bool:
        st = self._sessions.get(session_id)
        if st is None or st.locked_backend is None:
            return False
        st.unlock()
        return True
