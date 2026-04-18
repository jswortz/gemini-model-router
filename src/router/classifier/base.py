from __future__ import annotations

from typing import Protocol


class Classifier(Protocol):
    """Returns a per-backend quality_fit score in [0, 1] that sums (roughly) to 1."""

    def classify(self, prompt: str) -> dict[str, float]: ...
    def rebuild(self) -> None: ...
