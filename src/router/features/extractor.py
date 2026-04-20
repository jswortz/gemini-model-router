from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_AGENT_KEYWORDS = (
    "refactor",
    "fix",
    "run tests",
    "grep",
    "commit",
    "debug",
    "implement",
    "write a",
    "migrate",
    "trace through",
    "audit",
    "instrument",
    "extract",
    "split this",
)
_QA_KEYWORDS = (
    "what is",
    "explain",
    "summarize",
    "why does",
    "translate",
    "convert",
    "define",
    "give me",
    "list",
)

_PATH_RE = re.compile(
    r"\b[\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|rb|cpp|cc|h|hpp|md|yaml|yml|toml|json|sh)\b"
)
_URL_RE = re.compile(r"https?://\S+")
_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)

_SECRET_RES = (
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI-style secret
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack token
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),  # GitHub PAT
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),  # Google API key
)


@dataclass
class PromptFeatures:
    n_chars: int
    n_tokens_est: int
    code_fence_count: int
    code_ratio: float
    has_path_ref: bool
    has_url: bool
    agent_keywords: int
    qa_keywords: int
    tool_required: bool
    sensitive: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _count_keywords(text: str, words: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(lower.count(w) for w in words)


def extract(prompt: str) -> PromptFeatures:
    n_chars = len(prompt)
    n_tokens_est = max(1, n_chars // 4)

    fences = _FENCE_RE.findall(prompt)
    code_fence_count = len(fences)
    code_chars = sum(len(f) for f in fences)
    code_ratio = code_chars / n_chars if n_chars else 0.0

    has_path_ref = bool(_PATH_RE.search(prompt))
    has_url = bool(_URL_RE.search(prompt))

    agent_keywords = _count_keywords(prompt, _AGENT_KEYWORDS)
    qa_keywords = _count_keywords(prompt, _QA_KEYWORDS)
    tool_required = agent_keywords > 0 and (has_path_ref or code_fence_count > 0)

    sensitive = any(rx.search(prompt) for rx in _SECRET_RES)

    return PromptFeatures(
        n_chars=n_chars,
        n_tokens_est=n_tokens_est,
        code_fence_count=code_fence_count,
        code_ratio=round(code_ratio, 4),
        has_path_ref=has_path_ref,
        has_url=has_url,
        agent_keywords=agent_keywords,
        qa_keywords=qa_keywords,
        tool_required=tool_required,
        sensitive=sensitive,
    )
