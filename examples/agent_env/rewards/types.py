from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardResult:
    score: float
    components: dict[str, float] = field(default_factory=dict)
    raw: Any = None
    reason: str = ""
    returns_total: bool = True
    reward_version: str = ""

    def record(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "score": float(self.score),
            "components": {str(key): float(value) for key, value in self.components.items()},
            "returns_total": bool(self.returns_total),
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.reward_version:
            payload["reward_version"] = self.reward_version
        if self.raw is not None:
            payload["raw"] = self.raw
        return payload
