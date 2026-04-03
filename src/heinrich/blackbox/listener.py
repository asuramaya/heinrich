"""Token stream listener. Watches output tokens. Detects distribution shift."""
from __future__ import annotations
from collections import Counter
import math


class Listener:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.tokens: list[int] = []
        self.alerts: list[dict] = []

    def hear(self, token_id: int) -> dict | None:
        self.tokens.append(token_id)
        n = len(self.tokens)
        if n < self.window_size * 2:
            return None
        # Compare last window to previous window
        prev = self.tokens[n - 2 * self.window_size : n - self.window_size]
        curr = self.tokens[n - self.window_size : n]
        kl = _kl_from_counts(Counter(prev), Counter(curr))
        if kl > self._threshold():
            alert = {"position": n, "kl": round(kl, 4), "token": token_id}
            self.alerts.append(alert)
            return alert
        return None

    def _threshold(self) -> float:
        if not self.alerts:
            return 0.5
        # Adaptive: mean + 2*std of previous KLs
        kls = [a["kl"] for a in self.alerts[-20:]]
        mean = sum(kls) / len(kls)
        std = (sum((k - mean) ** 2 for k in kls) / len(kls)) ** 0.5
        return mean + 2 * std


def _kl_from_counts(p_counts: Counter, q_counts: Counter) -> float:
    total_p = sum(p_counts.values())
    total_q = sum(q_counts.values())
    if total_p == 0 or total_q == 0:
        return 0.0
    all_tokens = set(p_counts) | set(q_counts)
    kl = 0.0
    for t in all_tokens:
        p = (p_counts.get(t, 0) + 1e-10) / total_p
        q = (q_counts.get(t, 0) + 1e-10) / total_q
        kl += p * math.log(p / q)
    return kl
