"""Benders-cut data structures and deterministic normalization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BendersCut:
    """Scenario-specific optimality cut ``theta_s >= alpha + beta^T y``."""

    scenario_index: int
    alpha: float
    beta: tuple[float, ...]
    source_y: tuple[int, ...]
    source_recourse: float

    def rhs(self, y: tuple[float, ...] | tuple[int, ...]) -> float:
        if len(y) != len(self.beta):
            raise ValueError("first-stage vector length does not match cut coefficients")
        return self.alpha + sum(coefficient * value for coefficient, value in zip(self.beta, y))

    def violation(self, y: tuple[float, ...] | tuple[int, ...], theta: float) -> float:
        return self.rhs(y) - theta

    def normalized_key(self, digits: int = 10) -> tuple[object, ...]:
        return (
            self.scenario_index,
            round(self.alpha, digits),
            *(round(value, digits) for value in self.beta),
        )

    @property
    def fingerprint(self) -> str:
        payload = repr(self.normalized_key(12)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]
