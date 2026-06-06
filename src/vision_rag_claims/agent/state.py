"""Shared state dict that flows through every LangGraph node."""

from typing import TypedDict

import numpy as np

from vision_rag_claims.schemas import (
    ClaimReport,
    CoverageDecision,
    DetectionResult,
    PolicyChunk,
    SeverityLevel,
)


class AgentState(TypedDict, total=False):
    image: np.ndarray
    detection_result: DetectionResult
    severity: SeverityLevel | None
    retrieved_chunks: dict[str, list[PolicyChunk]]
    coverage_decisions: list[CoverageDecision]
    report: ClaimReport | None
