"""Pydantic models specific to the 5-agent generation pipeline.

These are the structured outputs that agents produce.  Phase-1 models
live in ``tribal_knowledge.models``; these are Phase-2 only.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Explorer ─────────────────────────────────────────────────────────


class ModuleMapEntry(BaseModel):
    """One module identified by the Explorer."""

    name: str
    description: str
    files: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    importance: float = 0.5


class ExplorerOutput(BaseModel):
    """Full output from the Explorer agent."""

    framework: str = ""
    modules: list[ModuleMapEntry] = Field(default_factory=list)


# ── Analyst ──────────────────────────────────────────────────────────


class PatternFinding(BaseModel):
    """A design pattern found by the Analyst."""

    name: str
    where: str
    why: str


class AnalystFindings(BaseModel):
    """Structured output from the Analyst for one module."""

    module: str
    overview: str = ""
    data_flow: str = ""
    patterns: list[PatternFinding] = Field(default_factory=list)
    tribal_knowledge: list[str] = Field(default_factory=list)
    cross_module_dependencies: list[str] = Field(default_factory=list)
    conventions: list[str] = Field(default_factory=list)
    modification_patterns: str = ""
    failure_patterns: list[str] = Field(default_factory=list)


# ── Critic ───────────────────────────────────────────────────────────


class CritiqueResult(BaseModel):
    """Structured output from the Critic."""

    overall_score: float = 0.0
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    critique: list[str] = Field(default_factory=list)
    passed: bool = False


# ── Quality report ───────────────────────────────────────────────────


class DocumentScore(BaseModel):
    """Quality score for a single generated document."""

    name: str
    score: float = 0.0
    iterations: int = 0
    tokens_used: int = 0


class QualityReport(BaseModel):
    """Final quality report for all generated documents."""

    documents: list[DocumentScore] = Field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    average_score: float = 0.0
