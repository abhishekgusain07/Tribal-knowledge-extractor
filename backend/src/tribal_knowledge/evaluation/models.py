"""Pydantic models for evaluation results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentEvaluation(BaseModel):
    """Evaluation scores for a single generated context document."""

    name: str
    entity_coverage: float = 0.0
    factual_accuracy: float = 0.0
    conciseness: float = 0.0
    structural_completeness: float = 0.0
    llm_usefulness: float = 0.0
    invalid_references: list[str] = Field(default_factory=list)
    missing_entities: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """Weighted composite of all dimension scores.

        Weights:
            factual_accuracy      0.30
            entity_coverage       0.25
            conciseness           0.20
            structural_completeness 0.10
            llm_usefulness        0.15
        """
        return (
            self.factual_accuracy * 0.30
            + self.entity_coverage * 0.25
            + self.conciseness * 0.20
            + self.structural_completeness * 0.10
            + self.llm_usefulness * 0.15
        )


class EvaluationReport(BaseModel):
    """Aggregate evaluation report across all generated documents."""

    documents: list[DocumentEvaluation] = Field(default_factory=list)
    average_composite: float = 0.0
    average_by_dimension: dict[str, float] = Field(default_factory=dict)
    total_invalid_references: int = 0
    total_missing_entities: int = 0
    evaluated_at: str = ""
