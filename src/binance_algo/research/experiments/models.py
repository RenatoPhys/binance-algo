"""Strict immutable models for experiment definitions and registry entities."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from binance_algo.common.errors import ResearchError
from binance_algo.research.datasets.references import DatasetReference
from binance_algo.research.experiments.canonical import canonicalize


class ImmutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HypothesisStatus(StrEnum):
    DRAFT = "DRAFT"
    READY = "READY"
    TESTED = "TESTED"
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALIDATED = "INVALIDATED"


class CampaignStatus(StrEnum):
    PLANNED = "PLANNED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunStatus(StrEnum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    STALE = "STALE"


class MetricScope(StrEnum):
    TRAIN = "TRAIN"
    INNER_VALIDATION = "INNER_VALIDATION"
    TEST = "TEST"
    LOCKBOX = "LOCKBOX"
    STRESS = "STRESS"
    CAMPAIGN = "CAMPAIGN"


class FeatureEvaluationType(StrEnum):
    UNIVARIATE = "UNIVARIATE"
    INCREMENTAL = "INCREMENTAL"
    ABLATION = "ABLATION"
    PERMUTATION = "PERMUTATION"
    REGIME_STABILITY = "REGIME_STABILITY"
    TURNOVER_IMPACT = "TURNOVER_IMPACT"
    COST_SENSITIVITY = "COST_SENSITIVITY"


class FeatureDecision(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONDITIONAL = "CONDITIONAL"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALIDATED = "INVALIDATED"
    RETEST_REQUIRED = "RETEST_REQUIRED"


class ArtifactPolicy(StrEnum):
    SUMMARY = "summary"
    FULL = "full"


class ProvenanceQuality(StrEnum):
    GIT_CLEAN = "git_clean"
    GIT_DIRTY = "git_dirty"
    FALLBACK_SOURCE_HASH = "fallback_source_hash"


class CodeFingerprint(ImmutableModel):
    git_commit: str | None = None
    git_dirty: bool
    git_diff_sha256: str | None = None
    source_tree_sha256: str | None = None
    provenance_quality: ProvenanceQuality

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if self.provenance_quality is ProvenanceQuality.GIT_CLEAN:
            if (
                self.git_dirty
                or not self.git_commit
                or self.git_diff_sha256 is not None
                or self.source_tree_sha256 is not None
            ):
                raise ValueError("clean Git provenance requires only a commit")
        elif self.provenance_quality is ProvenanceQuality.GIT_DIRTY:
            if (
                not self.git_dirty
                or not self.git_commit
                or not self.git_diff_sha256
                or self.source_tree_sha256 is not None
            ):
                raise ValueError("dirty Git provenance requires only commit and diff checksum")
        elif (
            self.git_commit is not None
            or self.git_dirty
            or self.git_diff_sha256 is not None
            or not self.source_tree_sha256
        ):
            raise ValueError("fallback provenance requires only a source-tree checksum")
        return self


class DatasetIdentity(ImmutableModel):
    dataset_id: str = Field(min_length=1)
    dataset_schema_version: int = Field(ge=1)
    feature_set_id: str = Field(min_length=1)
    label_id: str = Field(min_length=1)
    universe_version: str = Field(min_length=1)
    start_time_ms: int
    end_time_ms: int
    row_count: int = Field(ge=0)
    content_checksum: str = Field(min_length=1)
    fingerprint_method: str = Field(min_length=1)

    @classmethod
    def from_reference(cls, reference: DatasetReference) -> DatasetIdentity:
        return cls.model_validate(reference.identity_payload())

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_time_ms < self.start_time_ms:
            raise ValueError("dataset end must not precede start")
        return self


class VersionedComponent(ImmutableModel):
    component_id: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ParameterizedComponent(VersionedComponent):
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        normalized = canonicalize(
            self.parameters,
            field_path=f"{self.component_id}.parameters",
        )
        object.__setattr__(self, "parameters", cast(dict[str, Any], normalized))
        return self


class FeatureSetIdentity(ImmutableModel):
    feature_set_id: str = Field(min_length=1)
    canonical_checksum: str = Field(min_length=1)


class LabelIdentity(ImmutableModel):
    label_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    target_column: str = Field(min_length=1)


class HypothesisSpec(ImmutableModel):
    hypothesis_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    expected_direction: str | None = None
    expected_horizon: str | None = None
    target_universe: str | None = None
    preregistered_success_criteria: dict[str, Any]
    status: HypothesisStatus = HypothesisStatus.DRAFT
    notes: str | None = None

    @model_validator(mode="after")
    def validate_criteria(self) -> Self:
        normalized = canonicalize(
            self.preregistered_success_criteria,
            field_path="preregistered_success_criteria",
        )
        object.__setattr__(
            self,
            "preregistered_success_criteria",
            cast(dict[str, Any], normalized),
        )
        return self


class FeatureEvaluationSpec(ImmutableModel):
    run_id: str = Field(min_length=1)
    feature_id: str = Field(min_length=1)
    evaluation_type: FeatureEvaluationType
    scope: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    metric_value: float | None = None
    decision: FeatureDecision
    decision_reason: str = Field(min_length=1)
    context: dict[str, Any]

    @field_validator("metric_value")
    @classmethod
    def validate_metric_value(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("feature evaluation metric must be finite when present")
        return value

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        normalized = canonicalize(self.context, field_path="feature_evaluation.context")
        object.__setattr__(self, "context", cast(dict[str, Any], normalized))
        return self


class ExperimentSpec(ImmutableModel):
    hypothesis_id: str = Field(min_length=1)
    campaign_id: str | None = None
    dataset_reference: DatasetIdentity
    feature_set: FeatureSetIdentity
    label: LabelIdentity
    strategy: VersionedComponent
    strategy_parameters: dict[str, Any] = Field(default_factory=dict)
    portfolio_policy: VersionedComponent
    portfolio_parameters: dict[str, Any] = Field(default_factory=dict)
    execution_model: ParameterizedComponent
    cost_model: ParameterizedComponent
    split_plan: ParameterizedComponent
    validation_plan: ParameterizedComponent
    random_seed: int
    code_fingerprint: CodeFingerprint
    artifact_policy: ArtifactPolicy = ArtifactPolicy.SUMMARY

    @model_validator(mode="after")
    def validate_canonical_identity(self) -> Self:
        try:
            strategy_parameters = canonicalize(
                self.strategy_parameters,
                field_path="strategy_parameters",
            )
            portfolio_parameters = canonicalize(
                self.portfolio_parameters,
                field_path="portfolio_parameters",
            )
            object.__setattr__(
                self,
                "strategy_parameters",
                cast(dict[str, Any], strategy_parameters),
            )
            object.__setattr__(
                self,
                "portfolio_parameters",
                cast(dict[str, Any], portfolio_parameters),
            )
            canonicalize(self.model_dump(mode="python"), field_path="experiment_spec")
        except ResearchError as exc:
            raise ValueError(str(exc)) from exc
        if self.feature_set.feature_set_id != self.dataset_reference.feature_set_id:
            raise ValueError("experiment feature set differs from dataset reference")
        if self.label.label_id != self.dataset_reference.label_id:
            raise ValueError("experiment label differs from dataset reference")
        return self


__all__ = [
    "ArtifactPolicy",
    "CampaignStatus",
    "CodeFingerprint",
    "DatasetIdentity",
    "ExperimentSpec",
    "FeatureDecision",
    "FeatureEvaluationSpec",
    "FeatureEvaluationType",
    "FeatureSetIdentity",
    "HypothesisSpec",
    "HypothesisStatus",
    "LabelIdentity",
    "MetricScope",
    "ParameterizedComponent",
    "ProvenanceQuality",
    "RunStatus",
    "VersionedComponent",
]
