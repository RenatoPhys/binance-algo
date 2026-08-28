"""Strict YAML campaign schema and deterministic Cartesian expansion."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Self, cast

import orjson
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from binance_algo.common.errors import ResearchError
from binance_algo.config import ResearchConfig
from binance_algo.research.contracts import ValidationProfile
from binance_algo.research.datasets.references import DatasetReference, load_dataset_reference
from binance_algo.research.experiments.ablation import AblationDeclaration
from binance_algo.research.experiments.canonical import canonical_sha256, canonicalize
from binance_algo.research.experiments.ids import experiment_id
from binance_algo.research.experiments.models import (
    ArtifactPolicy,
    CodeFingerprint,
    DatasetIdentity,
    ExperimentSpec,
    FeatureSetIdentity,
    LabelIdentity,
    ParameterizedComponent,
    VersionedComponent,
)
from binance_algo.research.experiments.provenance import build_code_fingerprint
from binance_algo.research.features.registry import phase3_feature_set
from binance_algo.research.labels.forward_returns import PHASE3_LABEL_REGISTRY
from binance_algo.research.portfolio.registry import build_portfolio_policy
from binance_algo.research.strategies.registry import build_strategy


class StrictCampaignModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CampaignMetadata(StrictCampaignModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    artifact_policy: ArtifactPolicy = ArtifactPolicy.SUMMARY
    max_trials: int = Field(default=500, ge=1, le=100_000)


class CampaignDataset(StrictCampaignModel):
    manifest: str = Field(min_length=1)


class CampaignFeatureSet(StrictCampaignModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class CampaignLabel(StrictCampaignModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    horizon_minutes: int = Field(gt=0)


class ComponentSearch(StrictCampaignModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    fixed: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, tuple[Any, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_search_space(self) -> Self:
        overlap = set(self.fixed).intersection(self.grid)
        if overlap:
            raise ValueError(f"parameters cannot be both fixed and gridded: {sorted(overlap)}")
        if any(not values for values in self.grid.values()):
            raise ValueError("campaign grid axes cannot be empty")
        fixed = canonicalize(self.fixed, field_path=f"{self.name}.fixed")
        grid = canonicalize(self.grid, field_path=f"{self.name}.grid")
        object.__setattr__(self, "fixed", cast(dict[str, Any], fixed))
        object.__setattr__(
            self,
            "grid",
            {key: tuple(values) for key, values in cast(dict[str, list[Any]], grid).items()},
        )
        return self


class ParameterSumConstraint(StrictCampaignModel):
    fields: tuple[str, ...] = Field(min_length=1)
    equals: float
    tolerance: float = Field(default=1e-12, gt=0)


class CampaignValidation(StrictCampaignModel):
    profile: ValidationProfile = ValidationProfile.FULL
    split_plan: str = "expanding_walk_forward_v1"
    train_days: int | None = Field(default=None, ge=7, le=3650)
    test_days: int | None = Field(default=None, ge=1, le=365)
    embargo_bars: int | None = Field(default=None, ge=1, le=24)
    stress_cost_multipliers: tuple[float, ...] | None = None
    stress_signal_delay_bars: tuple[int, ...] | None = None
    bootstrap_samples: int | None = Field(default=None, ge=100, le=10_000)
    bootstrap_block_hours: int | None = Field(default=None, ge=2, le=720)
    require_parameter_sum: tuple[ParameterSumConstraint, ...] = ()

    @field_validator("require_parameter_sum", mode="before")
    @classmethod
    def normalize_constraints(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, dict):
            return (value,)
        return value

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        if self.profile is ValidationProfile.DISCOVERY:
            if self.stress_cost_multipliers not in {None, (1.5,)}:
                raise ValueError("discovery runs only the 1.5x cost stress")
            if self.stress_signal_delay_bars not in {None, (1,)}:
                raise ValueError("discovery runs only the one-bar signal-delay stress")
            if self.bootstrap_samples is not None or self.bootstrap_block_hours is not None:
                raise ValueError("discovery does not run bootstrap validation")
        return self

    def resolved_cost_multipliers(self) -> tuple[float, ...]:
        if self.stress_cost_multipliers is not None:
            return self.stress_cost_multipliers
        return (1.5,) if self.profile is ValidationProfile.DISCOVERY else (1.5, 2.0)

    def resolved_signal_delays(self) -> tuple[int, ...]:
        return self.stress_signal_delay_bars or (1,)


class CampaignRunnerSpec(StrictCampaignModel):
    max_workers: int = Field(default=1, ge=1, le=32)
    fail_fast: bool = False
    resume: bool = True


class CampaignSpec(StrictCampaignModel):
    campaign: CampaignMetadata
    dataset: CampaignDataset
    feature_set: CampaignFeatureSet
    label: CampaignLabel
    strategy: ComponentSearch
    portfolio: ComponentSearch
    execution: ComponentSearch
    costs: ComponentSearch
    validation: CampaignValidation
    runner: CampaignRunnerSpec = Field(default_factory=CampaignRunnerSpec)
    ablation: tuple[AblationDeclaration, ...] = ()

    @model_validator(mode="after")
    def enforce_discovery_limits(self) -> Self:
        if (
            self.validation.profile is ValidationProfile.DISCOVERY
            and self.campaign.artifact_policy is not ArtifactPolicy.SUMMARY
        ):
            raise ValueError("discovery requires artifact_policy: summary")
        return self


@dataclass(frozen=True, slots=True)
class CampaignTrial:
    ordinal: int
    experiment_id: str
    spec: ExperimentSpec
    tags: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    campaign_id: str
    source: CampaignSpec
    dataset_reference: DatasetReference
    possible_combinations: int
    valid_combinations: int
    rejected_by_constraints: int
    trials: tuple[CampaignTrial, ...]

    def stored_payload(self) -> dict[str, Any]:
        return {
            "source_spec": self.source.model_dump(mode="json"),
            "planning": {
                "possible_combinations": self.possible_combinations,
                "valid_combinations": self.valid_combinations,
                "rejected_by_constraints": self.rejected_by_constraints,
            },
        }


def load_campaign_spec(path: Path) -> CampaignSpec:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ResearchError("campaign YAML root must be a mapping")
        return CampaignSpec.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ResearchError(f"cannot load campaign file {path}: {exc}") from exc


def _manifest_path(spec: CampaignSpec, project_root: Path, data_root: Path) -> Path:
    if spec.dataset.manifest == "latest":
        candidates = sorted(
            data_root.glob("gold/binance/usdm/research_dataset/version=*/dataset.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not candidates:
            raise ResearchError(
                "campaign requests the latest dataset, but no local research dataset exists"
            )
        return candidates[-1]
    declared = Path(spec.dataset.manifest)
    path = declared if declared.is_absolute() else project_root / declared
    path = path.resolve()
    if not path.is_file():
        raise ResearchError(f"campaign dataset manifest does not exist: {path}")
    return path


def _label_identity(spec: CampaignLabel) -> LabelIdentity:
    version = spec.version if spec.version.startswith("v") else f"v{spec.version}"
    label_id = f"{spec.name}:{version}"
    definition = PHASE3_LABEL_REGISTRY.resolve_id(label_id)
    if definition.horizon_minutes != spec.horizon_minutes:
        raise ResearchError(
            f"campaign label horizon differs from {definition.label_id}: "
            f"{spec.horizon_minutes} != {definition.horizon_minutes}"
        )
    return LabelIdentity(
        label_id=definition.label_id,
        version=definition.version,
        target_column=definition.target_column,
    )


def _grid_combinations(spec: CampaignSpec) -> tuple[dict[str, dict[str, Any]], ...]:
    axes: list[tuple[str, str, tuple[Any, ...]]] = []
    for scope, component in (("strategy", spec.strategy), ("portfolio", spec.portfolio)):
        axes.extend((scope, name, tuple(values)) for name, values in sorted(component.grid.items()))
    if not axes:
        return ({"strategy": {}, "portfolio": {}},)
    combinations = []
    for values in itertools.product(*(axis[2] for axis in axes)):
        selected: dict[str, dict[str, Any]] = {"strategy": {}, "portfolio": {}}
        for (scope, name, _), value in zip(axes, values, strict=True):
            selected[scope][name] = value
        combinations.append(selected)
    return tuple(combinations)


def _constraint_values(
    strategy: Mapping[str, Any],
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    values = {
        **{f"strategy.{key}": value for key, value in strategy.items()},
        **{f"portfolio.{key}": value for key, value in portfolio.items()},
    }
    for key in set(strategy).union(portfolio):
        matches = [mapping[key] for mapping in (strategy, portfolio) if key in mapping]
        if len(matches) == 1:
            values[key] = matches[0]
    return values


def _constraints_pass(
    constraints: tuple[ParameterSumConstraint, ...],
    *,
    strategy: Mapping[str, Any],
    portfolio: Mapping[str, Any],
) -> bool:
    values = _constraint_values(strategy, portfolio)
    for constraint in constraints:
        try:
            total = sum(float(values[field]) for field in constraint.fields)
        except (KeyError, TypeError, ValueError) as exc:
            raise ResearchError(
                f"campaign constraint references an unknown or non-numeric field: "
                f"{constraint.fields}"
            ) from exc
        if abs(total - constraint.equals) > constraint.tolerance:
            return False
    return True


def _cost_parameters(
    component: ComponentSearch,
    config: ResearchConfig,
) -> dict[str, Any]:
    fixed = dict(component.fixed)
    try:
        multiplier = Decimal(str(fixed.pop("cost_multiplier", 1.0)))
    except (ValueError, ArithmeticError) as exc:
        raise ResearchError("cost_multiplier must be a finite decimal") from exc
    if fixed:
        raise ResearchError(f"unsupported fixed cost parameters: {sorted(fixed)}")
    if not multiplier.is_finite() or multiplier < 0:
        raise ResearchError("cost_multiplier must be finite and non-negative")
    fee_schedule = config.fee_schedule.model_copy(
        update={
            "maker_fee_rate": config.fee_schedule.maker_fee_rate * multiplier,
            "taker_fee_rate": config.fee_schedule.taker_fee_rate * multiplier,
        }
    )
    return {
        "spread_bps": config.spread_bps * multiplier,
        "slippage_bps": config.slippage_bps * multiplier,
        "initial_capital_usdt": config.initial_capital_usdt,
        "fee_schedule": fee_schedule.model_dump(mode="json"),
    }


def plan_campaign(
    source: CampaignSpec,
    *,
    project_root: Path,
    data_root: Path,
    research_config: ResearchConfig,
    allow_large_campaign: bool = False,
    code_fingerprint: CodeFingerprint | None = None,
) -> CampaignPlan:
    manifest_path = _manifest_path(source, project_root, data_root)
    dataset_reference = load_dataset_reference(manifest_path)
    feature_set = phase3_feature_set(research_config)
    declared_version = (
        source.feature_set.version
        if source.feature_set.version.startswith("v")
        else f"v{source.feature_set.version}"
    )
    declared_feature_set_id = f"{source.feature_set.name}:{declared_version}"
    if declared_feature_set_id != feature_set.feature_set_id:
        raise ResearchError(f"unsupported feature set: {declared_feature_set_id}")
    if dataset_reference.feature_set_id != feature_set.feature_set_id:
        raise ResearchError("campaign feature set differs from the dataset manifest")
    label = _label_identity(source.label)
    if dataset_reference.label_id != label.label_id:
        raise ResearchError("campaign label differs from the dataset manifest")
    fingerprint = code_fingerprint or build_code_fingerprint(project_root)
    semantic_payload = {
        "campaign": source.campaign.model_dump(mode="json", exclude={"max_trials"}),
        "dataset": dataset_reference.identity_payload(),
        "feature_set": {
            "feature_set_id": feature_set.feature_set_id,
            "canonical_checksum": feature_set.canonical_checksum,
        },
        "label": label.model_dump(mode="json"),
        "strategy": source.strategy.model_dump(mode="json"),
        "portfolio": source.portfolio.model_dump(mode="json"),
        "execution": source.execution.model_dump(mode="json"),
        "costs": source.costs.model_dump(mode="json"),
        "validation": source.validation.model_dump(mode="json"),
        "ablation": [item.model_dump(mode="json") for item in source.ablation],
        "code_fingerprint": fingerprint.model_dump(mode="json"),
    }
    campaign_id = canonical_sha256(semantic_payload)
    combinations = _grid_combinations(source)
    possible = len(combinations)
    valid: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for combination in combinations:
        strategy_parameters = {**source.strategy.fixed, **combination["strategy"]}
        portfolio_parameters = {**source.portfolio.fixed, **combination["portfolio"]}
        if _constraints_pass(
            source.validation.require_parameter_sum,
            strategy=strategy_parameters,
            portfolio=portfolio_parameters,
        ):
            build_strategy(source.strategy.name, source.strategy.version, strategy_parameters)
            build_portfolio_policy(
                source.portfolio.name,
                source.portfolio.version,
                portfolio_parameters,
            )
            valid.append((strategy_parameters, portfolio_parameters))
    if not valid:
        raise ResearchError("campaign constraints removed every parameter combination")
    if len(valid) > source.campaign.max_trials and not allow_large_campaign:
        raise ResearchError(
            f"campaign expands to {len(valid)} valid trials, above max_trials="
            f"{source.campaign.max_trials}; pass --allow-large-campaign to continue"
        )
    if source.execution.grid or source.costs.grid:
        raise ResearchError("execution and cost grids are not supported in campaign schema v1")
    if source.execution.name != "bar_next_open":
        raise ResearchError("only bar_next_open execution is supported")
    execution_parameters = {"lag_bars": 1, **source.execution.fixed}
    validation = source.validation
    if validation.split_plan != "expanding_walk_forward_v1":
        raise ResearchError(f"unsupported split plan: {validation.split_plan}")
    validation_parameters: dict[str, Any] = {
        "profile": validation.profile.value,
        "stress_cost_multipliers": list(validation.resolved_cost_multipliers()),
        "stress_signal_delay_bars": list(validation.resolved_signal_delays()),
    }
    if validation.profile is ValidationProfile.FULL:
        validation_parameters.update(
            {
                "bootstrap_samples": validation.bootstrap_samples
                or research_config.block_bootstrap_samples,
                "bootstrap_block_hours": validation.bootstrap_block_hours
                or research_config.block_bootstrap_hours,
            }
        )
    trials = []
    for ordinal, (strategy_parameters, portfolio_parameters) in enumerate(valid):
        spec = ExperimentSpec(
            hypothesis_id=source.campaign.hypothesis_id,
            campaign_id=campaign_id,
            dataset_reference=DatasetIdentity.from_reference(dataset_reference),
            feature_set=FeatureSetIdentity(
                feature_set_id=feature_set.feature_set_id,
                canonical_checksum=feature_set.canonical_checksum,
            ),
            label=label,
            strategy=VersionedComponent(
                component_id=source.strategy.name,
                version=source.strategy.version,
            ),
            strategy_parameters=strategy_parameters,
            portfolio_policy=VersionedComponent(
                component_id=source.portfolio.name,
                version=source.portfolio.version,
            ),
            portfolio_parameters=portfolio_parameters,
            execution_model=ParameterizedComponent(
                component_id=source.execution.name,
                version=source.execution.version,
                parameters=execution_parameters,
            ),
            cost_model=ParameterizedComponent(
                component_id=source.costs.name,
                version=source.costs.version,
                parameters=_cost_parameters(source.costs, research_config),
            ),
            split_plan=ParameterizedComponent(
                component_id="expanding_walk_forward",
                version="1",
                parameters={
                    "train_days": validation.train_days or research_config.walk_forward_train_days,
                    "test_days": validation.test_days or research_config.walk_forward_test_days,
                    "embargo_bars": validation.embargo_bars or research_config.embargo_bars,
                },
            ),
            validation_plan=ParameterizedComponent(
                component_id="phase3_validation",
                version="1",
                parameters=validation_parameters,
            ),
            random_seed=research_config.random_seed,
            code_fingerprint=fingerprint,
            artifact_policy=source.campaign.artifact_policy,
        )
        identifier = experiment_id(spec)
        trials.append(
            CampaignTrial(
                ordinal=ordinal,
                experiment_id=identifier,
                spec=spec,
                tags={
                    "strategy_parameters": strategy_parameters,
                    "portfolio_parameters": portfolio_parameters,
                },
            )
        )
    trials.sort(key=lambda trial: trial.experiment_id)
    trials = [
        CampaignTrial(
            ordinal=ordinal,
            experiment_id=trial.experiment_id,
            spec=trial.spec,
            tags=trial.tags,
        )
        for ordinal, trial in enumerate(trials)
    ]
    return CampaignPlan(
        campaign_id=campaign_id,
        source=source,
        dataset_reference=dataset_reference,
        possible_combinations=possible,
        valid_combinations=len(trials),
        rejected_by_constraints=possible - len(trials),
        trials=tuple(trials),
    )


def campaign_spec_from_stored_payload(payload_json: str) -> CampaignSpec:
    try:
        payload = orjson.loads(payload_json)
        return CampaignSpec.model_validate(payload["source_spec"])
    except (KeyError, TypeError, orjson.JSONDecodeError, ValidationError) as exc:
        raise ResearchError(f"stored campaign spec is invalid: {exc}") from exc


__all__ = [
    "CampaignDataset",
    "CampaignFeatureSet",
    "CampaignLabel",
    "CampaignMetadata",
    "CampaignPlan",
    "CampaignRunnerSpec",
    "CampaignSpec",
    "CampaignTrial",
    "CampaignValidation",
    "ComponentSearch",
    "ParameterSumConstraint",
    "campaign_spec_from_stored_payload",
    "load_campaign_spec",
    "plan_campaign",
]
