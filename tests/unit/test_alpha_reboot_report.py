from __future__ import annotations

from pathlib import Path

import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.alpha_reboot_report import (
    load_alpha_reboot_wave_config,
    select_champion_run_id,
)


def test_champion_resolution_fails_when_identity_is_ambiguous() -> None:
    with pytest.raises(ResearchError, match=r"configure champion\.run_id explicitly"):
        select_champion_run_id(("run-a", "run-b"), None)


def test_champion_resolution_never_uses_performance_order() -> None:
    assert select_champion_run_id(("run-z", "run-a"), "run-z") == "run-z"


def test_wave_configuration_enforces_the_total_trial_budget(tmp_path: Path) -> None:
    config = tmp_path / "wave.yaml"
    config.write_text(
        """
wave_id: alpha_reboot_wave1
research_context: development_seen
maximum_total_trials: 19
campaigns: [a, b, c, d]
champion:
  run_id: run-a
  strategy: carry_multi_horizon:v1
  portfolio: buffered_three_sleeve_neutral:v1
  portfolio_weights:
    carry_weight: 0.60
    fast_strength_weight: 0.30
    slow_strength_weight: 0.10
""",
        encoding="utf-8",
    )

    with pytest.raises(ResearchError, match="less than or equal to 18"):
        load_alpha_reboot_wave_config(config)
