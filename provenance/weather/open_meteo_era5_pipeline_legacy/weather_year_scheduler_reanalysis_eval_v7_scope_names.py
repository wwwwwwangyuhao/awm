"""Balanced per-episode weather-year scheduler.

Canonical manifest keys
-----------------------
- training_weather_years
- reanalysis_evaluation_years
- station_evaluation_years
- station_overlap_years

Only training_weather_years are eligible for policy training. Evaluation
years are loaded for audit metadata only and never enter the shuffled cycles.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolicyWeatherSpec:
    policy_idx: int
    label: str
    group: str
    years: tuple[str, ...]


class _ShuffledCycle:
    def __init__(self, values: list[str], seed: int):
        if not values:
            raise ValueError("Weather-year pool cannot be empty")
        self._values = list(values)
        self._rng = random.Random(seed)
        self._queue: list[str] = []
        self._refill()

    def _refill(self) -> None:
        self._queue = list(self._values)
        self._rng.shuffle(self._queue)

    def next(self) -> str:
        if not self._queue:
            self._refill()
        return self._queue.pop()

    def state_dict(self) -> dict[str, Any]:
        return {
            "values": list(self._values),
            "queue": list(self._queue),
            "rng_state": self._rng.getstate(),
        }


class WeatherYearScheduler:
    def __init__(self, manifest_path: str | Path, seed: int):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )

        self.source_model = str(
            self.manifest.get("source_model", "")
        ).strip().lower()
        supported_models = {
            "era5",
            "era5_seamless",
            "era5_land",
        }
        if self.source_model not in supported_models:
            raise ValueError(
                "Unsupported weather scenario source_model="
                f"{self.source_model!r}; expected one of "
                f"{sorted(supported_models)}"
            )

        splits = self.manifest.get("splits")
        if not isinstance(splits, dict):
            raise KeyError("Manifest has no valid 'splits' object")

        # Canonical key first; old key is a temporary compatibility fallback.
        training_values = splits.get(
            "training_weather_years",
            splits.get("train_years"),
        )
        if not isinstance(training_values, list) or not training_values:
            raise KeyError(
                "Manifest must define non-empty "
                "splits.training_weather_years"
            )

        self._training_years = tuple(
            str(year)
            for year in training_values
        )
        self._reanalysis_evaluation_years = tuple(
            str(year)
            for year in splits.get(
                "reanalysis_evaluation_years",
                splits.get("validation_years", []),
            )
        )
        self._station_evaluation_years = tuple(
            str(year)
            for year in splits.get(
                "station_evaluation_years",
                splits.get("holdout_test_years", []),
            )
        )
        self._station_overlap_years = tuple(
            str(year)
            for year in splits.get(
                "station_overlap_years",
                self._station_evaluation_years,
            )
        )

        self._validate_split_roles()

        pools = self.manifest["main_grouping"]["follower_pools"]
        self.specs = [
            PolicyWeatherSpec(
                0,
                "leader_all_weather",
                "all_train",
                self._training_years,
            ),
            PolicyWeatherSpec(
                1,
                "follower_low_et0",
                "low_et0",
                tuple(str(year) for year in pools["low_et0"]),
            ),
            PolicyWeatherSpec(
                2,
                "follower_medium_et0",
                "medium_et0",
                tuple(str(year) for year in pools["medium_et0"]),
            ),
            PolicyWeatherSpec(
                3,
                "follower_high_et0",
                "high_et0",
                tuple(str(year) for year in pools["high_et0"]),
            ),
        ]
        self._validate_follower_pools()

        self._cycles = {
            spec.policy_idx: _ShuffledCycle(
                list(spec.years),
                seed + 1009 * spec.policy_idx,
            )
            for spec in self.specs
        }

    def _validate_split_roles(self) -> None:
        training = set(self._training_years)
        reanalysis_eval = set(self._reanalysis_evaluation_years)
        station_eval = set(self._station_evaluation_years)

        overlaps = {
            "training_vs_reanalysis_evaluation": sorted(
                training & reanalysis_eval
            ),
            "training_vs_station_evaluation": sorted(
                training & station_eval
            ),
            "reanalysis_vs_station_evaluation": sorted(
                reanalysis_eval & station_eval
            ),
        }
        bad = {
            name: years
            for name, years in overlaps.items()
            if years
        }
        if bad:
            raise ValueError(
                f"Weather split roles overlap: {bad}"
            )

        if self._station_overlap_years:
            unknown = (
                set(self._station_overlap_years)
                - station_eval
            )
            if unknown:
                raise ValueError(
                    "station_overlap_years must be a subset of "
                    "station_evaluation_years; "
                    f"unexpected={sorted(unknown)}"
                )

    def _validate_follower_pools(self) -> None:
        training = set(self._training_years)
        follower_union: set[str] = set()
        group_sets: list[set[str]] = []

        for spec in self.specs[1:]:
            group = set(spec.years)
            if not group:
                raise ValueError(
                    f"Follower pool {spec.group} is empty"
                )
            outside = group - training
            if outside:
                raise ValueError(
                    f"Follower pool {spec.group} contains non-training "
                    f"years: {sorted(outside)}"
                )
            group_sets.append(group)
            follower_union |= group

        for index, left in enumerate(group_sets):
            for right in group_sets[index + 1:]:
                overlap = left & right
                if overlap:
                    raise ValueError(
                        "Follower ET0 pools overlap: "
                        f"{sorted(overlap)}"
                    )

        if follower_union != training:
            raise ValueError(
                "Follower ET0 pools must partition all training years; "
                f"missing={sorted(training - follower_union)}, "
                f"extra={sorted(follower_union - training)}"
            )

    @property
    def num_policies(self) -> int:
        return len(self.specs)

    @property
    def policy_labels(self) -> list[str]:
        return [spec.label for spec in self.specs]

    def training_years(self) -> tuple[str, ...]:
        return self._training_years

    def reanalysis_evaluation_years(self) -> tuple[str, ...]:
        return self._reanalysis_evaluation_years

    def station_evaluation_years(self) -> tuple[str, ...]:
        return self._station_evaluation_years

    def station_overlap_years(self) -> tuple[str, ...]:
        return self._station_overlap_years

    def years_for_policy(self, policy_idx: int) -> tuple[str, ...]:
        return self.specs[policy_idx].years

    def describe_weather_pools(self) -> dict[str, tuple[str, ...]]:
        return {
            spec.group: spec.years
            for spec in self.specs
        }

    def sample_episode(self, episode: int) -> list[dict[str, Any]]:
        assignments = []
        for spec in self.specs:
            assignments.append(
                {
                    "episode": int(episode) + 1,
                    "policy_idx": spec.policy_idx,
                    "policy_label": spec.label,
                    "weather_group": spec.group,
                    "year": self._cycles[spec.policy_idx].next(),
                }
            )
        return assignments

    def sample_rollout_batch(
        self,
        start_episode: int,
        episode_count: int,
    ) -> list[list[dict[str, Any]]]:
        if episode_count <= 0:
            raise ValueError("episode_count must be positive")
        return [
            self.sample_episode(start_episode + offset)
            for offset in range(episode_count)
        ]

    def workspace_names(self) -> dict[int, str]:
        return {
            spec.policy_idx: f"ppo_agent_{spec.policy_idx}"
            for spec in self.specs
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "source_model": self.source_model,
            "splits": {
                "training_weather_years": list(
                    self._training_years
                ),
                "reanalysis_evaluation_years": list(
                    self._reanalysis_evaluation_years
                ),
                "station_evaluation_years": list(
                    self._station_evaluation_years
                ),
                "station_overlap_years": list(
                    self._station_overlap_years
                ),
            },
            "cycles": {
                str(index): cycle.state_dict()
                for index, cycle in self._cycles.items()
            },
        }
