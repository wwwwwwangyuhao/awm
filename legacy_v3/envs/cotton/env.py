"""Canonical cotton DSSAT environment for ss_refactor.

Decision semantics
------------------
The observation at DAP d selects management applied on DAP d+1. There are
exactly 125 policy decisions: s_0 -> a_1 -> ... -> s_125.

DSSAT still simulates the complete biological season on every execution. The
policy sees daily OUT state only; mature Summary.OUT values are kept separate
and become available to reward/evaluation only on the terminal transition.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from ...configs.reward import REWARD_CONFIG
from ...dssat.dates import doy_to_date, yy_doy
from ...dssat.file_handler import DSSATFileHandler
from ...dssat.output_reader import CachedDSSATOutputReader
from ...dssat.runner import DSSATRunner, log_message
from .reward import CottonReward


class DSSATEnv(gymnasium.Env):
    """Continuous irrigation/nitrogen control over a 125-day decision horizon."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Dict[str, Any], action_logger=None):
        super().__init__()
        self.logger = action_logger
        self.config = dict(config)

        self.exp_year = str(config["exp-year"]).zfill(2)
        self.calendar_year = int(config["calendar-year"])
        self.data_dir = config.get("data-dir")
        self.irrigation_file = config.get("irrigation-file")
        self.fertilizer_file = config.get("fertilizer-file")
        self.weather_file = config.get("weather-file")
        self.soil_file = config.get("soil-file")
        self.output_dir = config.get("output-dir")
        self.cox_file = config.get("cox-file")
        self.summary_out = config.get("summary-out")
        self.out_list = config.get("out-list")

        self.plant_date = config.get("plant-date")
        self.emergence_date = config.get("emergence-date")
        self.total_days = int(
            config.get("decision-horizon", config.get("total-days", 125))
        )
        if self.total_days <= 0:
            raise ValueError("decision horizon must be a positive integer.")

        self.is_phosphorus = config.get("is-phosphorus")
        self.is_potassium = config.get("is-potassium")
        self.field_name = config.get("field-name")
        self.weather_name = config.get("weather-name")
        self.cox_name = config.get("cox-name")
        self.str_fields = config.get("str-fields")
        self.date_fields = config.get("date-fields")
        self.dssat_exec = config.get("dssat-exec")
        self.delete_dir = config.get("delete-dir", [])
        self.verbose = config.get("verbose", 0)
        self.action_type = str(config.get("action-type", "continuous")).lower()
        if self.action_type != "continuous":
            raise ValueError(
                "Current SAPG environment requires a 2-D continuous action; "
                f"received action-type={self.action_type!r}."
            )

        self.max_irrigation = float(config.get("max-irri", 50.0))
        self.max_nitrogen = float(config.get("max-fert", 30.0))
        if self.max_irrigation <= 0.0 or self.max_nitrogen <= 0.0:
            raise ValueError("max-irri and max-fert must be positive.")

        self.action_consistency_audit = self._as_bool(
            config.get("action-consistency-audit", False)
        )
        self.action_application_threshold = float(
            config.get("action-application-threshold", 1.0)
        )
        if self.action_application_threshold < 0.0:
            raise ValueError("action-application-threshold cannot be negative.")

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(0,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array(
                [self.max_irrigation, self.max_nitrogen], dtype=np.float32
            ),
            dtype=np.float32,
        )

        self.handler = DSSATFileHandler(
            output_path=self.cox_file,
            plant_date=self.plant_date,
            emergence_date=self.emergence_date,
            is_phosphorus=self.is_phosphorus,
            is_potassium=self.is_potassium,
            field_name=self.field_name,
            weather_name=self.weather_name,
            COX_name=self.cox_name,
            irrigation_file=self.irrigation_file,
            fertilizer_file=self.fertilizer_file,
            exp_year=self.exp_year,
            action_type=self.action_type,
            verbose=self.verbose,
        )
        self.runner = DSSATRunner(
            output_dir=self.output_dir,
            dssat_exec=self.dssat_exec,
            cox_path=self.cox_file,
            weather_file=self.weather_file,
            soil_file=self.soil_file,
            verbose=self.verbose,
        )
        self.output_reader = CachedDSSATOutputReader(
            summary_out=self.summary_out,
            out_files=list(self.out_list or []),
            str_fields=self.str_fields,
            date_fields=self.date_fields,
            verbose=self.verbose,
        )
        self.reward_model = CottonReward(
            config.get("reward-config", REWARD_CONFIG)
        )

        self.current_day = 0
        self.state_dict: Dict[str, Any] = {}
        self.initial_state: Dict[str, Any] | None = None
        self.done = False
        self.truncated = False
        self.episode = 0
        self.harvest = 0.0

        # Diagnostic accounting only; these cumulative values never enter the
        # actor/critic observation or future reward function.
        self.episode_return = 0.0
        self.total_irrigation_applied = 0.0
        self.total_nitrogen_applied = 0.0
        self.management_event_count = 0

        self.irri_run_days: list[int] = []
        self.irri_sim_days: list[int] = []
        self.irri_dates: list[str] = []
        self.irrigation_amounts: list[float] = []
        self.fert_run_days: list[int] = []
        self.fert_sim_days: list[int] = []
        self.fert_dates: list[str] = []
        self.fertilization_amounts: list[float] = []

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", ""}:
                return False
        raise ValueError(f"Cannot parse boolean configuration value: {value!r}")

    def _validate_policy_action(self, action: Any) -> tuple[float, float]:
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.shape != (2,):
            raise ValueError(
                "DSSATEnv expects [irrigation, nitrogen], "
                f"got shape={values.shape}."
            )
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Action contains NaN/Inf: {values}.")
        low = self.action_space.low.astype(np.float64)
        high = self.action_space.high.astype(np.float64)
        if np.any(values < low) or np.any(values > high):
            raise ValueError(
                "Action is outside declared environment bounds; silent clipping "
                f"is forbidden. action={values.tolist()}, low={low.tolist()}, "
                f"high={high.tolist()}."
            )
        return float(values[0]), float(values[1])

    def _resolve_applied_action(
        self, policy_action: tuple[float, float]
    ) -> tuple[float, float]:
        if self.action_consistency_audit:
            return policy_action
        threshold = self.action_application_threshold
        return (
            policy_action[0] if policy_action[0] >= threshold else 0.0,
            policy_action[1] if policy_action[1] >= threshold else 0.0,
        )

    def _run_dssat_and_refresh(self) -> None:
        self.handler.write_COX()
        run_status, run_stdout, run_stderr = self.runner.run()
        if run_status != 0:
            reason = run_stderr or run_stdout
            raise RuntimeError(
                f"DSSAT run failed: status={run_status}, reason={reason}"
            )
        self.output_reader.invalidate()
        self.output_reader.refresh()

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(int(seed))

        self.clean()
        self._run_dssat_and_refresh()

        self.current_day = 0
        target_yrdoy = self._get_current_yrdoy()
        self.initial_state = self.output_reader.daily_state(target_yrdoy)
        self._validate_state_dict(self.initial_state, target_yrdoy)
        self.state_dict = dict(self.initial_state)

        self.done = False
        self.truncated = False
        self.harvest = 0.0
        self.episode_return = 0.0
        self.total_irrigation_applied = 0.0
        self.total_nitrogen_applied = 0.0
        self.management_event_count = 0

        self.irri_run_days = []
        self.irri_sim_days = []
        self.irri_dates = []
        self.irrigation_amounts = []
        self.fert_run_days = []
        self.fert_sim_days = []
        self.fert_dates = []
        self.fertilization_amounts = []
        self.episode += 1

        cache = self.output_reader.metrics
        return self.state_dict, {
            "current_day": 0,
            "decision_horizon": self.total_days,
            "action_dict": {},  # temporary compatibility field; never observed
            "action_consistency_audit": self.action_consistency_audit,
            "action_threshold_enabled": not self.action_consistency_audit,
            "action_application_threshold": (
                0.0
                if self.action_consistency_audit
                else self.action_application_threshold
            ),
            "output_refresh_count": cache["refresh_count"],
            "output_disk_read_count": cache["disk_read_count"],
            "output_cache_hit_count": cache["cache_hit_count"],
            "output_last_refresh_seconds": cache["last_refresh_seconds"],
        }

    def _get_current_yrdoy(self) -> str:
        plant_date = str(self.plant_date).strip()
        if len(plant_date) != 5 or not plant_date.isdigit():
            raise ValueError(
                f"plant-date must be DSSAT YYDDD, got {self.plant_date!r}."
            )
        planting_doy = int(plant_date[-3:])
        return yy_doy(self.calendar_year, planting_doy + self.current_day)

    def _get_action_yrdoy(self) -> str:
        plant_date = str(self.plant_date).strip()
        if len(plant_date) != 5 or not plant_date.isdigit():
            raise ValueError(
                f"plant-date must be DSSAT YYDDD, got {self.plant_date!r}."
            )
        planting_doy = int(plant_date[-3:])
        return yy_doy(
            self.calendar_year,
            planting_doy + self.current_day + 1,
        )

    def _update_state(self, rerun_dssat: bool) -> None:
        if rerun_dssat:
            self._run_dssat_and_refresh()
        target_yrdoy = self._get_current_yrdoy()
        new_state = self.output_reader.daily_state(target_yrdoy)
        self._validate_state_dict(new_state, target_yrdoy)
        self.state_dict = new_state

    def step(
        self,
        action: Tuple[float, float],
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.done or self.truncated:
            raise RuntimeError("episode has ended; call reset() first.")

        previous_state = dict(self.state_dict)
        policy_action = self._validate_policy_action(action)
        applied_action = self._resolve_applied_action(policy_action)
        irrigation_applied, nitrogen_applied = applied_action
        difference = (
            abs(policy_action[0] - irrigation_applied),
            abs(policy_action[1] - nitrogen_applied),
        )
        management_changed = (
            irrigation_applied > 0.0 or nitrogen_applied > 0.0
        )

        action_day = self.current_day + 1
        action_yrdoy = self._get_action_yrdoy()
        action_doy = int(action_yrdoy[-3:])

        if irrigation_applied > 0.0:
            self.handler.write_irrigation(action_yrdoy, irrigation_applied)
            self.irri_run_days.append(action_day)
            self.irri_sim_days.append(action_doy)
            self.irri_dates.append(doy_to_date(action_doy, self.calendar_year))
            self.irrigation_amounts.append(float(irrigation_applied))

        if nitrogen_applied > 0.0:
            self.handler.write_fertilizer(action_yrdoy, nitrogen_applied)
            self.fert_run_days.append(action_day)
            self.fert_sim_days.append(action_doy)
            self.fert_dates.append(doy_to_date(action_doy, self.calendar_year))
            self.fertilization_amounts.append(float(nitrogen_applied))

        self.current_day += 1
        self._update_state(rerun_dssat=management_changed)
        self.done = self.current_day >= self.total_days

        season_summary = self.output_reader.season_summary() if self.done else None
        breakdown = self.reward_model.compute(
            previous_state=previous_state,
            next_state=self.state_dict,
            applied_action=applied_action,
            terminated=self.done,
            season_summary=season_summary,
        )
        reward = float(breakdown.total)

        self.total_irrigation_applied += float(irrigation_applied)
        self.total_nitrogen_applied += float(nitrogen_applied)
        self.management_event_count += int(breakdown.management_event)
        self.episode_return += reward

        cache = self.output_reader.metrics
        info: Dict[str, Any] = {
            "harvest": self.harvest,
            "action_dict": {},  # compatibility only; no history features remain
            "current_day": self.current_day,
            "decision_horizon": self.total_days,
            "decision_action_day": action_day,
            "decision_action_yrdoy": action_yrdoy,
            "decision_action_doy": action_doy,
            "decision_action_date": doy_to_date(
                action_doy, self.calendar_year
            ),
            "DOY": self.state_dict.get("DOY"),
            "management_changed": bool(management_changed),
            "dssat_rerun": bool(management_changed),
            "commanded_action": policy_action,
            "applied_action": applied_action,
            "commanded_applied_difference": difference,
            "action_transformed": any(value > 0.0 for value in difference),
            "irrigation_applied": float(irrigation_applied),
            "nitrogen_applied": float(nitrogen_applied),
            "action_consistency_audit": self.action_consistency_audit,
            "action_threshold_enabled": not self.action_consistency_audit,
            "action_application_threshold": (
                0.0
                if self.action_consistency_audit
                else self.action_application_threshold
            ),
            "output_refresh_count": cache["refresh_count"],
            "output_disk_read_count": cache["disk_read_count"],
            "output_cache_hit_count": cache["cache_hit_count"],
            "output_last_refresh_seconds": cache["last_refresh_seconds"],
            **breakdown.as_dict(),
            "final_reward": (
                float(breakdown.yield_reward) if self.done else 0.0
            ),
        }

        if self.action_consistency_audit:
            info["executed_action"] = applied_action

        if self.done:
            assert season_summary is not None
            required = ("HWAM", "IRCM", "NICM")
            missing = [key for key in required if key not in season_summary]
            if missing:
                raise KeyError(
                    "Terminal Summary.OUT is missing fields: "
                    + ", ".join(missing)
                )
            hwam = float(season_summary["HWAM"])
            ircm = float(season_summary["IRCM"])
            nicm = float(season_summary["NICM"])
            self.harvest = hwam

            direct_objective = self.reward_model.seasonal_objective(
                final_hwam=hwam,
                total_irrigation=self.total_irrigation_applied,
                total_nitrogen=self.total_nitrogen_applied,
                management_events=self.management_event_count,
            )
            telescoping_error = self.episode_return - direct_objective
            if not np.isclose(
                self.episode_return,
                direct_objective,
                rtol=0.0,
                atol=1e-5,
            ):
                raise RuntimeError(
                    "Reward telescoping audit failed: "
                    f"episode_return={self.episode_return}, "
                    f"direct_objective={direct_objective}, "
                    f"error={telescoping_error}."
                )

            info.update(
                {
                    "harvest": hwam,
                    "HWAM": hwam,
                    "IRCM": ircm,
                    "NICM": nicm,
                    "management_event_count": self.management_event_count,
                    "total_irrigation_applied_reward": self.total_irrigation_applied,
                    "total_nitrogen_applied_reward": self.total_nitrogen_applied,
                    "reward_episode_return": self.episode_return,
                    "reward_direct_season_objective": direct_objective,
                    "reward_telescoping_error": telescoping_error,
                }
            )
            if self.logger:
                self._log_action_details()

        return (
            self.state_dict,
            reward,
            bool(self.done),
            bool(self.truncated),
            info,
        )

    @staticmethod
    def _validate_state_dict(state: Any, yrdoy: str) -> None:
        if not isinstance(state, dict):
            raise TypeError(
                "DSSAT output reader must return a dictionary: "
                f"type={type(state).__name__}, yrdoy={yrdoy}."
            )
        if not state:
            raise RuntimeError(
                "DSSAT ran successfully but no daily state was parsed for "
                f"yrdoy={yrdoy}."
            )

    def sample_valid_action(
        self,
        seed: int | None = None,
    ) -> Tuple[float, float]:
        rng = np.random.RandomState(seed) if seed is not None else np.random
        irrigation = rng.uniform(
            self.action_space.low[0], self.action_space.high[0]
        )
        nitrogen = rng.uniform(
            self.action_space.low[1], self.action_space.high[1]
        )
        return float(irrigation), float(nitrogen)

    def _log_action_details(self) -> None:
        self.logger.debug(f"Episode {self.episode} irrigation details:")
        self.logger.debug(f"  events: {len(self.irri_run_days)}")
        self.logger.debug(f"  amount: {sum(self.irrigation_amounts):.4f}")
        self.logger.debug(f"  decision days: {self.irri_run_days}")
        self.logger.debug(f"  DOY: {self.irri_sim_days}")
        self.logger.debug(f"  dates: {self.irri_dates}")
        self.logger.debug(f"Episode {self.episode} nitrogen details:")
        self.logger.debug(f"  events: {len(self.fert_run_days)}")
        self.logger.debug(f"  amount: {sum(self.fertilization_amounts):.4f}")
        self.logger.debug(f"  decision days: {self.fert_run_days}")
        self.logger.debug(f"  DOY: {self.fert_sim_days}")
        self.logger.debug(f"  dates: {self.fert_dates}")
        self.logger.debug(
            f"Episode {self.episode} HWAM={self.harvest:.4f}, "
            f"management_days={self.management_event_count}, "
            f"reward={self.episode_return:.6f}"
        )

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            print(
                f"Day={self.current_day}, irrigation_events={len(self.irri_run_days)}, "
                f"nitrogen_events={len(self.fert_run_days)}"
            )

    def clean(self, *, strict: bool = False) -> dict[str, Any]:
        log_message(self.verbose, "Cleaning DSSAT writable directories...")
        removed: list[str] = []
        failures: list[dict[str, str]] = []
        for dir_name in self.delete_dir:
            dir_path = os.path.join(self.data_dir, dir_name)
            if not os.path.exists(dir_path):
                continue
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                        removed.append(file_path)
                except Exception as exc:
                    failures.append(
                        {
                            "path": file_path,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        report = {
            "removed_file_count": len(removed),
            "failure_count": len(failures),
            "removed_files": removed,
            "failures": failures,
            "strict": bool(strict),
        }
        if strict and failures:
            raise RuntimeError(f"Strict environment cleanup failed: {report}")
        return report

    def close(self) -> None:
        self.episode = 0


__all__ = ["DSSATEnv"]
