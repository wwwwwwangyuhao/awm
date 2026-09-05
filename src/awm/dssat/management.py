"""Site-neutral DSSAT experiment rendering for externally validated COX templates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IRRIGATION_MARKER = "{{AWM_IRRIGATION_EVENTS}}"


@dataclass(frozen=True, slots=True)
class IrrigationEvent:
    yrdoy: str
    amount_mm: float


class DSSATExperimentRenderer:
    """Inject policy irrigation into a validated external DSSAT COX template.

    Everything except the irrigation-event marker is preserved byte-for-byte.
    Cultivar, soil initial conditions, planting, mulch, fixed nitrogen and other
    management remain owned by the external experiment template.
    """

    def __init__(
        self,
        *,
        template_path: str,
        output_cox_path: str,
        marker: str = IRRIGATION_MARKER,
        irrigation_operation: str = "IR005",
    ) -> None:
        self.template_path = Path(template_path)
        self.output_cox_path = Path(output_cox_path)
        self.marker = str(marker)
        self.irrigation_operation = str(irrigation_operation)
        if not self.template_path.is_file():
            raise FileNotFoundError(self.template_path)
        text = self.template_path.read_text(encoding="utf-8")
        if text.count(self.marker) != 1:
            raise ValueError(
                f"COX template must contain marker exactly once: {self.marker}"
            )
        self._template = text
        self._events: list[IrrigationEvent] = []

    @property
    def events(self) -> tuple[IrrigationEvent, ...]:
        return tuple(self._events)

    @property
    def total_policy_irrigation_mm(self) -> float:
        return float(sum(event.amount_mm for event in self._events))

    def reset(self) -> None:
        self._events.clear()
        self.render()

    def add_irrigation(self, yrdoy: str, amount_mm: float) -> None:
        yrdoy = str(yrdoy).strip()
        amount = float(amount_mm)
        if len(yrdoy) != 5 or not yrdoy.isdigit():
            raise ValueError("irrigation date must be DSSAT YYDDD")
        if amount <= 0.0:
            raise ValueError("irrigation amount must be > 0")
        if any(event.yrdoy == yrdoy for event in self._events):
            raise ValueError(f"duplicate irrigation date: {yrdoy}")
        self._events.append(IrrigationEvent(yrdoy=yrdoy, amount_mm=amount))
        self.render()

    def render(self) -> None:
        rows = "\n".join(
            f" 1 {event.yrdoy} {self.irrigation_operation} {event.amount_mm:.2f}"
            for event in self._events
        )
        rendered = self._template.replace(self.marker, rows)
        self.output_cox_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_cox_path.write_text(rendered, encoding="utf-8", newline="\n")


__all__ = ["DSSATExperimentRenderer", "IRRIGATION_MARKER", "IrrigationEvent"]
