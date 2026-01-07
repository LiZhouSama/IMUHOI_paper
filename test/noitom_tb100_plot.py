"""
Utility script to parse ``noitom/OBJ/TB100_1.csv`` (acc in g, az includes -g)
and visualize acceleration, velocity, and displacement over time.
Run it directly:
    python test/noitom_tb100_plot.py
"""

from pathlib import Path
import csv
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


_EXPECTED_COLUMNS = [
    "time_s",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "mx",
    "my",
    "mz",
    "roll",
    "pitch",
    "yaw",
    "q1",
    "q2",
    "q3",
    "q4",
]

GRAVITY = 9.80665  # m/s^2 per g


def _candidate_paths() -> List[Path]:
    """Yield likely CSV locations, handling the datasets symlink on Windows."""
    test_file = Path(__file__).resolve()
    candidates: List[Path] = []
    for parent in test_file.parents:
        for root in ("noitom", "datasets/noitom"):
            candidate = parent / root / "OBJ" / "TB100_1.csv"
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _load_csv() -> Tuple[Path, Dict[str, np.ndarray]]:
    """Locate and parse the CSV, returning the resolved path and column arrays."""
    errors: List[str] = []
    for csv_path in _candidate_paths():
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError("CSV header missing.")
                missing = set(_EXPECTED_COLUMNS) - set(reader.fieldnames)
                if missing:
                    raise ValueError(f"Missing columns: {sorted(missing)}")

                values: Dict[str, List[float]] = {k: [] for k in _EXPECTED_COLUMNS}
                for row in reader:
                    try:
                        for key in _EXPECTED_COLUMNS:
                            values[key].append(float(row[key]))
                    except (TypeError, ValueError, KeyError):
                        # Skip malformed rows and keep parsing the rest.
                        continue

                if not values["time_s"]:
                    raise ValueError("No valid data rows parsed from CSV.")

                arrays = {k: np.asarray(v, dtype=np.float64) for k, v in values.items()}
                return csv_path, arrays
        except (FileNotFoundError, OSError, ValueError) as exc:
            errors.append(f"{csv_path}: {exc}")
            continue

    tried = "\n".join(f" - {err}" for err in errors)
    raise FileNotFoundError(f"Cannot locate TB100_1.csv. Tried:\n{tried}")


def _integrate_trapezoidal(series: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """
    Integrate a [N, 3] series with respect to time using the trapezoidal rule.
    time_s is expected to start at zero; call with shifted time values.
    """
    if series.shape[0] == 0:
        return np.zeros_like(series)
    result = np.zeros_like(series)
    if series.shape[0] == 1:
        return result

    dt = np.diff(time_s)
    if np.any(dt <= 0):
        raise ValueError("time_s must be strictly increasing after offsetting.")

    increments = 0.5 * (series[1:] + series[:-1]) * dt[:, None]
    result[1:] = np.cumsum(increments, axis=0)
    return result


def _plot_motion(
    time_s: np.ndarray,
    acc: np.ndarray,
    vel: np.ndarray,
    disp: np.ndarray,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axis_labels = ("x", "y", "z")

    for idx, label in enumerate(axis_labels):
        axes[0].plot(time_s, acc[:, idx], label=label)
        axes[1].plot(time_s, vel[:, idx], label=label)
        axes[2].plot(time_s, disp[:, idx], label=label)

    axes[0].set_ylabel("Acceleration (m/s^2)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[2].set_ylabel("Displacement (m)")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    fig.suptitle("TB100_1 Noitom OBJ: Acceleration → Velocity → Displacement")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    return output_path


def main() -> Path:
    csv_path, data = _load_csv()
    time_s = data["time_s"].astype(np.float64)
    time_s = time_s - time_s[0]  # Align to zero because the CSV time does not start at 0.

    # Convert from g to m/s^2 and remove the -g gravity component on az.
    acc_g = np.column_stack((data["ax"], data["ay"], data["az"]))
    acc_g[:, 2] += 1.0  # az has -g; offset so stationary becomes ~0
    acc = acc_g * GRAVITY

    vel = _integrate_trapezoidal(acc, time_s)
    disp = _integrate_trapezoidal(vel, time_s)

    output_path = Path(__file__).with_name("TB100_motion.png")
    saved = _plot_motion(time_s, acc, vel, disp, output_path)

    print(f"Loaded: {csv_path}")
    print(f"Figure saved to: {saved}")
    return saved


if __name__ == "__main__":
    main()
