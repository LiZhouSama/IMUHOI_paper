"""Generate the two comparison tables from unified online-evaluation reports.

The evaluator writes one JSON file per method/dataset pair.  This script keeps
the table schema in one place and fails loudly when a required report or metric
is missing, so a partially completed benchmark cannot be mistaken for a full
result table.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DATASETS = (
    ("omomo", "OMOMO"),
    ("imhd", r"$IMHD^{2}$"),
    ("behave", "BEHAVE"),
    ("hodome", "HODOME"),
    ("pahoi", "PAHOI"),
)
METHODS = (
    ("dip18", "DIP*"),
    ("tip", "TIP*"),
    ("transpose", "TransPose*"),
    ("globalpose", "GlobalPose*"),
    ("ours", r"\textbf{Ours (Fusion)}"),
)
OBJECT_METRICS = (
    ("obj_err_cm", "Obj Err"),
    ("hoi_err_cm", "HOI Err"),
    ("obj_trans_jitter_mm_frame2", "Jitter"),
)
HUMAN_METRICS = (
    ("human_ang_err_deg", "Ang Err"),
    ("human_pos_err_cm", "Pos Err"),
    ("human_jitter_mm_frame2", "Jitter"),
)
REQUIRED_METRICS = tuple(
    dict.fromkeys(key for key, _ in OBJECT_METRICS + HUMAN_METRICS)
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build object/human comparison LaTeX tables from eval JSON files"
    )
    parser.add_argument(
        "--report_root",
        required=True,
        help="Directory containing <method>_<dataset>.json reports",
    )
    parser.add_argument("--output", required=True, help="Output .tex path")
    parser.add_argument(
        "--summary_json",
        default=None,
        help="Optional path for the normalized numeric summary",
    )
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Use -- for missing reports/metrics (not recommended for final tables)",
    )
    args = parser.parse_args()
    if args.decimals < 0:
        raise ValueError("--decimals must be non-negative")
    return args


def _report_candidates(root: Path, method: str, dataset: str) -> list[Path]:
    # The first two forms are the canonical layout.  The remaining forms make
    # the utility usable with eval_comparison's default output name as well.
    return [
        root / f"{method}_{dataset}.json",
        root / method / f"{dataset}.json",
        root / method / f"eval_{dataset}_test_corrected_v2.json",
        root / f"eval_{dataset}_{method}_test_corrected_v2.json",
    ]


def _find_report(root: Path, method: str, dataset: str) -> Path | None:
    for candidate in _report_candidates(root, method, dataset):
        if candidate.is_file():
            return candidate
    # Do not rely on directory ordering when users collected reports in a
    # method-specific subdirectory.
    matches = sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file()
        and method in path.stem.lower()
        and dataset in path.stem.lower()
    )
    return matches[0] if len(matches) == 1 else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_values(
    root: Path,
    allow_missing: bool,
) -> tuple[dict[str, dict[str, dict[str, float | None]]], dict[str, str]]:
    values: dict[str, dict[str, dict[str, float | None]]] = {}
    report_paths: dict[str, str] = {}
    errors: list[str] = []
    for method, _ in METHODS:
        values[method] = {}
        if method == "ours":
            for dataset, _ in DATASETS:
                values[method][dataset] = {key: None for key in REQUIRED_METRICS}
            continue
        for dataset, _ in DATASETS:
            report_path = _find_report(root, method, dataset)
            identifier = f"{method}/{dataset}"
            if report_path is None:
                errors.append(f"missing report: {identifier}")
                values[method][dataset] = {key: None for key in REQUIRED_METRICS}
                continue
            try:
                with report_path.open() as handle:
                    report = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"cannot read {report_path}: {exc}")
                values[method][dataset] = {key: None for key in REQUIRED_METRICS}
                continue
            metrics = report.get("metrics") if isinstance(report, dict) else None
            if not isinstance(metrics, dict):
                errors.append(f"{report_path}: missing metrics object")
                metrics = {}
            row: dict[str, float | None] = {}
            for key in REQUIRED_METRICS:
                number = _finite_number(metrics.get(key))
                if number is None:
                    errors.append(f"{report_path}: missing/non-finite metric {key}")
                row[key] = number
            values[method][dataset] = row
            report_paths[identifier] = str(report_path.resolve())
    if errors and not allow_missing:
        raise RuntimeError("\n".join(errors))
    return values, report_paths


def _format_value(value: float | None, decimals: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def _cell(
    values: dict[str, dict[str, dict[str, float | None]]],
    metric: str,
    dataset: str,
    method: str,
    decimals: int,
) -> str:
    value = values[method][dataset].get(metric)
    text = _format_value(value, decimals)
    if value is None or method == "ours":
        return text
    comparison = [
        values[name][dataset].get(metric)
        for name, _ in METHODS
        if name != "ours"
    ]
    finite = [item for item in comparison if item is not None]
    if finite and math.isclose(value, min(finite), rel_tol=1e-12, abs_tol=1e-12):
        return rf"\textbf{{{text}}}"
    return text


def _header_lines() -> list[str]:
    return [
        r"\newcommand{\cw}{0.88cm}",
        r"\newcommand{\mw}{1.5cm}",
        r"\newcolumntype{C}{>{\centering\arraybackslash}m{\cw}}",
        r"\newcolumntype{L}{>{\raggedright\arraybackslash}m{\mw}}",
        "",
    ]


def _table(
    caption: str,
    label: str,
    metric_defs: tuple[tuple[str, str], ...],
    values: dict[str, dict[str, dict[str, float | None]]],
    decimals: int,
) -> list[str]:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\vspace{-3mm}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{L*{15}{C}}",
        r"\toprule",
        "& "
        + " & ".join(rf"\multicolumn{{3}}{{c}}{{{label}}}" for _, label in DATASETS)
        + " \\\\",
        r"\cmidrule(lr){2-4}",
        r"\cmidrule(lr){5-7}",
        r"\cmidrule(lr){8-10}",
        r"\cmidrule(lr){11-13}",
        r"\cmidrule(lr){14-16}",
        r"\textbf{Methods} & "
        + " & ".join(
            title for _dataset, _dataset_label in DATASETS for _metric, title in metric_defs
        )
        + " \\\\",
        r"\midrule",
    ]
    # The table has dataset-major columns, so transpose the metric/method loop
    # when rendering each row.
    for method, method_label in METHODS:
        cells = [
            _cell(values, metric, dataset, method, decimals)
            for dataset, _ in DATASETS
            for metric, _ in metric_defs
        ]
        lines.append(method_label + " & " + " & ".join(cells) + " \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{-0.6em}",
            r"\end{table*}",
        ]
    )
    return lines


def main() -> None:
    args = _parse_args()
    root = Path(args.report_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    values, report_paths = _load_values(root, args.allow_missing)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated from Comparisons.eval_comparison reports.",
        "% Ours (Fusion) is intentionally left as -- because it was not run.",
        *_header_lines(),
        *_table(
            "Comparison of object-centric performance on five HOI datasets. Numbers are lower-is-better. Best in \\textbf{bold}.",
            "tab:sota_object_metrics",
            OBJECT_METRICS,
            values,
            args.decimals,
        ),
        "",
        *_table(
            "Comparison of human motion estimation performance on five HOI datasets. Numbers are lower-is-better. Best in \\textbf{bold}.",
            "tab:sota_human_metrics",
            HUMAN_METRICS,
            values,
            args.decimals,
        ),
    ]
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "reports": report_paths,
                    "datasets": [name for name, _ in DATASETS],
                    "methods": [name for name, _ in METHODS],
                    "metrics": values,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="ascii",
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
