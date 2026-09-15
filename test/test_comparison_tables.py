import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Comparisons.generate_comparison_tables import main


def test_table_generator_formats_all_cells_and_bolds_minima(tmp_path, monkeypatch):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    datasets = ("omomo", "imhd", "behave", "hodome", "pahoi")
    methods = ("dip18", "tip", "transpose", "globalpose")
    metric_keys = (
        "obj_err_cm",
        "hoi_err_cm",
        "obj_trans_jitter_mm_frame2",
        "human_ang_err_deg",
        "human_pos_err_cm",
        "human_jitter_mm_frame2",
    )
    for method_index, method in enumerate(methods):
        for dataset_index, dataset in enumerate(datasets):
            values = {
                key: float(method_index + dataset_index + offset + 1)
                for offset, key in enumerate(metric_keys)
            }
            (report_root / f"{method}_{dataset}.json").write_text(
                json.dumps({"metrics": values})
            )

    output = tmp_path / "tables.tex"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_comparison_tables",
            "--report_root",
            str(report_root),
            "--output",
            str(output),
            "--summary_json",
            str(summary),
        ],
    )
    main()
    text = output.read_text()
    assert text.count(r"\begin{table*}") == 2
    assert "\\textbf{1.00}" in text
    assert "\\textbf{Ours (Fusion)}" in text
    # 30 table cells plus the explanatory comment.
    assert text.count("--") == 31
    assert json.loads(summary.read_text())["metrics"]["ours"]["omomo"]["obj_err_cm"] is None
