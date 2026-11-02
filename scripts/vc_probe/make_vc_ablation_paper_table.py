#!/usr/bin/env python3
"""Paper-format tables of the VC module ablation (experiments/06_vc_module_ablation).

Reads each dataset's metrics.json (written by summarize_vc_ablation.py) and writes paper_table.md and
paper_table.tex: one block per dataset, rows = variants, columns = left / right hand-contact F1, object-motion F1,
hand contact-event F1 (onsets and releases, +-10 frames), object and HOI error.  A dagger marks a difference to
the full model whose 95 % paired sequence-bootstrap interval excludes zero.  Datasets without results are skipped.

    python scripts/vc_probe/make_vc_ablation_paper_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/06_vc_module_ablation"
DATASETS = (("omomo", "OMOMO"), ("behave", "BEHAVE"), ("imhd", "IMHD"), ("pahoi", "PAHOI"), ("hodome", "HODome"))
ROWS = (("full", "IMUHOI++ (full)"), ("no_prior_bundle", "w/o HP prior"), ("direct", "w/o conditional contact"),
        ("no_larm", "w/o left arm"), ("no_rarm", "w/o right arm"), ("no_torso", "w/o torso"), ("no_legs", "w/o legs"))
# key, header, scale, digits, lower is better
COLUMNS = (("f1_lhand", "F1 L", 100, 1, False), ("f1_rhand", "F1 R", 100, 1, False),
           ("f1_obj", "F1 obj", 100, 1, False), ("ev_hand", "Event F1", 100, 1, False),
           ("obj", "Obj (cm)", 1, 2, True), ("hoi", "HOI (cm)", 1, 2, True))


def load(ds):
    path = OUT / ds / "metrics.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return data["rows"], data["deltas_vs_full"]


def cell(rows, deltas, name, key, scale, digits):
    if name not in rows:
        return None, False
    value = f"{rows[name][key] * scale:.{digits}f}"
    significant = False
    if name in deltas:
        lo, hi = deltas[name][key]["ci95"]
        significant = not (lo <= 0.0 <= hi)
    return value, significant


def main():
    md = ["# VC 模块消融：论文格式表", "",
          "F1 和事件 F1 为百分比，误差单位 cm；† 表示与 full 的差值 95% 配对 bootstrap 区间不含 0。",
          "事件 F1：手接触起止事件（左右手合并）在 ±10 帧内匹配的 F1，取起止平均。", ""]
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}", r"\toprule",
           "Variant & " + " & ".join(c[1] for c in COLUMNS) + r" \\", r"\midrule"]
    first = True
    for ds, name_ds in DATASETS:
        loaded = load(ds)
        if loaded is None:
            continue
        rows, deltas = loaded
        md += [f"**{name_ds}**", "", "| Variant | " + " | ".join(c[1] for c in COLUMNS) + " |",
               "|---" * (len(COLUMNS) + 1) + "|"]
        if not first:
            tex.append(r"\midrule")
        first = False
        tex.append(rf"\multicolumn{{{len(COLUMNS) + 1}}}{{l}}{{\textit{{{name_ds}}}}} \\")
        for name, label in ROWS:
            md_cells, tex_cells = [], []
            for key, _, scale, digits, _ in COLUMNS:
                value, significant = cell(rows, deltas, name, key, scale, digits)
                if value is None:
                    md_cells.append("–")
                    tex_cells.append("--")
                    continue
                md_cells.append(value + ("†" if significant else ""))
                tex_cells.append(value + (r"$^\dagger$" if significant else ""))
            md.append(f"| {label} | " + " | ".join(md_cells) + " |")
            tex.append(f"{label} & " + " & ".join(tex_cells) + r" \\")
        md.append("")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Ablation of the velocity-contact module. Every variant retrains its VC, OT and joint stage. "
            r"F1 in \%; event F1: contact onsets and releases matched within $\pm$10 frames. "
            r"$^\dagger$: difference to the full model with a 95\% paired bootstrap interval excluding zero.}",
            r"\label{tab:vc_ablation}", r"\end{table}"]
    (OUT / "paper_table.md").write_text("\n".join(md) + "\n")
    (OUT / "paper_table.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
