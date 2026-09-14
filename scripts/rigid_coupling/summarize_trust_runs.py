#!/usr/bin/env python3
"""Tabulate the 2026-09-10/11 state-trust experiments (OMOMO official + BEHAVE full sequences)."""
import json
import os

B = "outputs/IMUHOI++/RNN_cano/behave_longseq_20260909"
R = "outputs/IMUHOI++/RNN_cano/omomo_cano_ot_trust_20260910"
ROWS = [
    ("cur_hoi (old protocol 200x1)", "outputs/IMUHOI++/RNN_cano/omomo_cano_ot_variants_20260909_c/cur_hoi/eval_batch_online.json", f"{B}/cur_hoi_behave_eval.json", f"{B}/cur_hoi_final_behave_eval.json"),
    ("cur_hoi_p800 (baseline, 800x4+ES)", f"{R}_c/cur_hoi_p800/eval_batch_online.json", f"{B}/cur_hoi_p800_c_behave_eval.json", f"{B}/p800_c_final_behave_eval.json"),
    ("trust_noaug", f"{R}_e/cur_hoi_trust_noaug/eval_batch_online.json", f"{B}/cur_hoi_trust_noaug_behave_eval.json", f"{B}/trust_noaug_final_behave_eval.json"),
    ("trust_noaug + static gate", f"{B}/trust_noaug_gate_omomo_eval.json", f"{B}/trust_noaug_gate_behave_eval.json", None),
    ("trust_mild (init 0.10 m, 25%)", f"{R}_h/cur_hoi_trust_mild/eval_batch_online.json", f"{B}/cur_hoi_trust_mild_behave_eval.json", f"{B}/trust_mild_final_behave_eval.json"),
    ("trust_mild final (OMOMO) / final + gate", f"{B}/trust_mild_final_omomo_eval.json", f"{B}/trust_mild_final_gate_behave_eval.json", None),
    ("trust_mild final + gate (OMOMO)", f"{B}/trust_mild_final_gate_omomo_eval.json", None, None),
    ("trust (init 0.20 m, 50%) [a]", f"{R}/cur_hoi_trust/eval_batch_online.json", f"{B}/cur_hoi_trust_behave_eval.json", f"{B}/trust_a_final_behave_eval.json"),
    ("trust (init 0.20 m, 50%) [b]", f"{R}_b/cur_hoi_trust/eval_batch_online.json", f"{B}/cur_hoi_trust_b_behave_eval.json", f"{B}/trust_b_final_behave_eval.json"),
    ("trust_perturb (+rest 0.20 m)", f"{R}_d/cur_hoi_trust_perturb/eval_batch_online.json", f"{B}/cur_hoi_trust_perturb_behave_eval.json", f"{B}/trust_perturb_final_behave_eval.json"),
    ("trust_perturb + static gate", f"{B}/trust_perturb_gate_omomo_eval.json", f"{B}/trust_perturb_gate_behave_eval.json", None),
    ("trust_perturb40 (+rest 0.40 m)", f"{R}_f/cur_hoi_trust_perturb40/eval_batch_online.json", f"{B}/cur_hoi_trust_perturb40_behave_eval.json", f"{B}/trust_perturb40_final_behave_eval.json"),
]


def load(path, ds):
    if path and os.path.exists(path):
        j = json.load(open(path))
        return j["results"].get(ds) or next(iter(j["results"].values()))
    return None


def fmt(r, keys):
    return " / ".join(f"{r[k]:.2f}" for k in keys) if r else "—"


def main():
    keys = ("obj_trans_err_fusion", "hoi_err_fusion", "obj_jitter")
    print(f"| variant | OMOMO obj / HOI / jitter | BEHAVE best obj / HOI / jitter | BEHAVE final HOI |")
    print("|---|---|---|---|")
    for name, fo, fb, ff in ROWS:
        jo, jb, jf = load(fo, "omomo"), load(fb, "behave"), load(ff, "behave")
        print(f"| {name} | {fmt(jo, keys)} | {fmt(jb, keys)} | {jf['hoi_err_fusion']:.2f} |" if jf else f"| {name} | {fmt(jo, keys)} | {fmt(jb, keys)} | — |")


if __name__ == "__main__":
    main()
