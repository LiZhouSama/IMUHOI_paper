# VC × OT crossed ablation (BEHAVE and OMOMO)

The two diagonal baselines use their latest completed, domain-trained runs; the
off-diagonal cells retrain OT and perform the corresponding joint stage, reusing
the same dataset-specific pretrained HP and VC checkpoints. All rows use seed 42,
the canonical splits and full-sequence batch-online test evaluation. Lower is
better for errors/jitter; higher is better for contact F1.

## OMOMO

| VC | OT | Object (cm) | HOI (cm) | MPJPE (cm) | Root (cm) | L F1 | R F1 | Obj jitter (mm/frame²) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4 | v2_armrot_biclose | 13.366 | 10.443 | 2.986 | 6.044 | 0.949 | 0.952 | 5.335 |
| v4 | imuhoi_4heads | 17.942 | 16.274 | 3.012 | 10.800 | 0.946 | 0.947 | 4.536 |
| imuhoi | v2_armrot_biclose | 13.880 | 10.715 | 2.993 | 6.012 | 0.915 | 0.918 | 5.329 |
| imuhoi | imuhoi_4heads | 19.473 | 16.697 | 3.011 | 10.859 | 0.921 | 0.926 | 4.758 |

## BEHAVE

| VC | OT | Object (cm) | HOI (cm) | MPJPE (cm) | Root (cm) | L F1 | R F1 | Obj jitter (mm/frame²) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4 | v2_armrot_biclose | 23.970 | 17.815 | 3.678 | 16.129 | 0.802 | 0.796 | 6.487 |
| v4 | imuhoi_4heads | 39.837 | 24.801 | 3.716 | 28.273 | 0.787 | 0.778 | 8.563 |
| imuhoi | v2_armrot_biclose | 25.726 | 18.816 | 3.700 | 16.157 | 0.794 | 0.798 | 7.807 |
| imuhoi | imuhoi_4heads | 41.033 | 25.686 | 3.721 | 28.194 | 0.788 | 0.787 | 7.740 |

## Interpretation and provenance

This is a *pipeline-level* 2×2 crossing, not a strictly isolated architecture
main-effect experiment: the original joint-training recipe fine-tunes HP for
IMUHOI OT, but VC for v2 OT. BEHAVE uses HP smoothing half-width 7 and
object-velocity initialization diff 3 across all four cells; OMOMO uses 0
for both. The v2 rows include forearm/object rotation, bimanual closure,
and the fixed-fusion FK alpha-beta filter (α=0.35, β=0.05).

| Dataset | VC | OT | Run manifest | Selected OT checkpoint |
|---|---|---|---|---|
| omomo | v4 | v2_armrot_biclose | `outputs/IMUHOI++/RNN_cano/v2_armrot_biclose_20260923_omomo/experiment_manifest.json` | `best_object_trans.pt` |
| omomo | v4 | imuhoi_4heads | `outputs/IMUHOI++/RNN_cano/vc_ot_cross_ablation_omomo_v4_vc_imuhoi_ot_4heads/experiment_manifest.json` | `best_object_trans.pt` |
| omomo | imuhoi | v2_armrot_biclose | `outputs/IMUHOI++/RNN_cano/vc_ot_cross_ablation_omomo_imuhoi_v2_armrot_biclose/experiment_manifest.json` | `best_object_trans.pt` |
| omomo | imuhoi | imuhoi_4heads | `outputs/IMUHOI++/RNN_cano/relabel_20260920_omomo_imuhoi/experiment_manifest.json` | `best_object_trans.pt` |
| behave | v4 | v2_armrot_biclose | `outputs/IMUHOI++/RNN_cano/v2_armrot_biclose_20260923_behave/experiment_manifest.json` | `best_object_trans.pt` |
| behave | v4 | imuhoi_4heads | `outputs/IMUHOI++/RNN_cano/vc_ot_cross_ablation_behave_v4_vc_imuhoi_ot_4heads/experiment_manifest.json` | `best_object_trans.pt` |
| behave | imuhoi | v2_armrot_biclose | `outputs/IMUHOI++/RNN_cano/vc_ot_cross_ablation_behave_imuhoi_v2_armrot_biclose/experiment_manifest.json` | `best_object_trans.pt` |
| behave | imuhoi | imuhoi_4heads | `outputs/IMUHOI++/RNN_cano/relabel_20260922_behave_hpsmooth_imuhoi/experiment_manifest.json` | `best_object_trans.pt` |

Machine-readable full-precision metrics: `outputs/analysis/vc_ot_cross_ablation.csv`.

## VC-v5 rows (2026-09-30)

Same orchestrator command, HP, seed 42 and evaluation protocol as the `v4 × v2_armrot_biclose` rows above; only the VC changes (trained from scratch with the v4 VC config). v5a adds rotation-invariant forearm/object and bimanual coupling features to v4's pose prior; v5ar additionally perturbs the windows' velocity initializers during training (online exposure bias). Details, oracles and paired statistics: `docs/vc_v5_oracles_and_coupling_20260930.md`.

| Dataset | VC | OT | Object (cm) | HOI (cm) | MPJPE (cm) | Root (cm) | L F1 | R F1 | Obj jitter (mm/frame²) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OMOMO | v4 | v2_armrot_biclose | 13.366 | 10.443 | 2.986 | 6.044 | 0.949 | 0.952 | 5.335 |
| OMOMO | v5a | v2_armrot_biclose | 13.594 | 10.459 | 2.984 | 6.031 | 0.953 | 0.957 | 5.228 |
| OMOMO | v5ar | v2_armrot_biclose | 13.226 | 10.479 | 2.985 | 6.009 | 0.956 | 0.955 | 5.323 |
| BEHAVE | v4 | v2_armrot_biclose | 23.970 | 17.815 | 3.678 | 16.129 | 0.802 | 0.796 | 6.487 |
| BEHAVE | v5a | v2_armrot_biclose | 23.555 | 16.958 | 3.726 | 16.477 | 0.841 | 0.837 | 6.548 |
| BEHAVE | v5ar | v2_armrot_biclose | 22.934 | 17.186 | 3.682 | 16.133 | 0.838 | 0.845 | 5.739 |

Run roots: `outputs/IMUHOI++/RNN_cano/vc_v5_20260930_ot_{omomo,behave}_{v5a,v5ar}/` (manifest `experiment_manifest.json`, selected `best_object_trans.pt`); VC checkpoints under `outputs/IMUHOI++/RNN_cano/vc_v5_20260930/{omomo,behave}_{v5a,v5ar}/`. BEHAVE final checkpoints: v5a 23.395 / 16.999, v5ar 22.882 / 17.139 (v4 24.084 / 17.854).

Seed check (BEHAVE, training seed 43, same recipe, test-noise seed 42): v4 best 24.496 / 18.616, v5ar best 23.656 / 17.222; v4 final 24.059 / 18.362, v5ar final 23.418 / 16.661. Averaged over the two training seeds, the per-sequence paired difference v5ar − v4 (final) is object −0.92 cm [−1.56, −0.32] and HOI −1.21 cm [−1.92, −0.50]; with the initial contact state read from frame 1 (the frame-0 hand-contact label artifact, see the VC-v5 doc) it is −0.77 [−1.40, −0.19] / −0.96 [−1.63, −0.29].

## Original three-branch IMUHOI (2026-10-08)

VC-IMUHOI + OT-IMUHOI with the conference paper's three candidates (left FK, right FK, IMU integration; `object_trans_arch: imuhoi`, no static hold), trained from scratch (VC → OT → joint stage fine-tuning HP) with exactly the settings of the four-head rows above (seed 42, HP reused; OMOMO previous protocol, BEHAVE HP smoothing 7 + diff-3 velocity init). Roots `outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_{omomo,behave}/`, chain `chain_imuhoi3b_20261008.sh`, variant `vc_imuhoi_ot_3branch` in `scripts/run_behave_full_pipeline_trust.py`. The VC was retrained, so the 3- vs 4-branch difference also contains VC training noise.

| Dataset | ckpt | Object (cm) | HOI (cm) | Root (cm) | MPJPE (cm) | L F1 | R F1 | Obj jitter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OMOMO | best | 19.303 | 15.580 | 10.953 | 3.029 | 0.921 | 0.926 | 5.019 |
| OMOMO | final | 19.177 | 15.436 | 11.143 | 3.016 | 0.921 | 0.926 | 5.060 |
| BEHAVE | best | 95.455 | 79.903 | 26.673 | 3.723 | 0.799 | 0.799 | 3.237 |
| BEHAVE | final | 98.050 | 83.370 | 29.296 | 3.709 | 0.799 | 0.797 | 3.181 |

On BEHAVE the three-branch fusion has no static candidate: every frame without hand contact (including resting objects) goes to IMU integration, so the velocity bias integrates over the 1,400-frame sequences (IMU-candidate error 266 cm).
