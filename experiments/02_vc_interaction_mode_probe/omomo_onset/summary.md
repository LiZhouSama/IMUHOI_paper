# Onset imminence: OMOMO test

1173 sequences; 34546 non-contact hand-frames on sliding-window frames (30000 idle = no onset of that hand within 30 frames, subsampled). Published frame t has seen inputs up to t+10: leads 1-10 are inside the lookahead.

## AUC against idle frames (95% CI, sequence bootstrap)

| score | lead 1-10 (onset visible) | lead 11-20 | lead 21-30 |
|---|---|---|---|
| IMUHOI++ anticipation head | 0.933 [0.915, 0.946] | 0.865 [0.838, 0.887] | 0.746 [0.705, 0.794] |
| IMUHOI++ published contact prob. | 0.736 [0.702, 0.770] | 0.510 [0.448, 0.572] | 0.504 [0.446, 0.575] |
| IMUHOI published contact prob. | 0.745 [0.714, 0.769] | 0.600 [0.551, 0.652] | 0.587 [0.525, 0.648] |
| IMUHOI contact prob. at t+10 | 0.937 [0.925, 0.947] | 0.795 [0.763, 0.825] | 0.655 [0.592, 0.714] |
| probe: IMUHOI++ pose prior b_t | 0.906 [0.881, 0.926] | 0.887 [0.858, 0.913] | 0.786 [0.748, 0.824] |
| probe: IMUHOI++ hand-contact LSTM | 0.912 [0.892, 0.929] | 0.852 [0.822, 0.881] | 0.726 [0.678, 0.772] |
| probe: IMUHOI contact LSTM | 0.881 [0.855, 0.901] | 0.824 [0.788, 0.857] | 0.724 [0.667, 0.763] |
| probe: IMUHOI contact LSTM at t+10 | 0.930 [0.909, 0.949] | 0.866 [0.835, 0.892] | 0.735 [0.675, 0.778] |
| probe: HP pose input | 0.852 [0.817, 0.878] | 0.785 [0.747, 0.824] | 0.652 [0.597, 0.700] |
| probe: untrained b_t | 0.851 [0.822, 0.881] | 0.798 [0.762, 0.838] | 0.688 [0.647, 0.728] |

Same, restricted to frames where the other hand is not in contact either (no cue from a hand that already holds the object):

| score | lead 1-10 | lead 11-20 | lead 21-30 |
|---|---|---|---|
| IMUHOI++ anticipation head | 0.934 [0.914, 0.952] | 0.903 [0.876, 0.928] | 0.809 [0.765, 0.863] |
| IMUHOI++ published contact prob. | 0.735 [0.698, 0.768] | 0.486 [0.427, 0.555] | 0.474 [0.394, 0.565] |
| IMUHOI published contact prob. | 0.700 [0.658, 0.738] | 0.540 [0.481, 0.608] | 0.520 [0.440, 0.597] |
| IMUHOI contact prob. at t+10 | 0.945 [0.930, 0.957] | 0.771 [0.724, 0.813] | 0.610 [0.531, 0.693] |
| probe: IMUHOI++ pose prior b_t | 0.914 [0.884, 0.936] | 0.927 [0.899, 0.951] | 0.885 [0.855, 0.916] |
| probe: IMUHOI++ hand-contact LSTM | 0.937 [0.915, 0.954] | 0.909 [0.882, 0.939] | 0.808 [0.754, 0.868] |
| probe: IMUHOI contact LSTM | 0.917 [0.892, 0.938] | 0.879 [0.834, 0.914] | 0.771 [0.706, 0.815] |
| probe: IMUHOI contact LSTM at t+10 | 0.939 [0.914, 0.960] | 0.893 [0.856, 0.923] | 0.762 [0.686, 0.819] |
| probe: HP pose input | 0.886 [0.845, 0.917] | 0.884 [0.850, 0.917] | 0.746 [0.692, 0.797] |
| probe: untrained b_t | 0.876 [0.839, 0.909] | 0.885 [0.852, 0.915] | 0.769 [0.714, 0.815] |

Fresh approach: both hands free and no contact of either hand in the previous 30 frames (positives and idle alike; {'positives_1_10': 1074, 'positives_11_20': 688, 'idle': 8257}):

| score | lead 1-10 | lead 11-20 | lead 21-30 |
|---|---|---|---|
| IMUHOI++ anticipation head | 0.925 [0.899, 0.947] | 0.915 [0.882, 0.941] | 0.834 [0.786, 0.893] |
| IMUHOI++ published contact prob. | 0.767 [0.717, 0.812] | 0.462 [0.363, 0.555] | 0.440 [0.327, 0.534] |
| IMUHOI published contact prob. | 0.837 [0.799, 0.873] | 0.631 [0.544, 0.707] | 0.570 [0.460, 0.684] |
| IMUHOI contact prob. at t+10 | 0.978 [0.964, 0.988] | 0.853 [0.808, 0.899] | 0.641 [0.524, 0.747] |
| probe: IMUHOI++ pose prior b_t | 0.936 [0.911, 0.958] | 0.964 [0.940, 0.983] | 0.923 [0.884, 0.958] |
| probe: IMUHOI++ hand-contact LSTM | 0.947 [0.921, 0.964] | 0.935 [0.906, 0.964] | 0.823 [0.754, 0.898] |
| probe: IMUHOI contact LSTM | 0.935 [0.909, 0.956] | 0.917 [0.870, 0.953] | 0.828 [0.750, 0.888] |
| probe: IMUHOI contact LSTM at t+10 | 0.958 [0.936, 0.973] | 0.924 [0.884, 0.956] | 0.817 [0.724, 0.884] |
| probe: HP pose input | 0.923 [0.886, 0.951] | 0.954 [0.928, 0.977] | 0.821 [0.741, 0.894] |
| probe: untrained b_t | 0.901 [0.860, 0.935] | 0.936 [0.904, 0.968] | 0.807 [0.721, 0.867] |

Locomotion: share of walking frames (root speed >= 0.3 m/s) among fresh positives at lead 11-30 0.48, among fresh idle frames 0.37.

Fresh approach within locomotion strata (AUC [95% CI]):

| task | n positives / idle | IMUHOI++ anticipation head | probe: IMUHOI++ pose prior b_t | probe: HP pose input | probe: untrained b_t | probe: IMUHOI contact LSTM at t+10 | IMUHOI contact prob. at t+10 |
|---|---|---|---|---|---|---|---|
| fresh_11_20_moving | 318 / 3071 | 0.869 [0.811, 0.922] | 0.928 [0.853, 0.969] | 0.919 [0.855, 0.964] | 0.898 [0.830, 0.949] | 0.906 [0.841, 0.950] | 0.865 [0.804, 0.931] |
| fresh_11_20_still | 370 / 5186 | 0.944 [0.909, 0.972] | 0.985 [0.975, 0.994] | 0.980 [0.966, 0.991] | 0.963 [0.931, 0.986] | 0.935 [0.887, 0.967] | 0.842 [0.774, 0.900] |
| fresh_21_30_moving | 238 / 3071 | 0.793 [0.716, 0.863] | 0.923 [0.860, 0.962] | 0.847 [0.757, 0.921] | 0.819 [0.709, 0.908] | 0.801 [0.720, 0.879] | 0.587 [0.469, 0.707] |
| fresh_21_30_still | 237 / 5186 | 0.859 [0.806, 0.925] | 0.920 [0.866, 0.963] | 0.792 [0.679, 0.919] | 0.790 [0.676, 0.883] | 0.821 [0.665, 0.930] | 0.701 [0.518, 0.837] |

Paired AUC differences on fresh approaches (sequence bootstrap):

| task | difference | AUC difference [95% CI] |
|---|---|---|
| fresh_11_20 | probe: IMUHOI++ pose prior b_t minus probe: HP pose input | +0.010 [-0.005, +0.028] |
| fresh_11_20 | probe: IMUHOI++ pose prior b_t minus probe: untrained b_t | +0.028 [+0.012, +0.048] |
| fresh_11_20 | probe: IMUHOI++ pose prior b_t minus probe: IMUHOI contact LSTM at t+10 | +0.040 [+0.015, +0.072] |
| fresh_11_20 | IMUHOI++ anticipation head minus IMUHOI contact prob. at t+10 | +0.062 [+0.017, +0.111] |
| fresh_21_30 | probe: IMUHOI++ pose prior b_t minus probe: HP pose input | +0.102 [+0.053, +0.174] |
| fresh_21_30 | probe: IMUHOI++ pose prior b_t minus probe: untrained b_t | +0.117 [+0.075, +0.179] |
| fresh_21_30 | probe: IMUHOI++ pose prior b_t minus probe: IMUHOI contact LSTM at t+10 | +0.107 [+0.046, +0.179] |
| fresh_21_30 | IMUHOI++ anticipation head minus IMUHOI contact prob. at t+10 | +0.193 [+0.079, +0.308] |

## Detection lead at 5% false alarms on idle frames (112 onsets with 30 idle frames before)

| score | threshold | detected by the onset | >= 1 f early | >= 5 f early | >= 10 f early | >= 11 f early | median lead |
|---|---|---|---|---|---|---|---|
| IMUHOI++ anticipation head | 0.132 | 0.59 | 0.54 | 0.49 | 0.37 | 0.36 | +4 |
| IMUHOI++ published contact prob. | 0.511 | 0.46 | 0.39 | 0.13 | 0.08 | 0.07 | -1 |
| IMUHOI published contact prob. | 0.939 | 0.19 | 0.14 | 0.07 | 0.05 | 0.05 | -4 |
| IMUHOI contact prob. at t+10 | 0.880 | 0.78 | 0.71 | 0.54 | 0.19 | 0.13 | +6 |

Lead >= 11 frames = the score was above threshold before the published window could contain the onset.
Negative median lead = detected after the onset (frames).
