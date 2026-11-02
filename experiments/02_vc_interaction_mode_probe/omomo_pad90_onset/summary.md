# Onset imminence: OMOMO test

Sequences carry a 90-frame static prefix (copies of their first frame); only real frames are scored: candidates, idle frames and the 30 frames before a detection onset all lie after the prefix.

861 sequences; 59014 non-contact hand-frames on sliding-window frames (30000 idle = no onset of that hand within 30 frames, subsampled). Published frame t has seen inputs up to t+10: leads 1-10 are inside the lookahead.

## AUC against idle frames (95% CI, sequence bootstrap)

| score | lead 1-10 (onset visible) | lead 11-20 | lead 21-30 |
|---|---|---|---|
| IMUHOI++ anticipation head | 0.944 [0.937, 0.951] | 0.906 [0.897, 0.915] | 0.800 [0.782, 0.818] |
| IMUHOI++ published contact prob. | 0.727 [0.712, 0.744] | 0.373 [0.350, 0.398] | 0.307 [0.280, 0.335] |
| IMUHOI published contact prob. | 0.714 [0.693, 0.733] | 0.513 [0.488, 0.535] | 0.469 [0.444, 0.496] |
| IMUHOI contact prob. at t+10 | 0.941 [0.931, 0.950] | 0.741 [0.720, 0.760] | 0.549 [0.521, 0.582] |
| probe: IMUHOI++ pose prior b_t | 0.925 [0.916, 0.935] | 0.919 [0.910, 0.931] | 0.849 [0.832, 0.866] |
| probe: IMUHOI++ hand-contact LSTM | 0.916 [0.907, 0.927] | 0.894 [0.881, 0.906] | 0.802 [0.782, 0.821] |
| probe: IMUHOI contact LSTM | 0.903 [0.893, 0.913] | 0.893 [0.880, 0.904] | 0.816 [0.798, 0.831] |
| probe: IMUHOI contact LSTM at t+10 | 0.938 [0.932, 0.946] | 0.917 [0.907, 0.926] | 0.794 [0.773, 0.811] |
| probe: HP pose input | 0.896 [0.887, 0.907] | 0.881 [0.871, 0.894] | 0.779 [0.761, 0.797] |
| probe: untrained b_t | 0.868 [0.857, 0.882] | 0.862 [0.849, 0.877] | 0.777 [0.759, 0.795] |

Same, restricted to frames where the other hand is not in contact either (no cue from a hand that already holds the object):

| score | lead 1-10 | lead 11-20 | lead 21-30 |
|---|---|---|---|
| IMUHOI++ anticipation head | 0.941 [0.931, 0.949] | 0.908 [0.896, 0.917] | 0.805 [0.783, 0.822] |
| IMUHOI++ published contact prob. | 0.749 [0.732, 0.766] | 0.408 [0.381, 0.435] | 0.330 [0.303, 0.356] |
| IMUHOI published contact prob. | 0.685 [0.658, 0.705] | 0.489 [0.460, 0.515] | 0.445 [0.415, 0.474] |
| IMUHOI contact prob. at t+10 | 0.939 [0.928, 0.950] | 0.716 [0.687, 0.739] | 0.525 [0.493, 0.556] |
| probe: IMUHOI++ pose prior b_t | 0.923 [0.912, 0.933] | 0.928 [0.917, 0.939] | 0.869 [0.849, 0.887] |
| probe: IMUHOI++ hand-contact LSTM | 0.917 [0.905, 0.928] | 0.907 [0.893, 0.918] | 0.817 [0.797, 0.834] |
| probe: IMUHOI contact LSTM | 0.897 [0.885, 0.908] | 0.897 [0.886, 0.908] | 0.811 [0.790, 0.828] |
| probe: IMUHOI contact LSTM at t+10 | 0.937 [0.930, 0.946] | 0.915 [0.906, 0.925] | 0.796 [0.773, 0.815] |
| probe: HP pose input | 0.909 [0.898, 0.921] | 0.909 [0.896, 0.920] | 0.819 [0.797, 0.838] |
| probe: untrained b_t | 0.877 [0.863, 0.891] | 0.885 [0.871, 0.899] | 0.809 [0.788, 0.827] |

Fresh approach: both hands free and no contact of either hand in the previous 30 frames (positives and idle alike; {'positives_1_10': 10652, 'positives_11_20': 8578, 'idle': 14031}):

| score | lead 1-10 | lead 11-20 | lead 21-30 |
|---|---|---|---|
| IMUHOI++ anticipation head | 0.934 [0.923, 0.943] | 0.901 [0.888, 0.912] | 0.793 [0.769, 0.812] |
| IMUHOI++ published contact prob. | 0.777 [0.754, 0.794] | 0.421 [0.390, 0.451] | 0.333 [0.304, 0.362] |
| IMUHOI published contact prob. | 0.750 [0.719, 0.774] | 0.540 [0.508, 0.570] | 0.484 [0.453, 0.515] |
| IMUHOI contact prob. at t+10 | 0.952 [0.938, 0.964] | 0.750 [0.723, 0.775] | 0.548 [0.517, 0.583] |
| probe: IMUHOI++ pose prior b_t | 0.918 [0.905, 0.931] | 0.926 [0.913, 0.939] | 0.869 [0.848, 0.887] |
| probe: IMUHOI++ hand-contact LSTM | 0.912 [0.900, 0.925] | 0.904 [0.889, 0.917] | 0.816 [0.793, 0.834] |
| probe: IMUHOI contact LSTM | 0.885 [0.869, 0.898] | 0.889 [0.877, 0.901] | 0.806 [0.784, 0.826] |
| probe: IMUHOI contact LSTM at t+10 | 0.933 [0.924, 0.943] | 0.909 [0.895, 0.921] | 0.792 [0.767, 0.815] |
| probe: HP pose input | 0.916 [0.903, 0.929] | 0.922 [0.908, 0.933] | 0.841 [0.818, 0.861] |
| probe: untrained b_t | 0.876 [0.862, 0.892] | 0.892 [0.874, 0.905] | 0.818 [0.794, 0.840] |

Locomotion: share of walking frames (root speed >= 0.3 m/s) among fresh positives at lead 11-30 0.22, among fresh idle frames 0.32.

Fresh approach within locomotion strata (AUC [95% CI]):

| task | n positives / idle | IMUHOI++ anticipation head | probe: IMUHOI++ pose prior b_t | probe: HP pose input | probe: untrained b_t | probe: IMUHOI contact LSTM at t+10 | IMUHOI contact prob. at t+10 |
|---|---|---|---|---|---|---|---|
| fresh_11_20_moving | 1772 / 4467 | 0.874 [0.846, 0.898] | 0.937 [0.917, 0.955] | 0.939 [0.921, 0.955] | 0.910 [0.887, 0.933] | 0.912 [0.888, 0.931] | 0.765 [0.723, 0.803] |
| fresh_11_20_still | 6806 / 9564 | 0.908 [0.893, 0.919] | 0.921 [0.905, 0.936] | 0.914 [0.900, 0.927] | 0.884 [0.865, 0.900] | 0.905 [0.890, 0.918] | 0.739 [0.711, 0.767] |
| fresh_11_20_window_with_prefix | 7910 / 8171 | 0.887 [0.871, 0.901] | 0.895 [0.877, 0.910] | 0.892 [0.874, 0.906] | 0.860 [0.839, 0.876] | 0.871 [0.856, 0.887] | 0.720 [0.691, 0.744] |
| fresh_11_20_window_without_prefix | 668 / 5860 | 0.917 [0.884, 0.944] | 0.955 [0.927, 0.974] | 0.954 [0.925, 0.975] | 0.924 [0.885, 0.957] | 0.940 [0.921, 0.957] | 0.821 [0.762, 0.869] |
| fresh_21_30_moving | 1369 / 4467 | 0.751 [0.699, 0.787] | 0.869 [0.834, 0.901] | 0.830 [0.794, 0.869] | 0.808 [0.772, 0.846] | 0.779 [0.744, 0.820] | 0.587 [0.527, 0.639] |
| fresh_21_30_still | 4269 / 9564 | 0.808 [0.783, 0.829] | 0.866 [0.846, 0.885] | 0.840 [0.816, 0.861] | 0.818 [0.790, 0.841] | 0.793 [0.765, 0.815] | 0.529 [0.491, 0.566] |
| fresh_21_30_window_with_prefix | 5164 / 8171 | 0.771 [0.743, 0.796] | 0.819 [0.795, 0.838] | 0.796 [0.769, 0.817] | 0.778 [0.750, 0.800] | 0.725 [0.696, 0.754] | 0.514 [0.481, 0.545] |
| fresh_21_30_window_without_prefix | 474 / 5860 | 0.848 [0.793, 0.898] | 0.909 [0.865, 0.947] | 0.830 [0.757, 0.891] | 0.803 [0.722, 0.862] | 0.839 [0.777, 0.890] | 0.632 [0.532, 0.733] |

Paired AUC differences on fresh approaches (sequence bootstrap):

| task | difference | AUC difference [95% CI] |
|---|---|---|
| fresh_11_20 | probe: IMUHOI++ pose prior b_t minus probe: HP pose input | +0.005 [-0.002, +0.010] |
| fresh_11_20 | probe: IMUHOI++ pose prior b_t minus probe: untrained b_t | +0.035 [+0.027, +0.042] |
| fresh_11_20 | probe: IMUHOI++ pose prior b_t minus probe: IMUHOI contact LSTM at t+10 | +0.018 [+0.010, +0.026] |
| fresh_11_20 | IMUHOI++ anticipation head minus IMUHOI contact prob. at t+10 | +0.151 [+0.126, +0.184] |
| fresh_21_30 | probe: IMUHOI++ pose prior b_t minus probe: HP pose input | +0.028 [+0.017, +0.039] |
| fresh_21_30 | probe: IMUHOI++ pose prior b_t minus probe: untrained b_t | +0.050 [+0.037, +0.062] |
| fresh_21_30 | probe: IMUHOI++ pose prior b_t minus probe: IMUHOI contact LSTM at t+10 | +0.077 [+0.058, +0.096] |
| fresh_21_30 | IMUHOI++ anticipation head minus IMUHOI contact prob. at t+10 | +0.245 [+0.211, +0.277] |

## Detection lead at 5% false alarms on idle frames (526 onsets with 30 idle frames before)

| score | threshold | detected by the onset | >= 1 f early | >= 5 f early | >= 10 f early | >= 11 f early | median lead |
|---|---|---|---|---|---|---|---|
| IMUHOI++ anticipation head | 0.144 | 0.67 | 0.66 | 0.60 | 0.51 | 0.48 | +10 |
| IMUHOI++ published contact prob. | 0.331 | 0.61 | 0.47 | 0.10 | 0.04 | 0.03 | +0 |
| IMUHOI published contact prob. | 0.954 | 0.15 | 0.09 | 0.04 | 0.02 | 0.02 | -4 |
| IMUHOI contact prob. at t+10 | 0.925 | 0.78 | 0.75 | 0.61 | 0.17 | 0.12 | +7 |

Lead >= 11 frames = the score was above threshold before the published window could contain the onset.
Negative median lead = detected after the onset (frames).
