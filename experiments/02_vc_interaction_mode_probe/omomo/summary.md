# VC mode probe pilot: OMOMO test

Segments by mode (sit sequences excluded): {'ground': 214, 'lift': 635, 'tilt': 252, 'other': 19, 'ground_rot': 2}

## Segment decoding `onset/all`

n = {'segments': 849, 'lift': 635, 'ground': 214, 'sequences': 829, 'groups': 17}

| representation | balanced acc. [95% CI] | minus IMUHOI LSTM | minus IMUHOI LSTM (+10 f) |
|---|---|---|---|
| IMUHOI++ pose prior b_t | 0.917 [0.895, 0.938] | +0.019 [-0.010, +0.050] | +0.033 [+0.001, +0.066] |
| IMUHOI++ hand-contact LSTM | 0.791 [0.759, 0.821] | -0.107 [-0.146, -0.073] | -0.093 [-0.132, -0.056] |
| IMUHOI++ object-motion LSTM | 0.919 [0.894, 0.940] | +0.021 [-0.005, +0.048] | +0.034 [-0.000, +0.064] |
| IMUHOI contact LSTM | 0.897 [0.872, 0.920] | - | +0.014 [-0.012, +0.040] |
| IMUHOI contact LSTM, +10 f horizon | 0.883 [0.859, 0.909] | -0.014 [-0.040, +0.012] | - |
| input: HP pose stream | 0.913 [0.889, 0.935] | +0.015 [-0.015, +0.047] | +0.029 [-0.006, +0.062] |
| input: coupling features | 0.857 [0.825, 0.883] | -0.042 [-0.076, -0.009] | -0.028 [-0.065, +0.007] |
| input: forearm/object IMU + vel. | 0.894 [0.869, 0.917] | -0.003 [-0.034, +0.027] | +0.011 [-0.022, +0.041] |
| control: untrained b_t | 0.887 [0.861, 0.911] | -0.010 [-0.042, +0.022] | +0.003 [-0.030, +0.038] |
| group encoder: left_arm | 0.847 [0.819, 0.874] | -0.051 [-0.081, -0.020] | -0.037 [-0.072, -0.004] |
| group encoder: right_arm | 0.868 [0.839, 0.896] | -0.030 [-0.062, +0.002] | -0.016 [-0.051, +0.017] |
| group encoder: torso | 0.884 [0.859, 0.909] | -0.013 [-0.046, +0.020] | +0.001 [-0.034, +0.038] |
| group encoder: left_leg | 0.854 [0.825, 0.881] | -0.044 [-0.079, -0.009] | -0.030 [-0.064, +0.006] |
| group encoder: right_leg | 0.859 [0.832, 0.885] | -0.038 [-0.072, -0.003] | -0.024 [-0.059, +0.011] |

b_t minus untrained b_t: +0.030 [+0.009, +0.053]; b_t minus HP pose input: +0.004 [-0.018, +0.026]

## Segment decoding `onset/sliding`

n = {'segments': 79, 'lift': 48, 'ground': 31, 'sequences': 79, 'groups': 14}

| representation | balanced acc. [95% CI] | minus IMUHOI LSTM | minus IMUHOI LSTM (+10 f) |
|---|---|---|---|
| IMUHOI++ pose prior b_t | 0.889 [0.815, 0.952] | +0.069 [-0.020, +0.165] | +0.047 [-0.036, +0.134] |
| IMUHOI++ hand-contact LSTM | 0.815 [0.725, 0.900] | -0.004 [-0.097, +0.089] | -0.026 [-0.106, +0.052] |
| IMUHOI++ object-motion LSTM | 0.841 [0.748, 0.922] | +0.021 [-0.078, +0.120] | -0.001 [-0.109, +0.106] |
| IMUHOI contact LSTM | 0.820 [0.725, 0.904] | - | -0.022 [-0.083, +0.030] |
| IMUHOI contact LSTM, +10 f horizon | 0.842 [0.749, 0.919] | +0.022 [-0.030, +0.083] | - |
| input: HP pose stream | 0.751 [0.654, 0.844] | -0.068 [-0.174, +0.031] | -0.090 [-0.195, +0.007] |
| input: coupling features | 0.920 [0.850, 0.974] | +0.101 [+0.010, +0.189] | +0.079 [-0.014, +0.172] |
| input: forearm/object IMU + vel. | 0.719 [0.618, 0.812] | -0.101 [-0.209, +0.007] | -0.123 [-0.231, -0.022] |
| control: untrained b_t | 0.729 [0.629, 0.827] | -0.091 [-0.208, +0.027] | -0.114 [-0.226, -0.004] |

b_t minus untrained b_t: +0.160 [+0.078, +0.257]; b_t minus HP pose input: +0.137 [+0.045, +0.235]

## Segment decoding `onset/bimanual`

n = {'segments': 539, 'lift': 438, 'ground': 101, 'sequences': 534, 'groups': 17}

| representation | balanced acc. [95% CI] | minus IMUHOI LSTM | minus IMUHOI LSTM (+10 f) |
|---|---|---|---|
| IMUHOI++ pose prior b_t | 0.940 [0.911, 0.966] | +0.048 [+0.010, +0.087] | +0.042 [+0.003, +0.079] |
| IMUHOI++ hand-contact LSTM | 0.875 [0.839, 0.909] | -0.017 [-0.057, +0.024] | -0.023 [-0.064, +0.016] |
| IMUHOI++ object-motion LSTM | 0.907 [0.874, 0.939] | +0.015 [-0.033, +0.058] | +0.008 [-0.034, +0.048] |
| IMUHOI contact LSTM | 0.891 [0.854, 0.928] | - | -0.007 [-0.036, +0.022] |
| IMUHOI contact LSTM, +10 f horizon | 0.897 [0.862, 0.933] | +0.007 [-0.022, +0.036] | - |
| input: HP pose stream | 0.911 [0.877, 0.941] | +0.019 [-0.026, +0.063] | +0.012 [-0.031, +0.056] |
| input: coupling features | 0.892 [0.856, 0.924] | +0.000 [-0.047, +0.050] | -0.007 [-0.053, +0.041] |
| input: forearm/object IMU + vel. | 0.953 [0.930, 0.972] | +0.061 [+0.025, +0.102] | +0.055 [+0.017, +0.093] |
| control: untrained b_t | 0.906 [0.869, 0.939] | +0.014 [-0.032, +0.061] | +0.008 [-0.035, +0.051] |

b_t minus untrained b_t: +0.034 [-0.002, +0.073]; b_t minus HP pose input: +0.030 [-0.003, +0.062]

## Segment decoding `onset/loo_object`

n = {'segments': 567, 'lift': 363, 'ground': 204, 'sequences': 548, 'groups': 10, 'objects': ['floorlamp', 'largebox', 'largetable', 'mop', 'plasticbox', 'smallbox', 'smalltable', 'suitcase', 'trashcan', 'whitechair']}

| representation | balanced acc. [95% CI] | minus IMUHOI LSTM | minus IMUHOI LSTM (+10 f) |
|---|---|---|---|
| IMUHOI++ pose prior b_t | 0.911 [0.886, 0.936] | +0.003 [-0.027, +0.035] | +0.015 [-0.018, +0.048] |
| IMUHOI++ hand-contact LSTM | 0.830 [0.800, 0.859] | -0.078 [-0.114, -0.044] | -0.066 [-0.104, -0.030] |
| IMUHOI++ object-motion LSTM | 0.923 [0.899, 0.945] | +0.014 [-0.015, +0.045] | +0.026 [-0.006, +0.059] |
| IMUHOI contact LSTM | 0.908 [0.881, 0.932] | - | +0.012 [-0.013, +0.036] |
| IMUHOI contact LSTM, +10 f horizon | 0.896 [0.869, 0.921] | -0.012 [-0.036, +0.013] | - |
| input: HP pose stream | 0.902 [0.874, 0.927] | -0.006 [-0.036, +0.025] | +0.006 [-0.026, +0.041] |
| input: coupling features | 0.877 [0.847, 0.905] | -0.031 [-0.061, +0.001] | -0.019 [-0.055, +0.016] |
| input: forearm/object IMU + vel. | 0.887 [0.856, 0.916] | -0.021 [-0.053, +0.011] | -0.009 [-0.045, +0.026] |
| control: untrained b_t | 0.876 [0.845, 0.905] | -0.032 [-0.065, +0.002] | -0.020 [-0.058, +0.017] |

b_t minus untrained b_t: +0.034 [+0.010, +0.061]; b_t minus HP pose input: +0.009 [-0.015, +0.036]

## Segment decoding `hold/all`

n = {'segments': 839, 'lift': 635, 'ground': 204, 'sequences': 826, 'groups': 17}

| representation | balanced acc. [95% CI] | minus IMUHOI LSTM | minus IMUHOI LSTM (+10 f) |
|---|---|---|---|
| IMUHOI++ pose prior b_t | 0.915 [0.891, 0.937] | +0.051 [+0.020, +0.083] | +0.071 [+0.037, +0.109] |
| IMUHOI++ hand-contact LSTM | 0.795 [0.759, 0.828] | -0.069 [-0.114, -0.027] | -0.050 [-0.094, -0.009] |
| IMUHOI++ object-motion LSTM | 0.926 [0.905, 0.947] | +0.063 [+0.033, +0.092] | +0.083 [+0.051, +0.116] |
| IMUHOI contact LSTM | 0.864 [0.836, 0.890] | - | +0.020 [-0.007, +0.046] |
| IMUHOI contact LSTM, +10 f horizon | 0.843 [0.811, 0.871] | -0.020 [-0.046, +0.007] | - |
| input: HP pose stream | 0.904 [0.876, 0.929] | +0.040 [+0.006, +0.074] | +0.060 [+0.026, +0.095] |
| input: coupling features | 0.888 [0.863, 0.915] | +0.025 [-0.011, +0.057] | +0.045 [+0.008, +0.084] |
| input: forearm/object IMU + vel. | 0.876 [0.848, 0.900] | +0.012 [-0.022, +0.046] | +0.032 [-0.004, +0.067] |
| control: untrained b_t | 0.881 [0.851, 0.911] | +0.018 [-0.019, +0.052] | +0.038 [+0.005, +0.074] |
| group encoder: left_arm | 0.874 [0.844, 0.901] | +0.010 [-0.025, +0.046] | +0.030 [-0.004, +0.066] |
| group encoder: right_arm | 0.886 [0.858, 0.914] | +0.023 [-0.012, +0.058] | +0.043 [+0.010, +0.077] |
| group encoder: torso | 0.893 [0.866, 0.917] | +0.029 [-0.004, +0.062] | +0.049 [+0.017, +0.083] |
| group encoder: left_leg | 0.860 [0.831, 0.891] | -0.002 [-0.040, +0.032] | +0.017 [-0.019, +0.055] |
| group encoder: right_leg | 0.853 [0.821, 0.884] | -0.010 [-0.049, +0.026] | +0.010 [-0.027, +0.049] |

b_t minus untrained b_t: +0.033 [+0.010, +0.059]; b_t minus HP pose input: +0.011 [-0.008, +0.032]

## Segment decoding `hold/sliding`

n = {'segments': 76, 'lift': 48, 'ground': 28, 'sequences': 76, 'groups': 14}

| representation | balanced acc. [95% CI] | minus IMUHOI LSTM | minus IMUHOI LSTM (+10 f) |
|---|---|---|---|
| IMUHOI++ pose prior b_t | 0.923 [0.855, 0.980] | +0.148 [+0.055, +0.248] | +0.077 [-0.007, +0.165] |
| IMUHOI++ hand-contact LSTM | 0.856 [0.764, 0.930] | +0.080 [-0.040, +0.193] | +0.009 [-0.085, +0.109] |
| IMUHOI++ object-motion LSTM | 0.894 [0.816, 0.961] | +0.121 [+0.011, +0.226] | +0.050 [-0.053, +0.158] |
| IMUHOI contact LSTM | 0.774 [0.672, 0.864] | - | -0.071 [-0.170, +0.019] |
| IMUHOI contact LSTM, +10 f horizon | 0.845 [0.756, 0.922] | +0.071 [-0.019, +0.170] | - |
| input: HP pose stream | 0.845 [0.754, 0.927] | +0.071 [-0.048, +0.185] | +0.000 [-0.115, +0.110] |
| input: coupling features | 0.940 [0.882, 0.988] | +0.166 [+0.061, +0.274] | +0.095 [+0.009, +0.187] |
| input: forearm/object IMU + vel. | 0.784 [0.688, 0.872] | +0.010 [-0.120, +0.141] | -0.061 [-0.177, +0.070] |
| control: untrained b_t | 0.774 [0.670, 0.870] | -0.000 [-0.146, +0.129] | -0.071 [-0.200, +0.047] |

b_t minus untrained b_t: +0.148 [+0.069, +0.243]; b_t minus HP pose input: +0.077 [-0.000, +0.165]

## Segment decoding `hold/bimanual`

n = {'segments': 536, 'lift': 438, 'ground': 98, 'sequences': 532, 'groups': 17}

| representation | balanced acc. [95% CI] | minus IMUHOI LSTM | minus IMUHOI LSTM (+10 f) |
|---|---|---|---|
| IMUHOI++ pose prior b_t | 0.961 [0.934, 0.983] | +0.025 [-0.007, +0.058] | +0.051 [+0.019, +0.087] |
| IMUHOI++ hand-contact LSTM | 0.857 [0.813, 0.893] | -0.080 [-0.123, -0.038] | -0.055 [-0.101, -0.010] |
| IMUHOI++ object-motion LSTM | 0.943 [0.914, 0.968] | +0.007 [-0.021, +0.035] | +0.033 [+0.000, +0.066] |
| IMUHOI contact LSTM | 0.935 [0.908, 0.960] | - | +0.025 [+0.000, +0.054] |
| IMUHOI contact LSTM, +10 f horizon | 0.910 [0.877, 0.941] | -0.025 [-0.054, -0.000] | - |
| input: HP pose stream | 0.910 [0.872, 0.944] | -0.025 [-0.064, +0.015] | +0.000 [-0.040, +0.043] |
| input: coupling features | 0.915 [0.883, 0.943] | -0.020 [-0.053, +0.012] | +0.005 [-0.031, +0.042] |
| input: forearm/object IMU + vel. | 0.914 [0.884, 0.945] | -0.021 [-0.056, +0.015] | +0.004 [-0.038, +0.047] |
| control: untrained b_t | 0.918 [0.883, 0.950] | -0.019 [-0.058, +0.019] | +0.007 [-0.034, +0.046] |

b_t minus untrained b_t: +0.044 [+0.014, +0.077]; b_t minus HP pose input: +0.051 [+0.021, +0.085]

## Segment decoding `hold/loo_object`

n = {'segments': 558, 'lift': 363, 'ground': 195, 'sequences': 546, 'groups': 10, 'objects': ['floorlamp', 'largebox', 'largetable', 'mop', 'plasticbox', 'smallbox', 'smalltable', 'suitcase', 'trashcan', 'whitechair']}

| representation | balanced acc. [95% CI] | minus IMUHOI LSTM | minus IMUHOI LSTM (+10 f) |
|---|---|---|---|
| IMUHOI++ pose prior b_t | 0.922 [0.897, 0.945] | +0.051 [+0.016, +0.085] | +0.059 [+0.029, +0.092] |
| IMUHOI++ hand-contact LSTM | 0.808 [0.771, 0.842] | -0.064 [-0.107, -0.023] | -0.055 [-0.097, -0.017] |
| IMUHOI++ object-motion LSTM | 0.950 [0.929, 0.969] | +0.078 [+0.046, +0.110] | +0.087 [+0.055, +0.117] |
| IMUHOI contact LSTM | 0.871 [0.843, 0.899] | - | +0.008 [-0.017, +0.034] |
| IMUHOI contact LSTM, +10 f horizon | 0.863 [0.835, 0.890] | -0.008 [-0.034, +0.017] | - |
| input: HP pose stream | 0.905 [0.880, 0.931] | +0.035 [+0.000, +0.072] | +0.043 [+0.010, +0.076] |
| input: coupling features | 0.919 [0.895, 0.943] | +0.048 [+0.012, +0.085] | +0.056 [+0.022, +0.089] |
| input: forearm/object IMU + vel. | 0.869 [0.836, 0.900] | -0.002 [-0.041, +0.036] | +0.006 [-0.028, +0.041] |
| control: untrained b_t | 0.890 [0.862, 0.919] | +0.020 [-0.015, +0.058] | +0.029 [-0.004, +0.062] |

b_t minus untrained b_t: +0.031 [+0.010, +0.052]; b_t minus HP pose input: +0.016 [-0.005, +0.038]

## Time-resolved decoding (balanced accuracy)

n = {'segments': 845, 'lift': 633, 'ground': 212, 'sequences': 827, 'subjects': 17}

| tau | IMUHOI++ pose prior b_t | IMUHOI++ hand-contact LSTM | IMUHOI contact LSTM | IMUHOI contact LSTM, +10 f horizon | input: HP pose stream | input: forearm/object IMU + vel. | control: untrained b_t | b_t minus IMUHOI LSTM (+10 f) | b_t minus untrained b_t | segs in warmup |
|---|---|---|---|---|---|---|---|---|---|---|
| +0 | 0.859 | 0.668 | 0.751 | 0.882 | 0.778 | 0.797 | 0.746 | -0.023 [-0.059, +0.013] | +0.113 [+0.076, +0.148] | 769 |
| +5 | 0.874 | 0.755 | 0.866 | 0.820 | 0.846 | 0.847 | 0.816 | +0.052 [+0.014, +0.088] | +0.057 [+0.024, +0.089] | 759 |
| +10 | 0.902 | 0.776 | 0.877 | 0.841 | 0.871 | 0.864 | 0.849 | +0.061 [+0.023, +0.097] | +0.054 [+0.025, +0.081] | 753 |
| +15 | 0.894 | 0.791 | 0.820 | 0.879 | 0.883 | 0.882 | 0.868 | +0.016 [-0.018, +0.051] | +0.028 [+0.003, +0.053] | 731 |
| +20 | 0.910 | 0.768 | 0.841 | 0.867 | 0.891 | 0.881 | 0.873 | +0.043 [+0.011, +0.073] | +0.037 [+0.015, +0.061] | 707 |
| +25 | 0.908 | 0.758 | 0.879 | 0.849 | 0.898 | 0.879 | 0.880 | +0.059 [+0.024, +0.092] | +0.028 [+0.007, +0.050] | 676 |
| +30 | 0.904 | 0.731 | 0.867 | 0.853 | 0.891 | 0.870 | 0.886 | +0.052 [+0.018, +0.087] | +0.018 [-0.006, +0.044] | 633 |
| +35 | 0.915 | 0.704 | 0.849 | 0.838 | 0.882 | 0.867 | 0.879 | +0.078 [+0.045, +0.109] | +0.037 [+0.013, +0.060] | 557 |
| +40 | 0.888 | 0.731 | 0.853 | 0.807 | 0.883 | 0.863 | 0.881 | +0.081 [+0.041, +0.122] | +0.007 [-0.015, +0.030] | 481 |
| +45 | 0.894 | 0.718 | 0.838 | 0.795 | 0.884 | 0.864 | 0.873 | +0.099 [+0.058, +0.138] | +0.021 [+0.000, +0.044] | 406 |
| +50 | 0.887 | 0.714 | 0.807 | 0.773 | 0.900 | 0.858 | 0.873 | +0.115 [+0.076, +0.155] | +0.015 [-0.009, +0.037] | 320 |
| +55 | 0.887 | 0.737 | 0.795 | 0.764 | 0.888 | 0.866 | 0.873 | +0.123 [+0.082, +0.164] | +0.014 [-0.009, +0.039] | 244 |
| +60 | 0.883 | 0.757 | 0.773 | 0.753 | 0.883 | 0.852 | 0.871 | +0.131 [+0.088, +0.171] | +0.013 [-0.013, +0.039] | 0 |

## Unsupervised structure of onset-window embeddings

n = {'segments': 1101, 'ground': 214, 'lift': 635, 'tilt': 252, 'objects': 15, 'subjects': 17}

| representation | kNN purity (chance) | k-means AMI mode / object / subject | RSA mode / object / subject | RSA mode, partial |
|---|---|---|---|---|
| IMUHOI++ pose prior b_t | 0.589 (0.331) | 0.032 / 0.143 / 0.067 | 0.192 / 0.032 / 0.002 | 0.191 |
| IMUHOI++ hand-contact LSTM | 0.461 (0.331) | 0.024 / 0.058 / 0.038 | 0.093 / 0.028 / -0.002 | 0.092 |
| IMUHOI++ object-motion LSTM | 0.550 (0.331) | 0.141 / 0.064 / 0.050 | 0.269 / 0.036 / 0.022 | 0.268 |
| IMUHOI contact LSTM | 0.513 (0.331) | 0.028 / 0.084 / 0.072 | 0.171 / 0.044 / 0.013 | 0.170 |
| IMUHOI contact LSTM, +10 f horizon | 0.524 (0.331) | 0.046 / 0.086 / 0.066 | 0.213 / 0.038 / 0.007 | 0.212 |
| input: HP pose stream | 0.594 (0.331) | 0.280 / 0.163 / 0.066 | 0.170 / 0.099 / 0.054 | 0.168 |
| input: coupling features | 0.534 (0.331) | 0.075 / 0.079 / 0.059 | 0.067 / 0.028 / 0.035 | 0.067 |
| input: forearm/object IMU + vel. | 0.539 (0.331) | 0.111 / 0.118 / 0.056 | 0.078 / 0.079 / 0.017 | 0.076 |
| control: untrained b_t | 0.553 (0.331) | 0.004 / 0.122 / 0.052 | 0.193 / 0.057 / 0.031 | 0.192 |
