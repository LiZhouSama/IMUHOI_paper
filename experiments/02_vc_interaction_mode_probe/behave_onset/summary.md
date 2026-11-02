# Onset imminence: BEHAVE test

59 sequences; 39938 non-contact hand-frames on sliding-window frames (30000 idle = no onset of that hand within 30 frames, subsampled). Published frame t has seen inputs up to t+10: leads 1-10 are inside the lookahead.

## AUC against idle frames (95% CI, sequence bootstrap)

| score | lead 1-10 (onset visible) | lead 11-20 | lead 21-30 |
|---|---|---|---|
| IMUHOI++ anticipation head | 0.823 [0.785, 0.852] | 0.740 [0.706, 0.774] | 0.665 [0.624, 0.707] |
| IMUHOI++ published contact prob. | 0.804 [0.753, 0.852] | 0.684 [0.618, 0.744] | 0.677 [0.616, 0.732] |
| IMUHOI published contact prob. | 0.704 [0.651, 0.759] | 0.645 [0.586, 0.704] | 0.645 [0.587, 0.709] |
| IMUHOI contact prob. at t+10 | 0.850 [0.809, 0.893] | 0.716 [0.663, 0.769] | 0.657 [0.600, 0.713] |
| probe: IMUHOI++ pose prior b_t | 0.798 [0.755, 0.838] | 0.755 [0.709, 0.796] | 0.697 [0.652, 0.734] |
| probe: IMUHOI++ hand-contact LSTM | 0.757 [0.717, 0.797] | 0.709 [0.668, 0.743] | 0.662 [0.609, 0.701] |
| probe: IMUHOI contact LSTM | 0.701 [0.654, 0.747] | 0.660 [0.609, 0.711] | 0.645 [0.589, 0.697] |
| probe: IMUHOI contact LSTM at t+10 | 0.867 [0.832, 0.898] | 0.718 [0.671, 0.763] | 0.631 [0.583, 0.684] |
| probe: HP pose input | 0.771 [0.724, 0.810] | 0.736 [0.692, 0.778] | 0.689 [0.635, 0.734] |
| probe: untrained b_t | 0.750 [0.703, 0.788] | 0.710 [0.669, 0.753] | 0.657 [0.615, 0.696] |

Same, restricted to frames where the other hand is not in contact either (no cue from a hand that already holds the object):

| score | lead 1-10 | lead 11-20 | lead 21-30 |
|---|---|---|---|
| IMUHOI++ anticipation head | 0.825 [0.781, 0.863] | 0.755 [0.713, 0.798] | 0.706 [0.657, 0.755] |
| IMUHOI++ published contact prob. | 0.831 [0.769, 0.882] | 0.710 [0.637, 0.772] | 0.695 [0.627, 0.754] |
| IMUHOI published contact prob. | 0.733 [0.653, 0.795] | 0.676 [0.600, 0.740] | 0.672 [0.604, 0.743] |
| IMUHOI contact prob. at t+10 | 0.880 [0.831, 0.922] | 0.745 [0.664, 0.805] | 0.685 [0.604, 0.748] |
| probe: IMUHOI++ pose prior b_t | 0.785 [0.723, 0.827] | 0.765 [0.707, 0.809] | 0.736 [0.680, 0.779] |
| probe: IMUHOI++ hand-contact LSTM | 0.736 [0.684, 0.775] | 0.726 [0.665, 0.772] | 0.688 [0.612, 0.736] |
| probe: IMUHOI contact LSTM | 0.726 [0.656, 0.782] | 0.680 [0.618, 0.740] | 0.659 [0.589, 0.721] |
| probe: IMUHOI contact LSTM at t+10 | 0.887 [0.852, 0.920] | 0.739 [0.674, 0.794] | 0.640 [0.577, 0.705] |
| probe: HP pose input | 0.780 [0.714, 0.824] | 0.780 [0.714, 0.837] | 0.744 [0.675, 0.795] |
| probe: untrained b_t | 0.761 [0.700, 0.800] | 0.746 [0.688, 0.801] | 0.718 [0.663, 0.760] |

Fresh approach: both hands free and no contact of either hand in the previous 30 frames (positives and idle alike; {'positives_1_10': 1591, 'positives_11_20': 1408, 'idle': 18434}):

| score | lead 1-10 | lead 11-20 | lead 21-30 |
|---|---|---|---|
| IMUHOI++ anticipation head | 0.834 [0.786, 0.876] | 0.771 [0.723, 0.812] | 0.690 [0.635, 0.757] |
| IMUHOI++ published contact prob. | 0.863 [0.800, 0.912] | 0.721 [0.647, 0.780] | 0.664 [0.581, 0.733] |
| IMUHOI published contact prob. | 0.769 [0.686, 0.832] | 0.681 [0.598, 0.748] | 0.632 [0.549, 0.711] |
| IMUHOI contact prob. at t+10 | 0.899 [0.853, 0.940] | 0.755 [0.670, 0.820] | 0.653 [0.576, 0.723] |
| probe: IMUHOI++ pose prior b_t | 0.804 [0.734, 0.847] | 0.788 [0.717, 0.845] | 0.721 [0.661, 0.776] |
| probe: IMUHOI++ hand-contact LSTM | 0.752 [0.694, 0.797] | 0.725 [0.652, 0.772] | 0.639 [0.561, 0.693] |
| probe: IMUHOI contact LSTM | 0.756 [0.691, 0.809] | 0.679 [0.614, 0.749] | 0.625 [0.555, 0.697] |
| probe: IMUHOI contact LSTM at t+10 | 0.899 [0.867, 0.929] | 0.752 [0.679, 0.810] | 0.620 [0.540, 0.692] |
| probe: HP pose input | 0.794 [0.720, 0.841] | 0.789 [0.714, 0.847] | 0.739 [0.661, 0.797] |
| probe: untrained b_t | 0.776 [0.706, 0.823] | 0.753 [0.676, 0.810] | 0.712 [0.644, 0.761] |

Locomotion: share of walking frames (root speed >= 0.3 m/s) among fresh positives at lead 11-30 0.38, among fresh idle frames 0.27.

Fresh approach within locomotion strata (AUC [95% CI]):

| task | n positives / idle | IMUHOI++ anticipation head | probe: IMUHOI++ pose prior b_t | probe: HP pose input | probe: untrained b_t | probe: IMUHOI contact LSTM at t+10 | IMUHOI contact prob. at t+10 |
|---|---|---|---|---|---|---|---|
| fresh_11_20_moving | 526 / 4962 | 0.766 [0.670, 0.842] | 0.779 [0.683, 0.851] | 0.787 [0.669, 0.849] | 0.739 [0.632, 0.797] | 0.687 [0.584, 0.773] | 0.680 [0.570, 0.793] |
| fresh_11_20_still | 882 / 13472 | 0.777 [0.731, 0.829] | 0.792 [0.717, 0.860] | 0.785 [0.700, 0.859] | 0.754 [0.659, 0.830] | 0.775 [0.694, 0.842] | 0.781 [0.694, 0.834] |
| fresh_21_30_moving | 448 / 4962 | 0.684 [0.608, 0.770] | 0.701 [0.624, 0.772] | 0.743 [0.661, 0.810] | 0.686 [0.600, 0.749] | 0.538 [0.423, 0.647] | 0.547 [0.420, 0.669] |
| fresh_21_30_still | 677 / 13472 | 0.700 [0.626, 0.772] | 0.731 [0.649, 0.802] | 0.730 [0.633, 0.812] | 0.715 [0.620, 0.792] | 0.651 [0.548, 0.729] | 0.692 [0.590, 0.769] |

Paired AUC differences on fresh approaches (sequence bootstrap):

| task | difference | AUC difference [95% CI] |
|---|---|---|
| fresh_11_20 | probe: IMUHOI++ pose prior b_t minus probe: HP pose input | -0.002 [-0.039, +0.032] |
| fresh_11_20 | probe: IMUHOI++ pose prior b_t minus probe: untrained b_t | +0.034 [-0.004, +0.071] |
| fresh_11_20 | probe: IMUHOI++ pose prior b_t minus probe: IMUHOI contact LSTM at t+10 | +0.036 [-0.031, +0.092] |
| fresh_11_20 | IMUHOI++ anticipation head minus IMUHOI contact prob. at t+10 | +0.015 [-0.059, +0.098] |
| fresh_21_30 | probe: IMUHOI++ pose prior b_t minus probe: HP pose input | -0.019 [-0.051, +0.017] |
| fresh_21_30 | probe: IMUHOI++ pose prior b_t minus probe: untrained b_t | +0.009 [-0.032, +0.042] |
| fresh_21_30 | probe: IMUHOI++ pose prior b_t minus probe: IMUHOI contact LSTM at t+10 | +0.101 [+0.035, +0.162] |
| fresh_21_30 | IMUHOI++ anticipation head minus IMUHOI contact prob. at t+10 | +0.037 [-0.035, +0.109] |

## Detection lead at 5% false alarms on idle frames (329 onsets with 30 idle frames before)

| score | threshold | detected by the onset | >= 1 f early | >= 5 f early | >= 10 f early | >= 11 f early | median lead |
|---|---|---|---|---|---|---|---|
| IMUHOI++ anticipation head | 0.212 | 0.36 | 0.33 | 0.22 | 0.13 | 0.11 | -10 |
| IMUHOI++ published contact prob. | 0.876 | 0.27 | 0.22 | 0.12 | 0.07 | 0.07 | -10 |
| IMUHOI published contact prob. | 0.902 | 0.14 | 0.11 | 0.06 | 0.03 | 0.03 | -99 |
| IMUHOI contact prob. at t+10 | 0.889 | 0.41 | 0.37 | 0.25 | 0.09 | 0.09 | -4 |

Lead >= 11 frames = the score was above threshold before the published window could contain the onset.
Negative median lead = detected after the onset (frames).
