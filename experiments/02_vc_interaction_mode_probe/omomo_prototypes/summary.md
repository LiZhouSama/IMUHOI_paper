# Pose prototypes of interaction states: OMOMO test

Frames per state available on sliding-window frames: {'idle': 9946, 'approach': 959, 'pre-contact': 858, 'contact-start': 1764, 'holding': 83281, 'post-release': 5842}; at most 3000 sampled per state; k-means k = 8 on PCA-32.

## Alignment of the clusters with the interaction states

| representation | AMI with the six states |
|---|---|
| IMUHOI++ pose prior b_t | 0.337 |
| HP pose input | 0.175 |
| untrained b_t | 0.095 |
| IMUHOI++ hand-contact LSTM | 0.357 |
| IMUHOI++ object-motion LSTM | 0.280 |
| IMUHOI contact LSTM | 0.303 |

## Clusters of IMUHOI++ pose prior b_t (state shares, network outputs)

| cluster | n | idle | approach | pre-contact | contact-start | holding | post-release | published p(contact) | anticipation |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 665 | 0.01 | 0.00 | 0.02 | 0.82 | 0.13 | 0.01 | 0.95 | 0.148 |
| 4 | 1326 | 0.03 | 0.26 | 0.31 | 0.15 | 0.20 | 0.05 | 0.32 | 0.242 |
| 1 | 1338 | 0.17 | 0.10 | 0.10 | 0.22 | 0.30 | 0.10 | 0.54 | 0.096 |
| 7 | 1206 | 0.05 | 0.06 | 0.07 | 0.20 | 0.45 | 0.17 | 0.68 | 0.076 |
| 5 | 1623 | 0.04 | 0.04 | 0.08 | 0.18 | 0.15 | 0.52 | 0.37 | 0.105 |
| 6 | 1781 | 0.02 | 0.04 | 0.03 | 0.09 | 0.78 | 0.04 | 0.90 | 0.083 |
| 3 | 2584 | 0.85 | 0.08 | 0.01 | 0.01 | 0.02 | 0.02 | 0.03 | 0.059 |
| 0 | 2058 | 0.17 | 0.02 | 0.00 | 0.00 | 0.01 | 0.79 | 0.03 | 0.032 |

## Clusters of HP pose input (state shares, network outputs)

| cluster | n | idle | approach | pre-contact | contact-start | holding | post-release | published p(contact) | anticipation |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1383 | 0.02 | 0.11 | 0.30 | 0.23 | 0.19 | 0.15 | 0.39 | 0.216 |
| 7 | 602 | 0.03 | 0.06 | 0.12 | 0.35 | 0.17 | 0.27 | 0.59 | 0.129 |
| 3 | 1469 | 0.04 | 0.22 | 0.10 | 0.21 | 0.28 | 0.14 | 0.49 | 0.135 |
| 1 | 1970 | 0.04 | 0.01 | 0.02 | 0.27 | 0.07 | 0.58 | 0.36 | 0.067 |
| 5 | 1688 | 0.34 | 0.05 | 0.03 | 0.08 | 0.12 | 0.38 | 0.24 | 0.052 |
| 0 | 2068 | 0.52 | 0.07 | 0.03 | 0.06 | 0.24 | 0.09 | 0.30 | 0.071 |
| 6 | 2104 | 0.54 | 0.08 | 0.02 | 0.04 | 0.22 | 0.10 | 0.28 | 0.066 |
| 4 | 1297 | 0.04 | 0.02 | 0.02 | 0.03 | 0.71 | 0.18 | 0.76 | 0.074 |

## Top 1% anticipation frames (non-contact)

threshold 0.376, 177 frames; followed by an onset within 15 frames: 0.93 (base rate over all non-contact frames 0.07).

| pose group | n | onset within 15 f |
|---|---|---|
| 0 | 30 | 1.00 |
| 1 | 48 | 1.00 |
| 2 | 83 | 0.90 |
| 3 | 16 | 0.69 |
