# Pose prototypes of interaction states: BEHAVE test

Frames per state available on sliding-window frames: {'idle': 23413, 'approach': 2500, 'pre-contact': 1566, 'contact-start': 2421, 'holding': 49145, 'post-release': 2329}; at most 3000 sampled per state; k-means k = 8 on PCA-32.

## Alignment of the clusters with the interaction states

| representation | AMI with the six states |
|---|---|
| IMUHOI++ pose prior b_t | 0.093 |
| HP pose input | 0.060 |
| untrained b_t | 0.021 |
| IMUHOI++ hand-contact LSTM | 0.120 |
| IMUHOI++ object-motion LSTM | 0.233 |
| IMUHOI contact LSTM | 0.182 |

## Clusters of IMUHOI++ pose prior b_t (state shares, network outputs)

| cluster | n | idle | approach | pre-contact | contact-start | holding | post-release | published p(contact) | anticipation |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2913 | 0.07 | 0.16 | 0.15 | 0.25 | 0.13 | 0.24 | 0.45 | 0.212 |
| 2 | 1554 | 0.09 | 0.20 | 0.12 | 0.23 | 0.15 | 0.21 | 0.39 | 0.170 |
| 5 | 2359 | 0.11 | 0.23 | 0.13 | 0.21 | 0.10 | 0.22 | 0.32 | 0.173 |
| 7 | 1639 | 0.14 | 0.14 | 0.12 | 0.18 | 0.22 | 0.19 | 0.52 | 0.173 |
| 6 | 965 | 0.18 | 0.15 | 0.12 | 0.16 | 0.29 | 0.09 | 0.50 | 0.126 |
| 0 | 2011 | 0.33 | 0.22 | 0.07 | 0.07 | 0.25 | 0.07 | 0.34 | 0.113 |
| 4 | 2030 | 0.61 | 0.14 | 0.06 | 0.07 | 0.03 | 0.09 | 0.03 | 0.110 |
| 3 | 1345 | 0.07 | 0.06 | 0.05 | 0.08 | 0.69 | 0.06 | 0.85 | 0.107 |

## Clusters of HP pose input (state shares, network outputs)

| cluster | n | idle | approach | pre-contact | contact-start | holding | post-release | published p(contact) | anticipation |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 1787 | 0.06 | 0.11 | 0.11 | 0.31 | 0.10 | 0.31 | 0.46 | 0.189 |
| 2 | 2157 | 0.07 | 0.26 | 0.20 | 0.20 | 0.15 | 0.13 | 0.37 | 0.211 |
| 6 | 1117 | 0.11 | 0.18 | 0.19 | 0.20 | 0.09 | 0.23 | 0.27 | 0.182 |
| 1 | 2193 | 0.21 | 0.14 | 0.09 | 0.18 | 0.22 | 0.17 | 0.42 | 0.143 |
| 4 | 2220 | 0.22 | 0.19 | 0.10 | 0.15 | 0.22 | 0.11 | 0.40 | 0.139 |
| 7 | 2052 | 0.24 | 0.20 | 0.07 | 0.11 | 0.25 | 0.14 | 0.40 | 0.135 |
| 3 | 2637 | 0.24 | 0.14 | 0.07 | 0.09 | 0.34 | 0.12 | 0.50 | 0.128 |
| 0 | 653 | 0.84 | 0.06 | 0.01 | 0.02 | 0.04 | 0.04 | 0.03 | 0.090 |

## Top 1% anticipation frames (non-contact)

threshold 0.302, 299 frames; followed by an onset within 15 frames: 0.43 (base rate over all non-contact frames 0.08).

| pose group | n | onset within 15 f |
|---|---|---|
| 0 | 129 | 0.47 |
| 1 | 82 | 0.23 |
| 2 | 60 | 0.75 |
| 3 | 28 | 0.14 |
