# A2 body-part knockouts and attribution: BEHAVE test

59 sequences, IMUHOI++ pipeline (VC-v5ar + OT v2_armrot_biclose), 7/3 protocol.
Knockout = input replaced by its mean over all VC forward calls of the reference run. Event F1: tolerance
10 frames, threshold 0.5, sliding-window events only (as A1). Deltas: knockout minus reference with a
paired sequence-bootstrap 95% CI.

## Knockouts: contact F1 per hand

| knockout | frame F1 L | frame F1 R | onset F1 L | onset F1 R | offset F1 L | offset F1 R |
|---|---|---|---|---|---|---|
| reference | 0.873 | 0.890 | 0.270 | 0.289 | 0.309 | 0.348 |
| left arm | -0.086 [-0.110, -0.066] | -0.039 [-0.062, -0.018] | -0.119 [-0.179, -0.073] | -0.053 [-0.093, -0.015] | -0.128 [-0.193, -0.079] | -0.045 [-0.089, +0.001] |
| right arm | -0.039 [-0.055, -0.023] | -0.071 [-0.099, -0.048] | -0.063 [-0.104, -0.024] | -0.125 [-0.174, -0.081] | -0.069 [-0.116, -0.031] | -0.156 [-0.211, -0.111] |
| torso | -0.004 [-0.012, +0.004] | -0.011 [-0.021, -0.002] | -0.011 [-0.047, +0.019] | -0.016 [-0.058, +0.027] | -0.008 [-0.041, +0.023] | -0.042 [-0.078, -0.001] |
| both legs | -0.007 [-0.024, +0.006] | -0.005 [-0.014, +0.004] | -0.016 [-0.052, +0.015] | -0.008 [-0.049, +0.034] | -0.019 [-0.060, +0.015] | -0.017 [-0.057, +0.018] |
| all groups (b_t) | -0.079 [-0.103, -0.053] | -0.090 [-0.119, -0.065] | -0.098 [-0.154, -0.051] | -0.143 [-0.193, -0.100] | -0.118 [-0.188, -0.066] | -0.137 [-0.193, -0.088] |
| all HP inputs | -0.095 [-0.122, -0.066] | -0.126 [-0.161, -0.097] | -0.089 [-0.151, -0.031] | -0.178 [-0.243, -0.124] | -0.114 [-0.181, -0.057] | -0.193 [-0.251, -0.141] |

Arm knockouts, change of the ipsilateral hand minus change of the contralateral hand:

| knockout | frame F1 | onset F1 | offset F1 |
|---|---|---|---|
| left arm | -0.047 [-0.070, -0.025] | -0.065 [-0.125, -0.019] | -0.083 [-0.156, -0.026] |
| right arm | -0.032 [-0.063, -0.006] | -0.062 [-0.107, -0.020] | -0.087 [-0.140, -0.028] |

## Knockouts: object and HOI error (eval JSON)

| knockout | Object (cm) | HOI (cm) | F1 obj motion |
|---|---|---|---|
| reference | 22.93 | 17.19 | 0.922 |
| left arm | +1.59 | +1.80 | +0.003 |
| right arm | +1.21 | +1.27 | -0.002 |
| torso | +0.36 | +0.12 | -0.003 |
| both legs | +1.06 | +0.60 | -0.011 |
| all groups (b_t) | +4.76 | +3.48 | -0.009 |
| all HP inputs | +5.36 | +4.58 | -0.009 |

## Integrated gradients

653 windows at GT hand events; 434 used below. Left out: 195 where the whole pose pathway changes the published probability by < 0.05, 24 where the IG sum misses that change by > 10%.

| events | n | mean dp (pose vs mean pose) | left arm | right arm | torso | left leg | right leg | ipsilateral minus contralateral arm |
|---|---|---|---|---|---|---|---|---|
| left onset | 120 | -0.237 | 0.263 | 0.253 | 0.181 | 0.157 | 0.146 | +0.010 [-0.005, +0.022] |
| right onset | 115 | -0.323 | 0.233 | 0.257 | 0.185 | 0.164 | 0.161 | +0.024 [+0.010, +0.038] |
| left offset | 100 | -0.193 | 0.233 | 0.263 | 0.184 | 0.170 | 0.150 | -0.031 [-0.049, -0.013] |
| right offset | 99 | -0.334 | 0.217 | 0.256 | 0.205 | 0.168 | 0.154 | +0.039 [+0.025, +0.052] |

| events | |IG| share before the event frame | before -10 (window cannot see the event) | uniform share before the event |
|---|---|---|---|
| onset | 0.755 [0.747, 0.762] | 0.535 [0.523, 0.548] | 0.843 |
| offset | 0.690 [0.674, 0.705] | 0.441 [0.419, 0.460] | 0.843 |

| mode | n | left arm | right arm | torso | left leg | right leg |
|---|---|---|---|---|---|---|
| lift | 186 | 0.246 [0.236, 0.257] | 0.261 [0.249, 0.274] | 0.192 [0.177, 0.207] | 0.156 [0.149, 0.162] | 0.145 [0.139, 0.151] |
| ground | 141 | 0.223 [0.208, 0.235] | 0.252 [0.236, 0.269] | 0.187 [0.178, 0.200] | 0.175 [0.162, 0.190] | 0.164 [0.151, 0.176] |
| tilt | 51 | 0.249 [0.240, 0.266] | 0.264 [0.252, 0.279] | 0.174 [0.159, 0.192] | 0.158 [0.147, 0.167] | 0.155 [0.145, 0.161] |
