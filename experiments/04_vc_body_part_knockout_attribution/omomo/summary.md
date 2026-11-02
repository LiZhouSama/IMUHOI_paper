# A2 body-part knockouts and attribution: OMOMO test

1173 sequences, IMUHOI++ pipeline (VC-v5ar + OT v2_armrot_biclose), 7/3 protocol.
Knockout = input replaced by its mean over all VC forward calls of the reference run. Event F1: tolerance
10 frames, threshold 0.5, sliding-window events only (as A1). Deltas: knockout minus reference with a
paired sequence-bootstrap 95% CI.

## Knockouts: contact F1 per hand

| knockout | frame F1 L | frame F1 R | onset F1 L | onset F1 R | offset F1 L | offset F1 R |
|---|---|---|---|---|---|---|
| reference | 0.955 | 0.958 | 0.381 | 0.328 | 0.697 | 0.711 |
| left arm | -0.098 [-0.109, -0.088] | -0.010 [-0.013, -0.007] | -0.246 [-0.312, -0.181] | -0.029 [-0.075, +0.012] | -0.253 [-0.292, -0.215] | -0.018 [-0.051, +0.014] |
| right arm | -0.009 [-0.013, -0.005] | -0.051 [-0.058, -0.045] | -0.013 [-0.065, +0.041] | -0.195 [-0.261, -0.135] | -0.015 [-0.047, +0.017] | -0.212 [-0.250, -0.170] |
| torso | -0.005 [-0.007, -0.003] | -0.007 [-0.008, -0.005] | -0.041 [-0.093, +0.013] | -0.021 [-0.059, +0.017] | -0.036 [-0.070, -0.002] | -0.020 [-0.047, +0.009] |
| both legs | -0.016 [-0.020, -0.012] | -0.020 [-0.024, -0.015] | -0.087 [-0.143, -0.033] | -0.036 [-0.087, +0.014] | -0.031 [-0.065, +0.002] | -0.019 [-0.050, +0.010] |
| all groups (b_t) | -0.199 [-0.217, -0.182] | -0.086 [-0.095, -0.078] | -0.302 [-0.364, -0.237] | -0.129 [-0.190, -0.068] | -0.265 [-0.309, -0.223] | -0.148 [-0.191, -0.100] |
| all HP inputs | -0.404 [-0.434, -0.375] | -0.180 [-0.197, -0.163] | -0.323 [-0.392, -0.252] | -0.172 [-0.247, -0.100] | -0.387 [-0.441, -0.329] | -0.200 [-0.254, -0.146] |

Arm knockouts, change of the ipsilateral hand minus change of the contralateral hand:

| knockout | frame F1 | onset F1 | offset F1 |
|---|---|---|---|
| left arm | -0.088 [-0.100, -0.077] | -0.217 [-0.298, -0.134] | -0.235 [-0.283, -0.188] |
| right arm | -0.042 [-0.050, -0.035] | -0.183 [-0.268, -0.107] | -0.197 [-0.240, -0.149] |

## Knockouts: object and HOI error (eval JSON)

| knockout | Object (cm) | HOI (cm) | F1 obj motion |
|---|---|---|---|
| reference | 13.34 | 10.48 | 0.981 |
| left arm | +1.59 | +1.19 | -0.001 |
| right arm | +2.58 | +1.29 | -0.001 |
| torso | +0.26 | +0.11 | -0.000 |
| both legs | -0.16 | -0.04 | -0.004 |
| all groups (b_t) | +3.04 | +1.45 | -0.003 |
| all HP inputs | +2.31 | +1.78 | -0.003 |

## Integrated gradients

685 windows at GT hand events; 536 used below. Left out: 144 where the whole pose pathway changes the published probability by < 0.05, 5 where the IG sum misses that change by > 10%.

| events | n | mean dp (pose vs mean pose) | left arm | right arm | torso | left leg | right leg | ipsilateral minus contralateral arm |
|---|---|---|---|---|---|---|---|---|
| left onset | 49 | +0.365 | 0.276 | 0.207 | 0.200 | 0.160 | 0.158 | +0.069 [+0.049, +0.091] |
| right onset | 37 | +0.292 | 0.250 | 0.251 | 0.184 | 0.160 | 0.155 | +0.001 [-0.032, +0.033] |
| left offset | 221 | +0.016 | 0.237 | 0.223 | 0.203 | 0.168 | 0.169 | +0.014 [+0.003, +0.025] |
| right offset | 229 | +0.006 | 0.225 | 0.273 | 0.177 | 0.164 | 0.161 | +0.048 [+0.036, +0.061] |

| events | |IG| share before the event frame | before -10 (window cannot see the event) | uniform share before the event |
|---|---|---|---|
| onset | 0.660 [0.642, 0.677] | 0.416 [0.396, 0.438] | 0.843 |
| offset | 0.638 [0.631, 0.647] | 0.356 [0.347, 0.366] | 0.843 |

| mode | n | left arm | right arm | torso | left leg | right leg |
|---|---|---|---|---|---|---|
| lift | 319 | 0.240 [0.232, 0.249] | 0.250 [0.243, 0.258] | 0.190 [0.184, 0.198] | 0.159 [0.154, 0.164] | 0.160 [0.155, 0.166] |
| ground | 113 | 0.221 [0.210, 0.232] | 0.233 [0.221, 0.245] | 0.195 [0.186, 0.204] | 0.177 [0.168, 0.186] | 0.174 [0.164, 0.184] |
| tilt | 102 | 0.239 [0.229, 0.251] | 0.240 [0.228, 0.253] | 0.185 [0.174, 0.196] | 0.170 [0.162, 0.176] | 0.166 [0.158, 0.173] |
