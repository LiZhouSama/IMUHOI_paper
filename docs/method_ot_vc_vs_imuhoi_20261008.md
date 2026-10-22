# Velocity–Contact and Object-Translation Modules: Method Improvements over IMU-HOI

> **Status (2026-10-08).** Draft of the Method text for the journal extension.
> Described configurations: velocity–contact **VC-v5ar** (`velocity_contact_arch: v5`, `v5_init_noise_prob: 0.5`) and object translation **OT-v2 + forearm–object rotation + bimanual closure** (`object_trans_arch: v2`, `object_trans_forearm_obj_rotation: true`, `object_trans_bimanual_closure: true`; run name `v2_armrot_biclose`).
> Baselines: the IMU-HOI velocity–contact module (`model/rnn/velocity_contact_IMUHOI.py`) and the IMU-HOI object-translation module in our four-candidate re-implementation (`model/rnn/object_trans_IMUHOI_4heads.py`).
> Sections 3.1–3.4 are written as paper text. Section 3.5 and Appendices A–D are working material (results, code mapping, evidence behind each design choice, open issues) and are not meant to be pasted into the Method section as they are.

## 3.1 Overview

**Notation.** At frame $t$ the system observes six body IMUs $\mathbf X^{\mathrm{hum}}_t\in\mathbb R^{6\times 9}$ (pelvis, both lower legs, head and both forearms; acceleration and 6-D orientation each) and one object IMU with free acceleration $\mathbf a_O(t)$ and orientation $R_O(t)\equiv{}^{W}\!R_O(t)$. Starting from the initial state $\mathcal S_0$ defined in IMU-HOI, the model estimates the body pose, the root trajectory and the object position $\hat{\mathbf p}_O(t)$. For $s\in\{L,R\}$, $R_{F,s}(t)$ and $\mathbf a_{F,s}(t)$ denote the orientation and acceleration of the forearm IMU, and $\hat{\mathbf h}_s(t)$ the palm position estimated by the human-pose module; $\Delta t=1/30$ s. We call the estimates already output for earlier frames the *published stream*. Both redesigned modules rely on the forearm–object relative rotation

$$
B_s(t)=R_O(t)^{\top}R_{F,s}(t),
\tag{1}
$$

which is observable from the IMUs alone and invariant to a global rotation of the scene.

**Pipeline.** Like IMU-HOI, the model consists of a human-pose module (HP), a velocity–contact module (VC) and an object-translation module (OT), corresponding to Stages II, I and III of IMU-HOI. We keep HP and redesign VC and OT. Two changes affect the pipeline as a whole. First, VC runs after HP and is conditioned on its output (Sec. 3.2), whereas the contact estimator of IMU-HOI observes inertial signals only. Second, OT is formulated as a closed-loop estimator whose recurrence reads its own published object state (Sec. 3.3), and every module is trained with the fixed-lag sliding-window inference in mind, in which recurrent initializers come from the published stream rather than from the ground truth (Sec. 3.4). Table 1 summarizes the differences.

**Table 1.** Differences between IMU-HOI and the proposed VC/OT modules.

| Aspect | IMU-HOI | This work |
|---|---|---|
| Module order | VC independent of HP | VC conditioned on the HP output |
| Contact output | hand-contact (and object-motion) probabilities predicted jointly | object motion times motion-conditioned hand contact, $p_s=p_m\,p_{s\mid m}$ (Eq. 2) |
| Contact evidence | forearm/object IMUs, predicted velocities | + supervised pose prior with endpoint and anticipation heads; + rotation-invariant hand–object coupling features |
| Velocity regressors | trained with exact (ground-truth) initializers | same architecture, trained with perturbed initializers |
| Hand anchor of OT | wrist | palm |
| Grasp-offset predictor | one open-loop LSTM per hand | one predictor shared by both hands, closed loop through state feedback |
| Grasp cues | object orientation, pelvis-relative forearm IMU | + forearm–object rotation $B_s$; + segment-length statistics |
| Candidates | left FK, right FK, inertial integration (+ static hold in our re-implementation) | + state-free bimanual-closure candidate |
| Fusion | learned gate biased by the contact prior | closed-form contact-driven weights, per-hand FK α–β tracking, IMU-certified static gate |
| Upstream inputs of OT during training | predicted | scheduled from ground truth to predicted |
| Joint fine-tuning | HP + OT (VC frozen) | VC + OT (HP frozen) |

## 3.2 Interaction-Aware Velocity–Contact Estimation

IMU-HOI regresses the hand (wrist) velocities $\hat{\mathbf v}_{H,s}$ from all body IMUs and the object velocity $\hat{\mathbf v}_O$ from the object IMU with two recurrent networks, and predicts the contact probabilities jointly from the forearm and object IMUs and the predicted velocities. We retain the two velocity regressors and rebuild the contact branch and the training procedure around three observations: (i) for object translation, a contact matters only when it moves the object; (ii) reaching and releasing are visible in the body motion before the object starts moving and after it stops; and (iii) during a grasp, the forearm and the object move as one quasi-rigid body.

### 3.2.1 Motion-conditioned contact factorization

Let $m_t\in\{0,1\}$ indicate that the object moves and $c_{s,t}\in\{0,1\}$ that hand $s$ is in contact with it. We factorize the hand-contact probability through object motion,

$$
p_s(t)=P(c_{s,t}=1)=p_m(t)\,p_{s\mid m}(t),\qquad
p_m(t)=P(m_t=1),\quad
p_{s\mid m}(t)=P(c_{s,t}=1\mid m_t=1).
\tag{2}
$$

An object-motion network predicts $p_m$ and a causal hand-contact network predicts $p_{s\mid m}$. The object-motion network is bidirectional within the window; because frame $t$ is published from a window that ends at $t+L$ (Sec. 3.4), it accesses at most $L$ future frames outside the initial warm-up block. The factorization lets the object IMU, which observes motion directly, gate every hand decision; the hand network only has to decide which hand moves an object that is known to move; and the triplet $(p_L,p_R,p_m)$ is exactly what the OT fusion consumes (Sec. 3.3.4). Motion labels are derived from the ground-truth object pose: after zero-phase low-pass filtering (6 Hz), a frame is labeled as moving if the net displacement over $[t-5,t+5]$ exceeds 1 cm or 3°, and moving runs shorter than 15 frames are discarded. The contact loss is

$$
\mathcal L_{\mathrm{con}}=
\mathrm{BCE}(p_m,m)
+\frac{\sum_{t,s} m_t\,\omega_{s,t}\,\mathrm{BCE}\big(p_{s\mid m}(t),c_{s,t}\big)}{\sum_{t,s} m_t\,\omega_{s,t}}
+\lambda_{\mathrm{mar}}\sum_{s}\mathrm{BCE}(p_s,c_s),
\tag{3}
$$

where the conditional term is restricted to moving frames, $\omega_{s,t}$ equals 1 for positives, 2 for negatives and 4 for hard negatives (frames on which the object moves while no hand touches it), and the marginal term ($\lambda_{\mathrm{mar}}=0.5$) calibrates the product of Eq. (2) that is passed downstream.

### 3.2.2 Pose-conditioned interaction prior

To exploit observation (ii), VC reads the output of HP. The body is divided into five joint groups $g\in\mathcal G$ (two arms, torso, two legs). For each group we collect the joint rotations (6-D), the joint velocities and accelerations obtained by finite differences of the estimated joint positions followed by a causal 5-frame moving average, and the root velocity. All quantities are expressed in the pelvis heading frame—the yaw component of the estimated pelvis orientation—and each modality is normalized separately. A GRU per group, followed by a bidirectional GRU over the window, produces boundary features $\mathbf b_t$ that condition both contact networks; the hand-contact network additionally receives the palm velocity and acceleration. Heading normalization removes the absolute walking direction, which the network could otherwise memorize, and per-modality normalization prevents the large acceleration channels from dominating the feature statistics.

A pose stream that is merely concatenated to strong inertial features can simply be ignored. We therefore attach two auxiliary heads that read $\mathbf b_t$ only. The endpoint head predicts six heatmaps $\hat{\mathbf e}_t\in[0,1]^6$ that mark the start and the end of left-hand, right-hand and object-motion segments. It is trained with a focal loss $\mathcal L_{\mathrm{end}}$ against Gaussian-smoothed endpoint targets, a soft ordering penalty $\mathcal L_{\mathrm{ord}}=\mathrm{ReLU}(\bar t_{\mathrm{start}}-\bar t_{\mathrm{end}})$ on the expected endpoint times $\bar t=\sum_t t\,\hat e_t/\sum_t \hat e_t$, and a consistency loss $\mathcal L_{\mathrm{ec}}$ that asks the contact state decoded from the endpoints to agree with the (detached) per-frame contact prediction. The anticipation head predicts, for each of the six channels, whether a transition occurs within the next $k=15$ frames (focal loss $\mathcal L_{\mathrm{ant}}$; frames whose horizon leaves the window are masked). Since body motion leads the contact labels, these targets force $\mathbf b_t$ to encode transition evidence, and their accuracy directly measures how much interaction information the pose stream carries.

### 3.2.3 Rotation-invariant hand–object coupling evidence

Observation (iii) is not directly accessible from world-frame IMU channels, so we add scalar evidence that measures the coupling. With the rotation speed $\omega[R](t)=\angle\big(R(t-1)^{\top}R(t)\big)/\Delta t$ and $\angle(Q)=\arccos\frac{\operatorname{tr}Q-1}{2}$, we compute for each hand

$$
\boldsymbol\gamma_s(t)=\Big[\,\omega[B_s](t),\ \ \omega[R_{F,s}](t),\ \ \big\lVert\mathbf a_{F,s}(t)-\mathbf a_O(t)\big\rVert\,\Big],
\tag{4}
$$

the object signals $\boldsymbol\gamma_O(t)=\big[\omega[R_O](t),\ \lVert\mathbf a_O(t)\rVert\big]$, and the bimanual signals $\boldsymbol\gamma_{LR}(t)=\big[d(t),\ |\dot d(t)|,\ \lVert\dot{\hat{\mathbf h}}_L(t)-\dot{\hat{\mathbf h}}_R(t)\rVert,\ \omega[R_{F,L}^{\top}R_{F,R}](t)\big]$, where $d=\lVert\hat{\mathbf h}_L-\hat{\mathbf h}_R\rVert$ is the palm span. During a firm grasp $\omega[B_s]\approx 0$ even if the forearm and the object rotate, the two accelerations agree up to lever-arm terms, and a bimanual hold keeps $d$ nearly constant. Every signal $x$ is summarized at two temporal scales and two compressions,

$$
\mathcal M[x](t)=\Big[\bar x_{[t-5,\,t+5]},\ \ \bar x_{[t-20,\,t+10]},\ \ \log\!\big(1+\bar x_{[t-5,\,t+5]}\big),\ \ \log\!\big(1+\bar x_{[t-20,\,t+10]}\big)\Big],
\tag{5}
$$

where $\bar x_I$ denotes the mean over the interval $I$, which never extends beyond the 10-frame lookahead of the online protocol. The hand-contact network receives $\mathcal M$ of all twelve signals (48-D) and the object-motion network that of $\boldsymbol\gamma_O$ (8-D); gradients are stopped at these features. All of them are scalars invariant to a global rotation and therefore cannot encode the absolute heading of a sequence. In a preliminary probe on BEHAVE, a gradient-boosted classifier using only these features reached a per-sequence hand-contact F1 of 0.80, compared with 0.78 for the pose-conditioned VC, and 0.84 when stacked on its prediction, which indicates that the two sources are complementary.

### 3.2.4 Initializer-robust training

Sliding-window inference re-initializes the recurrent velocity regressors at every window start with the velocities $\mathbf v^{\mathrm{init}}_O$ and $\mathbf v^{\mathrm{init}}_{H,s}$ of the preceding frame. During training these initializers are taken from the ground truth; at test time they come from the published stream (Sec. 3.4), and on long sequences their error (0.23–0.35 m/s on BEHAVE) is as large as the object speed itself. A regressor that has only seen exact initializers propagates most of that error to its output: the least-squares slope of the output error on the initializer error is about 0.6. We therefore perturb the initializers of a random half of the training windows,

$$
\tilde{\mathbf v}^{\mathrm{init}}=\mathbf v^{\mathrm{init}}+\boldsymbol\epsilon,\qquad
\boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\sigma^2\mathbf I),\quad \sigma=0.2\ \mathrm{m/s},
\tag{6}
$$

for both the object and the hand velocities, during VC training and during the joint stage (Sec. 3.4). The network learns to weigh the initializer against the inertial evidence: the slope drops to about 0.2, the online object-velocity error on BEHAVE decreases by 31–41 %, and the result becomes insensitive to how the initializer is obtained online.

### 3.2.5 Objective and optimization

The VC objective is

$$
\mathcal L_{\mathrm{VC}}=\mathcal L_{\mathrm{vel}}+\mathcal L_{\mathrm{con}}+\mathcal L_{\mathrm{end}}
+\tfrac12\big(\mathcal L_{\mathrm{ord}}+\mathcal L_{\mathrm{ec}}+\mathcal L_{\mathrm{ant}}\big),\qquad
\mathcal L_{\mathrm{vel}}=\big\lVert\hat{\mathbf v}_O-\mathbf v_O\big\rVert^2+\sum_s\big\lVert\hat{\mathbf v}_{H,s}-\mathbf v_{H,s}\big\rVert^2 .
\tag{7}
$$

It is optimized in two stages. The velocity regressors are trained first with $\mathcal L_{\mathrm{vel}}$ and then frozen at their best state, while the contact branch (object-motion, hand-contact, endpoint and anticipation heads) is trained with group-wise learning-rate schedules. VC is finally fine-tuned together with OT (Sec. 3.4).

## 3.3 Closed-Loop Object Translation

IMU-HOI recovers the object position under a quasi-rigid contact assumption: an independent recurrent head per hand predicts an object-frame unit direction ${}^{O}\hat{\mathbf u}_s$ and a reach length $\hat\ell_s$, an inertial branch integrates the velocity from the previous fused position, and a learned gate biased by the contact prior blends the candidates. We keep the candidate-based formulation and the object-frame parameterization of the grasp offset (we drop the superscript $O$ below), and change how the offsets are predicted, which candidates are available, and how they are fused.

### 3.3.1 Candidates

Given the palm positions $\hat{\mathbf h}_s(t)$, the object orientation $R_O(t)$, the VC outputs and the previously published position $\hat{\mathbf p}_O(t-1)$, OT forms five candidates,

$$
\hat{\mathbf p}^{\mathrm{FK},s}(t)=\hat{\mathbf h}_s(t)+R_O(t)\,\hat{\mathbf u}_s(t)\,\hat\ell_s(t),\qquad s\in\{L,R\},
\tag{8}
$$

$$
\hat{\mathbf p}^{\mathrm{I}}(t)=\hat{\mathbf p}_O(t-1)+\hat{\mathbf v}_O(t)\,\Delta t,\qquad
\hat{\mathbf p}^{\mathrm{S}}(t)=\hat{\mathbf p}_O(t-1),\qquad
\hat{\mathbf p}^{\mathrm{B}}(t)=\hat{\mathbf m}(t)+R_O(t)\,\hat{\mathbf c}(t),
\tag{9}
$$

namely two kinematic (FK) candidates, inertial integration, static hold, and a bimanual-closure candidate built around the palm midpoint $\hat{\mathbf m}=\frac12(\hat{\mathbf h}_L+\hat{\mathbf h}_R)$ (Sec. 3.3.3). The direction $\hat{\mathbf u}_s\in\mathbb S^2$ is obtained by normalization and $\hat\ell_s=\mathrm{softplus}(\cdot)+\epsilon$. The FK candidates are anchored at the palms rather than at the wrists, which brings the anchor closer to the physical contact.

### 3.3.2 Shared grasp predictor with closed-loop state

**Shared predictor.** Instead of one head per hand, a single recurrent predictor $f_\theta$ (two-layer LSTM) estimates the grasp offsets of both hands,

$$
\big[\hat{\mathbf u}_L,\ \hat\ell_L,\ \hat{\mathbf u}_R,\ \hat\ell_R\big](t)=f_\theta\big(\mathbf x_t+g_\phi(\mathbf z_t)\big).
\tag{10}
$$

The input $\mathbf x_t$ concatenates the object IMU reading, the object rotation increment, the VC outputs $\hat{\mathbf v}_O$ and $(p_L,p_R,p_m)$, the palm positions and forearm IMU readings of both hands, the forearm–object rotation features $\boldsymbol\beta_t$ and the segment statistics $\boldsymbol\Lambda$ introduced below. At the window start $t_0$, an MLP maps the observed geometry—$\hat\ell_s(t_0)=\lVert\hat{\mathbf p}_O(t_0)-\hat{\mathbf h}_s(t_0)\rVert$ and $\hat{\mathbf u}_s(t_0)=R_O(t_0)^{\top}\big(\hat{\mathbf p}_O(t_0)-\hat{\mathbf h}_s(t_0)\big)/\hat\ell_s(t_0)$—together with the contact state, $\boldsymbol\Lambda$ and $B_s(t_0)$ to the initial LSTM state. A shared state lets the two offsets be estimated consistently in bimanual holds, which make up about half of the contact frames on OMOMO.

**State feedback.** The heads of IMU-HOI observe the object state only through the window-initial geometry. Within a contact segment, however, the hand re-settles and slides, so the true offset drifts. We feed the published state back into the predictor through

$$
\mathbf z_t=\Big[R_O(t)^{\top}\big(\hat{\mathbf p}_O(t-1)-\hat{\mathbf h}_L(t)\big),\ \
R_O(t)^{\top}\big(\hat{\mathbf p}_O(t-1)-\hat{\mathbf h}_R(t)\big),\ \
R_O(t)^{\top}\big(\hat{\mathbf p}_O(t-1)-\hat{\mathbf p}_O(t-2)\big),\ \
\mathbf w(t-1)\Big]\in\mathbb R^{13},
\tag{11}
$$

i.e., the currently observed offsets to both palms and the last displacement, all in the object frame, and the previous fusion weights. The feedback network $g_\phi$ is a two-layer MLP with a zero-initialized output layer, so that feedback starts as an exact no-op and is learned as a residual correction of the input. With feedback, OT becomes a nonlinear closed-loop recursion; it is trained on 60-frame windows and executed online by the same per-frame recursion (Sec. 3.4).

**Forearm–object relative rotation.** Consider a grip that pivots about a point $\mathbf g_s$ fixed in the object frame, with the palm at a fixed offset $\boldsymbol\nu_s$ from the pivot in the forearm frame. The object-frame offset then evolves as

$$
R_O(t)^{\top}\big(\mathbf p_O(t)-\mathbf h_s(t)\big)=-\mathbf g_s-B_s(t)\,\boldsymbol\nu_s,
\tag{12}
$$

so its change within a segment is driven by $B_s(t)$. This change is substantial: within bimanual segments on OMOMO, the ground-truth offset moves by 8.9 cm on average while the forearm rotates by 45° relative to the object. The heads of IMU-HOI cannot form $B_s$ directly, because their forearm orientations are expressed relative to the pelvis and the pelvis orientation is not among their inputs. We compute $B_s$ with Eq. (1), after mapping the forearm orientation back to the world frame with the pelvis IMU, and provide per hand

$$
\boldsymbol\beta_s(t)=\Big[\mathrm{6D}\big(B_s(t)\big),\ \ \mathrm{6D}\big(B_s(t-1)^{\top}B_s(t)\big),\ \ \mathrm{6D}\big(B_s(t_0)^{\top}B_s(t)\big),\ \ \angle\big(B_s(t_0)^{\top}B_s(t)\big)\Big]\in\mathbb R^{19},
\tag{13}
$$

i.e., the current relative rotation, its per-frame increment and its change since the window start; $B_s(t_0)$ also enters the initial state.

**Segment-length statistics.** As a weak prior on the reach length, $\boldsymbol\Lambda\in\mathbb R^{2\times 3}$ contains, per hand, the 10/50/90 % quantiles of $\lVert\hat{\mathbf p}_O(\tau)-\hat{\mathbf h}_s(\tau)\rVert$ over all previously published frames on which the hand was predicted to be in contact ($p_s>0.25$; at least 10 frames, otherwise the window-initial length is used). $\boldsymbol\Lambda$ is broadcast to every frame of the window. During training it is simulated from a random causal prefix of the ground-truth sequence with Gaussian noise (2 cm) and per-hand dropout (30 %), so that the predictor learns both to exploit it and to cope without it.

### 3.3.3 Bimanual closure candidate

When both hands hold the object, the two FK candidates estimate the same point. We add a structurally different estimate in the hand-pair frame (Eq. 9), in which $\hat{\mathbf c}(t)$ is the offset of the object center from the palm midpoint, expressed in the object frame. This common-mode offset is considerably less ambiguous across grasps than a per-hand offset (spread of 15.0 vs. 24.5 cm on the OMOMO training set). Crucially, $\hat{\mathbf c}$ is regressed by a GRU from state-free features only—the palm span $R_O^{\top}(\hat{\mathbf h}_R-\hat{\mathbf h}_L)$, its norm and temporal difference, the palm positions relative to the root and the object and forearm accelerations (all in the object frame), the object orientation, $B_L$ and $B_R$, and the contact probabilities (43-D in total)—and never reads the published object position, so it cannot inherit object drift on long sequences. Its share in the fusion is

$$
\eta_t=\sigma\Big(h_\psi\big[\mathbf s^{\mathrm{pred}}_t,\ \mathbf s^{\mathrm{clo}}_t,\ \boldsymbol\xi_t\big]\Big)\cdot\mathbb 1\big[\min(p_L,p_R)\ge 0.2\big],\qquad
\boldsymbol\xi_t=\big[\mathbf r_t,\ \lVert\mathbf r_t\rVert,\ \boldsymbol\delta_t,\ \lVert\boldsymbol\delta_t\rVert,\ \min(p_L,p_R)\big],
\tag{14}
$$

where $\mathbf s^{\mathrm{pred}}_t$ and $\mathbf s^{\mathrm{clo}}_t$ are the hidden states of the grasp predictor and of the closure GRU, $\mathbf r_t=R_O^{\top}\big(\hat{\mathbf p}^{\mathrm{FK},L}-\hat{\mathbf p}^{\mathrm{FK},R}\big)$ is the closure residual between the two FK proposals (hand inconsistency), and $\boldsymbol\delta_t=R_O^{\top}\big(\hat{\mathbf p}_O(t-1)-\hat{\mathbf m}\big)-\hat{\mathbf c}$ is the innovation between the observed and the predicted common-mode offset (state inconsistency). The output bias of $h_\psi$ is initialized to $-2$, and no latching is applied. For publication and feedback, the five-way fusion is folded exactly into four branches by assigning half of the pair weight to each hand,

$$
\tilde w_s=(1-\eta_t)\,w_s+\frac{\eta_t}{2},\qquad
\tilde{\mathbf p}^{\mathrm{FK},s}=\frac{(1-\eta_t)\,w_s\,\hat{\mathbf p}^{\mathrm{FK},s}+\frac{\eta_t}{2}\,\hat{\mathbf p}^{\mathrm{B}}}{\tilde w_s},
\tag{15}
$$

which leaves the fused position unchanged and lets all subsequent operations act on two effective FK candidates.

### 3.3.4 Contact-driven fusion, FK tracking and static gating

**Closed-form fusion weights.** IMU-HOI computes $\mathbf w_t=\mathrm{softmax}\big((\mathbf z_t+\beta\log\boldsymbol\pi_t)/\tau\big)$ from learned logits $\mathbf z_t$ and a contact prior $\boldsymbol\pi_t$. We remove the learned logits and derive the weights of the left-FK, right-FK, inertial and static candidates directly from Eq. (2),

$$
\mathbf w(t)\propto\big[\,p_L,\ \ p_R,\ \ p_m\,(1-\bar p_L)(1-\bar p_R),\ \ 1-p_m\,\big],\qquad
\bar p_s=\mathrm{clip}\big(p_s/p_m,\,0,\,1\big),
\tag{16}
$$

so that the inertial candidate receives the mass of "moving but not held"; the five-way weights are $\big[(1-\eta_t)\,\mathbf w(t),\ \eta_t\big]$. The motivation is the training regime. OT is trained on short windows that start from the ground-truth object state, where the recursive (inertial and static) candidates are almost exact. A learned gate adapts to this regime: in our experiments it assigned on average only 0.20 of the weight to the FK candidates on the contact frames of the long BEHAVE sequences, so that the estimate degenerated into inertial dead reckoning and drifted. Eq. (16) assigns 0.88 on the same frames and ties the fusion to the quantity that VC is trained to predict.

**FK tracking filter.** A high FK weight passes the frame-to-frame noise of the FK candidates—orientation noise propagated through the arm chain and fluctuations of the offset prediction—into the output. Each effective FK candidate is therefore tracked by a constant-gain α–β filter while its hand is in contact (hand index omitted),

$$
\bar{\mathbf x}_t=\mathbf x_{t-1}+\Delta t\,\dot{\mathbf x}_{t-1},\quad
\mathbf e_t=\tilde{\mathbf p}^{\mathrm{FK}}(t)-\bar{\mathbf x}_t,\quad
\mathbf x_t=\bar{\mathbf x}_t+\alpha\,\mathbf e_t,\quad
\dot{\mathbf x}_t=\dot{\mathbf x}_{t-1}+\frac{\beta}{\Delta t}\,\mathbf e_t,
\tag{17}
$$

with $\alpha=0.35$ and $\beta=0.05$. Contact segments are delimited by hysteresis on $p_s$ (entered at 0.7 and left below 0.3); at the first frame of a segment and outside segments, the filter is reset to the raw candidate with zero velocity. The filter state is part of the published stream and is restored on teacher-forced frames, so that it continues across windows during online inference.

**IMU-certified static gate.** When the object IMU certifies rest—a rotation below 0.6° over the last $k=5$ frames and an acceleration magnitude below 0.35 m/s² throughout the same causal window—the fusion is pinned to the static candidate ($\mathbf w=\mathbf e_{\mathrm S}$, $\eta_t=0$). The test has a precision above 0.995 on both datasets at a recall of about 0.55, so it stops the drift of resting objects caused by spurious contacts or velocity bias with little risk of freezing a moving object.

**Published estimate.** The published object position is

$$
\hat{\mathbf p}_O(t)=\sum_{s\in\{L,R\}}\tilde w_s(t)\,\mathbf x_s(t)
+\big(1-\eta_t\big)\Big[w_{\mathrm I}(t)\,\hat{\mathbf p}^{\mathrm I}(t)+w_{\mathrm S}(t)\,\hat{\mathbf p}^{\mathrm S}(t)\Big],
\tag{18}
$$

where $\mathbf x_s$ is the filtered effective FK candidate of hand $s$.

*Remark.* The closed-form fusion trades a small loss of hand–object accuracy on short clips, on which a learned gate rarely faces a drifted state, for robustness and smoothness on long sequences; the trade-off is quantified in [Sec. 4, ablation].

### 3.3.5 Objective

Let $\mathcal C_s$ denote the ground-truth contact frames of hand $s$, $\mathbf u_s$ and $\ell_s$ the ground-truth object-frame direction and length from the palm to the object center, and $\mathbf h_s$ and $\mathbf m$ the ground-truth palms and their midpoint. The OT objective is

$$
\mathcal L_{\mathrm{OT}}=
\lambda_p\big\lVert\hat{\mathbf p}_O-\mathbf p_O\big\rVert^2
+\lambda_v\Big\lVert\frac{\hat{\mathbf p}_O(t)-\hat{\mathbf p}_O(t-1)}{\Delta t}-\mathbf v_O\Big\rVert^2
+\sum_{s}\mathcal L^{s}_{\mathrm{grasp}}
+\lambda_B\,\mathcal L_B+\lambda_c\,\mathcal L_c+\mathcal L_{\mathrm{ref}},
\tag{19}
$$

with, on $\mathcal C_s$,

$$
\mathcal L^{s}_{\mathrm{grasp}}=
\lambda_u\big\lVert\hat{\mathbf u}_s-\mathbf u_s\big\rVert^2
+\lambda_\ell\big(\hat\ell_s-\ell_s\big)^2
+\lambda_\Delta\Big(\frac{\Delta\hat\ell_s-\Delta\ell_s}{\Delta t}\Big)^2
+\lambda_{\mathrm{hoi}}\big\lVert\hat{\mathbf u}_s\hat\ell_s-\mathbf u_s\ell_s\big\rVert,
\tag{20}
$$

and, on bimanual frames $\mathcal C_L\cap\mathcal C_R$,

$$
\mathcal L_B=\frac12\sum_s\Big\lVert\big(\hat{\mathbf p}^{\mathrm B}-\hat{\mathbf h}_s\big)-\big(\mathbf p_O-\mathbf h_s\big)\Big\rVert,\qquad
\mathcal L_c=\Big\lVert\hat{\mathbf c}-R_O^{\top}\big(\mathbf p_O-\mathbf m\big)\Big\rVert .
\tag{21}
$$

The offset-vector term of Eq. (20) penalizes exactly the hand–object relation evaluated by the HOI metric. The length-difference term matches the frame-to-frame change of the reach length to the ground truth instead of suppressing it, so that a sliding hand is followed rather than flattened. $\mathcal L_B$ evaluates the closure candidate with the same relative measure, and $\mathcal L_c$ supervises its common-mode offset. We use $\lambda_p=5$, $\lambda_v=0.01$, $\lambda_u=1$, $\lambda_\ell=10$, $\lambda_\Delta=1$, $\lambda_{\mathrm{hoi}}=10$, $\lambda_B=5$ and $\lambda_c=1$. $\mathcal L_{\mathrm{ref}}$ trains a small output-side head that predicts residual corrections of the reduced body pose and the root translation from the fused object state; its inputs are detached, so it cannot alter OT or VC, and its outputs are not fed back to later windows.

## 3.4 Training and Online Inference

**Online protocol.** All models run causally with a fixed lag of $L=10$ frames (0.33 s). The first $W=60$ frames are published together from a single pass over the first $W+L$ frames, initialized with $\mathcal S_0$. Afterwards, frame $t$ is published from the window $[t_0,\,t+L]$ with $t_0=t-W+1$: the recurrent states of HP and VC are re-initialized from the published stream, the object position and the contact state at $t_0$ are read from it, and OT is teacher-forced on the published prefix $[t_0,\,t-1]$, so that its closed-loop recursion—including the filter state of Eq. (17)—continues from the published state; only frame $t$ is published. The object-velocity initializer of VC is the three-frame backward difference of the published trajectory, $\mathbf v_O^{\mathrm{init}}=\big(\hat{\mathbf p}_O(t_0-1)-\hat{\mathbf p}_O(t_0-4)\big)/(3\Delta t)$. It replaces VC's own previous output, which forms a self-referential chain whose error performs a random walk across windows on long sequences, whereas the published trajectory is anchored by the FK and static candidates. In addition, the HP outputs are smoothed with a centered Savitzky–Golay filter (half-width 7 frames, order 2), which fits inside the lookahead and therefore adds no latency.

**Training schedule.** HP is trained first and kept frozen. VC is trained on the frozen HP outputs (Sec. 3.2.5). OT is then trained with HP and VC frozen, on 60-frame windows that start from the ground-truth state. Because OT consumes upstream estimates at test time, its upstream inputs (palm positions, body pose, root translation, contact probabilities and object velocity) follow a schedule: for each training window they are taken either from the ground truth or from the frozen upstream modules, the probability of the latter growing linearly from 0 to 1 over the first 100 epochs. A single draw per window is shared by the whole HP→VC→OT chain, so that inconsistent mixtures—e.g., contacts computed from predicted poses combined with ground-truth palms—never occur. Finally, VC and OT are fine-tuned jointly with $\mathcal L_{\mathrm{VC}}+\mathcal L_{\mathrm{OT}}$ at one tenth of the learning rate, with the initializer perturbation of Eq. (6) still active and HP frozen. IMU-HOI instead fine-tunes HP together with OT; in our re-implementation this degrades the root trajectory (root error 6.6 → 10.9 cm on OMOMO and 16.7 → 28–30 cm on BEHAVE), whereas fine-tuning VC adapts the contact and velocity estimates to the fusion without touching the body.

**Implementation details.** In VC, the velocity regressors are two-layer LSTMs with 128 units, the object-motion network is a one-layer bidirectional LSTM and the hand-contact network a one-layer causal LSTM (64 units each); the group encoders are GRUs with 64 units, and the boundary encoder is a bidirectional GRU with 128 units. In OT, the grasp predictor is a two-layer LSTM with 192 units, $g_\phi$ has 64 hidden units, and the closure encoder is a GRU with 96 units. The LSTMs obtain their initial states from their initializer vectors through MLPs; the GRUs start from zero. All modules are trained with AdamW (learning rate $2\times10^{-3}$, weight decay $10^{-4}$, gradient clipping at 1.0, bf16 mixed precision) on batches of 800 windows, sampling four windows per sequence and epoch, with step-wise learning-rate decay by a factor of 0.2. Inertial noise is simulated during training and testing (acceleration σ = 0.2 m/s²; orientation σ = 0.005 per 6-D component).

---

## 3.5 Summary of Empirical Gains (working material; belongs to Sec. 4)

**Table 2.** Full-sequence online evaluation (window 60, lookahead 10, simulated IMU noise, training seed 42). Obj: object position error (cm); HOI: hand–object relative error on ground-truth contact frames (cm); Root: root translation error (cm); Jit.: object jitter (mm/frame²); F1: hand-contact F1 (left/right). *IMU-HOI* denotes our re-implementation retrained on the same canonical data and contact labels; its OT uses four candidates (with static hold) and, when trained with its own recipe, a joint stage that fine-tunes HP. *VC (pose prior)* is VC without the coupling features and the initializer perturbation (`v4`); *VC (full)* is `v5ar`; *OT (ours)* is `v2_armrot_biclose`.

OMOMO (1,173 test sequences; previous online protocol for every row, i.e., no HP smoothing and VC's own previous output as velocity initializer; best = final checkpoint within 0.01 cm):

| VC | OT | Obj | HOI | Root | Jit. | F1 |
|---|---|---:|---:|---:|---:|---|
| IMU-HOI | IMU-HOI, own recipe (HP fine-tuned) | 19.47 | 16.70 | 10.86 | 4.76 | 0.921 / 0.926 |
| IMU-HOI | IMU-HOI, shared frozen HP | 18.43 | 16.42 | 6.65 | 5.10 | 0.921 / 0.926 |
| pose prior | IMU-HOI (HP fine-tuned) | 17.94 | 16.27 | 10.80 | 4.54 | 0.946 / 0.947 |
| IMU-HOI | ours | 13.88 | 10.71 | 6.01 | 5.33 | 0.915 / 0.918 |
| pose prior | ours | 13.37 | 10.44 | 6.04 | 5.34 | 0.949 / 0.952 |
| **full** | **ours** | **13.23** | **10.48** | **6.01** | **5.32** | **0.956 / 0.955** |

Under the unified protocol of Sec. 3.4 (HP smoothing, published-trajectory velocity initializer), the retrained full model reaches 13.33 / 10.48 cm (Obj / HOI), object jitter 4.80 and body jitter 2.96 instead of 7.34; the IMU-HOI rows have not been re-run under this protocol on OMOMO.

BEHAVE (59 full test sequences, median length 1,424 frames; unified protocol for every row; best checkpoints unless marked):

| VC | OT | Obj | HOI | Root | Jit. | F1 |
|---|---|---:|---:|---:|---:|---|
| IMU-HOI | IMU-HOI, own recipe (HP fine-tuned) | 41.03 | 25.69 | 28.19 | 7.74 | 0.788 / 0.787 |
| IMU-HOI | IMU-HOI, shared frozen HP (final) | 32.84 | 25.04 | 16.71 | 7.55 | 0.790 / 0.788 |
| pose prior | IMU-HOI (HP fine-tuned) | 39.84 | 24.80 | 28.27 | 8.56 | 0.787 / 0.778 |
| IMU-HOI | ours | 25.73 | 18.82 | 16.16 | 7.81 | 0.794 / 0.798 |
| pose prior | ours | 23.97 | 17.81 | 16.13 | 6.49 | 0.802 / 0.796 |
| **full** | **ours** | **22.93** | **17.19** | **16.13** | **5.74** | **0.838 / 0.845** |

Final checkpoints on BEHAVE: full model 22.88 / 17.14 (Obj / HOI), pose-prior VC + our OT 24.08 / 17.85, IMU-HOI own recipe 42.22 / 25.29 (root 30.42). MPJPE is unaffected by the VC/OT changes (OMOMO 2.99–3.02 cm, BEHAVE 3.68–3.74 cm).

**Overall.** Against IMU-HOI trained with its own recipe, the full model reduces the object error by 32 % (OMOMO) and 44 % (BEHAVE, best vs. best) and the HOI error by 37 % and 33 %. Against the stricter control in which IMU-HOI uses the same frozen HP, the reductions are 28 % / 36 % on OMOMO and 30 % / 32 % on BEHAVE (final vs. final).

**Attribution.** The OT redesign carries most of the gain: with the IMU-HOI VC fixed, it lowers Obj / HOI by 5.6 / 6.0 cm on OMOMO and by 15.3 / 6.9 cm on BEHAVE (this pipeline-level comparison includes the different joint-stage recipe, cf. the root column). On top of our OT, the VC redesign adds 0.65 / 0.24 cm on OMOMO and 2.79 / 1.63 cm on BEHAVE, raises the contact F1 by about 0.04 on both datasets and lowers the BEHAVE object jitter by 26 %. Two training seeds on BEHAVE (pose-prior VC vs. full VC, each with retrained OT and joint stage; sequence-paired bootstrap on seed-averaged final errors) give Obj −0.92 cm [−1.56, −0.32] and HOI −1.21 cm [−1.92, −0.50] for the coupling features plus initializer perturbation; on OMOMO the change is not significant (Obj −0.14 [−0.38, +0.09], HOI +0.04 [−0.16, +0.24]; one seed).

---

## Appendix A. Mapping to the code

| Paper element | Code / configuration |
|---|---|
| VC (full) | `model/rnn/velocity_contact_v5.py::V5VelocityContactModule`; `velocity_contact_arch: v5` |
| Eq. (2): object-motion and hand-contact networks | base class `model/rnn/velocity_contact_ablation.py::IMUHOIConditionalVelocityContactModule` (`obj_move_net`, `hand_contact_net`) |
| Eq. (3) | `train/rnn/loss/velocity_contact_loss.py`: `obj_move`, `hand_contact_cond` (`hand_contact_{pos,neg,hard_neg}_weight` = 1 / 2 / 4), `hand_contact_marginal` (0.5) |
| Pose prior (Sec. 3.2.2) | `model/rnn/velocity_contact_v4.py`: `_prepare_pose_streams`, `_encode_body_groups`, `contact_endpoint_head`, `contact_anticipation_head`; `v4_heading_normalize`, `v4_boundary_smooth_window: 5`, `v4_anticipation_horizon: 15` |
| Endpoint and anticipation losses | `contact_endpoint` (1.0), `contact_endpoint_order` (0.5), `endpoint_contact_consistency` (0.5), `contact_anticipation` (0.5) |
| Coupling features, Eqs. (4)–(5) | `velocity_contact_v5.py::_coupling_features`, `COUPLING_WINDOWS` |
| Initializer perturbation, Eq. (6) | `velocity_contact_v5.py::_perturb_inits`; `v5_init_noise_prob: 0.5`, `v5_obj_vel_init_noise_std: 0.2`, `v5_hand_vel_init_noise_std: 0.2`, `v5_contact_init_flip_prob: 0.0` |
| Two-stage VC optimization | `train/rnn/train_velocity_contact.py`; `vc_staged_training`, `vc_grouped_lr` |
| OT (full) | `model/rnn/object_trans_v2.py::V2ObjectTransModule` on top of `model/rnn/object_trans.py::ObjectTransModule` |
| Candidates, Eqs. (8)–(9); publication, Eq. (18) | `ObjectTransModule._predict_step`, `V2ObjectTransModule._predict_step` |
| Shared predictor, Eq. (10) | `object_trans_head` (`RNNWithInit`), `object_trans_hidden_dim: 192` |
| State feedback, Eq. (11) | `_feedback_features`, `feedback_embed`; `object_trans_state_feedback: true` |
| Forearm–object rotation, Eqs. (12)–(13) | `_forearm_object_rotation`, `_forearm_object_features`; `object_trans_forearm_obj_rotation: true` |
| Segment statistics $\boldsymbol\Lambda$ | `ObjectTransModule.compute_online_lb_quantiles`, `IMUHOIModel._compute_online_lb_stats`, `train/rnn/scheduled_inputs.py::sample_gt_lb_prior_stats`; `object_trans_lb_stats`, `lb_stats_contact_threshold: 0.25`, `lb_stats_noise_std: 0.02`, `lb_stats_drop_prob: 0.3` |
| Bimanual closure, Eqs. (14)–(15) | `model/rnn/bimanual_closure.py::BimanualClosure`; `object_trans_bimanual_closure: true` |
| Closed-form weights, Eq. (16) | `utils/object_fusion.py::build_contact_fusion_weights` (the same mapping is the gating prior of the four-candidate IMU-HOI re-implementation) |
| FK tracking filter, Eq. (17) | `V2ObjectTransModule._predict_step`; `v2_fk_filter_alpha: 0.35`, `v2_fk_filter_beta: 0.05` |
| Static gate | `_imu_rest_flags`, `_apply_static_gate`; `object_trans_static_gate: imu` (window 5, 0.6°, 0.35 m/s²) |
| Eqs. (19)–(21) | `train/rnn/loss/object_trans_loss.py`: `obj_trans` (5), `obj_vel_cons` (0.01), `{l,r}hand_obj_direction` (1), `{l,r}hand_lb` (10), `{l,r}hand_lb_cons` (1), `hoi_error_{l,r}` (10), `bimanual_hoi` (5), `bimanual_prior` (1), `refine_pose` (100), `refine_root_trans` (10) |
| Input scheduling | `train/rnn/train_object_trans.py::Stage3JointTrainer.forward_all`, `scheduled_inputs.prediction_mix_probability`; `input_mix_end_epoch: 99` |
| Joint stage | `train_object_trans.py::_joint_trainable_modules` (VC + OT for ours, HP + OT for IMU-HOI) |
| Online protocol | `model/rnn/imuhoi_model.py::_inference_online_sequence`, `_prepare_online_chunk`; `model/rnn/online.py::update_data_inits_from_history`; `online_obj_vel_init_diff_frames: 3`, `human_pose_smoothing_half_width: 7`, `human_pose_smoothing_polyorder: 2` |
| Runs | `outputs/IMUHOI++/RNN_cano/vc_v5_20260930_ot_{omomo,behave}_v5ar/`, `vc_v5_20260930_ot_omomo_v5ar_unified/`; VC×OT crossing in `docs/vc_ot_cross_ablation.md` and `outputs/analysis/vc_ot_cross_ablation.csv`; IMU-HOI frozen-HP controls in `behave_imuhoi_vs_ours_20260921/imuhoi_frozen_hp_omomo.json` and `behave_hpsmooth_vcinit_20260922/imuhoi_retrained_frozenhp.json` |

## Appendix B. Evidence behind each design choice

Single training seed unless stated; Obj / HOI in cm. "Learned-gate OT" is the `current` architecture (same predictor and feedback, four-way learned gate trained with a fused HOI loss).

| # | Change | Comparison | Measured effect | Caveat |
|---|---|---|---|---|
| V1 | Factorization + pose prior + auxiliary heads | VC IMU-HOI → pose-prior VC, OT ours | OMOMO 13.88/10.71 → 13.37/10.44, F1 +0.034 / +0.034; BEHAVE 25.73/18.82 → 23.97/17.81, F1 +0.008 / −0.002 | pipeline-level; no isolated ablation of each sub-part |
| V2 | Coupling features | pose-prior VC → + coupling (`v5a`) | BEHAVE F1 +0.039 [+0.011, +0.080] / +0.041 [+0.013, +0.076] (significant), HOI −0.86 [−1.82, +0.09]; OMOMO Obj +0.23 [+0.04, +0.41] (worse) | paired bootstrap over sequences, one seed |
| V3 | Initializer perturbation | `v5a` → `v5ar` | BEHAVE online velocity error 0.21–0.29 → 0.15–0.18 m/s, Obj −0.51 [−1.72, +0.53] (final), jitter −0.81 (best) / −0.32 (final); OMOMO Obj 13.59 → 13.23 | Obj gain on BEHAVE only significant for V2+V3 together |
| V2+V3 | Both | pose-prior VC → `v5ar` | BEHAVE, 2 seeds, final: Obj −0.92 [−1.56, −0.32], HOI −1.21 [−1.92, −0.50]; with frame-1 initial contact −0.77 / −0.96 (both significant); OMOMO n.s. | — |
| V4 | Initializer source (protocol) | own output vs. 3-frame difference | pose-prior VC on BEHAVE: own output +1.73 Obj [+0.57, +3.13]; `v5ar`: +0.02 [−0.04, +0.10] | — |
| O1 | Whole OT redesign | OT IMU-HOI → ours (pose-prior VC) | OMOMO 17.94/16.27 → 13.37/10.44; BEHAVE 39.84/24.80 → 23.97/17.81 | includes joint-stage recipe (root 10.8 → 6.0, 28.3 → 16.1) |
| O2 | Whole OT, HP-controlled | IMU-HOI frozen HP → ours with IMU-HOI VC | OMOMO 18.43/16.42 → 13.88/10.71; BEHAVE 32.84/25.04 → 25.73/18.82 | VC weights differ by the joint stage |
| O3 | State feedback | learned-gate OT, without → with feedback | OMOMO 13.49/10.08 → 13.32/9.38; BEHAVE (unified protocol, final) 23.69/18.68 → 27.16/24.05 (worse) | not ablated on the closed-form fusion |
| O4 | Closed-form fusion | learned-gate OT → v2 (both with feedback) | BEHAVE final 27.16/24.05 → 24.69/18.69 (jitter 3.47 → 10.72, no FK filter), contact-frame FK weight 0.20 → 0.88; OMOMO (with filter) 13.32/9.38 → 13.32/10.45 (jitter 7.34 → 5.02) | **OMOMO HOI +1.07 cm** (trade-off) |
| O5 | FK α–β filter | v2 without (2026-09-22) → with (2026-09-23) | BEHAVE final 24.69/18.69/jitter 10.72 → 24.49/18.32/6.11 | two separate trainings |
| O6 | Forearm–object rotation | learned-gate OT, without → with | OMOMO 13.32/9.38 → 13.41/9.11; BEHAVE final (old protocol) 37.90/33.44 → 34.01/31.76 | not ablated alone on v2 |
| O7 | Bimanual closure | learned-gate OT, without → with | OMOMO 13.32/9.38 → 12.76/9.15; BEHAVE final 37.90/33.44 → 36.71/32.52 | within checkpoint spread; probes bound bimanual-specific gains on OMOMO to ≈ 0.2–0.4 cm |
| O6+O7 | Both on v2 | v2 → `v2_armrot_biclose` | OMOMO final 13.34/10.47/jitter 5.03 → 13.37/10.44/5.34; BEHAVE final 24.49/18.32/6.11 → 24.08/17.85/6.45; BEHAVE bimanual-frame HOI −0.88 [−1.90, +0.12] | small, jitter +0.3 on both |
| O8 | IMU static gate | older learned-gate model, old labels (2026-09-09) | BEHAVE Obj 55.4 → 44.8, HOI 40.8 → 39.2; OMOMO Obj 13.72 → 13.54, HOI 9.44 → 9.55 | not re-ablated on the final OT |
| O9 | Segment statistics | test-time off, older model (2026-09-09) | OMOMO 13.72/9.44 → 13.71/9.45 (none); BEHAVE 55.4/40.8 → 57.3/41.9 when disabled | weak evidence |
| O10 | Refine head | refined vs. stage-1 human (BEHAVE, 8 sequences) | root 17.07 → 16.88 cm; essentially a constant +2.4 cm vertical lift | marginal |
| T1 | Joint stage VC + OT instead of HP + OT | IMU-HOI recipe vs. frozen HP | root OMOMO 6.65 → 10.86; BEHAVE 16.71 → 28.19 (best) / 30.42 (final) | — |

Sources: `docs/vc_ot_cross_ablation.md`, `docs/vc_v5_oracles_and_coupling_20260930.md` (§5, §7, §9), `docs/long_horizon_hoi_diagnosis_20260909.md` (§6.2, §13, §14, §15, §17).

## Appendix C. Directions tried and not adopted

- Event-anchored grasp latent (and its rigidity modes / bounded-residual variants): long-horizon gain on BEHAVE, but OMOMO HOI +0.7–1.4 cm and doubled jitter; the modes variant diverges on long sequences.
- Object-state confidence ("state trust") and window-initial / rest-state corruption training: a monotone OMOMO-vs-BEHAVE trade-off; the confidence head stays uncalibrated.
- Rollout-aligned (history-window) training: its BEHAVE gain disappeared after the contact labels were rebuilt.
- Anchor-state feedback: fine on BEHAVE, OMOMO HOI +0.30 cm.
- Vertical lock of the inertial candidate without hand contact: noise-level effect.
- Bimanual mid-plane losses and post-hoc drift correctors, mesh-free bimanual priors (2026-10-01): at most ≈ 0.1 cm on OMOMO and ≈ 0.3 cm Obj on BEHAVE as post-hoc correctors on frozen models; not integrated.
- Heading-normalized IMU inputs of the contact networks (`v5br`): no difference over two seeds.
- Rigid-coupling lever-arm physics, hard grasp latches, hand ZUPT: falsified by probes.
- Mamba backbones: earlier attempts lost to the LSTM.

## Appendix D. Open issues before submission

1. Model selection and early stopping currently use the test sequences (`best.pt`, early-stopping windows). A separate validation split, or a fixed training budget with final checkpoints, is needed for the reported numbers.
2. All rows except the BEHAVE VC comparison rely on one training seed; BEHAVE differences of about 1 cm are within the seed spread (pose-prior VC: HOI 17.8–18.6 over two seeds).
3. The two datasets use different online protocols in Table 2. Only the full model has been retrained under the unified protocol on OMOMO; the IMU-HOI rows on OMOMO still use the previous one.
4. Attribution inside OT is incomplete: feedback (O3), the static gate (O8), the segment statistics (O9) and the refine head (O10) lack ablations on the final architecture, and O6+O7 is small with a jitter cost. Components without positive evidence may be better dropped from the paper.
5. The closed-form fusion costs about 1 cm HOI on OMOMO relative to the learned gate (O4); the paper should present it as a long-horizon design choice and report both settings.
6. The frame-0 hand-contact labels are always 0 (backward-difference artifact in `process/preprocess.py::compute_improved_contact_labels`), so every test sequence starts from "no hand contact"; report the initial contact state used.
7. The simulated orientation noise (σ = 0.005 per 6-D component, about 0.5°/frame, white) is about five times the Noitom-style noise model and dominates the FK jitter; validate against real sensors.
8. The IMU-HOI numbers come from our re-implementation retrained on the rebuilt labels and evaluated on full sequences; they are not comparable to the tables of the conference paper.
