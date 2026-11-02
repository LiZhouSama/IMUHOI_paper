"""GlobalPose physics optimizer, ported from the released code to NumPy.

GlobalPose (Yi et al., SIGGRAPH 2025) refines every network frame with a
physics-based optimizer, ``GPNet.forward_frame`` in the released ``net.py``.
The optimizer is Python, but it runs on the C++ ``carticulate`` dynamics
library, which the authors ship only as a Windows / Python 3.8 binary.
``DynamicModel`` re-implements what the optimizer calls from
``dynamics/dynamic_model.cpp`` (github.com/Xinyu-Yi/carticulate @ ce5b204),
``GlobalPosePhysics.step`` ports the optimizer line by line, and
``armature_from_body_model`` ports articulate's ``physical_properties``.

Deviations forced by the data:
- the armature is built from the evaluation body model (the corpus' canonical
  skeleton) instead of SMPL male, because the network predicts that skeleton;
- the floor is the lowest contact joint of the first frame with the root at its
  ground-truth position (released: the same rule with the root at x = z = 0);
- the network runs at 30 fps while the released PD gains assume 1/60 s steps
  (with 1/30 s steps the controller diverges), so every frame is reached in two
  1/60 s steps toward targets interpolated causally from the previous frame;
- the friction-cone QP is solved exactly (bounded least squares) instead of
  with OSQP.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator
from scipy.linalg import block_diag
from scipy.optimize import lsq_linear
from scipy.sparse import csc_array
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation

from utils.human_pose import build_smpl24_joint_positions

SMPL24_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
CONTACT_JOINTS = (0, 10, 11, 22, 23)  # pelvis, feet, palms (released j_contact)


def _hat(v: np.ndarray) -> np.ndarray:
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # np.cross costs ~10 us per call; the recursions below call it per joint.
    return np.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


@dataclass
class Armature:
    """carticulate ``DynamicArmature`` plus the body model's rest root joint."""

    parent: tuple
    bone: np.ndarray         # [J, 3] joint position in the parent frame (root: 0)
    com: np.ndarray          # [J, 3] body centre of mass in the joint frame
    mass: np.ndarray         # [J]
    inertia: np.ndarray      # [J, 3, 3]
    root_offset: np.ndarray  # [3] rest root joint; body-model trans = root - root_offset
    gravity: np.ndarray = field(default_factory=lambda: np.array((0.0, -9.8, 0.0)))


def _inside_mesh(points: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """Winding-number inside test (articulate ``ParametricModel.is_inside``)."""
    tris = vertices[faces]
    inside = []
    for start in range(0, points.shape[0], chunk):
        c = tris[None] - points[start:start + chunk, None, None]
        norms = c.norm(dim=-1)
        numerator = (c[:, :, 0] * torch.cross(c[:, :, 1], c[:, :, 2], dim=-1)).sum(-1)
        dot01 = (c[:, :, 0] * c[:, :, 1]).sum(-1)
        dot12 = (c[:, :, 1] * c[:, :, 2]).sum(-1)
        dot02 = (c[:, :, 0] * c[:, :, 2]).sum(-1)
        denominator = (
            norms.prod(-1) + dot01 * norms[:, :, 2] + dot02 * norms[:, :, 1] + dot12 * norms[:, :, 0]
        )
        winding = torch.atan2(numerator, denominator).sum(-1) / (2 * torch.pi)
        inside.append(winding.round().bool())
    return torch.cat(inside)


def armature_from_body_model(body_model, res: float = 0.02, device: torch.device | None = None) -> Armature:
    """Mass, CoM and inertia of the 24 SMPL bodies of a rest-pose body model.

    Port of articulate ``ParametricModel.physical_properties`` (mesh voxelized
    at ``res``, 1000 kg/m^3, skinning weights interpolated inside the mesh).
    Native hand joints are merged into the palm bodies 22/23 and face joints
    into the head, so SMPL-H/SMPL-X models give the same 24 bodies as SMPL.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vert = body_model.init_v_template[0].detach().float().cpu()
    joint = build_smpl24_joint_positions(body_model.J_regressor.detach().float().cpu() @ vert)
    root_offset = joint[0].clone()
    vert, joint = vert - root_offset, joint - root_offset

    native_parent = body_model.kintree_table[0].tolist()
    body = []
    for k in range(len(native_parent)):
        j = k
        while j >= 22:
            j = native_parent[j]
        body.append(k if k < 22 else {20: 22, 21: 23}.get(j, j))
    native_weight = body_model.weights.detach().float().cpu()
    weight = torch.zeros(native_weight.shape[0], 24).index_add_(1, torch.tensor(body), native_weight)

    # Slice the mesh along x and keep the grid points inside it.
    faces = torch.as_tensor(body_model.f).long().cpu()
    vert_d, faces_d = vert.to(device), faces.to(device)
    points = []
    for x in torch.arange(vert[:, 0].min().item(), vert[:, 0].max().item(), res):
        v = vert[(vert[:, 0] - x).abs() < max(0.04, res)]
        grid = torch.meshgrid(
            x.view(1),
            torch.arange(v[:, 1].min().item(), v[:, 1].max().item(), res),
            torch.arange(v[:, 2].min().item(), v[:, 2].max().item(), res),
            indexing="ij",
        )
        grid = torch.stack(grid, dim=-1).view(-1, 3)
        points.append(grid[_inside_mesh(grid.to(device), vert_d, faces_d).cpu()])
    point = torch.cat(points).numpy().astype(np.float64)

    point_weight = np.nan_to_num(LinearNDInterpolator(vert.numpy(), weight.numpy())(point)).T
    weighted_mass = point_weight * 1000.0 * res ** 3
    joint = joint.numpy().astype(np.float64)
    mass = weighted_mass.sum(axis=1)
    com = np.einsum("ij,ijk->ik", weighted_mass, point[None] - joint[:, None]) / mass[:, None]
    # As released: the inertia is taken about ``point - com`` although ``com``
    # is joint-relative, which inflates the inertia of bodies away from the
    # root.  Kept, because the optimizer weights were tuned with these values.
    r = point[None] - com[:, None]
    inertia = (
        np.einsum("ij,ij->i", weighted_mass, (r * r).sum(-1))[:, None, None] * np.eye(3)
        - np.einsum("ij,ijk,ijl->ikl", weighted_mass, r, r)
    )
    bone = joint.copy()
    bone[1:] = joint[1:] - joint[list(SMPL24_PARENTS[1:])]
    return Armature(SMPL24_PARENTS, bone, com, mass, inertia, root_offset.numpy().astype(np.float64))


class DynamicModel:
    """NumPy port of carticulate ``DynamicModel`` (the parts GlobalPose uses).

    Generalized velocity: root linear velocity (world frame), then one angular
    velocity per joint expressed in that joint's own frame, so a state update
    right-multiplies every local rotation by exp(delta).
    """

    def __init__(self, armature: Armature):
        self.parent = tuple(int(p) for p in armature.parent)
        if self.parent[0] >= 0 or min(self.parent[1:]) < 0:
            raise ValueError("joint 0 must be the only root")
        self.n = len(self.parent)
        self.bone = np.asarray(armature.bone, dtype=np.float64)
        self.com = np.asarray(armature.com, dtype=np.float64)
        self.mass = np.asarray(armature.mass, dtype=np.float64)
        self.inertia = np.asarray(armature.inertia, dtype=np.float64)
        self.gravity = np.asarray(armature.gravity, dtype=np.float64)
        self.chain = []
        for j in range(self.n):
            chain = [j]
            while self.parent[chain[-1]] >= 0:
                chain.append(self.parent[chain[-1]])
            self.chain.append(np.array(chain[::-1]))
        self.ancestor = np.zeros((self.n, self.n))
        for j, chain in enumerate(self.chain):
            self.ancestor[j, chain] = 1.0
        # Mass-matrix factors: m_i = sqrt(m_i)^2 and I_i = L_i L_i^T.
        self._sqrt_mass = np.sqrt(self.mass)
        self._inertia_factor = np.linalg.cholesky(self.inertia)
        self.set_state_R(np.tile(np.eye(3), (self.n, 1, 1)), np.zeros(3), np.zeros(3 * self.n + 3))

    def _forward_kinematics(self) -> None:
        R_wb = np.empty_like(self.R_pb)
        t_wb = np.empty_like(self.t_pb)
        R_wb[0], t_wb[0] = self.R_pb[0], self.t_pb[0]
        for i in range(1, self.n):
            p = self.parent[i]
            R_wb[i] = R_wb[p] @ self.R_pb[i]
            t_wb[i] = R_wb[p] @ self.t_pb[i] + t_wb[p]
        self.R_wb, self.t_wb = R_wb, t_wb

    def set_state_R(self, pose, tran, vel) -> None:
        self.R_pb = np.array(pose, dtype=np.float64).reshape(self.n, 3, 3)
        self.t_pb = self.bone.copy()
        self.t_pb[0] += np.asarray(tran, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64).reshape(3 * self.n + 3)
        self._forward_kinematics()

    def get_state_R(self):
        return self.R_pb.copy(), self.t_pb[0] - self.bone[0], self.vel.copy()

    def update_state(self, acc: np.ndarray, delta_t: float) -> None:
        delta = self.vel * delta_t + 0.5 * acc * delta_t * delta_t
        self.R_pb = self.R_pb @ Rotation.from_rotvec(delta[3:].reshape(self.n, 3)).as_matrix()
        self.t_pb[0] += delta[:3]
        self._forward_kinematics()
        self.vel = self.vel + acc * delta_t

    def joint_rates(self) -> np.ndarray:
        """World-frame angular velocity contributed by each joint."""
        return np.einsum("nij,nj->ni", self.R_wb, self.vel[3:].reshape(self.n, 3))

    def get_position(self, joint: int) -> np.ndarray:
        return self.t_wb[joint].copy()

    def get_linear_velocity(self, joint: int, rates: np.ndarray | None = None) -> np.ndarray:
        rates = self.joint_rates() if rates is None else rates
        chain = self.chain[joint]
        return self.vel[:3] + np.cross(rates[chain], self.t_wb[joint] - self.t_wb[chain]).sum(axis=0)

    def _columns(self, chain: np.ndarray) -> np.ndarray:
        return (3 + 3 * chain[:, None] + np.arange(3)).reshape(-1)

    def get_linear_Jacobian(self, joint: int) -> np.ndarray:
        chain = self.chain[joint]
        J = np.zeros((3, 3 * self.n + 3))
        J[:, :3] = np.eye(3)
        blocks = -_hat(self.t_wb[joint] - self.t_wb[chain]) @ self.R_wb[chain]
        J[:, self._columns(chain)] = blocks.transpose(1, 0, 2).reshape(3, -1)
        return J

    def get_linear_Jacobian_dot(self, joint: int, rates: np.ndarray | None = None) -> np.ndarray:
        rates = self.joint_rates() if rates is None else rates
        chain = self.chain[joint]
        omega = np.cumsum(rates[chain], axis=0)  # angular velocity of each chain body
        t = self.t_wb[chain]
        segment = np.zeros_like(t)
        segment[:-1] = t[1:] - t[:-1]
        # velocity of the joint relative to each chain joint
        relative = np.cumsum(np.cross(omega, segment)[::-1], axis=0)[::-1]
        blocks = -(_hat(relative) + _hat(self.t_wb[joint] - t) @ _hat(omega)) @ self.R_wb[chain]
        J = np.zeros((3, 3 * self.n + 3))
        J[:, self._columns(chain)] = blocks.transpose(1, 0, 2).reshape(3, -1)
        return J

    def mass_matrix(self) -> np.ndarray:
        """Joint-space inertia: sum over bodies of m Jc'Jc + Jw' I_world Jw."""
        n = self.n
        mask = self.ancestor[..., None, None]
        com = self.t_wb + np.einsum("nij,nj->ni", self.R_wb, self.com)
        jc = (-_hat(com[:, None] - self.t_wb[None]) @ self.R_wb[None]) * mask  # [body, joint, 3, 3]
        jc = jc.transpose(0, 2, 1, 3).reshape(n, 3, 3 * n)
        jw = (self.R_wb[None] * mask).transpose(0, 2, 1, 3).reshape(n, 3, 3 * n)
        jc_scaled = (self._sqrt_mass[:, None, None] * jc).reshape(3 * n, 3 * n)
        factor = self.R_wb @ self._inertia_factor  # I_world = (R L)(R L)^T
        jw_scaled = np.einsum("nba,nbk->nak", factor, jw).reshape(3 * n, 3 * n)
        M = np.empty((3 * n + 3, 3 * n + 3))
        M[:3, :3] = np.eye(3) * self.mass.sum()
        M[:3, 3:] = np.einsum("i,iak->ak", self.mass, jc)
        M[3:, :3] = M[:3, 3:].T
        M[3:, 3:] = jc_scaled.T @ jc_scaled + jw_scaled.T @ jw_scaled
        return M

    def inverse_dynamics(self, acc: np.ndarray) -> np.ndarray:
        """Recursive Newton-Euler (carticulate ``inverse_dynamics``, no external force)."""
        n = self.n
        vel = self.vel[3:].reshape(n, 3)
        joint_acc = acc[3:].reshape(n, 3)
        w, wd, v, vd = (np.empty((n, 3)) for _ in range(4))
        w[0], wd[0], v[0] = vel[0], joint_acc[0], 0.0
        vd[0] = self.R_wb[0].T @ (acc[:3] - self.gravity)
        for i in range(1, n):
            p = self.parent[i]
            Rt, t = self.R_pb[i].T, self.t_pb[i]
            w[i] = Rt @ w[p] + vel[i]
            wd[i] = Rt @ wd[p] + _cross(w[i], vel[i]) + joint_acc[i]
            v[i] = Rt @ (v[p] + _cross(w[p], t))
            vd[i] = Rt @ (vd[p] + _cross(wd[p], t)) + _cross(v[i], vel[i])
        c = self.com
        fcom = self.mass[:, None] * (
            vd + np.cross(w, v) + np.cross(w, np.cross(w, c)) + np.cross(wd, c)
        )
        tau = (
            np.einsum("nij,nj->ni", self.inertia, wd)
            + np.cross(w, np.einsum("nij,nj->ni", self.inertia, w))
            + np.cross(c, fcom)
        )
        f = fcom
        for i in range(n - 1, 0, -1):
            p = self.parent[i]
            f_parent = self.R_pb[i] @ f[i]
            f[p] += f_parent
            tau[p] += self.R_pb[i] @ tau[i] + _cross(self.t_pb[i], f_parent)
        out = np.empty(3 * n + 3)
        out[:3] = self.R_pb[0] @ f[0]
        out[3:] = tau.reshape(-1)
        return out


class GlobalPosePhysics:
    """Physics part of ``GPNet.forward_frame`` (released net.py), one step per call."""

    dt = 1 / 60
    mu = 0.7               # environment friction coefficient
    kp_pose = 3600         # pose PD controller
    kd_pose = 60
    kp_tran = 3600         # contact-joint PD controller
    kd_tran = 60
    alpha_pd = 0.0         # stable-PD relaxation
    beta_cjoint = 1
    beta_extforce = 0.4
    beta_torque = 1e-3 / 80

    def __init__(self, armature: Armature):
        self.model = DynamicModel(armature)
        n_rot = 3 * self.model.n
        self.B = np.array([[self.mu, -self.mu, 0, 0],
                           [1, 1, 1, 1],
                           [0, 0, self.mu, -self.mu]]) / np.sqrt(1 + self.mu ** 2)  # friction cone basis
        self.pose_rows = np.hstack((np.zeros((n_rot, 3)), np.eye(n_rot)))
        self.floor_y = 0.0
        self.contact = np.zeros(len(CONTACT_JOINTS), dtype=bool)
        self.contact_counter = np.zeros(len(CONTACT_JOINTS), dtype=int)

    def reset(self, pose: np.ndarray, root: np.ndarray) -> None:
        """Place the first network pose with its root joint at ``root``.

        As released, the lowest contact joint of this frame defines the floor.
        """
        self.model.set_state_R(pose, root, np.zeros(3 * self.model.n + 3))
        self.floor_y = float(self.model.t_wb[list(CONTACT_JOINTS), 1].min())
        self.contact[:] = False
        self.contact_counter[:] = 0

    def state(self):
        pose, tran, _ = self.model.get_state_R()
        return pose, tran

    def _explain_residual_force(self, contact, contact_jacobian, contact_position, residual_force):
        if np.any(contact):
            J = contact_jacobian.reshape(len(CONTACT_JOINTS), 3, -1)[contact, :, :6].reshape(-1, 6)
            B, lb = [], []
            for i in np.where(contact)[0]:
                # Released condition; ``i > 3`` admits only the right palm as a grasp.
                if i > 3 and contact_position[i, 1] > self.floor_y + 0.15:  # hand grasp
                    B.append(np.eye(3))
                    lb.append(-np.ones(3) * np.inf)
                else:
                    B.append(self.B)
                    lb.append(np.zeros(4))
            JTB = J.T @ block_diag(*B)
            # solve_qp(P = JTB'JTB + beta I, q = -JTB'r, lb) as bounded least squares
            n_force = JTB.shape[1]
            A = np.vstack((JTB, np.sqrt(self.beta_extforce) * np.eye(n_force)))
            b = np.concatenate((residual_force, np.zeros(n_force)))
            lb = np.concatenate(lb)
            if np.isinf(lb).all():
                force = np.linalg.lstsq(A, b, rcond=None)[0]
            else:
                force = lsq_linear(A, b, bounds=(lb, np.inf), method="bvls").x
            error = np.linalg.norm(JTB @ force - residual_force)
        else:
            B = None
            force = np.zeros(0)
            error = np.linalg.norm(residual_force)
        return force, error, B

    def step(self, pose: np.ndarray, velocity: np.ndarray, stationary_prob: np.ndarray, cjoint: np.ndarray) -> None:
        """Advance the simulated character one ``dt`` toward a network frame.

        ``pose`` [24, 3, 3] local rotations, ``velocity`` [3] fused root velocity,
        ``stationary_prob`` [5], ``cjoint`` [5, 3] contact joints relative to the
        root joint in world orientation; all from the same network frame.
        """
        m = self.model
        pose_cur, tran_cur, qdot = m.get_state_R()
        rates = m.joint_rates()
        cjoint_cur = m.t_wb[list(CONTACT_JOINTS)].copy()
        cvel_cur = np.stack([m.get_linear_velocity(j, rates) for j in CONTACT_JOINTS])
        cJ_cur = np.vstack([m.get_linear_Jacobian(j) for j in CONTACT_JOINTS])
        cJdot_cur = np.vstack([m.get_linear_Jacobian_dot(j, rates) for j in CONTACT_JOINTS])
        M = m.mass_matrix()
        h = m.inverse_dynamics(np.zeros_like(qdot))
        stationary = stationary_prob > 0.7
        stationary_weight = np.clip(stationary_prob * 5 - 3, 0, 1)

        # dual RSPD controller
        R = Rotation.from_rotvec(qdot[3:].reshape(-1, 3) * self.alpha_pd * self.dt).as_matrix()
        delta_pose = Rotation.from_matrix(np.swapaxes(pose_cur @ R, 1, 2) @ pose).as_rotvec().ravel()
        thetaddotdes = (self.kp_pose * delta_pose - self.kd_pose * qdot[3:]) / (1 + self.kd_pose * self.alpha_pd * self.dt)
        target = tran_cur + velocity * self.dt + cjoint
        target = target + (cjoint_cur - target) * stationary_weight[:, None]
        delta_tran = target - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
        rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)

        # unconstrained tracking
        cjoint_cur[0, 1] -= 0.15
        target[0, 1] -= 0.15
        k = np.ones((len(CONTACT_JOINTS), 3)) * self.beta_cjoint
        k[self.contact] *= 10
        sqrt_k = np.sqrt(k.reshape(-1))
        A = np.vstack((self.pose_rows, sqrt_k[:, None] * cJ_cur, np.sqrt(self.beta_torque) * M))
        b = np.concatenate((thetaddotdes, sqrt_k * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque) * (-h)))
        qddot = lsqr(csc_array(A), b)[0]
        residual_force = M[:6] @ qddot + h[:6]

        # determine potential contact
        vdist = np.abs(cjoint_cur[np.newaxis, :, 1] - cjoint_cur[:, np.newaxis, 1])
        contact = stationary & (self.contact | (cjoint_cur[:, 1] < self.floor_y + 0.05))
        if np.any(contact):
            contact |= stationary & (vdist[contact].min(axis=0) < 0.05)
        potential_contact = stationary & ~contact
        if contact[0] or potential_contact[0]:  # root joint
            lleg = m.t_wb[4] - m.t_wb[1]
            rleg = m.t_wb[5] - m.t_wb[2]
            if min(np.arccos(-lleg[1] / np.linalg.norm(lleg)), np.arccos(-rleg[1] / np.linalg.norm(rleg))) < np.pi / 4:
                contact[0], potential_contact[0] = False, False

        # explain residual force by contacts
        force, err, forceB = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
        for i in np.argsort(cjoint_cur[:, 1]):  # add potential contact from lowest to highest
            if err > 400 and potential_contact[i]:
                contact[i] = True
                force_new, err_new, forceB_new = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
                self.contact_counter[i] = self.contact_counter[i] + 1 if err_new / err < 0.6 else 0
                if self.contact_counter[i] >= 5:
                    force, err, forceB = force_new, err_new, forceB_new
                else:
                    contact[i] = False
            else:
                self.contact_counter[i] = 0

        # update contact position
        near_ground = target[:, 1] < self.floor_y + 0.15
        for i in np.where(contact & near_ground)[0]:
            target[i, 1] += (self.floor_y - target[i, 1]) * 0.1
        target[target[:, 1] < self.floor_y, 1] = self.floor_y

        # re-optimization and update state
        delta_tran = target - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
        rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)
        if np.any(contact):
            J = cJ_cur.reshape(len(CONTACT_JOINTS), 3, -1)[contact].reshape(-1, cJ_cur.shape[1])
            torque = J.T @ (block_diag(*forceB) @ force)
        else:
            torque = np.zeros_like(qdot)
        A = np.vstack((self.pose_rows, sqrt_k[:, None] * cJ_cur, np.sqrt(self.beta_torque * 3) * M))
        b = np.concatenate((thetaddotdes, sqrt_k * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque * 3) * (-h + torque)))
        qddot = lsqr(csc_array(A), b)[0]
        m.update_state(qddot, self.dt)
        self.contact = contact


def run_globalpose_physics(
    armature: Armature,
    pose: np.ndarray,
    velocity: np.ndarray,
    stationary_prob: np.ndarray,
    cjoint: np.ndarray,
    root: np.ndarray,
    fps: float,
):
    """Causal physics refinement of one sequence of network frames.

    Frame 0 places the network pose with its root joint at ``root``.  Between
    frames, the optimizer takes ``1 / (fps * dt)`` steps toward targets
    interpolated from the previous to the current frame (slerp for rotations),
    so the last step of every frame tracks that frame exactly and no future
    frame is used.  Returns the refined local pose [T, 24, 3, 3] and root joint
    positions [T, 3].
    """
    steps = round(1.0 / (fps * GlobalPosePhysics.dt))
    if steps < 1 or abs(steps * fps * GlobalPosePhysics.dt - 1.0) > 1e-6:
        raise ValueError(f"fps {fps} is not a divisor of the physics rate {1 / GlobalPosePhysics.dt:g} Hz")
    physics = GlobalPosePhysics(armature)
    physics.reset(pose[0], root)
    refined_pose = np.empty((pose.shape[0], physics.model.n, 3, 3))
    refined_root = np.empty((pose.shape[0], 3))
    refined_pose[0], refined_root[0] = physics.state()
    for frame in range(1, pose.shape[0]):
        previous, current = pose[frame - 1], pose[frame]
        pose_delta = Rotation.from_matrix(np.swapaxes(previous, 1, 2) @ current).as_rotvec()
        for step in range(1, steps + 1):
            alpha = step / steps
            target_pose = (
                current if step == steps
                else previous @ Rotation.from_rotvec(pose_delta * alpha).as_matrix()
            )
            physics.step(
                target_pose,
                *(x[frame - 1] + (x[frame] - x[frame - 1]) * alpha for x in (velocity, stationary_prob, cjoint)),
            )
        refined_pose[frame], refined_root[frame] = physics.state()
    return refined_pose, refined_root
