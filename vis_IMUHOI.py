import torch
import os
import sys
import threading
import time as _walltime
import numpy as np
import random
import argparse
from dataclasses import dataclass
import yaml
import re
import types
from pathlib import Path
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.viewer import Viewer
from aitviewer.scene.camera import Camera
from moderngl_window.context.base import KeyModifiers
import trimesh
from easydict import EasyDict as edict

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytorch3d.transforms as transforms
except Exception:
    from utils import rotation_conversions as _rot

    pytorch3d_mod = types.ModuleType("pytorch3d")
    transforms = types.ModuleType("pytorch3d.transforms")
    transforms.rotation_6d_to_matrix = _rot.rotation_6d_to_matrix
    transforms.matrix_to_rotation_6d = _rot.matrix_to_rotation_6d
    transforms.matrix_to_axis_angle = _rot.matrix_to_axis_angle
    pytorch3d_mod.transforms = transforms
    sys.modules["pytorch3d"] = pytorch3d_mod
    sys.modules["pytorch3d.transforms"] = transforms

from dataset.dataset_IMUHOI import IMUDataset
from process.preprocess import load_object_geometry
from model import IMUHOIModel, load_model
from model.rnn.velocity_contact import (
    VELOCITY_CONTACT_ARCH_CHOICES,
    normalize_velocity_contact_arch,
)
from model.rnn.object_trans_IMUHOI import is_imuhoi_object_trans_arch
from utils.dataset_config import (
    get_dataset_configs,
    get_full_sequence_test_path,
    resolve_dataset_path,
)
from utils.vis_model_spec import resolve_vis_models
from utils.utils import (
    apply_eval_imu_noise_to_sequence,
    load_checkpoint,
    load_config,
    load_smpl_model,
    build_model_input_dict,
)
from configs import (
    FRAME_RATE,
    _SENSOR_NAMES,
    _SENSOR_VEL_NAMES,
    _REDUCED_POSE_NAMES,
    _REDUCED_INDICES,
    _IGNORED_INDICES,
    _SENSOR_ROT_INDICES,
    _SENSOR_POS_INDICES,
    _VEL_SELECTION_INDICES,
)

PROJECT_ROOT = Path(__file__).resolve().parent

import imgui
try:
    from aitviewer.renderables.lines import Lines
except Exception as exc:
    raise ImportError(
        "aitviewer.renderables.lines.Lines is required for trajectory visualization. "
        "Please install an aitviewer version that provides this renderable."
    ) from exc

R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

HAND_TRAJ_RADIUS = 0.004
# Probability -> opacity mapping shared by --show_hand_traj and the predicted
# object under --show_obj_move: alpha = MIN + (MAX - MIN) * prob.
PROB_MIN_ALPHA = 0.08
PROB_MAX_ALPHA = 0.95

HAND_TRAJ_COLORS = {
    "gt_l": (1.0, 0.0, 0.0, 1),
    "gt_r": (0.0, 0.0, 1.0, 1),
    "pred_l": (1.0, 0.5, 0.0, 1),
    "pred_r": (0.0, 0.6, 1.0, 1),
}

GT_HUMAN_COLOR = (118/255, 147/255, 248/255, 0.9)
GT_OBJECT_COLOR = (186/255, 161/255, 246/255, 0.9)
# --show_obj_move: GT and predicted objects are coloured per frame from obj_move
# (GT: obj_contact label; prediction: pred_obj_move_prob > 0.5).
OBJ_MOVE_MOVING_COLOR = (0.95, 0.40, 0.15, 0.9)
OBJ_MOVE_STATIC_COLOR = GT_OBJECT_COLOR

PRED_HUMAN_COLOR = (246 / 255, 153 / 255, 19 / 255, 153/255)
PRED_OBJECT_COLOR = (255 / 255, 82 / 255, 0 / 255, 153/255)

# One colour set per prediction lane (repeated --model); lane 1 is also the
# single-model set.  GT owns blue/lavender, so no lane uses them.  Under
# --show_obj_move the object keeps the shared moving/static colours instead.
PRED_LANE_COLORS = (
    {"name": "orange", "human": PRED_HUMAN_COLOR, "object": PRED_OBJECT_COLOR,
     "lhand": HAND_TRAJ_COLORS["pred_l"], "rhand": HAND_TRAJ_COLORS["pred_r"]},
    {"name": "green", "human": (70 / 255, 185 / 255, 90 / 255, 153 / 255),
     "object": (0 / 255, 125 / 255, 55 / 255, 153 / 255),
     "lhand": (0.45, 0.85, 0.2, 1), "rhand": (0.0, 0.6, 0.45, 1)},
    {"name": "magenta", "human": (220 / 255, 90 / 255, 200 / 255, 153 / 255),
     "object": (150 / 255, 30 / 255, 135 / 255, 153 / 255),
     "lhand": (1.0, 0.35, 0.7, 1), "rhand": (0.65, 0.3, 1.0, 1)},
    {"name": "yellow", "human": (230 / 255, 205 / 255, 50 / 255, 153 / 255),
     "object": (170 / 255, 135 / 255, 0 / 255, 153 / 255),
     "lhand": (1.0, 0.9, 0.0, 1), "rhand": (0.8, 0.5, 0.0, 1)},
    {"name": "teal", "human": (40 / 255, 190 / 255, 190 / 255, 153 / 255),
     "object": (0 / 255, 120 / 255, 120 / 255, 153 / 255),
     "lhand": (0.2, 0.9, 0.9, 1), "rhand": (0.0, 0.55, 0.6, 1)},
)


def _lane_colors(lane):
    """Colour set of 1-based prediction lane ``lane`` (cycles past the palette)."""
    return PRED_LANE_COLORS[(int(lane) - 1) % len(PRED_LANE_COLORS)]


def _lane_offset(pred_offset, lane):
    """Lane ``lane`` sits ``lane`` steps of --pred_offset from GT."""
    step = np.asarray((1.0, 0.0, 0.0) if pred_offset is None else pred_offset, dtype=np.float32)
    return (int(lane) * step).astype(np.float32)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _get_human_pose_module(model):
    core = _unwrap_model(model)
    module = getattr(core, "human_pose_module", None)
    if module is not None:
        return module
    if core.__class__.__name__.lower() == "humanposemodule":
        return core
    wrapped = getattr(core, "module", None)
    if wrapped is not None:
        return getattr(wrapped, "human_pose_module", None)
    return None


def _infer_model_kind(model):
    core = _unwrap_model(model)
    name = core.__class__.__name__.lower()
    if "humanposemodule" in name:
        return "human_pose"
    if "imuhoimixmodule" in name:
        return "mix"
    if "imuhoimodel" in name:
        return "pipeline"
    if "interactionmodule" in name:
        return "interaction"
    if hasattr(core, "human_pose_module") and hasattr(core, "interaction_module"):
        return "pipeline"
    return "unknown"


@dataclass
class ComparisonModel:
    label: str
    config: object
    model: object
    ot_refine: bool


def _load_comparison_models(specs, dataset, no_trans=False):
    """Strictly load each manifest variant, parked on CPU between predictions."""
    cpu = torch.device("cpu")
    bundles = []
    for spec in specs:
        config = load_config(str(spec.config_path))
        if str(getattr(config, "model_arch", "rnn")).lower() != "rnn":
            raise ValueError(f"{spec.label}: --model variant directories currently require an RNN pipeline")
        ot_arch = _normalize_object_trans_arch(getattr(config, "object_trans_arch", spec.object_trans_arch))
        vc_arch = normalize_velocity_contact_arch(getattr(config, "velocity_contact_arch", spec.velocity_contact_arch))
        if ot_arch.lower() != spec.object_trans_arch.lower() or vc_arch != normalize_velocity_contact_arch(spec.velocity_contact_arch):
            raise ValueError(f"{spec.label}: config OT/VC architectures do not match its experiment manifest")
        if dataset not in get_dataset_configs(config):
            raise ValueError(f"{spec.label}: dataset {dataset!r} is absent from its config")
        config.device = str(cpu)
        config.object_trans_arch = ot_arch
        config.velocity_contact_arch = vc_arch
        model = load_model(config, cpu, no_trans=no_trans)
        for module_name, path in spec.checkpoints.items():
            module = getattr(_unwrap_model(model), f"{module_name}_module", None)
            if module is None:
                raise ValueError(f"{spec.label}: missing {module_name} module for checkpoint {path}")
            # Modular RNN default loading is non-strict for VC/OT and can leave
            # randomly initialized layers in a mismatched architecture. A
            # comparison must fail instead of displaying that partial model.
            load_checkpoint(module, str(path), cpu, strict=True)
        model.eval()
        print(f"[Vis] {spec.label}: OT={ot_arch}, VC={vc_arch}, config={spec.config_path}")
        for module_name, path in spec.checkpoints.items():
            print(f"[Vis]   {module_name}: {path}")
        bundles.append(ComparisonModel(
            spec.label, config, model,
            False if is_imuhoi_object_trans_arch(ot_arch) else bool(getattr(config, "enable_ot_refine", False)),
        ))
    return bundles


def _comparison_prediction_length(prediction, batch_length):
    """Shared display length across the full GT sequence and every prediction."""
    source = prediction.get("p_pred")
    if not isinstance(source, torch.Tensor):
        source = prediction.get("root_trans_pred")
    if not isinstance(source, torch.Tensor) or source.ndim < 2:
        raise ValueError("Pipeline prediction is missing a time-indexed human pose/root")
    length = min(int(batch_length), int(source.shape[1]))
    if length < 1:
        raise ValueError("Online inference produced no publishable frames")
    return length


def _move_comparison_model(bundle, device):
    bundle.model.to(device)
    core = _unwrap_model(bundle.model)
    core.device = device
    human = _get_human_pose_module(bundle.model)
    if human is not None:
        human.device = device
        if getattr(human, "body_model", None) is not None:
            human.body_model_device = device


def _global_to_local_rotmat(global_rotmats, parents):
    if global_rotmats.dim() != 4 or global_rotmats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [T, J, 3, 3], got {tuple(global_rotmats.shape)}")
    seq_len, num_joints = global_rotmats.shape[:2]
    local_rotmats = torch.zeros_like(global_rotmats)
    local_rotmats[:, 0] = global_rotmats[:, 0]
    for i in range(1, num_joints):
        parent_idx = int(parents[i]) if i < len(parents) else 0
        parent_idx = max(0, min(parent_idx, num_joints - 1))
        local_rotmats[:, i] = torch.matmul(global_rotmats[:, parent_idx].transpose(-1, -2), global_rotmats[:, i])
    valid_ignored = [idx for idx in _IGNORED_INDICES if idx < num_joints]
    if valid_ignored:
        eye = torch.eye(3, device=global_rotmats.device, dtype=global_rotmats.dtype).view(1, 1, 3, 3)
        local_rotmats[:, valid_ignored] = eye.expand(seq_len, len(valid_ignored), 3, 3)
    return local_rotmats


class SequenceCancelled(BaseException):
    """Raised inside a sequence job once the user has asked for another one.

    It derives from BaseException on purpose: the visualization pipeline wraps
    every stage in ``except Exception`` so that one failing branch still shows
    the rest, and a cancellation must pass straight through those handlers.
    """


_JOB_CONTEXT = threading.local()


def _raise_if_cancelled():
    """Cancellation checkpoint; a no-op outside a background sequence job."""
    job = getattr(_JOB_CONTEXT, "job", None)
    if job is not None and job.cancel.is_set():
        raise SequenceCancelled()


def _cancel_check_callback(_completed=None, _total=None):
    """Online-inference progress hook: checked once per published frame."""
    _raise_if_cancelled()


def _format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}m{rest:02d}s"


class _ProgressLine:
    """A single terminal line that long inference passes keep overwriting.

    Worker and GL thread both print, so writes go through one lock, and any
    other message first calls ``break_line`` so it does not land on the end of
    a half-drawn progress line.  When stdout is not a terminal, ``\\r`` would
    only produce one enormous line, so updates become ordinary lines at a
    slower rate instead.
    """

    TTY_INTERVAL = 0.25
    PLAIN_INTERVAL = 10.0

    def __init__(self, stream=None):
        self._stream = stream
        self._lock = threading.Lock()
        self._open = False

    @property
    def stream(self):
        return self._stream if self._stream is not None else sys.stdout

    def is_tty(self):
        try:
            return bool(self.stream.isatty())
        except Exception:
            return False

    def interval(self):
        return self.TTY_INTERVAL if self.is_tty() else self.PLAIN_INTERVAL

    @staticmethod
    def _fit(label, body, width):
        """``[label] body`` within ``width``, shortening the label first."""
        text = f"[{label}] {body}"
        if len(text) <= width:
            return text
        room = width - len(body) - 3
        if room >= 8:
            return f"[{label[:room - 3]}...] {body}"
        return body[:width]

    def update(self, label, body, final=False):
        with self._lock:
            stream = self.stream
            if self.is_tty():
                import shutil

                # A wrapped line cannot be overwritten with \r, so clip it,
                # keeping the numbers.  A pty with no size set reports 0
                # columns: treat that as unknown.
                columns = shutil.get_terminal_size((120, 20)).columns
                width = (columns if columns > 0 else 120) - 1
                text = self._fit(label, body, width)
                stream.write("\r" + text + "\x1b[K" + ("\n" if final else ""))
                self._open = not final
            else:
                stream.write(f"[{label}] {body}\n")
                self._open = False
            stream.flush()

    def break_line(self):
        with self._lock:
            if self._open:
                self.stream.write("\n")
                self.stream.flush()
                self._open = False


_PROGRESS_LINE = _ProgressLine()


class _InferenceProgress:
    """Online progress callback: cancellation checkpoint plus a live ETA line.

    The pipeline reports the warmup block in one call and then every published
    frame, so the rate is measured from the first report: the warmup is a
    single batched forward and would make the ETA far too optimistic.
    """

    def __init__(self, label, line=None, clock=None):
        self.label = label
        self.line = line if line is not None else _PROGRESS_LINE
        self.clock = clock if clock is not None else _walltime.perf_counter
        self.started = self.clock()
        self._first = None
        self._last_draw = None
        self._finished = False

    def __call__(self, completed=None, total=None):
        _raise_if_cancelled()
        if completed is None or not total or self._finished:
            return
        now = self.clock()
        completed, total = int(completed), int(total)
        if self._first is None:
            self._first = (completed, now)
        final = completed >= total
        if not final and self._last_draw is not None and now - self._last_draw < self.line.interval():
            return
        self._last_draw = now
        elapsed = now - self.started
        first_done, first_at = self._first
        if final:
            self._finished = True
            fps = completed / elapsed if elapsed > 0 else 0.0
            body = f"{completed}/{total} frames in {_format_seconds(elapsed)} ({fps:.1f} fps)"
        else:
            span = now - first_at
            rate = (completed - first_done) / span if span > 0 else 0.0
            eta = _format_seconds((total - completed) / rate) if rate > 0 else "--"
            body = (f"{completed}/{total} frames {100.0 * completed / total:5.1f}% | "
                    f"{rate:.1f} fps | elapsed {_format_seconds(elapsed)} | ETA {eta}")
        self.line.update(self.label, body, final=final)

    def close(self):
        """End a line that an interruption or an error left half drawn."""
        self.line.break_line()


def _progress_label(model_label=None):
    job = getattr(_JOB_CONTEXT, "job", None)
    parts = [part for part in (getattr(job, "label", None), model_label) if part]
    return " | ".join(parts) or "inference"


class _RecordingScene:
    """Scene stand-in for a worker thread: reads pass through, edits are queued.

    aitviewer nodes may only be added, removed and released on the GL thread,
    so a background job records its scene edits and the render loop applies
    them in one go once the job has finished.
    """

    def __init__(self, scene, ops):
        self._scene = scene
        self._ops = ops

    def add(self, *nodes, **kwargs):
        self._ops.append(("add", nodes, kwargs))

    def remove(self, *nodes):
        self._ops.append(("remove", nodes, {}))

    def collect_nodes(self, *args, **kwargs):
        return list(self._scene.collect_nodes(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._scene, name)


class _WorkerViewerProxy:
    """The viewer as a background job sees it: a recording scene, shared state."""

    def __init__(self, viewer, ops):
        object.__setattr__(self, "_viewer", viewer)
        object.__setattr__(self, "scene", _RecordingScene(viewer.scene, ops))

    def __getattr__(self, name):
        return getattr(self._viewer, name)

    def __setattr__(self, name, value):
        setattr(self._viewer, name, value)


class _LaneScene:
    """Name every prediction node after its model, without touching GT nodes."""

    def __init__(self, scene, label):
        self._scene = scene
        self._label = label

    def add(self, *nodes, **kwargs):
        for node in nodes:
            if getattr(node, "name", None) and not node.name.startswith("GT-"):
                node.name = f"Pred-{self._label}::{node.name}"
        self._scene.add(*nodes, **kwargs)

    def __getattr__(self, name):
        return getattr(self._scene, name)


class _LaneViewer:
    def __init__(self, viewer, label, config):
        object.__setattr__(self, "_viewer", viewer)
        object.__setattr__(self, "scene", _LaneScene(viewer.scene, label))
        object.__setattr__(self, "config", config)

    def __getattr__(self, name):
        return getattr(self._viewer, name)

    def __setattr__(self, name, value):
        setattr(self._viewer, name, value)


class _SequenceJob:
    def __init__(self, index, generation):
        self.index = index
        self.generation = generation
        self.label = None            # sequence name, prefixes the progress line
        self.cancel = threading.Event()
        self.ops = []
        self.status = "running"      # running | done | cancelled | failed
        self.title = None
        self.started_at = _walltime.perf_counter()
        self.finished_at = None
        self.cancel_requested_at = None


def _run_model_prediction(
    model,
    model_input,
    batch_device,
    compute_fk_flag,
    model_arch,
    ot_refine=False,
    inference_mode="online",
    online_window=None,
    online_lookahead=5,
    online_chunk=1,
    progress_label=None,
):
    model_kind = _infer_model_kind(model)

    def _call_inference(*args, **kwargs):
        kwargs.setdefault("inference_mode", inference_mode)
        kwargs.setdefault("online_window", online_window)
        kwargs.setdefault("online_lookahead", online_lookahead)
        kwargs.setdefault("online_chunk", online_chunk)
        # Called once per published frame, which is what lets a key press
        # interrupt a long online pass instead of waiting for it; it also
        # redraws the progress/ETA line.
        progress = _InferenceProgress(_progress_label(progress_label))
        kwargs.setdefault("online_progress_callback", progress)
        try:
            return model.inference(*args, **kwargs)
        except TypeError as exc:
            progress.close()
            print(f"[vis] model.inference rejected the online arguments ({exc!r}); "
                  "retrying without them", flush=True)
            import traceback
            traceback.print_exc()
            kwargs.pop("inference_mode", None)
            kwargs.pop("online_window", None)
            kwargs.pop("online_lookahead", None)
            kwargs.pop("online_chunk", None)
            kwargs.pop("online_progress_callback", None)
            return model.inference(*args, **kwargs)
        finally:
            # A cancellation or an error leaves the line half drawn.
            progress.close()

    if model_kind == "human_pose":
        if hasattr(model, "inference"):
            return _call_inference(model_input, gt_targets=batch_device)
        return model(model_input)
    if model_kind == "pipeline":
        if hasattr(model, "inference"):
            return _call_inference(
                model_input,
                gt_targets=batch_device,
                use_object_data=True,
                compute_fk=compute_fk_flag,
                refine_human=ot_refine,
            )
        return model(model_input, use_object_data=True, compute_fk=compute_fk_flag, refine_human=ot_refine)
    if model_arch == "dit":
        return _call_inference(
            model_input,
            gt_targets=batch_device,
            use_object_data=True,
            compute_fk=compute_fk_flag,
            refine_human=ot_refine,
        )
    return model(model_input, use_object_data=True, compute_fk=compute_fk_flag, refine_human=ot_refine)


def compute_virtual_bone_info(wrist_pos, obj_trans, obj_rot_mat):
    """Compute virtual bone length and direction."""
    v_HO_world = obj_trans - wrist_pos
    bone_length = torch.norm(v_HO_world, dim=1)
    v_HO_world_unit = v_HO_world / (bone_length.unsqueeze(-1) + 1e-8)
    obj_rot_inv = obj_rot_mat.transpose(-1, -2)
    obj_direction = torch.bmm(obj_rot_inv, v_HO_world_unit.unsqueeze(-1)).squeeze(-1)
    return bone_length, obj_direction


def _to_np_yup(points_torch, device=None):
    if points_torch is None or points_torch.dim() != 2 or points_torch.shape[0] < 2:
        return None
    points_yup = torch.matmul(points_torch, R_yup.T.to(points_torch.device))
    return points_yup.detach().cpu().numpy()


def _add_prob_traj_nodes(
    viewer,
    points_torch,
    prob_torch,
    device,
    name_prefix,
    base_color,
    r_base,
    min_alpha=PROB_MIN_ALPHA,
    max_alpha=PROB_MAX_ALPHA,
):
    if points_torch is None or prob_torch is None:
        return
    if points_torch.dim() != 2 or points_torch.shape[0] < 2:
        return

    prob = prob_torch
    if not isinstance(prob, torch.Tensor):
        prob = torch.as_tensor(prob, device=points_torch.device)
    else:
        prob = prob.to(points_torch.device)
    if prob.dim() > 1:
        prob = prob.reshape(prob.shape[0], -1)[:, 0]
    prob = prob.to(dtype=points_torch.dtype).clamp(0.0, 1.0)

    valid_len = min(points_torch.shape[0], prob.shape[0])
    if valid_len < 2:
        return

    pts = points_torch[:valid_len]
    prob = prob[:valid_len]
    seg_prob = 0.5 * (prob[:-1] + prob[1:])
    rgb = base_color[:3]
    alpha_range = float(max_alpha) - float(min_alpha)
    segments = torch.stack([pts[:-1], pts[1:]], dim=1).reshape(-1, 3)
    points_np = _to_np_yup(segments, device)
    if points_np is None:
        return

    alpha = float(min_alpha) + alpha_range * seg_prob.detach().cpu().numpy()
    colors = np.stack(
        (
            np.full_like(alpha, float(rgb[0]), dtype=np.float32),
            np.full_like(alpha, float(rgb[1]), dtype=np.float32),
            np.full_like(alpha, float(rgb[2]), dtype=np.float32),
            alpha.astype(np.float32),
        ),
        axis=1,
    )
    line_node = Lines(
        lines=points_np,
        mode="lines",
        r_base=r_base,
        color=colors,
        cast_shadow=False,
        name=name_prefix,
        gui_affine=False,
        is_selectable=False,
    )
    viewer.scene.add(line_node)


def _normalize_overlay_frames(frame_ids, total_frames):
    if frame_ids is None:
        return None
    if total_frames <= 0:
        raise SystemExit(f"Invalid sequence length: {total_frames}")

    unique_ids = []
    seen = set()
    invalid = []
    for raw_idx in frame_ids:
        idx = int(raw_idx)
        if idx in seen:
            continue
        seen.add(idx)
        if idx < 0 or idx >= total_frames:
            invalid.append(idx)
        else:
            unique_ids.append(idx)

    if invalid:
        raise SystemExit(
            f"Invalid --overlay_frames: {invalid}. "
            f"Valid frame range is [0, {total_frames - 1}]."
        )
    if not unique_ids:
        raise SystemExit("No valid frame index found in --overlay_frames.")
    return unique_ids


def _build_alpha_gradient(n, alpha_min=0.30, alpha_max=0.95):
    if n <= 0:
        return []
    if n == 1:
        return [float(alpha_max)]
    return np.linspace(alpha_min, alpha_max, n, dtype=np.float32).tolist()


def _add_overlay_meshes(viewer, verts_seq, faces_np, frame_ids, base_name, base_rgb, alpha_values, device):
    if verts_seq is None or faces_np is None:
        return
    max_frames = verts_seq.shape[0]
    rgb = (float(base_rgb[0]), float(base_rgb[1]), float(base_rgb[2]))
    for idx, frame_id in enumerate(frame_ids):
        if frame_id < 0 or frame_id >= max_frames:
            continue
        frame_verts = verts_seq[frame_id:frame_id + 1]
        frame_verts_yup = torch.matmul(frame_verts, R_yup.T.to(frame_verts.device))
        alpha = float(alpha_values[idx]) if idx < len(alpha_values) else 0.95
        mesh = Meshes(
            frame_verts_yup.detach().cpu().numpy(), faces_np,
            name=f"{base_name}-F{frame_id}",
            color=(rgb[0], rgb[1], rgb[2], alpha),
            gui_affine=False, is_selectable=False
        )
        mesh.material.ambient = 0.26
        mesh.material.diffuse = 0.41
        viewer.scene.add(mesh)


class StableMeshes(Meshes):
    """Meshes that re-upload only what actually changed.

    aitviewer rewrites positions, vertex normals *and* vertex colors on every
    buffer upload.  Here the colors are one uniform value for the whole
    sequence, so that write is pure traffic, and under WSL's D3D12 OpenGL
    translation every write costs driver memory that is reclaimed lazily.  On a
    6890-vertex body the colors are 110 KB of the 276 KB written per frame.
    """

    def _upload_buffers(self):
        if not self.is_renderable or not self._need_upload:
            return
        if (
            self.face_colors is not None
            or self.has_texture
            or self.instance_transforms is not None
        ):
            super()._upload_buffers()
            return

        frame_id = self.current_frame_id if self.vertices.shape[0] > 1 else 0
        color_key = tuple(np.asarray(self.material.color).reshape(-1).tolist())
        per_vertex_colors = not self._use_uniform_color

        if frame_id != getattr(self, "_uploaded_frame_id", None):
            self.vbo_vertices.write(self.current_vertices.astype("f4").tobytes())
            if not self.flat_shading:
                self.vbo_normals.write(
                    self.vertex_normals_at(frame_id).astype("f4").tobytes()
                )
        if per_vertex_colors or color_key != getattr(self, "_uploaded_color_key", None):
            self.vbo_colors.write(self.current_vertex_colors.astype("f4").tobytes())
            self._uploaded_color_key = color_key

        self._uploaded_frame_id = frame_id
        self._need_upload = False

    def release_normals_cache(self):
        """Drop the per-frame normal cache, which is 250 KB per cached frame."""
        self.compute_vertex_and_face_normals.cache_clear()


class FrameColoredMeshes(StableMeshes):
    """Uniform-colour mesh whose colour follows a per-frame label.

    Only ``material.color`` is swapped on frame change (the shader reads it as a
    uniform), so a per-frame label costs no per-vertex colour buffer.
    """

    def __init__(self, vertices, faces, frame_colors, **kwargs):
        frame_colors = np.asarray(frame_colors, dtype=np.float32).reshape(-1, 4)
        if frame_colors.shape[0] == 0:
            raise ValueError("frame_colors must not be empty")
        self._frame_colors = frame_colors
        super().__init__(vertices, faces, color=tuple(frame_colors[0].tolist()), **kwargs)

    def on_frame_update(self):
        index = min(int(self.current_frame_id), self._frame_colors.shape[0] - 1)
        self.material.color = tuple(self._frame_colors[index].tolist())
        super().on_frame_update()


def _obj_move_frame_colors(obj_move_seq, num_frames):
    """[num_frames, 4] colours from a bool obj_move sequence (padded with the last label)."""
    labels = obj_move_seq.detach().cpu().bool().numpy().reshape(-1)[:num_frames]
    if labels.shape[0] < num_frames:
        pad_value = labels[-1] if labels.shape[0] else False
        labels = np.concatenate((labels, np.full(num_frames - labels.shape[0], pad_value)))
    return np.where(
        labels[:, None],
        np.asarray(OBJ_MOVE_MOVING_COLOR, dtype=np.float32),
        np.asarray(OBJ_MOVE_STATIC_COLOR, dtype=np.float32),
    )


def _format_obj_move_segments(obj_move_seq):
    """Summarise obj_move as '[start-end]' runs of moving frames (inclusive, 0-based)."""
    labels = obj_move_seq.detach().cpu().bool().numpy().reshape(-1)
    runs, start = [], None
    for index, value in enumerate(labels):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append(f"{start}-{index - 1}")
            start = None
    if start is not None:
        runs.append(f"{start}-{labels.shape[0] - 1}")
    return runs


def _smpl_forward_chunked(smpl_model, pose_body, root_orient, trans, chunk=256):
    """SMPL over a long sequence in slices.

    The body model's skinning intermediates scale with frames x vertices, so a
    whole sequence in one call peaks at gigabytes (3.5 GiB at 3920 frames) and
    the caching allocator then keeps that pool for the rest of the session.
    Slicing keeps the peak flat without changing the result.
    """
    lengths = {int(pose_body.shape[0]), int(root_orient.shape[0])}
    if trans is not None:
        lengths.add(int(trans.shape[0]))
    if len(lengths) != 1:
        raise ValueError(
            f"SMPL inputs must share their frame count, got {sorted(lengths)}"
        )
    total = int(pose_body.shape[0])
    step = max(1, int(chunk))
    if total <= step:
        out = smpl_model(pose_body=pose_body, root_orient=root_orient, trans=trans)
        return out.v, out.Jtr
    verts = joints = None
    for start in range(0, total, step):
        _raise_if_cancelled()
        stop = min(start + step, total)
        out = smpl_model(
            pose_body=pose_body[start:stop],
            root_orient=root_orient[start:stop],
            trans=trans[start:stop] if trans is not None else None,
        )
        if verts is None:
            # Write into one preallocated buffer: torch.cat would hold the
            # slices and the result at the same time.
            verts = out.v.new_empty((total,) + tuple(out.v.shape[1:]))
            joints = out.Jtr.new_empty((total,) + tuple(out.Jtr.shape[1:]))
        verts[start:stop] = out.v
        joints[start:stop] = out.Jtr
        del out
    return verts, joints


def visualize_batch_data(viewer, batch, model, smpl_model, device, obj_geo_root, 
                         show_objects=True, vis_gt_only=False, show_foot_contact=False, 
                         show_hand_traj=False, use_fk=False, compare_3=False,
                         pred_offset_np=None, no_trans=False, overlay_frames=None, ot_refine=False,
                         inference_mode="online", online_window=None, online_lookahead=5,
                         online_chunk=1, obj_max_faces=0, show_obj_move=False,
                         prediction_override=None, display_length=None, render_gt=True, clear_scene=True,
                         lane_colors=None):
    """Visualize one batch sequence or one lane of a prepared comparison."""
    if clear_scene:
        _clear_visual_nodes(viewer)
    with torch.no_grad():
        return _render_batch_sequence(
            viewer, batch, model, smpl_model, device, obj_geo_root,
            show_objects=show_objects, vis_gt_only=vis_gt_only,
            show_foot_contact=show_foot_contact, show_hand_traj=show_hand_traj,
            use_fk=use_fk, compare_3=compare_3, pred_offset_np=pred_offset_np,
            no_trans=no_trans, overlay_frames=overlay_frames, ot_refine=ot_refine,
            inference_mode=inference_mode, online_window=online_window,
            online_lookahead=online_lookahead, online_chunk=online_chunk,
            obj_max_faces=obj_max_faces, show_obj_move=show_obj_move,
            prediction_override=prediction_override, display_length=display_length,
            render_gt=render_gt, lane_colors=lane_colors,
        )


def _clear_removed_node_selection(scene, nodes):
    """Detach selection references before a removed subtree releases its GL buffers."""
    removed_ids = set()
    pending = list(nodes)
    while pending:
        node = pending.pop()
        if id(node) in removed_ids:
            continue
        removed_ids.add(id(node))
        pending.extend(getattr(node, "nodes", ()))
    if id(getattr(scene, "selected_object", None)) in removed_ids:
        select = getattr(scene, "select", None)
        if callable(select):
            select(None)
        else:
            scene.selected_object = None
    # Scene.select(None) intentionally keeps the property-panel selection.
    if id(getattr(scene, "gui_selected_object", None)) in removed_ids:
        scene.gui_selected_object = None

def _clear_visual_nodes(viewer):
    try:
        nodes_to_remove = [
            node for node in viewer.scene.collect_nodes()
            if hasattr(node, "name") and node.name is not None
            and (node.name.startswith("GT-") or node.name.startswith("Pred-") or node.name.startswith("FK-")
                 or "Contact" in node.name or "Indicator" in node.name or "Traj" in node.name)
        ]
        for node in nodes_to_remove:
            try:
                viewer.scene.remove(node)
            except Exception:
                pass
        # aitviewer caches per-frame normals with @lru_cache on a *method*, so
        # `self` sits in the cache key and every mesh that ever rendered stays
        # reachable, pinning its whole vertex array.  Clearing the class-level
        # cache when a sequence goes away is what actually frees it.
        try:
            Meshes.compute_vertex_and_face_normals.cache_clear()
        except Exception:
            pass
    except Exception:
        pass

    # The previous sequence's tensors are gone with its scene nodes; give the
    # allocator's pool back so switching sequences does not pin the card.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _render_batch_sequence(viewer, batch, model, smpl_model, device, obj_geo_root,
                           show_objects=True, vis_gt_only=False, show_foot_contact=False,
                           show_hand_traj=False, use_fk=False, compare_3=False,
                           pred_offset_np=None, no_trans=False, overlay_frames=None,
                           ot_refine=False, inference_mode="online", online_window=None,
                           online_lookahead=5, online_chunk=1, obj_max_faces=0,
                           show_obj_move=False, prediction_override=None,
                           display_length=None, render_gt=True, lane_colors=None):
    colors = lane_colors or PRED_LANE_COLORS[0]
    with torch.no_grad():
        config = getattr(viewer, "config", edict({}))
        
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(device)
            else:
                batch_device[key] = value

        human_imu_batch = batch_device.get("human_imu")
        if human_imu_batch is None:
            print("Error: Batch missing 'human_imu'.")
            return
        if human_imu_batch.dim() == 3:
            human_imu_batch = human_imu_batch.unsqueeze(0)
        batch_size, T = human_imu_batch.shape[:2]
        if display_length is not None:
            T = min(T, int(display_length))
        bs = 0
        human_imu_seq = human_imu_batch[bs]
        overlay_frame_ids = _normalize_overlay_frames(overlay_frames, T)
        overlay_mode = overlay_frame_ids is not None
        overlay_alpha_values = _build_alpha_gradient(len(overlay_frame_ids)) if overlay_mode else None
        if overlay_mode:
            print(f"Overlay mesh frames: {overlay_frame_ids}")

        pose_batch = batch_device.get("pose")
        trans_batch = batch_device.get("trans")
        obj_trans_batch = batch_device.get("obj_trans")
        obj_rot_batch = batch_device.get("obj_rot")
        obj_scale_batch = batch_device.get("obj_scale")
        lhand_contact_batch = batch_device.get("lhand_contact")
        rhand_contact_batch = batch_device.get("rhand_contact")
        obj_contact_batch = batch_device.get("obj_contact")

        def _get_bool_sequence(tensor):
            if tensor is None:
                return None
            seq = tensor[bs]
            if seq.dtype != torch.bool:
                seq = seq > 0.5
            return seq

        lhand_contact_seq = _get_bool_sequence(lhand_contact_batch)
        rhand_contact_seq = _get_bool_sequence(rhand_contact_batch)
        obj_contact_seq = _get_bool_sequence(obj_contact_batch)

        has_object_value = batch_device.get("has_object")
        has_object_bool = False
        if has_object_value is not None:
            if isinstance(has_object_value, torch.Tensor):
                has_object_bool = bool(has_object_value[bs].item() if has_object_value.dim() > 0 else has_object_value.item())
            else:
                has_object_bool = bool(has_object_value)
        elif obj_trans_batch is not None:
            has_object_bool = True

        obj_name = "object"
        if "obj_name" in batch_device:
            obj_name_raw = batch_device["obj_name"]
            if isinstance(obj_name_raw, (list, tuple)):
                obj_name_candidate = obj_name_raw[bs]
            else:
                obj_name_candidate = obj_name_raw
            if isinstance(obj_name_candidate, bytes):
                obj_name_candidate = obj_name_candidate.decode("utf-8")
            if obj_name_candidate:
                obj_name = str(obj_name_candidate)

        faces_attr = getattr(smpl_model, "f", None)
        if isinstance(faces_attr, torch.Tensor):
            faces_gt_np = faces_attr.detach().cpu().numpy()
        elif faces_attr is not None:
            faces_gt_np = faces_attr
        else:
            faces_gt_np = smpl_model.faces_tensor.detach().cpu().numpy()

        verts_gt_seq = None
        Jtr_gt_seq = None
        if render_gt and pose_batch is not None and trans_batch is not None:
            try:
                gt_pose_seq = pose_batch[bs, :T]
                gt_trans_seq = trans_batch[bs, :T]
                if gt_pose_seq.dim() == 2 and gt_trans_seq.dim() == 2:
                    root_orient = gt_pose_seq[:, :3]
                    pose_body = gt_pose_seq[:, 3:]
                    if pose_body.shape[1] < 63:
                        pose_body_padded = torch.zeros(T, 63, device=device, dtype=pose_body.dtype)
                        pose_body_padded[:, :pose_body.shape[1]] = pose_body
                        pose_body = pose_body_padded
                    elif pose_body.shape[1] > 63:
                        pose_body = pose_body[:, :63]
                    verts_gt_seq, Jtr_gt_seq = _smpl_forward_chunked(
                        smpl_model, pose_body, root_orient, gt_trans_seq
                    )
            except Exception as exc:
                print(f"Failed to build GT SMPL mesh: {exc}")

        pred_dict = prediction_override
        verts_pred_seq = None
        Jtr_pred_seq = None
        pred_obj_trans_seq = None
        pred_obj_trans_fk_seq = None
        pred_obj_trans_imu_seq = None
        pred_hand_contact_prob_seq = None
        pred_obj_move_prob_seq = None
        model_input = None

        if not vis_gt_only and pred_dict is None:
            try:
                model_input = build_model_input_dict(batch_device, config, device, add_noise=False)
                compute_fk_flag = bool(use_fk or compare_3)
                model_arch = str(getattr(config, "model_arch", "rnn")).lower()
                pred_dict = _run_model_prediction(
                    model,
                    model_input,
                    batch_device,
                    compute_fk_flag,
                    model_arch,
                    ot_refine=ot_refine,
                    inference_mode=inference_mode,
                    online_window=online_window,
                    online_lookahead=online_lookahead,
                    online_chunk=online_chunk,
                )
            except Exception as exc:
                print(f"Model inference failed: {exc}")
                pred_dict = None
        _raise_if_cancelled()

        if pred_dict is not None:
            pred_length_source = pred_dict.get("p_pred")
            if not isinstance(pred_length_source, torch.Tensor):
                pred_length_source = pred_dict.get("root_trans_pred")
            if isinstance(pred_length_source, torch.Tensor) and pred_length_source.dim() >= 2:
                T = min(T, int(pred_length_source.shape[1]))
                if T <= 0:
                    print("Online strict mode has no publishable frame for this sequence.")
                    return
                human_imu_seq = human_imu_seq[:T]
                verts_gt_seq = verts_gt_seq[:T] if verts_gt_seq is not None else None
                Jtr_gt_seq = Jtr_gt_seq[:T] if Jtr_gt_seq is not None else None
                lhand_contact_seq = lhand_contact_seq[:T] if lhand_contact_seq is not None else None
                rhand_contact_seq = rhand_contact_seq[:T] if rhand_contact_seq is not None else None
                obj_contact_seq = obj_contact_seq[:T] if obj_contact_seq is not None else None
                overlay_frame_ids = _normalize_overlay_frames(overlay_frames, T)
                overlay_mode = overlay_frame_ids is not None
                overlay_alpha_values = _build_alpha_gradient(len(overlay_frame_ids)) if overlay_mode else None

        if pred_dict is not None:
            p_pred_seq = pred_dict.get("p_pred")
            if p_pred_seq is not None:
                p_pred_seq = p_pred_seq[bs].to(device)
            
            if no_trans:
                pred_root_trans_seq = trans_batch[bs, :T] if trans_batch is not None else None
            else:
                pred_root_trans_all = pred_dict.get("root_trans_pred")
                pred_root_trans_seq = pred_root_trans_all[bs].to(device) if pred_root_trans_all is not None else None
            
            if "pred_obj_trans" in pred_dict:
                pred_obj_trans_seq = pred_dict["pred_obj_trans"][bs].to(device)
            if compare_3 and "pred_obj_trans_fk" in pred_dict:
                pred_obj_trans_fk_seq = pred_dict["pred_obj_trans_fk"][bs].to(device)

            pred_hand_contact_prob_all = pred_dict.get("pred_hand_contact_prob")
            if pred_hand_contact_prob_all is not None:
                pred_hand_contact_prob_seq = pred_hand_contact_prob_all[bs].to(device)
            pred_obj_move_prob_all = pred_dict.get("pred_obj_move_prob")
            if (
                not isinstance(pred_obj_move_prob_all, torch.Tensor)
                and isinstance(pred_hand_contact_prob_all, torch.Tensor)
                and pred_hand_contact_prob_all.shape[-1] >= 3
            ):
                # VCs without a separate obj_move head (the IMUHOI baseline)
                # predict the same obj_contact label as contact channel 2;
                # v4's pred_obj_move_prob is that channel too.
                pred_obj_move_prob_all = pred_hand_contact_prob_all[..., 2]
            if isinstance(pred_obj_move_prob_all, torch.Tensor):
                pred_obj_move_prob_seq = pred_obj_move_prob_all[bs].reshape(-1)[:T].to(device)

            if compare_3 and "pred_obj_vel" in pred_dict and model_input is not None:
                try:
                    pred_obj_vel_seq = pred_dict["pred_obj_vel"][bs].to(device)
                    obj_trans_init = model_input["obj_trans_init"][bs].to(device)
                    delta = pred_obj_vel_seq * (1.0 / FRAME_RATE)
                    disp = torch.zeros_like(delta)
                    if T > 1:
                        cumulative = torch.cumsum(delta, dim=0)
                        disp[1:] = cumulative[:-1]
                    pred_obj_trans_imu_seq = obj_trans_init.unsqueeze(0) + disp
                except Exception:
                    pass
            
            human_module = _get_human_pose_module(model)

            pred_full_pose_rotmat_all = pred_dict.get("pred_full_pose_rotmat")
            pred_full_pose_rotmat_seq = (
                pred_full_pose_rotmat_all[bs].to(device)
                if isinstance(pred_full_pose_rotmat_all, torch.Tensor)
                else None
            )
            if pred_full_pose_rotmat_seq is not None and pred_root_trans_seq is not None:
                try:
                    full_glb = pred_full_pose_rotmat_seq
                    if full_glb.dim() != 4 or full_glb.shape[0] != T:
                        raise ValueError(
                            f"Invalid pred_full_pose_rotmat shape: {tuple(full_glb.shape)}, expected [{T}, J, 3, 3]"
                        )
                    if human_module is not None and hasattr(human_module, "_global2local"):
                        parents = human_module.smpl_parents.tolist()
                        local_rot = human_module._global2local(full_glb, parents)
                    else:
                        default_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
                        local_rot = _global_to_local_rotmat(full_glb, default_parents)
                    pose_axis = transforms.matrix_to_axis_angle(
                        local_rot.reshape(T * full_glb.shape[1], 3, 3)
                    ).reshape(T, full_glb.shape[1], 3)
                    root_axis = pose_axis[:, 0, :]
                    pose_body_axis = pose_axis[:, 1:22, :].reshape(T, -1)
                    verts_pred_seq, Jtr_pred_seq = _smpl_forward_chunked(
                        smpl_model, pose_body_axis, root_axis, pred_root_trans_seq
                    )
                except Exception as exc:
                    print(f"Predicted SMPL reconstruction from pred_full_pose_rotmat failed: {exc}")

            if (
                verts_pred_seq is None
                and p_pred_seq is not None
                and pred_root_trans_seq is not None
                and human_module is not None
                and hasattr(human_module, "_reduced_glb_6d_to_full_glb_mat")
                and hasattr(human_module, "_global2local")
            ):
                try:
                    reduced_pose = p_pred_seq.view(T, len(_REDUCED_POSE_NAMES), 6)
                    orientation_6d = human_imu_batch[bs, :T, :, -6:]
                    orientation_mat = transforms.rotation_6d_to_matrix(
                        orientation_6d.reshape(-1, 6)
                    ).reshape(T, human_imu_batch.shape[2], 3, 3)
                    orientation_subset = orientation_mat[:, :len(_SENSOR_ROT_INDICES), :, :]
                    full_glb = human_module._reduced_glb_6d_to_full_glb_mat(
                        reduced_pose,
                        orientation_subset.reshape(T, len(_SENSOR_ROT_INDICES), 3, 3)
                    )
                    parents = human_module.smpl_parents.tolist()
                    local_rot = human_module._global2local(full_glb, parents)
                    pose_axis = transforms.matrix_to_axis_angle(
                        local_rot.reshape(T * full_glb.shape[1], 3, 3)
                    ).reshape(T, full_glb.shape[1], 3)
                    root_axis = pose_axis[:, 0, :]
                    pose_body_axis = pose_axis[:, 1:22, :].reshape(T, -1)
                    verts_pred_seq, Jtr_pred_seq = _smpl_forward_chunked(
                        smpl_model, pose_body_axis, root_axis, pred_root_trans_seq
                    )
                except Exception as exc:
                    print(f"Predicted SMPL reconstruction failed: {exc}")

        if pred_offset_np is not None:
            pred_offset = torch.tensor(pred_offset_np, device=device, dtype=torch.float32)
        else:
            pred_offset = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)

        # BEHAVE object meshes are large (chairwood has 132857 vertices), so a
        # transformed sequence is gigabytes.  The renderer only ever sees the
        # numpy copy, so keep the whole object-vertex pipeline off the GPU.
        obj_verts_device = torch.device("cpu")
        gt_obj_verts_seq = None
        pred_obj_verts_seq_mesh = None
        pred_obj_verts_seq_fk = None
        pred_obj_verts_seq_imu = None
        obj_faces_np = None
        gt_obj_trans_seq = None
        gt_obj_rot_mat_seq = None

        if has_object_bool and obj_trans_batch is not None:
            gt_obj_trans_seq = obj_trans_batch[bs, :T]

        _raise_if_cancelled()
        if show_objects and has_object_bool and obj_trans_batch is not None and obj_rot_batch is not None:
            gt_obj_rot_6d_seq = obj_rot_batch[bs, :T]
            gt_obj_rot_mat_seq = transforms.rotation_6d_to_matrix(gt_obj_rot_6d_seq)
            gt_obj_scale_seq = obj_scale_batch[bs] if obj_scale_batch is not None else None
            if (
                isinstance(gt_obj_scale_seq, torch.Tensor)
                and gt_obj_scale_seq.dim() >= 1
                and gt_obj_scale_seq.shape[0] >= T
            ):
                gt_obj_scale_seq = gt_obj_scale_seq[:T]
            try:
                gt_obj_verts_seq, obj_faces_np = load_object_geometry(
                    obj_name, gt_obj_rot_mat_seq, gt_obj_trans_seq, gt_obj_scale_seq,
                    device=obj_verts_device, obj_geo_root=obj_geo_root,
                    max_faces=obj_max_faces or None
                )
            except Exception as exc:
                print(f"Failed to load GT object geometry: {exc}")
            
            if not vis_gt_only and pred_obj_trans_seq is not None and obj_faces_np is not None:
                try:
                    pred_obj_verts_seq_mesh, _ = load_object_geometry(
                        obj_name, gt_obj_rot_mat_seq, pred_obj_trans_seq, gt_obj_scale_seq,
                        device=obj_verts_device, obj_geo_root=obj_geo_root,
                    max_faces=obj_max_faces or None
                    )
                except Exception:
                    pass
            
            if not vis_gt_only and compare_3 and pred_obj_trans_fk_seq is not None and obj_faces_np is not None:
                try:
                    pred_obj_verts_seq_fk, _ = load_object_geometry(
                        obj_name, gt_obj_rot_mat_seq, pred_obj_trans_fk_seq, gt_obj_scale_seq,
                        device=obj_verts_device, obj_geo_root=obj_geo_root,
                    max_faces=obj_max_faces or None
                    )
                except Exception:
                    pass
            
            if not vis_gt_only and compare_3 and pred_obj_trans_imu_seq is not None and obj_faces_np is not None:
                try:
                    pred_obj_verts_seq_imu, _ = load_object_geometry(
                        obj_name, gt_obj_rot_mat_seq, pred_obj_trans_imu_seq, gt_obj_scale_seq,
                        device=obj_verts_device, obj_geo_root=obj_geo_root,
                    max_faces=obj_max_faces or None
                    )
                except Exception:
                    pass

        if verts_gt_seq is not None:
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, verts_gt_seq, faces_gt_np, overlay_frame_ids,
                    "GT-Human", GT_HUMAN_COLOR, overlay_alpha_values, device
                )
            else:
                verts_gt_yup = torch.matmul(verts_gt_seq, R_yup.T.to(device))
                gt_human_mesh = StableMeshes(
                    verts_gt_yup.detach().cpu().numpy(), faces_gt_np,
                    name="GT-Human", color=GT_HUMAN_COLOR,
                    gui_affine=False, is_selectable=False
                )
                gt_human_mesh.material.ambient = 0.26
                gt_human_mesh.material.diffuse = 0.41
                viewer.scene.add(gt_human_mesh)

        if verts_pred_seq is not None and not vis_gt_only:
            verts_pred_shifted = verts_pred_seq + pred_offset
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, verts_pred_shifted, faces_gt_np, overlay_frame_ids,
                    "Pred-Human", colors["human"], overlay_alpha_values, device
                )
            else:
                verts_pred_yup = torch.matmul(verts_pred_shifted, R_yup.T.to(device))
                pred_human_mesh = StableMeshes(
                    verts_pred_yup.detach().cpu().numpy(), faces_gt_np,
                    name="Pred-Human", color=colors["human"],
                    gui_affine=False, is_selectable=False
                )
                viewer.scene.add(pred_human_mesh)

        if show_obj_move:
            if obj_contact_seq is None:
                print("[obj_move] this sequence has no obj_contact label")
            else:
                runs = _format_obj_move_segments(obj_contact_seq)
                print(
                    f"[obj_move] moving {int(obj_contact_seq.sum())}/{int(obj_contact_seq.numel())} frames; "
                    f"moving runs: {', '.join(runs) if runs else 'none'}"
                    + (" (overlay mode: mesh colouring disabled)" if overlay_mode else ""),
                    flush=True,
                )
            if not vis_gt_only:
                if pred_obj_move_prob_seq is None:
                    print("[obj_move] prediction has no pred_obj_move_prob", flush=True)
                else:
                    pred_move = pred_obj_move_prob_seq > 0.5
                    pred_runs = _format_obj_move_segments(pred_move)
                    agreement = ""
                    if obj_contact_seq is not None:
                        count = min(int(pred_move.numel()), int(obj_contact_seq.numel()))
                        match = (pred_move[:count].cpu() == obj_contact_seq[:count].cpu()).float().mean()
                        agreement = f"; frame agreement with GT {float(match):.1%}"
                    print(
                        f"[obj_move] pred (prob>0.5) moving {int(pred_move.sum())}/{int(pred_move.numel())} frames; "
                        f"moving runs: {', '.join(pred_runs) if pred_runs else 'none'}{agreement}",
                        flush=True,
                    )

        if render_gt and gt_obj_verts_seq is not None and obj_faces_np is not None:
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, gt_obj_verts_seq, obj_faces_np, overlay_frame_ids,
                    f"GT-{obj_name}", GT_OBJECT_COLOR, overlay_alpha_values, device
                )
            else:
                gt_obj_verts_yup = torch.matmul(gt_obj_verts_seq, R_yup.T.to(gt_obj_verts_seq.device))
                if show_obj_move and obj_contact_seq is not None:
                    gt_obj_mesh = FrameColoredMeshes(
                        gt_obj_verts_yup.detach().cpu().numpy(), obj_faces_np,
                        _obj_move_frame_colors(obj_contact_seq, gt_obj_verts_yup.shape[0]),
                        name=f"GT-{obj_name}-ObjMove", gui_affine=False, is_selectable=False
                    )
                else:
                    gt_obj_mesh = StableMeshes(
                        gt_obj_verts_yup.detach().cpu().numpy(), obj_faces_np,
                        name=f"GT-{obj_name}", color=GT_OBJECT_COLOR,
                        gui_affine=False, is_selectable=False
                    )
                gt_obj_mesh.material.ambient = 0.24
                viewer.scene.add(gt_obj_mesh)

        if pred_obj_verts_seq_mesh is not None and obj_faces_np is not None and not vis_gt_only:
            pred_obj_verts_shifted = pred_obj_verts_seq_mesh + pred_offset.to(pred_obj_verts_seq_mesh.device)
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, pred_obj_verts_shifted, obj_faces_np, overlay_frame_ids,
                    f"Pred-{obj_name}", colors["object"], overlay_alpha_values, device
                )
            else:
                pred_obj_verts_yup = torch.matmul(pred_obj_verts_shifted, R_yup.T.to(pred_obj_verts_shifted.device))
                if show_obj_move and pred_obj_move_prob_seq is not None:
                    # Same palette as the GT object: orange = moving (prob > 0.5), purple = static.
                    pred_obj_mesh = FrameColoredMeshes(
                        pred_obj_verts_yup.detach().cpu().numpy(), obj_faces_np,
                        _obj_move_frame_colors(
                            pred_obj_move_prob_seq > 0.5, pred_obj_verts_yup.shape[0]
                        ),
                        name=f"Pred-{obj_name}-ObjMove", gui_affine=False, is_selectable=False
                    )
                else:
                    pred_obj_mesh = StableMeshes(
                        pred_obj_verts_yup.detach().cpu().numpy(), obj_faces_np,
                        name=f"Pred-{obj_name}", color=colors["object"],
                        gui_affine=False, is_selectable=False
                    )
                viewer.scene.add(pred_obj_mesh)

        if compare_3 and pred_obj_verts_seq_imu is not None and obj_faces_np is not None and not vis_gt_only:
            obj_verts_imu_shifted = pred_obj_verts_seq_imu + pred_offset.to(pred_obj_verts_seq_imu.device)
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, obj_verts_imu_shifted, obj_faces_np, overlay_frame_ids,
                    f"Pred-IMU-{obj_name}", (0.2, 0.2, 0.9), overlay_alpha_values, device
                )
            else:
                obj_verts_imu_yup = torch.matmul(obj_verts_imu_shifted, R_yup.T.to(obj_verts_imu_shifted.device))
                imu_mesh = StableMeshes(
                    obj_verts_imu_yup.detach().cpu().numpy(), obj_faces_np,
                    name=f"Pred-IMU-{obj_name}", color=(0.2, 0.2, 0.9, 0.8),
                    gui_affine=False, is_selectable=False
                )
                viewer.scene.add(imu_mesh)

        if compare_3 and pred_obj_verts_seq_fk is not None and obj_faces_np is not None and not vis_gt_only:
            obj_verts_fk_shifted = pred_obj_verts_seq_fk + pred_offset.to(pred_obj_verts_seq_fk.device)
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, obj_verts_fk_shifted, obj_faces_np, overlay_frame_ids,
                    f"FK-{obj_name}", (1.0, 1.0, 0.0), overlay_alpha_values, device
                )
            else:
                obj_verts_fk_yup = torch.matmul(obj_verts_fk_shifted, R_yup.T.to(obj_verts_fk_shifted.device))
                fk_mesh = StableMeshes(
                    obj_verts_fk_yup.detach().cpu().numpy(), obj_faces_np,
                    name=f"FK-{obj_name}", color=(1.0, 1.0, 0.0, 0.8),
                    gui_affine=False, is_selectable=False
                )
                viewer.scene.add(fk_mesh)

        if show_hand_traj:
            lhand_idx, rhand_idx = 20, 21

            if render_gt and Jtr_gt_seq is not None:
                if lhand_contact_seq is not None:
                    _add_prob_traj_nodes(
                        viewer,
                        Jtr_gt_seq[:, lhand_idx],
                        lhand_contact_seq.float(),
                        device,
                        "GT-LHandTraj",
                        HAND_TRAJ_COLORS["gt_l"],
                        HAND_TRAJ_RADIUS,
                    )

                if rhand_contact_seq is not None:
                    _add_prob_traj_nodes(
                        viewer,
                        Jtr_gt_seq[:, rhand_idx],
                        rhand_contact_seq.float(),
                        device,
                        "GT-RHandTraj",
                        HAND_TRAJ_COLORS["gt_r"],
                        HAND_TRAJ_RADIUS,
                    )

            if not vis_gt_only and Jtr_pred_seq is not None:
                if pred_hand_contact_prob_seq is not None and pred_hand_contact_prob_seq.shape[-1] >= 2:
                    _add_prob_traj_nodes(
                        viewer,
                        Jtr_pred_seq[:, lhand_idx] + pred_offset,
                        pred_hand_contact_prob_seq[:, 0],
                        device,
                        "Pred-LHandTraj",
                        colors["lhand"],
                        HAND_TRAJ_RADIUS,
                    )
                    _add_prob_traj_nodes(
                        viewer,
                        Jtr_pred_seq[:, rhand_idx] + pred_offset,
                        pred_hand_contact_prob_seq[:, 1],
                        device,
                        "Pred-RHandTraj",
                        colors["rhand"],
                        HAND_TRAJ_RADIUS,
                    )

        viewer.virtual_bone_info = {'has_data': False}


def visualize_comparison_data(viewer, batch, bundles, smpl_model, device, obj_geo_root):
    """Infer N model lanes sequentially, then draw one GT and synchronized predictions."""
    from model.rnn.online import resolve_online_lookahead, resolve_online_window

    batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    human_imu = batch_device.get("human_imu")
    if not isinstance(human_imu, torch.Tensor):
        raise ValueError("Comparison batch has no human_imu")
    total_frames = int(human_imu.shape[1])
    predictions = []
    for lane, bundle in enumerate(bundles, 1):
        _raise_if_cancelled()
        _move_comparison_model(bundle, device)
        try:
            config = bundle.config
            model_input = build_model_input_dict(batch_device, config, device, add_noise=False)
            window = resolve_online_window(config, viewer.online_window)
            lookahead = resolve_online_lookahead(config, viewer.online_lookahead)
            chunk = viewer.online_chunk if viewer.online_chunk is not None else min(59, window - 1)
            with torch.inference_mode():
                pred = _run_model_prediction(
                    bundle.model, model_input, batch_device, bool(viewer.use_fk),
                    "rnn", ot_refine=bundle.ot_refine, inference_mode=viewer.inference_mode,
                    online_window=window, online_lookahead=lookahead, online_chunk=chunk,
                    progress_label=f"{lane}/{len(bundles)} {bundle.label}",
                )
            _raise_if_cancelled()
            length = _comparison_prediction_length(pred, total_frames)
            for key in ("pred_obj_trans", "root_trans_pred"):
                value = pred.get(key)
                if not isinstance(value, torch.Tensor) or value.shape[1] < length:
                    raise ValueError(f"{bundle.label}: missing/short {key} prediction")
            predictions.append((bundle, {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                         for k, v in pred.items()}, length))
        finally:
            _move_comparison_model(bundle, torch.device("cpu"))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    common_length = min(length for _, _, length in predictions)
    _normalize_overlay_frames(viewer.overlay_frames, common_length)
    from model.rnn.online import slice_time_dict
    predictions = [
        (bundle, slice_time_dict(pred, 0, common_length, int(human_imu.shape[0]), length), length)
        for bundle, pred, length in predictions
    ]
    # Object vertex arrays remain resident for every visible lane. Divide the
    # default face budget among lanes rather than multiplying its VRAM cost.
    face_budget = viewer.obj_max_faces
    if face_budget:
        face_budget = min(face_budget, max(2000, 20000 // len(bundles)))
    for index, (bundle, pred, _) in enumerate(predictions, 1):
        _raise_if_cancelled()
        # The worker only records scene changes; the GL thread commits all lanes
        # together after every model has succeeded. A cancelled job commits none.
        _move_comparison_model(bundle, device)
        try:
            lane = _LaneViewer(viewer, bundle.label, bundle.config)
            visualize_batch_data(
                lane, batch, bundle.model, smpl_model, device, obj_geo_root,
                show_objects=viewer.show_objects, show_foot_contact=viewer.show_foot_contact,
                show_hand_traj=viewer.show_hand_traj, show_obj_move=viewer.show_obj_move,
                pred_offset_np=_lane_offset(getattr(viewer, "pred_offset", None), index),
                no_trans=viewer.no_trans, overlay_frames=viewer.overlay_frames,
                ot_refine=bundle.ot_refine, obj_max_faces=face_budget,
                prediction_override=pred, display_length=common_length,
                render_gt=(index == 1), clear_scene=False, lane_colors=_lane_colors(index),
            )
        finally:
            _move_comparison_model(bundle, torch.device("cpu"))
            if device.type == "cuda":
                torch.cuda.empty_cache()


class InteractiveViewer(Viewer):
    def __init__(self, data_list, model, smpl_model, config, device, obj_geo_root, 
                 show_objects=True, vis_gt_only=False, show_foot_contact=False,
                 show_hand_traj=False, use_fk=False, compare_3=False,
                 pred_offset=None, no_trans=False, overlay_frames=None, ot_refine=False,
                 inference_mode="online", online_window=None, online_lookahead=5,
                 online_chunk=1, auto_advance=0.0, obj_max_faces=0, show_obj_move=False,
                 comparison_models=None, **kwargs):
        super().__init__(**kwargs)
        # Debug/instrumentation: step through sequences on a timer, which is how
        # sequence-switch memory behaviour gets measured without a keyboard.
        self.auto_advance = float(auto_advance or 0.0)
        self._last_advance_at = None
        self.data_list = data_list
        self.current_index = 0
        self.model = model
        self.comparison_models = comparison_models or []
        self._comparison_nodes = []
        self.smpl_model = smpl_model
        self.config = config
        self.device = device
        self.show_objects = show_objects
        self.vis_gt_only = vis_gt_only
        self.show_obj_move = show_obj_move
        self.show_hand_traj = show_hand_traj
        self.show_foot_contact = show_foot_contact
        self.obj_geo_root = obj_geo_root
        self.use_fk = use_fk
        self.compare_3 = compare_3
        self.pred_offset = pred_offset
        self.no_trans = no_trans
        self.overlay_frames = overlay_frames
        self.ot_refine = ot_refine
        self.inference_mode = inference_mode
        self.online_window = online_window
        self.online_lookahead = online_lookahead
        self.online_chunk = online_chunk
        self.obj_max_faces = int(obj_max_faces or 0)
        self.virtual_bone_info = {'has_data': False}

        # Sequences are computed on a background thread so that a key press can
        # interrupt one that is still running; the render loop schedules jobs
        # and applies a finished job's scene edits on the GL thread.
        self._job = None
        self._worker = None
        self._pending_index = None
        self._generation = 0
        self.request_sequence(self.current_index)

        light_front = self.scene.lights[1]
        light_front.strength = 1.32 
        light_front.position = (15, 10, 15)
        floor = self.scene.floor
        floor.material.diffuse = 0.1
        floor.tiling = False
        floor.material.color = (132/255, 150/255, 183/255, 1.0)
        floor.material.ambient = 0.30
        self.scene.background_color = (0.5, 0.5, 0.5, 1.0)

    def visualize_current_sequence(self):
        """Kept for callers of the old API: schedules the current sequence."""
        self.request_sequence(self.current_index)

    # ------------------------------------------------------------------
    # Background sequence jobs
    # ------------------------------------------------------------------
    def _sequence_label(self, index):
        entry = self.data_list[index]
        name = ""
        if isinstance(entry, dict):
            name = entry.get("seq_file_name") or os.path.basename(entry.get("seq_file_path", ""))
        return name or f"index {index}"

    def _target_index(self):
        """Where the next q/e press steps from: the newest request, not the screen."""
        if self._pending_index is not None:
            return self._pending_index
        if self._job is not None and self._job.status == "running":
            return self._job.index
        return self.current_index

    def request_sequence(self, index):
        """Ask for a sequence; interrupts the one being computed, if any."""
        if not self.data_list:
            print("Error: Data list is empty.")
            return
        index = max(0, min(len(self.data_list) - 1, int(index)))
        job = self._job
        if job is not None and job.status == "running":
            if job.index == index and not job.cancel.is_set():
                return
            if not job.cancel.is_set():
                _PROGRESS_LINE.break_line()
                print(
                    f"Interrupted sequence {self._sequence_label(job.index)} "
                    f"after {_walltime.perf_counter() - job.started_at:.1f}s; "
                    f"switching to {self._sequence_label(index)}",
                    flush=True,
                )
                job.cancel_requested_at = _walltime.perf_counter()
                job.cancel.set()
            else:
                _PROGRESS_LINE.break_line()
                print(f"Switching to {self._sequence_label(index)} instead", flush=True)
        self._pending_index = index

    def _start_job(self, index):
        self._generation += 1
        job = _SequenceJob(index, self._generation)
        self._job = job
        self._worker = threading.Thread(
            target=self._run_sequence_job, args=(job,), name=f"vis-seq-{index}", daemon=True
        )
        self._worker.start()

    def _run_sequence_job(self, job):
        _JOB_CONTEXT.job = job
        try:
            entry = self.data_list[job.index]
            batch = entry["batch"] if isinstance(entry, dict) and "batch" in entry else entry
            mode_str = " (GT only)" if self.vis_gt_only else " (GT+Pred)"
            label = self._sequence_label(job.index)
            job.label = os.path.splitext(label)[0]
            print(f"Visualizing sequence: {label}{mode_str}", flush=True)
            proxy = _WorkerViewerProxy(self, job.ops)
            if self.comparison_models:
                for node in self._comparison_nodes:
                    proxy.scene.remove(node)
                visualize_comparison_data(
                    proxy, batch, self.comparison_models, self.smpl_model,
                    self.device, self.obj_geo_root,
                )
            else:
                visualize_batch_data(
                    proxy, batch, self.model, self.smpl_model, self.device,
                    self.obj_geo_root, self.show_objects, self.vis_gt_only,
                    self.show_foot_contact, self.show_hand_traj,
                    self.use_fk, self.compare_3, self.pred_offset, self.no_trans, self.overlay_frames,
                    self.ot_refine, self.inference_mode, self.online_window, self.online_lookahead,
                    self.online_chunk, self.obj_max_faces, self.show_obj_move,
                )
            _raise_if_cancelled()
            job.title = f"Sequence: {label}{mode_str} (q/e: +-1, Ctrl+q/e: +-10, Alt+q/e: +-50)"
            job.status = "done"
        except SequenceCancelled:
            job.status = "cancelled"
            job.ops.clear()
            if job.cancel_requested_at is not None:
                print(
                    f"Stopped {self._sequence_label(job.index)} "
                    f"{(_walltime.perf_counter() - job.cancel_requested_at) * 1000:.0f} ms after the request",
                    flush=True,
                )
        except Exception as exc:
            job.status = "failed"
            job.ops.clear()
            print(f"Error visualizing sequence {job.index}: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            job.finished_at = _walltime.perf_counter()
            _JOB_CONTEXT.job = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _apply_finished_job(self, job):
        """Swap the new sequence in; runs on the GL thread."""
        _clear_removed_node_selection(
            self.scene,
            [node for op, nodes, _ in job.ops if op == "remove" for node in nodes],
        )
        for op, nodes, kwargs in job.ops:
            if op == "remove":
                for node in nodes:
                    try:
                        self.scene.remove(node)
                    except Exception:
                        pass
            else:
                self.scene.add(*nodes, **kwargs)
        # The previous sequence's meshes are gone now; drop the normal cache
        # that would otherwise keep them (and their vertex arrays) alive.
        try:
            Meshes.compute_vertex_and_face_normals.cache_clear()
        except Exception:
            pass
        if self.comparison_models:
            self._comparison_nodes = [node for op, nodes, _ in job.ops if op == "add" for node in nodes]
        job.ops.clear()
        self.current_index = job.index
        self.scene.current_frame_id = 0
        if self.comparison_models:
            self._frame_comparison_camera()
        if job.title:
            self.title = job.title
        print(
            f"Ready: {self._sequence_label(job.index)} "
            f"({job.finished_at - job.started_at:.1f}s)",
            flush=True,
        )

    def _frame_comparison_camera(self):
        """Show the GT and every prediction lane when a sequence first opens."""
        bounds = []
        for node in self._comparison_nodes:
            if getattr(node, "name", "").startswith(("GT-", "Pred-")) and hasattr(node, "current_bounds"):
                bound = np.asarray(node.current_bounds, dtype=np.float64)
                if bound.shape == (3, 2) and np.isfinite(bound).all():
                    bounds.append(bound)
        if not bounds:
            return
        lower = np.min([b[:, 0] for b in bounds], axis=0)
        upper = np.max([b[:, 1] for b in bounds], axis=0)
        center = (lower + upper) * 0.5
        # Keep the leftmost GT lane clear of aitviewer's scene-tree sidebar.
        center[0] -= 0.10 * float(upper[0] - lower[0])
        diagonal = float(np.linalg.norm(upper - lower))
        camera = self.scene.camera
        forward = np.asarray(camera.forward, dtype=np.float64)
        camera.target = center
        camera.position = center - forward * max(5.0, diagonal * 1.4)

    def _pump_sequence_jobs(self):
        job = self._job
        if job is not None and job.status != "running" and not (
            self._worker is not None and self._worker.is_alive()
        ):
            self._job = None
            self._worker = None
            superseded = self._pending_index is not None and self._pending_index != job.index
            if job.status == "done" and not superseded:
                self._apply_finished_job(job)
        if self._job is None and self._pending_index is not None:
            index = self._pending_index
            self._pending_index = None
            self._start_job(index)

    def render(self, time, frame_time, **kwargs):
        if self.auto_advance > 0.0:
            if self._last_advance_at is None:
                self._last_advance_at = time
            elif time - self._last_advance_at >= self.auto_advance:
                self._last_advance_at = time
                new_index = min(len(self.data_list) - 1, self._target_index() + 1)
                if new_index != self._target_index():
                    print(f"[auto_advance] requesting sequence index {new_index}", flush=True)
                    self.request_sequence(new_index)
        self._pump_sequence_jobs()
        self._yield_to_running_job()
        return super().render(time, frame_time, **kwargs)

    # The render loop and a background job share the GIL.  Left alone the loop
    # spins far faster than the screen needs and starves the job: in testing the
    # job ran over 30x slower and noticed a cancellation 40 s late.  While a job
    # is running, pace the loop so most of each frame is spent off the GIL; at
    # 15 fps the job runs as fast as it does without a window at all.
    BUSY_FRAME_PERIOD = 1.0 / 15.0

    def _yield_to_running_job(self):
        job = self._job
        now = _walltime.perf_counter()
        last = getattr(self, "_last_render_wall", None)
        self._last_render_wall = now
        if job is None or job.status != "running" or last is None:
            return
        remaining = self.BUSY_FRAME_PERIOD - (now - last)
        if remaining > 0:
            _walltime.sleep(remaining)
            self._last_render_wall = _walltime.perf_counter()

    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        
        io = imgui.get_io()
        if self.render_gui and (io.want_capture_keyboard or io.want_text_input):
            return
        
        is_press = action == self.wnd.keys.ACTION_PRESS
        
        if is_press:
            ctrl_pressed = modifiers.ctrl
            alt_pressed = modifiers.alt
            
            if key in (self.wnd.keys.Q, self.wnd.keys.E):
                step = 50 if alt_pressed else (10 if ctrl_pressed else 1)
                if key == self.wnd.keys.Q:
                    step = -step
                # Step from the newest request, so repeated presses while a
                # sequence is still computing keep moving instead of piling up.
                base = self._target_index()
                new_index = max(0, min(len(self.data_list) - 1, base + step))
                if new_index != base:
                    self.request_sequence(new_index)


OBJECT_TRANS_ARCH_CHOICES = (
    "current",
    "v2",
    "v3",
    "legacy",
    "legacy_mesh",
    "mergeHeads_obs",
    "imuhoi",
    "imuhoi_4heads",
)


def _normalize_object_trans_arch(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "mergeheads_obs":
        return "mergeHeads_obs"
    return normalized


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Interactive IMUHOI Visualization Tool')
    parser.add_argument('--model', action='append', metavar='VARIANT_DIR',
                        help='Repeat for each model lane; resolve config and final HP/VC/OT checkpoints from its experiment manifest')
    parser.add_argument('--config', type=str, default='configs/IMUHOI_train_rnn.yaml', help='Path to config file')
    parser.add_argument('--model_arch', type=str, choices=['rnn', 'dit', 'mamba'], default=None, help='Override model architecture in config')
    parser.add_argument('--hp_ckpt', type=str, default=None, help='Override HumanPose checkpoint')
    parser.add_argument('--vc_ckpt', type=str, default=None, help='Override VelocityContact checkpoint')
    parser.add_argument('--ot_ckpt', type=str, default=None, help='Override ObjectTrans checkpoint')
    parser.add_argument('--interaction_ckpt', type=str, default=None, help='Override Interaction/Object checkpoint')
    parser.add_argument(
        '--object_trans_arch',
        '--object-trans-arch',
        type=_normalize_object_trans_arch,
        choices=OBJECT_TRANS_ARCH_CHOICES,
        default=None,
        help=(
            'ObjectTrans architecture (case-insensitive); imuhoi selects the '
            'original EgoIMU three-branch model and imuhoi_4heads selects the '
            'static-hold four-branch ablation; v2 uses fixed VC fusion weights; '
            'v3 uses uncertainty-aware hierarchical FK fusion.'
        ),
    )
    parser.add_argument(
        '--velocity_contact_arch',
        '--velocity-contact-arch',
        type=normalize_velocity_contact_arch,
        choices=VELOCITY_CONTACT_ARCH_CHOICES,
        default=None,
        help=(
            'VelocityContact architecture (case-insensitive); imuhoi selects '
            'the EgoIMU VC model; v2 selects the causal velocity-first '
            'direct-contact cascade.'
        ),
    )
    parser.add_argument('--smpl_model_path', type=str, default=None, help='Path to SMPL model file')
    parser.add_argument('--dataset', type=str, default='omomo', help='Dataset key under config.datasets')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help=(
            'Override the dataset input path with a single .pt file or a directory '
            'of .pt files; takes precedence over --dataset.'
        ),
    )
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--no_objects', action='store_true', help='Disable object mesh rendering')
    parser.add_argument('--vis_gt_only', action='store_true', help='Render GT only')
    parser.add_argument('--show_foot_contact', action='store_true', help='Show foot contact')
    parser.add_argument(
        '--show_obj_move', action='store_true',
        help=(
            'Colour the GT object by its obj_move label and the predicted object by '
            'pred_obj_move_prob > 0.5 (orange = moving, purple = static); also prints moving runs.'
        ),
    )
    parser.add_argument('--show_hand_traj', action='store_true', help='Show hand contact trajectory')
    parser.add_argument('--use_fk', action='store_true', help='Enable FK branch')
    parser.add_argument('--compare_3', action='store_true', help='Compare three object branches')
    parser.add_argument('--limit_sequences', type=int, default=500, help='Limit number of loaded sequences')
    parser.add_argument('--pred_offset', type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        help='Prediction translation offset; with repeated --model, lane i is drawn '
                             'at i * offset, so neighbouring lanes (GT first) are one offset apart')
    parser.add_argument('--overlay_frames', type=int, nargs='+', default=None, help='Overlay selected 0-based frames in one scene')
    parser.add_argument('--no_trans', action='store_true', help='Enable noTrans mode')
    parser.add_argument('--obj_max_faces', '--obj-max-faces', type=int, default=20000,
                        help='Decimate object meshes to at most N faces for display; 0 keeps the '
                             'captured resolution. BEHAVE chairwood is 226563 faces and a 1000-frame '
                             'sequence of it costs 10.8 GB of RAM at full resolution against 1.3 GB '
                             'at 20000 faces, with 0.84 mm mean surface error')
    parser.add_argument('--auto_advance', '--auto-advance', type=float, default=0.0,
                        help='Debug: step to the next sequence every N seconds (0 = off)')
    parser.add_argument('--no_shadows', '--no-shadows', action='store_true',
                        help='Disable the scene lights shadow maps; each is an 8192x8192 '
                             'depth texture worth about 257 MiB of GPU memory')
    parser.add_argument('--inference_mode', '--inference-mode', choices=['online', 'offline'], default='online', help='RNN inference mode')
    parser.add_argument('--online_window', '--online-window', type=int, default=None, help='RNN online sliding window size')
    parser.add_argument(
        '--online_chunk', '--online-chunk', type=int, default=None,
        help='Target frames whose HumanPose/VelocityContact windows run as one batch '
             '(1 = one window per frame; capped at online_window - 1)',
    )
    parser.add_argument(
        '--online_lookahead',
        '--online-lookahead',
        type=int,
        default=10,
        help='Future frames appended after the legacy RNN online update frame (strict tail)',
    )
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        '--simulate_imu_noise',
        '--simulate-imu-noise',
        action='store_true',
        help='Enable Dataset IMU noise simulation once per full sequence.',
    )
    noise_group.add_argument(
        '--eval_imu_noise',
        '--eval-imu-noise',
        action='store_true',
        help='Apply one fixed split-Gaussian IMU noise draw per full sequence before visualization.',
    )
    parser.add_argument(
        '--simulate_imu_noise_seed',
        '--simulate-imu-noise-seed',
        type=int,
        default=42,
        help='Base seed for deterministic Dataset IMU noise; sequence path is mixed in.',
    )
    parser.add_argument(
        '--eval_imu_noise_seed',
        '--eval-imu-noise-seed',
        type=int,
        default=42,
        help='Base seed for sequence-level noisy eval; sequence path is mixed in for stable per-sequence noise.',
    )
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument(
        '--enable_ot_refine',
        dest='ot_refine',
        action='store_true',
        default=None,
        help='Allow ObjectTrans refine to override human pose/root trans',
    )
    refine_group.add_argument(
        '--disable_ot_refine',
        dest='ot_refine',
        action='store_false',
        help='Disable ObjectTrans human refine and keep HumanPose output',
    )
    return parser


def main():
    args = _make_parser().parse_args()
    comparison_specs = resolve_vis_models(args.model, PROJECT_ROOT) if args.model else []
    if comparison_specs and (
        args.config != 'configs/IMUHOI_train_rnn.yaml'
        or args.hp_ckpt or args.vc_ckpt or args.ot_ckpt or args.interaction_ckpt
        or args.model_arch or args.object_trans_arch or args.velocity_contact_arch
        or args.ot_refine is not None or args.compare_3 or args.vis_gt_only
    ):
        raise SystemExit('[Vis] --model cannot be combined with --config, single-model checkpoint/architecture/refine overrides, --compare_3 or --vis_gt_only')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Mode: {'noTrans' if args.no_trans else 'Normal'}")
    print(
        f"Inference mode: {args.inference_mode} | online_window: "
        f"{args.online_window or 'config/default'} | online_lookahead: {args.online_lookahead}"
    )
    print(
        "Eval IMU noise: "
        f"{'enabled' if args.eval_imu_noise else 'disabled'}"
        + (f" | base_seed={args.eval_imu_noise_seed}" if args.eval_imu_noise else "")
    )
    print(
        "simulate_imu_noise: "
        f"{'enabled' if args.simulate_imu_noise else 'disabled'}"
        + (f" | base_seed={args.simulate_imu_noise_seed}" if args.simulate_imu_noise else "")
        + " (sequence-level)"
    )

    print(f"Loading config from: {comparison_specs[0].config_path if comparison_specs else args.config}")
    config = load_config(str(comparison_specs[0].config_path) if comparison_specs else args.config)
    try:
        # ``--data_dir`` is an explicit input override.  It intentionally does
        # not require a valid dataset key, so a standalone PT can be visualized
        # with a model-only config as well.  When the key exists, retain its
        # object mesh root as a fallback for relative object names.
        if args.data_dir is not None:
            override_path = Path(args.data_dir).expanduser()
            if not override_path.is_absolute():
                override_path = Path.cwd() / override_path
            test_data_input = str(override_path.resolve())
            try:
                dataset_configs = get_dataset_configs(config)
            except (TypeError, ValueError):
                dataset_configs = {}
            dataset_config = dataset_configs.get(args.dataset, {})
        else:
            dataset_configs = get_dataset_configs(config)
            if args.dataset not in dataset_configs:
                valid = ', '.join(sorted(dataset_configs))
                raise ValueError(f"Unknown dataset '{args.dataset}'. Valid datasets: {valid}")
            dataset_config = dataset_configs[args.dataset]
            test_data_input = str(resolve_dataset_path(
                get_full_sequence_test_path(dataset_config),
                PROJECT_ROOT,
            ))
        obj_geo_setting = dataset_config.get('object_geo_root')
        obj_geo_root = (
            resolve_dataset_path(obj_geo_setting, PROJECT_ROOT)
            if obj_geo_setting
            else None
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"[Vis] Invalid dataset configuration: {exc}") from exc

    if args.data_dir is not None:
        print(f"Data source override (--data_dir): {test_data_input}")
    else:
        print(f"Dataset: {args.dataset}")
        print(f"Test data path: {test_data_input}")
    if obj_geo_root:
        print(f"Object mesh root: {obj_geo_root}")
    elif not args.no_objects:
        print("[Vis] No object mesh root configured for this dataset; object meshes may be unavailable.")

    if args.model_arch is not None:
        config.model_arch = args.model_arch
    if args.object_trans_arch is not None:
        config.object_trans_arch = args.object_trans_arch
    config.object_trans_arch = _normalize_object_trans_arch(
        getattr(config, 'object_trans_arch', 'current')
    )
    if args.velocity_contact_arch is not None:
        config.velocity_contact_arch = args.velocity_contact_arch
    config.velocity_contact_arch = normalize_velocity_contact_arch(
        getattr(config, 'velocity_contact_arch', 'current')
    )
    print(f"ObjectTrans architecture: {config.object_trans_arch}")
    print(f"VelocityContact architecture: {config.velocity_contact_arch}")
    ot_refine = (
        False
        if is_imuhoi_object_trans_arch(config.object_trans_arch)
        else (
            bool(args.ot_refine)
            if args.ot_refine is not None
            else bool(getattr(config, 'enable_ot_refine', False))
        )
    )
    config.enable_ot_refine = ot_refine
    print(f"OT human refine: {'enabled' if ot_refine else 'disabled'}")
    module_paths = None
    if hasattr(config, "pretrained_modules") and config.pretrained_modules:
        module_paths = dict(config.pretrained_modules)
    if args.hp_ckpt:
        module_paths = dict(module_paths) if isinstance(module_paths, dict) else {}
        module_paths["human_pose"] = args.hp_ckpt
    if args.vc_ckpt:
        module_paths = dict(module_paths) if isinstance(module_paths, dict) else {}
        module_paths["velocity_contact"] = args.vc_ckpt
    if args.ot_ckpt:
        module_paths = dict(module_paths) if isinstance(module_paths, dict) else {}
        module_paths["object_trans"] = args.ot_ckpt
    if args.interaction_ckpt:
        module_paths = dict(module_paths) if isinstance(module_paths, dict) else {}
        module_paths["interaction"] = args.interaction_ckpt
    model_arch = str(getattr(config, "model_arch", "rnn")).lower()
    if model_arch in {"dit", "mamba"} and isinstance(module_paths, dict):
        allowed = {"human_pose", "interaction", "velocity_contact", "object_trans"}
        module_paths = {key: value for key, value in module_paths.items() if key in allowed and value}
        module_paths = module_paths or None
    
    if args.smpl_model_path:
        config.body_model_path = args.smpl_model_path
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    config.device = str(device)

    smpl_model_path = config.get('body_model_path', 'datasets/smpl_models/smplh/neutral/model.npz')
    smpl_model = load_smpl_model(smpl_model_path, device)

    comparison_models = _load_comparison_models(comparison_specs, args.dataset, args.no_trans) if comparison_specs else []
    for lane, bundle in enumerate(comparison_models, 1):
        offset = ", ".join(f"{value:g}" for value in _lane_offset(args.pred_offset, lane))
        print(f"[Vis] lane {lane}: {bundle.label} -> {_lane_colors(lane)['name']}, offset ({offset})")
    model = comparison_models[0].model if comparison_models else load_model(
        config, device, no_trans=args.no_trans, module_paths=module_paths
    )
    if not comparison_models and _infer_model_kind(model) == "human_pose" and not args.no_objects:
        print("[Vis] 当前加载的是 human-only 模型；如需渲染预测物体，请提供 pretrained_modules.interaction 或 --interaction_ckpt。")

    if comparison_models:
        for bundle in comparison_models[1:]:
            other = get_dataset_configs(bundle.config)[args.dataset]
            other_path = resolve_dataset_path(get_full_sequence_test_path(other), PROJECT_ROOT)
            if args.data_dir is None and Path(other_path).resolve() != Path(test_data_input).resolve():
                raise ValueError(f"{bundle.label}: test split {other_path} differs from shared data {test_data_input}")
            for key in ("body_model_type", "num_human_imus", "imu_dim"):
                first = dataset_config.get(key) if key == "body_model_type" else getattr(config, key, None)
                second = other.get(key) if key == "body_model_type" else getattr(bundle.config, key, None)
                if first != second:
                    raise ValueError(f"{bundle.label}: incompatible shared data convention {key}: {second} != {first}")

    if not test_data_input or not os.path.exists(test_data_input):
        print(f"Error: Dataset test path not found: {test_data_input}")
        return

    test_window_size = config.test.get('window', config.train.get('window', 60))
    dataset_debug = config.get('debug', False)

    if os.path.isdir(test_data_input):
        print(f"Loading test dataset from directory: {test_data_input}")
        print("Dataset mode: directory")
        test_dataset = IMUDataset(
            data_dir=test_data_input,
            window_size=test_window_size,
            debug=dataset_debug,
            full_sequence=True,
            simulate_imu_noise=bool(args.simulate_imu_noise),
            simulate_imu_noise_seed=args.simulate_imu_noise_seed,
            resolve_bimanual_contact_conflicts=config.get("resolve_bimanual_contact_conflicts", False),
        )
    elif os.path.isfile(test_data_input) and test_data_input.lower().endswith(".pt"):
        target_pt = os.path.normcase(os.path.normpath(os.path.abspath(test_data_input)))
        parent_dir = os.path.dirname(target_pt)
        print(f"Loading test dataset from single file: {target_pt}")
        print("Dataset mode: single-pt")
        test_dataset = IMUDataset(
            data_dir=parent_dir,
            window_size=test_window_size,
            debug=dataset_debug,
            full_sequence=True,
            sequence_paths=[target_pt],
            simulate_imu_noise=bool(args.simulate_imu_noise),
            simulate_imu_noise_seed=args.simulate_imu_noise_seed,
            resolve_bimanual_contact_conflicts=config.get("resolve_bimanual_contact_conflicts", False),
        )
        if len(test_dataset.sequence_info) != 1:
            try:
                pt_data = torch.load(target_pt, map_location="cpu")
            except Exception as exc:
                raise SystemExit(f"Failed to load pt file: {target_pt}. Error: {exc}")
            required_key = "pose"
            if not isinstance(pt_data, dict) or required_key not in pt_data:
                raise SystemExit(
                    f"Unsupported pt format for dataset test_path: {target_pt}. "
                    f"Missing required key '{required_key}'."
                )
            raise SystemExit(
                f"Failed to build single-sequence dataset from file: {target_pt}. "
                f"Valid sequence count after filtering: {len(test_dataset.sequence_info)}"
            )
    else:
        raise SystemExit(
            f"Visualization data path must be a directory or a .pt file: {test_data_input}"
        )

    def _natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    try:
        test_dataset.sequence_info.sort(
            key=lambda info: _natural_key(os.path.basename(info.get('file_path', '')))
        )
    except Exception:
        pass

    if len(test_dataset) == 0:
        print("Error: Test dataset is empty.")
        return

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=False, drop_last=False
    )

    print(f"Loading data into memory (limit={args.limit_sequences})...")
    data_list = []
    for i, batch in enumerate(test_loader):
        if args.limit_sequences is not None and i >= args.limit_sequences:
            break
        try:
            seq_info_i = test_dataset.sequence_info[i]
            file_path_i = seq_info_i.get('file_path', '')
            file_name_i = os.path.basename(file_path_i) if file_path_i else ''
        except Exception:
            file_path_i, file_name_i = '', ''
        if args.eval_imu_noise:
            batch = apply_eval_imu_noise_to_sequence(
                batch,
                config,
                seed=args.eval_imu_noise_seed,
                sequence_key=os.path.realpath(os.path.abspath(file_path_i)) if file_path_i else i,
            )
        data_list.append({
            'batch': batch,
            'seq_file_path': file_path_i,
            'seq_file_name': file_name_i,
        })
        if i % 50 == 0 and i > 0:
            print(f"  Loaded {i+1} sequences...")
    print(f"Finished loading {len(data_list)} sequences.")

    if not data_list:
        print("Error: No data loaded.")
        return

    print("Initializing Interactive Viewer...")
    pred_offset_np = np.array(args.pred_offset, dtype=np.float32)
    if args.overlay_frames is not None:
        print(f"Overlay frame mode enabled: {args.overlay_frames}")
    
    viewer_instance = InteractiveViewer(
        data_list=data_list,
        model=model,
        smpl_model=smpl_model,
        config=config,
        device=device,
        obj_geo_root=obj_geo_root,
        show_objects=(not args.no_objects),
        vis_gt_only=args.vis_gt_only,
        show_foot_contact=args.show_foot_contact,
        show_obj_move=args.show_obj_move,
        show_hand_traj=args.show_hand_traj,
        use_fk=args.use_fk,
        compare_3=args.compare_3,
        pred_offset=pred_offset_np,
        no_trans=args.no_trans,
        overlay_frames=args.overlay_frames,
        ot_refine=ot_refine,
        inference_mode=args.inference_mode,
        online_window=args.online_window if comparison_models else (args.online_window or 60),
        online_lookahead=args.online_lookahead if comparison_models else (10 if args.online_lookahead is None else args.online_lookahead),
        online_chunk=args.online_chunk if comparison_models else (args.online_chunk or 59),
        comparison_models=comparison_models,
        auto_advance=args.auto_advance,
        obj_max_faces=args.obj_max_faces,
        window_size=(1920, 1080)
    )

    if args.no_shadows:
        for light in getattr(viewer_instance.scene, "lights", []):
            light.shadow_enabled = False
        print(f"Shadow maps disabled for {len(viewer_instance.scene.lights)} light(s) "
              "(about 257 MiB of GPU memory each)")

    print("Viewer Initialized. Controls:")
    print("  q/e: Previous/Next 1 sequence")
    print("  Ctrl+q/e: Previous/Next 10 sequences")
    print("  Alt+q/e: Previous/Next 50 sequences")
    viewer_instance.run()


if __name__ == "__main__":
    main()


