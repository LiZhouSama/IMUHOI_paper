"""Load a trained comparison checkpoint, run full-sequence inference, and evaluate.

Example:
    python -m Comparisons.eval_comparison --method transpose \
        --checkpoint outputs/TRANSPOSE/OMOMO/best.pt --dataset omomo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Comparisons.common.adapters import tensor_to_device
from Comparisons.common.inference import infer_comparison
from Comparisons.common.metrics import (
    METRIC_UNITS,
    MetricAccumulator,
    add_compatibility_aliases,
    comparison_metric_stats,
)
from Comparisons.dip18 import DIP18HOIModel, DIP18ReferenceHOIModel
from Comparisons.globalpose import GlobalPoseHOIModel
from Comparisons.tip import TIPHOIModel, TIPReferenceHOIModel
from Comparisons.transpose import TransPoseHOIModel
from Comparisons.common.dataset import IMUDataset
from utils.human_pose import load_body_model
from utils.dataset_config import get_full_sequence_test_path
from utils.utils import setup_seed


METHODS = ("dip18", "tip", "transpose", "globalpose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-sequence checkpoint inference/evaluation for comparisons"
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cfg", default="configs/IMUHOI_train_rnn.yaml")
    parser.add_argument("--dataset", default=None, help="Dataset key from cfg.datasets")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument(
        "--data_dir",
        action="append",
        default=None,
        help="Explicit processed sequence directory; repeat for multiple directories",
    )
    parser.add_argument("--body_model_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--n_sbps", type=int, default=None)
    parser.add_argument(
        "--tip_n_heads",
        type=int,
        default=None,
        help="Override TIP attention heads for checkpoints without architecture metadata",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--dip_past_frames", type=int, default=20)
    parser.add_argument("--dip_future_frames", type=int, default=5)
    parser.add_argument("--dip_window_batch_size", type=int, default=128)
    parser.add_argument(
        "--dip_variant",
        "--dip-variant",
        choices=("legacy", "official"),
        default=None,
        help="Override DIP architecture detection (normally inferred from the checkpoint).",
    )
    parser.add_argument("--tip_window", type=int, default=40)
    parser.add_argument(
        "--tip_variant",
        "--tip-variant",
        choices=("legacy", "official"),
        default=None,
        help="Override TIP architecture/protocol detection.",
    )
    parser.add_argument("--transpose_past_frames", type=int, default=20)
    parser.add_argument("--transpose_future_frames", type=int, default=5)
    parser.add_argument("--transpose_window_batch_size", type=int, default=256)
    parser.add_argument(
        "--transpose_variant",
        "--transpose-variant",
        choices=("legacy", "official"),
        default=None,
        help="Override TransPose architecture/protocol detection.",
    )
    parser.add_argument(
        "--globalpose_variant",
        "--globalpose-variant",
        choices=("legacy", "official"),
        default=None,
        help="Override GlobalPose object protocol detection.",
    )
    parser.add_argument(
        "--globalpose_init_mode",
        "--globalpose-init-mode",
        choices=("reference", "deploy"),
        default="reference",
        help=(
            "GlobalPose RNN initialization. 'reference' matches the released "
            "online test (first-frame pose, zero VR labels); 'deploy' uses "
            "mean-shape T-pose and zero motion."
        ),
    )
    parser.add_argument(
        "--protocol_version",
        type=int,
        choices=(1, 2),
        default=None,
        help="Override checkpoint protocol: 1=legacy, 2=corrected",
    )
    parser.add_argument(
        "--object_output_mode",
        choices=("absolute", "relative_displacement", "velocity"),
        default=None,
        help="Override checkpoint object-head semantics",
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--save_predictions", default=None)
    parser.add_argument("--seed", type=int, default=42)
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--simulate_imu_noise",
        "--simulate-imu-noise",
        action="store_true",
        help=(
            "Enable Dataset.simulate_imu_noise for full-sequence evaluation. "
            "Default: disabled."
        ),
    )
    noise_group.add_argument(
        "--eval_imu_noise",
        "--eval-imu-noise",
        action="store_true",
        help=(
            "Reserved for the main model's sequence-level noise protocol; "
            "comparison evaluators only support --simulate-imu-noise."
        ),
    )
    parser.add_argument(
        "--simulate_imu_noise_seed",
        "--simulate-imu-noise-seed",
        type=int,
        default=42,
        help="Base seed for deterministic Dataset IMU noise.",
    )
    return parser.parse_args()


def _load_cfg(path: str):
    with open(path, "r") as handle:
        return edict(yaml.safe_load(handle))


def _resolve_data_dirs(args: argparse.Namespace, cfg) -> list[str]:
    if args.data_dir:
        raw_paths = args.data_dir
    else:
        dataset_names = (
            [args.dataset]
            if args.dataset
            else (getattr(cfg, "train_datasets", None) or list(cfg.datasets.keys()))
        )
        raw_paths = []
        for name in dataset_names:
            if name not in cfg.datasets:
                raise KeyError(f"Dataset {name!r} is not present in {args.cfg}")
            if args.split == "test":
                relative = get_full_sequence_test_path(cfg.datasets[name])
            else:
                relative = getattr(cfg.datasets[name], "train_path", None)
            if relative:
                raw_paths.append(relative)

    paths = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Evaluation data path does not exist: {path}")
        paths.append(str(path.resolve()))
    if not paths:
        raise RuntimeError("No evaluation data paths were resolved")
    return paths


def _build_model(
    method: str,
    n_sbps: int,
    tip_n_heads: int,
    tip_object_state_feedback: bool,
    globalpose_object_bidirectional: bool,
    globalpose_object_with_init: bool,
    transpose_object_bidirectional: bool,
    transpose_object_condition_on_full: bool,
    human_condition_on_object: bool,
    dip_variant: str = "legacy",
    tip_variant: str = "legacy",
    transpose_variant: str = "legacy",
    globalpose_obj_imu_dim: int = 12,
) -> torch.nn.Module:
    if method == "dip18":
        if dip_variant == "official":
            return DIP18ReferenceHOIModel()
        if dip_variant != "legacy":
            raise ValueError(f"Unknown DIP variant: {dip_variant!r}")
        return DIP18HOIModel(
            human_condition_on_object=human_condition_on_object
        )
    if method == "tip":
        if tip_variant == "official":
            # Official eval_tip_format.py disables the history dropout at
            # inference (the training script uses p=0.8).  Dropout has no
            # checkpoint parameters, so this must be selected at construction.
            return TIPReferenceHOIModel(n_heads=tip_n_heads, past_state_dropout=0.0)
        return TIPHOIModel(
            n_sbps=n_sbps,
            n_heads=tip_n_heads,
            object_state_feedback=tip_object_state_feedback,
        )
    if method == "transpose":
        return TransPoseHOIModel(
            object_bidirectional=transpose_object_bidirectional,
            object_condition_on_full=transpose_object_condition_on_full,
            human_condition_on_object=human_condition_on_object,
            transpose_reference=transpose_variant == "official",
        )
    if method == "globalpose":
        return GlobalPoseHOIModel(
            obj_imu_dim=globalpose_obj_imu_dim,
            object_bidirectional=globalpose_object_bidirectional,
            object_with_init=globalpose_object_with_init,
            human_condition_on_object=human_condition_on_object,
        )
    raise ValueError(method)


def _checkpoint_transpose_variant(args: argparse.Namespace, checkpoint) -> str:
    if getattr(args, "transpose_variant", None) is not None:
        return str(args.transpose_variant)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    stored = metadata.get("transpose_variant")
    if stored is None:
        stored = (metadata.get("model_kwargs") or {}).get("transpose_variant")
    if stored in ("legacy", "official"):
        return str(stored)
    try:
        state = _checkpoint_state(checkpoint)
    except ValueError:
        state = {}
    # Released TransPose-with-object checkpoints use the nested ``base`` and
    # ``object_net`` names and the exact 104-tensor signature.
    if (
        "base.pose_s1.linear1.weight" in state
        and "object_net.net.linear1.weight" in state
        and tuple(state["base.pose_s1.linear1.weight"].shape) == (256, 72)
        and tuple(state["object_net.net.linear1.weight"].shape) == (128, 84)
    ):
        return "official"
    return "legacy"


def _checkpoint_tip_variant(args: argparse.Namespace, checkpoint) -> str:
    if getattr(args, "tip_variant", None) is not None:
        return str(args.tip_variant)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    stored = metadata.get("tip_variant")
    if stored in ("legacy", "official"):
        return str(stored)
    try:
        state = _checkpoint_state(checkpoint)
    except ValueError:
        state = {}
    # Public transformer-inertial-poser object checkpoints have exactly these
    # dimensions: Linear(63+114,256) and Linear(512,114).
    if (
        state.get("in_linear.weight") is not None
        and state.get("linear.weight") is not None
        and tuple(state["in_linear.weight"].shape) == (256, 177)
        and tuple(state["linear.weight"].shape) == (114, 512)
    ):
        return "official"
    return "legacy"


def _checkpoint_state(checkpoint) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return {
                    key[7:] if key.startswith("module.") else key: value
                    for key, value in state.items()
                }
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return {
                key[7:] if key.startswith("module.") else key: value
                for key, value in checkpoint.items()
            }
    raise ValueError("Checkpoint does not contain a model state dict")


def _map_transpose_reference_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map RefCodes/TransPose's nested names to the comparison wrapper."""
    mapped = {}
    for key, value in state.items():
        if key.startswith("base."):
            key = key[len("base.") :]
        elif key.startswith("object_net.net."):
            key = "obj_head." + key[len("object_net.net.") :]
        mapped[key] = value
    return mapped


def _map_globalpose_reference_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map the released GlobalPose ModuleDict names to the local wrapper."""
    mapped = {}
    for key, value in state.items():
        if key.startswith("iknet.net1."):
            key = "iknet1." + key[len("iknet.net1.") :]
        elif key.startswith("iknet.net2."):
            key = "iknet2." + key[len("iknet.net2.") :]
        if key in mapped:
            raise ValueError(f"Duplicate GlobalPose checkpoint key after mapping: {key}")
        mapped[key] = value
    return mapped


def _checkpoint_globalpose_variant(args: argparse.Namespace, checkpoint) -> str:
    """Infer whether a GlobalPose checkpoint uses the released 9-D object path."""
    explicit = getattr(args, "globalpose_variant", None)
    if explicit is not None:
        return str(explicit)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    stored = metadata.get("globalpose_variant")
    if stored is None:
        stored = (metadata.get("model_kwargs") or {}).get("globalpose_variant")
    if stored in ("legacy", "official"):
        return str(stored)
    try:
        state = _checkpoint_state(checkpoint)
    except ValueError:
        state = {}
    object_input = state.get("obj_head.rnn.weight_ih_l0")
    if object_input is not None:
        # LSTM gate matrix has four times hidden rows; its second dimension is
        # the concatenated VR state plus object IMU width.
        if object_input.ndim == 2 and int(object_input.shape[1]) == 252:
            return "official"
        if object_input.ndim == 2 and int(object_input.shape[1]) == 255:
            return "legacy"
    # The released public file contains human weights only and therefore has
    # no obj_head keys; its PL input signature is unambiguous.
    if "plnet.rnn.weight_ih_l0" in state and int(state["plnet.rnn.weight_ih_l0"].shape[1]) == 84:
        return "official"
    return "legacy"


def _checkpoint_dip_variant(args: argparse.Namespace, checkpoint) -> str:
    """Resolve the DIP architecture without guessing from a loss value.

    The released OMOMO file is a raw state_dict and has no metadata, so the
    distinctive ``glb.linear1`` input width is the fallback discriminator.
    """
    if getattr(args, "dip_variant", None) is not None:
        return str(args.dip_variant)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    stored = metadata.get("dip_variant")
    if stored in ("legacy", "official"):
        return str(stored)
    try:
        state = _checkpoint_state(checkpoint)
    except ValueError:
        state = {}
    glb = state.get("glb.linear1.weight")
    if glb is not None and glb.ndim == 2 and tuple(glb.shape) == (36, 72):
        return "official"
    return "legacy"


def _checkpoint_protocol(
    args: argparse.Namespace, checkpoint
) -> tuple[int, str, int, int, bool, bool, bool, bool, bool, bool]:
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    model_kwargs = metadata.get("model_kwargs") or {}
    try:
        state = _checkpoint_state(checkpoint)
    except ValueError:
        state = {}

    stored_n_sbps = metadata.get("n_sbps")
    if stored_n_sbps is None:
        stored_n_sbps = model_kwargs.get("n_sbps", 5)
    if (
        args.n_sbps is None
        and metadata.get("n_sbps") is None
        and model_kwargs.get("n_sbps") is None
        and "linear.weight" in state
    ):
        output_size = int(state["linear.weight"].shape[0])
        # TIP emits [18*6 pose, root velocity, 4*n_sbps, object velocity].
        if output_size >= 114 and (output_size - 114) % 4 == 0:
            stored_n_sbps = (output_size - 114) // 4
    n_sbps = int(
        args.n_sbps
        if args.n_sbps is not None
        else stored_n_sbps
    )

    if args.protocol_version is not None:
        protocol_version = int(args.protocol_version)
    elif "comparison_protocol_version" in metadata:
        protocol_version = int(metadata["comparison_protocol_version"])
    else:
        # Raw state_dict files have no semantic metadata. The corrected training
        # architectures have method-specific input widths that distinguish them
        # from every legacy architecture produced by this repository.
        protocol_version = 1
        if getattr(args, "method", None) == "dip18" and "glb.linear1.weight" in state:
            # Raw public OMOMO DIP checkpoint: glb is Linear(72, 36).
            if tuple(state["glb.linear1.weight"].shape) == (36, 72):
                protocol_version = 2
        elif "temporal.input_layer.0.weight" in state:
            if int(state["temporal.input_layer.0.weight"].shape[1]) == 60:
                protocol_version = 2
        elif "pose_s1.linear1.weight" in state:
            if int(state["pose_s1.linear1.weight"].shape[1]) == 72:
                protocol_version = 2
        elif "base.pose_s1.linear1.weight" in state:
            if int(state["base.pose_s1.linear1.weight"].shape[1]) == 72:
                # Raw RefCodes/TransPose-with-object checkpoint.  Although
                # the released container has no metadata, its 72-channel
                # reference input and velocity object branch are the
                # corrected official protocol.
                protocol_version = 2
        elif "plnet.rnn.weight_ih_l0" in state:
            if int(state["plnet.rnn.weight_ih_l0"].shape[1]) == 84:
                protocol_version = 2
        elif (
            getattr(args, "method", None) == "tip"
            and tuple(state.get("in_linear.weight", torch.empty(0)).shape)
            == (256, 177)
            and tuple(state.get("linear.weight", torch.empty(0)).shape)
            == (114, 512)
        ):
            # Raw public TIP object checkpoint: 63 IMU + 114 state channels.
            protocol_version = 2
        elif "in_linear.weight" in state:
            tip_base_input = 90 + 12 + 111 + 4 * n_sbps
            if int(state["in_linear.weight"].shape[1]) > tip_base_input:
                protocol_version = 2
        if any(key.startswith("obj_head.init_net.") for key in state):
            protocol_version = 2

    object_output_mode = args.object_output_mode or metadata.get("object_output_mode")
    if object_output_mode is None:
        object_output_mode = "velocity" if protocol_version >= 2 else "absolute"
    stored_tip_n_heads = metadata.get("tip_n_heads")
    if stored_tip_n_heads is None:
        stored_tip_n_heads = model_kwargs.get("n_heads", 16)
    tip_n_heads = int(
        args.tip_n_heads
        if args.tip_n_heads is not None
        else stored_tip_n_heads
    )
    tip_object_state_feedback = bool(
        model_kwargs.get("object_state_feedback", protocol_version >= 2)
    )
    default_object_bidirectional = (
        not bool(metadata["object_head_causal"])
        if "object_head_causal" in metadata
        else protocol_version < 2
    )
    object_bidirectional = bool(
        model_kwargs.get("object_bidirectional", default_object_bidirectional)
    )
    globalpose_object_bidirectional = object_bidirectional
    globalpose_object_with_init = bool(
        model_kwargs.get("object_with_init", protocol_version >= 2)
    )
    transpose_object_bidirectional = object_bidirectional
    transpose_object_condition_on_full = bool(
        model_kwargs.get("object_condition_on_full", protocol_version < 2)
    )
    human_condition_on_object = bool(
        model_kwargs.get("human_condition_on_object", protocol_version < 2)
    )

    # Architecture metadata was not present in the original comparison
    # checkpoints. Weight shapes are authoritative when they are available.
    has_reverse_object_rnn = any(
        (key.startswith("obj_head.rnn.") or key.startswith("object_net.net.rnn."))
        and key.endswith("_reverse")
        for key in state
    )
    if "in_linear.weight" in state and "linear.weight" in state:
        if (
            tuple(state["in_linear.weight"].shape) == (256, 177)
            and tuple(state["linear.weight"].shape) == (114, 512)
        ):
            tip_object_state_feedback = False
        else:
            tip_base_input = 90 + 12 + 111 + 4 * n_sbps
            tip_object_state_feedback = (
                state["in_linear.weight"].shape[1] > tip_base_input
            )
    transpose_reference_checkpoint = (
        "base.pose_s1.linear1.weight" in state
        and "object_net.net.linear1.weight" in state
    )
    transpose_prefix = (
        "base.pose_s1.linear1.weight"
        if "base.pose_s1.linear1.weight" in state
        else "pose_s1.linear1.weight"
    )
    if transpose_prefix in state:
        human_condition_on_object = (
            state[transpose_prefix].shape[1] > 72
        )
        transpose_object_bidirectional = has_reverse_object_rnn
        if "obj_head.linear1.weight" in state:
            transpose_object_condition_on_full = (
                state["obj_head.linear1.weight"].shape[1] > 84
            )
    if transpose_reference_checkpoint:
        transpose_object_condition_on_full = False
        transpose_object_bidirectional = has_reverse_object_rnn
        human_condition_on_object = False
    elif "plnet.rnn.weight_ih_l0" in state:
        human_condition_on_object = (
            state["plnet.rnn.weight_ih_l0"].shape[1] > 84
        )
        globalpose_object_bidirectional = has_reverse_object_rnn
        if any(key.startswith("obj_head.init_net.") for key in state):
            globalpose_object_with_init = True
        elif any(key.startswith("obj_head.linear1.") for key in state):
            globalpose_object_with_init = False
    elif "temporal.input_layer.0.weight" in state:
        human_condition_on_object = (
            state["temporal.input_layer.0.weight"].shape[1] > 60
        )
    return (
        protocol_version,
        object_output_mode,
        n_sbps,
        tip_n_heads,
        tip_object_state_feedback,
        globalpose_object_bidirectional,
        globalpose_object_with_init,
        transpose_object_bidirectional,
        transpose_object_condition_on_full,
        human_condition_on_object,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_sequences is not None and args.max_sequences <= 0:
        raise ValueError("--max_sequences must be positive")
    if args.n_sbps is not None and args.n_sbps < 0:
        raise ValueError("--n_sbps must be non-negative")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.dip_past_frames < 0 or args.dip_future_frames < 0:
        raise ValueError("DIP past/future frame counts must be non-negative")
    if args.dip_window_batch_size <= 0:
        raise ValueError("--dip_window_batch_size must be positive")
    if args.tip_window <= 0:
        raise ValueError("--tip_window must be positive")
    if args.transpose_past_frames < 0 or args.transpose_future_frames < 0:
        raise ValueError("TransPose past/future frame counts must be non-negative")
    if args.transpose_window_batch_size <= 0:
        raise ValueError("--transpose_window_batch_size must be positive")
    if args.eval_imu_noise:
        raise ValueError(
            "Comparisons.eval_comparison supports --simulate-imu-noise; "
            "use eval_batch_online.py for --eval-imu-noise."
        )


def _sequence_name(batch: dict, index: int = 0) -> str:
    value = batch.get("seq_name", f"sequence_{index:05d}")
    if isinstance(value, (list, tuple)):
        return str(value[index])
    return str(value)


def _serialize_predictions(
    destination: Path,
    batch: dict,
    prediction: dict,
    method: str,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    name = _sequence_name(batch)
    payload = {
        "method": method,
        "seq_name": name,
        "local_rot": prediction["local_rot"][0].detach().cpu(),
        "root_trans": prediction["root_trans"][0].detach().cpu(),
        "joints": prediction["joints"][0].detach().cpu(),
        "target_joints": prediction["target_joints"][0].detach().cpu(),
        "target_root_trans": batch["trans"][0].detach().cpu(),
        "obj_trans": prediction["obj_trans"][0].detach().cpu(),
        "has_object": bool(batch["has_object"][0]),
    }
    safe_name = name.replace("/", "_").replace("\\", "_")
    torch.save(payload, destination / f"{safe_name}.pt")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    cfg = _load_cfg(args.cfg)
    setup_seed(args.seed)
    device_name = args.device or getattr(cfg, "device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    (
        protocol_version,
        object_output_mode,
        n_sbps,
        tip_n_heads,
        tip_object_state_feedback,
        globalpose_object_bidirectional,
        globalpose_object_with_init,
        transpose_object_bidirectional,
        transpose_object_condition_on_full,
        human_condition_on_object,
    ) = _checkpoint_protocol(args, checkpoint)
    dip_variant = (
        _checkpoint_dip_variant(args, checkpoint)
        if args.method == "dip18"
        else "legacy"
    )
    tip_variant = (
        _checkpoint_tip_variant(args, checkpoint)
        if args.method == "tip"
        else "legacy"
    )
    transpose_variant = (
        _checkpoint_transpose_variant(args, checkpoint)
        if args.method == "transpose"
        else "legacy"
    )
    globalpose_variant = (
        _checkpoint_globalpose_variant(args, checkpoint)
        if args.method == "globalpose"
        else "legacy"
    )
    globalpose_obj_imu_dim = 9 if globalpose_variant == "official" else 12
    if args.method == "transpose" and transpose_variant == "official":
        # RefCodes/TransPose's object branch is trained on world velocity,
        # while the raw checkpoint has no metadata describing this contract.
        object_output_mode = "velocity"
        # New unified checkpoints retain corrected-v2 metadata; raw public
        # checkpoints use v1 as a container marker but still run the explicit
        # reference adapter selected above.
    if isinstance(checkpoint, dict) and checkpoint.get("method"):
        checkpoint_method = str(checkpoint["method"]).lower()
        if checkpoint_method != args.method:
            raise ValueError(
                f"Checkpoint method is {checkpoint_method!r}, not {args.method!r}"
            )
    model = _build_model(
        args.method,
        n_sbps=n_sbps,
        tip_n_heads=tip_n_heads,
        tip_object_state_feedback=tip_object_state_feedback,
        globalpose_object_bidirectional=globalpose_object_bidirectional,
        globalpose_object_with_init=globalpose_object_with_init,
        transpose_object_bidirectional=transpose_object_bidirectional,
        transpose_object_condition_on_full=transpose_object_condition_on_full,
        human_condition_on_object=human_condition_on_object,
        dip_variant=dip_variant,
        tip_variant=tip_variant,
        transpose_variant=transpose_variant,
        globalpose_obj_imu_dim=globalpose_obj_imu_dim,
    ).to(device)
    checkpoint_state = _checkpoint_state(checkpoint)
    if args.method == "transpose" and transpose_variant == "official":
        checkpoint_state = _map_transpose_reference_state(checkpoint_state)
    if args.method == "globalpose":
        checkpoint_state = _map_globalpose_reference_state(checkpoint_state)
        # Public GlobalPose weights contain only the 68 human tensors.  The
        # object extension has no released checkpoint; allow exactly those
        # object keys to remain freshly initialized, while rejecting every
        # other mismatch.
        model_keys = set(model.state_dict())
        human_keys = {key for key in model_keys if not key.startswith("obj_head.")}
        missing_human = sorted(human_keys - set(checkpoint_state))
        unexpected = sorted(set(checkpoint_state) - model_keys)
        shape_errors = sorted(
            key
            for key in human_keys.intersection(checkpoint_state)
            if tuple(checkpoint_state[key].shape) != tuple(model.state_dict()[key].shape)
        )
        if missing_human or unexpected or shape_errors:
            raise RuntimeError(
                "GlobalPose human checkpoint is incompatible "
                f"(missing={missing_human[:3]}, unexpected={unexpected[:3]}, "
                f"shape_errors={shape_errors[:3]})"
            )
        model.load_state_dict(checkpoint_state, strict=False)
        object_parameters_initialized = bool(
            set(key for key in checkpoint_state if key.startswith("obj_head."))
        )
    else:
        model.load_state_dict(checkpoint_state, strict=True)
        object_parameters_initialized = True
    model.eval()

    body_model_path = args.body_model_path or getattr(cfg, "body_model_path", None)
    if not body_model_path:
        raise ValueError("body_model_path is missing from arguments and config")
    body_model_path = Path(body_model_path)
    if not body_model_path.is_absolute():
        body_model_path = PROJECT_ROOT / body_model_path
    body_model = load_body_model(str(body_model_path)).to(device)

    data_dirs = _resolve_data_dirs(args, cfg)
    sequence_paths = None
    if args.max_sequences is not None:
        sequence_paths = []
        for data_dir in data_dirs:
            sequence_paths.extend(sorted(str(path) for path in Path(data_dir).glob("*.pt")))
        sequence_paths = sequence_paths[: args.max_sequences]
    dataset = IMUDataset(
        data_dir=data_dirs,
        sequence_paths=sequence_paths,
        full_sequence=True,
        window_size=1,
        random_window=False,
        debug=False,
        # full_sequence=True applies one deterministic noise draw per sequence,
        # matching eval_batch_online.py's Dataset noise protocol.
        simulate_imu_noise=bool(args.simulate_imu_noise),
        simulate_imu_noise_seed=args.simulate_imu_noise_seed,
        obj_points_sample_count=int(getattr(cfg, "mesh_downsample_points", 256)),
        body_model_path=(
            str(body_model_path)
            if args.method == "globalpose" and globalpose_variant == "official"
            else None
        ),
        resolve_bimanual_contact_conflicts=bool(
            getattr(cfg, "resolve_bimanual_contact_conflicts", False)
        ),
        imu_protocol=(
            "dip18_reference"
            if args.method == "dip18" and dip_variant == "official"
            else "tip_reference"
            if args.method == "tip" and tip_variant == "official"
            else "transpose_reference"
            if args.method == "transpose" and transpose_variant == "official"
            else "globalpose_reference"
            if args.method == "globalpose" and globalpose_variant == "official"
            else "imuhoi"
        ),
    )
    if len(dataset) == 0:
        raise RuntimeError("The evaluation dataset is empty")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    protocol_name = "corrected_v2" if protocol_version >= 2 else "legacy_v1"
    print(
        f"method={args.method} checkpoint={checkpoint_path} protocol={protocol_name} "
        f"object_output={object_output_mode} sequences={len(dataset)} device={device}"
    )
    if args.method == "dip18":
        print(f"dip_variant={dip_variant}")
    if args.method == "tip":
        print(f"tip_variant={tip_variant}")
    if args.method == "globalpose":
        print(
            f"globalpose_variant={globalpose_variant} "
            f"object_imu_dim={globalpose_obj_imu_dim} "
            f"object_parameters_initialized={object_parameters_initialized}"
        )
    if protocol_version < 2:
        print(
            "warning: checkpoint has no corrected protocol metadata; using legacy "
            "input/object semantics for compatibility"
        )
    if args.method == "globalpose":
        print(
            "note: GlobalPose evaluation uses neural VR/stationary fusion; the "
            "Windows-only physics optimizer is not applied"
        )

    aggregate = MetricAccumulator()
    per_sequence = []
    translation_modes = set()
    initialization_modes = set()
    object_velocity_frames = set()
    prediction_dir = Path(args.save_predictions) if args.save_predictions else None
    with torch.inference_mode():
        for raw_batch in tqdm(loader, desc=f"eval {args.method}"):
            batch = tensor_to_device(raw_batch, device)
            prediction = infer_comparison(
                args.method,
                model,
                batch,
                body_model,
                n_sbps=n_sbps,
                fps=args.fps,
                protocol_version=protocol_version,
                object_output_mode=object_output_mode,
                dip_past_frames=args.dip_past_frames,
                dip_future_frames=args.dip_future_frames,
                dip_window_batch_size=args.dip_window_batch_size,
                tip_window=args.tip_window,
                transpose_past_frames=args.transpose_past_frames,
                transpose_future_frames=args.transpose_future_frames,
                transpose_window_batch_size=args.transpose_window_batch_size,
                dip_variant=dip_variant,
                tip_variant=tip_variant,
                transpose_variant=transpose_variant,
                globalpose_obj_imu_dim=globalpose_obj_imu_dim,
                globalpose_init_mode=args.globalpose_init_mode,
            )
            prediction["object_metrics_available"] = bool(
                object_parameters_initialized
                if args.method == "globalpose"
                else True
            )
            stats = comparison_metric_stats(prediction, batch)
            sequence_raw_metrics = stats.means()
            aggregate.add_sequence_means(stats)
            sequence_metrics = add_compatibility_aliases(sequence_raw_metrics)
            per_sequence.append(
                {
                    "seq_name": _sequence_name(raw_batch),
                    "num_frames": int(batch["human_imu"].shape[1]),
                    "has_object": bool(batch["has_object"][0]),
                    "metrics": sequence_metrics,
                }
            )
            translation_modes.add(str(prediction["translation_mode"]))
            initialization_modes.add(str(prediction["initialization_mode"]))
            object_velocity_frames.add(str(prediction["object_velocity_frame"]))
            if prediction_dir is not None:
                _serialize_predictions(
                    prediction_dir, raw_batch, prediction, args.method
                )

    metrics = add_compatibility_aliases(aggregate.means())
    print("\nComparison evaluation")
    display_order = [
        "human_ang_err_deg",
        "human_geodesic_err_deg",
        "human_pos_err_cm",
        "human_jitter_mm_frame2",
        "root_trans_err_cm",
        "obj_err_cm",
        "hoi_err_cm",
        "obj_trans_jitter_mm_frame2",
    ]
    for name in display_order:
        if metrics[name] is not None:
            print(
                f"{name:32s} {metrics[name]:12.6f} {METRIC_UNITS[name]} "
                f"(n={aggregate.counts.get(name, 0)})"
            )
        else:
            print(f"{name:32s} {'unavailable':>12s} {METRIC_UNITS[name]} (n=0)")

    report = {
        "method": args.method,
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "protocol": protocol_name,
        "protocol_version": protocol_version,
        "object_output_mode": object_output_mode,
        "dip_variant": dip_variant if args.method == "dip18" else None,
        "tip_variant": tip_variant if args.method == "tip" else None,
        "transpose_variant": transpose_variant if args.method == "transpose" else None,
        "globalpose_variant": globalpose_variant if args.method == "globalpose" else None,
        "globalpose_init_mode": (
            args.globalpose_init_mode if args.method == "globalpose" else None
        ),
        "globalpose_object_imu_dim": (
            globalpose_obj_imu_dim if args.method == "globalpose" else None
        ),
        "globalpose_object_parameters_initialized": (
            object_parameters_initialized if args.method == "globalpose" else None
        ),
        "object_metrics_available": bool(
            object_parameters_initialized if args.method == "globalpose" else True
        ),
        "simulate_imu_noise": bool(args.simulate_imu_noise),
        "simulate_imu_noise_seed": (
            args.simulate_imu_noise_seed if args.simulate_imu_noise else None
        ),
        "object_velocity_frame": sorted(object_velocity_frames),
        "tip_n_heads": tip_n_heads if args.method == "tip" else None,
        "tip_object_state_feedback": (
            model.object_state_feedback if args.method == "tip" else None
        ),
        "online_window": (
            {
                "past_frames": args.dip_past_frames,
                "future_frames": args.dip_future_frames,
            }
            if args.method == "dip18"
            else {"history_frames": args.tip_window}
            if args.method == "tip"
            else {
                "past_frames": args.transpose_past_frames,
                "future_frames": args.transpose_future_frames,
            }
            if args.method == "transpose"
            else {"history": "recurrent_state"}
        ),
        "translation_mode": sorted(translation_modes),
        "initialization_mode": sorted(initialization_modes),
        "globalpose_physics_applied": False if args.method == "globalpose" else None,
        "globalpose_object_head_causal": (
            not model.object_bidirectional if args.method == "globalpose" else None
        ),
        "globalpose_object_with_init": (
            model.object_with_init if args.method == "globalpose" else None
        ),
        "transpose_object_head_causal": (
            not model.object_bidirectional if args.method == "transpose" else None
        ),
        "transpose_object_condition_on_full": (
            model.object_condition_on_full if args.method == "transpose" else None
        ),
        "human_condition_on_object": (
            True
            if args.method == "tip"
            else getattr(model, "human_condition_on_object", False)
        ),
        "data_dirs": data_dirs,
        "num_sequences": len(per_sequence),
        "aggregation": "macro_mean_over_sequences",
        "metrics": metrics,
        "metric_units": METRIC_UNITS,
        "metric_counts": {
            name: aggregate.counts.get(name, 0) for name in METRIC_UNITS
        },
        "per_sequence": per_sequence,
    }
    output_json = args.output_json
    if output_json is None:
        output_json = str(
            checkpoint_path.parent
            / f"eval_{args.dataset or 'custom'}_{args.split}_{protocol_name}.json"
        )
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f"saved evaluation report to {output_path}")


if __name__ == "__main__":
    main()
