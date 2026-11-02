"""Render GT and predicted body meshes from eval_comparison predictions.

Rows are GT followed by one row per ``--prediction``; columns are frames.
Every panel is centred on that body's own root, so a malformed pose ("a
twisted blob") is visible even when the predicted translation drifts.  The
camera direction is fixed per sequence (in front of the GT body at the first
rendered frame) and shared by all rows.

Example:
    python -m Comparisons.render_comparison_meshes \
        --prediction dip18=outputs/Comparisons/<run>/predictions/<seq>.pt \
        --prediction tip=outputs/Comparisons/<run2>/predictions/<seq>.pt \
        --output outputs/Comparisons/vis/<seq>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.human_pose import load_body_model
from utils.rotation_conversions import matrix_to_axis_angle

ROW_COLORS = {
    "GT": (0.62, 0.66, 0.78),
}
PRED_COLOR = (0.86, 0.45, 0.38)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="label=path to a sequence .pt written by eval_comparison --save_predictions",
    )
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument(
        "--body_model_path",
        default="datasets/smpl_models/smplx/SMPLX_NEUTRAL.npz",
    )
    parser.add_argument("--frames", default=None, help="Comma-separated frame indices")
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _vertices(body_model, local_rot: torch.Tensor, trans: torch.Tensor, device) -> torch.Tensor:
    pose = matrix_to_axis_angle(local_rot[:, :22].reshape(-1, 3, 3).double()).float()
    pose = pose.reshape(local_rot.shape[0], 22, 3).to(device)
    output = body_model(
        root_orient=pose[:, 0],
        pose_body=pose[:, 1:22].reshape(pose.shape[0], 63),
        trans=trans.to(device),
    )
    return output.v


def main() -> None:
    args = parse_args()
    from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        HardPhongShader,
        MeshRasterizer,
        MeshRenderer,
        PointLights,
        RasterizationSettings,
        TexturesVertex,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    device = torch.device(args.device)
    entries = []
    for item in args.prediction:
        label, _, path = item.partition("=")
        if not path:
            label, path = Path(item).stem, item
        entries.append((label, _load(path)))
    reference = entries[0][1]
    if "target_local_rot" not in reference:
        raise ValueError("prediction files lack target_local_rot; rerun eval_comparison")
    length = min(int(payload["local_rot"].shape[0]) for _, payload in entries)
    if args.frames:
        frames = [int(token) for token in args.frames.split(",") if token.strip()]
    else:
        frames = np.linspace(0, length - 1, args.num_frames).round().astype(int).tolist()
    frames = [frame for frame in frames if 0 <= frame < length]

    body_model_path = Path(args.body_model_path)
    if not body_model_path.is_absolute():
        body_model_path = PROJECT_ROOT / body_model_path
    body_model = load_body_model(str(body_model_path)).to(device)
    faces = None

    rows = [("GT", reference["target_local_rot"], reference["target_root_trans"])]
    rows += [(label, payload["local_rot"], payload["root_trans"]) for label, payload in entries]
    vertices = []
    for _, local_rot, trans in rows:
        with torch.no_grad():
            vertices.append(_vertices(body_model, local_rot[frames], trans[frames], device))
    faces = body_model.f.long().to(device)

    # View from the front of the GT body at the first rendered frame.
    root0 = reference["target_local_rot"][frames[0], 0]
    forward = root0[:, 2].clone()
    forward[1] = 0.0
    forward = forward / forward.norm().clamp_min(1e-6)
    raster = RasterizationSettings(image_size=args.image_size, blur_radius=0.0, faces_per_pixel=1)

    tiles = []
    for row_index, (label, _, _) in enumerate(rows):
        color = torch.tensor(ROW_COLORS.get(label, PRED_COLOR), device=device)
        row_tiles = []
        for column in range(len(frames)):
            verts = vertices[row_index][column]
            center = verts.mean(dim=0)
            eye = center + forward.to(device) * 3.2 + torch.tensor([0.0, 0.4, 0.0], device=device)
            rotation, translation = look_at_view_transform(
                eye=eye[None], at=center[None], up=((0.0, 1.0, 0.0),), device=device
            )
            cameras = FoVPerspectiveCameras(R=rotation, T=translation, fov=38.0, device=device)
            renderer = MeshRenderer(
                MeshRasterizer(cameras=cameras, raster_settings=raster),
                HardPhongShader(
                    cameras=cameras,
                    lights=PointLights(location=(eye + torch.tensor([0.0, 2.0, 0.0], device=device))[None], device=device),
                    device=device,
                ),
            )
            mesh = Meshes(
                verts=verts[None],
                faces=faces[None],
                textures=TexturesVertex(verts_features=color.expand_as(verts)[None]),
            )
            with torch.no_grad():
                image = renderer(mesh)[0, ..., :3].clamp(0.0, 1.0)
            row_tiles.append((image.cpu().numpy() * 255).astype(np.uint8))
        tiles.append(np.concatenate(row_tiles, axis=1))
    canvas = np.concatenate(tiles, axis=0)
    canvas = _annotate(canvas, [label for label, _, _ in rows], frames, args.image_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output, canvas)
    print(f"saved {output} rows={[label for label, _, _ in rows]} frames={frames}")


def _annotate(canvas: np.ndarray, labels: list[str], frames: list[int], size: int) -> np.ndarray:
    """Write row labels and frame indices with matplotlib's text renderer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height, width = canvas.shape[:2]
    figure = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.imshow(canvas)
    axis.axis("off")
    for row, label in enumerate(labels):
        axis.text(6, row * size + 18, label, fontsize=11, color="black", weight="bold")
    for column, frame in enumerate(frames):
        axis.text(column * size + size - 70, 16, f"f{frame}", fontsize=9, color="dimgray")
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return image


if __name__ == "__main__":
    main()
