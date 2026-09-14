"""Dense surface-sample nearest-neighbour queries (no rtree needed)."""
from __future__ import annotations

import os

import numpy as np
import torch
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MESH_DIR = os.path.join(ROOT, "datasets/OMOMO/captured_objects")


def load_mesh(name: str) -> trimesh.Trimesh:
    path = os.path.join(MESH_DIR, f"{name}_cleaned_simplified.obj")
    m = trimesh.load_mesh(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    v = np.asarray(m.vertices, dtype=np.float64)
    v = v - v.mean(axis=0)  # preprocess centres at the vertex centroid
    return trimesh.Trimesh(vertices=v, faces=np.asarray(m.faces), process=False)


class SurfaceNN:
    """Nearest surface point via a dense surface sampling, in the canonical
    (centroid-centred, unscaled) mesh frame."""

    def __init__(self, mesh: trimesh.Trimesh, n_samples: int = 60000, device="cuda", seed: int = 0):
        pts, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
        # always include vertices so thin parts are represented
        allpts = np.concatenate((np.asarray(pts), np.asarray(mesh.vertices)), axis=0).astype(np.float32)
        allpts = allpts[np.isfinite(allpts).all(axis=1)]
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        verts = verts[np.isfinite(verts).all(axis=1)]
        self.pts = torch.from_numpy(allpts).to(device)
        self.verts = torch.from_numpy(verts).to(device)
        self.device = device

    @torch.no_grad()
    def query(self, q: torch.Tensor, chunk: int = 2048):
        """q: [M,3] canonical unscaled -> (closest [M,3], dist [M])."""
        q = q.to(self.device, torch.float32)
        out_c, out_d = [], []
        for i in range(0, q.shape[0], chunk):
            d = torch.cdist(q[i : i + chunk], self.pts)
            dmin, idx = d.min(dim=1)
            out_c.append(self.pts[idx])
            out_d.append(dmin)
        if not out_c:
            return q.new_zeros(0, 3), q.new_zeros(0)
        return torch.cat(out_c), torch.cat(out_d)

    @torch.no_grad()
    def min_y_world(self, R: torch.Tensor, s: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Lowest vertex height after Rotate->Scale->Translate. R [T,3,3], s [T], p [T,3] -> [T]."""
        R = R.to(self.device, torch.float32)
        s = s.to(self.device, torch.float32)
        p = p.to(self.device, torch.float32)
        y = torch.einsum("tj,nj->tn", R[:, 1, :], self.verts) * s.view(-1, 1) + p[:, 1:2]
        return y.min(dim=1).values
