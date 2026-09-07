from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from plyfile import PlyData

from mast3r_slam.evaluate import save_ply

NODE_COLORS = {
    "O": np.array([255, 0, 0], dtype=np.uint8),
    "C": np.array([255, 255, 0], dtype=np.uint8),
    "U": np.array([0, 0, 255], dtype=np.uint8),
}

EDGE_COLORS = {
    "O-C": np.array([0, 255, 0], dtype=np.uint8),
    "C-U": np.array([255, 0, 255], dtype=np.uint8),
    "O-U": np.array([255, 165, 0], dtype=np.uint8),
    "remote": np.array([255, 255, 255], dtype=np.uint8),
}


def _edge_render_style(
    edge,
    *,
    base_radius: float,
    base_ring_samples: int,
    remote_tentative_style: bool = False,
) -> tuple[np.ndarray, float, int]:
    base_color = EDGE_COLORS.get(edge.edge_type, EDGE_COLORS["remote"])
    if remote_tentative_style and edge.edge_type == "remote":
        return (
            np.array([170, 220, 255], dtype=np.uint8),
            max(base_radius * 0.68, 0.0025),
            max(6, int(round(base_ring_samples * 0.65))),
        )
    status = str(getattr(edge, "status", "committed") or "committed").strip().lower()
    if status != "tentative":
        return base_color, base_radius, base_ring_samples

    if edge.edge_type == "remote":
        color = np.array([170, 220, 255], dtype=np.uint8)
    else:
        color = np.clip(base_color.astype(np.float32) * 0.6 + 255.0 * 0.4, 0.0, 255.0).astype(np.uint8)
    radius = max(base_radius * 0.68, 0.0025)
    ring_samples = max(6, int(round(base_ring_samples * 0.65)))
    return color, radius, ring_samples


@dataclass
class OverlayVizConfig:
    scene_keep_ratio: float = 0.15
    clear_scene_near_graph: bool = True
    clear_node_radius_scale: float = 1.8
    clear_edge_radius_scale: float = 1.6
    show_fused_bbox_wireframe: bool = False
    show_auxiliary_anchor_centroids: bool = False
    remote_edges_tentative_style: bool = False
    export_mode: str = "balanced"
    random_seed: int = 0


def infer_scene_base_ply_path(dataset_path: str | Path, sequence_name: str | Path) -> Optional[Path]:
    dataset_path = Path(str(dataset_path))
    sequence_name = Path(str(sequence_name))
    scene_name = sequence_name.parts[0] if len(sequence_name.parts) > 0 else ""
    if scene_name:
        scene_candidate = dataset_path.parents[1] / f"{scene_name}.ply"
        if scene_candidate.exists():
            return scene_candidate
    for parent in dataset_path.parents:
        candidate = parent / f"{parent.name}.ply"
        if candidate.exists():
            return candidate
    return None


def build_pred_world_reconstruction(
    keyframes,
    *,
    c_conf_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    pointclouds: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if keyframe is None:
            continue
        has_real_rgb = getattr(keyframe, "has_real_rgb", None)
        if has_real_rgb is False:
            continue
        uimg_source = getattr(keyframe, "uimg_source", None)
        if uimg_source is not None and str(uimg_source) != "rgb":
            continue
        X_canon = getattr(keyframe, "X_canon", None)
        C = getattr(keyframe, "C", None)
        uimg = getattr(keyframe, "uimg", None)
        T_WC = getattr(keyframe, "T_WC", None)
        if X_canon is None or C is None or uimg is None or T_WC is None:
            continue
        if not torch.is_tensor(X_canon) or X_canon.numel() == 0:
            continue
        if not torch.is_tensor(C) or C.numel() == 0:
            continue
        if not torch.is_tensor(uimg) or uimg.numel() == 0:
            continue
        if hasattr(T_WC, "act"):
            points_world_t = T_WC.act(X_canon)
        else:
            pose = torch.as_tensor(T_WC, dtype=X_canon.dtype, device=X_canon.device).reshape(4, 4)
            ones = torch.ones((X_canon.shape[0], 1), dtype=X_canon.dtype, device=X_canon.device)
            homo = torch.cat([X_canon.reshape(-1, 3), ones], dim=-1)
            points_world_t = (homo @ pose.transpose(-1, -2))[..., :3]
        points_world = points_world_t.detach().cpu().numpy().reshape(-1, 3)
        color = (uimg.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8).reshape(-1, 3)
        conf = C.detach().cpu().numpy().astype(np.float32).reshape(-1)
        valid = conf > float(c_conf_threshold)
        if int(valid.sum()) == 0:
            continue
        pointclouds.append(points_world[valid])
        colors.append(color[valid])
    if not pointclouds:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(pointclouds, axis=0), np.concatenate(colors, axis=0)


def _as_numpy_xyz(vertex_data) -> np.ndarray:
    return np.stack([vertex_data[axis] for axis in ("x", "y", "z")], axis=1).astype(np.float32)


def _as_numpy_rgb(vertex_data) -> np.ndarray:
    return np.stack([vertex_data[channel] for channel in ("red", "green", "blue")], axis=1).astype(np.uint8)


def _subsample_scene_points(
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    keep_ratio: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if scene_points.size == 0:
        return scene_points, scene_colors
    ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    if ratio >= 0.999:
        return scene_points, scene_colors
    n = int(scene_points.shape[0])
    keep_n = max(1, int(round(n * ratio)))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=keep_n, replace=False))
    return scene_points[indices], scene_colors[indices]


def _point_to_segment_dist2(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 <= 1e-12:
        return np.sum((points - a[None, :]) ** 2, axis=1)
    t = np.sum((points - a[None, :]) * ab[None, :], axis=1) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.sum((points - proj) ** 2, axis=1)


def _mask_points_near_nodes(
    scene_points: np.ndarray,
    node_positions: dict[str, np.ndarray],
    radius: float,
) -> np.ndarray:
    keep = np.ones(scene_points.shape[0], dtype=bool)
    if not node_positions or scene_points.size == 0 or radius <= 0.0:
        return keep
    centers = np.stack(list(node_positions.values()), axis=0)
    for center in centers:
        dist2 = np.sum((scene_points - center[None, :]) ** 2, axis=1)
        keep &= dist2 > (radius * radius)
    return keep


def _mask_points_near_edges(
    scene_points: np.ndarray,
    edge_segments: list[tuple[np.ndarray, np.ndarray]],
    radius: float,
) -> np.ndarray:
    keep = np.ones(scene_points.shape[0], dtype=bool)
    if not edge_segments or scene_points.size == 0 or radius <= 0.0:
        return keep
    thr2 = float(radius * radius)
    for a, b in edge_segments:
        dist2 = _point_to_segment_dist2(scene_points, a, b)
        keep &= dist2 > thr2
    return keep


def _fibonacci_sphere(center: np.ndarray, radius: float, samples: int) -> np.ndarray:
    if samples <= 1:
        return center.reshape(1, 3).astype(np.float32)
    idx = np.arange(samples, dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * idx) / max(1.0, samples - 1.0)
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * idx
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    pts = np.stack([x, y, z], axis=1)
    return (center[None, :] + radius * pts).astype(np.float32)


def _sample_ball_points(center: np.ndarray, radius: float, shell_samples: int, shell_count: int) -> np.ndarray:
    if shell_count <= 1:
        return _fibonacci_sphere(center, radius, shell_samples)
    layers = [center.reshape(1, 3).astype(np.float32)]
    for shell_idx in range(1, shell_count + 1):
        frac = float(shell_idx) / float(shell_count)
        shell_radius = radius * frac
        samples = max(24, int(round(shell_samples * frac * frac)))
        layers.append(_fibonacci_sphere(center, shell_radius, samples))
    return np.concatenate(layers, axis=0)


def _sample_segment_points(start: np.ndarray, end: np.ndarray, spacing: float) -> np.ndarray:
    dist = float(np.linalg.norm(end - start))
    if dist <= 1e-9:
        return start.reshape(1, 3).astype(np.float32)
    n_steps = max(2, int(np.ceil(dist / max(spacing, 1e-4))) + 1)
    t = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    pts = start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]
    return pts.astype(np.float32)


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction.astype(np.float32)
    direction = direction / max(np.linalg.norm(direction), 1e-9)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(direction, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    basis1 = np.cross(direction, ref)
    basis1 = basis1 / max(np.linalg.norm(basis1), 1e-9)
    basis2 = np.cross(direction, basis1)
    basis2 = basis2 / max(np.linalg.norm(basis2), 1e-9)
    return basis1.astype(np.float32), basis2.astype(np.float32)


def _sample_tube_points(
    start: np.ndarray,
    end: np.ndarray,
    *,
    spacing: float,
    radius: float,
    ring_samples: int,
) -> np.ndarray:
    centers = _sample_segment_points(start, end, spacing)
    if centers.shape[0] == 1 or radius <= 1e-9 or ring_samples <= 2:
        return centers

    direction = end - start
    basis1, basis2 = _orthonormal_basis(direction)
    angles = np.linspace(0.0, 2.0 * np.pi, ring_samples, endpoint=False, dtype=np.float32)
    radial_levels = (0.55 * radius, radius)

    tube_parts = [centers]
    for radial in radial_levels:
        offsets = []
        for angle in angles:
            offset = np.cos(angle) * basis1 + np.sin(angle) * basis2
            offsets.append(centers + radial * offset[None, :])
        tube_parts.append(np.concatenate(offsets, axis=0))
    return np.concatenate(tube_parts, axis=0).astype(np.float32)


def _track_anchor_frame(track, keyframes, anchor_kf_id: Optional[int]):
    if anchor_kf_id is None:
        return None
    anchor_frame_fn = getattr(track, "_anchor_frame", None)
    if callable(anchor_frame_fn):
        try:
            return anchor_frame_fn(keyframes, anchor_kf_id)
        except Exception:
            pass
    try:
        if 0 <= int(anchor_kf_id) < len(keyframes):
            return keyframes[int(anchor_kf_id)]
    except Exception:
        pass
    try:
        for frame in keyframes:
            if int(getattr(frame, "frame_id", -1)) == int(anchor_kf_id):
                return frame
    except Exception:
        pass
    try:
        if len(keyframes) == 1:
            return keyframes[0]
    except Exception:
        pass
    return None


def _points_tensor_from_track_attr(track, attr_name: str, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
    value = getattr(track, attr_name, None)
    if value is None:
        return None
    try:
        points = torch.as_tensor(value, device=device, dtype=dtype)
    except Exception:
        return None
    if points.ndim == 1:
        if points.numel() != 3:
            return None
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or int(points.shape[0]) == 0:
        return None
    finite = torch.isfinite(points).all(dim=1)
    if not bool(finite.any()):
        return None
    return points[finite]


def _pose_act_points(pose, points: torch.Tensor) -> Optional[torch.Tensor]:
    if pose is None or points is None or int(points.shape[0]) == 0:
        return None
    try:
        pose_data = getattr(pose, "data", None)
        if torch.is_tensor(pose_data):
            points = points.to(device=pose_data.device, dtype=pose_data.dtype)
        elif torch.is_tensor(pose):
            points = points.to(device=pose.device, dtype=pose.dtype)
        if hasattr(pose, "act"):
            return pose.act(points)
        pose_t = torch.as_tensor(pose, dtype=points.dtype, device=points.device).reshape(4, 4)
        ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
        homo = torch.cat([points, ones], dim=-1)
        return (homo @ pose_t.transpose(-1, -2))[..., :3]
    except Exception:
        return None


def _resolve_track_point_cloud_centroid_world(track, keyframes) -> tuple[Optional[np.ndarray], Optional[str]]:
    # For paper-oriented overlays, prefer the observed object surface point
    # centroid over stored centroid fields that may have fallen back to a 3D
    # bbox center. This only changes marker placement, not graph geometry.
    candidates: list[tuple[str, str, Optional[int]]] = []
    anchor_kf_id = getattr(track, "anchor_kf_id", None)
    if anchor_kf_id is not None:
        candidates.extend(
            [
                ("points_fused_local", "fused_point_cloud_centroid", anchor_kf_id),
                ("points_anchor", "anchor_point_cloud_centroid", anchor_kf_id),
            ]
        )
    candidate_kf_id = getattr(track, "candidate_anchor_kf_id", None)
    if candidate_kf_id is not None:
        candidates.extend(
            [
                ("provisional_points_frame", "provisional_point_cloud_centroid", candidate_kf_id),
                ("candidate_points_frame", "candidate_point_cloud_centroid", candidate_kf_id),
            ]
        )

    for attr_name, source, kf_id in candidates:
        frame = _track_anchor_frame(track, keyframes, kf_id)
        if frame is None:
            continue
        pose = getattr(frame, "T_WC", None)
        if pose is None:
            continue
        points = _points_tensor_from_track_attr(track, attr_name)
        if points is None:
            continue
        world_points = _pose_act_points(pose, points)
        if world_points is None or int(world_points.shape[0]) == 0:
            continue
        centroid = world_points.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        return centroid, source

    return None, None


def _resolve_node_world_position_with_source(track, keyframes) -> tuple[Optional[np.ndarray], Optional[str]]:
    override = getattr(track, "temp_visual_centroid_world", None)
    if override is not None:
        try:
            arr = np.asarray(override, dtype=np.float32).reshape(3)
            source = str(getattr(track, "temp_visual_centroid_source", "") or "temp_point_cloud_centroid")
            return arr, f"temp_{source}"
        except Exception:
            pass
    point_centroid, source = _resolve_track_point_cloud_centroid_world(track, keyframes)
    if point_centroid is not None:
        return point_centroid, source
    try:
        predicted = track.predict_centroid_world(keyframes)
    except Exception:
        predicted = None
    if predicted is not None:
        return predicted.detach().cpu().numpy().astype(np.float32), "predicted_centroid"
    if track.centroid_world_est is not None:
        return np.asarray(track.centroid_world_est, dtype=np.float32), "centroid_world_est"
    if track.anchor_kf_id is not None and track.centroid_anchor is not None:
        keyframe = _track_anchor_frame(track, keyframes, track.anchor_kf_id)
        if keyframe is None:
            return None
        anchor = torch.tensor(track.centroid_anchor, dtype=keyframe.X_canon.dtype, device=keyframe.X_canon.device).reshape(1, 3)
        pos = keyframe.T_WC.act(anchor).reshape(-1, 3)[0].detach().cpu().numpy().astype(np.float32)
        return pos, "centroid_anchor"
    return None, None


def _resolve_node_world_position(track, keyframes) -> Optional[np.ndarray]:
    pos, _ = _resolve_node_world_position_with_source(track, keyframes)
    return pos


def _resolve_track_world_bbox_corners(track, keyframes) -> Optional[np.ndarray]:
    anchor_kf_id = getattr(track, "anchor_kf_id", None)
    if anchor_kf_id is None or anchor_kf_id < 0 or anchor_kf_id >= len(keyframes):
        return None
    bbox_local = getattr(track, "bbox_fused_local", None) or getattr(track, "bbox_anchor", None)
    if not isinstance(bbox_local, dict):
        return None
    bmin = torch.as_tensor(bbox_local.get("min", []), dtype=torch.float32)
    bmax = torch.as_tensor(bbox_local.get("max", []), dtype=torch.float32)
    if bmin.numel() != 3 or bmax.numel() != 3:
        return None
    corners = torch.stack(
        [
            torch.tensor([bmin[0], bmin[1], bmin[2]], dtype=torch.float32),
            torch.tensor([bmin[0], bmin[1], bmax[2]], dtype=torch.float32),
            torch.tensor([bmin[0], bmax[1], bmin[2]], dtype=torch.float32),
            torch.tensor([bmin[0], bmax[1], bmax[2]], dtype=torch.float32),
            torch.tensor([bmax[0], bmin[1], bmin[2]], dtype=torch.float32),
            torch.tensor([bmax[0], bmin[1], bmax[2]], dtype=torch.float32),
            torch.tensor([bmax[0], bmax[1], bmin[2]], dtype=torch.float32),
            torch.tensor([bmax[0], bmax[1], bmax[2]], dtype=torch.float32),
        ],
        dim=0,
    )
    keyframe = keyframes[anchor_kf_id]
    pose = getattr(keyframe, "T_WC", None)
    if pose is None:
        return None
    if hasattr(pose, "act"):
        world = pose.act(corners)
    else:
        pose_t = torch.as_tensor(pose, dtype=corners.dtype).reshape(4, 4)
        ones = torch.ones((corners.shape[0], 1), dtype=corners.dtype)
        homo = torch.cat([corners, ones], dim=-1)
        world = (homo @ pose_t.transpose(-1, -2))[..., :3]
    return world.detach().cpu().numpy().astype(np.float32)


def _bbox_wireframe_segments(corners: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if corners.shape != (8, 3):
        return []
    edge_indices = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    return [(corners[i], corners[j]) for i, j in edge_indices]


def _iter_local_edges(graph) -> Iterable:
    return getattr(graph, "local_edges", {}).values()


def _iter_remote_edges(graph) -> Iterable:
    return getattr(graph, "remote_edges", {}).values()


def export_functional_graph_overlay_from_points(
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    output_ply_path: str | Path,
    online_state,
    keyframes,
    *,
    viz_config: Optional[OverlayVizConfig] = None,
    sphere_shell_samples: int = 384,
    sphere_shell_count: int = 5,
    edge_ring_samples: int = 12,
) -> Path:
    output_ply_path = Path(output_ply_path)
    viz_config = viz_config or OverlayVizConfig()

    scene_points, scene_colors = _subsample_scene_points(
        scene_points,
        scene_colors,
        keep_ratio=viz_config.scene_keep_ratio,
        seed=viz_config.random_seed,
    )

    bbox_min = scene_points.min(axis=0)
    bbox_max = scene_points.max(axis=0)
    scene_diag = float(np.linalg.norm(bbox_max - bbox_min))
    node_radius = float(np.clip(scene_diag * 0.0045, 0.01, 0.03))
    edge_radius = float(np.clip(node_radius * 0.45, 0.004, 0.015))
    edge_spacing = max(edge_radius * 0.6, 0.003)

    overlay_points: list[np.ndarray] = []
    overlay_colors: list[np.ndarray] = []
    node_positions: dict[str, np.ndarray] = {}
    edge_segments: list[tuple[np.ndarray, np.ndarray]] = []
    hidden_nodes: set[str] = set()
    policy = getattr(online_state, "graph_policy", None)

    for node_id, track in online_state.node_tracks.items():
        if policy is not None and policy.should_hide_overlay_node(
            node_id=node_id,
            label=getattr(track, "label", ""),
            origin=getattr(track, "origin", "standard"),
        ):
            hidden_nodes.add(node_id)
            continue
        pos = _resolve_node_world_position(track, keyframes)
        if pos is None:
            continue
        node_positions[node_id] = pos
        sphere = _sample_ball_points(pos, node_radius, sphere_shell_samples, sphere_shell_count)
        color = NODE_COLORS.get(track.role, np.array([255, 255, 255], dtype=np.uint8))
        overlay_points.append(sphere)
        overlay_colors.append(np.repeat(color[None, :], sphere.shape[0], axis=0))
        if viz_config.show_fused_bbox_wireframe:
            corners = _resolve_track_world_bbox_corners(track, keyframes)
            if corners is not None:
                for start, end in _bbox_wireframe_segments(corners):
                    wire = _sample_segment_points(start, end, max(edge_spacing * 0.9, 0.0025))
                    overlay_points.append(wire)
                    overlay_colors.append(np.repeat(color[None, :], wire.shape[0], axis=0))

    # Auxiliary anchor centroids: render as small cyan spheres when enabled.
    # Default off – does not affect primary/fused geometry rendering.
    if viz_config.show_auxiliary_anchor_centroids:
        _AUX_COLOR = np.array([0, 255, 255], dtype=np.uint8)
        _aux_radius = float(node_radius * 0.45)
        for _node_id, _track in online_state.node_tracks.items():
            if _node_id in hidden_nodes:
                continue
            for _aux in getattr(_track, "auxiliary_anchors", []):
                _akf = _aux.get("kf_id")
                _ac = _aux.get("source_centroid_frame")
                if _akf is None or _ac is None:
                    continue
                try:
                    _kf = keyframes[_akf]
                except (IndexError, TypeError):
                    continue
                _apose = getattr(_kf, "T_WC", None)
                if _apose is None:
                    continue
                _ct = torch.tensor(_ac, dtype=torch.float32).reshape(1, 3)
                if hasattr(_apose, "act"):
                    _cw = _apose.act(_ct)
                else:
                    _pt = torch.as_tensor(_apose, dtype=_ct.dtype).reshape(4, 4)
                    _ones = torch.ones((1, 1), dtype=_ct.dtype)
                    _homo = torch.cat([_ct, _ones], dim=-1)
                    _cw = (_homo @ _pt.transpose(-1, -2))[..., :3]
                _pos = _cw[0].detach().cpu().numpy().astype(np.float32)
                _sphere = _fibonacci_sphere(_pos, _aux_radius, 96)
                overlay_points.append(_sphere)
                overlay_colors.append(np.repeat(_AUX_COLOR[None, :], _sphere.shape[0], axis=0))

    for edge in _iter_local_edges(online_state.graph):
        if edge.src_node_id in hidden_nodes or edge.dst_node_id in hidden_nodes:
            continue
        src = node_positions.get(edge.src_node_id)
        dst = node_positions.get(edge.dst_node_id)
        if src is None or dst is None:
            continue
        edge_segments.append((src, dst))
        color, edge_radius_now, edge_ring_samples_now = _edge_render_style(
            edge,
            base_radius=edge_radius,
            base_ring_samples=edge_ring_samples,
            remote_tentative_style=viz_config.remote_edges_tentative_style,
        )
        segment = _sample_tube_points(
            src,
            dst,
            spacing=edge_spacing,
            radius=edge_radius_now,
            ring_samples=edge_ring_samples_now,
        )
        overlay_points.append(segment)
        overlay_colors.append(np.repeat(color[None, :], segment.shape[0], axis=0))

    for edge in _iter_remote_edges(online_state.graph):
        if edge.src_node_id in hidden_nodes or edge.dst_node_id in hidden_nodes:
            continue
        src = node_positions.get(edge.src_node_id)
        dst = node_positions.get(edge.dst_node_id)
        if src is None or dst is None:
            continue
        edge_segments.append((src, dst))
        color, edge_radius_now, edge_ring_samples_now = _edge_render_style(
            edge,
            base_radius=edge_radius,
            base_ring_samples=edge_ring_samples,
            remote_tentative_style=viz_config.remote_edges_tentative_style,
        )
        segment = _sample_tube_points(
            src,
            dst,
            spacing=edge_spacing,
            radius=edge_radius_now,
            ring_samples=edge_ring_samples_now,
        )
        overlay_points.append(segment)
        overlay_colors.append(np.repeat(color[None, :], segment.shape[0], axis=0))

    if viz_config.clear_scene_near_graph and scene_points.size > 0:
        node_clear_r = float(node_radius * viz_config.clear_node_radius_scale)
        edge_clear_r = float(edge_radius * viz_config.clear_edge_radius_scale)
        keep_node = _mask_points_near_nodes(scene_points, node_positions, node_clear_r)
        keep_edge = _mask_points_near_edges(scene_points, edge_segments, edge_clear_r)
        keep = keep_node & keep_edge
        scene_points = scene_points[keep]
        scene_colors = scene_colors[keep]

    if overlay_points:
        points = np.concatenate([scene_points, *overlay_points], axis=0)
        colors = np.concatenate([scene_colors, *overlay_colors], axis=0)
    else:
        points = scene_points
        colors = scene_colors

    output_ply_path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(output_ply_path, points, colors)
    return output_ply_path


def export_functional_graph_overlay_ply(
    base_ply_path: str | Path,
    output_ply_path: str | Path,
    online_state,
    keyframes,
    *,
    viz_config: Optional[OverlayVizConfig] = None,
    sphere_shell_samples: int = 384,
    sphere_shell_count: int = 5,
    edge_ring_samples: int = 12,
) -> Path:
    base_ply_path = Path(base_ply_path)
    if not base_ply_path.exists():
        raise FileNotFoundError(f"base ply not found: {base_ply_path}")

    ply_data = PlyData.read(base_ply_path)
    vertex = ply_data["vertex"].data
    scene_points = _as_numpy_xyz(vertex)
    scene_colors = _as_numpy_rgb(vertex)
    return export_functional_graph_overlay_from_points(
        scene_points=scene_points,
        scene_colors=scene_colors,
        output_ply_path=output_ply_path,
        online_state=online_state,
        keyframes=keyframes,
        viz_config=viz_config,
        sphere_shell_samples=sphere_shell_samples,
        sphere_shell_count=sphere_shell_count,
        edge_ring_samples=edge_ring_samples,
    )


def export_progressive_functional_graph_overlay(
    base_ply_path: str | Path,
    output_dir: str | Path,
    *,
    frame_idx: int,
    online_state,
    keyframes,
    viz_config: Optional[OverlayVizConfig] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"assoc_overlay_frame_{int(frame_idx):06d}.ply"
    return export_functional_graph_overlay_ply(
        base_ply_path=base_ply_path,
        output_ply_path=output_path,
        online_state=online_state,
        keyframes=keyframes,
        viz_config=viz_config,
    )


def export_progressive_functional_graph_overlay_pred_world(
    output_dir: str | Path,
    *,
    frame_idx: int,
    online_state,
    keyframes,
    viz_config: Optional[OverlayVizConfig] = None,
    c_conf_threshold: float = 0.3,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"assoc_overlay_frame_{int(frame_idx):06d}.ply"
    scene_points, scene_colors = build_pred_world_reconstruction(keyframes, c_conf_threshold=c_conf_threshold)
    return export_functional_graph_overlay_from_points(
        scene_points=scene_points,
        scene_colors=scene_colors,
        output_ply_path=output_path,
        online_state=online_state,
        keyframes=keyframes,
        viz_config=viz_config,
    )


def export_functional_graph_overlay_pred_world(
    output_ply_path: str | Path,
    online_state,
    keyframes,
    *,
    viz_config: Optional[OverlayVizConfig] = None,
    c_conf_threshold: float = 0.3,
) -> Path:
    scene_points, scene_colors = build_pred_world_reconstruction(keyframes, c_conf_threshold=c_conf_threshold)
    return export_functional_graph_overlay_from_points(
        scene_points=scene_points,
        scene_colors=scene_colors,
        output_ply_path=output_ply_path,
        online_state=online_state,
        keyframes=keyframes,
        viz_config=viz_config,
    )


def export_overlay_edge_sidecar(
    output_path: str | Path,
    online_state,
    keyframes,
    *,
    viz_config: Optional[OverlayVizConfig] = None,
) -> Path:
    """Write a JSON sidecar describing every overlay-rendered edge plus the
    set of edges that *would* have rendered but were skipped because one or
    both endpoints had no resolvable world position (``skip_reason``).

    Output schema::

        {
          "frame_idx": int | null,
          "nodes": {node_id: {role, label, pos:[x,y,z]}, ...},
          "hidden_nodes": [...],
          "edges": [
            {
              "kind": "local" | "remote",
              "src": str, "dst": str,
              "edge_type": str, "relation_text": str,
              "status": str, "support_count": int, "margin": float,
              "first_seen_frame": int, "last_seen_frame": int,
              "rendered": bool, "skip_reason": str | null
            }, ...
          ],
          "summary": {"num_local": int, "num_remote": int, "num_skipped": int}
        }
    """
    output_path = Path(output_path)
    viz_config = viz_config or OverlayVizConfig()
    policy = getattr(online_state, "graph_policy", None)

    node_payload: dict = {}
    hidden_nodes: list[str] = []
    node_positions: dict = {}
    for node_id, track in online_state.node_tracks.items():
        if policy is not None and policy.should_hide_overlay_node(
            node_id=node_id,
            label=getattr(track, "label", ""),
            origin=getattr(track, "origin", "standard"),
        ):
            hidden_nodes.append(node_id)
            continue
        pos, pos_source = _resolve_node_world_position_with_source(track, keyframes)
        if pos is None:
            continue
        node_positions[node_id] = pos
        node_payload[node_id] = {
            "role": getattr(track, "role", None),
            "label": getattr(track, "label", None),
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "pos_source": pos_source,
        }

    edges_payload: list[dict] = []

    def _edge_skip_reason(edge) -> Optional[str]:
        if edge.src_node_id in hidden_nodes or edge.dst_node_id in hidden_nodes:
            return "hidden_endpoint"
        if (
            edge.src_node_id not in node_positions
            or edge.dst_node_id not in node_positions
        ):
            return "missing_endpoint_position"
        return None

    def _edge_record(edge, kind: str) -> dict:
        skip = _edge_skip_reason(edge)
        render_color, render_radius_scale, render_ring_samples = _edge_render_style(
            edge,
            base_radius=1.0,
            base_ring_samples=16,
            remote_tentative_style=viz_config.remote_edges_tentative_style,
        )
        return {
            "kind": kind,
            "src": str(edge.src_node_id),
            "dst": str(edge.dst_node_id),
            "edge_type": str(getattr(edge, "edge_type", "") or ""),
            "relation_text": str(getattr(edge, "relation_text", "") or ""),
            "status": str(getattr(edge, "status", "") or ""),
            "support_count": int(getattr(edge, "support_count", 0) or 0),
            "recent_support_count": int(
                getattr(edge, "recent_support_count", 0) or 0
            ),
            "margin": float(getattr(edge, "margin", 0.0) or 0.0),
            "latest_margin": float(getattr(edge, "latest_margin", 0.0) or 0.0),
            "evidence_score": float(getattr(edge, "evidence_score", 0.0) or 0.0),
            "first_seen_frame": int(getattr(edge, "first_seen_frame", -1) or -1),
            "last_seen_frame": int(getattr(edge, "last_seen_frame", -1) or -1),
            "last_update_source": str(
                getattr(edge, "last_update_source", "") or ""
            ),
            "retention_policy": str(
                getattr(edge, "retention_policy", "") or ""
            ),
            "render_color_rgb": [int(v) for v in render_color.tolist()],
            "render_radius_scale": float(render_radius_scale),
            "render_ring_samples": int(render_ring_samples),
            "rendered": skip is None,
            "skip_reason": skip,
        }

    num_local = num_remote = num_skipped = 0
    for edge in _iter_local_edges(online_state.graph):
        rec = _edge_record(edge, "local")
        edges_payload.append(rec)
        num_local += 1
        if not rec["rendered"]:
            num_skipped += 1
    for edge in _iter_remote_edges(online_state.graph):
        rec = _edge_record(edge, "remote")
        edges_payload.append(rec)
        num_remote += 1
        if not rec["rendered"]:
            num_skipped += 1

    frame_order = getattr(online_state, "_frame_order", None) or []
    latest_frame = int(frame_order[-1]) if len(frame_order) else -1
    payload = {
        "frame_idx": latest_frame,
        "nodes": node_payload,
        "hidden_nodes": list(hidden_nodes),
        "edges": edges_payload,
        "summary": {
            "num_local": int(num_local),
            "num_remote": int(num_remote),
            "num_skipped": int(num_skipped),
            "num_rendered": int(num_local + num_remote - num_skipped),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
