from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
import torch

from mast3r_slam.functional_graph.policy import FunctionalGraphPolicy

from .debug_baseline_types import BaselineFrameDump, BaselineMeta


BASELINE_DUMP_VERSION = "pose_frozen_debug_v1"
FUNGRAPH_SCENE_ID_HINTS = {
    "0kitchen": "420683",
}


def _to_numpy(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_list_matrix(value) -> Optional[list[list[float]]]:
    if value is None:
        return None
    arr = _to_numpy(value)
    return arr.astype(np.float32).tolist()


def _optional_array(value, *, empty_shape: tuple[int, ...]) -> tuple[np.ndarray, bool]:
    arr = _to_numpy(value)
    if arr is None:
        return np.zeros(empty_shape, dtype=np.float32), True
    return arr.astype(np.float32), False


def _pose_to_matrix(pose) -> list[list[float]]:
    if pose is None:
        return np.eye(4, dtype=np.float32).tolist()
    if torch.is_tensor(pose):
        return _to_numpy(pose.reshape(4, 4)).astype(np.float32).tolist()
    if hasattr(pose, "matrix"):
        mat = pose.matrix()
        if torch.is_tensor(mat) and mat.ndim == 3:
            mat = mat[0]
        return _to_numpy(mat).reshape(4, 4).astype(np.float32).tolist()
    if hasattr(pose, "translation"):
        translation = _to_numpy(getattr(pose, "translation")).reshape(-1)
        if translation.size == 3:
            mat = np.eye(4, dtype=np.float32)
            mat[:3, 3] = translation.astype(np.float32)
            return mat.tolist()
    return _to_list_matrix(pose)


def _pack_masks(masks) -> tuple[np.ndarray, np.ndarray]:
    masks_np = _to_numpy(masks)
    if masks_np is None:
        return np.zeros((0, 0), dtype=np.uint8), np.asarray([0, 0, 0], dtype=np.int64)
    if masks_np.size == 0:
        return np.zeros((0, 0), dtype=np.uint8), np.asarray(masks_np.shape, dtype=np.int64)
    masks_u8 = masks_np.astype(np.uint8).reshape(masks_np.shape[0], -1)
    packed = np.packbits(masks_u8, axis=1)
    return packed, np.asarray(masks_np.shape, dtype=np.int64)


def _unpack_masks(packed: np.ndarray, shape: np.ndarray) -> np.ndarray:
    shape_t = tuple(int(v) for v in shape.reshape(-1).tolist())
    if len(shape_t) not in (3, 4) or shape_t[0] == 0:
        if len(shape_t) in (3, 4):
            return np.zeros(shape_t, dtype=np.float32)
        return np.zeros((0, 0, 0), dtype=np.float32)
    flat = np.unpackbits(packed, axis=1)[:, : int(np.prod(shape_t[1:]))]
    return flat.reshape(shape_t).astype(np.float32)


def serialize_sam3_det_payload(det) -> tuple[dict, dict[str, np.ndarray]]:
    if det is None:
        return {"labels": []}, {"sam3_boxes": np.zeros((0, 4), dtype=np.float32), "sam3_scores": np.zeros((0,), dtype=np.float32)}
    payload = {
        "labels": [str(v) for v in (getattr(det, "labels", None) or [])],
        "label_to_rank": {str(k): int(v) for k, v in (getattr(det, "label_to_rank", None) or {}).items()},
        "u_to_allowed_parents": {str(k): sorted(str(x) for x in v) for k, v in (getattr(det, "u_to_allowed_parents", None) or {}).items()},
        "c_to_allowed_parents": {str(k): sorted(str(x) for x in v) for k, v in (getattr(det, "c_to_allowed_parents", None) or {}).items()},
        "local_rel_debug": getattr(det, "local_rel_debug", None) or {},
        "cabinet_hints": getattr(det, "cabinet_hints", None) or {"carrier_hits": []},
    }
    boxes = _to_numpy(getattr(det, "boxes", None))
    scores = _to_numpy(getattr(det, "scores", None))
    masks_packed, masks_shape = _pack_masks(getattr(det, "masks", None))
    arrays = {
        "sam3_boxes": np.zeros((0, 4), dtype=np.float32) if boxes is None else boxes.astype(np.float32),
        "sam3_scores": np.zeros((0,), dtype=np.float32) if scores is None else scores.astype(np.float32),
        "sam3_masks_packed": masks_packed,
        "sam3_masks_shape": masks_shape,
    }
    return payload, arrays


def deserialize_sam3_det_payload(payload: dict, arrays: dict[str, np.ndarray]) -> dict:
    restored = dict(payload or {})
    restored["boxes"] = arrays.get("sam3_boxes", np.zeros((0, 4), dtype=np.float32)).astype(np.float32)
    restored["scores"] = arrays.get("sam3_scores", np.zeros((0,), dtype=np.float32)).astype(np.float32)
    restored["masks"] = _unpack_masks(
        arrays.get("sam3_masks_packed", np.zeros((0, 0), dtype=np.uint8)),
        arrays.get("sam3_masks_shape", np.asarray([0, 0, 0], dtype=np.int64)),
    )
    return restored


def save_baseline_frame_dump(
    baseline_dir: str | Path,
    *,
    frame_idx: int,
    frame,
    frame_result: dict,
    sam3_out,
    is_keyframe: bool,
    keyframe_slot: Optional[int],
    frame_assoc_debug: Optional[dict] = None,
    frame_extract_debug: Optional[list[dict]] = None,
    vis2d_image_path: Optional[str] = None,
    rgb_image_path: Optional[str] = None,
    debug_image_path: Optional[str] = None,
) -> BaselineFrameDump:
    base = Path(baseline_dir)
    frames_dir = base / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frames_dir / f"frame_{frame_idx:06d}.json"
    array_path = frames_dir / f"frame_{frame_idx:06d}.npz"

    det_payload, det_arrays = serialize_sam3_det_payload(getattr(sam3_out, "det", None))
    x_canon, x_canon_missing = _optional_array(getattr(frame, "X_canon", None), empty_shape=(0, 3))
    c_map, c_missing = _optional_array(getattr(frame, "C", None), empty_shape=(0, 1))
    arrays = {
        "X_canon": x_canon,
        "C": c_map,
        **det_arrays,
    }
    np.savez_compressed(array_path, **arrays)

    dump = BaselineFrameDump(
        frame_idx=frame_idx,
        is_keyframe=is_keyframe,
        keyframe_slot=keyframe_slot,
        T_WC=_pose_to_matrix(getattr(frame, "T_WC", None)),
        K=_to_list_matrix(getattr(frame, "K", None)),
        img_shape=[int(v) for v in getattr(frame, "img_shape").reshape(-1).tolist()],
        X_canon=arrays["X_canon"],
        C=arrays["C"],
        frame_result=frame_result or {},
        sam3_det_payload=deserialize_sam3_det_payload(det_payload, arrays),
        remote_rel_2d=getattr(sam3_out, "remote_rel_2d", None) or {},
        frame_assoc_debug=frame_assoc_debug,
        frame_extract_debug=list(frame_extract_debug or []),
        vis2d_image_path=None if vis2d_image_path is None else str(vis2d_image_path),
        rgb_image_path=None if rgb_image_path is None else str(rgb_image_path),
        debug_image_path=None if debug_image_path is None else str(debug_image_path),
        x_canon_missing=x_canon_missing,
        c_missing=c_missing,
    )

    with frame_path.open("w", encoding="utf-8") as f:
        json.dump(dump.to_json_dict(), f, ensure_ascii=False, indent=2)
    return dump


def load_baseline_frame_dump(baseline_dir: str | Path, frame_idx: int) -> BaselineFrameDump:
    base = Path(baseline_dir)
    frame_path = base / "frames" / f"frame_{frame_idx:06d}.json"
    array_path = base / "frames" / f"frame_{frame_idx:06d}.npz"
    with frame_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    arrays_npz = np.load(array_path)
    arrays = {key: arrays_npz[key] for key in arrays_npz.files}
    sam3_payload = deserialize_sam3_det_payload(payload.get("sam3_det_payload", {}), arrays)
    x_canon_missing = bool(payload.get("x_canon_missing", False))
    c_missing = bool(payload.get("c_missing", False))
    return BaselineFrameDump(
        frame_idx=int(payload["frame_idx"]),
        is_keyframe=bool(payload["is_keyframe"]),
        keyframe_slot=payload.get("keyframe_slot"),
        T_WC=payload["T_WC"],
        K=payload.get("K"),
        img_shape=[int(v) for v in payload["img_shape"]],
        X_canon=None if x_canon_missing else arrays.get("X_canon"),
        C=None if c_missing else arrays.get("C"),
        frame_result=payload.get("frame_result", {}),
        sam3_det_payload=sam3_payload,
        remote_rel_2d=payload.get("remote_rel_2d", {}),
        frame_assoc_debug=payload.get("frame_assoc_debug"),
        frame_extract_debug=payload.get("frame_extract_debug", []),
        vis2d_image_path=payload.get("vis2d_image_path"),
        rgb_image_path=payload.get("rgb_image_path"),
        debug_image_path=payload.get("debug_image_path"),
        x_canon_missing=x_canon_missing,
        c_missing=c_missing,
    )


def iter_baseline_frame_dumps(
    baseline_dir: str | Path,
    *,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Iterable[BaselineFrameDump]:
    meta = load_baseline_meta(baseline_dir)
    start = 0 if start_frame is None else int(start_frame)
    end = (meta.num_frames - 1) if end_frame is None else int(end_frame)
    for frame_idx in range(start, end + 1):
        yield load_baseline_frame_dump(baseline_dir, frame_idx)


def save_baseline_meta(baseline_dir: str | Path, meta: BaselineMeta) -> None:
    base = Path(baseline_dir)
    base.mkdir(parents=True, exist_ok=True)
    with (base / "baseline_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta.to_json_dict(), f, ensure_ascii=False, indent=2)


def load_baseline_meta(baseline_dir: str | Path) -> BaselineMeta:
    with (Path(baseline_dir) / "baseline_meta.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return BaselineMeta(
        version=payload["version"],
        sequence_name=payload["sequence_name"],
        dataset_path=payload["dataset_path"],
        num_frames=int(payload["num_frames"]),
        frame_idx_to_gt_idx={int(k): int(v) for k, v in payload.get("frame_idx_to_gt_idx", {}).items()},
        keyframe_indices=[int(v) for v in payload.get("keyframe_indices", [])],
        frame_timestamps=[float(v) for v in payload.get("frame_timestamps", [])],
        frame_dump_path_template=payload["frame_dump_path_template"],
        graph_policy=payload.get("graph_policy", {}),
        K=payload.get("K"),
        img_shape=payload.get("img_shape"),
        A_base=payload.get("A_base"),
        gt_file=payload.get("gt_file"),
        gt_cache_path_gt_world=payload.get("gt_cache_path_gt_world"),
        gt_cache_path_baseline_world=payload.get("gt_cache_path_baseline_world"),
    )


def replay_graph_policy_from_meta(meta: BaselineMeta) -> FunctionalGraphPolicy:
    policy_payload = meta.graph_policy or {}
    legacy_aggregate_objects = set(policy_payload.get("aggregate_output_objects", []))
    enable_cabinet_aggregation = bool(
        policy_payload.get("enable_cabinet_aggregation", bool(legacy_aggregate_objects))
    )
    return FunctionalGraphPolicy(
        suppress_objects=set(policy_payload.get("suppress_objects", [])),
        hint_only_objects=set(policy_payload.get("hint_only_objects", [])),
        enable_cabinet_aggregation=enable_cabinet_aggregation,
        aggregate_output_objects=legacy_aggregate_objects,
        cabinet_seed_carriers=set(policy_payload.get("cabinet_seed_carriers", [])),
        cabinet_unit_labels=set(policy_payload.get("cabinet_unit_labels", [])),
    )


def graph_policy_to_dict(policy: FunctionalGraphPolicy) -> dict[str, object]:
    return {
        "suppress_objects": sorted(policy.suppress_objects),
        "hint_only_objects": sorted(policy.hint_only_objects),
        "enable_cabinet_aggregation": bool(policy.should_enable_cabinet_aggregation()),
        "aggregate_output_objects": sorted(policy.aggregate_output_objects),
        "cabinet_seed_carriers": sorted(policy.cabinet_seed_carriers),
        "cabinet_unit_labels": sorted(policy.cabinet_unit_labels),
    }


def pose_matrix_from_tum_components(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    quat = quat / max(1e-12, np.linalg.norm(quat))
    qw, qx, qy, qz = quat.tolist()
    rot = np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = np.asarray([tx, ty, tz], dtype=np.float64)
    return pose


def load_tum_trajectory(path: str | Path) -> tuple[np.ndarray, list[np.ndarray]]:
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) < 8:
                continue
            t, tx, ty, tz, qx, qy, qz, qw = [float(v) for v in vals[:8]]
            timestamps.append(t)
            poses.append(pose_matrix_from_tum_components(tx, ty, tz, qx, qy, qz, qw))
    return np.asarray(timestamps, dtype=np.float64), poses


def match_frame_timestamps_to_gt(frame_timestamps: list[float], gt_timestamps: np.ndarray, max_delta: float = 0.05) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if gt_timestamps.size == 0:
        return mapping
    for frame_idx, timestamp in enumerate(frame_timestamps):
        deltas = np.abs(gt_timestamps - float(timestamp))
        gt_idx = int(np.argmin(deltas))
        if float(deltas[gt_idx]) <= float(max_delta):
            mapping[frame_idx] = gt_idx
    return mapping


def _umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> dict:
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need at least three matched 3D points for alignment")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / float(src.shape[0])
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1.0
    rot = u @ s @ vt
    scale = 1.0
    if with_scale:
        var = np.sum(src_c * src_c) / float(src.shape[0])
        scale = float(np.trace(np.diag(d) @ s) / max(var, 1e-12))
    trans = dst_mean - scale * (rot @ src_mean)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = scale * rot
    mat[:3, 3] = trans
    aligned = apply_similarity_transform(src, {"matrix": mat.tolist(), "scale": scale, "kind": "sim3"})
    rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
    return {"kind": "sim3" if with_scale else "se3", "scale": scale, "matrix": mat.tolist(), "rmse": rmse}


def estimate_pred_to_gt_alignment(pred_pose_mats: list[np.ndarray], gt_pose_mats: list[np.ndarray], allow_scale: bool = True) -> Optional[dict]:
    if len(pred_pose_mats) != len(gt_pose_mats) or len(pred_pose_mats) < 3:
        return None
    pred_centers = np.asarray([pose[:3, 3] for pose in pred_pose_mats], dtype=np.float64)
    gt_centers = np.asarray([pose[:3, 3] for pose in gt_pose_mats], dtype=np.float64)
    return _umeyama_alignment(pred_centers, gt_centers, with_scale=allow_scale)


def apply_similarity_transform(points: np.ndarray, transform: dict) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(-1, 3)
    mat = np.asarray(transform["matrix"], dtype=np.float64).reshape(4, 4)
    homo = np.concatenate([pts.reshape(-1, 3), np.ones((pts.reshape(-1, 3).shape[0], 1), dtype=np.float64)], axis=1)
    return (homo @ mat.T)[:, :3]


def invert_similarity_transform(transform: dict) -> dict:
    mat = np.asarray(transform["matrix"], dtype=np.float64).reshape(4, 4)
    inv = np.linalg.inv(mat)
    scale = float(transform.get("scale", 1.0))
    inv_scale = 1.0 / scale if abs(scale) > 1e-12 else 1.0
    return {
        "kind": transform.get("kind", "sim3"),
        "scale": inv_scale,
        "matrix": inv.tolist(),
        "rmse": float(transform.get("rmse", 0.0)),
    }


def infer_fungraph_scene_id(sequence_name: str, fungraph_root: str | Path, scenefun3d_root: str | Path) -> Optional[str]:
    sequence = Path(str(sequence_name))
    scene_name = sequence.parts[0] if len(sequence.parts) > 0 else str(sequence_name)
    if scene_name in FUNGRAPH_SCENE_ID_HINTS:
        return FUNGRAPH_SCENE_ID_HINTS[scene_name]
    scene_ply = Path(fungraph_root) / scene_name / f"{scene_name}.ply"
    if not scene_ply.exists():
        return None
    scene_size = scene_ply.stat().st_size
    candidates = list(Path(scenefun3d_root).rglob("*_laser_scan.ply"))
    size_matches = [path for path in candidates if path.stat().st_size == scene_size]
    if not size_matches:
        return None
    if len(size_matches) == 1:
        return size_matches[0].parent.name
    src_digest = hashlib.sha256(scene_ply.read_bytes()[: 1024 * 1024]).hexdigest()
    for candidate in size_matches:
        cand_digest = hashlib.sha256(candidate.read_bytes()[: 1024 * 1024]).hexdigest()
        if cand_digest == src_digest:
            return candidate.parent.name
    return size_matches[0].parent.name


class BaselineDebugDumper:
    def __init__(self, baseline_dir: str | Path, graph_policy: FunctionalGraphPolicy) -> None:
        self.baseline_dir = Path(baseline_dir)
        self.graph_policy = graph_policy
        self.pred_pose_mats: list[np.ndarray] = []
        self.frame_timestamps: list[float] = []
        self.keyframe_indices: list[int] = []
        self.common_K: Optional[list[list[float]]] = None
        self.common_img_shape: Optional[list[int]] = None

    def record_frame(
        self,
        *,
        frame_idx: int,
        frame,
        timestamp: float,
        frame_result: dict,
        sam3_out,
        is_keyframe: bool,
        keyframe_slot: Optional[int],
        frame_assoc_debug: Optional[dict] = None,
        frame_extract_debug: Optional[list[dict]] = None,
        vis2d_image_path: Optional[str] = None,
        rgb_image_path: Optional[str] = None,
        debug_image_path: Optional[str] = None,
    ) -> BaselineFrameDump:
        dump = save_baseline_frame_dump(
            self.baseline_dir,
            frame_idx=frame_idx,
            frame=frame,
            frame_result=frame_result,
            sam3_out=sam3_out,
            is_keyframe=is_keyframe,
            keyframe_slot=keyframe_slot,
            frame_assoc_debug=frame_assoc_debug,
            frame_extract_debug=frame_extract_debug,
            vis2d_image_path=vis2d_image_path,
            rgb_image_path=rgb_image_path,
            debug_image_path=debug_image_path,
        )
        self.pred_pose_mats.append(np.asarray(dump.T_WC, dtype=np.float64))
        self.frame_timestamps.append(float(timestamp))
        if dump.is_keyframe:
            self.keyframe_indices.append(int(frame_idx))
        if self.common_K is None:
            self.common_K = dump.K
        if self.common_img_shape is None:
            self.common_img_shape = list(dump.img_shape)
        return dump

    def finalize(
        self,
        *,
        sequence_name: str,
        dataset_path: str,
        gt_file: Optional[str] = None,
        gt_cache_path_gt_world: Optional[str] = None,
        gt_cache_path_baseline_world: Optional[str] = None,
    ) -> BaselineMeta:
        frame_idx_to_gt_idx: dict[int, int] = {}
        A_base = None
        if gt_file:
            gt_timestamps, gt_poses = load_tum_trajectory(gt_file)
            frame_idx_to_gt_idx = match_frame_timestamps_to_gt(self.frame_timestamps, gt_timestamps)
            matched_pred = [self.pred_pose_mats[frame_idx] for frame_idx in sorted(frame_idx_to_gt_idx.keys())]
            matched_gt = [gt_poses[frame_idx_to_gt_idx[frame_idx]] for frame_idx in sorted(frame_idx_to_gt_idx.keys())]
            if len(matched_pred) >= 3:
                A_base = estimate_pred_to_gt_alignment(matched_pred, matched_gt, allow_scale=True)

        meta = BaselineMeta(
            version=BASELINE_DUMP_VERSION,
            sequence_name=sequence_name,
            dataset_path=dataset_path,
            num_frames=len(self.frame_timestamps),
            frame_idx_to_gt_idx=frame_idx_to_gt_idx,
            keyframe_indices=list(self.keyframe_indices),
            frame_timestamps=list(self.frame_timestamps),
            frame_dump_path_template="frames/frame_{frame_idx:06d}.json",
            graph_policy=graph_policy_to_dict(self.graph_policy),
            K=self.common_K,
            img_shape=self.common_img_shape,
            A_base=A_base,
            gt_file=gt_file,
            gt_cache_path_gt_world=gt_cache_path_gt_world,
            gt_cache_path_baseline_world=gt_cache_path_baseline_world,
        )
        save_baseline_meta(self.baseline_dir, meta)
        return meta
