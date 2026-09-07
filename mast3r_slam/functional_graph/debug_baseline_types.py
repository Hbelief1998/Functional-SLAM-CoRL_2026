from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def normalize_debug_label(label: str | None) -> str:
    text = str(label or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def label_family(label: str | None) -> str:
    text = normalize_debug_label(label)
    if not text:
        return ""
    families = (
        "drawer",
        "door",
        "handle",
        "knob",
        "button",
        "switch",
        "panel",
        "cabinet",
        "shelf",
        "window",
        "stove",
    )
    for family in families:
        if family in text:
            if family == "switch":
                return "button"
            if family == "panel" and "switch" in text:
                return "button"
            return family
    return text


def infer_role_from_label(label: str | None) -> str:
    family = label_family(label)
    if family in {"handle", "knob", "button"}:
        return "U"
    if family in {"drawer", "door", "shelf", "panel"}:
        return "C"
    return "O"


@dataclass
class BaselineFrameDump:
    frame_idx: int
    is_keyframe: bool
    keyframe_slot: Optional[int]
    T_WC: list[list[float]]
    K: Optional[list[list[float]]]
    img_shape: list[int]
    X_canon: Any
    C: Any
    frame_result: dict
    sam3_det_payload: dict
    remote_rel_2d: dict = field(default_factory=dict)
    frame_assoc_debug: Optional[dict] = None
    frame_extract_debug: list[dict] = field(default_factory=list)
    vis2d_image_path: Optional[str] = None
    rgb_image_path: Optional[str] = None
    debug_image_path: Optional[str] = None
    x_canon_missing: bool = False
    c_missing: bool = False

    def to_json_dict(self) -> dict:
        return {
            "frame_idx": int(self.frame_idx),
            "is_keyframe": bool(self.is_keyframe),
            "keyframe_slot": None if self.keyframe_slot is None else int(self.keyframe_slot),
            "T_WC": self.T_WC,
            "K": self.K,
            "img_shape": [int(v) for v in self.img_shape],
            "frame_result": self.frame_result,
            "sam3_det_payload": {
                key: value
                for key, value in self.sam3_det_payload.items()
                if key not in {"masks", "boxes", "scores"}
            },
            "remote_rel_2d": self.remote_rel_2d,
            "frame_assoc_debug": self.frame_assoc_debug,
            "frame_extract_debug": self.frame_extract_debug,
            "vis2d_image_path": self.vis2d_image_path,
            "rgb_image_path": self.rgb_image_path,
            "debug_image_path": self.debug_image_path,
            "x_canon_missing": bool(self.x_canon_missing),
            "c_missing": bool(self.c_missing),
        }


@dataclass
class BaselineMeta:
    version: str
    sequence_name: str
    dataset_path: str
    num_frames: int
    frame_idx_to_gt_idx: dict[int, int]
    keyframe_indices: list[int]
    frame_timestamps: list[float]
    frame_dump_path_template: str
    graph_policy: dict[str, Any] = field(default_factory=dict)
    K: Optional[list[list[float]]] = None
    img_shape: Optional[list[int]] = None
    A_base: Optional[dict] = None
    gt_file: Optional[str] = None
    gt_cache_path_gt_world: Optional[str] = None
    gt_cache_path_baseline_world: Optional[str] = None

    def to_json_dict(self) -> dict:
        return {
            "version": self.version,
            "sequence_name": self.sequence_name,
            "dataset_path": self.dataset_path,
            "num_frames": int(self.num_frames),
            "frame_idx_to_gt_idx": {str(k): int(v) for k, v in self.frame_idx_to_gt_idx.items()},
            "keyframe_indices": [int(v) for v in self.keyframe_indices],
            "frame_timestamps": [float(v) for v in self.frame_timestamps],
            "frame_dump_path_template": self.frame_dump_path_template,
            "graph_policy": self.graph_policy,
            "K": self.K,
            "img_shape": self.img_shape,
            "A_base": self.A_base,
            "gt_file": self.gt_file,
            "gt_cache_path_gt_world": self.gt_cache_path_gt_world,
            "gt_cache_path_baseline_world": self.gt_cache_path_baseline_world,
        }


@dataclass
class GTNode3D:
    gt_node_id: str
    role: str
    label: str
    label_family: str
    points_world_gt: list[list[float]]
    parent_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "gt_node_id": self.gt_node_id,
            "role": self.role,
            "label": self.label,
            "label_family": self.label_family,
            "points_world_gt": self.points_world_gt,
            "parent_ids": self.parent_ids,
            "metadata": self.metadata,
        }


@dataclass
class GTVisibleNode:
    frame_idx: int
    gt_node_id: str
    role: str
    label: str
    label_family: str
    num_visible_points: int
    visible_ratio: float
    box_xyxy_gt: list[float]
    centroid_world_gt_visible: list[float]
    bbox_world_gt_visible: dict[str, list[float]]
    depth_minmax: list[float]
    points_world_gt_visible_sample: list[list[float]] = field(default_factory=list)
    centroid_world_base: Optional[list[float]] = None
    bbox_world_base: Optional[dict[str, list[float]]] = None
    points_world_base_sample: list[list[float]] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        return {
            "frame_idx": int(self.frame_idx),
            "gt_node_id": self.gt_node_id,
            "role": self.role,
            "label": self.label,
            "label_family": self.label_family,
            "num_visible_points": int(self.num_visible_points),
            "visible_ratio": float(self.visible_ratio),
            "box_xyxy_gt": [float(v) for v in self.box_xyxy_gt],
            "centroid_world_gt_visible": [float(v) for v in self.centroid_world_gt_visible],
            "bbox_world_gt_visible": {
                "min": [float(v) for v in self.bbox_world_gt_visible.get("min", [])],
                "max": [float(v) for v in self.bbox_world_gt_visible.get("max", [])],
            },
            "depth_minmax": [float(v) for v in self.depth_minmax],
            "points_world_gt_visible_sample": [
                [float(c) for c in point] for point in self.points_world_gt_visible_sample
            ],
            "centroid_world_base": None if self.centroid_world_base is None else [float(v) for v in self.centroid_world_base],
            "bbox_world_base": None
            if self.bbox_world_base is None
            else {
                "min": [float(v) for v in self.bbox_world_base.get("min", [])],
                "max": [float(v) for v in self.bbox_world_base.get("max", [])],
            },
            "points_world_base_sample": [[float(c) for c in point] for point in self.points_world_base_sample],
        }


@dataclass
class ObservationGTBinding:
    frame_idx: int
    det_idx: int
    pred_role: str
    pred_label: str
    pred_node_id: Optional[str]
    gt_node_id: Optional[str]
    gt_role: Optional[str]
    gt_label: Optional[str]
    match_score: float
    score_terms: dict[str, float] = field(default_factory=dict)
    bind_status: str = "unbound"
    binding_mode: Optional[str] = None

    def to_json_dict(self) -> dict:
        return {
            "frame_idx": int(self.frame_idx),
            "det_idx": int(self.det_idx),
            "pred_role": self.pred_role,
            "pred_label": self.pred_label,
            "pred_node_id": self.pred_node_id,
            "gt_node_id": self.gt_node_id,
            "gt_role": self.gt_role,
            "gt_label": self.gt_label,
            "match_score": float(self.match_score),
            "score_terms": {str(k): float(v) for k, v in self.score_terms.items()},
            "bind_status": self.bind_status,
            "binding_mode": self.binding_mode,
        }
