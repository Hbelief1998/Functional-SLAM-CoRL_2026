from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


ROLE_RANK = {"U": 0, "C": 1, "O": 2}
RANK_ROLE = {value: key for key, value in ROLE_RANK.items()}


def normalize_label(label: str | None) -> str:
    text = str(label or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def role_from_rank(rank: int | None) -> str:
    return RANK_ROLE.get(int(rank) if rank is not None else ROLE_RANK["O"], "O")


@dataclass
class NodeObservation:
    det_idx: int
    label: str
    raw_label: str
    role: str
    score: float
    mask_area: int
    box_xyxy: Optional[List[float]]
    box_touches_border: bool
    num_points: int
    dbscan_filtered: bool
    points_frame: torch.Tensor
    centroid_frame: torch.Tensor
    bbox3d_frame: Tuple[torch.Tensor, torch.Tensor]
    points_world: torch.Tensor
    centroid_world: torch.Tensor
    bbox3d_world: Tuple[torch.Tensor, torch.Tensor]
    bbox_diag_world: float
    allowed_parent_labels: set[str] = field(default_factory=set)
    preferred_parent_labels: set[str] = field(default_factory=set)
    fallback_parent_labels: set[str] = field(default_factory=set)
    preferred_parent_role: Optional[str] = None
    is_direct_unit: bool = False
    semantic_owner_mode: str = "prefer_object"
    cabinet_box_marked: bool = False
    cabinet_box_ids: list[int] = field(default_factory=list)
    cabinet_box_scores: list[float] = field(default_factory=list)
    local_parent_det_idxs: list[int] = field(default_factory=list)
    strong_parent_det_idxs: list[int] = field(default_factory=list)
    semantic_subtype: Optional[str] = None
    semantic_subtype_conf: float = 0.0
    semantic_parent_label: Optional[str] = None
    semantic_parent_role: Optional[str] = None
    semantic_owner_object_label: Optional[str] = None
    view_score: float = 0.0
    matched_node_id: Optional[str] = None


@dataclass
class CabinetCarrierObservation:
    node_id: str
    label: str
    det_idx: int
    frame_idx: int
    box_xyxy: Optional[List[float]]
    view_score: float
    cabinet_box_marked: bool = False
    cabinet_box_scores: list[float] = field(default_factory=list)
    cabinet_box_ids: list[int] = field(default_factory=list)
    box_touches_border: bool = False
    centroid_world: Optional[torch.Tensor] = None
    bbox3d_world: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


@dataclass
class RemoteObservation:
    relation_key: str
    relation_text: str
    src_label: str
    dst_label: str
    pair_scores: Dict[Tuple[str, str], float]
    frame_idx: int
    best_view_frame: Optional[int] = None
