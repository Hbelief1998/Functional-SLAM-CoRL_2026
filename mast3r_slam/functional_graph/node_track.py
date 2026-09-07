from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch

from .geometry_utils import bbox_diag, bbox_from_points, bbox_size, frame_hw, frame_intrinsics, project_box_xyxy, tensor_to_list


@dataclass
class OnlineNodeTrack:
    node_id: str
    label: str
    role: str
    origin: str = "standard"
    anchor_kf_id: Optional[int] = None
    last_seen_frame: Optional[int] = None
    last_seen_kf: Optional[int] = None
    obs_count: int = 0
    visible_count: int = 0
    centroid_world_est: Optional[list[float]] = None
    bbox_world_est: Optional[Dict[str, list[float]]] = None
    bbox_diag_world_est: float = 0.0
    best_view_frame: Optional[int] = None
    best_view_score: float = 0.0
    best_view_crop_meta: Optional[Dict[str, Any]] = None
    last_box_xyxy: Optional[list[float]] = None
    last_obs_pose: Any = None
    last_obs_K: Optional[list[list[float]]] = None
    last_obs_hw: Optional[list[int]] = None
    last_obs_depth: Optional[float] = None
    stable_geom_ready: bool = False
    # Primary anchor snapshot kept for backward compatibility. Stable tracks should
    # consume fused canonical local geometry first and treat these as mirrors.
    points_anchor: Optional[list[list[float]]] = None
    centroid_anchor: Optional[list[float]] = None
    bbox_anchor: Optional[Dict[str, list[float]]] = None
    # Primary anchor frame defines the canonical local frame. Fused local geometry
    # is the source of truth for stable tracks in that frame.
    points_fused_local: Optional[list[list[float]]] = None
    centroid_fused_local: Optional[list[float]] = None
    bbox_fused_local: Optional[Dict[str, list[float]]] = None
    fused_support_kfs: list[int] = field(default_factory=list)
    fused_num_points: int = 0
    fused_geom_ready: bool = False
    candidate_support_kfs: list[int] = field(default_factory=list)
    candidate_anchor_kf_id: Optional[int] = None
    candidate_points_frame: Optional[list[list[float]]] = None
    candidate_centroid_frame: Optional[list[float]] = None
    candidate_bbox_frame: Optional[Dict[str, list[float]]] = None
    candidate_view_score: float = 0.0
    candidate_num_points: int = 0
    candidate_border_touched: bool = True
    candidate_box_touches_border: bool = True
    provisional_points_frame: Optional[list[list[float]]] = None
    provisional_centroid_frame: Optional[list[float]] = None
    provisional_bbox_frame: Optional[Dict[str, list[float]]] = None
    provisional_pose: Any = None
    provisional_view_score: float = 0.0
    provisional_num_points: int = 0
    provisional_border_touched: bool = True
    provisional_support_kfs: list[int] = field(default_factory=list)
    anchor_support_kfs: list[int] = field(default_factory=list)
    primary_anchor_view_score: float = 0.0
    primary_anchor_num_points: int = 0
    primary_anchor_border_touched: bool = True
    primary_anchor_created_frame: Optional[int] = None
    primary_anchor_support_kfs: list[int] = field(default_factory=list)
    fused_quality_score: float = 0.0
    last_fused_kf: Optional[int] = None
    disable_3d_fusion: bool = False
    num_fusion_accepts: int = 0
    num_fusion_rejects: int = 0
    last_fusion_reject_reason: Optional[str] = None
    last_pre_stable_reject_reason: Optional[str] = None
    # Promote bookkeeping
    num_promote_rejects: int = 0
    last_promote_reject_reason: Optional[str] = None
    promote_reject_reason_counts: Dict[str, int] = field(default_factory=dict)
    last_promote_kf: Optional[int] = None
    primary_anchor_created_kf: Optional[int] = None
    primary_anchor_source: Optional[str] = None  # "candidate" or "pre_stable"
    primary_anchor_promoted_from_pre_stable: bool = False
    # Reject reason counts for fusion and pre-stable
    fusion_reject_reason_counts: Dict[str, int] = field(default_factory=dict)
    pre_stable_reject_reason_counts: Dict[str, int] = field(default_factory=dict)
    last_accept_quality_score: float = 0.0
    last_pre_stable_accept_kf: Optional[int] = None
    # -- Auxiliary anchors: side cache for multi-view coverage ---------
    # These are NOT the source of truth.  The single primary anchor +
    # fused canonical local geometry remain authoritative.
    auxiliary_anchors: list[dict] = field(default_factory=list)
    aux_anchor_budget: int = 3
    aux_anchor_created_kfs: list[int] = field(default_factory=list)
    # Bookkeeping for auxiliary anchor updates (per-frame diagnostics)
    aux_anchor_last_attempt_kf: Optional[int] = None
    aux_anchor_last_attempt_result: Optional[str] = None  # "added" | "updated" | "skipped:<reason>"
    aux_anchor_update_reason_counts: Dict[str, int] = field(default_factory=dict)
    aux_anchor_total_attempts: int = 0
    aux_anchor_total_adds: int = 0
    aux_anchor_total_updates: int = 0
    aux_anchor_total_trims: int = 0
    # -- Conservative re-anchor interface (default ON, low-frequency) --
    # Default behaviour: enable_reanchor=True so long-range sequences can
    # use an occasional re-anchor as a *corrective* mechanism.  The gate
    # is deliberately conservative (cooldown + composite margin + no
    # border) so re-anchor should fire rarely and only when a clearly
    # better primary frame becomes available.
    # Global kill-switch: set FG_DISABLE_REANCHOR=1.
    # Per-track override: set track.enable_reanchor=False.
    enable_reanchor: bool = True
    num_reanchors: int = 0
    last_reanchor_kf: Optional[int] = None
    last_reanchor_reject_reason: Optional[str] = None
    reanchor_reject_reason_counts: Dict[str, int] = field(default_factory=dict)
    reanchor_min_kf_interval: int = 5  # minimum kf gap between re-anchors
    # -- Support-anchor assist bookkeeping (read-path, weak assist) ----
    # Counts how many times the online association main chain applied a
    # support-anchor bonus to this track's stage1 score.  Purely
    # observational: the bonus is always capped and never changes
    # source of truth.  See ``online_state._apply_multi_anchor_assoc_assist``.
    support_anchor_assist_applied_count: int = 0
    last_support_anchor_assist_kf: Optional[int] = None
    last_support_anchor_assist_bonus: float = 0.0
    last_support_anchor_assist_best_kf: Optional[int] = None
    last_support_anchor_assist_reason: Optional[str] = None
    support_anchor_assist_reason_counts: Dict[str, int] = field(default_factory=dict)
    label_counts: Dict[str, int] = field(default_factory=dict)
    display_label_ratio: float = 3.0
    semantic_subtype: Optional[str] = None
    semantic_subtype_scores: Dict[str, float] = field(default_factory=dict)
    semantic_subtype_ratio: float = 1.5
    plane_normal_world_est: Optional[list[float]] = None
    plane_normal_quality: float = 0.0
    plane_normal_source: Optional[str] = None
    plane_normal_kf_id: Optional[int] = None
    plane_normal_version: int = 0
    plane_normal_num_points: int = 0
    plane_normal_eigenvalues: Optional[list[float]] = None
    plane_normal_last_update_reason: Optional[str] = None
    cabinet_group_id: Optional[str] = None
    cabinet_seeded_count: int = 0
    cabinet_member_count: int = 0
    # -- Co-visibility bookkeeping (used by node consolidation /
    # sibling-duplicate-risk gate). Stores the most recent ``observed_frame_cap``
    # frame indices in which this track was associated to an observation. The
    # *count* of observations is still tracked by ``obs_count``; this list is a
    # strictly-bounded ring buffer used purely for cross-track co-visibility
    # tests. ``first_seen_frame`` is the earliest associated frame.
    observed_frames: list[int] = field(default_factory=list)
    observed_frame_cap: int = 256
    first_seen_frame: Optional[int] = None
    # ---- Lightweight observation-evidence cache ----------------------
    # Kept as bounded debug/inspection metadata. It is not used for
    # large-object duplicate merging.
    recent_observations: list[dict] = field(default_factory=list)
    best_observations: list[dict] = field(default_factory=list)
    recent_observations_cap: int = 16
    best_observations_cap: int = 5

    @staticmethod
    def _bump_reason_count(counter: Dict[str, int], reason: Optional[str]) -> None:
        if reason:
            counter[reason] = counter.get(reason, 0) + 1

    def _record_promote_reject(self, reason: str) -> None:
        self.num_promote_rejects += 1
        self.last_promote_reject_reason = reason
        self._bump_reason_count(self.promote_reject_reason_counts, reason)

    def _record_fusion_reject(self, reason: str) -> None:
        self.num_fusion_rejects += 1
        self.last_fusion_reject_reason = reason
        self._bump_reason_count(self.fusion_reject_reason_counts, reason)

    def _record_pre_stable_reject(self, reason: str) -> None:
        self.last_pre_stable_reject_reason = reason
        self._bump_reason_count(self.pre_stable_reject_reason_counts, reason)

    @staticmethod
    def _safe_tensor_to_list(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.is_cuda:
                return None
            try:
                return value.detach().cpu().tolist()
            except RuntimeError:
                return None
        try:
            return tensor_to_list(value)
        except Exception:
            return None

    def _as_tensor(self, value, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor

    # ------------------------------------------------------------------
    # Points tensor cache
    # ------------------------------------------------------------------
    # Several hot paths (`consider_provisional_anchor`, `fuse_anchor_observation`,
    # `_anchor_fusion_gate`, candidate/provisional/fused tensor accessors) repeatedly
    # rebuild tensors from ``list[list[float]]`` storage via ``torch.as_tensor``.
    # With ≤512 × 3 points that round-trip dominates the per-keyframe commit cost.
    # Keep a CPU float32 mirror per list-backed field and return a device-specific
    # view on demand. The list storage remains the JSON source of truth; the cache
    # is strictly an accelerator and is invalidated whenever the list length no
    # longer matches the cached tensor shape.
    def _points_cache_dict(self) -> Dict[str, torch.Tensor]:
        cache = getattr(self, "_points_tensor_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_points_tensor_cache", cache)
        return cache

    def _cache_points_tensor(self, attr_name: str, tensor: Optional[torch.Tensor]) -> None:
        cache = self._points_cache_dict()
        if tensor is None:
            cache.pop(attr_name, None)
            return
        if not torch.is_tensor(tensor):
            cache.pop(attr_name, None)
            return
        try:
            cpu_t = tensor.detach()
            if cpu_t.device.type != "cpu":
                cpu_t = cpu_t.cpu()
            if cpu_t.dtype != torch.float32:
                cpu_t = cpu_t.to(dtype=torch.float32)
            cache[attr_name] = cpu_t.contiguous()
        except Exception:
            cache.pop(attr_name, None)

    def _cached_points_tensor(
        self,
        attr_name: str,
        *,
        device=None,
        dtype=torch.float32,
    ) -> Optional[torch.Tensor]:
        cache = getattr(self, "_points_tensor_cache", None)
        if not cache:
            return None
        cached = cache.get(attr_name)
        if cached is None:
            return None
        raw = getattr(self, attr_name, None)
        if raw is None:
            cache.pop(attr_name, None)
            return None
        try:
            if len(raw) != int(cached.shape[0]):
                cache.pop(attr_name, None)
                return None
        except Exception:
            cache.pop(attr_name, None)
            return None
        out = cached
        if device is not None and out.device != device:
            out = out.to(device=device, dtype=dtype)
        elif out.dtype != dtype:
            out = out.to(dtype=dtype)
        return out

    def anchor_points_tensor(self, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        cached = self._cached_points_tensor("points_anchor", device=device, dtype=dtype)
        if cached is not None:
            return cached
        return self._as_tensor(self.points_anchor, device=device, dtype=dtype)

    def anchor_centroid_tensor(self, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        return self._as_tensor(self.centroid_anchor, device=device, dtype=dtype)

    def anchor_bbox_tensors(self, *, device=None, dtype=torch.float32) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._bbox_tensors(self.bbox_anchor, device=device, dtype=dtype)

    def fused_points_tensor(self, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        cached = self._cached_points_tensor("points_fused_local", device=device, dtype=dtype)
        if cached is not None:
            return cached
        return self._as_tensor(self.points_fused_local, device=device, dtype=dtype)

    def fused_centroid_tensor(self, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        return self._as_tensor(self.centroid_fused_local, device=device, dtype=dtype)

    def fused_bbox_tensors(self, *, device=None, dtype=torch.float32) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._bbox_tensors(self.bbox_fused_local, device=device, dtype=dtype)

    def _bbox_tensors(self, bbox_value, *, device=None, dtype=torch.float32) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if bbox_value is None:
            return None, None
        bmin = self._as_tensor(bbox_value.get("min"), device=device, dtype=dtype)
        bmax = self._as_tensor(bbox_value.get("max"), device=device, dtype=dtype)
        if bmin is None or bmax is None or bmin.numel() != 3 or bmax.numel() != 3:
            return None, None
        return bmin, bmax

    def candidate_points_tensor(self, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        cached = self._cached_points_tensor("candidate_points_frame", device=device, dtype=dtype)
        if cached is not None:
            return cached
        return self._as_tensor(self.candidate_points_frame, device=device, dtype=dtype)

    def candidate_centroid_tensor(self, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        return self._as_tensor(self.candidate_centroid_frame, device=device, dtype=dtype)

    def candidate_bbox_tensors(self, *, device=None, dtype=torch.float32) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.candidate_bbox_frame is None:
            return None, None
        bmin = self._as_tensor(self.candidate_bbox_frame.get("min"), device=device, dtype=dtype)
        bmax = self._as_tensor(self.candidate_bbox_frame.get("max"), device=device, dtype=dtype)
        if bmin is None or bmax is None or bmin.numel() != 3 or bmax.numel() != 3:
            return None, None
        return bmin, bmax

    def provisional_points_tensor(self, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        cached = self._cached_points_tensor("provisional_points_frame", device=device, dtype=dtype)
        if cached is not None:
            return cached
        return self._as_tensor(self.provisional_points_frame, device=device, dtype=dtype)

    def provisional_centroid_tensor(self, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        return self._as_tensor(self.provisional_centroid_frame, device=device, dtype=dtype)

    def provisional_bbox_tensors(self, *, device=None, dtype=torch.float32) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.provisional_bbox_frame is None:
            return None, None
        bmin = self._as_tensor(self.provisional_bbox_frame.get("min"), device=device, dtype=dtype)
        bmax = self._as_tensor(self.provisional_bbox_frame.get("max"), device=device, dtype=dtype)
        if bmin is None or bmax is None or bmin.numel() != 3 or bmax.numel() != 3:
            return None, None
        return bmin, bmax

    def world_centroid_tensor(self, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        return self._as_tensor(self.centroid_world_est, device=device, dtype=dtype)

    def update_label_hist(self, label: str) -> None:
        if not label:
            return
        self.label_counts[label] = int(self.label_counts.get(label, 0)) + 1

    def dominant_label(self) -> str:
        if not self.label_counts:
            return self.label
        ranked = sorted(self.label_counts.items(), key=lambda item: item[1], reverse=True)
        top1_label, top1_count = ranked[0]
        top2_count = ranked[1][1] if len(ranked) > 1 else 0
        if float(top1_count) >= float(self.display_label_ratio) * float(max(1, top2_count)):
            return top1_label
        return self.label

    def label_soft_score(self, label: str) -> float:
        if not label:
            return 0.1
        dominant = self.dominant_label()
        # Prefer the dominant label so the main association path and the hard
        # label gate reason about the same canonical track label.  Keep weaker
        # support for the legacy display label / histogram entries to avoid
        # throwing away useful history when the histogram is still mixed.
        if label == dominant:
            return 1.0
        if label == self.label:
            return 0.85
        if label in self.label_counts:
            return 0.7
        return 0.1

    def update_semantic_subtype_hist(self, subtype: Optional[str], conf: float) -> None:
        if not subtype:
            return
        score = float(conf)
        if score <= 0.0:
            return
        self.semantic_subtype_scores[subtype] = float(self.semantic_subtype_scores.get(subtype, 0.0)) + score

    def dominant_semantic_subtype(self) -> Optional[str]:
        if not self.semantic_subtype_scores:
            return None
        ranked = sorted(self.semantic_subtype_scores.items(), key=lambda item: item[1], reverse=True)
        top1_label, top1_score = ranked[0]
        top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if float(top1_score) >= float(self.semantic_subtype_ratio) * float(max(top2_score, 1e-6)):
            return top1_label
        return top1_label

    def semantic_subtype_soft_score(self, obs) -> float:
        obs_subtype = str(getattr(obs, "semantic_subtype", "") or "").strip()
        obs_subtype_conf = float(getattr(obs, "semantic_subtype_conf", 0.0) or 0.0)
        track_subtype = str(self.semantic_subtype or "").strip()
        if not obs_subtype or obs_subtype_conf <= 0.0 or not track_subtype:
            return 0.5
        if obs_subtype == track_subtype:
            return 1.0
        return 0.0

    def update_bbox_world_est(self, obs, momentum: float = 0.7) -> None:
        obs_min = obs.bbox3d_world[0].detach().cpu()
        obs_max = obs.bbox3d_world[1].detach().cpu()
        if self.bbox_world_est is None:
            bmin = obs_min
            bmax = obs_max
        else:
            old_min = torch.as_tensor(self.bbox_world_est.get("min", []), dtype=obs_min.dtype)
            old_max = torch.as_tensor(self.bbox_world_est.get("max", []), dtype=obs_max.dtype)
            if old_min.numel() != 3 or old_max.numel() != 3:
                bmin = obs_min
                bmax = obs_max
            else:
                bmin = momentum * old_min + (1.0 - momentum) * obs_min
                bmax = momentum * old_max + (1.0 - momentum) * obs_max

        self.bbox_world_est = {"min": bmin.tolist(), "max": bmax.tolist()}
        self.bbox_diag_world_est = bbox_diag((bmin, bmax))

    @staticmethod
    def _pose_act(pose, points: torch.Tensor) -> torch.Tensor:
        if hasattr(pose, "act"):
            return pose.act(points)
        if torch.is_tensor(pose) and pose.shape[-2:] == (4, 4):
            ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
            homo = torch.cat([points, ones], dim=-1)
            return (homo @ pose.transpose(-1, -2))[..., :3]
        return points

    @staticmethod
    def _pose_inv_act(pose, points: torch.Tensor) -> torch.Tensor:
        if hasattr(pose, "inv"):
            return pose.inv().act(points)
        if torch.is_tensor(pose) and pose.shape[-2:] == (4, 4):
            pose_inv = torch.linalg.inv(pose)
            ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
            homo = torch.cat([points, ones], dim=-1)
            return (homo @ pose_inv.transpose(-1, -2))[..., :3]
        return points

    @staticmethod
    def _bbox_corners(bmin: torch.Tensor, bmax: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                torch.stack([bmin[0], bmin[1], bmin[2]]),
                torch.stack([bmin[0], bmin[1], bmax[2]]),
                torch.stack([bmin[0], bmax[1], bmin[2]]),
                torch.stack([bmin[0], bmax[1], bmax[2]]),
                torch.stack([bmax[0], bmin[1], bmin[2]]),
                torch.stack([bmax[0], bmin[1], bmax[2]]),
                torch.stack([bmax[0], bmax[1], bmin[2]]),
                torch.stack([bmax[0], bmax[1], bmax[2]]),
            ],
            dim=0,
        )

    @staticmethod
    def _anchor_frame(keyframes, anchor_kf_id: Optional[int]):
        if anchor_kf_id is None:
            return None
        try:
            return keyframes[anchor_kf_id]
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

    def _canonical_local_geometry_tensors(
        self,
        *,
        device=None,
        dtype=torch.float32,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
        fused_points = self.fused_points_tensor(device=device, dtype=dtype)
        fused_centroid = self.fused_centroid_tensor(device=device, dtype=dtype)
        fused_min, fused_max = self.fused_bbox_tensors(device=device, dtype=dtype)
        if fused_points is not None or fused_centroid is not None or fused_min is not None:
            return fused_points, fused_centroid, fused_min, fused_max, "stable_anchor_fused"

        anchor_points = self.anchor_points_tensor(device=device, dtype=dtype)
        anchor_centroid = self.anchor_centroid_tensor(device=device, dtype=dtype)
        anchor_min, anchor_max = self.anchor_bbox_tensors(device=device, dtype=dtype)
        if anchor_points is not None or anchor_centroid is not None or anchor_min is not None:
            return anchor_points, anchor_centroid, anchor_min, anchor_max, "stable_anchor"
        return None, None, None, None, None

    def _set_fused_local_geometry(
        self,
        *,
        points_local,
        centroid_local,
        bbox_local,
        support_kfs,
        quality_score: Optional[float] = None,
        last_fused_kf: Optional[int] = None,
        points_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        # ``points_local`` is typically the output of ``tensor_to_list(merged)``
        # which is already a fresh list-of-lists, so copying it again via
        # ``[list(p) for p in points_local]`` just doubles the Python cost on
        # every keyframe commit.  Use the incoming container directly (callers
        # do not mutate it in place) and cache the tensor view when available.
        if points_local is None:
            self.points_fused_local = None
        else:
            self.points_fused_local = points_local
        self.centroid_fused_local = None if centroid_local is None else list(centroid_local)
        self.bbox_fused_local = None if bbox_local is None else {
            "min": list(bbox_local.get("min", [])),
            "max": list(bbox_local.get("max", [])),
        }
        self._cache_points_tensor("points_fused_local", points_tensor)
        self.fused_support_kfs = sorted({int(kf) for kf in (support_kfs or [])})
        self.anchor_support_kfs = list(self.fused_support_kfs)
        self.primary_anchor_support_kfs = list(self.anchor_support_kfs)
        self.fused_num_points = 0 if self.points_fused_local is None else int(len(self.points_fused_local))
        self.fused_geom_ready = bool(
            self.points_fused_local is not None
            or self.centroid_fused_local is not None
            or self.bbox_fused_local is not None
        )
        if quality_score is not None:
            self.fused_quality_score = float(quality_score)
        if last_fused_kf is not None:
            self.last_fused_kf = int(last_fused_kf)

    def _sync_anchor_snapshot_from_fused(self) -> None:
        if not self.fused_geom_ready:
            return
        # Compatibility path: legacy consumers still read points_anchor / bbox_anchor /
        # centroid_anchor. The formal stable geometry is points_fused_local / bbox_fused_local.
        # These list payloads are never mutated in place downstream, so aliasing the
        # underlying containers is safe and avoids an O(N) per-keyframe deep copy.
        self.points_anchor = self.points_fused_local
        self.centroid_anchor = None if self.centroid_fused_local is None else list(self.centroid_fused_local)
        self.bbox_anchor = None if self.bbox_fused_local is None else {
            "min": list(self.bbox_fused_local.get("min", [])),
            "max": list(self.bbox_fused_local.get("max", [])),
        }
        # Mirror the tensor cache so ``anchor_points_tensor`` hits the same
        # zero-copy fast path as ``fused_points_tensor``.
        cache = self._points_cache_dict()
        fused_tensor = cache.get("points_fused_local")
        if fused_tensor is None:
            cache.pop("points_anchor", None)
        else:
            cache["points_anchor"] = fused_tensor
        self.primary_anchor_support_kfs = list(self.anchor_support_kfs)

    def assoc_geom_source(self) -> str:
        if self.stable_geom_ready and self.anchor_kf_id is not None:
            _, _, _, _, stable_source = self._canonical_local_geometry_tensors()
            if stable_source is not None:
                return stable_source
        if self.stable_geom_ready and self.anchor_kf_id is not None and (
            self.points_anchor is not None or self.bbox_anchor is not None or self.centroid_anchor is not None
        ):
            return "stable_anchor"
        if self.has_provisional_anchor():
            return "provisional_anchor"
        if self._candidate_anchor_assoc_ready() and (
            self.candidate_points_frame is not None
            or self.candidate_bbox_frame is not None
            or self.candidate_centroid_frame is not None
        ):
            return "candidate_anchor"
        if self.bbox_world_est is not None:
            return "world_bbox"
        if self.centroid_world_est is not None:
            return "world_centroid"
        return "none"

    @staticmethod
    def _downsample_points_local(points: torch.Tensor, *, max_points: int = 512, voxel_size: float = 0.005) -> torch.Tensor:
        if points.shape[0] <= max_points:
            return points
        pts = points.detach().cpu().numpy()
        order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
        pts_sorted = pts[order]
        voxel_keys = np.floor(pts_sorted / float(voxel_size) + 0.5).astype(np.int64)
        kept = []
        seen = set()
        for idx, key in enumerate(voxel_keys):
            key_t = (int(key[0]), int(key[1]), int(key[2]))
            if key_t in seen:
                continue
            seen.add(key_t)
            kept.append(pts_sorted[idx])
        if not kept:
            kept = pts_sorted.tolist()
        kept_np = np.asarray(kept, dtype=np.float32)
        if kept_np.shape[0] > max_points:
            sample_idx = np.linspace(0, kept_np.shape[0] - 1, num=max_points, dtype=np.int64)
            kept_np = kept_np[sample_idx]
        return torch.as_tensor(kept_np, device=points.device, dtype=points.dtype)

    def _plane_pca_source_points_local(
        self,
        *,
        device=None,
        dtype=torch.float32,
    ) -> tuple[Optional[torch.Tensor], Optional[str], Optional[int], Optional[str]]:
        if self.stable_geom_ready and self.anchor_kf_id is not None:
            fused_points = self.fused_points_tensor(device=device, dtype=dtype)
            if fused_points is not None and int(fused_points.shape[0]) > 0:
                return fused_points, "stable_anchor_fused", int(self.anchor_kf_id), None
            anchor_points = self.anchor_points_tensor(device=device, dtype=dtype)
            if anchor_points is not None and int(anchor_points.shape[0]) > 0:
                return anchor_points, "stable_anchor", int(self.anchor_kf_id), None

        if self.has_provisional_anchor() and self.candidate_anchor_kf_id is not None:
            provisional_points = self.provisional_points_tensor(device=device, dtype=dtype)
            if provisional_points is not None and int(provisional_points.shape[0]) > 0:
                return provisional_points, "provisional_anchor", int(self.candidate_anchor_kf_id), None

        if self._candidate_anchor_assoc_ready() and self.candidate_anchor_kf_id is not None:
            candidate_points = self.candidate_points_tensor(device=device, dtype=dtype)
            if candidate_points is not None and int(candidate_points.shape[0]) > 0:
                return candidate_points, "candidate_anchor", int(self.candidate_anchor_kf_id), None

        best_aux = None
        for aux in self.auxiliary_anchors:
            pts = aux.get("source_points_frame")
            if pts is None:
                continue
            if best_aux is None or float(aux.get("quality_score", 0.0)) > float(best_aux.get("quality_score", 0.0)):
                best_aux = aux
        if best_aux is not None:
            aux_points = self._as_tensor(best_aux.get("source_points_frame"), device=device, dtype=dtype)
            if aux_points is not None and int(aux_points.shape[0]) > 0:
                return aux_points, "auxiliary_anchor", int(best_aux.get("kf_id")), None

        return None, None, None, "no_plane_pca_source_points"

    @staticmethod
    def _voxel_downsample_for_plane_pca(
        points: torch.Tensor,
        *,
        max_points: int = 256,
        voxel_size: float = 0.01,
    ) -> torch.Tensor:
        if points is None or int(points.shape[0]) == 0:
            return points
        pts = points.detach().cpu().numpy().astype(np.float32, copy=False)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return points[:0]
        order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
        pts_sorted = pts[order]
        voxel = max(float(voxel_size), 1e-6)
        voxel_keys = np.floor(pts_sorted / voxel + 0.5).astype(np.int64)
        kept = []
        seen = set()
        for idx, key in enumerate(voxel_keys):
            key_t = (int(key[0]), int(key[1]), int(key[2]))
            if key_t in seen:
                continue
            seen.add(key_t)
            kept.append(pts_sorted[idx])
        if not kept:
            kept = pts_sorted.tolist()
        kept_np = np.asarray(kept, dtype=np.float32)
        if kept_np.shape[0] < 3 and pts_sorted.shape[0] >= 3:
            kept_np = pts_sorted
        limit = max(1, int(max_points))
        if kept_np.shape[0] > limit:
            sample_idx = np.linspace(0, kept_np.shape[0] - 1, num=limit, dtype=np.int64)
            kept_np = kept_np[sample_idx]
        return torch.as_tensor(kept_np, device=points.device, dtype=points.dtype)

    @staticmethod
    def _estimate_plane_normal_from_points(points: torch.Tensor) -> dict:
        if points is None or int(points.shape[0]) < 3:
            return {
                "valid": False,
                "normal_local": None,
                "quality": 0.0,
                "eigenvalues": None,
                "reason": "insufficient_plane_pca_points",
            }
        pts = points.detach().cpu().numpy().astype(np.float64, copy=False)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return {
                "valid": False,
                "normal_local": None,
                "quality": 0.0,
                "eigenvalues": None,
                "reason": "invalid_plane_pca_points",
            }
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(1, int(centered.shape[0]) - 1)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return {
                "valid": False,
                "normal_local": None,
                "quality": 0.0,
                "eigenvalues": None,
                "reason": "plane_pca_eigh_failed",
            }
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        normal = eigvecs[:, 0]
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            return {
                "valid": False,
                "normal_local": None,
                "quality": 0.0,
                "eigenvalues": eigvals.astype(float).tolist(),
                "reason": "degenerate_plane_normal",
            }
        normal = normal / norm
        quality = 1.0 - float(eigvals[0]) / max(float(eigvals[1]), 1e-9)
        return {
            "valid": True,
            "normal_local": normal.astype(np.float32).tolist(),
            "quality": float(max(0.0, min(1.0, quality))),
            "eigenvalues": eigvals.astype(float).tolist(),
            "reason": "updated",
        }

    @classmethod
    def _pose_act_cpu_safe(cls, pose, points: torch.Tensor) -> torch.Tensor:
        if hasattr(pose, "act") and hasattr(pose, "data") and torch.is_tensor(pose.data):
            import lietorch

            pose_cpu = lietorch.Sim3(pose.data.detach().cpu().clone())
            return cls._pose_act(pose_cpu, points.detach().cpu())
        return cls._pose_act(pose, points)

    @classmethod
    def _local_direction_to_world(cls, pose, direction_local: torch.Tensor) -> Optional[torch.Tensor]:
        if direction_local is None or direction_local.numel() != 3:
            return None
        origin = torch.zeros((1, 3), dtype=torch.float32)
        tip = direction_local.detach().cpu().to(dtype=torch.float32).reshape(1, 3)
        points = torch.cat([origin, tip], dim=0)
        try:
            world = cls._pose_act_cpu_safe(pose, points)
        except Exception:
            return None
        direction = world[1] - world[0]
        norm = torch.linalg.norm(direction)
        if float(norm.item()) <= 1e-9:
            return None
        return direction / norm

    def update_plane_normal_cache(self, kf_idx: int, keyframes, policy) -> dict:
        debug = {
            "plane_normal_update_attempted": True,
            "plane_normal_source": None,
            "plane_normal_quality": float(self.plane_normal_quality),
            "plane_normal_num_points": int(self.plane_normal_num_points),
            "plane_normal_version": int(self.plane_normal_version),
            "plane_normal_update_reason": None,
        }
        points_local, source, source_kf_id, reason = self._plane_pca_source_points_local(dtype=torch.float32)
        if points_local is None or source is None or source_kf_id is None:
            reason = reason or "no_plane_pca_source_points"
            self.plane_normal_last_update_reason = reason
            debug["plane_normal_update_reason"] = reason
            return debug

        anchor_frame = self._anchor_frame(keyframes, source_kf_id)
        pose = getattr(anchor_frame, "T_WC", None) if anchor_frame is not None else None
        if pose is None:
            reason = "no_plane_pca_pose"
            self.plane_normal_last_update_reason = reason
            debug.update(
                {
                    "plane_normal_source": source,
                    "plane_normal_update_reason": reason,
                }
            )
            return debug

        downsampled = self._voxel_downsample_for_plane_pca(
            points_local,
            max_points=int(getattr(policy, "cabinet_plane_normal_pca_max_points", 256)),
            voxel_size=float(getattr(policy, "cabinet_plane_normal_voxel_size", 0.01)),
        )
        est = self._estimate_plane_normal_from_points(downsampled)
        normal_local = est.get("normal_local")
        normal_world = None
        if normal_local is not None:
            normal_world_t = self._local_direction_to_world(pose, torch.as_tensor(normal_local, dtype=torch.float32))
            if normal_world_t is not None:
                normal_world = normal_world_t.detach().cpu().tolist()

        quality = float(est.get("quality", 0.0) or 0.0)
        min_quality = float(getattr(policy, "cabinet_min_plane_normal_quality", 0.15))
        if normal_world is None:
            reason = est.get("reason") if est.get("reason") != "updated" else "plane_normal_transform_failed"
        elif quality < min_quality:
            reason = "low_plane_normal_quality"
        else:
            reason = "updated"

        self.plane_normal_world_est = None if normal_world is None else [float(v) for v in normal_world]
        self.plane_normal_quality = quality
        self.plane_normal_source = source
        self.plane_normal_kf_id = int(kf_idx)
        self.plane_normal_version += 1
        self.plane_normal_num_points = int(0 if downsampled is None else downsampled.shape[0])
        eigvals = est.get("eigenvalues")
        self.plane_normal_eigenvalues = None if eigvals is None else [float(v) for v in eigvals]
        self.plane_normal_last_update_reason = reason

        debug.update(
            {
                "plane_normal_source": self.plane_normal_source,
                "plane_normal_quality": float(self.plane_normal_quality),
                "plane_normal_num_points": int(self.plane_normal_num_points),
                "plane_normal_version": int(self.plane_normal_version),
                "plane_normal_update_reason": self.plane_normal_last_update_reason,
            }
        )
        return debug

    def _anchor_fusion_gate(self, points_local: torch.Tensor) -> bool:
        anchor_min, anchor_max = self.fused_bbox_tensors(device=points_local.device, dtype=points_local.dtype)
        if anchor_min is None or anchor_max is None:
            anchor_min, anchor_max = self.anchor_bbox_tensors(device=points_local.device, dtype=points_local.dtype)
        if anchor_min is None or anchor_max is None:
            return True
        return self._local_fusion_gate(points_local, anchor_min, anchor_max)

    @staticmethod
    def _bbox_center_tensor(bmin: torch.Tensor, bmax: torch.Tensor) -> torch.Tensor:
        return 0.5 * (bmin + bmax)

    @staticmethod
    def _local_fusion_gate(points_local: torch.Tensor, ref_min: torch.Tensor, ref_max: torch.Tensor) -> bool:
        obs_min, obs_max = bbox_from_points(points_local)
        anchor_extent = bbox_size((ref_min, ref_max))
        obs_extent = bbox_size((obs_min, obs_max))
        anchor_center = 0.5 * (ref_min + ref_max)
        obs_center = 0.5 * (obs_min + obs_max)
        center_dist = float(torch.linalg.norm(obs_center - anchor_center).item())
        anchor_scale = float(torch.linalg.norm(anchor_extent).item())
        obs_scale = float(torch.linalg.norm(obs_extent).item())
        geom_scale = max(anchor_scale, obs_scale, 1e-6)
        if center_dist > max(0.05, 1.5 * geom_scale):
            return False
        min_scale = max(1e-6, min(anchor_scale, obs_scale))
        max_scale = max(anchor_scale, obs_scale)
        return (max_scale / min_scale) <= 4.0

    @staticmethod
    def _bbox_overlap_or_near(obs_min: torch.Tensor, obs_max: torch.Tensor, ref_min: torch.Tensor, ref_max: torch.Tensor) -> bool:
        inter_min = torch.maximum(obs_min, ref_min)
        inter_max = torch.minimum(obs_max, ref_max)
        inter_extent = torch.clamp(inter_max - inter_min, min=0.0)
        if bool(torch.all(inter_extent > 1e-6)):
            return True
        ref_extent = bbox_size((ref_min, ref_max))
        obs_extent = bbox_size((obs_min, obs_max))
        margin = 0.35 * torch.maximum(ref_extent, obs_extent)
        expanded_min = ref_min - margin
        expanded_max = ref_max + margin
        return bool(torch.all(obs_max >= expanded_min) and torch.all(obs_min <= expanded_max))

    @staticmethod
    def _observation_quality_score(*, num_points: int, view_score: float, border_touched: bool) -> float:
        point_term = min(1.0, float(num_points) / 128.0)
        view_term = min(1.0, float(view_score) / 0.25)
        border_term = 0.0 if border_touched else 1.0
        return float(0.50 * point_term + 0.35 * view_term + 0.15 * border_term)

    def _anchor_observation_quality_ok(
        self,
        obs,
        *,
        min_points: int,
        min_view_score: float,
        allow_border: bool,
        border_min_points: int,
        border_min_view_score: float,
    ) -> tuple[bool, str]:
        num_points = int(getattr(obs, "num_points", 0) or 0)
        view_score = float(getattr(obs, "view_score", 0.0) or 0.0)
        border_touched = bool(getattr(obs, "box_touches_border", False))
        if num_points < int(min_points):
            return False, "low_num_points"
        if view_score < float(min_view_score):
            return False, "low_view_score"
        if border_touched:
            if not allow_border:
                return False, "border_touched"
            if num_points < int(border_min_points):
                return False, "border_low_num_points"
            if view_score < float(border_min_view_score):
                return False, "border_low_view_score"
        return True, "ok"

    def _anchor_fusion_accept(
        self,
        obs,
        points_local: torch.Tensor,
        *,
        ref_min: torch.Tensor,
        ref_max: torch.Tensor,
        ref_centroid: Optional[torch.Tensor],
        min_points: int,
        min_view_score: float,
        allow_border: bool,
        border_min_points: int,
        border_min_view_score: float,
    ) -> tuple[bool, str, float]:
        ok, reason = self._anchor_observation_quality_ok(
            obs,
            min_points=min_points,
            min_view_score=min_view_score,
            allow_border=allow_border,
            border_min_points=border_min_points,
            border_min_view_score=border_min_view_score,
        )
        if not ok:
            return False, reason, 0.0
        if not self._local_fusion_gate(points_local, ref_min, ref_max):
            return False, "geom_center_scale", 0.0
        obs_min, obs_max = bbox_from_points(points_local)
        obs_centroid = points_local.mean(dim=0)
        if ref_centroid is not None:
            ref_extent = bbox_size((ref_min, ref_max))
            geom_scale = max(float(torch.linalg.norm(ref_extent).item()), 0.05)
            centroid_dist = float(torch.linalg.norm(obs_centroid - ref_centroid).item())
            if centroid_dist > max(0.08, 1.25 * geom_scale):
                return False, "centroid_inconsistent", 0.0
        if not self._bbox_overlap_or_near(obs_min, obs_max, ref_min, ref_max):
            return False, "bbox_disjoint", 0.0
        quality = self._observation_quality_score(
            num_points=int(getattr(obs, "num_points", 0) or int(points_local.shape[0])),
            view_score=float(getattr(obs, "view_score", 0.0) or 0.0),
            border_touched=bool(getattr(obs, "box_touches_border", False)),
        )
        return True, "accept", quality

    def has_provisional_anchor(self) -> bool:
        return bool(
            self.candidate_anchor_kf_id is not None
            and len(self.provisional_support_kfs) >= 2
            and (
                self.provisional_points_frame is not None
                or self.provisional_bbox_frame is not None
                or self.provisional_centroid_frame is not None
            )
        )

    def _candidate_anchor_assoc_ready(self) -> bool:
        if self.candidate_anchor_kf_id is None:
            return False
        if (
            self.candidate_points_frame is None
            and self.candidate_bbox_frame is None
            and self.candidate_centroid_frame is None
        ):
            return False
        if not self.candidate_border_touched:
            return True
        # Relaxed thresholds for border-touched C nodes so that door/drawer
        # candidates can compete earlier, without overriding stable anchor
        # priority in assoc_geom_source().
        return bool(
            self.role == "C"
            and int(self.obs_count) >= 2
            and int(self.candidate_num_points) >= 384
            and float(self.candidate_view_score) >= 0.08
        )

    def _pre_stable_source_stats(self) -> tuple[Optional[list[list[float]]], Optional[list[float]], Optional[dict], float, int, bool, list[int]]:
        if self.has_provisional_anchor():
            return (
                self.provisional_points_frame,
                self.provisional_centroid_frame,
                self.provisional_bbox_frame,
                float(self.provisional_view_score),
                int(self.provisional_num_points),
                bool(self.provisional_border_touched),
                list(self.provisional_support_kfs),
            )
        return (
            self.candidate_points_frame,
            self.candidate_centroid_frame,
            self.candidate_bbox_frame,
            float(self.candidate_view_score),
            int(self.candidate_num_points),
            bool(self.candidate_border_touched),
            list(self.candidate_support_kfs or ([int(self.candidate_anchor_kf_id)] if self.candidate_anchor_kf_id is not None else [])),
        )

    def _local_centroid_to_world(
        self,
        centroid_local,
        *,
        keyframes,
        anchor_kf_id: Optional[int],
    ) -> Optional[torch.Tensor]:
        if centroid_local is None or anchor_kf_id is None:
            return None
        anchor_frame = self._anchor_frame(keyframes, anchor_kf_id)
        if anchor_frame is None:
            return None
        pose = getattr(anchor_frame, "T_WC", None)
        if pose is None:
            return None
        centroid_t = torch.as_tensor(centroid_local, dtype=torch.float32).reshape(1, 3)
        if hasattr(pose, "act") and hasattr(pose, "data") and torch.is_tensor(pose.data):
            import lietorch
            pose_cpu = lietorch.Sim3(pose.data.detach().cpu().clone())
            return self._pose_act(pose_cpu, centroid_t.cpu())[0]
        return self._pose_act(pose, centroid_t)[0]

    def _local_bbox_to_world(
        self,
        bbox_local,
        *,
        keyframes,
        anchor_kf_id: Optional[int],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if bbox_local is None or anchor_kf_id is None:
            return None, None
        anchor_frame = self._anchor_frame(keyframes, anchor_kf_id)
        if anchor_frame is None:
            return None, None
        pose = getattr(anchor_frame, "T_WC", None)
        if pose is None:
            return None, None
        bmin = torch.as_tensor(bbox_local.get("min", []), dtype=torch.float32)
        bmax = torch.as_tensor(bbox_local.get("max", []), dtype=torch.float32)
        if bmin.numel() != 3 or bmax.numel() != 3:
            return None, None
        corners = self._bbox_corners(bmin, bmax)
        if hasattr(pose, "act") and hasattr(pose, "data") and torch.is_tensor(pose.data):
            import lietorch
            pose_cpu = lietorch.Sim3(pose.data.detach().cpu().clone())
            corners_world = self._pose_act(pose_cpu, corners.cpu())
        else:
            corners_world = self._pose_act(pose, corners)
        return bbox_from_points(corners_world)

    def _candidate_anchor_quality_ok(
        self,
        *,
        min_points: int,
        min_view_score: float,
        allow_border_touched: bool,
        border_min_points: int,
        border_min_view_score: float,
    ) -> tuple[bool, str]:
        _, _, _, view_score, num_points, border_touched, _ = self._pre_stable_source_stats()
        if num_points < int(min_points):
            return False, "candidate_low_num_points"
        if view_score < float(min_view_score):
            return False, "candidate_low_view_score"
        if border_touched:
            if not allow_border_touched:
                return False, "candidate_border_touched"
            if num_points < int(border_min_points):
                return False, "candidate_border_low_num_points"
            if view_score < float(border_min_view_score):
                return False, "candidate_border_low_view_score"
        return True, "ok"

    def _candidate_anchor_stability_ok(
        self,
        *,
        min_kf_support: int,
        keyframes,
        center_dev_max: float,
        centroid_dev_max: float,
    ) -> tuple[bool, str]:
        _, centroid_local, bbox_local, _, _, _, support_kfs = self._pre_stable_source_stats()
        if len(support_kfs) < int(min_kf_support):
            return False, "insufficient_kf_support"
        if keyframes is None or self.candidate_anchor_kf_id is None:
            return True, "ok"
        bbox_world_min, bbox_world_max = self._local_bbox_to_world(
            bbox_local,
            keyframes=keyframes,
            anchor_kf_id=self.candidate_anchor_kf_id,
        )
        if self.bbox_world_est is not None and bbox_world_min is not None and bbox_world_max is not None:
            est_min = torch.as_tensor(self.bbox_world_est.get("min", []), dtype=torch.float32)
            est_max = torch.as_tensor(self.bbox_world_est.get("max", []), dtype=torch.float32)
            if est_min.numel() == 3 and est_max.numel() == 3:
                cand_center = self._bbox_center_tensor(bbox_world_min, bbox_world_max)
                est_center = self._bbox_center_tensor(est_min, est_max)
                center_dist = float(torch.linalg.norm(cand_center - est_center).item())
                cand_diag = bbox_diag((bbox_world_min, bbox_world_max))
                est_diag = bbox_diag((est_min, est_max))
                max_center_dev = max(float(center_dev_max), 0.75 * max(cand_diag, est_diag, 1e-3))
                if center_dist > max_center_dev:
                    return False, "bbox_world_deviation"
        centroid_world = self._local_centroid_to_world(
            centroid_local,
            keyframes=keyframes,
            anchor_kf_id=self.candidate_anchor_kf_id,
        )
        if self.centroid_world_est is not None and centroid_world is not None:
            world_est = torch.as_tensor(self.centroid_world_est, dtype=torch.float32)
            if world_est.numel() == 3:
                centroid_dist = float(torch.linalg.norm(centroid_world - world_est).item())
                max_centroid_dev = max(float(centroid_dev_max), 0.75 * max(float(self.bbox_diag_world_est), 1e-3), 0.05)
                if centroid_dist > max_centroid_dev:
                    return False, "centroid_world_deviation"
        return True, "ok"

    def predict_centroid_world(self, keyframes, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        stable_points, stable_centroid, stable_min, stable_max, _ = self._canonical_local_geometry_tensors(
            device=device,
            dtype=dtype,
        )
        if self.stable_geom_ready and self.anchor_kf_id is not None and (
            stable_centroid is not None or stable_points is not None or stable_min is not None
        ):
            anchor_frame = self._anchor_frame(keyframes, self.anchor_kf_id)
            if anchor_frame is not None:
                pose = getattr(anchor_frame, "T_WC", None)
                if pose is not None:
                    local_centroid = stable_centroid
                    if local_centroid is None and stable_points is not None and int(stable_points.shape[0]) > 0:
                        local_centroid = stable_points.mean(dim=0)
                    if local_centroid is None and stable_min is not None and stable_max is not None:
                        local_centroid = 0.5 * (stable_min + stable_max)
                    if local_centroid is not None:
                        return self._pose_act(pose, local_centroid[None])[0]
        provisional_centroid = self.provisional_centroid_tensor(device=device, dtype=dtype)
        if self.has_provisional_anchor() and self.candidate_anchor_kf_id is not None and provisional_centroid is not None:
            anchor_frame = self._anchor_frame(keyframes, self.candidate_anchor_kf_id)
            if anchor_frame is not None:
                pose = getattr(anchor_frame, "T_WC", None)
                if pose is not None:
                    return self._pose_act(pose, provisional_centroid[None])[0]
        candidate_centroid = self.candidate_centroid_tensor(device=device, dtype=dtype)
        if self._candidate_anchor_assoc_ready() and self.candidate_anchor_kf_id is not None and candidate_centroid is not None:
            anchor_frame = self._anchor_frame(keyframes, self.candidate_anchor_kf_id)
            if anchor_frame is not None:
                pose = getattr(anchor_frame, "T_WC", None)
                if pose is not None:
                    return self._pose_act(pose, candidate_centroid[None])[0]
        return self.world_centroid_tensor(device=device, dtype=dtype)

    def predict_points_world(self, keyframes, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        stable_points, _, stable_min, stable_max, _ = self._canonical_local_geometry_tensors(
            device=device,
            dtype=dtype,
        )
        if self.stable_geom_ready and self.anchor_kf_id is not None and (stable_points is not None or stable_min is not None):
            anchor_frame = self._anchor_frame(keyframes, self.anchor_kf_id)
            if anchor_frame is not None:
                pose = getattr(anchor_frame, "T_WC", None)
                if pose is not None:
                    local_points = []
                    if stable_points is not None:
                        local_points.append(stable_points)
                    if stable_min is not None and stable_max is not None:
                        local_points.append(self._bbox_corners(stable_min, stable_max))
                    if local_points:
                        return self._pose_act(pose, torch.cat(local_points, dim=0))
        provisional_points = self.provisional_points_tensor(device=device, dtype=dtype)
        provisional_min, provisional_max = self.provisional_bbox_tensors(device=device, dtype=dtype)
        if self.has_provisional_anchor() and self.candidate_anchor_kf_id is not None and (
            provisional_points is not None or provisional_min is not None
        ):
            anchor_frame = self._anchor_frame(keyframes, self.candidate_anchor_kf_id)
            if anchor_frame is not None:
                pose = getattr(anchor_frame, "T_WC", None)
                if pose is not None:
                    local_points = []
                    if provisional_points is not None:
                        local_points.append(provisional_points)
                    if provisional_min is not None and provisional_max is not None:
                        local_points.append(self._bbox_corners(provisional_min, provisional_max))
                    if local_points:
                        return self._pose_act(pose, torch.cat(local_points, dim=0))
        candidate_points = self.candidate_points_tensor(device=device, dtype=dtype)
        candidate_min, candidate_max = self.candidate_bbox_tensors(device=device, dtype=dtype)
        if self._candidate_anchor_assoc_ready() and self.candidate_anchor_kf_id is not None and (
            candidate_points is not None or candidate_min is not None
        ):
            anchor_frame = self._anchor_frame(keyframes, self.candidate_anchor_kf_id)
            if anchor_frame is not None:
                pose = getattr(anchor_frame, "T_WC", None)
                if pose is not None:
                    local_points = []
                    if candidate_points is not None:
                        local_points.append(candidate_points)
                    if candidate_min is not None and candidate_max is not None:
                        local_points.append(self._bbox_corners(candidate_min, candidate_max))
                    if local_points:
                        return self._pose_act(pose, torch.cat(local_points, dim=0))
        centroid = self.world_centroid_tensor(device=device, dtype=dtype)
        if centroid is None:
            return None
        if self.bbox_world_est is not None:
            bmin = torch.as_tensor(self.bbox_world_est.get("min", []), device=centroid.device, dtype=centroid.dtype)
            bmax = torch.as_tensor(self.bbox_world_est.get("max", []), device=centroid.device, dtype=centroid.dtype)
            if bmin.numel() == 3 and bmax.numel() == 3:
                return self._bbox_corners(bmin, bmax)
        return centroid[None]

    def _last_obs_intrinsics_tensor(self, *, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        if self.last_obs_K is not None:
            K = torch.as_tensor(self.last_obs_K, device=device, dtype=dtype)
            if K.numel() == 9:
                return K.reshape(3, 3)
        if self.last_obs_hw is None:
            return None
        h, w = [int(v) for v in self.last_obs_hw[:2]]
        focal = float(max(h, w)) * 0.5
        return torch.tensor(
            [[focal, 0.0, float(w) * 0.5], [0.0, focal, float(h) * 0.5], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )

    def predict_recent_box_xyxy(self, frame, *, device=None, dtype=torch.float32) -> Optional[list[float]]:
        if frame is None or self.last_obs_pose is None or self.last_box_xyxy is None:
            return None
        if self.last_obs_depth is None or float(self.last_obs_depth) <= 1e-6:
            return None
        K_prev = self._last_obs_intrinsics_tensor(device=device, dtype=dtype)
        if K_prev is None:
            return None
        x1, y1, x2, y2 = [float(v) for v in self.last_box_xyxy]
        prev_uv = torch.tensor(
            [[x1, y1], [x1, y2], [x2, y1], [x2, y2]],
            device=device,
            dtype=dtype,
        )
        z = float(self.last_obs_depth)
        fx = float(K_prev[0, 0].item())
        fy = float(K_prev[1, 1].item())
        cx = float(K_prev[0, 2].item())
        cy = float(K_prev[1, 2].item())
        if abs(fx) <= 1e-6 or abs(fy) <= 1e-6:
            return None
        prev_points = torch.stack(
            [
                (prev_uv[:, 0] - cx) * z / fx,
                (prev_uv[:, 1] - cy) * z / fy,
                torch.full((prev_uv.shape[0],), z, device=device, dtype=dtype),
            ],
            dim=-1,
        )
        world_points = self._pose_act(self.last_obs_pose, prev_points)
        return project_box_xyxy(frame, world_points)

    def _record_observed_frame(self, frame_idx: int) -> None:
        """Append ``frame_idx`` to the bounded ``observed_frames`` ring
        buffer with last-entry deduplication, and update
        ``first_seen_frame``.  Public for direct unit testing of the
        co-visibility book-keeping path."""
        try:
            f_int = int(frame_idx)
        except Exception:
            return
        if self.first_seen_frame is None or f_int < int(self.first_seen_frame):
            self.first_seen_frame = f_int
        if not self.observed_frames or self.observed_frames[-1] != f_int:
            self.observed_frames.append(f_int)
            cap = int(self.observed_frame_cap or 0) or 256
            if len(self.observed_frames) > cap:
                del self.observed_frames[: len(self.observed_frames) - cap]

    def _record_observation_evidence(self, obs, frame_idx: int, kf_idx: Optional[int] = None) -> None:
        """Append a small observation-evidence record (bbox, area,
        centroid, view score, point count) to ``recent_observations``
        and refresh the top-K ``best_observations`` heap-by-list.

        Defensive: any field that cannot be extracted falls back to
        ``None`` / ``0`` rather than raising, so this is safe to call
        from the hot association path.
        """
        try:
            view_score = float(getattr(obs, "score", 0.0) or 0.0)
        except Exception:
            view_score = 0.0
        try:
            num_points = int(getattr(obs, "num_points", 0) or 0)
        except Exception:
            num_points = 0
        try:
            mask_area = getattr(obs, "mask_area", None)
            mask_area = float(mask_area) if mask_area is not None else None
        except Exception:
            mask_area = None
        bbox_xyxy = None
        try:
            box = getattr(obs, "box_xyxy", None)
            if box is not None:
                bbox_xyxy = [float(v) for v in list(box)[:4]]
        except Exception:
            bbox_xyxy = None
        centroid_world = None
        try:
            cw = getattr(obs, "centroid_world", None)
            if cw is not None:
                centroid_world = [float(v) for v in cw.detach().cpu().tolist()]
        except Exception:
            centroid_world = None
        centroid_frame_depth = None
        try:
            cf = getattr(obs, "centroid_frame", None)
            if cf is not None:
                centroid_frame_depth = float(cf.reshape(-1)[2].detach().cpu().item())
        except Exception:
            centroid_frame_depth = None
        geom_source = ""
        try:
            geom_source = str(self.assoc_geom_source() or "")
        except Exception:
            geom_source = ""
        record = {
            "frame_idx": int(frame_idx) if frame_idx is not None else -1,
            "kf_idx": int(kf_idx) if kf_idx is not None else None,
            "bbox_xyxy": bbox_xyxy,
            "mask_area": mask_area,
            "centroid_world": centroid_world,
            "centroid_frame_depth": centroid_frame_depth,
            "view_score": view_score,
            "num_points": num_points,
            "geom_source": geom_source,
            "matched_node_id": str(self.node_id),
            "label": str(self.label or ""),
            "role": str(self.role or ""),
        }
        cap = int(self.recent_observations_cap or 16)
        self.recent_observations.append(record)
        if len(self.recent_observations) > cap:
            del self.recent_observations[: len(self.recent_observations) - cap]
        # Refresh best_observations: keep top-K by (view_score, num_points).
        best_cap = int(self.best_observations_cap or 5)
        candidates = list(self.best_observations) + [record]
        candidates.sort(
            key=lambda r: (
                float(r.get("view_score", 0.0) or 0.0),
                float(r.get("num_points", 0) or 0),
            ),
            reverse=True,
        )
        # De-duplicate by frame_idx (keep highest scoring per frame).
        seen_frames: set[int] = set()
        deduped: list[dict] = []
        for r in candidates:
            f = int(r.get("frame_idx", -1))
            if f in seen_frames:
                continue
            seen_frames.add(f)
            deduped.append(r)
            if len(deduped) >= best_cap:
                break
        self.best_observations = deduped

    def update_observation(self, obs, frame_idx: int, frame=None) -> None:
        was_empty = self.obs_count == 0
        self.last_seen_frame = frame_idx
        self.obs_count += 1
        self.visible_count += 1
        # Co-visibility / first_seen bookkeeping.
        self._record_observed_frame(frame_idx)
        # Section A: record a small observation-evidence summary so node
        # consolidation can compare best/recent-view geometry across
        # potentially-duplicate tracks.
        try:
            self._record_observation_evidence(obs, frame_idx, kf_idx=getattr(frame, "frame_id", None))
        except Exception:
            pass
        self.last_box_xyxy = list(obs.box_xyxy or []) if obs.box_xyxy is not None else None
        self.last_obs_pose = getattr(frame, "T_WC", None) if frame is not None else None
        self.last_obs_depth = float(obs.centroid_frame[2].item()) if getattr(obs, "centroid_frame", None) is not None else None
        if frame is not None:
            K = frame_intrinsics(frame)
            self.last_obs_K = tensor_to_list(K) if K is not None else None
            try:
                h, w = frame_hw(frame)
                self.last_obs_hw = [int(h), int(w)]
            except Exception:
                self.last_obs_hw = None

        if not (was_empty and self.label_counts.get(obs.label, 0) > 0):
            self.update_label_hist(obs.label)
        self.label = self.dominant_label()
        self.update_semantic_subtype_hist(
            getattr(obs, "semantic_subtype", None),
            float(getattr(obs, "semantic_subtype_conf", 0.0) or 0.0),
        )
        self.semantic_subtype = self.dominant_semantic_subtype()

        centroid_world = obs.centroid_world.detach().cpu()
        if self.centroid_world_est is None:
            self.centroid_world_est = centroid_world.tolist()
        else:
            old = centroid_world.new_tensor(self.centroid_world_est)
            self.centroid_world_est = (0.7 * old + 0.3 * centroid_world).tolist()

        self.update_bbox_world_est(obs)

        if obs.view_score >= self.best_view_score:
            self.best_view_score = float(obs.view_score)
            self.best_view_frame = frame_idx
            self.best_view_crop_meta = {
                "det_idx": int(obs.det_idx),
                "box_xyxy": list(obs.box_xyxy or []),
            }

    def _is_candidate_better(self, obs) -> bool:
        if self.candidate_anchor_kf_id is None:
            return True
        current_key = (
            1 if not self.candidate_border_touched else 0,
            float(self.candidate_view_score),
            int(self.candidate_num_points),
        )
        new_key = (
            1 if not bool(obs.box_touches_border) else 0,
            float(obs.view_score),
            int(obs.num_points),
        )
        return new_key > current_key

    def consider_anchor_candidate(self, kf_idx: int, obs) -> None:
        self.last_seen_kf = kf_idx
        if not self.stable_geom_ready and self.provisional_support_kfs:
            return
        if not self._is_candidate_better(obs):
            return
        self.candidate_anchor_kf_id = kf_idx
        self.candidate_support_kfs = [int(kf_idx)]
        obs_points_frame = obs.points_frame
        self.candidate_points_frame = tensor_to_list(obs_points_frame)
        # Cache the CPU tensor so subsequent ``candidate_points_tensor`` /
        # ``provisional_points_tensor`` calls during the same commit skip the
        # expensive ``torch.as_tensor(list_of_lists)`` rebuild.
        self._cache_points_tensor("candidate_points_frame", obs_points_frame)
        self.candidate_centroid_frame = tensor_to_list(obs.centroid_frame)
        self.candidate_bbox_frame = {
            "min": tensor_to_list(obs.bbox3d_frame[0]),
            "max": tensor_to_list(obs.bbox3d_frame[1]),
        }
        self.candidate_view_score = float(obs.view_score)
        self.candidate_num_points = int(obs.num_points)
        self.candidate_border_touched = bool(obs.box_touches_border)
        self.candidate_box_touches_border = self.candidate_border_touched
        self.provisional_points_frame = None
        self._cache_points_tensor("provisional_points_frame", None)
        self.provisional_centroid_frame = None
        self.provisional_bbox_frame = None
        self.provisional_pose = None
        self.provisional_view_score = 0.0
        self.provisional_num_points = 0
        self.provisional_border_touched = True
        self.provisional_support_kfs = []
        self.last_pre_stable_reject_reason = None

    def consider_provisional_anchor(
        self,
        kf_idx: int,
        keyframes,
        obs,
        *,
        max_points: int = 512,
        enable_local_accum: bool = True,
        min_points: int = 8,
        min_view_score: float = 0.0,
        allow_border: bool = False,
        border_min_points: int = 0,
        border_min_view_score: float = 0.0,
    ) -> bool:
        if self.disable_3d_fusion:
            self._record_pre_stable_reject("disabled_by_policy")
            return False
        if self.stable_geom_ready:
            return False
        if self.candidate_anchor_kf_id is None:
            return False
        anchor_frame = self._anchor_frame(keyframes, self.candidate_anchor_kf_id)
        if anchor_frame is None:
            return False
        pose = getattr(anchor_frame, "T_WC", None)
        if pose is None:
            return False

        if not self.candidate_support_kfs:
            self.candidate_support_kfs = [int(self.candidate_anchor_kf_id)]

        if enable_local_accum and not self.provisional_support_kfs:
            if self.candidate_points_frame is not None:
                # The candidate list is never mutated in place, so aliasing it
                # as the provisional seed avoids an O(N) Python copy per commit.
                self.provisional_points_frame = self.candidate_points_frame
                candidate_cache = self._points_cache_dict().get("candidate_points_frame")
                if candidate_cache is not None:
                    self._points_cache_dict()["provisional_points_frame"] = candidate_cache
                else:
                    self._cache_points_tensor("provisional_points_frame", None)
            else:
                self.provisional_points_frame = None
                self._cache_points_tensor("provisional_points_frame", None)
            self.provisional_centroid_frame = list(self.candidate_centroid_frame) if self.candidate_centroid_frame is not None else None
            self.provisional_bbox_frame = dict(self.candidate_bbox_frame) if self.candidate_bbox_frame is not None else None
            self.provisional_pose = pose
            self.provisional_view_score = float(self.candidate_view_score)
            self.provisional_num_points = int(self.candidate_num_points)
            self.provisional_border_touched = bool(self.candidate_border_touched)
            self.provisional_support_kfs = [int(self.candidate_anchor_kf_id)]

        if int(kf_idx) in self.candidate_support_kfs and (not enable_local_accum or int(kf_idx) in self.provisional_support_kfs):
            return True
        points_world = getattr(obs, "points_world", None)
        if points_world is None or int(points_world.shape[0]) == 0:
            return False
        if int(kf_idx) == int(self.candidate_anchor_kf_id):
            points_local = obs.points_frame
        else:
            points_local = self._pose_inv_act(pose, points_world)

        ref_min, ref_max = self.provisional_bbox_tensors(device=points_local.device, dtype=points_local.dtype)
        if ref_min is None or ref_max is None:
            ref_min, ref_max = self.candidate_bbox_tensors(device=points_local.device, dtype=points_local.dtype)
        ref_centroid = self.provisional_centroid_tensor(device=points_local.device, dtype=points_local.dtype)
        if ref_centroid is None:
            ref_centroid = self.candidate_centroid_tensor(device=points_local.device, dtype=points_local.dtype)
        if ref_min is not None and ref_max is not None:
            accepted, reason, _ = self._anchor_fusion_accept(
                obs,
                points_local,
                ref_min=ref_min,
                ref_max=ref_max,
                ref_centroid=ref_centroid,
                min_points=min_points,
                min_view_score=min_view_score,
                allow_border=allow_border,
                border_min_points=border_min_points,
                border_min_view_score=border_min_view_score,
            )
            if not accepted:
                self._record_pre_stable_reject(reason)
                return False
        self.last_pre_stable_reject_reason = None
        self.last_pre_stable_accept_kf = int(kf_idx)
        if int(kf_idx) not in self.candidate_support_kfs:
            self.candidate_support_kfs = sorted(set([*self.candidate_support_kfs, int(kf_idx)]))
        if not enable_local_accum:
            return True
        if int(kf_idx) in self.provisional_support_kfs:
            return True

        provisional_points = self.provisional_points_tensor(device=points_local.device, dtype=points_local.dtype)
        if provisional_points is None:
            provisional_points = self.candidate_points_tensor(device=points_local.device, dtype=points_local.dtype)
        merged = points_local if provisional_points is None else torch.cat([provisional_points, points_local], dim=0)
        merged = self._downsample_points_local(merged, max_points=max_points)
        bmin, bmax = bbox_from_points(merged)
        self.provisional_points_frame = tensor_to_list(merged)
        self._cache_points_tensor("provisional_points_frame", merged)
        self.provisional_centroid_frame = tensor_to_list(merged.mean(dim=0))
        self.provisional_bbox_frame = {
            "min": tensor_to_list(bmin),
            "max": tensor_to_list(bmax),
        }
        self.provisional_pose = pose
        self.provisional_view_score = max(float(self.provisional_view_score), float(obs.view_score))
        self.provisional_num_points = int(merged.shape[0])
        self.provisional_border_touched = bool(self.provisional_border_touched and bool(obs.box_touches_border))
        self.provisional_support_kfs = sorted(set([*self.provisional_support_kfs, int(kf_idx)]))
        return True

    def maybe_promote_candidate_anchor(
        self,
        min_obs_count: int = 2,
        *,
        min_kf_support: int = 1,
        keyframes=None,
        allow_border_touched: bool = False,
        border_min_obs_count: Optional[int] = None,
        border_min_points: int = 0,
        border_min_view_score: float = 0.0,
        min_points: int = 0,
        min_view_score: float = 0.0,
        center_dev_max: float = 0.2,
        centroid_dev_max: float = 0.2,
        created_frame: Optional[int] = None,
    ) -> bool:
        if self.disable_3d_fusion:
            self._record_promote_reject("disabled_by_policy")
            return False
        if self.stable_geom_ready:
            return False
        if self.obs_count < int(min_obs_count):
            self._record_promote_reject("insufficient_obs_count")
            return False
        if self.candidate_anchor_kf_id is None:
            return False
        _, _, _, source_view_score, source_num_points, source_border_touched, source_support_kfs = self._pre_stable_source_stats()
        if source_border_touched:
            if not allow_border_touched:
                self._record_promote_reject("candidate_border_touched")
                return False
            required_obs = int(min_obs_count if border_min_obs_count is None else border_min_obs_count)
            if self.obs_count < required_obs:
                self._record_promote_reject("candidate_border_touched")
                return False
            if int(source_num_points) < int(border_min_points):
                self._record_promote_reject("candidate_border_low_num_points")
                return False
            if float(source_view_score) < float(border_min_view_score):
                self._record_promote_reject("candidate_border_low_view_score")
                return False
        quality_ok, quality_reason = self._candidate_anchor_quality_ok(
            min_points=min_points,
            min_view_score=min_view_score,
            allow_border_touched=allow_border_touched,
            border_min_points=border_min_points,
            border_min_view_score=border_min_view_score,
        )
        if not quality_ok:
            self._record_promote_reject(quality_reason)
            return False
        stability_ok, stability_reason = self._candidate_anchor_stability_ok(
            min_kf_support=min_kf_support,
            keyframes=keyframes,
            center_dev_max=center_dev_max,
            centroid_dev_max=centroid_dev_max,
        )
        if not stability_ok:
            self._record_promote_reject(stability_reason)
            return False

        # Promote success: record lifecycle metadata
        is_pre_stable = self.has_provisional_anchor()
        self.last_promote_kf = int(created_frame) if created_frame is not None else int(self.candidate_anchor_kf_id)
        self.primary_anchor_created_kf = int(self.candidate_anchor_kf_id)
        self.primary_anchor_source = "pre_stable" if is_pre_stable else "candidate"
        self.primary_anchor_promoted_from_pre_stable = is_pre_stable
        self.last_promote_reject_reason = None

        self.anchor_kf_id = self.candidate_anchor_kf_id
        self.primary_anchor_created_frame = int(created_frame) if created_frame is not None else int(self.candidate_anchor_kf_id)
        self.primary_anchor_view_score = float(source_view_score)
        self.primary_anchor_num_points = int(source_num_points)
        self.primary_anchor_border_touched = bool(source_border_touched)
        if self.has_provisional_anchor():
            # Compatibility mirror of the fused canonical local geometry.
            self.points_anchor = self.provisional_points_frame
            self.centroid_anchor = self.provisional_centroid_frame
            self.bbox_anchor = self.provisional_bbox_frame
            provisional_cache = self._points_cache_dict().get("provisional_points_frame")
            self._set_fused_local_geometry(
                points_local=self.provisional_points_frame,
                centroid_local=self.provisional_centroid_frame,
                bbox_local=self.provisional_bbox_frame,
                support_kfs=self.provisional_support_kfs,
                quality_score=self._observation_quality_score(
                    num_points=int(self.provisional_num_points),
                    view_score=float(self.provisional_view_score),
                    border_touched=bool(self.provisional_border_touched),
                ),
                last_fused_kf=max(self.provisional_support_kfs) if self.provisional_support_kfs else None,
                points_tensor=provisional_cache,
            )
        else:
            # Compatibility mirror of the fused canonical local geometry.
            self.points_anchor = self.candidate_points_frame
            self.centroid_anchor = self.candidate_centroid_frame
            self.bbox_anchor = self.candidate_bbox_frame
            candidate_cache = self._points_cache_dict().get("candidate_points_frame")
            self._set_fused_local_geometry(
                points_local=self.candidate_points_frame,
                centroid_local=self.candidate_centroid_frame,
                bbox_local=self.candidate_bbox_frame,
                support_kfs=source_support_kfs,
                quality_score=self._observation_quality_score(
                    num_points=int(self.candidate_num_points),
                    view_score=float(self.candidate_view_score),
                    border_touched=bool(self.candidate_border_touched),
                ),
                last_fused_kf=max(source_support_kfs) if source_support_kfs else None,
                points_tensor=candidate_cache,
            )
        self.stable_geom_ready = True
        self.primary_anchor_support_kfs = list(self.anchor_support_kfs)
        self.last_fusion_reject_reason = None
        self.last_pre_stable_reject_reason = None
        return True

    def fuse_anchor_observation(
        self,
        kf_idx: int,
        keyframes,
        obs,
        *,
        max_points: int = 512,
        min_points: int = 8,
        min_view_score: float = 0.0,
        allow_border: bool = False,
        border_min_points: int = 0,
        border_min_view_score: float = 0.0,
    ) -> bool:
        if self.disable_3d_fusion:
            self._record_fusion_reject("disabled_by_policy")
            return False
        if not self.stable_geom_ready or self.anchor_kf_id is None:
            return False
        support_kfs = self.fused_support_kfs or self.anchor_support_kfs
        if int(kf_idx) in support_kfs:
            return False
        anchor_frame = self._anchor_frame(keyframes, self.anchor_kf_id)
        if anchor_frame is None:
            return False
        pose = getattr(anchor_frame, "T_WC", None)
        if pose is None:
            return False
        points_world = getattr(obs, "points_world", None)
        if points_world is None or int(points_world.shape[0]) == 0:
            return False
        points_local = self._pose_inv_act(pose, points_world)
        ref_min, ref_max = self.fused_bbox_tensors(device=points_local.device, dtype=points_local.dtype)
        if ref_min is None or ref_max is None:
            ref_min, ref_max = self.anchor_bbox_tensors(device=points_local.device, dtype=points_local.dtype)
        ref_centroid = self.fused_centroid_tensor(device=points_local.device, dtype=points_local.dtype)
        if ref_centroid is None:
            ref_centroid = self.anchor_centroid_tensor(device=points_local.device, dtype=points_local.dtype)
        if ref_min is not None and ref_max is not None:
            accepted, reason, quality_score = self._anchor_fusion_accept(
                obs,
                points_local,
                ref_min=ref_min,
                ref_max=ref_max,
                ref_centroid=ref_centroid,
                min_points=min_points,
                min_view_score=min_view_score,
                allow_border=allow_border,
                border_min_points=border_min_points,
                border_min_view_score=border_min_view_score,
            )
            if not accepted:
                self._record_fusion_reject(reason)
                return False
        else:
            quality_score = self._observation_quality_score(
                num_points=int(getattr(obs, "num_points", 0) or int(points_local.shape[0])),
                view_score=float(getattr(obs, "view_score", 0.0) or 0.0),
                border_touched=bool(getattr(obs, "box_touches_border", False)),
            )
        if not self._anchor_fusion_gate(points_local):
            self._record_fusion_reject("anchor_fusion_gate")
            return False
        anchor_points = self.fused_points_tensor(device=points_local.device, dtype=points_local.dtype)
        if anchor_points is None:
            anchor_points = self.anchor_points_tensor(device=points_local.device, dtype=points_local.dtype)
        if anchor_points is None:
            merged = points_local
        else:
            merged = torch.cat([anchor_points, points_local], dim=0)
        merged = self._downsample_points_local(merged, max_points=max_points)
        bmin, bmax = bbox_from_points(merged)
        merged_centroid = merged.mean(dim=0)
        self._set_fused_local_geometry(
            points_local=tensor_to_list(merged),
            centroid_local=tensor_to_list(merged_centroid),
            bbox_local={
                "min": tensor_to_list(bmin),
                "max": tensor_to_list(bmax),
            },
            support_kfs=[*support_kfs, int(kf_idx)],
            quality_score=max(float(self.fused_quality_score), float(quality_score)),
            last_fused_kf=int(kf_idx),
            points_tensor=merged,
        )
        self._sync_anchor_snapshot_from_fused()
        self.num_fusion_accepts += 1
        self.last_accept_quality_score = float(quality_score)
        self.last_fusion_reject_reason = None
        return True

    # ------------------------------------------------------------------
    # Auxiliary anchor helpers
    # ------------------------------------------------------------------
    # Auxiliary anchors are a side cache for multi-view coverage.  They
    # do NOT replace the single primary anchor + fused canonical local
    # geometry which remain the source of truth in this stage.

    def _score_auxiliary_anchor_candidate(self, obs) -> float:
        """Score an observation for auxiliary anchor candidacy (0..1)."""
        return self._observation_quality_score(
            num_points=int(getattr(obs, "num_points", 0) or 0),
            view_score=float(getattr(obs, "view_score", 0.0) or 0.0),
            border_touched=bool(getattr(obs, "box_touches_border", False)),
        )

    def _coverage_hint_from_obs(self, obs, frame=None) -> str:
        """Classify coverage using the 2D box position in the image plane.

        This avoids touching GPU-resident world tensors during live keyframe
        commit.  The hint is intentionally coarse and only used as a lightweight
        novelty signal for the auxiliary-anchor side cache.
        """
        box = getattr(obs, "box_xyxy", None)
        if box is None or len(box) != 4:
            return "border" if bool(getattr(obs, "box_touches_border", False)) else "unknown"
        try:
            if frame is not None:
                h, w = frame_hw(frame)
            else:
                h, w = None, None
        except Exception:
            h, w = None, None
        if h is None or w is None or int(h) <= 0 or int(w) <= 0:
            return "border" if bool(getattr(obs, "box_touches_border", False)) else "unknown"
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = 0.5 * (x1 + x2) / float(w)
        cy = 0.5 * (y1 + y2) / float(h)
        if cx < 0.33:
            horiz = "left"
        elif cx > 0.67:
            horiz = "right"
        else:
            horiz = "center"
        if cy < 0.33:
            vert = "top"
        elif cy > 0.67:
            vert = "bottom"
        else:
            vert = "center"
        if horiz == "center" and vert == "center":
            return "center"
        if horiz != "center" and vert != "center":
            return f"{vert}_{horiz}"
        return horiz if horiz != "center" else vert

    def _should_add_auxiliary_anchor(self, kf_idx: int, obs, keyframes) -> tuple[bool, str]:
        """Check if an observation qualifies as a new auxiliary anchor.

        Only stable tracks may collect auxiliary anchors.  The primary
        anchor frame itself is never added.  Basic quality and coverage
        novelty gates are applied.

        Returns (accept, reason) where reason is a discrete string from
        the stable bookkeeping vocabulary:

            accept reasons  : first_support, novel_coverage, under_budget,
                              beat_worst, same_kf_update
            reject reasons  : not_stable, is_primary_frame,
                              missing_geometry, low_quality,
                              duplicate_coverage
        """
        if not self.stable_geom_ready:
            return False, "not_stable"
        if kf_idx == self.anchor_kf_id:
            return False, "is_primary_frame"
        if int(getattr(obs, "num_points", 0)) < 8:
            return False, "missing_geometry"
        quality = self._score_auxiliary_anchor_candidate(obs)
        if quality < 0.3:
            return False, "low_quality"
        # First aux anchor for this track -> always accept
        if not self.auxiliary_anchors:
            return True, "first_support"
        # Allow update path for same kf
        for aux in self.auxiliary_anchors:
            if aux.get("kf_id") == kf_idx:
                return True, "same_kf_update"
        # Coverage hint novelty
        frame = self._anchor_frame(keyframes, kf_idx) if keyframes else None
        new_hint = self._coverage_hint_from_obs(obs, frame)
        existing_hints = {a.get("coverage_hint") for a in self.auxiliary_anchors}
        if new_hint != "unknown" and new_hint not in existing_hints:
            return True, "novel_coverage"
        # Same-hint observations only qualify if they are clearly better than
        # the weakest existing auxiliary entry or if budget is not yet full.
        if len(self.auxiliary_anchors) < self.aux_anchor_budget:
            return True, "under_budget"
        worst_quality = min(float(a.get("quality_score", 0.0)) for a in self.auxiliary_anchors)
        if quality > (worst_quality + 0.05):
            return True, "beat_worst"
        return False, "duplicate_coverage"

    def _add_or_update_auxiliary_anchor(self, kf_idx: int, obs, keyframes) -> bool:
        """Try to add or update an auxiliary anchor from this observation.

        Auxiliary anchors are a side cache for multi-view coverage.  They
        do NOT replace the single primary anchor + fused canonical local
        geometry which remain the source of truth.

        Bookkeeping vocabulary recorded in ``aux_anchor_update_reason_counts``:

            * accept outcomes  : first_support, novel_coverage, under_budget,
                                 beat_worst, updated_better_quality
            * skip outcomes    : not_stable, is_primary_frame,
                                 missing_geometry, low_quality,
                                 duplicate_coverage, duplicate_kf
        Budget trim events are bumped under the ``trim_lowest`` reason
        and mirrored in ``aux_anchor_total_trims``.
        """
        self.aux_anchor_total_attempts += 1
        self.aux_anchor_last_attempt_kf = int(kf_idx)

        accept, reason = self._should_add_auxiliary_anchor(kf_idx, obs, keyframes)
        if not accept:
            self.aux_anchor_last_attempt_result = f"skipped:{reason}"
            self.aux_anchor_update_reason_counts[reason] = (
                self.aux_anchor_update_reason_counts.get(reason, 0) + 1
            )
            return False

        quality = self._score_auxiliary_anchor_candidate(obs)
        frame = self._anchor_frame(keyframes, kf_idx) if keyframes else None
        hint = self._coverage_hint_from_obs(obs, frame)
        pts_frame = getattr(obs, "points_frame", None)
        centroid_frame = getattr(obs, "centroid_frame", None)
        bbox3d = getattr(obs, "bbox3d_frame", None)
        src_points = self._safe_tensor_to_list(pts_frame)
        src_centroid = self._safe_tensor_to_list(centroid_frame)
        src_bbox = None
        if bbox3d is not None:
            src_bbox = {
                "min": self._safe_tensor_to_list(bbox3d[0]),
                "max": self._safe_tensor_to_list(bbox3d[1]),
            }
            if src_bbox["min"] is None or src_bbox["max"] is None:
                src_bbox = None
        # Update existing entry for same kf_id
        for aux in self.auxiliary_anchors:
            if aux.get("kf_id") == kf_idx:
                if quality > float(aux.get("quality_score", 0.0)):
                    aux["view_score"] = float(getattr(obs, "view_score", 0.0) or 0.0)
                    aux["num_points"] = int(getattr(obs, "num_points", 0) or 0)
                    aux["border_touched"] = bool(getattr(obs, "box_touches_border", False))
                    aux["coverage_hint"] = hint
                    aux["quality_score"] = quality
                    aux["source_points_frame"] = src_points
                    aux["source_centroid_frame"] = src_centroid
                    aux["source_bbox_frame"] = src_bbox
                    self.aux_anchor_last_attempt_result = "updated"
                    self.aux_anchor_total_updates += 1
                    self.aux_anchor_update_reason_counts["updated_better_quality"] = (
                        self.aux_anchor_update_reason_counts.get("updated_better_quality", 0) + 1
                    )
                else:
                    self.aux_anchor_last_attempt_result = "skipped:duplicate_kf"
                    self.aux_anchor_update_reason_counts["duplicate_kf"] = (
                        self.aux_anchor_update_reason_counts.get("duplicate_kf", 0) + 1
                    )
                return True
        # New auxiliary anchor entry
        count_before = len(self.auxiliary_anchors)
        self.auxiliary_anchors.append({
            "kf_id": int(kf_idx),
            "view_score": float(getattr(obs, "view_score", 0.0) or 0.0),
            "num_points": int(getattr(obs, "num_points", 0) or 0),
            "border_touched": bool(getattr(obs, "box_touches_border", False)),
            "coverage_hint": hint,
            "quality_score": quality,
            "source_points_frame": src_points,
            "source_centroid_frame": src_centroid,
            "source_bbox_frame": src_bbox,
        })
        if int(kf_idx) not in self.aux_anchor_created_kfs:
            self.aux_anchor_created_kfs.append(int(kf_idx))
        self._trim_auxiliary_anchors()
        trimmed = (count_before + 1) - len(self.auxiliary_anchors)
        if trimmed > 0:
            self.aux_anchor_total_trims += trimmed
            self.aux_anchor_update_reason_counts["trim_lowest"] = (
                self.aux_anchor_update_reason_counts.get("trim_lowest", 0) + trimmed
            )
        self.aux_anchor_last_attempt_result = "added"
        self.aux_anchor_total_adds += 1
        self.aux_anchor_update_reason_counts[reason] = (
            self.aux_anchor_update_reason_counts.get(reason, 0) + 1
        )
        return True

    def _trim_auxiliary_anchors(self) -> None:
        """Remove lowest-quality auxiliary anchors if over budget."""
        if len(self.auxiliary_anchors) <= self.aux_anchor_budget:
            return
        self.auxiliary_anchors.sort(key=lambda a: float(a.get("quality_score", 0.0)), reverse=True)
        self.auxiliary_anchors = self.auxiliary_anchors[:self.aux_anchor_budget]
        self.aux_anchor_created_kfs = [int(a["kf_id"]) for a in self.auxiliary_anchors]

    # ------------------------------------------------------------------
    # Multi-anchor weak association assist (first-stage helpers)
    # ------------------------------------------------------------------
    # Auxiliary anchors are *never* a source of truth.  They may however
    # be used to provide a tiny *weak* support bonus to the association
    # score when the primary anchor is visibly weak (border-touched,
    # few points, or low view score).  The bonus is capped tight so it
    # cannot override the primary geometry.

    def _primary_anchor_is_weak(self) -> bool:
        """Heuristic: current primary anchor is a weaker observation.

        Used only to decide whether multi-anchor assist is worth applying.
        Never used to change primary source of truth.
        """
        if not self.stable_geom_ready:
            return False
        if bool(self.primary_anchor_border_touched):
            return True
        if int(self.primary_anchor_num_points) < 64:
            return True
        if float(self.primary_anchor_view_score) < 0.08:
            return True
        return False

    def _score_aux_anchor_assoc_support(self, aux_anchor: dict, obs) -> tuple[float, str]:
        """Return a small weak-support score for association (0..1) plus reason.

        The score is intentionally small — it is meant to be used as a
        bonus/fallback in the association main chain, never as a
        standalone decision signal.
        """
        if aux_anchor is None:
            return 0.0, "no_aux"
        quality = float(aux_anchor.get("quality_score", 0.0))
        if quality <= 0.0:
            return 0.0, "zero_quality"
        num_points = int(aux_anchor.get("num_points", 0) or 0)
        view_score = float(aux_anchor.get("view_score", 0.0) or 0.0)
        # quality_factor dominates
        quality_factor = max(0.0, min(1.0, quality))
        # small boost from strong observations
        point_factor = min(1.0, float(num_points) / 200.0)
        view_factor = min(1.0, view_score / 0.3)
        agreement = 1.0
        # If obs and aux share coarse coverage hint, the support is a bit stronger.
        obs_border = bool(getattr(obs, "box_touches_border", False))
        aux_border = bool(aux_anchor.get("border_touched", True))
        if obs_border != aux_border and obs_border:
            agreement *= 0.85
        # Geometric agreement: compare aux source centroid to obs centroid_frame
        # in the (rough) local frame.  Both may live in different anchor frames;
        # we only use the magnitude as a soft sanity check.
        try:
            aux_centroid = aux_anchor.get("source_centroid_frame")
            obs_centroid = getattr(obs, "centroid_frame", None)
            if aux_centroid is not None and obs_centroid is not None:
                ac = torch.as_tensor(aux_centroid, dtype=torch.float32).reshape(-1)
                oc = obs_centroid
                if torch.is_tensor(oc):
                    oc_cpu = oc.detach().cpu().reshape(-1)
                else:
                    oc_cpu = torch.as_tensor(oc, dtype=torch.float32).reshape(-1)
                if ac.numel() == 3 and oc_cpu.numel() == 3:
                    depth_diff = abs(float(ac[2].item()) - float(oc_cpu[2].item()))
                    if depth_diff > 1.5:
                        agreement *= 0.75
        except Exception:
            pass
        support = 0.55 * quality_factor + 0.25 * point_factor + 0.20 * view_factor
        support *= agreement
        support = max(0.0, min(1.0, support))
        reason = "aux_support"
        if not aux_border and not obs_border:
            reason = "aux_support_centered"
        return float(support), reason

    def _best_aux_assoc_support(self, obs) -> tuple[float, Optional[int], str]:
        """Return (best_support_score, best_kf_id, reason) over aux anchors.

        Only returns a non-zero score when the primary is visibly weak —
        otherwise this helper is a no-op by design.
        """
        if not self.stable_geom_ready or not self.auxiliary_anchors:
            return 0.0, None, "no_aux_or_not_stable"
        if not self._primary_anchor_is_weak():
            return 0.0, None, "primary_strong"
        best_score = 0.0
        best_kf: Optional[int] = None
        best_reason = "no_viable_aux"
        for aux in self.auxiliary_anchors:
            score, reason = self._score_aux_anchor_assoc_support(aux, obs)
            if score > best_score:
                best_score = score
                best_kf = int(aux.get("kf_id")) if aux.get("kf_id") is not None else None
                best_reason = reason
        return float(best_score), best_kf, best_reason

    # ------------------------------------------------------------------
    # Conservative re-anchor interface
    # ------------------------------------------------------------------
    # Re-anchor allows upgrading the primary anchor to a better auxiliary
    # anchor.  It is ENABLED by default (enable_reanchor=True) and is
    # expected to fire only as a low-frequency corrective mechanism.
    # Global kill-switch: FG_DISABLE_REANCHOR=1.
    # Per-track override: track.enable_reanchor=False.
    # When triggered, the fused canonical local geometry is re-expressed
    # in the new primary frame via world-coordinate intermediate.

    def _score_reanchor_candidate(self, aux_anchor: dict, keyframes) -> float:
        """Composite score for a re-anchor candidate auxiliary anchor (0..1).

        Combines fusion quality, view score, point count, and a border
        penalty.  A strong, centrally-framed observation with many points
        scores near 1.0; a weak or border-touched observation scores low.
        """
        quality = float(aux_anchor.get("quality_score", 0.0))
        border_penalty = 0.15 if bool(aux_anchor.get("border_touched", True)) else 0.0
        view_score = float(aux_anchor.get("view_score", 0.0) or 0.0)
        view_bonus = 0.10 * min(1.0, view_score / 0.30)
        num_points = int(aux_anchor.get("num_points", 0) or 0)
        points_bonus = 0.10 * min(1.0, num_points / 200.0)
        return max(0.0, min(1.0, quality - border_penalty + view_bonus + points_bonus))

    def _primary_composite_score(self) -> float:
        """Composite score for the current primary anchor (0..1).

        Same formula as :meth:`_score_reanchor_candidate` so the two are
        directly comparable when deciding whether to upgrade.
        """
        base = self._observation_quality_score(
            num_points=int(self.primary_anchor_num_points),
            view_score=float(self.primary_anchor_view_score),
            border_touched=bool(self.primary_anchor_border_touched),
        )
        border_penalty = 0.15 if bool(self.primary_anchor_border_touched) else 0.0
        view_bonus = 0.10 * min(1.0, float(self.primary_anchor_view_score) / 0.30)
        points_bonus = 0.10 * min(1.0, float(self.primary_anchor_num_points) / 200.0)
        return max(0.0, min(1.0, base - border_penalty + view_bonus + points_bonus))

    def _best_reanchor_candidate(self, keyframes) -> Optional[dict]:
        """Return the highest-scoring auxiliary anchor suitable for re-anchor."""
        if not self.auxiliary_anchors:
            return None
        scored = [(self._score_reanchor_candidate(a, keyframes), a) for a in self.auxiliary_anchors]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1] if scored[0][0] > 0.0 else None

    def _can_upgrade_primary_anchor(self, aux_anchor: dict, keyframes, *, current_kf: Optional[int] = None) -> tuple[bool, str]:
        """Conservative, low-frequency re-anchor gate.

        Designed as a *corrective* mechanism — it should fire rarely.
        All conditions must hold:

          - ``enable_reanchor`` is True
          - track is stable with non-empty fused geometry
          - at least 2 fused support kfs (track has been stable a while)
          - cooldown: not too recently re-anchored
          - aux candidate is not border-touched
          - aux composite score exceeds primary composite by >= 0.2

        Returns ``(can_upgrade, reason)``.  Reasons are discrete, stable
        strings suitable for bookkeeping / snapshot export.
        """
        if not self.enable_reanchor:
            return False, "disabled"
        if not self.stable_geom_ready or self.anchor_kf_id is None:
            return False, "not_stable"
        if self.points_fused_local is None and self.centroid_fused_local is None:
            return False, "no_fused_geometry"
        if len(self.fused_support_kfs) < 2:
            return False, "insufficient_support"
        # Cooldown: don't re-anchor too frequently
        if self.last_reanchor_kf is not None and current_kf is not None:
            if (int(current_kf) - int(self.last_reanchor_kf)) < self.reanchor_min_kf_interval:
                return False, "cooldown"
        if bool(aux_anchor.get("border_touched", True)):
            return False, "border_touched"
        aux_score = self._score_reanchor_candidate(aux_anchor, keyframes)
        primary_score = self._primary_composite_score()
        if aux_score < primary_score + 0.2:
            return False, "not_significantly_better"
        return True, "approved"

    def _reexpress_fused_local_to_new_primary(self, new_anchor_kf_id: int, keyframes) -> bool:
        """Transform fused local geometry from current primary frame to new primary.

        Steps:
          1. points_fused_local → world  (via old primary T_WC)
          2. world → new primary local   (via new primary T_WC inverse)
          3. Rewrite anchor_kf_id, fused geometry, and re-anchor metadata

        Single-primary remains the source of truth after this operation;
        only the canonical frame changes.
        """
        if not self.stable_geom_ready or self.anchor_kf_id is None:
            return False
        if self.points_fused_local is None and self.centroid_fused_local is None and self.bbox_fused_local is None:
            return False
        old_frame = self._anchor_frame(keyframes, self.anchor_kf_id)
        new_frame = self._anchor_frame(keyframes, new_anchor_kf_id)
        if old_frame is None or new_frame is None:
            return False
        old_pose = getattr(old_frame, "T_WC", None)
        new_pose = getattr(new_frame, "T_WC", None)
        if old_pose is None or new_pose is None:
            return False

        # Determine the device of the pose (lietorch or tensor)
        pose_device = None
        if hasattr(old_pose, "data") and torch.is_tensor(old_pose.data):
            pose_device = old_pose.data.device
        elif torch.is_tensor(old_pose):
            pose_device = old_pose.device

        def _to_pose_device(t: torch.Tensor) -> torch.Tensor:
            if pose_device is not None and t.device != pose_device:
                return t.to(pose_device)
            return t

        # Transform points
        new_points = None
        if self.points_fused_local is not None:
            pts = _to_pose_device(torch.as_tensor(self.points_fused_local, dtype=torch.float32))
            pts_world = self._pose_act(old_pose, pts)
            new_local = self._pose_inv_act(new_pose, pts_world)
            new_points = tensor_to_list(new_local.detach().cpu())

        # Transform centroid
        new_centroid = None
        if self.centroid_fused_local is not None:
            c = _to_pose_device(torch.as_tensor(self.centroid_fused_local, dtype=torch.float32).reshape(1, 3))
            c_world = self._pose_act(old_pose, c)
            c_new = self._pose_inv_act(new_pose, c_world)
            new_centroid = tensor_to_list(c_new[0].detach().cpu())

        # Transform bbox via 8-corner round-trip
        new_bbox = None
        if self.bbox_fused_local is not None:
            bmin = torch.as_tensor(self.bbox_fused_local.get("min", []), dtype=torch.float32)
            bmax = torch.as_tensor(self.bbox_fused_local.get("max", []), dtype=torch.float32)
            if bmin.numel() == 3 and bmax.numel() == 3:
                corners = _to_pose_device(self._bbox_corners(bmin, bmax))
                corners_world = self._pose_act(old_pose, corners)
                corners_new = self._pose_inv_act(new_pose, corners_world)
                nb_min, nb_max = bbox_from_points(corners_new.detach().cpu())
                new_bbox = {"min": tensor_to_list(nb_min), "max": tensor_to_list(nb_max)}

        # Commit new primary
        self.anchor_kf_id = new_anchor_kf_id
        self._set_fused_local_geometry(
            points_local=new_points,
            centroid_local=new_centroid,
            bbox_local=new_bbox,
            support_kfs=self.fused_support_kfs,
            quality_score=self.fused_quality_score,
            last_fused_kf=self.last_fused_kf,
        )
        self._sync_anchor_snapshot_from_fused()
        self.primary_anchor_created_kf = new_anchor_kf_id
        self.primary_anchor_source = "reanchor"
        self.num_reanchors += 1
        self.last_reanchor_kf = new_anchor_kf_id
        return True

    def _try_runtime_reanchor(self, kf_idx: int, keyframes) -> bool:
        """Main-flow entry point: attempt a conservative runtime re-anchor.

        Called from commit_keyframe after auxiliary anchor update.
        Respects enable_reanchor flag, cooldown, quality gates.
        Records bookkeeping for reject/accept.
        """
        if not self.enable_reanchor:
            self.last_reanchor_reject_reason = "disabled"
            self._record_reanchor_reject("disabled")
            return False
        if not self.auxiliary_anchors:
            self.last_reanchor_reject_reason = "no_aux_anchors"
            self._record_reanchor_reject("no_aux_anchors")
            return False
        best = self._best_reanchor_candidate(keyframes)
        if best is None:
            self.last_reanchor_reject_reason = "no_viable_candidate"
            self._record_reanchor_reject("no_viable_candidate")
            return False
        can_upgrade, reason = self._can_upgrade_primary_anchor(best, keyframes, current_kf=kf_idx)
        if not can_upgrade:
            self.last_reanchor_reject_reason = reason
            self._record_reanchor_reject(reason)
            return False
        success = self._reexpress_fused_local_to_new_primary(best["kf_id"], keyframes)
        if success:
            self.last_reanchor_reject_reason = None
        else:
            self.last_reanchor_reject_reason = "reexpress_failed"
            self._record_reanchor_reject("reexpress_failed")
        return success

    def _record_reanchor_reject(self, reason: str) -> None:
        self.reanchor_reject_reason_counts[reason] = (
            self.reanchor_reject_reason_counts.get(reason, 0) + 1
        )

    def record_support_anchor_assist(
        self,
        *,
        frame_idx: Optional[int],
        best_kf: Optional[int],
        score: float,
        bonus: float,
        reason: str,
    ) -> None:
        """Record that a support-anchor assist bonus was applied to this track.

        Purely observational bookkeeping called from the online association
        main chain (``_apply_multi_anchor_assoc_assist``).  Never influences
        source of truth.  ``frame_idx`` is the current frame index at the
        time of the call; ``best_kf`` is the auxiliary anchor's kf_id.
        """
        self.support_anchor_assist_applied_count += 1
        self.last_support_anchor_assist_kf = (
            int(frame_idx) if frame_idx is not None else None
        )
        self.last_support_anchor_assist_bonus = float(bonus)
        self.last_support_anchor_assist_best_kf = (
            int(best_kf) if best_kf is not None else None
        )
        self.last_support_anchor_assist_reason = str(reason) if reason else None
        if reason:
            key = str(reason)
            self.support_anchor_assist_reason_counts[key] = (
                self.support_anchor_assist_reason_counts.get(key, 0) + 1
            )

    def commit_anchor(self, kf_idx: int, obs) -> None:
        # Deprecated compatibility path: use candidate-anchor workflow instead.
        self.consider_anchor_candidate(kf_idx, obs)
        self.maybe_promote_candidate_anchor(min_obs_count=1)

    @staticmethod
    def _tensor_points_to_list(points: Optional[torch.Tensor]) -> Optional[list[list[float]]]:
        if points is None or points.numel() == 0:
            return None
        if points.ndim == 1:
            if points.shape[0] != 3:
                return None
            points = points.reshape(1, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            return None
        return points.detach().cpu().tolist()

    def _eval_geometry_payload(self, keyframes) -> tuple[Optional[list[list[float]]], str]:
        predicted_points = None
        if keyframes:
            try:
                predicted_points = self.predict_points_world(keyframes)
            except Exception:
                predicted_points = None

        assoc_source = self.assoc_geom_source()
        points_world_eval = self._tensor_points_to_list(predicted_points)
        if points_world_eval:
            if assoc_source == "world_centroid":
                return None, "centroid_only"
            if assoc_source == "world_bbox":
                return points_world_eval, "bbox_world_est"
            return points_world_eval, assoc_source

        if self.bbox_world_est is not None:
            bmin = torch.as_tensor(self.bbox_world_est.get("min", []), dtype=torch.float32)
            bmax = torch.as_tensor(self.bbox_world_est.get("max", []), dtype=torch.float32)
            if bmin.numel() == 3 and bmax.numel() == 3:
                return self._bbox_corners(bmin, bmax).detach().cpu().tolist(), "bbox_world_est"

        if self.centroid_world_est is not None:
            centroid = torch.as_tensor(self.centroid_world_est, dtype=torch.float32)
            if centroid.numel() == 3:
                return None, "centroid_only"

        return None, "empty"

    def to_dict(self, *, keyframes=None, include_eval_geometry: bool = False) -> dict:
        points_world_eval = None
        geom_source = None
        if include_eval_geometry:
            points_world_eval, geom_source = self._eval_geometry_payload(keyframes)

        return {
            "node_id": self.node_id,
            "label": self.label,
            "role": self.role,
            "origin": self.origin,
            "anchor_kf_id": self.anchor_kf_id,
            "last_seen_frame": self.last_seen_frame,
            "last_seen_kf": self.last_seen_kf,
            "obs_count": self.obs_count,
            "visible_count": self.visible_count,
            "centroid_world_est": self.centroid_world_est,
            "bbox_world_est": self.bbox_world_est,
            "bbox_diag_world_est": self.bbox_diag_world_est,
            "best_view_frame": self.best_view_frame,
            "best_view_score": self.best_view_score,
            "best_view_crop_meta": self.best_view_crop_meta,
            "last_box_xyxy": self.last_box_xyxy,
            "last_obs_K": self.last_obs_K,
            "last_obs_hw": self.last_obs_hw,
            "last_obs_depth": self.last_obs_depth,
            "stable_geom_ready": self.stable_geom_ready,
            "points_anchor": self.points_anchor,
            "centroid_anchor": self.centroid_anchor,
            "bbox_anchor": self.bbox_anchor,
            "points_fused_local": self.points_fused_local,
            "centroid_fused_local": self.centroid_fused_local,
            "bbox_fused_local": self.bbox_fused_local,
            "fused_support_kfs": list(self.fused_support_kfs),
            "fused_num_points": int(self.fused_num_points),
            "fused_geom_ready": self.fused_geom_ready,
            "candidate_support_kfs": list(self.candidate_support_kfs),
            "candidate_anchor_kf_id": self.candidate_anchor_kf_id,
            "candidate_points_frame": self.candidate_points_frame,
            "candidate_centroid_frame": self.candidate_centroid_frame,
            "candidate_bbox_frame": self.candidate_bbox_frame,
            "candidate_view_score": self.candidate_view_score,
            "candidate_num_points": self.candidate_num_points,
            "candidate_border_touched": self.candidate_border_touched,
            "candidate_box_touches_border": self.candidate_box_touches_border,
            "provisional_points_frame": self.provisional_points_frame,
            "provisional_centroid_frame": self.provisional_centroid_frame,
            "provisional_bbox_frame": self.provisional_bbox_frame,
            "provisional_view_score": self.provisional_view_score,
            "provisional_num_points": self.provisional_num_points,
            "provisional_border_touched": self.provisional_border_touched,
            "provisional_support_kfs": list(self.provisional_support_kfs),
            "anchor_support_kfs": list(self.anchor_support_kfs),
            "primary_anchor_view_score": float(self.primary_anchor_view_score),
            "primary_anchor_num_points": int(self.primary_anchor_num_points),
            "primary_anchor_border_touched": bool(self.primary_anchor_border_touched),
            "primary_anchor_created_frame": self.primary_anchor_created_frame,
            "primary_anchor_support_kfs": list(self.primary_anchor_support_kfs),
            "fused_quality_score": float(self.fused_quality_score),
            "last_fused_kf": self.last_fused_kf,
            "num_fusion_accepts": int(self.num_fusion_accepts),
            "num_fusion_rejects": int(self.num_fusion_rejects),
            "last_fusion_reject_reason": self.last_fusion_reject_reason,
            "last_pre_stable_reject_reason": self.last_pre_stable_reject_reason,
            "num_promote_rejects": int(self.num_promote_rejects),
            "last_promote_reject_reason": self.last_promote_reject_reason,
            "promote_reject_reason_counts": dict(self.promote_reject_reason_counts),
            "last_promote_kf": self.last_promote_kf,
            "primary_anchor_created_kf": self.primary_anchor_created_kf,
            "primary_anchor_source": self.primary_anchor_source,
            "primary_anchor_promoted_from_pre_stable": bool(self.primary_anchor_promoted_from_pre_stable),
            "fusion_reject_reason_counts": dict(self.fusion_reject_reason_counts),
            "pre_stable_reject_reason_counts": dict(self.pre_stable_reject_reason_counts),
            "last_accept_quality_score": float(self.last_accept_quality_score),
            "last_pre_stable_accept_kf": self.last_pre_stable_accept_kf,
            "auxiliary_anchors": list(self.auxiliary_anchors),
            "aux_anchor_count": len(self.auxiliary_anchors),
            "aux_anchor_budget": int(self.aux_anchor_budget),
            "aux_anchor_created_kfs": list(self.aux_anchor_created_kfs),
            "aux_anchor_last_attempt_kf": self.aux_anchor_last_attempt_kf,
            "aux_anchor_last_attempt_result": self.aux_anchor_last_attempt_result,
            "aux_anchor_update_reason_counts": dict(self.aux_anchor_update_reason_counts),
            "aux_anchor_total_attempts": int(self.aux_anchor_total_attempts),
            "aux_anchor_total_adds": int(self.aux_anchor_total_adds),
            "aux_anchor_total_updates": int(self.aux_anchor_total_updates),
            "aux_anchor_total_trims": int(self.aux_anchor_total_trims),
            "enable_reanchor": bool(self.enable_reanchor),
            "num_reanchors": int(self.num_reanchors),
            "last_reanchor_kf": self.last_reanchor_kf,
            "last_reanchor_reject_reason": self.last_reanchor_reject_reason,
            "reanchor_reject_reason_counts": dict(self.reanchor_reject_reason_counts),
            "reanchor_min_kf_interval": int(self.reanchor_min_kf_interval),
            "support_anchor_assist_applied_count": int(self.support_anchor_assist_applied_count),
            "last_support_anchor_assist_kf": self.last_support_anchor_assist_kf,
            "last_support_anchor_assist_bonus": float(self.last_support_anchor_assist_bonus),
            "last_support_anchor_assist_best_kf": self.last_support_anchor_assist_best_kf,
            "last_support_anchor_assist_reason": self.last_support_anchor_assist_reason,
            "support_anchor_assist_reason_counts": dict(self.support_anchor_assist_reason_counts),
            "label_counts": dict(self.label_counts),
            "display_label_ratio": self.display_label_ratio,
            "semantic_subtype": self.semantic_subtype,
            "semantic_subtype_scores": {str(k): float(v) for k, v in self.semantic_subtype_scores.items()},
            "semantic_subtype_ratio": float(self.semantic_subtype_ratio),
            "plane_normal_world_est": self.plane_normal_world_est,
            "plane_normal_quality": float(self.plane_normal_quality),
            "plane_normal_source": self.plane_normal_source,
            "plane_normal_kf_id": self.plane_normal_kf_id,
            "plane_normal_version": int(self.plane_normal_version),
            "plane_normal_num_points": int(self.plane_normal_num_points),
            "plane_normal_eigenvalues": self.plane_normal_eigenvalues,
            "plane_normal_last_update_reason": self.plane_normal_last_update_reason,
            "cabinet_group_id": self.cabinet_group_id,
            "cabinet_seeded_count": self.cabinet_seeded_count,
            "cabinet_member_count": self.cabinet_member_count,
            "observed_frames": list(self.observed_frames),
            "first_seen_frame": self.first_seen_frame,
            "recent_observations": list(self.recent_observations),
            "best_observations": list(self.best_observations),
            "points_world_eval": points_world_eval,
            "geom_source": geom_source,
        }
