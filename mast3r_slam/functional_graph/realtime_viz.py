from __future__ import annotations

import time
import os
from typing import Any

import numpy as np

from mast3r_slam.functional_graph.graph_commit import PersistentEdge, PersistentFunctionalGraph
from mast3r_slam.functional_graph.ply_overlay import (
    EDGE_COLORS,
    NODE_COLORS,
    _iter_local_edges,
    _iter_remote_edges,
    _resolve_node_world_position_with_source,
)
from mast3r_slam.functional_graph.types import normalize_label


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _rgba_u8(color: np.ndarray, alpha: float = 1.0) -> list[float]:
    rgb = np.asarray(color, dtype=np.float32).reshape(3) / 255.0
    return [float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha)]


def _track_label(track: Any) -> str:
    for name in ("canonical_label", "label", "object_label", "name"):
        value = getattr(track, name, None)
        if value:
            return str(value)
    return ""


def _track_role(track: Any) -> str:
    role = getattr(track, "role", "")
    if role:
        role_s = str(role).strip()
        role_map = {
            "object": "O",
            "obj": "O",
            "carrier": "C",
            "functional_carrier": "C",
            "unit": "U",
            "interactive_unit": "U",
        }
        return role_map.get(role_s.lower(), role_s.upper() if role_s.upper() in NODE_COLORS else role_s)
    node_id = str(getattr(track, "node_id", "") or "")
    if node_id and "-" in node_id:
        prefix = node_id.split("-", 1)[0].upper()
        if prefix in NODE_COLORS:
            return prefix
    return ""


def _node_color_rgba(role: str, *, simplified_colors: bool) -> list[float]:
    if simplified_colors:
        if role == "U":
            return [0.0, 0.25, 1.0, 1.0]
        if role in {"O", "C"}:
            return [1.0, 0.0, 0.0, 1.0]
    color = NODE_COLORS.get(role, np.array([210, 210, 210], dtype=np.uint8))
    return _rgba_u8(color)


def _edge_color_rgba(edge, *, kind: str, simplified_colors: bool) -> list[float]:
    status = str(getattr(edge, "status", "committed") or "committed").strip().lower()
    if simplified_colors:
        return [0.0, 1.0, 0.0, 0.85 if status == "tentative" else 1.0]
    edge_type = str(getattr(edge, "edge_type", "") or "")
    base_color = EDGE_COLORS.get(edge_type, EDGE_COLORS["remote"])
    if status == "tentative":
        if edge_type == "remote" or kind == "remote":
            return [0.55, 0.82, 1.0, 0.82]
        rgb = np.clip(base_color.astype(np.float32) * 0.62 + 255.0 * 0.38, 0.0, 255.0)
        return _rgba_u8(rgb.astype(np.uint8), alpha=0.82)
    return _rgba_u8(base_color, alpha=1.0)


def _edge_strength(edge) -> tuple[int, int, float]:
    status_rank = 1 if str(getattr(edge, "status", "") or "").strip().lower() == "committed" else 0
    support_count = int(getattr(edge, "support_count", 0) or 0)
    evidence_score = float(getattr(edge, "evidence_score", 0.0) or 0.0)
    return status_rank, support_count, evidence_score


def _select_best_edge(edges: list[Any]) -> Any | None:
    if not edges:
        return None
    return max(edges, key=_edge_strength)


def _edge_strength_for_final_viz(edge) -> tuple[float, ...]:
    try:
        base = PersistentFunctionalGraph._edge_strength_tuple(edge)
    except Exception:
        status_rank, support_count, evidence_score = _edge_strength(edge)
        base = (float(status_rank), 0.0, float(support_count), 0.0, 0.0, float(evidence_score))
    return (
        *tuple(float(v) for v in base),
        float(getattr(edge, "last_seen_frame", -1) or -1),
        float(getattr(edge, "first_seen_frame", -1) or -1),
    )


def _select_best_final_edge(edges: list[Any]) -> Any | None:
    if not edges:
        return None
    return max(edges, key=_edge_strength_for_final_viz)


def _edge_key(edge) -> tuple[str, str, str]:
    return (
        str(getattr(edge, "src_node_id", "") or ""),
        str(getattr(edge, "dst_node_id", "") or ""),
        str(getattr(edge, "edge_type", "") or ""),
    )


def _policy_bool(policy, method_name: str) -> bool:
    if policy is None:
        return False
    method = getattr(policy, method_name, None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            return False
    return False


def _track_norm_label(node_tracks: dict, node_id: str) -> str:
    track = node_tracks.get(str(node_id))
    return normalize_label(_track_label(track)) if track is not None else ""


def _track_origin(node_tracks: dict, node_id: str) -> str:
    track = node_tracks.get(str(node_id))
    return str(getattr(track, "origin", "") or "") if track is not None else ""


def _track_role_by_id(node_tracks: dict, node_id: str) -> str:
    track = node_tracks.get(str(node_id))
    return _track_role(track) if track is not None else ""


def _is_cup_like(label: str) -> bool:
    norm = normalize_label(label)
    return norm in {"cup", "mug"} or norm.endswith(" cup")


def _edge_type_for_roles(parent_role: str, child_role: str) -> str | None:
    parent = str(parent_role or "").upper()
    child = str(child_role or "").upper()
    if parent == "O" and child == "C":
        return "O-C"
    if parent == "C" and child == "U":
        return "C-U"
    if parent == "O" and child == "U":
        return "O-U"
    return None


def _posterior_support_count(posterior, parent_id: str) -> int:
    method = getattr(posterior, "support_count", None)
    if callable(method):
        try:
            return int(method(parent_id))
        except Exception:
            pass
    frames = getattr(posterior, "candidate_support_frames", {}) or {}
    return len(list(frames.get(parent_id, []) or []))


def _posterior_recent_support_count(posterior, parent_id: str) -> int:
    method = getattr(posterior, "recent_support_count", None)
    if callable(method):
        try:
            return int(method(parent_id, window=3))
        except Exception:
            pass
    return 0


def _posterior_recent_switch_count(posterior) -> int:
    method = getattr(posterior, "recent_switch_count", None)
    if callable(method):
        try:
            return int(method(window=4))
        except Exception:
            pass
    return 0


def _materialize_realtime_single_frame_cup_handle_edges(
    local_edges: list[Any],
    online_state,
    *,
    kf_idx: int,
) -> tuple[list[Any], int]:
    node_tracks = getattr(online_state, "node_tracks", {}) or {}
    local_posteriors = getattr(online_state, "local_posteriors", {}) or {}
    to_remove: set[tuple[str, str, str]] = set()
    added: list[Any] = []

    local_by_child: dict[str, list[Any]] = {}
    for edge in local_edges:
        local_by_child.setdefault(str(getattr(edge, "dst_node_id", "") or ""), []).append(edge)

    for child_id, posterior in list(local_posteriors.items()):
        child_id_s = str(child_id)
        child_track = node_tracks.get(child_id_s)
        if child_track is None:
            continue
        if _track_role(child_track) != "U" or normalize_label(_track_label(child_track)) != "handle":
            continue

        candidate_scores = getattr(posterior, "candidate_parent_scores", {}) or {}
        candidate_frames = getattr(posterior, "candidate_support_frames", {}) or {}
        candidate_parent_ids = {str(pid) for pid in candidate_scores} | {str(pid) for pid in candidate_frames}
        candidates = []
        for parent_id in candidate_parent_ids:
            parent_track = node_tracks.get(parent_id)
            if parent_track is None:
                continue
            if not _is_cup_like(_track_label(parent_track)):
                continue
            edge_type = _edge_type_for_roles(_track_role(parent_track), _track_role(child_track))
            if edge_type != "O-U":
                continue
            support_frames = [int(f) for f in list(candidate_frames.get(parent_id, []) or [])]
            support_count = _posterior_support_count(posterior, parent_id)
            if support_count < 1 and not support_frames:
                continue
            candidates.append(
                (
                    int(support_count),
                    _posterior_recent_support_count(posterior, parent_id),
                    float(candidate_scores.get(parent_id, 0.0) or 0.0),
                    parent_id,
                    support_frames,
                )
            )
        if not candidates:
            continue
        candidates.sort(reverse=True)
        support_count, recent_support_count, evidence_score, parent_id, support_frames = candidates[0]

        existing_edge = _select_best_final_edge(local_by_child.get(child_id_s, []))
        if existing_edge is not None:
            existing_parent_label = _track_norm_label(node_tracks, getattr(existing_edge, "src_node_id", ""))
            if _is_cup_like(existing_parent_label):
                continue
            if str(getattr(existing_edge, "status", "") or "").strip().lower() == "committed":
                continue
            to_remove.add(_edge_key(existing_edge))

        first_seen = int(support_frames[0]) if support_frames else int(kf_idx)
        last_seen = int(support_frames[-1]) if support_frames else int(kf_idx)
        parent_label = _track_norm_label(node_tracks, parent_id) or "cup"
        child_label = _track_norm_label(node_tracks, child_id_s) or "handle"
        added.append(
            PersistentEdge(
                src_node_id=parent_id,
                dst_node_id=child_id_s,
                edge_type="O-U",
                relation_text=f"{parent_label} has {child_label}",
                committed_kf=-1,
                support_count=max(1, int(support_count)),
                evidence_score=float(evidence_score),
                status="tentative",
                first_seen_frame=first_seen,
                last_seen_frame=last_seen,
                last_update_source="realtime_single_frame_cup_handle_viz",
                margin=float(getattr(posterior, "margin", 0.0) or 0.0),
                latest_margin=float(getattr(posterior, "latest_evidence_margin", 0.0) or 0.0),
                recent_support_count=max(1, int(recent_support_count)),
                switch_count=_posterior_recent_switch_count(posterior),
                retention_policy="until_contradicted",
            )
        )

    if to_remove:
        local_edges = [edge for edge in local_edges if _edge_key(edge) not in to_remove]
    if added:
        local_edges = [*local_edges, *added]
    return local_edges, len(added)


def _materialize_realtime_merged_drawer_unit_edges(
    local_edges: list[Any],
    online_state,
    *,
    kf_idx: int,
) -> tuple[list[Any], int]:
    node_tracks = getattr(online_state, "node_tracks", {}) or {}
    local_posteriors = getattr(online_state, "local_posteriors", {}) or {}
    existing_keys = {_edge_key(edge) for edge in local_edges}
    added: list[Any] = []

    for child_id, posterior in list(local_posteriors.items()):
        child_id_s = str(child_id)
        child_track = node_tracks.get(child_id_s)
        if child_track is None:
            continue
        child_label = normalize_label(_track_label(child_track))
        if child_label not in {"handle", "knob"}:
            continue
        if _track_role(child_track) != "U":
            continue
        candidate_scores = getattr(posterior, "candidate_parent_scores", {}) or {}
        candidate_frames = getattr(posterior, "candidate_support_frames", {}) or {}
        candidate_parent_ids = {str(pid) for pid in candidate_scores} | {str(pid) for pid in candidate_frames}
        for parent_id in sorted(candidate_parent_ids):
            parent_track = node_tracks.get(parent_id)
            if parent_track is None:
                continue
            if _track_norm_label(node_tracks, parent_id) != "drawer" or _track_origin(node_tracks, parent_id) != "temp_merged_drawer":
                continue
            edge_type = _edge_type_for_roles(_track_role(parent_track), _track_role(child_track))
            if edge_type is None:
                continue
            support_frames = [int(f) for f in list(candidate_frames.get(parent_id, []) or [])]
            support_count = _posterior_support_count(posterior, parent_id)
            if support_count < 1 and not support_frames:
                continue
            key = (parent_id, child_id_s, edge_type)
            if key in existing_keys:
                continue
            first_seen = int(support_frames[0]) if support_frames else int(kf_idx)
            last_seen = int(support_frames[-1]) if support_frames else int(kf_idx)
            parent_label = _track_norm_label(node_tracks, parent_id) or "drawer"
            added.append(
                PersistentEdge(
                    src_node_id=parent_id,
                    dst_node_id=child_id_s,
                    edge_type=edge_type,
                    relation_text=f"{parent_label} has {child_label}",
                    committed_kf=-1,
                    support_count=max(1, int(support_count)),
                    evidence_score=float(candidate_scores.get(parent_id, 0.0) or 0.0),
                    status="tentative",
                    first_seen_frame=first_seen,
                    last_seen_frame=last_seen,
                    last_update_source="realtime_merged_drawer_unit_posterior",
                    margin=float(getattr(posterior, "margin", 0.0) or 0.0),
                    latest_margin=float(getattr(posterior, "latest_evidence_margin", 0.0) or 0.0),
                    recent_support_count=max(1, int(_posterior_recent_support_count(posterior, parent_id))),
                    switch_count=_posterior_recent_switch_count(posterior),
                    retention_policy="until_contradicted",
                )
            )
            existing_keys.add(key)
    if added:
        local_edges = [*local_edges, *added]
    return local_edges, len(added)


def _limit_realtime_door_drawer_unit_edges(local_edges: list[Any], node_tracks: dict) -> tuple[list[Any], int]:
    grouped: dict[str, list[Any]] = {}
    for edge in local_edges:
        parent_id = str(getattr(edge, "src_node_id", "") or "")
        parent_label = _track_norm_label(node_tracks, parent_id)
        parent_origin = _track_origin(node_tracks, parent_id)
        child_label = _track_norm_label(node_tracks, getattr(edge, "dst_node_id", ""))
        if parent_label == "drawer" and parent_origin == "temp_merged_drawer":
            continue
        if parent_label in {"door", "drawer"} and child_label in {"handle", "knob"}:
            grouped.setdefault(parent_id, []).append(edge)

    remove_keys: set[tuple[str, str, str]] = set()
    for edges in grouped.values():
        if len(edges) <= 1:
            continue
        winner = _select_best_final_edge(edges)
        for edge in edges:
            if edge is not winner:
                remove_keys.add(_edge_key(edge))
    if not remove_keys:
        return local_edges, 0
    return [edge for edge in local_edges if _edge_key(edge) not in remove_keys], len(remove_keys)


def _limit_realtime_parent_child_label_edges(local_edges: list[Any], node_tracks: dict) -> tuple[list[Any], int]:
    grouped: dict[tuple[str, str], list[Any]] = {}
    for edge in local_edges:
        parent_id = str(getattr(edge, "src_node_id", "") or "")
        child_id = str(getattr(edge, "dst_node_id", "") or "")
        parent_role = _track_role_by_id(node_tracks, parent_id)
        child_role = _track_role_by_id(node_tracks, child_id)
        if parent_role != "O" or child_role not in {"U", "C"}:
            continue
        parent_label = _track_norm_label(node_tracks, parent_id)
        if parent_label == "drawer" and _track_origin(node_tracks, parent_id) == "temp_merged_drawer":
            continue
        child_label = _track_norm_label(node_tracks, child_id)
        if not child_label:
            continue
        grouped.setdefault((parent_id, child_label), []).append(edge)

    remove_keys: set[tuple[str, str, str]] = set()
    for edges in grouped.values():
        if len(edges) <= 1:
            continue
        winner = _select_best_final_edge(edges)
        for edge in edges:
            if edge is not winner:
                remove_keys.add(_edge_key(edge))
    if not remove_keys:
        return local_edges, 0
    return [edge for edge in local_edges if _edge_key(edge) not in remove_keys], len(remove_keys)


def _limit_realtime_merged_drawer_unit_edges(
    local_edges: list[Any],
    node_tracks: dict,
    keyframes,
    kf_idx: int = -1,
) -> tuple[list[Any], int]:
    grouped: dict[str, list[Any]] = {}
    for edge in local_edges:
        parent_id = str(getattr(edge, "src_node_id", "") or "")
        child_id = str(getattr(edge, "dst_node_id", "") or "")
        parent_label = _track_norm_label(node_tracks, parent_id)
        child_label = _track_norm_label(node_tracks, child_id)
        if parent_label == "drawer" and _track_origin(node_tracks, parent_id) == "temp_merged_drawer" and child_label in {"handle", "knob"}:
            grouped.setdefault(parent_id, []).append(edge)

    def _track_max_observation_depth(track) -> float | None:
        depths: list[float] = []
        for record in list(getattr(track, "recent_observations", []) or []) + list(getattr(track, "best_observations", []) or []):
            value = record.get("centroid_frame_depth") if isinstance(record, dict) else None
            if value is None:
                continue
            try:
                depth = float(value)
                if depth == depth:
                    depths.append(depth)
            except Exception:
                continue
        if depths:
            return max(depths)
        value = getattr(track, "last_obs_depth", None)
        if value is not None:
            try:
                depth = float(value)
                if depth == depth:
                    return depth
            except Exception:
                pass
        for attr in ("candidate_centroid_frame", "provisional_centroid_frame", "centroid_anchor"):
            value = getattr(track, attr, None)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    depth = float(value[2])
                    if depth == depth:
                        return depth
                except Exception:
                    pass
        return None

    def _track_visual_side_x(track, fallback_x: float) -> float:
        records = []
        for record in list(getattr(track, "recent_observations", []) or []) + list(getattr(track, "best_observations", []) or []):
            if not isinstance(record, dict):
                continue
            bbox = record.get("bbox_xyxy")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
                frame_idx = int(record.get("frame_idx", -1) or -1)
                if cx == cx:
                    records.append((frame_idx, cx))
            except Exception:
                continue
        if records:
            records.sort(key=lambda item: item[0])
            return float(records[-1][1])
        return float(fallback_x)

    def _track_observation_frames(track) -> set[int]:
        frames: set[int] = set()
        for record in list(getattr(track, "recent_observations", []) or []) + list(getattr(track, "best_observations", []) or []):
            if not isinstance(record, dict):
                continue
            for key in ("frame_idx", "kf_idx"):
                try:
                    value = int(record.get(key, -1))
                except Exception:
                    value = -1
                if value >= 0:
                    frames.add(value)
                    break
        return frames

    remove_keys: set[tuple[str, str, str]] = set()
    for edges in grouped.values():
        positioned = []
        for edge in edges:
            child = node_tracks.get(str(getattr(edge, "dst_node_id", "") or ""))
            pos = None
            if child is not None:
                try:
                    pos, _ = _resolve_node_world_position_with_source(child, keyframes)
                except Exception:
                    pos = None
            x = float(pos[0]) if pos is not None else 0.0
            display_depth = float(pos[2]) if pos is not None and len(pos) >= 3 else None
            obs_depth = _track_max_observation_depth(child) if child is not None else None
            depth = display_depth if display_depth is not None else (obs_depth if obs_depth is not None else 0.0)
            camera_distance = float(depth)
            side_x = _track_visual_side_x(child, x) if child is not None else x
            frames = _track_observation_frames(child) if child is not None else set()
            positioned.append((side_x, camera_distance, edge, frames))
        positioned_sorted = sorted(positioned, key=lambda item: item[0])
        if len(positioned_sorted) <= 1:
            keep_edge_ids = {id(item[2]) for item in positioned_sorted}
        else:
            covisible_pairs = []
            for i, left in enumerate(positioned_sorted):
                for right in positioned_sorted[i + 1 :]:
                    shared = set(left[3]) & set(right[3])
                    if not shared:
                        continue
                    side_gap = abs(float(right[0] - left[0]))
                    depth_score = float(left[1] + right[1])
                    strength_score = _edge_strength(left[2]) + _edge_strength(right[2])
                    covisible_pairs.append((side_gap, min(float(left[1]), float(right[1])), depth_score, strength_score, left, right))
            if covisible_pairs:
                _, _, _, _, left_item, right_item = max(covisible_pairs, key=lambda item: item[:4])
                keep_edge_ids = {id(left_item[2]), id(right_item[2])}
            else:
                best_item = max(positioned_sorted, key=lambda item: (item[1], _edge_strength(item[2])))
                keep_edge_ids = {id(best_item[2])}
        for _, _, edge, _ in positioned:
            if id(edge) not in keep_edge_ids:
                remove_keys.add(_edge_key(edge))
    if not remove_keys:
        return local_edges, 0
    return [edge for edge in local_edges if _edge_key(edge) not in remove_keys], len(remove_keys)


def _edge_has_realtime_2d_support(edge: Any, online_state) -> bool:
    child_id = str(getattr(edge, "dst_node_id", "") or "")
    parent_id = str(getattr(edge, "src_node_id", "") or "")
    if not child_id or not parent_id:
        return False
    posterior = (getattr(online_state, "local_posteriors", {}) or {}).get(child_id)
    if posterior is None:
        return False
    support_frames = list(getattr(posterior, "candidate_support_frames", {}).get(parent_id, []) or [])
    if support_frames:
        return True
    try:
        if int(posterior.support_count(parent_id)) > 0:
            return True
    except Exception:
        pass
    try:
        score = float(getattr(posterior, "candidate_parent_scores", {}).get(parent_id, 0.0) or 0.0)
    except Exception:
        score = 0.0
    return score > 0.0 and int(getattr(edge, "support_count", 0) or 0) > 0


def _drop_realtime_edges_without_2d_support(local_edges: list[Any], online_state) -> tuple[list[Any], int]:
    kept = []
    removed_count = 0
    for edge in local_edges:
        if _edge_has_realtime_2d_support(edge, online_state):
            kept.append(edge)
        else:
            removed_count += 1
    return kept, removed_count


def _apply_realtime_local_edge_invariant(local_edges: list[Any]) -> tuple[list[Any], int]:
    by_child: dict[str, list[Any]] = {}
    for edge in local_edges:
        by_child.setdefault(str(getattr(edge, "dst_node_id", "") or ""), []).append(edge)

    kept: list[Any] = []
    removed_count = 0
    for edges in by_child.values():
        if len(edges) <= 1:
            kept.extend(edges)
            continue
        winner = _select_best_final_edge(edges)
        kept.append(winner)
        removed_count += len(edges) - 1
    return kept, removed_count


def _apply_realtime_final_edge_adjustments(
    local_edges: list[Any],
    online_state,
    *,
    kf_idx: int,
    keyframes=None,
) -> tuple[list[Any], dict[str, int]]:
    policy = getattr(online_state, "graph_policy", None)
    if not _policy_bool(policy, "should_apply_realtime_final_adjustments"):
        return local_edges, {}

    node_tracks = getattr(online_state, "node_tracks", {}) or {}
    summary: dict[str, int] = {}
    if _policy_bool(policy, "should_keep_single_frame_cup_handle"):
        local_edges, added_count = _materialize_realtime_single_frame_cup_handle_edges(
            local_edges,
            online_state,
            kf_idx=kf_idx,
        )
        summary["single_frame_cup_handle_added"] = int(added_count)
    if _policy_bool(policy, "should_limit_door_drawer_units"):
        local_edges, removed_count = _limit_realtime_door_drawer_unit_edges(local_edges, node_tracks)
        summary["door_drawer_unit_edges_removed"] = int(removed_count)
    if _policy_bool(policy, "should_apply_final_local_edge_invariant"):
        local_edges, removed_count = _drop_realtime_edges_without_2d_support(local_edges, online_state)
        summary["no_2d_support_edges_removed"] = int(removed_count)
    if _policy_bool(policy, "should_limit_parent_child_label_final"):
        local_edges, added_count = _materialize_realtime_merged_drawer_unit_edges(
            local_edges,
            online_state,
            kf_idx=int(kf_idx),
        )
        summary["merged_drawer_unit_posterior_added"] = int(added_count)
        local_edges, removed_count = _limit_realtime_parent_child_label_edges(local_edges, node_tracks)
        summary["parent_single_child_label_removed"] = int(removed_count)
        local_edges, removed_count = _limit_realtime_merged_drawer_unit_edges(
            local_edges,
            node_tracks,
            keyframes,
            kf_idx=int(kf_idx),
        )
        summary["merged_drawer_extra_units_removed"] = int(removed_count)
    if _policy_bool(policy, "should_apply_final_local_edge_invariant"):
        local_edges, removed_count = _apply_realtime_local_edge_invariant(local_edges)
        summary["local_edge_invariant_removed"] = int(removed_count)
    return local_edges, summary


def _formed_hierarchy_chains_from_graph(graph) -> list[tuple[str, str, str]]:
    """Return O-C-U chains already formed in the persistent functional graph."""

    chains: list[tuple[str, str, str]] = []
    hierarchy = getattr(graph, "hierarchy", None)
    for chain in list(getattr(hierarchy, "chains_uco", []) or []):
        try:
            u_node_id, c_node_id, o_node_id = chain
        except Exception:
            continue
        chains.append((str(u_node_id), str(c_node_id), str(o_node_id)))
    return chains


def build_functional_graph_viz_snapshot(
    online_state,
    keyframes,
    *,
    kf_idx: int = -1,
    simplified_colors: bool | None = None,
    collapse_ocu: bool | None = None,
    collapse_hide_c: bool | None = None,
    show_tentative_remote: bool | None = None,
) -> dict:
    """Build a small, Manager-serializable snapshot for realtime visualization."""

    if simplified_colors is None:
        simplified_colors = _env_bool("MAST3R_SLAM_VIZ_FG_SIMPLIFIED_COLORS", True)
    if collapse_ocu is None:
        collapse_ocu = _env_bool("MAST3R_SLAM_VIZ_FG_COLLAPSE_OCU", True)
    if collapse_hide_c is None:
        collapse_hide_c = _env_bool("MAST3R_SLAM_VIZ_FG_COLLAPSE_HIDE_C", True)
    if show_tentative_remote is None:
        show_tentative_remote = _env_bool("MAST3R_SLAM_VIZ_FG_SHOW_TENTATIVE_REMOTE", False)
    linked_nodes_only = _env_bool("MAST3R_SLAM_VIZ_FG_LINKED_NODES_ONLY", False)

    node_tracks = getattr(online_state, "node_tracks", {}) or {}
    graph = getattr(online_state, "graph", None)
    policy = getattr(online_state, "graph_policy", None)
    apply_realtime_final_adjustments = _policy_bool(policy, "should_apply_realtime_final_adjustments")
    realtime_final_adjustment_summary: dict[str, int] = {}

    nodes: list[dict] = []
    node_positions: dict[str, list[float]] = {}
    role_counts: dict[str, int] = {}

    for node_id, track in list(node_tracks.items()):
        node_id_s = str(node_id)
        label = _track_label(track)
        role = _track_role(track)
        origin = str(getattr(track, "origin", "") or "")
        try:
            if policy is not None and policy.should_hide_overlay_node(node_id_s, label, origin):
                continue
        except Exception:
            pass
        if apply_realtime_final_adjustments:
            try:
                if policy is not None and policy.should_drop_final_label(label):
                    continue
            except Exception:
                pass
        try:
            pos, source = _resolve_node_world_position_with_source(track, keyframes)
        except Exception:
            pos, source = None, None
        if pos is None:
            continue
        pos_np = np.asarray(pos, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(pos_np)):
            continue
        node_positions[node_id_s] = [float(pos_np[0]), float(pos_np[1]), float(pos_np[2])]
        role_counts[role or "unknown"] = role_counts.get(role or "unknown", 0) + 1
        nodes.append(
            {
                "id": node_id_s,
                "label": label,
                "role": role,
                "origin": origin,
                "pos": node_positions[node_id_s],
                "source": str(source or ""),
                "color_rgba": _node_color_rgba(role, simplified_colors=bool(simplified_colors)),
                "obs_count": int(getattr(track, "obs_count", 0) or 0),
                "visible_count": int(getattr(track, "visible_count", 0) or 0),
            }
        )

    edges: list[dict] = []
    edge_status_counts: dict[str, int] = {}
    rendered_edge_keys: set[tuple[str, str, str, str]] = set()

    def _append_edge(edge, kind: str, *, synthetic: bool = False, src_override: str | None = None, dst_override: str | None = None) -> None:
        src = str(getattr(edge, "src_node_id", "") or "")
        dst = str(getattr(edge, "dst_node_id", "") or "")
        if src_override is not None:
            src = str(src_override)
        if dst_override is not None:
            dst = str(dst_override)
        if not src or not dst or src not in node_positions or dst not in node_positions:
            return
        status = str(getattr(edge, "status", "committed") or "committed")
        if kind == "remote" and not show_tentative_remote and status.strip().lower() != "committed":
            edge_status_counts["remote_hidden_tentative"] = edge_status_counts.get("remote_hidden_tentative", 0) + 1
            return
        edge_type = str(getattr(edge, "edge_type", "") or "")
        if synthetic:
            edge_type = "O-U"
        rendered_key = (str(kind), src, dst, edge_type)
        if rendered_key in rendered_edge_keys:
            return
        rendered_edge_keys.add(rendered_key)
        edge_status_counts[f"{kind}_{status.lower()}"] = edge_status_counts.get(f"{kind}_{status.lower()}", 0) + 1
        edges.append(
            {
                "kind": kind,
                "synthetic": bool(synthetic),
                "src": src,
                "dst": dst,
                "src_pos": node_positions[src],
                "dst_pos": node_positions[dst],
                "edge_type": edge_type,
                "relation_text": str(getattr(edge, "relation_text", "") or ""),
                "status": status,
                "support_count": int(getattr(edge, "support_count", 0) or 0),
                "color_rgba": _edge_color_rgba(edge, kind=kind, simplified_colors=bool(simplified_colors)),
            }
        )

    if graph is not None:
        local_edges = list(_iter_local_edges(graph))
        local_edges, realtime_final_adjustment_summary = _apply_realtime_final_edge_adjustments(
            local_edges,
            online_state,
            kf_idx=int(kf_idx),
            keyframes=keyframes,
        )
        skip_local_keys: set[tuple[str, str, str]] = set()
        synthetic_pairs: set[tuple[str, str]] = set()
        hidden_node_ids: set[str] = set()

        if collapse_ocu:
            local_by_key: dict[tuple[str, str, str], list[Any]] = {}
            for edge in local_edges:
                key = (
                    str(getattr(edge, "src_node_id", "") or ""),
                    str(getattr(edge, "dst_node_id", "") or ""),
                    str(getattr(edge, "edge_type", "") or ""),
                )
                local_by_key.setdefault(key, []).append(edge)

            for u_node_id, c_node_id, o_node_id in _formed_hierarchy_chains_from_graph(graph):
                oc_key = (o_node_id, c_node_id, "O-C")
                cu_key = (c_node_id, u_node_id, "C-U")
                if o_node_id not in node_positions or u_node_id not in node_positions:
                    continue
                oc_edge = _select_best_edge(local_by_key.get(oc_key, []))
                cu_edge = _select_best_edge(local_by_key.get(cu_key, []))
                if oc_edge is None or cu_edge is None:
                    continue
                skip_local_keys.add(oc_key)
                skip_local_keys.add(cu_key)
                if collapse_hide_c:
                    hidden_node_ids.add(c_node_id)
                if (o_node_id, u_node_id) not in synthetic_pairs:
                    synthetic_pairs.add((o_node_id, u_node_id))
                    _append_edge(
                        cu_edge,
                        "local",
                        synthetic=True,
                        src_override=o_node_id,
                        dst_override=u_node_id,
                    )

        if hidden_node_ids:
            nodes = [node for node in nodes if str(node.get("id", "")) not in hidden_node_ids]
            for node_id in hidden_node_ids:
                node_positions.pop(node_id, None)

        for edge in local_edges:
            local_key = (
                str(getattr(edge, "src_node_id", "") or ""),
                str(getattr(edge, "dst_node_id", "") or ""),
                str(getattr(edge, "edge_type", "") or ""),
            )
            if local_key in skip_local_keys:
                continue
            _append_edge(edge, "local")
        for edge in _iter_remote_edges(graph):
            _append_edge(edge, "remote")

    linked_nodes_removed = 0
    if linked_nodes_only:
        linked_node_ids = {
            str(edge.get("src", "") or "")
            for edge in edges
            if str(edge.get("src", "") or "")
        }
        linked_node_ids.update(
            str(edge.get("dst", "") or "")
            for edge in edges
            if str(edge.get("dst", "") or "")
        )
        before_count = len(nodes)
        nodes = [node for node in nodes if str(node.get("id", "") or "") in linked_node_ids]
        linked_nodes_removed = before_count - len(nodes)

    role_counts = {}
    for node in nodes:
        role = str(node.get("role", "") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "kf_idx": int(kf_idx),
        "updated_at": float(time.time()),
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "n_nodes": int(len(nodes)),
            "n_edges": int(len(edges)),
            "n_local_edges": int(sum(1 for e in edges if e.get("kind") == "local")),
            "n_remote_edges": int(sum(1 for e in edges if e.get("kind") == "remote")),
            "role_counts": role_counts,
            "edge_status_counts": edge_status_counts,
            "simplified_colors": bool(simplified_colors),
            "collapse_ocu": bool(collapse_ocu),
            "collapse_hide_c": bool(collapse_hide_c),
            "show_tentative_remote": bool(show_tentative_remote),
            "linked_nodes_only": bool(linked_nodes_only),
            "linked_nodes_removed": int(linked_nodes_removed),
            "realtime_final_adjustments": bool(apply_realtime_final_adjustments),
            "realtime_final_adjustment_summary": realtime_final_adjustment_summary,
        },
    }


def publish_functional_graph_viz_snapshot(shared_state, online_state, keyframes, *, kf_idx: int = -1) -> None:
    if shared_state is None:
        return
    try:
        snapshot = build_functional_graph_viz_snapshot(online_state, keyframes, kf_idx=kf_idx)
    except Exception as exc:
        print(f"[FG_VIZ] failed to build snapshot at kf={kf_idx}: {type(exc).__name__}: {exc}", flush=True)
        return
    try:
        version = int(shared_state.get("version", 0)) + 1
    except Exception:
        version = 1
    snapshot["version"] = version
    shared_state["snapshot"] = snapshot
    shared_state["version"] = version
