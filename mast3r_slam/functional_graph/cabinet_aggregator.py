from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

import numpy as np

from .policy import FunctionalGraphPolicy
from .types import CabinetCarrierObservation


@dataclass
class CabinetMemberEvidence:
    node_id: str
    label: str
    seeded_by_box: bool = False
    support_frames: list[int] = field(default_factory=list)
    last_seen_frame: Optional[int] = None
    recent_boxes: list[dict] = field(default_factory=list)
    recent_support_scores: list[float] = field(default_factory=list)
    last_overlap_support: float = 0.0
    last_touch_support: float = 0.0
    last_view_score: float = 0.0


@dataclass
class CabinetTrack:
    cabinet_id: str
    member_node_ids: set[str] = field(default_factory=set)
    seeded_member_ids: set[str] = field(default_factory=set)
    pending_plane_pairs: set[str] = field(default_factory=set)
    validated_plane_pairs: set[str] = field(default_factory=set)
    failed_plane_pairs: set[str] = field(default_factory=set)
    plane_validation_status: str = "pending"
    support_frames: list[int] = field(default_factory=list)
    anchor_kf_id: Optional[int] = None
    centroid_world_est: Optional[list[float]] = None
    bbox_world_est: Optional[dict] = None
    stable_geom_ready: bool = False
    last_seen_frame: Optional[int] = None
    last_commit_kf: Optional[int] = None


class CabinetAggregator:
    """Online incremental cabinet aggregation over committed carrier tracks."""

    def __init__(self, policy: Optional[FunctionalGraphPolicy] = None) -> None:
        self.policy = policy or FunctionalGraphPolicy()
        self.cabinet_tracks: Dict[str, CabinetTrack] = {}
        self.member_to_cabinet: Dict[str, str] = {}
        self.member_evidence: Dict[str, CabinetMemberEvidence] = {}
        self._next_idx = 1
        self.last_update_debug: dict = {}
        self.last_commit_debug: dict = {}

    @staticmethod
    def _append_frame_once(frames: list[int], frame_idx: int) -> None:
        if int(frame_idx) not in frames:
            frames.append(int(frame_idx))
            frames.sort()

    @staticmethod
    def _pair_key(a: str, b: str) -> str:
        return "|".join(sorted([str(a), str(b)]))

    @staticmethod
    def _append_recent_score(scores: list[float], score: float, max_len: int = 6) -> None:
        scores.append(float(score))
        if len(scores) > max_len:
            del scores[:-max_len]

    @staticmethod
    def _boxes_touch(box_a: list[float], box_b: list[float], margin_px: float = 6.0) -> bool:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        return not (
            ax2 < bx1 - margin_px
            or bx2 < ax1 - margin_px
            or ay2 < by1 - margin_px
            or by2 < ay1 - margin_px
        )

    @classmethod
    def _box_iou(cls, box_a: Optional[list[float]], box_b: Optional[list[float]]) -> float:
        if box_a is None or box_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 1e-9:
            return 0.0
        return float(inter / union)

    def _record_recent_box(self, evidence: CabinetMemberEvidence, frame_idx: int, box_xyxy: Optional[list[float]]) -> None:
        if box_xyxy is None:
            return
        payload = {"frame_idx": int(frame_idx), "box_xyxy": [float(v) for v in box_xyxy]}
        if evidence.recent_boxes and evidence.recent_boxes[-1]["frame_idx"] == int(frame_idx):
            evidence.recent_boxes[-1] = payload
        else:
            evidence.recent_boxes.append(payload)
        if len(evidence.recent_boxes) > 4:
            del evidence.recent_boxes[:-4]

    def _new_group_id(self) -> str:
        gid = f"CAB_GROUP_{self._next_idx:04d}"
        self._next_idx += 1
        return gid

    def _ensure_group(self, member_node_id: str, frame_idx: int) -> str:
        gid = self.member_to_cabinet.get(member_node_id)
        if gid is not None and gid in self.cabinet_tracks:
            group = self.cabinet_tracks[gid]
            self._append_frame_once(group.support_frames, frame_idx)
            group.last_seen_frame = int(frame_idx)
            return gid
        gid = self._new_group_id()
        self.cabinet_tracks[gid] = CabinetTrack(
            cabinet_id=f"O_CABINET_{gid.split('_')[-1]}",
            member_node_ids={member_node_id},
            support_frames=[int(frame_idx)],
            last_seen_frame=int(frame_idx),
        )
        self.member_to_cabinet[member_node_id] = gid
        return gid

    def _attach_member(self, gid: str, payload: CabinetCarrierObservation, *, seeded: bool) -> None:
        if gid not in self.cabinet_tracks:
            return
        group = self.cabinet_tracks[gid]
        group.member_node_ids.add(payload.node_id)
        if seeded:
            group.seeded_member_ids.add(payload.node_id)
        self._append_frame_once(group.support_frames, payload.frame_idx)
        group.last_seen_frame = int(payload.frame_idx)
        self.member_to_cabinet[payload.node_id] = gid

    def _drop_group_if_invalid(self, gid: str) -> None:
        group = self.cabinet_tracks.get(gid)
        if group is None:
            return
        if group.member_node_ids and group.seeded_member_ids:
            return
        for member_id in list(group.member_node_ids):
            if self.member_to_cabinet.get(member_id) == gid:
                self.member_to_cabinet.pop(member_id, None)
        self.cabinet_tracks.pop(gid, None)

    def _merge_groups(self, dst_gid: str, src_gid: str) -> None:
        if dst_gid == src_gid or dst_gid not in self.cabinet_tracks or src_gid not in self.cabinet_tracks:
            return
        dst = self.cabinet_tracks[dst_gid]
        src = self.cabinet_tracks[src_gid]
        dst.member_node_ids.update(src.member_node_ids)
        dst.seeded_member_ids.update(src.seeded_member_ids)
        dst.pending_plane_pairs.update(src.pending_plane_pairs)
        dst.validated_plane_pairs.update(src.validated_plane_pairs)
        dst.failed_plane_pairs.update(src.failed_plane_pairs)
        dst.support_frames = sorted(set(dst.support_frames).union(src.support_frames))
        dst.last_seen_frame = max(
            int(dst.last_seen_frame or -1),
            int(src.last_seen_frame or -1),
        )
        for member_id in sorted(src.member_node_ids):
            self.member_to_cabinet[member_id] = dst_gid
        self.cabinet_tracks.pop(src_gid, None)
        self._refresh_group_plane_validation_status(dst)

    def _refresh_group_plane_validation_status(self, group: CabinetTrack) -> None:
        if group.pending_plane_pairs:
            group.plane_validation_status = "pending"
        elif group.validated_plane_pairs:
            group.plane_validation_status = "validated"
        elif group.failed_plane_pairs:
            group.plane_validation_status = "failed"
        else:
            group.plane_validation_status = "pending"

    def _record_group_pair_validation(self, group: CabinetTrack, pair_debug: dict) -> None:
        pair_ids = pair_debug.get("pair_ids") or []
        if len(pair_ids) != 2:
            return
        key = self._pair_key(pair_ids[0], pair_ids[1])
        status = str(pair_debug.get("plane_validation_status") or "")
        group.pending_plane_pairs.discard(key)
        group.validated_plane_pairs.discard(key)
        group.failed_plane_pairs.discard(key)
        if status == "pass":
            group.validated_plane_pairs.add(key)
        elif status in {"fail_angle", "fail_coplanarity"}:
            group.failed_plane_pairs.add(key)
        elif status == "missing_pending":
            group.pending_plane_pairs.add(key)
        self._refresh_group_plane_validation_status(group)

    def _member_has_validated_plane_support(self, group: CabinetTrack, member_id: str, node_tracks: Dict[str, object]) -> bool:
        if len(group.member_node_ids) <= 1:
            track = node_tracks.get(member_id)
            return bool(self._track_plane_normal_debug(track)["valid"])
        for other_id in group.member_node_ids:
            if other_id == member_id:
                continue
            if self._pair_key(member_id, other_id) in group.validated_plane_pairs:
                return True
        return False

    @staticmethod
    def _track_centroid(track) -> Optional[np.ndarray]:
        centroid = getattr(track, "centroid_world_est", None)
        if centroid is None:
            return None
        arr = np.asarray(centroid, dtype=np.float32)
        if arr.shape != (3,):
            return None
        return arr

    @staticmethod
    def _track_bbox(track) -> Optional[dict]:
        bbox = getattr(track, "bbox_world_est", None) or getattr(track, "bbox_anchor", None)
        if bbox is None:
            return None
        bmin = np.asarray(bbox.get("min") or [], dtype=np.float32)
        bmax = np.asarray(bbox.get("max") or [], dtype=np.float32)
        if bmin.shape != (3,) or bmax.shape != (3,):
            return None
        return {"min": bmin, "max": bmax}

    @staticmethod
    def _payload_or_track_centroid(payload: CabinetCarrierObservation, track) -> Optional[np.ndarray]:
        centroid_payload = getattr(payload, "centroid_world", None)
        if centroid_payload is not None:
            try:
                arr = np.asarray(centroid_payload, dtype=np.float32).reshape(-1)
                if arr.shape == (3,):
                    return arr
            except Exception:
                pass
        return CabinetAggregator._track_centroid(track)

    def _track_plane_normal_debug(self, track) -> dict:
        normal = getattr(track, "plane_normal_world_est", None)
        quality = float(getattr(track, "plane_normal_quality", 0.0) or 0.0)
        version = int(getattr(track, "plane_normal_version", 0) or 0)
        source = getattr(track, "plane_normal_source", None)
        arr = None
        if normal is not None:
            try:
                arr = np.asarray(normal, dtype=np.float32).reshape(-1)
            except Exception:
                arr = None
        valid = False
        if arr is not None and arr.shape == (3,) and np.all(np.isfinite(arr)):
            norm = float(np.linalg.norm(arr))
            if norm > 1e-6:
                arr = arr / norm
                valid = bool(quality >= float(self.policy.cabinet_min_plane_normal_quality) and version > 0)
        if arr is None or arr.shape != (3,) or not np.all(np.isfinite(arr)):
            arr_list = None
        else:
            arr_list = arr.astype(float).tolist()
        return {
            "normal": arr,
            "normal_list": arr_list,
            "quality": quality,
            "source": source,
            "version": version,
            "valid": bool(valid),
        }

    def _plane_normal_gate(
        self,
        payload_a: CabinetCarrierObservation,
        payload_b: CabinetCarrierObservation,
        track_a,
        track_b,
        *,
        front_adjacency_pass: bool,
    ) -> dict:
        normal_a = self._track_plane_normal_debug(track_a)
        normal_b = self._track_plane_normal_debug(track_b)
        plane_debug = {
            "plane_normal_a": normal_a["normal_list"],
            "plane_normal_b": normal_b["normal_list"],
            "plane_normal_quality_a": float(normal_a["quality"]),
            "plane_normal_quality_b": float(normal_b["quality"]),
            "plane_normal_source_a": normal_a["source"],
            "plane_normal_source_b": normal_b["source"],
            "plane_normal_version_a": int(normal_a["version"]),
            "plane_normal_version_b": int(normal_b["version"]),
            "plane_normal_angle_deg": None,
            "plane_normal_angle_gate_pass": False,
            "plane_normal_missing": False,
            "plane_normal_missing_fallback": False,
            "plane_offset_m": None,
            "coplanarity_gate_pass": False,
            "adjacent_before_plane_gate": bool(front_adjacency_pass),
            "plane_validation_status": "front_adjacency_fail",
            "plane_validation_is_final": True,
            "plane_validation_blocks_adjacency": True,
            "adjacent_is_pending_plane_validation": False,
        }
        if not front_adjacency_pass:
            return plane_debug

        missing = not bool(normal_a["valid"]) or not bool(normal_b["valid"])
        plane_debug["plane_normal_missing"] = bool(missing)
        if missing:
            plane_debug["plane_normal_missing_fallback"] = True
            plane_debug["plane_normal_angle_gate_pass"] = True
            plane_debug["coplanarity_gate_pass"] = True
            plane_debug["plane_validation_status"] = "missing_pending"
            plane_debug["plane_validation_is_final"] = False
            plane_debug["plane_validation_blocks_adjacency"] = False
            plane_debug["adjacent_is_pending_plane_validation"] = True
            return plane_debug

        n1 = normal_a["normal"]
        n2 = normal_b["normal"]
        cos = float(np.clip(abs(float(np.dot(n1, n2))), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos)))
        angle_pass = bool(angle_deg <= float(self.policy.cabinet_max_plane_normal_angle_deg))
        plane_debug["plane_normal_angle_deg"] = angle_deg
        plane_debug["plane_normal_angle_gate_pass"] = angle_pass
        if not angle_pass:
            plane_debug["plane_validation_status"] = "fail_angle"
            plane_debug["plane_validation_is_final"] = True
            plane_debug["plane_validation_blocks_adjacency"] = True
            return plane_debug

        c1 = self._payload_or_track_centroid(payload_a, track_a)
        c2 = self._payload_or_track_centroid(payload_b, track_b)
        if c1 is None or c2 is None:
            plane_debug["plane_validation_status"] = "missing_pending"
            plane_debug["plane_validation_is_final"] = False
            plane_debug["plane_validation_blocks_adjacency"] = False
            plane_debug["adjacent_is_pending_plane_validation"] = True
            plane_debug["plane_normal_missing_fallback"] = True
            plane_debug["coplanarity_gate_pass"] = True
            return plane_debug
        n2_aligned = n2 if float(np.dot(n1, n2)) >= 0.0 else -n2
        n_avg = n1 + n2_aligned
        n_norm = float(np.linalg.norm(n_avg))
        if n_norm <= 1e-6:
            n_avg = n1
            n_norm = float(np.linalg.norm(n_avg))
        n_avg = n_avg / max(n_norm, 1e-6)
        offset = float(abs(np.dot(c2 - c1, n_avg)))
        offset_pass = bool(offset <= float(self.policy.cabinet_max_plane_offset_m))
        plane_debug["plane_offset_m"] = offset
        plane_debug["coplanarity_gate_pass"] = offset_pass
        if offset_pass:
            plane_debug["plane_validation_status"] = "pass"
            plane_debug["plane_validation_is_final"] = True
            plane_debug["plane_validation_blocks_adjacency"] = False
        else:
            plane_debug["plane_validation_status"] = "fail_coplanarity"
            plane_debug["plane_validation_is_final"] = True
            plane_debug["plane_validation_blocks_adjacency"] = True
        return plane_debug

    def _bbox3d_from_payload_or_track(self, payload: CabinetCarrierObservation, track) -> tuple[Optional[dict], Optional[str]]:
        bbox_payload = getattr(payload, "bbox3d_world", None)
        if bbox_payload is not None:
            try:
                bmin = np.asarray(bbox_payload[0], dtype=np.float32)
                bmax = np.asarray(bbox_payload[1], dtype=np.float32)
                if bmin.shape == (3,) and bmax.shape == (3,):
                    return {"min": bmin, "max": bmax}, "payload.bbox3d_world"
            except Exception:
                pass

        bbox_world_est = getattr(track, "bbox_world_est", None)
        if bbox_world_est is not None:
            bbox = self._track_bbox(track)
            if bbox is not None and getattr(track, "bbox_world_est", None) is not None:
                return bbox, "track.bbox_world_est"

        bbox_anchor = getattr(track, "bbox_anchor", None)
        if bbox_anchor is not None:
            bbox = self._track_bbox(track)
            if bbox is not None and getattr(track, "bbox_anchor", None) is not None:
                return bbox, "track.bbox_anchor"
        return None, None

    def _bbox3d_overlap_and_gap(
        self,
        bbox_a: Optional[dict],
        bbox_b: Optional[dict],
        *,
        margin: Optional[float] = None,
    ) -> dict:
        eps = float(self.policy.cabinet_expand_bbox_intersection_eps if margin is None else margin)
        if bbox_a is None or bbox_b is None:
            return {
                "valid": False,
                "overlap_xyz": None,
                "gap_xyz": None,
                "intersection_volume": 0.0,
                "extent_a_xyz": None,
                "extent_b_xyz": None,
            }

        raw_delta = np.minimum(bbox_a["max"], bbox_b["max"]) - np.maximum(bbox_a["min"], bbox_b["min"])
        stabilized_delta = np.where(np.abs(raw_delta) <= eps, 0.0, raw_delta)
        overlap = np.maximum(stabilized_delta, 0.0)
        gap = np.maximum(-stabilized_delta, 0.0)
        return {
            "valid": True,
            "overlap_xyz": overlap.tolist(),
            "gap_xyz": gap.tolist(),
            "intersection_volume": float(np.prod(overlap)),
            "extent_a_xyz": np.maximum(bbox_a["max"] - bbox_a["min"], 0.0).tolist(),
            "extent_b_xyz": np.maximum(bbox_b["max"] - bbox_b["min"], 0.0).tolist(),
        }

    def _front_adjacency_3d(self, bbox_debug: dict) -> dict:
        if not bool(bbox_debug.get("valid")):
            return {
                "passes_depth_overlap": False,
                "passes_planar_adjacency": False,
                "adjacent": False,
            }

        overlap_xyz = bbox_debug.get("overlap_xyz") or [0.0, 0.0, 0.0]
        gap_xyz = bbox_debug.get("gap_xyz") or [0.0, 0.0, 0.0]
        extent_a_xyz = bbox_debug.get("extent_a_xyz") or [0.0, 0.0, 0.0]
        extent_b_xyz = bbox_debug.get("extent_b_xyz") or [0.0, 0.0, 0.0]
        overlap_x, overlap_y, overlap_z = [float(v) for v in overlap_xyz]
        gap_x, gap_y, gap_z = [float(v) for v in gap_xyz]
        extent_ax, extent_ay, extent_az = [float(v) for v in extent_a_xyz]
        extent_bx, extent_by, extent_bz = [float(v) for v in extent_b_xyz]

        depth_overlap_thr = min(float(self.policy.cabinet_min_depth_overlap), 0.25 * min(extent_az, extent_bz)) if min(extent_az, extent_bz) > 0.0 else 0.0
        planar_overlap_thr_x = min(float(self.policy.cabinet_min_planar_overlap), 0.25 * min(extent_ax, extent_bx)) if min(extent_ax, extent_bx) > 0.0 else 0.0
        planar_overlap_thr_y = min(float(self.policy.cabinet_min_planar_overlap), 0.25 * min(extent_ay, extent_by)) if min(extent_ay, extent_by) > 0.0 else 0.0
        planar_gap_thr = float(self.policy.cabinet_max_planar_gap)
        passes_depth_overlap = overlap_z >= depth_overlap_thr and gap_z <= float(self.policy.cabinet_expand_bbox_intersection_eps)
        if depth_overlap_thr == 0.0:
            passes_depth_overlap = gap_z <= float(self.policy.cabinet_expand_bbox_intersection_eps)
        horizontal_neighbor = overlap_y >= planar_overlap_thr_y and gap_x <= planar_gap_thr
        vertical_neighbor = overlap_x >= planar_overlap_thr_x and gap_y <= planar_gap_thr
        passes_planar_adjacency = bool(horizontal_neighbor or vertical_neighbor)
        return {
            "passes_depth_overlap": bool(passes_depth_overlap),
            "passes_planar_adjacency": bool(passes_planar_adjacency),
            "adjacent": bool(passes_depth_overlap and passes_planar_adjacency),
            "horizontal_neighbor": bool(horizontal_neighbor),
            "vertical_neighbor": bool(vertical_neighbor),
            "gap_z": gap_z,
            "depth_overlap_thr": float(depth_overlap_thr),
            "planar_overlap_thr_x": float(planar_overlap_thr_x),
            "planar_overlap_thr_y": float(planar_overlap_thr_y),
        }

    def _box_support_score(self, payload: CabinetCarrierObservation, evidence: Optional[CabinetMemberEvidence]) -> tuple[float, float, float]:
        if payload.box_xyxy is None or evidence is None or not evidence.recent_boxes:
            return 0.0, 0.0, 0.0

        best_iou = 0.0
        best_touch = 0.0
        for item in evidence.recent_boxes:
            hist_box = item.get("box_xyxy")
            if hist_box is None:
                continue
            iou = self._box_iou(payload.box_xyxy, hist_box)
            touch = 1.0 if self._boxes_touch(payload.box_xyxy, hist_box) else 0.0
            best_iou = max(best_iou, iou)
            best_touch = max(best_touch, touch)
        return max(best_iou, 0.35 * best_touch), best_iou, best_touch

    def _pair_adjacency(self, payload_a: CabinetCarrierObservation, payload_b: CabinetCarrierObservation, node_tracks: Dict[str, object]) -> dict:
        track_a = node_tracks.get(payload_a.node_id)
        track_b = node_tracks.get(payload_b.node_id)
        bbox_a, bbox_source_a = self._bbox3d_from_payload_or_track(payload_a, track_a)
        bbox_b, bbox_source_b = self._bbox3d_from_payload_or_track(payload_b, track_b)
        bbox_debug = self._bbox3d_overlap_and_gap(bbox_a, bbox_b)
        front_debug = self._front_adjacency_3d(bbox_debug)

        overlap_xyz = bbox_debug.get("overlap_xyz") or [0.0, 0.0, 0.0]
        extent_a_xyz = bbox_debug.get("extent_a_xyz") or [0.0, 0.0, 0.0]
        extent_b_xyz = bbox_debug.get("extent_b_xyz") or [0.0, 0.0, 0.0]
        intersection_volume = float(bbox_debug.get("intersection_volume", 0.0) or 0.0)
        volume_a = float(np.prod(np.maximum(np.asarray(extent_a_xyz, dtype=np.float32), 0.0)))
        volume_b = float(np.prod(np.maximum(np.asarray(extent_b_xyz, dtype=np.float32), 0.0)))
        union_volume = max(0.0, volume_a + volume_b - intersection_volume)
        bbox3d_iou = float(intersection_volume / union_volume) if union_volume > 1e-12 else 0.0
        iou_only_policy = bool(self.policy.should_use_cabinet_iou_only_aggregation())

        current_iou = self._box_iou(payload_a.box_xyxy, payload_b.box_xyxy)
        current_touch = 1.0 if payload_a.box_xyxy is not None and payload_b.box_xyxy is not None and self._boxes_touch(payload_a.box_xyxy, payload_b.box_xyxy) else 0.0

        evidence_a = self.member_evidence.get(payload_a.node_id)
        evidence_b = self.member_evidence.get(payload_b.node_id)
        recent_ab, recent_iou_ab, recent_touch_ab = self._box_support_score(payload_a, evidence_b)
        recent_ba, recent_iou_ba, recent_touch_ba = self._box_support_score(payload_b, evidence_a)

        support_score = float(bbox_debug["intersection_volume"])
        if iou_only_policy:
            adjacent = bool(bbox3d_iou > 0.0)
            plane_debug = {
                "plane_normal_a": None,
                "plane_normal_b": None,
                "plane_normal_quality_a": 0.0,
                "plane_normal_quality_b": 0.0,
                "plane_normal_source_a": None,
                "plane_normal_source_b": None,
                "plane_normal_version_a": 0,
                "plane_normal_version_b": 0,
                "plane_normal_angle_deg": None,
                "plane_normal_angle_gate_pass": True,
                "plane_normal_missing": False,
                "plane_normal_missing_fallback": False,
                "plane_offset_m": None,
                "coplanarity_gate_pass": True,
                "adjacent_before_plane_gate": bool(adjacent),
                "plane_validation_status": "pass" if adjacent else "iou_only_no_overlap",
                "plane_validation_is_final": True,
                "plane_validation_blocks_adjacency": False,
                "adjacent_is_pending_plane_validation": False,
            }
        else:
            plane_debug = self._plane_normal_gate(
                payload_a,
                payload_b,
                track_a,
                track_b,
                front_adjacency_pass=bool(front_debug["adjacent"]),
            )
            adjacent = bool(front_debug["adjacent"] and not plane_debug["plane_validation_blocks_adjacency"])
        support_ok = bool(adjacent)

        if evidence_a is not None:
            evidence_a.last_overlap_support = float(max(current_iou, recent_iou_ba))
            evidence_a.last_touch_support = float(max(current_touch, recent_touch_ba))
            self._append_recent_score(evidence_a.recent_support_scores, support_score)
        if evidence_b is not None:
            evidence_b.last_overlap_support = float(max(current_iou, recent_iou_ab))
            evidence_b.last_touch_support = float(max(current_touch, recent_touch_ab))
            self._append_recent_score(evidence_b.recent_support_scores, support_score)

        return {
            "pair_ids": [payload_a.node_id, payload_b.node_id],
            "geom_ok": bool(adjacent),
            "support_ok": bool(support_ok),
            "adjacent": adjacent,
            "cabinet_iou_only_policy": bool(iou_only_policy),
            "bbox3d_intersects": bool(bbox_debug.get("valid") and all(float(v) > 0.0 for v in (bbox_debug.get("overlap_xyz") or [0.0, 0.0, 0.0]))),
            "bbox3d_iou": float(bbox3d_iou),
            "overlap_x": None if bbox_debug["overlap_xyz"] is None else float(bbox_debug["overlap_xyz"][0]),
            "overlap_y": None if bbox_debug["overlap_xyz"] is None else float(bbox_debug["overlap_xyz"][1]),
            "overlap_z": None if bbox_debug["overlap_xyz"] is None else float(bbox_debug["overlap_xyz"][2]),
            "bbox3d_overlap_x": None if bbox_debug["overlap_xyz"] is None else float(bbox_debug["overlap_xyz"][0]),
            "bbox3d_overlap_y": None if bbox_debug["overlap_xyz"] is None else float(bbox_debug["overlap_xyz"][1]),
            "bbox3d_overlap_z": None if bbox_debug["overlap_xyz"] is None else float(bbox_debug["overlap_xyz"][2]),
            "gap_x": None if bbox_debug["gap_xyz"] is None else float(bbox_debug["gap_xyz"][0]),
            "gap_y": None if bbox_debug["gap_xyz"] is None else float(bbox_debug["gap_xyz"][1]),
            "gap_z": None if bbox_debug["gap_xyz"] is None else float(bbox_debug["gap_xyz"][2]),
            "bbox3d_intersection_volume": float(bbox_debug["intersection_volume"]),
            "bbox_source_a": bbox_source_a,
            "bbox_source_b": bbox_source_b,
            "passes_depth_overlap": bool(front_debug["passes_depth_overlap"]),
            "passes_planar_adjacency": bool(front_debug["passes_planar_adjacency"]),
            "depth_overlap_thr": float(front_debug["depth_overlap_thr"]),
            "planar_overlap_thr_x": float(front_debug["planar_overlap_thr_x"]),
            "planar_overlap_thr_y": float(front_debug["planar_overlap_thr_y"]),
            "box_iou": float(current_iou),
            "box_touch": float(current_touch),
            "recent_support_ab": float(recent_ab),
            "recent_support_ba": float(recent_ba),
            "support_score": float(support_score),
            **plane_debug,
        }

    def _refresh_group_geometry(self, group: CabinetTrack, node_tracks: Dict[str, object]) -> None:
        centroids = []
        mins = []
        maxs = []
        for member_id in sorted(group.member_node_ids):
            track = node_tracks.get(member_id)
            if track is None or getattr(track, "role", None) != "C":
                continue
            centroid = self._track_centroid(track)
            bbox = self._track_bbox(track)
            if centroid is not None:
                centroids.append(centroid)
            if bbox is not None:
                mins.append(bbox["min"])
                maxs.append(bbox["max"])

        group.stable_geom_ready = False
        group.centroid_world_est = None
        group.bbox_world_est = None
        if centroids:
            group.centroid_world_est = np.mean(np.stack(centroids, axis=0), axis=0).tolist()
        if mins and maxs:
            group.bbox_world_est = {
                "min": np.min(np.stack(mins, axis=0), axis=0).tolist(),
                "max": np.max(np.stack(maxs, axis=0), axis=0).tolist(),
            }
            group.stable_geom_ready = True

    def prune_members(
        self,
        member_ids: Iterable[str],
        *,
        frame_idx: Optional[int] = None,
        reason: str = "",
        node_tracks: Optional[Dict[str, object]] = None,
    ) -> dict:
        removed_members: list[dict] = []
        touched_groups: set[str] = set()
        for member_id in sorted(set(member_ids)):
            gid = self.member_to_cabinet.pop(member_id, None)
            if gid is None:
                continue
            group = self.cabinet_tracks.get(gid)
            if group is None:
                continue
            touched_groups.add(gid)
            group.member_node_ids.discard(member_id)
            group.seeded_member_ids.discard(member_id)
            for pair_set in (group.pending_plane_pairs, group.validated_plane_pairs, group.failed_plane_pairs):
                for key in list(pair_set):
                    if member_id in key.split("|"):
                        pair_set.discard(key)
            self._refresh_group_plane_validation_status(group)
            removed_members.append(
                {
                    "node_id": member_id,
                    "group_id": gid,
                    "frame_idx": None if frame_idx is None else int(frame_idx),
                    "reason": reason,
                }
            )

        dropped_groups: list[str] = []
        for gid in sorted(touched_groups):
            group = self.cabinet_tracks.get(gid)
            if group is None:
                continue
            if node_tracks is not None:
                self._refresh_group_geometry(group, node_tracks)
            if not group.member_node_ids or not group.seeded_member_ids:
                dropped_groups.append(gid)
            self._drop_group_if_invalid(gid)

        return {
            "removed_members": removed_members,
            "dropped_groups": dropped_groups,
        }

    def _prune_unavailable_members(
        self,
        *,
        frame_idx: int,
        node_tracks: Dict[str, object],
        is_member_available: Optional[Callable[[str], bool]],
        reason: str,
    ) -> dict:
        if is_member_available is None:
            return {"removed_members": [], "dropped_groups": []}
        unavailable = [
            member_id
            for member_id in sorted(self.member_to_cabinet.keys())
            if not bool(is_member_available(member_id))
        ]
        if not unavailable:
            return {"removed_members": [], "dropped_groups": []}
        return self.prune_members(
            unavailable,
            frame_idx=frame_idx,
            reason=reason,
            node_tracks=node_tracks,
        )

    def _payload_for_existing_member(
        self,
        member_id: str,
        *,
        frame_idx: int,
        node_tracks: Dict[str, object],
        payload_by_id: dict[str, CabinetCarrierObservation],
    ) -> Optional[CabinetCarrierObservation]:
        payload = payload_by_id.get(member_id)
        if payload is not None:
            return payload
        track = node_tracks.get(member_id)
        if track is None or getattr(track, "role", None) != "C":
            return None
        label = str(getattr(track, "label", "") or "")
        if label not in self.policy.cabinet_seed_carriers:
            return None
        centroid = self._track_centroid(track)
        bbox = self._track_bbox(track)
        bbox_payload = None
        if bbox is not None:
            bbox_payload = (bbox["min"], bbox["max"])
        return CabinetCarrierObservation(
            node_id=member_id,
            label=label,
            det_idx=-1,
            frame_idx=int(frame_idx),
            box_xyxy=getattr(track, "last_box_xyxy", None),
            view_score=0.0,
            cabinet_box_marked=False,
            centroid_world=centroid,
            bbox3d_world=bbox_payload,
        )

    def _merge_existing_groups_by_member_adjacency(
        self,
        *,
        frame_idx: int,
        node_tracks: Dict[str, object],
        payload_by_id: dict[str, CabinetCarrierObservation],
        is_member_available: Optional[Callable[[str], bool]] = None,
    ) -> dict:
        gids = sorted(self.cabinet_tracks.keys())
        debug = {
            "global_group_merge_candidates": [],
            "merged_group_pairs": [],
            "merge_reason": "member_pair_adjacency",
        }
        if len(gids) < 2:
            return debug

        parent: dict[str, str] = {gid: gid for gid in gids}

        def find(gid: str) -> str:
            root = parent[gid]
            while root != parent[root]:
                root = parent[root]
            while gid != root:
                nxt = parent[gid]
                parent[gid] = root
                gid = nxt
            return root

        def union(a: str, b: str) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for idx, gid_a in enumerate(gids):
            group_a = self.cabinet_tracks.get(gid_a)
            if group_a is None:
                continue
            for gid_b in gids[idx + 1 :]:
                group_b = self.cabinet_tracks.get(gid_b)
                if group_b is None:
                    continue
                trigger_debug = None
                members_a = [
                    member_id
                    for member_id in sorted(group_a.member_node_ids)
                    if is_member_available is None or bool(is_member_available(member_id))
                ]
                members_b = [
                    member_id
                    for member_id in sorted(group_b.member_node_ids)
                    if is_member_available is None or bool(is_member_available(member_id))
                ]
                for member_a in members_a:
                    payload_a = self._payload_for_existing_member(
                        member_a,
                        frame_idx=frame_idx,
                        node_tracks=node_tracks,
                        payload_by_id=payload_by_id,
                    )
                    if payload_a is None:
                        continue
                    for member_b in members_b:
                        payload_b = self._payload_for_existing_member(
                            member_b,
                            frame_idx=frame_idx,
                            node_tracks=node_tracks,
                            payload_by_id=payload_by_id,
                        )
                        if payload_b is None:
                            continue
                        pair_debug = self._pair_adjacency(payload_a, payload_b, node_tracks)
                        if pair_debug["adjacent"]:
                            trigger_debug = pair_debug
                            break
                    if trigger_debug is not None:
                        break
                candidate = {
                    "group_ids": [gid_a, gid_b],
                    "merge": bool(trigger_debug is not None),
                    "trigger_member_pair": None if trigger_debug is None else list(trigger_debug.get("pair_ids", [])),
                    "trigger_pair_plane_normal_angle_deg": None
                    if trigger_debug is None
                    else trigger_debug.get("plane_normal_angle_deg"),
                    "trigger_pair_plane_offset_m": None if trigger_debug is None else trigger_debug.get("plane_offset_m"),
                }
                debug["global_group_merge_candidates"].append(candidate)
                if trigger_debug is not None:
                    union(gid_a, gid_b)
                    debug["merged_group_pairs"].append(
                        {
                            "group_ids": [gid_a, gid_b],
                            "merge_reason": "member_pair_adjacency",
                            "trigger_member_pair": list(trigger_debug.get("pair_ids", [])),
                            "trigger_pair_plane_normal_angle_deg": trigger_debug.get("plane_normal_angle_deg"),
                            "trigger_pair_plane_offset_m": trigger_debug.get("plane_offset_m"),
                        }
                    )

        groups_by_root: dict[str, list[str]] = {}
        for gid in gids:
            groups_by_root.setdefault(find(gid), []).append(gid)
        for root, members in sorted(groups_by_root.items()):
            existing = [gid for gid in sorted(members) if gid in self.cabinet_tracks]
            if len(existing) <= 1:
                continue
            dst_gid = existing[0]
            for src_gid in existing[1:]:
                self._merge_groups(dst_gid, src_gid)
            if dst_gid in self.cabinet_tracks:
                self._refresh_group_geometry(self.cabinet_tracks[dst_gid], node_tracks)
        return debug

    def revalidate_groups_with_plane_normals(
        self,
        node_tracks: Dict[str, object],
        frame_idx: int,
        *,
        is_member_available: Optional[Callable[[str], bool]] = None,
    ) -> dict:
        debug = {
            "frame_idx": int(frame_idx),
            "revalidated_pairs": [],
            "pruned_members_due_to_plane_failure": [],
            "groups_dropped_due_to_plane_failure": [],
            "pruned_members_due_to_unavailable": [],
            "groups_dropped_due_to_unavailable": [],
        }
        unavailable_prune = self._prune_unavailable_members(
            frame_idx=frame_idx,
            node_tracks=node_tracks,
            is_member_available=is_member_available,
            reason="member_unavailable_for_cabinet_revalidate",
        )
        debug["pruned_members_due_to_unavailable"] = list(unavailable_prune.get("removed_members", []))
        debug["groups_dropped_due_to_unavailable"] = list(unavailable_prune.get("dropped_groups", []))
        members_to_prune: set[str] = set()
        for gid, group in sorted(list(self.cabinet_tracks.items())):
            if gid not in self.cabinet_tracks:
                continue
            members = sorted(group.member_node_ids)
            group.pending_plane_pairs.clear()
            group.validated_plane_pairs.clear()
            group.failed_plane_pairs.clear()
            for idx, member_a in enumerate(members):
                payload_a = self._payload_for_existing_member(
                    member_a,
                    frame_idx=frame_idx,
                    node_tracks=node_tracks,
                    payload_by_id={},
                )
                if payload_a is None:
                    continue
                for member_b in members[idx + 1 :]:
                    payload_b = self._payload_for_existing_member(
                        member_b,
                        frame_idx=frame_idx,
                        node_tracks=node_tracks,
                        payload_by_id={},
                    )
                    if payload_b is None:
                        continue
                    pair_debug = self._pair_adjacency(payload_a, payload_b, node_tracks)
                    self._record_group_pair_validation(group, pair_debug)
                    debug["revalidated_pairs"].append(
                        {
                            "group_id": gid,
                            "pair_ids": list(pair_debug.get("pair_ids", [])),
                            "adjacent_before_plane_gate": bool(pair_debug.get("adjacent_before_plane_gate")),
                            "adjacent": bool(pair_debug.get("adjacent")),
                            "plane_validation_status": pair_debug.get("plane_validation_status"),
                            "plane_normal_angle_deg": pair_debug.get("plane_normal_angle_deg"),
                            "plane_offset_m": pair_debug.get("plane_offset_m"),
                        }
                    )
            for member_id in members:
                track = node_tracks.get(member_id)
                if track is None:
                    continue
                is_door = str(getattr(track, "label", "") or "").strip().lower() == "door"
                if not is_door:
                    continue
                has_failed_pair = any(member_id in key.split("|") for key in group.failed_plane_pairs)
                has_validated_pair = self._member_has_validated_plane_support(group, member_id, node_tracks)
                if has_failed_pair and not has_validated_pair:
                    members_to_prune.add(member_id)

        if members_to_prune:
            prune_debug = self.prune_members(
                members_to_prune,
                frame_idx=frame_idx,
                reason="plane_validation_failed",
                node_tracks=node_tracks,
            )
            debug["pruned_members_due_to_plane_failure"] = list(prune_debug.get("removed_members", []))
            debug["groups_dropped_due_to_plane_failure"] = list(prune_debug.get("dropped_groups", []))

        for group in self.cabinet_tracks.values():
            self._refresh_group_geometry(group, node_tracks)
            self._refresh_group_plane_validation_status(group)
        self.last_update_debug["plane_revalidate"] = debug
        return debug

    def update_frame(
        self,
        frame_idx: int,
        carrier_node_ids: Iterable[str],
        seeded_node_ids: set[str],
        node_tracks: Dict[str, object],
        carrier_obs_payloads: Optional[Iterable[CabinetCarrierObservation]] = None,
        scene_allows_cabinet: bool = True,
        is_member_available: Optional[Callable[[str], bool]] = None,
    ) -> dict:
        debug = {
            "frame_idx": int(frame_idx),
            "scene_allows_cabinet": bool(scene_allows_cabinet),
            "carrier_candidates": [],
            "pair_adjacency": [],
            "components": [],
            "pruned_members_due_to_unavailable": [],
            "groups_dropped_due_to_unavailable": [],
        }
        if not scene_allows_cabinet:
            self.last_update_debug = debug
            return debug

        unavailable_prune = self._prune_unavailable_members(
            frame_idx=frame_idx,
            node_tracks=node_tracks,
            is_member_available=is_member_available,
            reason="member_unavailable_for_cabinet_update",
        )
        debug["pruned_members_due_to_unavailable"] = list(unavailable_prune.get("removed_members", []))
        debug["groups_dropped_due_to_unavailable"] = list(unavailable_prune.get("dropped_groups", []))

        if carrier_obs_payloads is not None:
            payloads = list(carrier_obs_payloads)
        else:
            payloads = [
                CabinetCarrierObservation(
                    node_id=node_id,
                    label=getattr(node_tracks.get(node_id), "label", ""),
                    det_idx=-1,
                    frame_idx=int(frame_idx),
                    box_xyxy=None,
                    view_score=0.0,
                    cabinet_box_marked=node_id in seeded_node_ids,
                )
                for node_id in carrier_node_ids
            ]

        payload_by_id: dict[str, CabinetCarrierObservation] = {}
        for payload in payloads:
            if is_member_available is not None and not bool(is_member_available(payload.node_id)):
                debug["carrier_candidates"].append(
                    {
                        "node_id": payload.node_id,
                        "label": payload.label,
                        "seeded_by_box": bool(payload.cabinet_box_marked or payload.node_id in seeded_node_ids),
                        "view_score": float(payload.view_score),
                        "eligible_for_cabinet": False,
                        "skip_reason": "member_unavailable_for_cabinet_update",
                    }
                )
                continue
            track = node_tracks.get(payload.node_id)
            if track is None or getattr(track, "role", None) != "C":
                continue
            if getattr(track, "label", payload.label) not in self.policy.cabinet_seed_carriers:
                continue
            evidence = self.member_evidence.setdefault(
                payload.node_id,
                CabinetMemberEvidence(node_id=payload.node_id, label=getattr(track, "label", payload.label)),
            )
            evidence.label = getattr(track, "label", payload.label)
            evidence.last_seen_frame = int(frame_idx)
            evidence.last_view_score = float(payload.view_score)
            evidence.seeded_by_box = bool(evidence.seeded_by_box or payload.cabinet_box_marked or payload.node_id in seeded_node_ids)
            self._append_frame_once(evidence.support_frames, frame_idx)
            self._record_recent_box(evidence, frame_idx, payload.box_xyxy)
            payload_by_id[payload.node_id] = payload
            debug["carrier_candidates"].append(
                {
                    "node_id": payload.node_id,
                    "label": evidence.label,
                    "seeded_by_box": bool(evidence.seeded_by_box),
                    "view_score": float(payload.view_score),
                }
            )

        node_ids = sorted(payload_by_id.keys())
        parent: dict[str, str] = {node_id: node_id for node_id in node_ids}

        def find(node_id: str) -> str:
            root = parent[node_id]
            while root != parent[root]:
                root = parent[root]
            while node_id != root:
                nxt = parent[node_id]
                parent[node_id] = root
                node_id = nxt
            return root

        def union(lhs: str, rhs: str) -> None:
            root_l = find(lhs)
            root_r = find(rhs)
            if root_l != root_r:
                parent[root_r] = root_l

        for idx, node_a in enumerate(node_ids):
            payload_a = payload_by_id[node_a]
            for node_b in node_ids[idx + 1 :]:
                payload_b = payload_by_id[node_b]
                pair_debug = self._pair_adjacency(payload_a, payload_b, node_tracks)
                debug["pair_adjacency"].append(pair_debug)
                if pair_debug["adjacent"]:
                    union(node_a, node_b)

        components: dict[str, set[str]] = {}
        for node_id in node_ids:
            components.setdefault(find(node_id), set()).add(node_id)

        for members in sorted((sorted(values) for values in components.values()), key=lambda items: items[0] if items else ""):
            intersecting_groups = sorted(
                {
                    self.member_to_cabinet.get(member_id)
                    for member_id in members
                    if self.member_to_cabinet.get(member_id) in self.cabinet_tracks
                }
            )
            seeded_member_ids = sorted(
                {
                    member_id
                    for member_id in members
                    if self.member_evidence.get(member_id) is not None and self.member_evidence[member_id].seeded_by_box
                }
            )
            accepted = bool(seeded_member_ids)
            if not accepted:
                accepted = any(bool(self.cabinet_tracks.get(gid).seeded_member_ids) for gid in intersecting_groups if gid in self.cabinet_tracks)

            assigned_gid = None
            if accepted and members:
                if intersecting_groups:
                    assigned_gid = intersecting_groups[0]
                    for other_gid in intersecting_groups[1:]:
                        self._merge_groups(assigned_gid, other_gid)
                else:
                    assigned_gid = self._ensure_group(members[0], frame_idx)
                for member_id in members:
                    evidence = self.member_evidence.get(member_id)
                    payload = payload_by_id[member_id]
                    self._attach_member(assigned_gid, payload, seeded=bool(evidence and evidence.seeded_by_box))

            debug["components"].append(
                {
                    "member_ids": members,
                    "seeded_member_ids": seeded_member_ids,
                    "intersecting_groups": intersecting_groups,
                    "accepted_as_cabinet_component": bool(accepted),
                    "assigned_group_id": assigned_gid,
                }
            )

        component_id_by_member: dict[str, str] = {}
        for component_index, component in enumerate(debug["components"]):
            component_id = f"component_{component_index:04d}"
            component["component_id"] = component_id
            for member_id in component["member_ids"]:
                component_id_by_member[member_id] = component_id
        for pair_debug in debug["pair_adjacency"]:
            pair_debug["unioned"] = bool(pair_debug["adjacent"])
            pair_debug["component_id_a"] = component_id_by_member.get(pair_debug["pair_ids"][0])
            pair_debug["component_id_b"] = component_id_by_member.get(pair_debug["pair_ids"][1])

        for group in self.cabinet_tracks.values():
            self._refresh_group_geometry(group, node_tracks)
            for pair_debug in debug["pair_adjacency"]:
                pair_ids = pair_debug.get("pair_ids") or []
                if len(pair_ids) == 2 and pair_ids[0] in group.member_node_ids and pair_ids[1] in group.member_node_ids:
                    self._record_group_pair_validation(group, pair_debug)

        merge_debug = self._merge_existing_groups_by_member_adjacency(
            frame_idx=frame_idx,
            node_tracks=node_tracks,
            payload_by_id=payload_by_id,
            is_member_available=is_member_available,
        )
        debug.update(merge_debug)

        self.last_update_debug = debug
        return debug

    def commit_keyframe(
        self,
        kf_idx: int,
        node_tracks: Dict[str, object],
        graph,
        *,
        can_commit_member: Optional[Callable[[str], bool]] = None,
        relation_text_lookup: Optional[Callable[[str, str, str], str]] = None,
    ) -> dict:
        debug = {
            "kf_idx": int(kf_idx),
            "groups": [],
        }
        for gid, group in sorted(list(self.cabinet_tracks.items())):
            if gid not in self.cabinet_tracks:
                continue
            carrier_members = {
                member_id
                for member_id in group.member_node_ids
                if member_id in node_tracks and getattr(node_tracks[member_id], "role", None) == "C"
            }
            seeded_members = {
                member_id
                for member_id in group.seeded_member_ids
                if member_id in carrier_members
            }

            committed_members: list[str] = []
            skipped_members_due_to_existing_parent: list[str] = []
            skipped_members_due_to_missing_plane_normal: list[str] = []
            skipped_members_due_to_pending_plane_validation: list[str] = []
            for member_id in sorted(carrier_members):
                if can_commit_member is not None and not bool(can_commit_member(member_id)):
                    skipped_members_due_to_existing_parent.append(member_id)
                    continue
                member_track = node_tracks.get(member_id)
                if bool(self.policy.cabinet_require_plane_normal_for_door) and member_track is not None:
                    is_door = str(getattr(member_track, "label", "") or "").strip().lower() == "door"
                    if is_door and not self._member_has_validated_plane_support(group, member_id, node_tracks):
                        if self._track_plane_normal_debug(member_track)["valid"]:
                            skipped_members_due_to_pending_plane_validation.append(member_id)
                        else:
                            skipped_members_due_to_missing_plane_normal.append(member_id)
                        continue
                committed_members.append(member_id)

            committed_seeded_members = [member_id for member_id in committed_members if member_id in seeded_members]
            group_debug = {
                "group_id": gid,
                "carrier_members": sorted(carrier_members),
                "seeded_members": sorted(seeded_members),
                "committed_members": committed_members,
                "skipped_members_due_to_existing_parent": skipped_members_due_to_existing_parent,
                "skipped_members_due_to_missing_plane_normal": skipped_members_due_to_missing_plane_normal,
                "skipped_members_due_to_pending_plane_validation": skipped_members_due_to_pending_plane_validation,
            }
            debug["groups"].append(group_debug)

            unstable_skipped_members = sorted(
                set(skipped_members_due_to_missing_plane_normal).union(skipped_members_due_to_pending_plane_validation)
            )
            if unstable_skipped_members:
                removed_unstable_edges = []
                local_edges = getattr(graph, "local_edges", None)
                if isinstance(local_edges, dict):
                    for key, edge in list(local_edges.items()):
                        if edge.dst_node_id not in unstable_skipped_members:
                            continue
                        if edge.relation_text != "aggregate_cabinet" and not str(edge.src_node_id).startswith("O_CABINET_"):
                            continue
                        removed_unstable_edges.append(
                            {
                                "src_node_id": edge.src_node_id,
                                "dst_node_id": edge.dst_node_id,
                                "edge_type": edge.edge_type,
                                "status": str(getattr(edge, "status", "")),
                                "skip_reason": (
                                    "missing_plane_normal"
                                    if edge.dst_node_id in skipped_members_due_to_missing_plane_normal
                                    else "pending_plane_validation"
                                ),
                            }
                        )
                        local_edges.pop(key, None)
                group_debug["removed_unstable_skipped_member_edges"] = removed_unstable_edges

            skipped_members_to_prune = sorted(set(skipped_members_due_to_existing_parent))
            if skipped_members_to_prune:
                prune_debug = self.prune_members(
                    skipped_members_to_prune,
                    frame_idx=kf_idx,
                    reason="commit_skip_member",
                    node_tracks=node_tracks,
                )
                group_debug["pruned_skipped_members"] = list(prune_debug.get("removed_members", []))
                group_debug["dropped_groups_after_skip_prune"] = list(prune_debug.get("dropped_groups", []))
                for member_id in skipped_members_to_prune:
                    member_track = node_tracks.get(member_id)
                    if member_track is not None:
                        member_track.cabinet_group_id = None
                graph.local_edges = {
                    key: edge
                    for key, edge in graph.local_edges.items()
                    if not (
                        edge.dst_node_id in skipped_members_to_prune
                        and (
                            edge.relation_text == "aggregate_cabinet"
                            or str(edge.src_node_id).startswith("O_CABINET_")
                        )
                    )
                }
                group = self.cabinet_tracks.get(gid)

            if group is None:
                continue
            if not committed_members or not committed_seeded_members:
                continue

            self._refresh_group_geometry(group, node_tracks)
            cabinet_id = group.cabinet_id
            if cabinet_id not in node_tracks:
                from .node_track import OnlineNodeTrack

                node_tracks[cabinet_id] = OnlineNodeTrack(
                    node_id=cabinet_id,
                    label="cabinet",
                    role="O",
                    origin="aggregate",
                )

            cabinet_track = node_tracks[cabinet_id]
            cabinet_track.origin = "aggregate"
            cabinet_track.label = "cabinet"
            cabinet_track.role = "O"
            cabinet_track.last_seen_frame = group.last_seen_frame
            cabinet_track.last_seen_kf = int(kf_idx)
            cabinet_track.cabinet_group_id = gid
            cabinet_track.cabinet_member_count = len(committed_members)
            cabinet_track.cabinet_seeded_count = len(committed_seeded_members)
            cabinet_track.centroid_world_est = list(group.centroid_world_est) if group.centroid_world_est is not None else None
            cabinet_track.bbox_world_est = dict(group.bbox_world_est) if group.bbox_world_est is not None else None
            cabinet_track.bbox_anchor = dict(group.bbox_world_est) if group.bbox_world_est is not None else None
            if group.bbox_world_est is not None:
                bmin = np.asarray(group.bbox_world_est.get("min") or [], dtype=np.float32)
                bmax = np.asarray(group.bbox_world_est.get("max") or [], dtype=np.float32)
                if bmin.shape == (3,) and bmax.shape == (3,):
                    cabinet_track.bbox_diag_world_est = float(np.linalg.norm(bmax - bmin))
            cabinet_track.stable_geom_ready = bool(group.bbox_world_est is not None or group.centroid_world_est is not None)

            group.last_commit_kf = int(kf_idx)
            group.last_seen_frame = int(group.last_seen_frame if group.last_seen_frame is not None else kf_idx)

            committed_member_set = set(committed_members)
            removed_conflicting_parent_edges = []
            local_edges = getattr(graph, "local_edges", None)
            if isinstance(local_edges, dict):
                for key, edge in list(local_edges.items()):
                    if edge.edge_type != "O-C" or edge.dst_node_id not in committed_member_set or edge.src_node_id == cabinet_id:
                        continue
                    if edge.relation_text == "aggregate_cabinet" or str(edge.src_node_id).startswith("O_CABINET_"):
                        continue
                    removed_conflicting_parent_edges.append(
                        {
                            "src_node_id": edge.src_node_id,
                            "dst_node_id": edge.dst_node_id,
                            "edge_type": edge.edge_type,
                            "status": str(getattr(edge, "status", "")),
                            "support_count": int(getattr(edge, "support_count", 0) or 0),
                        }
                    )
                    local_edges.pop(key, None)
            group_debug["removed_conflicting_parent_edges"] = removed_conflicting_parent_edges

            for member_id in committed_members:
                member_track = node_tracks.get(member_id)
                if member_track is None or getattr(member_track, "role", None) != "C":
                    continue
                member_track.cabinet_group_id = gid
                member_track.cabinet_member_count = len(committed_members)
                member_track.cabinet_seeded_count = len(committed_seeded_members)
                graph.upsert_local_edge(
                    cabinet_id,
                    member_id,
                    "O-C",
                    relation_text=(
                        (relation_text_lookup(cabinet_id, member_id, "O-C") if relation_text_lookup else "")
                        or "part of"
                    ),
                    committed_kf=kf_idx,
                    support_count=max(1, len(group.support_frames)),
                    evidence_score=float(len(committed_seeded_members)),
                    status="committed",
                    first_seen_frame=int(group.support_frames[0]) if group.support_frames else int(kf_idx),
                    last_seen_frame=int(group.support_frames[-1]) if group.support_frames else int(kf_idx),
                    last_update_source="cabinet_commit",
                    margin=0.0,
                    latest_margin=0.0,
                    recent_support_count=min(3, max(1, len(group.support_frames))),
                    switch_count=0,
                )

        self.last_commit_debug = debug
        return debug

    def to_dict(self) -> dict:
        return {
            "cabinet_tracks": {
                gid: {
                    "cabinet_id": track.cabinet_id,
                    "member_node_ids": sorted(track.member_node_ids),
                    "seeded_member_ids": sorted(track.seeded_member_ids),
                    "support_frames": list(track.support_frames),
                    "anchor_kf_id": track.anchor_kf_id,
                    "centroid_world_est": track.centroid_world_est,
                    "bbox_world_est": track.bbox_world_est,
                    "stable_geom_ready": track.stable_geom_ready,
                    "pending_plane_pairs": sorted(track.pending_plane_pairs),
                    "validated_plane_pairs": sorted(track.validated_plane_pairs),
                    "failed_plane_pairs": sorted(track.failed_plane_pairs),
                    "plane_validation_status": track.plane_validation_status,
                    "last_seen_frame": track.last_seen_frame,
                    "last_commit_kf": track.last_commit_kf,
                }
                for gid, track in sorted(self.cabinet_tracks.items())
            },
            "member_to_cabinet": {key: value for key, value in sorted(self.member_to_cabinet.items())},
            "member_evidence": {
                node_id: {
                    "label": evidence.label,
                    "seeded_by_box": bool(evidence.seeded_by_box),
                    "support_frames": list(evidence.support_frames),
                    "last_seen_frame": evidence.last_seen_frame,
                    "recent_boxes": list(evidence.recent_boxes),
                    "recent_support_scores": list(evidence.recent_support_scores),
                    "last_overlap_support": float(evidence.last_overlap_support),
                    "last_touch_support": float(evidence.last_touch_support),
                    "last_view_score": float(evidence.last_view_score),
                }
                for node_id, evidence in sorted(self.member_evidence.items())
            },
            "last_update_debug": dict(self.last_update_debug),
            "last_commit_debug": dict(self.last_commit_debug),
        }
