from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set


def _norm_label(label: str) -> str:
    text = (label or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


@dataclass
class FunctionalGraphPolicy:
    suppress_objects: set[str] = field(default_factory=lambda: {"stove"})
    hint_only_objects: set[str] = field(default_factory=lambda: {"cabinet"})
    # Cabinet aggregation is disabled by default in the current phase.
    enable_cabinet_aggregation: bool = False
    aggregate_output_objects: set[str] = field(default_factory=set)
    aggregate_candidate_objects: set[str] = field(default_factory=lambda: {"cabinet"})
    disable_fusion_labels: set[str] = field(default_factory=set)
    temp_disable_fusion_observations: set[str] = field(default_factory=set)
    drop_lid_nodes: bool = False
    drop_lid_when_overlaps_cap: bool = False
    cap_lid_overlap_iou_thr: float = 0.9
    remote_edge_block_labels: set[str] = field(default_factory=set)
    temp_final_local_edge_invariant: bool = False
    temp_final_drop_labels: set[str] = field(default_factory=set)
    temp_drop_border_u_nodes: bool = False
    temp_keep_single_frame_cup_handle: bool = False
    temp_limit_door_drawer_units: bool = False
    temp_merge_drawers_final: bool = False
    temp_bottle_single_cap_final: bool = False
    temp_bottle_cap_min_distance: float = 0.0
    temp_parent_single_child_per_label_final: bool = False
    temp_node_pos_point_centroid: bool = False
    temp_realtime_final_adjustments: bool = False
    temp_cabinet_iou_only_aggregation: bool = False
    temp_exclude_cabinet_aggregation_observations: set[str] = field(default_factory=set)
    temp_stage1_min_projected_iou: float = 0.0
    cabinet_scene_prior_labels: set[str] = field(default_factory=lambda: {"cabinet"})
    cabinet_seed_carriers: set[str] = field(default_factory=lambda: {"door", "drawer"})
    cabinet_unit_labels: set[str] = field(default_factory=lambda: {"handle", "knob"})
    cabinet_carrier_contain_thr: float = 0.6
    cabinet_expand_bbox_intersection_eps: float = 0.01
    cabinet_min_depth_overlap: float = 0.03
    cabinet_min_planar_overlap: float = 0.05
    cabinet_max_planar_gap: float = 0.025
    cabinet_plane_normal_pca_max_points: int = 256
    cabinet_plane_normal_voxel_size: float = 0.01
    cabinet_min_plane_normal_quality: float = 0.15
    cabinet_max_plane_normal_angle_deg: float = 35.0
    cabinet_max_plane_offset_m: float = 0.12
    cabinet_require_plane_normal_for_door: bool = True
    cabinet_expand_dist_thr: float = 0.35
    cabinet_expand_depth_thr: float = 0.20
    cabinet_expand_size_ratio_thr: float = 2.5
    label_aliases: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "stove": {"stove", "kitchen stove", "gas stove", "stove top", "cooktop"},
            "cabinet": {"cabinet", "kitchen cabinet", "wall cabinet", "base cabinet"},
        }
    )

    @property
    def defer_group_objects(self) -> set[str]:
        # Backward-compatible alias. Prefer hint_only_objects in new code.
        return self.hint_only_objects

    def canonical_object_match(self, label: str) -> set[str]:
        text = _norm_label(label)
        if not text:
            return set()
        matched: set[str] = set()
        for canonical, aliases in self.label_aliases.items():
            alias_norm = {_norm_label(a) for a in aliases}
            if text in alias_norm:
                matched.add(canonical)
        if not matched:
            matched.add(text)
        return matched

    def _matches(self, label: str, canonical_set: set[str]) -> bool:
        if not canonical_set:
            return False
        normalized_set = {_norm_label(name) for name in canonical_set}
        return bool(self.canonical_object_match(label).intersection(normalized_set))

    def is_suppressed_object(self, label: str) -> bool:
        return self._matches(label, self.suppress_objects)

    def is_hint_only_object(self, label: str) -> bool:
        return self._matches(label, self.hint_only_objects)

    def is_aggregate_output_object(self, label: str) -> bool:
        return self._matches(label, self.aggregate_output_objects)

    def should_enable_cabinet_aggregation(self) -> bool:
        return bool(self.enable_cabinet_aggregation)

    def should_disable_3d_fusion(self, label: str) -> bool:
        return self._matches(label, self.disable_fusion_labels)

    def should_disable_3d_fusion_for_observation(
        self,
        label: str,
        frame_idx: int | None,
        det_idx: int | None,
    ) -> bool:
        if self.should_disable_3d_fusion(label):
            return True
        if not self.temp_disable_fusion_observations:
            return False
        try:
            frame_i = int(frame_idx)
            det_i = int(det_idx)
        except Exception:
            return False
        for selector in self.temp_disable_fusion_observations:
            text = str(selector or "").strip().lower().replace(" ", "")
            text = text.replace("frame_", "").replace("frame-", "").replace("frame", "")
            if ":" not in text:
                continue
            frame_s, det_s = text.split(":", 1)
            try:
                if int(frame_s.lstrip("0") or "0") == frame_i and int(det_s.lstrip("0") or "0") == det_i:
                    return True
            except Exception:
                continue
        return False

    def should_drop_lid_cap_overlap(self) -> bool:
        return bool(self.drop_lid_when_overlaps_cap)

    def should_drop_lid_node(self, label: str) -> bool:
        return bool(self.drop_lid_nodes) and _norm_label(label) == "lid"

    def should_block_remote_edge_label(self, label: str) -> bool:
        return self._matches(label, self.remote_edge_block_labels)

    def should_block_remote_edge(self, src_label: str, dst_label: str) -> bool:
        return self.should_block_remote_edge_label(src_label) or self.should_block_remote_edge_label(dst_label)

    def should_apply_final_local_edge_invariant(self) -> bool:
        return bool(self.temp_final_local_edge_invariant)

    def should_drop_final_label(self, label: str) -> bool:
        return self._matches(label, self.temp_final_drop_labels)

    def should_drop_border_u_node(self) -> bool:
        return bool(self.temp_drop_border_u_nodes)

    def should_keep_single_frame_cup_handle(self) -> bool:
        return bool(self.temp_keep_single_frame_cup_handle)

    def should_limit_door_drawer_units(self) -> bool:
        return bool(self.temp_limit_door_drawer_units)

    def should_merge_drawers_final(self) -> bool:
        return bool(self.temp_merge_drawers_final)

    def should_keep_single_cap_per_bottle_final(self) -> bool:
        return bool(self.temp_bottle_single_cap_final)

    def bottle_cap_min_display_distance(self) -> float:
        return max(0.0, float(self.temp_bottle_cap_min_distance or 0.0))

    def should_limit_parent_child_label_final(self) -> bool:
        return bool(self.temp_parent_single_child_per_label_final)

    def should_use_point_cloud_centroid_node_positions(self) -> bool:
        return bool(self.temp_node_pos_point_centroid)

    def should_apply_realtime_final_adjustments(self) -> bool:
        return bool(self.temp_realtime_final_adjustments)

    def should_use_cabinet_iou_only_aggregation(self) -> bool:
        return bool(self.temp_cabinet_iou_only_aggregation)

    @staticmethod
    def _selector_matches_frame_det(selector: str, frame_idx: int | None, det_idx: int | None) -> bool:
        try:
            frame_i = int(frame_idx)
            det_i = int(det_idx)
        except Exception:
            return False
        text = str(selector or "").strip().lower().replace(" ", "")
        text = text.replace("frame_", "").replace("frame-", "").replace("frame", "")
        if ":" not in text:
            return False
        frame_s, det_s = text.split(":", 1)
        try:
            return int(frame_s.lstrip("0") or "0") == frame_i and int(det_s.lstrip("0") or "0") == det_i
        except Exception:
            return False

    def should_exclude_cabinet_aggregation_observation(
        self,
        frame_idx: int | None,
        det_idx: int | None,
    ) -> bool:
        return any(
            self._selector_matches_frame_det(selector, frame_idx, det_idx)
            for selector in self.temp_exclude_cabinet_aggregation_observations
        )

    def stage1_min_projected_iou(self) -> float:
        return max(0.0, float(self.temp_stage1_min_projected_iou or 0.0))

    def should_enable_cabinet_aggregation_for_frame(self, frame_result: dict | None) -> bool:
        if not self.should_enable_cabinet_aggregation():
            return False

        prior_labels = self.cabinet_scene_prior_labels or self.aggregate_candidate_objects or {"cabinet"}
        known_tags = (frame_result or {}).get("known_tags")
        if isinstance(known_tags, list):
            if any(self._matches(str(tag), prior_labels) for tag in known_tags):
                return True

        present = (frame_result or {}).get("present", []) or []
        for entry in present:
            if self._matches(str(entry.get("object") or ""), prior_labels):
                return True
        return False

    def should_emit_aggregate_object(self, label: str) -> bool:
        return self.should_enable_cabinet_aggregation() and self.is_aggregate_output_object(label)

    def is_detect_only_object(self, label: str) -> bool:
        return self.is_hint_only_object(label) and not self.should_emit_aggregate_object(label)

    def should_skip_standard_node(self, label: str) -> bool:
        return self.is_suppressed_object(label) or self.is_hint_only_object(label)

    def should_hide_overlay_node(self, node_id: str, label: str, origin: str) -> bool:
        if self.is_suppressed_object(label):
            return True
        if self.is_hint_only_object(label):
            is_aggregate = origin == "aggregate" or str(node_id).startswith("O_CABINET_")
            if is_aggregate:
                return not self.should_emit_aggregate_object(label)
            return True
        return False

    def should_filter_from_standard_graph(self, label: str) -> bool:
        # Backward-compatible alias. Prefer should_skip_standard_node in new code.
        return self.should_skip_standard_node(label)

    def is_deferred_group_object(self, label: str) -> bool:
        # Backward-compatible alias. Prefer is_hint_only_object in new code.
        return self.is_hint_only_object(label)
