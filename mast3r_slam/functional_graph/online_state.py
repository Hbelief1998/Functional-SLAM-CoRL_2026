from __future__ import annotations

import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

from mast3r_slam.semantic.io_utils import write_json_atomic

# Environment variable to enable runtime re-anchor for A/B experiments.
# Re-anchor is enabled by default. Set FG_DISABLE_REANCHOR=1 to disable.
_ENV_DISABLE_REANCHOR = os.environ.get("FG_DISABLE_REANCHOR", "0") == "1"

# Environment variable to enable multi-anchor association assist.
# Multi-anchor assist is enabled by default.  It provides a tiny bonus to
# stable tracks whose primary anchor is visibly weak.  Set
# FG_DISABLE_AUX_ASSIST=1 to disable for A/B experiments.
_ENV_DISABLE_AUX_ASSIST = os.environ.get("FG_DISABLE_AUX_ASSIST", "0") == "1"

# Maximum bonus (in raw stage1 score units) the multi-anchor assist can
# add to a single candidate.  Kept tight by design so it cannot override
# the primary anchor.
_MULTI_ANCHOR_ASSIST_MAX_BONUS = 0.04

# Intermediate snapshot throttling.  The online functional graph JSON is
# only consumed by offline tools after `flush_graph`, so intermediate
# writes on every keyframe (which can hit 2-3s each as the graph grows
# past tens of MB) are pure overhead.
#




#


try:
    _ENV_SNAPSHOT_EVERY_KF = int(os.environ.get("FG_SNAPSHOT_EVERY_KF", "5"))
except ValueError:
    _ENV_SNAPSHOT_EVERY_KF = 5

# Node consolidation pass (sibling duplicate / orphan low-quality node merge
# or prune).  Conservative by default (only acts when co-visibility is zero,
# label/role match, geometry is close, and quality gap is clear).
# Disable globally by setting FG_DISABLE_NODE_CONSOLIDATION=1.
# Force dry-run only (compute candidates + log, never mutate) by setting
# FG_NODE_CONSOLIDATION_DRY_RUN=1.
_ENV_DISABLE_NODE_CONSOLIDATION = os.environ.get("FG_DISABLE_NODE_CONSOLIDATION", "0") == "1"
_ENV_NODE_CONSOLIDATION_DRY_RUN = os.environ.get("FG_NODE_CONSOLIDATION_DRY_RUN", "0") == "1"
try:
    _ENV_NODE_CONSOLIDATION_EVERY_KF = int(
        os.environ.get("FG_NODE_CONSOLIDATION_EVERY_KF", "3")
    )
except ValueError:
    _ENV_NODE_CONSOLIDATION_EVERY_KF = 3

# Parent-level unstable-sibling arbitration over the low-shot tentative path.
# When multiple unstable child candidates compete for the same parent under
# the same role/label group with no co-visibility between them, only the
# clearly-best one keeps its low-shot edge.  Disable with
# ``FG_DISABLE_UNSTABLE_SIBLING_ARBITRATION=1``.
_ENV_DISABLE_UNSTABLE_SIBLING_ARBITRATION = (
    os.environ.get("FG_DISABLE_UNSTABLE_SIBLING_ARBITRATION", "0") == "1"
)

# Conservative orphan-U pruning over nodes that lost parent-level
# competitions repeatedly (sibling-duplicate-risk + unstable-sibling
# arbitration).  Disabled via FG_DISABLE_ORPHAN_U_PRUNE=1.
_ENV_DISABLE_ORPHAN_U_PRUNE = (
    os.environ.get("FG_DISABLE_ORPHAN_U_PRUNE", "0") == "1"
)

# Conservative tentative remote-edge recall path that lets strong 2D remote
# evidence land in ``graph.remote_edges`` (status=tentative, TTL retention)
# even before the remote posterior reaches its committed-stable bar.
_ENV_DISABLE_REMOTE_TENTATIVE = (
    os.environ.get("FG_DISABLE_REMOTE_TENTATIVE", "0") == "1"
)

# Association hot-path optimization toggles for A/B validation.  The combined
# switch restores the conservative per-call geometry path without changing
# scoring thresholds or graph policy.
_ENV_DISABLE_ASSOC_HOTPATH_OPT = os.environ.get("FG_DISABLE_ASSOC_HOTPATH_OPT", "0") == "1"
_ENV_DISABLE_ASSOC_GEOM_CACHE = (
    _ENV_DISABLE_ASSOC_HOTPATH_OPT
    or os.environ.get("FG_DISABLE_ASSOC_GEOM_CACHE", "0") == "1"
)
_ENV_DISABLE_BATCH_PROJECTABILITY = (
    _ENV_DISABLE_ASSOC_HOTPATH_OPT
    or os.environ.get("FG_DISABLE_BATCH_PROJECTABILITY", "0") == "1"
)

from .geometry_utils import (
    bbox_diag,
    bbox_from_points,
    bbox_size,
    box_iou_xyxy,
    box_overlap_over_obs_area,
    box_overlap_ratio_xyxy,
    box_touches_image_border,
    box_to_mask,
    depth_boxplot_filter_mask,
    dbscan_filter_points,
    expanded_box_contains,
    extract_points_from_mask,
    frame_hw,
    frame_intrinsics,
    observation_view_score,
    transform_points_frame,
    transform_points_world,
    world_distance,
)
from .graph_commit import PersistentEdge, PersistentFunctionalGraph
from .cabinet_aggregator import CabinetAggregator
from .llava_scheduler import LlavaScheduler
from .local_posterior import LocalParentPosterior
from .node_track import OnlineNodeTrack
from .policy import FunctionalGraphPolicy
from .remote_posterior import RemotePairPosterior
from .types import ROLE_RANK, CabinetCarrierObservation, NodeObservation, normalize_label, role_from_rank

# ---------------------------------------------------------------------------
# Label semantic similarity helper (Section A — config-driven aliases)
# ---------------------------------------------------------------------------
#
# The legacy hard-coded alias table now lives in
# ``mast3r_slam.functional_graph.node_consolidation``.  ``label_semantic_-
# similarity`` keeps using that legacy table by *default* for backward
# compatibility (the system has been depending on door/handle/knob synonyms
# for a long time), but new code paths should call
# ``label_similarity(a, b, alias_groups=...)`` from ``node_consolidation``
# with an explicit alias list loaded via ``load_label_alias_groups``.
from .node_consolidation import (
    LEGACY_ALIAS_GROUPS as _LABEL_ALIAS_GROUPS,
    label_similarity as _label_similarity_helper,
    load_label_alias_groups,
)

# Build a fast lookup: label → group id
_LABEL_TO_GROUP: dict[str, int] = {}
for _gid, _group in enumerate(_LABEL_ALIAS_GROUPS):
    for _lbl in _group:
        _LABEL_TO_GROUP[_lbl] = _gid


def label_semantic_similarity(
    obs_label: str,
    track_label: str,
    *,
    alias_groups: Optional[list] = None,
) -> float:
    """Deterministic similarity between two *normalized* labels.

    * 1.0 – exact string match.
    * 0.9 – same alias / synonym group.
    * 0.0 – different.

    ``alias_groups=None`` falls back to the legacy table for backward
    compatibility.  Callers that want strictly config-driven matching
    should pass an explicit list (typically loaded via
    ``load_label_alias_groups``).
    """
    if alias_groups is not None:
        return _label_similarity_helper(obs_label, track_label, alias_groups=alias_groups)
    if obs_label == track_label:
        return 1.0
    obs_gid = _LABEL_TO_GROUP.get(obs_label)
    trk_gid = _LABEL_TO_GROUP.get(track_label)
    if obs_gid is not None and obs_gid == trk_gid:
        return 0.9
    return 0.0


class OnlineFunctionalState:
    def __init__(
        self,
        *,
        output_path: Optional[str | Path] = None,
        llava_scheduler: Optional[LlavaScheduler] = None,
        point_conf_thr: float = 0.3,
        min_points_per_obs: int = 8,
        max_points_per_obs: int = 256,
        obs_depth_boxplot_enable: bool = True,
        obs_depth_boxplot_iqr_scale: float = 1.5,
        obs_depth_boxplot_near_iqr_scale: Optional[float] = None,
        obs_depth_boxplot_min_keep_ratio: float = 0.35,
        obs_dbscan_enable: bool = True,
        obs_dbscan_pre_voxel: float = 0.01,
        obs_dbscan_eps: float = 0.03,
        obs_dbscan_min_samples: int = 8,
        obs_dbscan_min_keep_ratio: float = 0.35,
        kf_geom_dbscan_enable: bool = True,
        kf_geom_dbscan_pre_voxel: float = 0.008,
        kf_geom_dbscan_eps: float = 0.025,
        kf_geom_dbscan_min_samples: int = 10,
        kf_geom_dbscan_min_keep_ratio: float = 0.4,
        assoc_weight_iou: float = 0.65,
        assoc_weight_dist: float = 0.25,
        assoc_weight_label: float = 0.10,
        assoc_stage1_weight_iou: float = 0.45,
        assoc_stage1_weight_dist: float = 0.15,
        assoc_stage1_weight_label: float = 0.30,
        assoc_stage1_weight_subtype: float = 0.10,
        assoc_stage1_geom_penalty_weight_c: float = 0.12,
        assoc_stage1_continuity_overlap_weight_c: float = 0.15,
        assoc_stage1_continuity_tau_boost_c: float = 0.15,
        assoc_stage1_continuity_min_obs_count_c: int = 3,
        assoc_stage2_weight_dist: float = 0.45,
        assoc_stage2_weight_label: float = 0.40,
        assoc_stage2_weight_subtype: float = 0.10,
        assoc_diag_gate_scale: float = 1.2,
        assoc_diag_gate_min_by_role: Optional[Dict[str, float]] = None,
        assoc_diag_gate_max_by_role: Optional[Dict[str, float]] = None,
        assoc_birth_gate_cap_by_role: Optional[Dict[str, float]] = None,
        assoc_require_positive_overlap: bool = True,
        anchor_min_obs_count: int = 2,
        anchor_min_obs_count_by_role: Optional[Dict[str, int]] = None,
        anchor_min_kf_support_by_role: Optional[Dict[str, int]] = None,
        anchor_pre_stable_accum_roles: Optional[Dict[str, bool]] = None,
        anchor_candidate_min_points_by_role: Optional[Dict[str, int]] = None,
        anchor_candidate_min_view_score_by_role: Optional[Dict[str, float]] = None,
        anchor_candidate_border_min_points_by_role: Optional[Dict[str, int]] = None,
        anchor_candidate_border_min_view_score_by_role: Optional[Dict[str, float]] = None,
        anchor_candidate_bbox_center_gate_by_role: Optional[Dict[str, float]] = None,
        anchor_candidate_centroid_gate_by_role: Optional[Dict[str, float]] = None,
        anchor_carrier_border_promote_min_obs_count: int = 3,
        anchor_carrier_border_promote_min_points: int = 1024,
        anchor_carrier_border_promote_min_view_score: float = 0.12,
        anchor_fusion_min_points_by_role: Optional[Dict[str, int]] = None,
        anchor_fusion_min_view_score_by_role: Optional[Dict[str, float]] = None,
        anchor_fusion_border_min_points_by_role: Optional[Dict[str, int]] = None,
        anchor_fusion_border_min_view_score_by_role: Optional[Dict[str, float]] = None,
        local_decay: float = 0.95,
        local_min_support_frames: int = 3,
        local_stable_margin: float = 1.5,
        tentative_margin_local: float = 0.75,
        tentative_local_ttl_frames: int = 6,
        tentative_local_replace_margin: float = 0.2,
        tentative_local_lowshot_enable: bool = True,
        tentative_local_lowshot_edge_types: Optional[set[str]] = None,
        tentative_local_lowshot_min_status_strength_by_edge_type: Optional[dict[str, float]] = None,
        tentative_local_lowshot_require_no_competition: bool = True,
        tentative_local_lowshot_retention_policy: str = "until_contradicted",
        remote_decay: float = 0.95,
        remote_min_support_frames: int = 2,
        remote_stable_margin: float = 0.8,
        tentative_margin_remote: float = 0.4,
        tentative_remote_ttl_frames: int = 8,
        tentative_remote_replace_margin: float = 0.15,
        unresolved_frames_for_llava: int = 5,
        llava_margin_local: float = 0.5,
        llava_margin_remote: float = 0.5,
        graph_policy: Optional[FunctionalGraphPolicy] = None,
        save_intermediate_snapshots: bool = True,
        enable_delta_log: bool = True,
        keep_border_no_link_observations: bool = False,
    ) -> None:
        self.output_path = Path(output_path) if output_path is not None else None
        self.snapshot_every_kf = _ENV_SNAPSHOT_EVERY_KF if save_intermediate_snapshots else 0
        # Background snapshot worker: single slot, latest-wins.  Hides
        # JSON serialisation + fsync behind the main loop so real-time
        # users can keep writing periodic snapshots without blocking.
        import threading as _threading
        self._snapshot_lock = _threading.Lock()
        self._snapshot_pending: Optional[tuple[dict, bool]] = None  # (payload, pretty)
        self._snapshot_event = _threading.Event()
        self._snapshot_shutdown = False
        self._snapshot_worker: Optional[_threading.Thread] = None
        self._snapshot_worker_done = _threading.Event()
        self._snapshot_worker_done.set()
        # Optional delta log path (graph_delta.jsonl next to the snapshot).
        self._delta_log_path: Optional[Path] = None
        if self.output_path is not None and enable_delta_log:
            self._delta_log_path = self.output_path.parent / "graph_delta.jsonl"
            # Truncate on fresh run.
            try:
                self._delta_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._delta_log_path, "w", encoding="utf-8") as _f:
                    _f.write("")
            except Exception:
                self._delta_log_path = None
            self._snapshot_worker = _threading.Thread(
                target=self._snapshot_worker_loop, name="fg-snapshot", daemon=True
            )
            self._snapshot_worker.start()
        elif self.output_path is not None:
            self._snapshot_worker = _threading.Thread(
                target=self._snapshot_worker_loop, name="fg-snapshot", daemon=True
            )
            self._snapshot_worker.start()
        self.runtime_profile_totals: dict[str, float] = {}
        self.runtime_profile_counts: dict[str, int] = {}
        self.llava_scheduler = llava_scheduler or LlavaScheduler()
        self.keep_border_no_link_observations = bool(keep_border_no_link_observations)
        self.point_conf_thr = float(point_conf_thr)
        self.min_points_per_obs = int(min_points_per_obs)
        self.max_points_per_obs = int(max_points_per_obs)
        self.obs_depth_boxplot_enable = bool(obs_depth_boxplot_enable)
        self.obs_depth_boxplot_iqr_scale = float(obs_depth_boxplot_iqr_scale)
        self.obs_depth_boxplot_near_iqr_scale = (
            None if obs_depth_boxplot_near_iqr_scale is None else float(obs_depth_boxplot_near_iqr_scale)
        )
        self.obs_depth_boxplot_min_keep_ratio = float(obs_depth_boxplot_min_keep_ratio)
        self.obs_dbscan_enable = bool(obs_dbscan_enable)
        self.obs_dbscan_pre_voxel = float(obs_dbscan_pre_voxel)
        self.obs_dbscan_eps = float(obs_dbscan_eps)
        self.obs_dbscan_min_samples = int(obs_dbscan_min_samples)
        self.obs_dbscan_min_keep_ratio = float(obs_dbscan_min_keep_ratio)
        self.kf_geom_dbscan_enable = bool(kf_geom_dbscan_enable)
        self.kf_geom_dbscan_pre_voxel = float(kf_geom_dbscan_pre_voxel)
        self.kf_geom_dbscan_eps = float(kf_geom_dbscan_eps)
        self.kf_geom_dbscan_min_samples = int(kf_geom_dbscan_min_samples)
        self.kf_geom_dbscan_min_keep_ratio = float(kf_geom_dbscan_min_keep_ratio)
        self.assoc_weight_iou = float(assoc_weight_iou)
        self.assoc_weight_dist = float(assoc_weight_dist)
        self.assoc_weight_label = float(assoc_weight_label)
        self.assoc_stage1_weight_iou = float(assoc_stage1_weight_iou)
        self.assoc_stage1_weight_dist = float(assoc_stage1_weight_dist)
        self.assoc_stage1_weight_label = float(assoc_stage1_weight_label)
        self.assoc_stage1_weight_subtype = float(assoc_stage1_weight_subtype)
        self.assoc_stage1_geom_penalty_weight_c = float(assoc_stage1_geom_penalty_weight_c)
        self.assoc_stage1_continuity_overlap_weight_c = float(assoc_stage1_continuity_overlap_weight_c)
        self.assoc_stage1_continuity_tau_boost_c = float(assoc_stage1_continuity_tau_boost_c)
        self.assoc_stage1_continuity_min_obs_count_c = int(assoc_stage1_continuity_min_obs_count_c)
        self.assoc_stage2_weight_dist = float(assoc_stage2_weight_dist)
        self.assoc_stage2_weight_label = float(assoc_stage2_weight_label)
        self.assoc_stage2_weight_subtype = float(assoc_stage2_weight_subtype)
        self.assoc_diag_gate_scale = float(assoc_diag_gate_scale)
        self.assoc_diag_gate_min_by_role = assoc_diag_gate_min_by_role or {"O": 0.15, "C": 0.08, "U": 0.04}
        self.assoc_diag_gate_max_by_role = assoc_diag_gate_max_by_role or {"O": 1.2, "C": 0.6, "U": 0.35}
        self.assoc_birth_gate_cap_by_role = assoc_birth_gate_cap_by_role or {"O": 0.4, "C": 0.18, "U": 0.10}
        self.assoc_require_positive_overlap = bool(assoc_require_positive_overlap)
        self.assoc_parent_weight_stage1 = 0.10
        self.assoc_parent_weight_stage2 = 0.20
        self.assoc_parent_penalty_weight_stage1 = 0.05
        self.assoc_parent_penalty_weight_stage2 = 0.08
        self.assoc_parent_conflict_thr_stage1 = 0.30
        self.assoc_parent_conflict_thr_stage2 = 0.35
        self.assoc_recent_bbox_gap_max = 2
        self.assoc_recent_bbox_min_overlap = 0.15
        self.assoc_recent_bbox_min_obs_count = 2
        self.anchor_min_obs_count = int(anchor_min_obs_count)
        self.anchor_min_obs_count_by_role = anchor_min_obs_count_by_role or {"O": 2, "C": 2, "U": 3}
        self.anchor_min_kf_support_by_role = anchor_min_kf_support_by_role or {"O": 2, "C": 2, "U": 2}
        self.anchor_pre_stable_accum_roles = anchor_pre_stable_accum_roles or {"O": False, "C": True, "U": True}
        self.anchor_candidate_min_points_by_role = anchor_candidate_min_points_by_role or {"O": 8, "C": 8, "U": 8}
        self.anchor_candidate_min_view_score_by_role = anchor_candidate_min_view_score_by_role or {"O": 0.01, "C": 0.01, "U": 0.01}
        self.anchor_candidate_border_min_points_by_role = anchor_candidate_border_min_points_by_role or {"O": 8, "C": 8, "U": 8}
        self.anchor_candidate_border_min_view_score_by_role = anchor_candidate_border_min_view_score_by_role or {"O": 0.01, "C": 0.01, "U": 0.01}
        self.anchor_candidate_bbox_center_gate_by_role = anchor_candidate_bbox_center_gate_by_role or {"O": 0.35, "C": 0.20, "U": 0.12}
        self.anchor_candidate_centroid_gate_by_role = anchor_candidate_centroid_gate_by_role or {"O": 0.35, "C": 0.20, "U": 0.12}
        self.anchor_carrier_border_promote_min_obs_count = int(anchor_carrier_border_promote_min_obs_count)
        self.anchor_carrier_border_promote_min_points = int(anchor_carrier_border_promote_min_points)
        self.anchor_carrier_border_promote_min_view_score = float(anchor_carrier_border_promote_min_view_score)
        self.anchor_fusion_min_points_by_role = anchor_fusion_min_points_by_role or {"O": 8, "C": 8, "U": 8}
        self.anchor_fusion_min_view_score_by_role = anchor_fusion_min_view_score_by_role or {"O": 0.01, "C": 0.01, "U": 0.01}
        self.anchor_fusion_border_min_points_by_role = anchor_fusion_border_min_points_by_role or {"O": 8, "C": 8, "U": 8}
        self.anchor_fusion_border_min_view_score_by_role = anchor_fusion_border_min_view_score_by_role or {"O": 0.01, "C": 0.01, "U": 0.01}
        self.local_decay = float(local_decay)
        self.local_min_support_frames = int(local_min_support_frames)
        self.local_stable_margin = float(local_stable_margin)
        self.tentative_margin_local = float(tentative_margin_local)
        self.tentative_local_ttl_frames = int(tentative_local_ttl_frames)
        self.tentative_local_replace_margin = float(tentative_local_replace_margin)
        self.tentative_local_lowshot_enable = bool(tentative_local_lowshot_enable)
        self.tentative_local_lowshot_edge_types = set(
            tentative_local_lowshot_edge_types
            if tentative_local_lowshot_edge_types is not None
            else {"O-C", "C-U", "O-U"}
        )
        self.tentative_local_lowshot_min_status_strength_by_edge_type = dict(
            tentative_local_lowshot_min_status_strength_by_edge_type
            if tentative_local_lowshot_min_status_strength_by_edge_type is not None
            else {"C-U": 0.85, "O-C": 0.90, "O-U": 0.95}
        )
        self.tentative_local_lowshot_require_no_competition = bool(
            tentative_local_lowshot_require_no_competition
        )
        self.tentative_local_lowshot_retention_policy = str(
            tentative_local_lowshot_retention_policy or "until_contradicted"
        )
        self.remote_decay = float(remote_decay)
        self.remote_min_support_frames = int(remote_min_support_frames)
        self.remote_stable_margin = float(remote_stable_margin)
        self.tentative_margin_remote = float(tentative_margin_remote)
        self.tentative_remote_ttl_frames = int(tentative_remote_ttl_frames)
        self.tentative_remote_replace_margin = float(tentative_remote_replace_margin)
        self.unresolved_frames_for_llava = int(unresolved_frames_for_llava)
        self.llava_margin_local = float(llava_margin_local)
        self.llava_margin_remote = float(llava_margin_remote)
        self.graph_policy = graph_policy or FunctionalGraphPolicy()

        self.local_weights = {
            "selected": 2.0,
            "mask": 1.0,
            "box": 0.5,
            "vis": 0.2,
            "geom": 0.5,
            "bypass": 0.35,
            "owner": 0.75,
        }
        self.remote_weights = {
            "det": 1.0,
            "conf": 1.0,
            "co_visibility": 0.35,
            "endpoint_maturity": 0.35,
            "distance": 0.45,
        }
        self.local_history_size = 6
        self.local_recent_consistency_frames = 3
        self.local_max_switches = 1
        self.preferred_tau_mask = 0.6
        self.preferred_tau_box = 0.5
        self.preferred_min_recent_support = 2

        self.node_tracks: Dict[str, OnlineNodeTrack] = {}
        self.enable_reanchor = not _ENV_DISABLE_REANCHOR
        self.enable_multi_anchor_assist = not _ENV_DISABLE_AUX_ASSIST
        self.multi_anchor_assist_max_bonus = float(_MULTI_ANCHOR_ASSIST_MAX_BONUS)
        # Node consolidation switches (see module-level _ENV_*).
        self.enable_node_consolidation = not _ENV_DISABLE_NODE_CONSOLIDATION
        self.node_consolidation_dry_run = bool(_ENV_NODE_CONSOLIDATION_DRY_RUN)
        self.node_consolidation_every_kf = int(_ENV_NODE_CONSOLIDATION_EVERY_KF)
        # Per-event log: list of dict events (action / survivor / loser /
        # reason / metrics).  Bounded by simple length cap.
        self.node_consolidation_events: list[dict] = []
        # Bounded log of low-shot edge sibling-duplicate-risk blocks so the
        # final summary JSON can reference each rejection.
        self.lowshot_sibling_block_events: list[dict] = []
        # Bounded log of parent-level unstable-sibling arbitration events
        # produced by ``_refresh_tentative_local_edges``.  Each event
        # describes one (parent, edge_type, role, label) group together
        # with the per-child scores and the accepted / blocked partition.
        self.unstable_sibling_arbitration_events: list[dict] = []
        # Toggles + thresholds for the unstable-sibling arbitration pass.
        self.unstable_sibling_arbitration_enable = (
            not _ENV_DISABLE_UNSTABLE_SIBLING_ARBITRATION
        )
        self.unstable_sibling_best_margin = 0.75
        self.unstable_sibling_require_no_covisibility = True
        # C-role merge safety used by the competition-duplicate path.
        # This remains as a false-merge guard, independent from the removed
        # large-object instance-envelope duplicate pipeline.
        self.carrier_merge_centroid_abs_cap = 0.40
        self.carrier_merge_min_iou = 0.10
        self.carrier_merge_min_overlap = 0.20
        # Configurable label alias groups (defaults to the legacy table for
        # backward compatibility; pass ``alias_groups`` via constructor or
        # set ``label_alias_groups`` after construction to override).  The
        # production default is intentionally NOT empty here because many
        # downstream call-sites still rely on door/handle/knob synonyms;
        # the new helpers in ``node_consolidation`` always default to an
        # empty table per requirement A.
        self.label_alias_groups = list(_LABEL_ALIAS_GROUPS)
        # ---- Orphan-U competition-loser pruning ---------------------
        self.orphan_u_prune_enable = not _ENV_DISABLE_ORPHAN_U_PRUNE
        self.orphan_u_prune_min_loss_count = 2
        self.orphan_u_prune_stale_frames = 8
        self.orphan_u_prune_min_margin = 0.75
        self.orphan_u_prune_quality_gap = 1.0
        self.orphan_u_prune_allow_if_no_remote = True
        # Per-node parent-competition loss history.  Populated by
        # ``_record_parent_competition_loss`` whenever a low-shot edge is
        # blocked by sibling-duplicate-risk or unstable-sibling
        # arbitration.  Consumed by orphan-U pruning.
        self.node_parent_competition_losses: Dict[str, dict] = {}
        # Bounded log of orphan-U pruning events.
        self.orphan_u_prune_events: list[dict] = []
        # ---- Competition-based duplicate consolidation ---------------
        # Triggered when two nodes (1) have no co-visibility, (2) are
        # close in 3D, and (3) have produced an assignment-competition
        # event (parent or child).  Default ON.
        self.enable_competition_duplicate_consolidation = True
        # 3D IoU threshold above which the pair is consolidated by merge
        # (action="merge"); below threshold but with centroid distance
        # passing the half-diag rule, the worse node is absorbed/pruned.
        self.competition_duplicate_iou_thr = 0.25
        # Per-pair competition events ledger and pair counter.
        self.node_assignment_competition_events: list[dict] = []
        self.node_assignment_competition_pair_counts: Dict[
            tuple[str, str], dict
        ] = {}
        self.competition_duplicate_events: list[dict] = []
        self.competition_duplicate_summary: dict = {
            "candidates": 0,
            "rejected_covisible": 0,
            "rejected_missing_geometry": 0,
            "rejected_not_close": 0,
            "rejected_no_competition": 0,
            "rejected_carrier_gate": 0,
            "rejected_aggregate_cabinet": 0,
            "iou_merges": 0,
            "distance_absorb_prunes": 0,
            "dangling_ref_errors": 0,
        }
        # ---- Conservative tentative remote-edge recall ---------------
        self.remote_tentative_enable = not _ENV_DISABLE_REMOTE_TENTATIVE
        self.remote_tentative_min_support_frames = 1
        self.remote_tentative_min_margin = 0.35
        self.remote_tentative_min_endpoint_maturity = 0.25
        self.remote_tentative_min_distance_score = 0.15
        self.remote_tentative_require_recent_covisibility = True
        self.remote_tentative_recent_covis_window = 6
        # Defer to existing ``tentative_remote_ttl_frames`` for TTL.
        # Bounded log of tentative-remote candidacy outcomes.
        self.remote_tentative_events: list[dict] = []
        # ----- Section C: 2D remote evidence backlog -----
        # Some 2D-observed remote relations cannot be turned into a posterior
        # update at the moment they are seen because at least one endpoint
        # detection has no matched node yet (immature track / missing
        # association).  We persist the evidence keyed by relation_key and
        # try to re-resolve it on every keyframe so it can still feed the
        # posterior once the endpoint nodes mature.
        self.remote_2d_backlog_enable = True
        self.remote_2d_evidence_max_per_key = 32
        self.remote_2d_evidence_global_cap = 1024
        self.remote_2d_evidence_ttl_frames = 200
        self.remote_2d_evidence_ledger: dict[str, list[dict]] = {}
        self.remote_2d_backlog_events: list[dict] = []
        self.remote_2d_backlog_summary: dict = {
            "remote_2d_evidence_seen": 0,
            "remote_2d_backlog_size": 0,
            "remote_2d_backlog_resolved": 0,
            "remote_2d_backlog_ambiguous": 0,
            "remote_2d_backlog_expired": 0,
            "remote_2d_backlog_written_tentative": 0,
        }
        # Aggregated counters over a single refresh pass (reset every
        # call to ``_refresh_tentative_remote_edges``).  Surfaced via
        # ``to_dict``.
        self.remote_tentative_summary: dict = {
            "remote_2d_candidates_seen": 0,
            "remote_pairs_scored": 0,
            "remote_tentative_written": 0,
            "remote_committed_written": 0,
            "remote_rejected_endpoint_missing": 0,
            "remote_rejected_no_position": 0,
            "remote_rejected_low_maturity": 0,
            "remote_rejected_distance": 0,
            "remote_rejected_margin": 0,
            "remote_rejected_no_recent_covis": 0,
            "remote_rejected_low_support": 0,
            "remote_rejected_switch_count": 0,
        }
        # ---- Section D: atlas-driven proactive remote candidate ------
        # When LLM/DeepSeek atlas / scene reasoning enumerates allowed
        # remote relations between two object labels, we treat them as
        # *templates*.  Once the corresponding stable nodes form, we
        # generate candidate edges from spatial priors and (optionally)
        # ask LLAVA to confirm at low frequency.  Templates and candidate
        # events are bounded ring-buffers; nothing here writes a
        # *committed* edge directly — the strongest path is a tentative
        # remote edge with ``last_update_source="remote_atlas_spatial"``.
        _ENV_DISABLE_REMOTE_ATLAS = (
            os.environ.get("FG_DISABLE_REMOTE_ATLAS", "0") == "1"
        )
        self.remote_atlas_enable = not _ENV_DISABLE_REMOTE_ATLAS
        self.remote_atlas_min_template_support = 1
        self.remote_atlas_min_endpoint_maturity = 0.35
        self.remote_atlas_min_spatial_score = 0.35
        self.remote_atlas_require_mature_nodes = True
        self.remote_atlas_max_candidates_per_kf = 5
        self.remote_atlas_write_min_score = 0.55
        self.remote_atlas_llava_min_score = 0.35
        self.remote_atlas_llava_every_kf = 5
        self.remote_atlas_llava_max_per_round = 2
        self.remote_atlas_template_max = 1024
        self.remote_atlas_template_source_frames_cap = 8
        self.remote_atlas_distance_scale_default = 2.0
        self.remote_atlas_templates: dict[str, dict] = {}
        self.remote_atlas_candidates_recent: list[dict] = []
        self.remote_atlas_llava_events: list[dict] = []
        self.remote_atlas_summary: dict = {
            "templates_seen": 0,
            "candidates_generated": 0,
            "candidates_written_tentative": 0,
            "candidates_queued_llava": 0,
            "candidates_skipped_low_maturity": 0,
            "candidates_skipped_low_score": 0,
        }
        self.final_temp_cleanup_events: list[dict] = []
        self._remote_atlas_llava_last_kf: int = -1
        self._remote_atlas_llava_dedupe: set[str] = set()
        self._node_consolidation_kf_counter: int = 0
        self.local_posteriors: Dict[str, LocalParentPosterior] = {}
        self.remote_posteriors: Dict[str, RemotePairPosterior] = {}
        # Lightweight cache mapping (edge_type, src_label, dst_label) ->
        # {relation_text: count}. Used to bind 2D-derived relation texts to
        # final 3D persistent edges without re-running any LLM.
        # edge_type ∈ {"O-C", "C-U", "O-U", "remote"}.
        self._relation_text_counts: Dict[tuple, Dict[str, int]] = {}
        self.graph = PersistentFunctionalGraph()
        self.cabinet_aggregator = CabinetAggregator(policy=self.graph_policy)
        self.frame_observations: Dict[int, list[NodeObservation]] = {}
        self._frame_order: deque[int] = deque(maxlen=64)
        self.frame_assoc_debug: Dict[int, dict] = {}
        self._assoc_debug_order: deque[int] = deque(maxlen=128)
        self.frame_extract_debug: Dict[int, list[dict]] = {}
        self._extract_debug_order: deque[int] = deque(maxlen=128)
        self._last_obs_extract_stats: dict[str, int] = {
            "num_obs_dropped_low_points": 0,
            "num_obs_depth_boxplot_changed": 0,
            "num_obs_dbscan_changed": 0,
        }
        self._last_obs_extract_debug: list[dict] = []
        self._next_node_idx = 1
        self._assoc_det_to_obs_context: Optional[Dict[int, NodeObservation]] = None
        self._latest_keyframes = None
        self._latest_local_assignment_signals: Dict[str, Dict[str, dict]] = {}
        self._latest_current_parent_claims: Dict[str, Dict[str, dict]] = {}
        self._latest_current_parent_claims_frame_idx: Optional[int] = None
        # Debug snapshot of lowshot decisions per (child_id, parent_id), latest frame only.
        self._latest_local_lowshot_debug: Dict[str, Dict[str, dict]] = {}

    @staticmethod
    def _role_cfg_int(mapping: Dict[str, int], role: str, default: int) -> int:
        try:
            return int(mapping.get(role, default))
        except Exception:
            return int(default)

    @staticmethod
    def _role_cfg_float(mapping: Dict[str, float], role: str, default: float) -> float:
        try:
            return float(mapping.get(role, default))
        except Exception:
            return float(default)

    def _should_enable_pre_stable_accum(self, track: OnlineNodeTrack, obs: NodeObservation) -> bool:
        if getattr(track, "stable_geom_ready", False):
            return False
        if getattr(track, "candidate_anchor_kf_id", None) is None:
            return False
        role = str(getattr(track, "role", getattr(obs, "role", "U")) or "U")
        return bool(self.anchor_pre_stable_accum_roles.get(role, False))

    def _update_pre_stable_local_geometry(self, track: OnlineNodeTrack, kf_idx: int, obs: NodeObservation, keyframes) -> bool:
        role = str(getattr(track, "role", getattr(obs, "role", "U")) or "U")
        return track.consider_provisional_anchor(
            kf_idx,
            keyframes,
            obs,
            enable_local_accum=self._should_enable_pre_stable_accum(track, obs),
            min_points=self._role_cfg_int(self.anchor_fusion_min_points_by_role, role, self.min_points_per_obs),
            min_view_score=self._role_cfg_float(self.anchor_fusion_min_view_score_by_role, role, 0.0),
            allow_border=(role == "C"),
            border_min_points=self._role_cfg_int(self.anchor_fusion_border_min_points_by_role, role, self.min_points_per_obs),
            border_min_view_score=self._role_cfg_float(self.anchor_fusion_border_min_view_score_by_role, role, 0.0),
        )

    def _new_node_id(self, role: str) -> str:
        node_id = f"{role}{self._next_node_idx:04d}"
        self._next_node_idx += 1
        return node_id

    def _remember_frame_observations(self, frame_idx: int, observations: list[NodeObservation]) -> None:
        self.frame_observations[frame_idx] = observations
        self._frame_order.append(frame_idx)
        while len(self._frame_order) > self._frame_order.maxlen:
            stale = self._frame_order.popleft()
            self.frame_observations.pop(stale, None)

    @staticmethod
    def _pick_local_stage(local_rel_debug: dict) -> dict:
        local_rel = local_rel_debug.get("local_rel", {}) if isinstance(local_rel_debug, dict) else {}
        for key in ("after_cov", "after_r4", "after_r3", "after_r2", "after_r1", "raw"):
            if key in local_rel:
                return local_rel[key]
        return {}

    @staticmethod
    def _local_assignment_group(local_stage: dict, child_idx: int) -> Optional[dict]:
        assignments = local_stage.get("assignments", {}) if isinstance(local_stage, dict) else {}
        if not isinstance(assignments, dict):
            return None
        group = assignments.get(child_idx)
        if group is None:
            group = assignments.get(str(child_idx))
        return group if isinstance(group, dict) else None

    @staticmethod
    def _local_assignment_status_strength(status: str, edge: Optional[dict]) -> float:
        if edge is not None and bool(edge.get("selected")):
            return 1.0
        text = str(status or "").strip().lower()
        if text == "confirmed_mask":
            return 1.0
        if text == "confirmed":
            return 0.85
        return 0.0

    def _chosen_edge_from_assignment(self, local_stage: dict, child_idx: int, edge_type: str) -> Optional[dict]:
        group = self._local_assignment_group(local_stage, child_idx)
        if group is None:
            return None
        assignment = group.get(edge_type)
        if not isinstance(assignment, dict):
            return None

        edges = local_stage.get("edges", []) if isinstance(local_stage, dict) else []
        if not isinstance(edges, list):
            return None

        def _edge_by_eid(eid: Any) -> Optional[dict]:
            try:
                idx = int(eid)
            except Exception:
                return None
            if idx < 0 or idx >= len(edges):
                return None
            edge = edges[idx]
            if not isinstance(edge, dict):
                return None
            if int(edge.get("child_idx", -1)) != int(child_idx):
                return None
            if str(edge.get("type") or "") != edge_type:
                return None
            return edge

        chosen = _edge_by_eid(assignment.get("chosen_eid"))
        if chosen is not None:
            return chosen

        chosen_parent_idx = assignment.get("chosen_parent_idx")
        candidate_eids = assignment.get("candidate_eids", [])
        if isinstance(candidate_eids, list):
            for eid in candidate_eids:
                edge = _edge_by_eid(eid)
                if edge is None:
                    continue
                try:
                    if chosen_parent_idx is not None and int(edge.get("parent_idx", -1)) == int(chosen_parent_idx):
                        return edge
                except Exception:
                    continue

            for eid in candidate_eids:
                edge = _edge_by_eid(eid)
                if edge is not None and bool(edge.get("selected")):
                    return edge

            for eid in candidate_eids:
                edge = _edge_by_eid(eid)
                if edge is not None:
                    return edge

        return None

    def _derive_observation_semantic_subtype(self, det_idx: int, label: str, role: str, local_stage: dict) -> dict:
        payload = {
            "semantic_subtype": None,
            "semantic_subtype_conf": 0.0,
            "semantic_parent_label": None,
            "semantic_parent_role": None,
            "semantic_owner_object_label": None,
        }
        if role not in {"C", "U"}:
            return payload

        norm_label = normalize_label(label)
        if not norm_label:
            return payload
        if self.graph_policy.is_suppressed_object(norm_label) or self.graph_policy.is_hint_only_object(norm_label):
            return payload

        if not isinstance(local_stage, dict):
            return payload

        edge_priority = [("O-C", "O")] if role == "C" else [("C-U", "C"), ("O-U", "O")]
        for edge_type, default_parent_role in edge_priority:
            group = self._local_assignment_group(local_stage, det_idx)
            assignment = group.get(edge_type) if isinstance(group, dict) else None
            if not isinstance(assignment, dict):
                continue

            edge = self._chosen_edge_from_assignment(local_stage, det_idx, edge_type)
            conf = self._local_assignment_status_strength(str(assignment.get("status") or ""), edge)
            if edge is None or conf <= 0.0:
                continue

            parent_label = normalize_label(edge.get("parent_label"))
            if not parent_label:
                continue
            if self.graph_policy.is_suppressed_object(parent_label) or self.graph_policy.is_hint_only_object(parent_label):
                continue

            parent_role = str(edge.get("parent_role") or default_parent_role)
            subtype = f"{norm_label}@{parent_label}"
            owner_object_label: Optional[str] = None
            if role == "C":
                owner_object_label = parent_label if parent_role == "O" else None
            elif role == "U":
                if parent_role == "O":
                    owner_object_label = parent_label
                elif parent_role == "C":
                    carrier_idx = int(edge.get("parent_idx", -1))
                    carrier_group = self._local_assignment_group(local_stage, carrier_idx)
                    carrier_assignment = carrier_group.get("O-C") if isinstance(carrier_group, dict) else None
                    if isinstance(carrier_assignment, dict):
                        carrier_edge = self._chosen_edge_from_assignment(local_stage, carrier_idx, "O-C")
                        carrier_conf = self._local_assignment_status_strength(
                            str(carrier_assignment.get("status") or ""),
                            carrier_edge,
                        )
                        if carrier_edge is not None and carrier_conf > 0.0:
                            owner_candidate = normalize_label(carrier_edge.get("parent_label"))
                            if owner_candidate and not self.graph_policy.is_suppressed_object(owner_candidate) and not self.graph_policy.is_hint_only_object(owner_candidate):
                                owner_object_label = owner_candidate

            payload.update(
                {
                    "semantic_subtype": subtype,
                    "semantic_subtype_conf": float(conf),
                    "semantic_parent_label": parent_label,
                    "semantic_parent_role": parent_role,
                    "semantic_owner_object_label": owner_object_label,
                }
            )
            return payload

        return payload

    # ------------------------------------------------------------------
    # Relation-text cache: bind 2D-derived oc/cu/ou/remote relation strings
    # to instance-level edges in the persistent graph.  No LLM is invoked;
    # all texts come straight from frame_result.
    # ------------------------------------------------------------------
    def _record_relation_text(
        self,
        edge_type: str,
        src_label: str,
        dst_label: str,
        relation_text: Optional[str],
        source: str = "",
    ) -> None:
        del source  # reserved for future debugging; kept for API stability
        if relation_text is None:
            return
        text = str(relation_text).strip()
        if not text:
            return
        src = normalize_label(src_label)
        dst = normalize_label(dst_label)
        if not src or not dst:
            return
        key = (str(edge_type), src, dst)
        bucket = self._relation_text_counts.setdefault(key, {})
        bucket[text] = int(bucket.get(text, 0)) + 1

    def _best_relation_text(self, edge_type: str, src_label: str, dst_label: str) -> str:
        src = normalize_label(src_label)
        dst = normalize_label(dst_label)
        if not src or not dst:
            return ""
        bucket = self._relation_text_counts.get((str(edge_type), src, dst))
        if not bucket:
            return ""
        # Most-frequent text wins; ties broken by lexical order for determinism.
        best_text = ""
        best_count = -1
        for text, count in sorted(bucket.items()):
            if int(count) > best_count:
                best_text = text
                best_count = int(count)
        return best_text

    def _relation_text_for_instance_edge(
        self,
        src_node_id: str,
        dst_node_id: str,
        edge_type: str,
    ) -> str:
        src_track = self.node_tracks.get(src_node_id)
        dst_track = self.node_tracks.get(dst_node_id)
        src_label = getattr(src_track, "label", "") if src_track is not None else ""
        dst_label = getattr(dst_track, "label", "") if dst_track is not None else ""
        # Synthetic aggregate-cabinet parent has no detection; the 2D relation
        # cache is keyed on the literal label "cabinet".
        if not src_label and str(src_node_id).startswith("O_CABINET_"):
            src_label = "cabinet"
        return self._best_relation_text(edge_type, src_label, dst_label)

    def _update_relation_text_cache_from_frame_result(self, frame_result: dict) -> dict:
        """Populate the relation-text cache with oc/cu/ou and remote relation
        strings carried inside ``frame_result``.

        Returns a small debug dict summarising how many entries were added,
        purely for inspection in tests.
        """
        added = {"O-C": 0, "C-U": 0, "O-U": 0, "remote": 0}
        if not isinstance(frame_result, dict):
            return added

        for entry in frame_result.get("present") or []:
            if not isinstance(entry, dict):
                continue
            obj_label = entry.get("object")
            for carrier_entry in entry.get("functional_carriers", []) or []:
                if not isinstance(carrier_entry, dict):
                    continue
                carrier_label = carrier_entry.get("carrier")
                oc_relation = carrier_entry.get("oc_relation")
                if oc_relation:
                    self._record_relation_text("O-C", obj_label, carrier_label, oc_relation, "present")
                    added["O-C"] += 1
                for unit_entry in carrier_entry.get("interactive_units", []) or []:
                    if not isinstance(unit_entry, dict):
                        continue
                    unit_label = unit_entry.get("unit")
                    cu_relation = unit_entry.get("cu_relation")
                    if cu_relation:
                        self._record_relation_text("C-U", carrier_label, unit_label, cu_relation, "present")
                        added["C-U"] += 1
                    ou_relation = unit_entry.get("ou_relation")
                    if ou_relation:
                        self._record_relation_text("O-U", obj_label, unit_label, ou_relation, "present")
                        added["O-U"] += 1
            for direct_unit in entry.get("direct_interactive_units", []) or []:
                if not isinstance(direct_unit, dict):
                    continue
                unit_label = direct_unit.get("unit")
                ou_relation = direct_unit.get("ou_relation")
                if ou_relation:
                    self._record_relation_text("O-U", obj_label, unit_label, ou_relation, "present")
                    added["O-U"] += 1

        chains = frame_result.get("local_function_chains") or []
        if isinstance(chains, list):
            for chain in chains:
                if not isinstance(chain, dict):
                    continue
                obj_label = chain.get("object")
                carrier_label = chain.get("carrier")
                unit_label = chain.get("unit")
                oc_relation = chain.get("oc_relation")
                cu_relation = chain.get("cu_relation")
                ou_relation = chain.get("ou_relation")
                if oc_relation and obj_label and carrier_label:
                    self._record_relation_text("O-C", obj_label, carrier_label, oc_relation, "chain")
                    added["O-C"] += 1
                if cu_relation and carrier_label and unit_label:
                    self._record_relation_text("C-U", carrier_label, unit_label, cu_relation, "chain")
                    added["C-U"] += 1
                if ou_relation and obj_label and unit_label:
                    self._record_relation_text("O-U", obj_label, unit_label, ou_relation, "chain")
                    added["O-U"] += 1

        for item in frame_result.get("remote_relation_candidates") or []:
            if not isinstance(item, dict):
                continue
            from_obj = item.get("from_object")
            to_obj = item.get("to_object")
            relation_text = item.get("relation")
            if relation_text:
                self._record_relation_text("remote", from_obj, to_obj, relation_text, "remote_candidate")
                added["remote"] += 1

        return added

    def derive_owner_prior_from_present(self, frame_result: dict) -> dict:
        u_preferred: dict[str, set[str]] = defaultdict(set)
        u_fallback: dict[str, set[str]] = defaultdict(set)
        u_owner_mode: dict[str, str] = {}
        u_preferred_role: dict[str, str] = {}
        u_is_direct: dict[str, bool] = defaultdict(bool)
        c_preferred: dict[str, set[str]] = defaultdict(set)
        u_direct_object_contexts: dict[str, set[str]] = defaultdict(set)
        u_carrier_contexts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        c_object_contexts: dict[str, set[str]] = defaultdict(set)

        for entry in frame_result.get("present", []) or []:
            obj = normalize_label(entry.get("object"))
            if not obj:
                continue
            if self.graph_policy.is_suppressed_object(obj):
                continue
            is_hint_only = self.graph_policy.is_hint_only_object(obj)

            for carrier_entry in entry.get("functional_carriers", []) or []:
                carrier = normalize_label(carrier_entry.get("carrier"))
                if not carrier:
                    continue
                if not is_hint_only:
                    c_preferred[carrier].add(obj)
                    c_object_contexts[carrier].add(obj)
                for unit_entry in carrier_entry.get("interactive_units", []) or []:
                    unit = unit_entry.get("unit") if isinstance(unit_entry, dict) else unit_entry
                    unit = normalize_label(unit)
                    if not unit:
                        continue
                    u_preferred[unit].add(carrier)
                    if not is_hint_only:
                        u_fallback[unit].add(obj)
                        u_carrier_contexts[unit][carrier].add(obj)
                    u_owner_mode[unit] = "prefer_carrier"
                    u_preferred_role[unit] = "C"

            for unit_entry in entry.get("direct_interactive_units", []) or []:
                unit = unit_entry.get("unit") if isinstance(unit_entry, dict) else unit_entry
                unit = normalize_label(unit)
                if not unit:
                    continue
                if is_hint_only:
                    continue
                u_direct_object_contexts[unit].add(obj)
                if u_owner_mode.get(unit) != "prefer_carrier":
                    u_preferred[unit].add(obj)
                    u_owner_mode[unit] = "prefer_object"
                    u_preferred_role[unit] = "O"
                u_is_direct[unit] = True

        return {
            "u_preferred_parents": {key: set(value) for key, value in u_preferred.items()},
            "u_fallback_parents": {key: set(value) for key, value in u_fallback.items()},
            "u_owner_mode": dict(u_owner_mode),
            "u_preferred_parent_role": dict(u_preferred_role),
            "u_is_direct": dict(u_is_direct),
            "c_preferred_parents": {key: set(value) for key, value in c_preferred.items()},
            "u_direct_object_contexts": {key: set(value) for key, value in u_direct_object_contexts.items()},
            "u_carrier_contexts": {
                unit: {carrier: set(objects) for carrier, objects in carriers.items()}
                for unit, carriers in u_carrier_contexts.items()
            },
            "c_object_contexts": {key: set(value) for key, value in c_object_contexts.items()},
        }

    def _local_owner_context_edges(self, local_stage: dict, det_idx: int, edge_type: str) -> list[dict]:
        if not isinstance(local_stage, dict):
            return []
        edges = local_stage.get("edges", []) if isinstance(local_stage, dict) else []
        candidates: list[dict] = []
        seen: set[tuple] = set()

        def _append(edge: Optional[dict]) -> None:
            if not isinstance(edge, dict):
                return
            try:
                if int(edge.get("child_idx", -1)) != int(det_idx):
                    return
            except Exception:
                return
            if str(edge.get("type") or "") != edge_type:
                return
            key = (
                int(edge.get("child_idx", -1)),
                int(edge.get("parent_idx", -1)),
                str(edge.get("type") or ""),
                normalize_label(edge.get("parent_label")),
            )
            if key in seen:
                return
            seen.add(key)
            candidates.append(edge)

        _append(self._chosen_edge_from_assignment(local_stage, int(det_idx), edge_type))
        for edge in edges:
            _append(edge)

        def _strength(edge: dict) -> tuple[float, float, float]:
            selected = 1.0 if edge.get("selected") else 0.0
            pass_thr = 1.0 if edge.get("pass_thr") else 0.0
            contain = max(float(edge.get("contain") or 0.0), float(edge.get("mask_contain") or 0.0))
            return (max(selected, pass_thr, contain), selected, contain)

        candidates.sort(key=_strength, reverse=True)
        return candidates

    def _edge_has_strong_owner_context(self, edge: dict) -> bool:
        return (
            bool(edge.get("selected"))
            or bool(edge.get("pass_thr"))
            or float(edge.get("mask_contain") or 0.0) >= self.preferred_tau_mask
            or float(edge.get("contain") or 0.0) >= self.preferred_tau_box
        )

    def _resolve_u_owner_context(
        self,
        *,
        det_idx: int,
        unit_label: str,
        owner_prior: dict,
        local_stage: dict,
    ) -> Optional[dict]:
        unit = normalize_label(unit_label)
        if not unit:
            return None

        direct_objects = set(owner_prior.get("u_direct_object_contexts", {}).get(unit, set()))
        carrier_contexts = owner_prior.get("u_carrier_contexts", {}).get(unit, {}) or {}
        carrier_contexts = {normalize_label(carrier): set(objects) for carrier, objects in carrier_contexts.items()}

        # Prefer the parent role actually selected for this detection. This makes
        # owner mode instance/context-level: a kettle handle may be direct O-U
        # while a cabinet door handle in the same frame remains C-U.
        for edge in self._local_owner_context_edges(local_stage, det_idx, "O-U"):
            if not self._edge_has_strong_owner_context(edge):
                continue
            parent_label = normalize_label(edge.get("parent_label"))
            if parent_label and parent_label in direct_objects:
                return {
                    "allowed_parent_labels": {parent_label},
                    "preferred_parent_labels": {parent_label},
                    "fallback_parent_labels": set(),
                    "preferred_parent_role": "O",
                    "semantic_owner_mode": "prefer_object",
                    "is_direct_unit": True,
                    "source": "direct_object_context",
                }

        for edge in self._local_owner_context_edges(local_stage, det_idx, "C-U"):
            if not self._edge_has_strong_owner_context(edge):
                continue
            parent_label = normalize_label(edge.get("parent_label"))
            if parent_label and parent_label in carrier_contexts:
                fallback_objects = set(carrier_contexts.get(parent_label, set()))
                return {
                    "allowed_parent_labels": {parent_label, *fallback_objects},
                    "preferred_parent_labels": {parent_label},
                    "fallback_parent_labels": fallback_objects,
                    "preferred_parent_role": "C",
                    "semantic_owner_mode": "prefer_carrier",
                    "is_direct_unit": False,
                    "source": "carrier_context",
                }

        if direct_objects and not carrier_contexts:
            return {
                "allowed_parent_labels": set(direct_objects),
                "preferred_parent_labels": set(direct_objects),
                "fallback_parent_labels": set(),
                "preferred_parent_role": "O",
                "semantic_owner_mode": "prefer_object",
                "is_direct_unit": True,
                "source": "direct_object_context_label_only",
            }
        return None

    def _build_semantic_metadata(self, frame_result: dict) -> tuple[Dict[str, int], Dict[str, set[str]], Dict[str, set[str]]]:
        label_to_rank: Dict[str, int] = {}
        owner_prior = self.derive_owner_prior_from_present(frame_result)
        u_preferred = owner_prior.get("u_preferred_parents", {})
        u_fallback = owner_prior.get("u_fallback_parents", {})
        c_preferred = owner_prior.get("c_preferred_parents", {})

        u_to_allowed: dict[str, set[str]] = defaultdict(set, {key: set(value) for key, value in u_preferred.items()})
        c_to_allowed: dict[str, set[str]] = defaultdict(set, {key: set(value) for key, value in c_preferred.items()})

        for key, parents in u_fallback.items():
            u_to_allowed[key].update(parents)

        for entry in frame_result.get("present", []) or []:
            obj = normalize_label(entry.get("object"))
            if not obj:
                continue
            if self.graph_policy.is_suppressed_object(obj):
                continue
            label_to_rank[obj] = min(label_to_rank.get(obj, ROLE_RANK["O"]), ROLE_RANK["O"])

            for carrier_entry in entry.get("functional_carriers", []) or []:
                carrier = normalize_label(carrier_entry.get("carrier"))
                if not carrier:
                    continue
                label_to_rank[carrier] = min(label_to_rank.get(carrier, ROLE_RANK["C"]), ROLE_RANK["C"])
                for unit_entry in carrier_entry.get("interactive_units", []) or []:
                    unit = unit_entry.get("unit") if isinstance(unit_entry, dict) else unit_entry
                    unit = normalize_label(unit)
                    if not unit:
                        continue
                    label_to_rank[unit] = min(label_to_rank.get(unit, ROLE_RANK["U"]), ROLE_RANK["U"])

            for unit_entry in entry.get("direct_interactive_units", []) or []:
                unit = unit_entry.get("unit") if isinstance(unit_entry, dict) else unit_entry
                unit = normalize_label(unit)
                if not unit:
                    continue
                label_to_rank[unit] = min(label_to_rank.get(unit, ROLE_RANK["U"]), ROLE_RANK["U"])

        return label_to_rank, {key: set(value) for key, value in u_to_allowed.items()}, {key: set(value) for key, value in c_to_allowed.items()}

    @staticmethod
    def _edge_type_for_roles(parent_role: str, child_role: str) -> Optional[str]:
        mapping = {
            ("O", "C"): "O-C",
            ("C", "U"): "C-U",
            ("O", "U"): "O-U",
        }
        return mapping.get((parent_role, child_role))

    @staticmethod
    def _remote_relation_key(src_label: str, relation_text: str, dst_label: str) -> str:
        return "|".join([normalize_label(src_label), normalize_label(relation_text), normalize_label(dst_label)])

    @staticmethod
    def _remote_graph_edge_key(relation_key: str, src_node_id: str, dst_node_id: str) -> str:
        """Pair-specific key used for ``graph.remote_edges``.

        The label-level ``relation_key`` (src_label|relation_text|dst_label)
        groups posteriors but is too coarse for the graph dictionary itself:
        if two source instances both target the same destination (e.g. two
        ``electric outlet`` nodes both linked to the same ``microwave``),
        a label-level key would silently overwrite one with the other.
        Encode the endpoint node ids in the dictionary key so distinct node
        pairs survive as independent edges.  Node ids never contain ``|``,
        so the original relation_key can be recovered with ``rsplit('|', 2)``.
        """
        return f"{relation_key}|{src_node_id}|{dst_node_id}"

    @staticmethod
    def _split_remote_graph_edge_key(graph_edge_key: str) -> tuple[str, str, str]:
        """Inverse of ``_remote_graph_edge_key`` (best-effort).

        Returns ``(relation_key, src_node_id, dst_node_id)`` for keys that
        carry the new pair-specific suffix; for legacy/label-only keys the
        node-id slots come back empty.
        """
        parts = graph_edge_key.rsplit("|", 2)
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        return graph_edge_key, "", ""

    def _track_geom_maturity_score(self, track: Optional[OnlineNodeTrack]) -> float:
        if track is None:
            return 0.0
        source = str(track.assoc_geom_source() or "")
        if source == "stable_anchor_fused":
            return 1.0
        if source == "stable_anchor":
            return 0.9
        if source == "provisional_anchor":
            return 0.65
        if source == "candidate_anchor":
            return 0.55
        if source == "world_bbox":
            return 0.30
        if source == "world_centroid":
            return 0.20
        return 0.0

    def _track_world_position(
        self,
        node_id: str,
        *,
        obs_by_node_id: Optional[Dict[str, NodeObservation]] = None,
    ) -> Optional[torch.Tensor]:
        if obs_by_node_id is not None:
            obs = obs_by_node_id.get(node_id)
            if obs is not None and getattr(obs, "centroid_world", None) is not None:
                return obs.centroid_world
        track = self.node_tracks.get(node_id)
        if track is None:
            return None
        keyframes = self._latest_keyframes
        try:
            predicted = track.predict_centroid_world(keyframes)
        except Exception:
            predicted = None
        if predicted is not None:
            return predicted
        return track.world_centroid_tensor()

    @staticmethod
    def _relation_aware_distance_prior(relation_text: str, distance_m: Optional[float]) -> float:
        if distance_m is None:
            return 0.0
        relation = normalize_label(relation_text)
        dist = max(0.0, float(distance_m))
        if relation in {"attached to", "mounted on", "mounted to"}:
            return max(0.0, 1.0 - min(dist / 0.45, 1.0))
        if relation in {"provide power", "provides power", "controls", "control", "switches"}:
            target = 0.75
            spread = 0.85
            return max(0.0, 1.0 - min(abs(dist - target) / spread, 1.0))
        return max(0.0, 1.0 - min(dist / 1.5, 1.0))

    def _score_remote_pair_candidate(
        self,
        item: dict,
        *,
        src_id: str,
        dst_id: str,
        obs_by_node_id: Dict[str, NodeObservation],
    ) -> dict:
        semantic_relation_score = self.remote_weights["det"] * 1.0 + self.remote_weights["conf"] * min(
            float(item.get("from_score") or 0.0),
            float(item.get("to_score") or 0.0),
        )
        co_visibility_score = self.remote_weights["co_visibility"] if src_id in obs_by_node_id and dst_id in obs_by_node_id else 0.0
        src_track = self.node_tracks.get(src_id)
        dst_track = self.node_tracks.get(dst_id)
        endpoint_maturity_score = self.remote_weights["endpoint_maturity"] * 0.5 * (
            self._track_geom_maturity_score(src_track) + self._track_geom_maturity_score(dst_track)
        )
        src_pos = self._track_world_position(src_id, obs_by_node_id=obs_by_node_id)
        dst_pos = self._track_world_position(dst_id, obs_by_node_id=obs_by_node_id)
        distance_m = None
        if src_pos is not None and dst_pos is not None:
            distance_m = float(world_distance(src_pos, dst_pos))
        relation_aware_3d_distance_score = self.remote_weights["distance"] * self._relation_aware_distance_prior(
            str(item.get("relation") or ""),
            distance_m,
        )
        total = semantic_relation_score + co_visibility_score + endpoint_maturity_score + relation_aware_3d_distance_score
        return {
            "semantic_relation_score": float(semantic_relation_score),
            "co_visibility_score": float(co_visibility_score),
            "endpoint_maturity_score": float(endpoint_maturity_score),
            "relation_aware_3d_distance_score": float(relation_aware_3d_distance_score),
            "distance_m": None if distance_m is None else float(distance_m),
            "total_score": float(total),
        }

    def extract_frame_observations(self, frame, frame_result: dict, sam3_out) -> list[NodeObservation]:
        det = getattr(sam3_out, "det", None)
        if det is None or getattr(det, "labels", None) is None or frame.X_canon is None or frame.C is None:
            self._last_obs_extract_debug = []
            return []

        label_to_rank = getattr(det, "label_to_rank", None)
        u_to_allowed = getattr(det, "u_to_allowed_parents", None)
        c_to_allowed = getattr(det, "c_to_allowed_parents", None)
        owner_prior = self.derive_owner_prior_from_present(frame_result)
        if label_to_rank is None or u_to_allowed is None or c_to_allowed is None:
            label_to_rank, u_to_allowed, c_to_allowed = self._build_semantic_metadata(frame_result)

        boxes = getattr(det, "boxes", None)
        masks = getattr(det, "masks", None)
        scores = getattr(det, "scores", None)
        height, width = frame_hw(frame)
        frame_area = height * width
        observations: list[NodeObservation] = []
        dropped_low_points = 0
        depth_boxplot_changed = 0
        dbscan_changed = 0
        extract_debug: list[dict] = []
        cabinet_hits = (getattr(det, "cabinet_hints", None) or {}).get("carrier_hits", []) or []
        cabinet_aggregation_enabled = self.graph_policy.should_enable_cabinet_aggregation()
        local_stage = self._pick_local_stage(getattr(det, "local_rel_debug", {}) or {})
        local_edges = local_stage.get("edges", []) if isinstance(local_stage, dict) else []
        local_parent_det_idxs_by_child: dict[int, set[int]] = defaultdict(set)
        local_child_det_idxs_by_parent: dict[int, set[int]] = defaultdict(set)
        strong_parent_det_idxs_by_child: dict[int, set[int]] = defaultdict(set)
        selected_linked_idxs: set[int] = set()
        drop_det_idxs: set[int] = set()
        if self.graph_policy.should_drop_lid_cap_overlap() and boxes is not None:
            labels_norm = [normalize_label(label) for label in (det.labels or [])]

            def _box_list(idx: int) -> Optional[list[float]]:
                try:
                    box = boxes[idx]
                    if torch.is_tensor(box):
                        return [float(v) for v in box.detach().cpu().reshape(-1).tolist()[:4]]
                    return [float(v) for v in list(box)[:4]]
                except Exception:
                    return None

            cap_boxes = [_box_list(i) for i, lab in enumerate(labels_norm) if lab == "cap"]
            cap_boxes = [box for box in cap_boxes if box is not None]
            if cap_boxes:
                thr = float(getattr(self.graph_policy, "cap_lid_overlap_iou_thr", 0.9) or 0.9)
                for i, lab in enumerate(labels_norm):
                    if lab != "lid":
                        continue
                    lid_box = _box_list(i)
                    if lid_box is None:
                        continue
                    if any(box_iou_xyxy(lid_box, cap_box) >= thr for cap_box in cap_boxes):
                        drop_det_idxs.add(int(i))
        for edge in local_edges:
            child_idx = int(edge.get("child_idx", -1))
            parent_idx = int(edge.get("parent_idx", -1))
            if child_idx < 0 or parent_idx < 0 or child_idx == parent_idx:
                continue
            local_parent_det_idxs_by_child[child_idx].add(parent_idx)
            local_child_det_idxs_by_parent[parent_idx].add(child_idx)
            strong_now = (
                bool(edge.get("selected"))
                or float(edge.get("mask_contain") or 0.0) >= self.preferred_tau_mask
                or float(edge.get("contain") or 0.0) >= self.preferred_tau_box
            )
            if strong_now:
                strong_parent_det_idxs_by_child[child_idx].add(parent_idx)
            if edge.get("selected"):
                selected_linked_idxs.add(child_idx)
                selected_linked_idxs.add(parent_idx)
        carrier_hit_by_idx: dict[int, dict[str, Any]] = {}
        if cabinet_aggregation_enabled:
            for hit in cabinet_hits:
                if not hit.get("pass_thr"):
                    continue
                child_idx = int(hit.get("child_idx", -1))
                if child_idx < 0:
                    continue
                cur = carrier_hit_by_idx.get(child_idx)
                score = float(hit.get("contain") or 0.0)
                if cur is None or score > float(cur.get("contain") or 0.0):
                    carrier_hit_by_idx[child_idx] = hit

        for det_idx, raw_label in enumerate(det.labels or []):
            label = normalize_label(raw_label)
            if not label:
                continue
            if self.graph_policy.should_drop_lid_node(label):
                extract_debug.append(
                    {
                        "det_idx": int(det_idx),
                        "raw_label": str(raw_label),
                        "label": label,
                        "role": role_from_rank(label_to_rank.get(label, ROLE_RANK["O"])),
                        "status": "skipped_lid_policy",
                    }
                )
                continue
            if int(det_idx) in drop_det_idxs:
                extract_debug.append(
                    {
                        "det_idx": int(det_idx),
                        "raw_label": str(raw_label),
                        "label": label,
                        "role": role_from_rank(label_to_rank.get(label, ROLE_RANK["O"])),
                        "status": "skipped_lid_cap_overlap",
                    }
                )
                continue
            if self.graph_policy.should_skip_standard_node(label):
                extract_debug.append(
                    {
                        "det_idx": int(det_idx),
                        "raw_label": str(raw_label),
                        "label": label,
                        "role": role_from_rank(label_to_rank.get(label, ROLE_RANK["O"])),
                        "status": "skipped_policy",
                    }
                )
                continue
            role = role_from_rank(label_to_rank.get(label, ROLE_RANK["O"]))
            if masks is not None:
                mask = masks[det_idx]
            elif boxes is not None:
                mask = box_to_mask(boxes[det_idx], height, width)
            else:
                extract_debug.append(
                    {
                        "det_idx": int(det_idx),
                        "raw_label": str(raw_label),
                        "label": label,
                        "role": role,
                        "status": "missing_mask_and_box",
                    }
                )
                continue

            extracted = extract_points_from_mask(
                frame,
                mask,
                max_points=self.max_points_per_obs,
                conf_thr=self.point_conf_thr,
                min_points=self.min_points_per_obs,
            )
            if extracted is None:
                dropped_low_points += 1
                extract_debug.append(
                    {
                        "det_idx": int(det_idx),
                        "raw_label": str(raw_label),
                        "label": label,
                        "role": role,
                        "status": "low_points",
                    }
                )
                continue
            points_frame, conf, mask_area = extracted
            depth_boxplot_filtered = False
            if self.obs_depth_boxplot_enable:
                depth_keep_mask = depth_boxplot_filter_mask(
                    points_frame,
                    iqr_scale=self.obs_depth_boxplot_iqr_scale,
                    near_iqr_scale=self.obs_depth_boxplot_near_iqr_scale,
                    min_keep_ratio=self.obs_depth_boxplot_min_keep_ratio,
                )
                depth_boxplot_filtered = int(depth_keep_mask.sum().item()) < points_frame.shape[0]
                if depth_boxplot_filtered:
                    depth_boxplot_changed += 1
                    points_frame = points_frame[depth_keep_mask]
                    conf = conf[depth_keep_mask]

            if points_frame.shape[0] < self.min_points_per_obs:
                extract_debug.append(
                    {
                        "det_idx": int(det_idx),
                        "raw_label": str(raw_label),
                        "label": label,
                        "role": role,
                        "status": "after_depth_boxplot_low_points",
                        "depth_boxplot_filtered": bool(depth_boxplot_filtered),
                    }
                )
                continue

            dbscan_filtered = False
            if self.obs_dbscan_enable and not points_frame.is_cuda:
                points_frame_clean = dbscan_filter_points(
                    points_frame,
                    eps=self.obs_dbscan_eps,
                    min_samples=self.obs_dbscan_min_samples,
                    min_keep_ratio=self.obs_dbscan_min_keep_ratio,
                    pre_voxel_size=self.obs_dbscan_pre_voxel,
                )
                if points_frame_clean.shape[0] >= self.min_points_per_obs:
                    dbscan_filtered = points_frame_clean.shape[0] < points_frame.shape[0]
                    if dbscan_filtered:
                        dbscan_changed += 1
                    points_frame = points_frame_clean

            if points_frame.shape[0] < self.min_points_per_obs:
                extract_debug.append(
                    {
                        "det_idx": int(det_idx),
                        "raw_label": str(raw_label),
                        "label": label,
                        "role": role,
                        "status": "after_dbscan_low_points",
                        "depth_boxplot_filtered": bool(depth_boxplot_filtered),
                    }
                )
                continue

            points_world = transform_points_world(frame, points_frame)
            centroid_frame = points_frame.mean(dim=0)
            centroid_world = points_world.mean(dim=0)
            bbox_frame = bbox_from_points(points_frame)
            bbox_world = bbox_from_points(points_world)
            bbox_diag_world = bbox_diag(bbox_world)
            score = float(scores[det_idx].item()) if scores is not None else 0.0
            preferred_parent_labels: set[str] = set()
            fallback_parent_labels: set[str] = set()
            preferred_parent_role: Optional[str] = None
            owner_mode = "prefer_object"
            is_direct_unit = False
            cabinet_box_marked = False
            cabinet_box_ids: list[int] = []
            cabinet_box_scores: list[float] = []
            if role == "U":
                allowed_parent_labels = set(u_to_allowed.get(label, set()))
                preferred_parent_labels = set(owner_prior.get("u_preferred_parents", {}).get(label, set()))
                fallback_parent_labels = set(owner_prior.get("u_fallback_parents", {}).get(label, set()))
                preferred_parent_role = owner_prior.get("u_preferred_parent_role", {}).get(label)
                owner_mode = owner_prior.get("u_owner_mode", {}).get(label, "prefer_object")
                is_direct_unit = bool(owner_prior.get("u_is_direct", {}).get(label, False))
            elif role == "C":
                allowed_parent_labels = set(c_to_allowed.get(label, set()))
                preferred_parent_labels = set(owner_prior.get("c_preferred_parents", {}).get(label, set()))
                preferred_parent_role = "O"
                owner_mode = "prefer_object"
                if cabinet_aggregation_enabled:
                    hit = carrier_hit_by_idx.get(det_idx)
                    if hit is not None:
                        cabinet_box_marked = True
                        cabinet_box_ids.append(int(hit.get("cabinet_parent_idx", -1)))
                        cabinet_box_scores.append(float(hit.get("contain") or 0.0))
            else:
                allowed_parent_labels = set()
            box_xyxy = boxes[det_idx].detach().cpu().tolist() if boxes is not None else None
            border_touched = box_touches_image_border(box_xyxy, height, width, margin_px=2)
            if border_touched:
                if role == "U" and self.graph_policy.should_drop_border_u_node():
                    extract_debug.append(
                        {
                            "det_idx": int(det_idx),
                            "raw_label": str(raw_label),
                            "label": label,
                            "role": role,
                            "status": "skipped_border_u_policy",
                        }
                    )
                    continue
                if int(det_idx) not in selected_linked_idxs and not self.keep_border_no_link_observations:
                    extract_debug.append(
                        {
                            "det_idx": int(det_idx),
                            "raw_label": str(raw_label),
                            "label": label,
                            "role": role,
                            "status": "skipped_border_no_link",
                        }
                    )
                    continue
            view_score = observation_view_score(score, mask_area, float(conf.mean().item()), frame_area)
            semantic_subtype_info = self._derive_observation_semantic_subtype(
                det_idx=int(det_idx),
                label=label,
                role=role,
                local_stage=local_stage,
            )
            owner_context_source = "label_prior"
            if role == "U":
                owner_context = self._resolve_u_owner_context(
                    det_idx=int(det_idx),
                    unit_label=label,
                    owner_prior=owner_prior,
                    local_stage=local_stage,
                )
                if owner_context is not None:
                    allowed_parent_labels = set(owner_context.get("allowed_parent_labels", set()))
                    preferred_parent_labels = set(owner_context.get("preferred_parent_labels", set()))
                    fallback_parent_labels = set(owner_context.get("fallback_parent_labels", set()))
                    preferred_parent_role = owner_context.get("preferred_parent_role")
                    owner_mode = str(owner_context.get("semantic_owner_mode") or owner_mode)
                    is_direct_unit = bool(owner_context.get("is_direct_unit", is_direct_unit))
                    owner_context_source = str(owner_context.get("source") or "context_prior")
            observations.append(
                NodeObservation(
                    det_idx=det_idx,
                    label=label,
                    raw_label=str(raw_label),
                    role=role,
                    score=score,
                    mask_area=mask_area,
                    box_xyxy=box_xyxy,
                    box_touches_border=border_touched,
                    num_points=int(points_frame.shape[0]),
                    dbscan_filtered=dbscan_filtered,
                    points_frame=points_frame,
                    centroid_frame=centroid_frame,
                    bbox3d_frame=bbox_frame,
                    points_world=points_world,
                    centroid_world=centroid_world,
                    bbox3d_world=bbox_world,
                    bbox_diag_world=bbox_diag_world,
                    allowed_parent_labels=allowed_parent_labels,
                    preferred_parent_labels=preferred_parent_labels,
                    fallback_parent_labels=fallback_parent_labels,
                    preferred_parent_role=preferred_parent_role,
                    is_direct_unit=is_direct_unit,
                    semantic_owner_mode=owner_mode,
                    cabinet_box_marked=cabinet_box_marked,
                    cabinet_box_ids=cabinet_box_ids,
                    cabinet_box_scores=cabinet_box_scores,
                    local_parent_det_idxs=sorted(int(idx) for idx in local_parent_det_idxs_by_child.get(int(det_idx), set())),
                    strong_parent_det_idxs=sorted(int(idx) for idx in strong_parent_det_idxs_by_child.get(int(det_idx), set())),
                    semantic_subtype=semantic_subtype_info.get("semantic_subtype"),
                    semantic_subtype_conf=float(semantic_subtype_info.get("semantic_subtype_conf") or 0.0),
                    semantic_parent_label=semantic_subtype_info.get("semantic_parent_label"),
                    semantic_parent_role=semantic_subtype_info.get("semantic_parent_role"),
                    semantic_owner_object_label=semantic_subtype_info.get("semantic_owner_object_label"),
                    view_score=view_score,
                )
            )
            extract_debug.append(
                {
                    "det_idx": int(det_idx),
                    "raw_label": str(raw_label),
                    "label": label,
                    "role": role,
                    "status": "ok",
                    "num_points": int(points_frame.shape[0]),
                    "depth_boxplot_filtered": bool(depth_boxplot_filtered),
                    "dbscan_filtered": bool(dbscan_filtered),
                    "owner_context_source": owner_context_source,
                    "allowed_parent_labels": sorted(allowed_parent_labels),
                    "preferred_parent_labels": sorted(preferred_parent_labels),
                    "fallback_parent_labels": sorted(fallback_parent_labels),
                    "semantic_owner_mode": owner_mode,
                }
            )

        self._last_obs_extract_stats = {
            "num_obs_dropped_low_points": int(dropped_low_points),
            "num_obs_depth_boxplot_changed": int(depth_boxplot_changed),
            "num_obs_dbscan_changed": int(dbscan_changed),
        }
        self._last_obs_extract_debug = extract_debug
        return observations

    def _track_stable_parent_id(self, node_id: str) -> Optional[str]:
        posterior = self.local_posteriors.get(node_id)
        return posterior.stable_parent_id if posterior is not None else None

    def _track_assoc_parent(self, node_id: str) -> tuple[Optional[OnlineNodeTrack], Optional[str], int]:
        posterior = self.local_posteriors.get(node_id)
        if posterior is None:
            return None, None, 0
        if posterior.stable_parent_id is not None:
            parent_track = self.node_tracks.get(posterior.stable_parent_id)
            if parent_track is not None:
                return parent_track, "stable", posterior.support_count(posterior.stable_parent_id)
        if posterior.top1_parent_id is not None:
            parent_track = self.node_tracks.get(posterior.top1_parent_id)
            if parent_track is not None:
                return parent_track, "top1", posterior.support_count(posterior.top1_parent_id)
        return None, None, 0

    def _assoc_parent_penalty(self, track: OnlineNodeTrack, obs: NodeObservation) -> float:
        if not obs.allowed_parent_labels:
            return 0.0
        parent_track, _, _ = self._track_assoc_parent(track.node_id)
        if parent_track is None:
            return 0.0
        penalty = 0.0 if parent_track.label in obs.allowed_parent_labels else 1.0
        if obs.preferred_parent_labels:
            if parent_track.label in obs.preferred_parent_labels:
                penalty -= 0.4
            elif (
                obs.fallback_parent_labels
                and parent_track.label in obs.fallback_parent_labels
                and obs.semantic_owner_mode == "prefer_carrier"
                and self._has_visible_preferred_parent(obs)
            ):
                penalty += 0.6
        return max(0.0, penalty)

    def _strong_current_parent_observations(self, obs: NodeObservation) -> list[NodeObservation]:
        if obs.role != "U":
            return []
        det_to_obs = self._assoc_det_to_obs_context or {}
        parent_det_idxs = list(obs.strong_parent_det_idxs or obs.local_parent_det_idxs or [])
        parent_obs = [det_to_obs.get(int(det_idx)) for det_idx in parent_det_idxs]
        parent_obs = [parent for parent in parent_obs if parent is not None and parent.det_idx != obs.det_idx]
        if obs.preferred_parent_role:
            preferred = [parent for parent in parent_obs if parent.role == obs.preferred_parent_role]
            if preferred:
                parent_obs = preferred
        if obs.allowed_parent_labels:
            allowed = [parent for parent in parent_obs if parent.label in obs.allowed_parent_labels]
            if allowed:
                parent_obs = allowed
        parent_obs.sort(key=lambda parent: (parent.det_idx, parent.role, parent.label))
        return parent_obs

    def _assoc_parent_context_default(self, track: OnlineNodeTrack, obs: NodeObservation) -> dict[str, Any]:
        return {
            "stable_parent_id": None,
            "stable_parent_label": None,
            "stable_parent_role": None,
            "assoc_parent_source": None,
            "assoc_parent_support": 0,
            "current_parent_det_idxs": [],
            "best_parent_det_idx": None,
            "best_parent_label": None,
            "best_parent_role": None,
            "parent_context_score": 0.0,
            "parent_geom_score": 0.0,
            "parent_overlap_score": 0.0,
            "parent_dist": None,
            "parent_tau": None,
            "parent_label_penalty": float(self._assoc_parent_penalty(track, obs)),
            "parent_geom_available": False,
            "parent_has_current_evidence": False,
            "parent_hard_conflict_stage1": False,
            "parent_hard_conflict_stage2": False,
        }

    def _assoc_parent_context(self, track: OnlineNodeTrack, obs: NodeObservation, frame, keyframes) -> dict[str, Any]:
        info: dict[str, Any] = self._assoc_parent_context_default(track, obs)
        if obs.role != "U":
            return info

        current_parent_obs = self._strong_current_parent_observations(obs)
        if not current_parent_obs:
            return info
        info["parent_has_current_evidence"] = True
        info["current_parent_det_idxs"] = [int(parent.det_idx) for parent in current_parent_obs]

        stable_parent_track, parent_source, parent_support = self._track_assoc_parent(track.node_id)
        if stable_parent_track is None:
            return info

        info["stable_parent_id"] = stable_parent_track.node_id
        info["stable_parent_label"] = stable_parent_track.label
        info["stable_parent_role"] = stable_parent_track.role
        info["assoc_parent_source"] = parent_source
        info["assoc_parent_support"] = int(parent_support)

        best_score = -1.0
        best_payload: dict[str, Any] = {}
        for parent_obs in current_parent_obs:
            role_score = 1.0 if stable_parent_track.role == parent_obs.role else 0.0
            label_score = 1.0 if stable_parent_track.label == parent_obs.label else 0.0
            predicted_parent = self._cached_predicted_centroid(
                stable_parent_track,
                keyframes,
                parent_obs.centroid_world.device,
                parent_obs.centroid_world.dtype,
            )
            dist = None
            tau = None
            dist_score = 0.0
            if predicted_parent is not None:
                tau = self._adaptive_assoc_dist_gate(stable_parent_track, parent_obs)
                dist = self._assoc_world_distance_to_track_centroid(
                    parent_obs,
                    stable_parent_track,
                    predicted_parent,
                    keyframes,
                )
                dist_score = max(0.0, 1.0 - (dist / max(tau, 1e-6)))
            projected_parent_box = self._projected_box_for_track(stable_parent_track, frame, keyframes, parent_obs)
            overlap_score = self._assoc_overlap_score(projected_parent_box, parent_obs) if projected_parent_box is not None else 0.0
            geom_score = max(float(overlap_score), float(dist_score))
            compat = 0.80 * geom_score + 0.15 * role_score + 0.05 * label_score
            if compat <= best_score:
                continue
            best_score = float(compat)
            best_payload = {
                "best_parent_det_idx": int(parent_obs.det_idx),
                "best_parent_label": parent_obs.label,
                "best_parent_role": parent_obs.role,
                "parent_context_score": float(compat),
                "parent_geom_score": float(geom_score),
                "parent_overlap_score": float(overlap_score),
                "parent_dist": None if dist is None else float(dist),
                "parent_tau": None if tau is None else float(tau),
                "parent_geom_available": bool(predicted_parent is not None or projected_parent_box is not None),
            }

        if not best_payload:
            return info
        info.update(best_payload)
        if info["parent_geom_available"] and (
            info["assoc_parent_source"] == "stable" or int(info["assoc_parent_support"]) >= 2
        ):
            info["parent_hard_conflict_stage1"] = info["parent_context_score"] < self.assoc_parent_conflict_thr_stage1
            info["parent_hard_conflict_stage2"] = info["parent_context_score"] < self.assoc_parent_conflict_thr_stage2
        return info

    @staticmethod
    def _assoc_gate_family(reject_reason: Optional[str], *, stage: str) -> str:
        reason = str(reject_reason or "")
        if reason in {"ok", ""}:
            return "pass"
        if reason in {"no_pred_centroid", "no_projected_box", "overlap_zero"}:
            return "projection"
        if reason == "dist_fail":
            return "distance"
        if reason == "birth_gate_fail":
            return "birth"
        if reason == "parent_conflict":
            return "context"
        if stage == "stage2":
            return "stage2_other"
        return "stage1_other"

    @staticmethod
    def _assoc_context_family(parent_ctx: dict[str, Any], *, stage: str) -> str:
        hard_key = "parent_hard_conflict_stage1" if stage == "stage1" else "parent_hard_conflict_stage2"
        if bool(parent_ctx.get(hard_key, False)):
            return "conflicting"
        if bool(parent_ctx.get("parent_has_current_evidence", False)):
            if str(parent_ctx.get("assoc_parent_source") or ""):
                return "supportive"
            return "current_only"
        return "absent"

    def _annotate_assoc_candidate_debug(
        self,
        track: OnlineNodeTrack,
        info: dict[str, Any],
        *,
        stage: str,
    ) -> dict[str, Any]:
        info = dict(info)
        info["projected_box_available"] = bool(info.get("projectable", False))
        info["track_stable_geom_ready"] = bool(getattr(track, "stable_geom_ready", False))
        info["track_bbox_diag_world_est"] = float(getattr(track, "bbox_diag_world_est", 0.0))
        info["track_anchor_kf_id"] = getattr(track, "anchor_kf_id", None)
        info["track_anchor_support_count"] = int(len(getattr(track, "anchor_support_kfs", []) or []))
        info["track_obs_count"] = int(getattr(track, "obs_count", 0))
        info["track_visible_count"] = int(getattr(track, "visible_count", 0))
        info["track_last_seen_frame"] = getattr(track, "last_seen_frame", None)
        info["track_geom_source"] = str(track.assoc_geom_source())
        info["gate_family"] = self._assoc_gate_family(info.get("reject_reason"), stage=stage)
        info["context_family"] = self._assoc_context_family(info, stage=stage)
        info.setdefault("won_hungarian", False)
        info.setdefault("lost_to_node_id", None)
        return info

    @staticmethod
    def _mark_assoc_debug_outcome(candidates: list[dict], *, matched_node_id: Optional[str], assigned_stage: Optional[str], stage: str) -> None:
        for candidate in candidates:
            is_winner = (
                matched_node_id is not None
                and assigned_stage == stage
                and candidate.get("candidate_node_id") == matched_node_id
            )
            candidate["won_hungarian"] = bool(is_winner)
            candidate["lost_to_node_id"] = None if is_winner or matched_node_id is None else matched_node_id

    def _record_assoc_observation_competition(
        self,
        frame_idx: int,
        observation_debug: list[dict],
    ) -> int:
        """Record per-observation node-association competition.

        This captures same-role node pairs that competed for the same 2D
        observation during stage1 / stage2 association and therefore can
        serve as duplicate-consolidation evidence even when they never
        co-occurred inside a local posterior.
        """
        recorded = 0
        for entry in observation_debug:
            matched_node_id = str(entry.get("matched_node_id") or "")
            if not matched_node_id or matched_node_id not in self.node_tracks:
                continue
            if self._is_aggregate_cabinet_node(matched_node_id):
                continue
            role = str(entry.get("role") or "")
            det_idx = int(entry.get("det_idx", -1) or -1)
            competitor_ids: set[str] = set()
            for stage_key in ("stage1_candidates", "stage2_candidates"):
                for candidate in entry.get(stage_key, []) or []:
                    candidate_node_id = str(candidate.get("candidate_node_id") or "")
                    if (
                        not candidate_node_id
                        or candidate_node_id == matched_node_id
                        or candidate_node_id not in self.node_tracks
                    ):
                        continue
                    if self._is_aggregate_cabinet_node(candidate_node_id):
                        continue
                    candidate_track = self.node_tracks.get(candidate_node_id)
                    if candidate_track is None or str(getattr(candidate_track, "role", "")) != role:
                        continue
                    # Only record plausible competitions: either the
                    # candidate actually lost to the matched node, or it was
                    # otherwise a valid Hungarian candidate.
                    if (
                        str(candidate.get("lost_to_node_id") or "") != matched_node_id
                        and not bool(candidate.get("valid", False))
                    ):
                        continue
                    competitor_ids.add(candidate_node_id)
            for competitor_id in sorted(competitor_ids):
                self._record_assignment_competition_event(
                    frame_idx=int(frame_idx),
                    type="child_competition",
                    node_a=matched_node_id,
                    node_b=competitor_id,
                    shared_child_id=f"assoc_obs:{int(frame_idx)}:{det_idx}",
                    source="association_debug",
                    strength=1.0,
                )
                recorded += 1
        return recorded

    def _has_visible_preferred_parent(self, obs: NodeObservation) -> bool:
        if obs.role != "U" or obs.semantic_owner_mode != "prefer_carrier":
            return False
        for track in self.node_tracks.values():
            if track.role != "C" or track.label not in obs.preferred_parent_labels:
                continue
            if track.last_seen_frame is None:
                continue
            return True
        return False

    def _assoc_geom_mismatch(self, track: OnlineNodeTrack, obs: NodeObservation) -> float:
        if not track.stable_geom_ready or track.bbox_anchor is None:
            return 0.0
        anchor_min = torch.as_tensor(track.bbox_anchor.get("min"), device=obs.centroid_world.device, dtype=obs.centroid_world.dtype)
        anchor_max = torch.as_tensor(track.bbox_anchor.get("max"), device=obs.centroid_world.device, dtype=obs.centroid_world.dtype)
        anchor_extent = bbox_size((anchor_min, anchor_max))
        obs_extent = bbox_size(obs.bbox3d_frame)
        denom = max(float(torch.linalg.norm(anchor_extent).item()), float(torch.linalg.norm(obs_extent).item()), 1e-6)
        mismatch = float(torch.linalg.norm(anchor_extent - obs_extent).item()) / denom
        return min(1.0, mismatch)

    def _adaptive_assoc_dist_gate(self, track: OnlineNodeTrack, obs: NodeObservation) -> float:
        role = obs.role
        role_min = float(self.assoc_diag_gate_min_by_role.get(role, 0.05))
        role_max = float(self.assoc_diag_gate_max_by_role.get(role, 0.5))
        if float(track.bbox_diag_world_est) > 0.0:
            tau = float(self.assoc_diag_gate_scale) * float(track.bbox_diag_world_est)
            return float(max(role_min, min(role_max, tau)))
        return role_max

    def _assoc_overlap_score(self, projected_box: Optional[list[float]], obs: NodeObservation) -> float:
        std_iou = box_iou_xyxy(projected_box, obs.box_xyxy)
        if obs.box_touches_border:
            obs_cov = box_overlap_over_obs_area(projected_box, obs.box_xyxy)
            return float(max(std_iou, obs_cov))
        return float(std_iou)

    # ------------------------------------------------------------------
    # Per-frame geometry cache (Step 1 + Step 4).
    #
    # Stage1/stage2/projectability and their helpers independently call
    # `predict_points_world`, `predict_centroid_world`, `predict_recent_box_xyxy`
    # and `_projected_box_for_track` per (obs, track) pair.  Because all of
    # these depend only on (track, frame, keyframes) (the `obs` arg is used
    # merely to forward device/dtype), the result is identical across all
    # observations in a single frame.  Memoising on `track.node_id` collapses
    # the call count from O(N_obs * N_track) -> O(N_track) without changing
    # any numerical output and without affecting matching decisions.
    # ------------------------------------------------------------------

    def _reset_assoc_frame_cache(self) -> None:
        if _ENV_DISABLE_ASSOC_GEOM_CACHE:
            self._assoc_frame_cache = None
            return
        self._assoc_frame_cache = {
            "centroid": {},
            "centroid_cpu": {},
            "obs_centroid_cpu": {},
            "projected_box": {},
            "recent_bbox": {},
            "frame_hw": {},
            "frame_K": {},
        }

    def _cached_predicted_centroid(
        self,
        track: OnlineNodeTrack,
        keyframes,
        device,
        dtype,
    ) -> Optional[torch.Tensor]:
        cache = getattr(self, "_assoc_frame_cache", None)
        if cache is None:
            return track.predict_centroid_world(keyframes, device=device, dtype=dtype)
        key = track.node_id
        bucket = cache["centroid"]
        if key in bucket:
            return bucket[key]
        value = track.predict_centroid_world(keyframes, device=device, dtype=dtype)
        bucket[key] = value
        return value

    def _cached_predicted_centroid_cpu(
        self,
        track: OnlineNodeTrack,
        keyframes,
        device,
        dtype,
    ) -> Optional[torch.Tensor]:
        cache = getattr(self, "_assoc_frame_cache", None)
        value = self._cached_predicted_centroid(track, keyframes, device, dtype)
        if value is None:
            return None
        if cache is None:
            return value.detach().cpu()
        key = track.node_id
        bucket = cache["centroid_cpu"]
        if key in bucket:
            return bucket[key]
        cpu_value = value.detach().cpu()
        bucket[key] = cpu_value
        return cpu_value

    def _cached_obs_centroid_cpu(self, obs: NodeObservation) -> torch.Tensor:
        cache = getattr(self, "_assoc_frame_cache", None)
        if cache is None:
            return obs.centroid_world.detach().cpu()
        bucket = cache["obs_centroid_cpu"]
        obs_key = int(obs.det_idx)
        if obs_key not in bucket:
            bucket[obs_key] = obs.centroid_world.detach().cpu()
        return bucket[obs_key]

    def _assoc_world_distance_to_track_centroid(
        self,
        obs: NodeObservation,
        track: OnlineNodeTrack,
        predicted: torch.Tensor,
        keyframes,
    ) -> float:
        cache = getattr(self, "_assoc_frame_cache", None)
        if cache is None:
            return world_distance(obs.centroid_world, predicted)
        obs_cpu = self._cached_obs_centroid_cpu(obs)
        pred_cpu = self._cached_predicted_centroid_cpu(
            track,
            keyframes,
            obs.centroid_world.device,
            obs.centroid_world.dtype,
        )
        if pred_cpu is None:
            pred_cpu = predicted.detach().cpu()
        return float(torch.linalg.norm(obs_cpu - pred_cpu).item())

    def _assoc_frame_hw_cached(self, frame) -> tuple[int, int]:
        cache = getattr(self, "_assoc_frame_cache", None)
        if cache is None:
            return frame_hw(frame)
        bucket = cache["frame_hw"]
        key = id(frame)
        if key not in bucket:
            bucket[key] = frame_hw(frame)
        return bucket[key]

    def _assoc_frame_intrinsics_cached(self, frame, *, device, dtype) -> Optional[torch.Tensor]:
        cache = getattr(self, "_assoc_frame_cache", None)
        if cache is None:
            K = frame_intrinsics(frame)
            return None if K is None else K.to(device=device, dtype=dtype)
        bucket = cache["frame_K"]
        key = (id(frame), str(device), str(dtype))
        if key in bucket:
            return bucket[key]
        K = frame_intrinsics(frame)
        value = None if K is None else K.to(device=device, dtype=dtype)
        bucket[key] = value
        return value

    def _cached_projected_box(
        self,
        track: OnlineNodeTrack,
        frame,
        keyframes,
        device,
        dtype,
    ) -> Optional[list[float]]:
        cache = getattr(self, "_assoc_frame_cache", None)
        if cache is None:
            return self._projected_box_for_track_uncached(track, frame, keyframes, device, dtype)
        key = track.node_id
        bucket = cache["projected_box"]
        if key in bucket:
            return bucket[key]
        value = self._projected_box_for_track_uncached(track, frame, keyframes, device, dtype)
        bucket[key] = value
        return value

    def _cached_recent_bbox(
        self,
        track: OnlineNodeTrack,
        frame,
        device,
        dtype,
    ) -> Optional[list[float]]:
        cache = getattr(self, "_assoc_frame_cache", None)
        if cache is None:
            return track.predict_recent_box_xyxy(frame, device=device, dtype=dtype)
        key = track.node_id
        bucket = cache["recent_bbox"]
        if key in bucket:
            return bucket[key]
        value = track.predict_recent_box_xyxy(frame, device=device, dtype=dtype)
        bucket[key] = value
        return value

    def _projected_box_for_track_uncached(
        self,
        track: OnlineNodeTrack,
        frame,
        keyframes,
        device,
        dtype,
    ) -> Optional[list[float]]:
        predicted_points = self._projectability_points_world_for_track(
            track,
            keyframes,
            device=device,
            dtype=dtype,
        )
        return self._project_world_points_to_box(predicted_points, frame)

    def _projectability_points_world_for_track(
        self,
        track: OnlineNodeTrack,
        keyframes,
        *,
        device,
        dtype,
    ) -> Optional[torch.Tensor]:
        return track.predict_points_world(keyframes, device=device, dtype=dtype)

    def _project_world_points_to_box(
        self,
        points_world: Optional[torch.Tensor],
        frame,
    ) -> Optional[list[float]]:
        if points_world is None or points_world.numel() == 0:
            return None
        points_frame = transform_points_frame(frame, points_world)
        valid = torch.isfinite(points_frame).all(dim=-1) & (points_frame[:, 2] > 1e-6)
        if int(valid.sum().detach().cpu().item()) == 0:
            return None
        points_frame = points_frame[valid]
        z = points_frame[:, 2]
        h, w = self._assoc_frame_hw_cached(frame)
        K = self._assoc_frame_intrinsics_cached(frame, device=points_frame.device, dtype=points_frame.dtype)
        if K is None:
            focal = float(max(h, w)) * 0.5
            K = torch.tensor(
                [[focal, 0.0, float(w) * 0.5], [0.0, focal, float(h) * 0.5], [0.0, 0.0, 1.0]],
                device=points_frame.device,
                dtype=points_frame.dtype,
            )
        x = points_frame[:, 0]
        y = points_frame[:, 1]
        u = K[0, 0] * (x / z) + K[0, 2]
        v = K[1, 1] * (y / z) + K[1, 2]
        x_clamped = u.clamp(0.0, float(w - 1))
        y_clamped = v.clamp(0.0, float(h - 1))
        if x_clamped.numel() == 0 or y_clamped.numel() == 0:
            return None
        x1, y1, x2, y2 = [
            float(v)
            for v in torch.stack(
                [x_clamped.min(), y_clamped.min(), x_clamped.max(), y_clamped.max()]
            ).detach().cpu().tolist()
        ]
        if (x2 - x1) <= 1e-6 or (y2 - y1) <= 1e-6:
            return None
        return [x1, y1, x2, y2]

    def _projected_box_for_track(self, track: OnlineNodeTrack, frame, keyframes, obs: NodeObservation) -> Optional[list[float]]:
        return self._cached_projected_box(
            track,
            frame,
            keyframes,
            obs.centroid_world.device,
            obs.centroid_world.dtype,
        )

    def _recent_bbox_overlap_prior(
        self,
        track: OnlineNodeTrack,
        frame,
        obs: NodeObservation,
    ) -> tuple[float, Optional[int], Optional[str], Optional[list[float]]]:
        if frame is None or track.role != obs.role:
            return 0.0, None, None, None
        if int(getattr(track, "obs_count", 0)) < int(self.assoc_recent_bbox_min_obs_count):
            return 0.0, None, None, None
        if track.last_seen_frame is None:
            return 0.0, None, None, None
        gap = int(getattr(frame, "frame_id", -1)) - int(track.last_seen_frame)
        if gap <= 0 or gap > int(self.assoc_recent_bbox_gap_max):
            return 0.0, gap, None, None
        recent_box = self._cached_recent_bbox(track, frame, obs.centroid_world.device, obs.centroid_world.dtype)
        if recent_box is None:
            return 0.0, gap, None, None
        overlap = self._assoc_overlap_score(recent_box, obs)
        if overlap <= 0.0:
            return 0.0, gap, "recent_bbox_motion", recent_box
        return float(overlap), gap, "recent_bbox_motion", recent_box

    def _carrier_stage1_geom_confidence(self, track: OnlineNodeTrack, candidate_debug: dict[str, Any]) -> float:
        source = str(candidate_debug.get("track_geom_source") or track.assoc_geom_source())
        if source == "stable_anchor_fused":
            confidence = 0.98
        elif source == "stable_anchor":
            confidence = 0.95
        elif source == "provisional_anchor":
            confidence = 0.75
        elif source == "candidate_anchor":
            confidence = 0.60
        elif source == "world_bbox":
            confidence = 0.35
        elif source == "world_centroid":
            confidence = 0.15
        else:
            confidence = 0.05
        obs_count = int(getattr(track, "obs_count", 0))
        confidence += 0.10 * min(obs_count, 4) / 4.0
        support = int(len(getattr(track, "anchor_support_kfs", []) or []))
        confidence += 0.10 * min(support, 3) / 3.0
        if bool(getattr(track, "stable_geom_ready", False)):
            confidence += 0.10
        recent_overlap = float(candidate_debug.get("recent_bbox_overlap_score") or 0.0)
        confidence += 0.15 * min(1.0, recent_overlap)
        return float(max(0.0, min(1.0, confidence)))

    def _carrier_stage1_continuity_rescue(
        self,
        track: OnlineNodeTrack,
        obs: NodeObservation,
        *,
        overlap_score: float,
        recent_overlap_score: float,
        tau: float,
        dist: float,
    ) -> tuple[float, float, bool, Optional[str]]:
        if track.role != obs.role:
            return float(overlap_score), float(tau), False, None

        if obs.role != "C":
            effective_overlap = float(overlap_score)
            if effective_overlap > 0.0:
                return effective_overlap, float(tau), False, None
            if int(getattr(track, "obs_count", 0)) < int(self.assoc_recent_bbox_min_obs_count):
                return float(overlap_score), float(tau), False, None
            if float(recent_overlap_score) < float(self.assoc_recent_bbox_min_overlap):
                return float(overlap_score), float(tau), False, None
            effective_overlap = max(effective_overlap, float(recent_overlap_score))
            return float(effective_overlap), float(tau), bool(effective_overlap > 0.0), "recent_bbox_overlap"

        source = str(track.assoc_geom_source())
        obs_count = int(getattr(track, "obs_count", 0))
        mature_track = bool(getattr(track, "stable_geom_ready", False)) or source in {"stable_anchor_fused", "stable_anchor", "provisional_anchor", "candidate_anchor"} or obs_count >= int(
            self.assoc_stage1_continuity_min_obs_count_c
        )
        if not mature_track:
            return float(overlap_score), float(tau), False, None

        effective_overlap = float(overlap_score)
        effective_tau = float(tau)
        near_miss = (effective_overlap <= 0.0) or (float(dist) > float(tau))
        if not near_miss:
            return float(overlap_score), float(tau), False, None
        rescue_kind: Optional[str] = None
        if effective_overlap <= 0.0:
            if float(recent_overlap_score) < float(self.assoc_recent_bbox_min_overlap):
                return float(overlap_score), float(tau), False, None
            effective_overlap = max(
                effective_overlap,
                float(self.assoc_stage1_continuity_overlap_weight_c) * float(recent_overlap_score),
            )
            if effective_overlap > 0.0:
                rescue_kind = "recent_bbox_overlap"
        tau_boost = 0.0
        if source == "provisional_anchor" and float(overlap_score) > 0.0:
            tau_boost += 0.10
        elif source == "candidate_anchor" and float(overlap_score) > 0.0:
            tau_boost += 0.08
        elif source == "stable_anchor_fused" and float(overlap_score) > 0.0:
            tau_boost += 0.06
        elif source == "stable_anchor" and float(overlap_score) > 0.0:
            tau_boost += 0.05
        if float(recent_overlap_score) >= float(self.assoc_recent_bbox_min_overlap):
            tau_boost += float(self.assoc_stage1_continuity_tau_boost_c) * min(1.0, float(recent_overlap_score))
        boosted_tau = float(tau) * (1.0 + tau_boost)
        if boosted_tau > effective_tau + 1e-6:
            effective_tau = boosted_tau
            if rescue_kind is None:
                rescue_kind = "anchor_tau" if tau_boost > 0.0 and float(recent_overlap_score) < float(self.assoc_recent_bbox_min_overlap) else "recent_bbox_tau"
            elif rescue_kind == "recent_bbox_overlap":
                rescue_kind = "recent_bbox_overlap+tau"
        return float(effective_overlap), float(effective_tau), rescue_kind is not None, rescue_kind

    def _apply_multi_anchor_assoc_assist(
        self,
        *,
        obs_indices: list[int],
        stage1_track_ids: list[str],
        stage1_valid: torch.Tensor,
        stage1_score: torch.Tensor,
        obs_debug: dict[int, dict],
        observations: list[NodeObservation],
        frame_idx: Optional[int] = None,
    ) -> None:
        """Apply a small multi-anchor assist bonus to stage1 scores.

        First-stage weak assist: for stable tracks whose primary anchor
        is visibly weak (border-touched / few points / low view score),
        we compute a tiny bonus from the best auxiliary anchor's support
        and add it to the existing stage1 score.

        Hard constraints:
          * Only applied to stable tracks with ``stable_geom_ready=True``.
          * Bonus is capped at ``multi_anchor_assist_max_bonus`` (default
            0.04) so it can never override the primary geometry.
          * Debug metadata is annotated on the matching candidate entry.
          * If ``self.enable_multi_anchor_assist`` is False the helper is
            a no-op.
        """
        if not self.enable_multi_anchor_assist:
            return
        if not stage1_track_ids or stage1_valid.numel() == 0:
            return
        max_bonus = float(self.multi_anchor_assist_max_bonus)
        if max_bonus <= 0.0:
            return
        for r, obs_idx in enumerate(obs_indices):
            obs = observations[obs_idx]
            for c, node_id in enumerate(stage1_track_ids):
                if not bool(stage1_valid[r, c]):
                    continue
                track = self.node_tracks.get(node_id)
                if track is None:
                    continue
                if not bool(getattr(track, "stable_geom_ready", False)):
                    continue
                if not getattr(track, "auxiliary_anchors", None):
                    continue
                support, best_kf, reason = track._best_aux_assoc_support(obs)
                if support <= 0.0:
                    continue
                bonus = float(max_bonus) * float(support)
                if bonus <= 0.0:
                    continue
                stage1_score[r, c] = float(stage1_score[r, c].item()) + bonus
                # Record on the track itself so stats/exporter can observe
                # that assist was applied to this stable track at this frame.
                try:
                    track.record_support_anchor_assist(
                        frame_idx=frame_idx,
                        best_kf=best_kf,
                        score=float(support),
                        bonus=float(bonus),
                        reason=str(reason),
                    )
                except Exception:
                    pass
                # Annotate the debug record for this candidate so the
                # assist is visible in snapshots / reports.
                for candidate_debug in obs_debug[obs_idx]["stage1_candidates"]:
                    if candidate_debug.get("candidate_node_id") == node_id:
                        # Primary support-anchor debug field names (per
                        # anchor_req.md v2).  We keep the ``multi_anchor_*``
                        # aliases for backward compatibility.
                        candidate_debug["support_anchor_used"] = True
                        candidate_debug["support_anchor_best_kf"] = best_kf
                        candidate_debug["support_anchor_best_score"] = float(support)
                        candidate_debug["support_anchor_bonus_applied"] = float(bonus)
                        candidate_debug["support_anchor_reason"] = str(reason)
                        candidate_debug["multi_anchor_support_used"] = True
                        candidate_debug["multi_anchor_best_kf"] = best_kf
                        candidate_debug["multi_anchor_best_score"] = float(support)
                        candidate_debug["multi_anchor_support_bonus"] = float(bonus)
                        candidate_debug["multi_anchor_support_reason"] = str(reason)
                        existing_score = float(candidate_debug.get("score", 0.0) or 0.0)
                        candidate_debug["score"] = existing_score + bonus
                        break

    def _apply_stage1_carrier_geom_rerank(
        self,
        *,
        obs_indices: list[int],
        stage1_track_ids: list[str],
        stage1_valid: torch.Tensor,
        stage1_score: torch.Tensor,
        obs_debug: dict[int, dict],
    ) -> None:
        if not stage1_track_ids or stage1_valid.numel() == 0:
            return
        for r, obs_idx in enumerate(obs_indices):
            valid_cols = [c for c in range(len(stage1_track_ids)) if bool(stage1_valid[r, c])]
            if len(valid_cols) < 2:
                for c in valid_cols:
                    node_id = stage1_track_ids[c]
                    candidate_debug = next(
                        (
                            cand
                            for cand in obs_debug[obs_idx]["stage1_candidates"]
                            if cand.get("candidate_node_id") == node_id
                        ),
                        None,
                    )
                    if candidate_debug is None:
                        continue
                    conf = self._carrier_stage1_geom_confidence(self.node_tracks[node_id], candidate_debug)
                    candidate_debug["geom_confidence"] = float(conf)
                    candidate_debug["geom_confidence_penalty"] = 0.0
                    candidate_debug["score_pre_rerank"] = float(stage1_score[r, c].item())
                    candidate_debug["score_post_rerank"] = float(stage1_score[r, c].item())
                continue

            confidence_by_col: dict[int, float] = {}
            for c in valid_cols:
                node_id = stage1_track_ids[c]
                candidate_debug = next(
                    (
                        cand
                        for cand in obs_debug[obs_idx]["stage1_candidates"]
                        if cand.get("candidate_node_id") == node_id
                    ),
                    None,
                )
                if candidate_debug is None:
                    continue
                confidence_by_col[c] = self._carrier_stage1_geom_confidence(self.node_tracks[node_id], candidate_debug)
            if not confidence_by_col:
                continue
            best_conf = max(confidence_by_col.values())
            for c in valid_cols:
                node_id = stage1_track_ids[c]
                track = self.node_tracks[node_id]
                candidate_debug = next(
                    (
                        cand
                        for cand in obs_debug[obs_idx]["stage1_candidates"]
                        if cand.get("candidate_node_id") == node_id
                    ),
                    None,
                )
                if candidate_debug is None:
                    continue
                conf = float(confidence_by_col.get(c, 0.0))
                pre_score = float(stage1_score[r, c].item())
                penalty = 0.0
                if conf + 1e-6 < best_conf:
                    if (not bool(getattr(track, "stable_geom_ready", False))) or int(getattr(track, "obs_count", 0)) <= 3:
                        penalty = self.assoc_stage1_geom_penalty_weight_c * float(best_conf - conf)
                        stage1_score[r, c] = float(pre_score - penalty)
                candidate_debug["geom_confidence"] = float(conf)
                candidate_debug["geom_confidence_penalty"] = float(penalty)
                candidate_debug["score_pre_rerank"] = float(pre_score)
                candidate_debug["score_post_rerank"] = float(stage1_score[r, c].item())

    def _pair_assoc_score_stage1(self, obs: NodeObservation, track: OnlineNodeTrack, frame, keyframes) -> tuple[Optional[float], str]:
        debug = self._pair_assoc_debug_stage1(obs, track, frame, keyframes)
        score = debug.get("score")
        if score is None:
            return None, str(debug.get("reject_reason") or "invalid")
        return float(score), "ok"

    def _pair_assoc_debug_stage1(self, obs: NodeObservation, track: OnlineNodeTrack, frame, keyframes) -> dict[str, Any]:
        predicted = self._cached_predicted_centroid(track, keyframes, obs.centroid_world.device, obs.centroid_world.dtype)
        label_score = track.label_soft_score(obs.label)
        subtype_score = track.semantic_subtype_soft_score(obs)
        parent_ctx = self._assoc_parent_context_default(track, obs)
        dominant_track_label = track.dominant_label()
        label_exact = (obs.label == dominant_track_label)
        label_sim = label_semantic_similarity(obs.label, dominant_track_label)
        info: dict[str, Any] = {
            "candidate_node_id": track.node_id,
            "projectable": False,
            "projected_box_xyxy": None,
            "overlap_score": 0.0,
            "overlap_source": "projected_box",
            "recent_bbox_overlap_score": 0.0,
            "recent_bbox_gap": None,
            "dist": None,
            "tau": None,
            "label_soft_score": float(label_score),
            "obs_semantic_subtype": obs.semantic_subtype,
            "obs_semantic_subtype_conf": float(obs.semantic_subtype_conf),
            "track_semantic_subtype": track.semantic_subtype,
            "semantic_subtype_soft_score": float(subtype_score),
            "parent_conflict_softened": False,
            "track_dominant_label": dominant_track_label,
            "label_exact_match": bool(label_exact),
            "label_semantic_similarity": float(label_sim),
            "hard_label_gate_iou": 0.0,
            "stage1_projected_iou": 0.0,
            "stage1_min_projected_iou": float(self.graph_policy.stage1_min_projected_iou()),
            "hard_label_gate_pass": True,
            "hard_label_gate_reason": "label_ok",
            "hard_label_gate_bypass_high_iou": False,
            "entered_hungarian": False,
            "valid": False,
            "reject_reason": "invalid",
            "score": None,
            **parent_ctx,
        }
        info = self._annotate_assoc_candidate_debug(track, info, stage="stage1")
        if predicted is None:
            info["reject_reason"] = "no_pred_centroid"
            info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage1")
            return info

        projected_box = self._projected_box_for_track(track, frame, keyframes, obs)
        if projected_box is None:
            info["reject_reason"] = "no_projected_box"
            info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage1")
            return info
        info["projectable"] = True
        info["projected_box_available"] = True
        info["projected_box_xyxy"] = [float(v) for v in projected_box]

        # --- Hard label / semantic gate (B2) ---
        projected_iou = box_iou_xyxy(projected_box, obs.box_xyxy)
        gate_iou = self._assoc_overlap_score(projected_box, obs)
        info["stage1_projected_iou"] = float(projected_iou)
        info["hard_label_gate_iou"] = float(gate_iou)
        min_projected_iou = float(self.graph_policy.stage1_min_projected_iou())
        info["stage1_min_projected_iou"] = float(min_projected_iou)
        if min_projected_iou > 0.0 and float(projected_iou) < min_projected_iou:
            info["reject_reason"] = "stage1_projected_iou_fail"
            info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage1")
            return info
        if not label_exact and label_sim < 0.8:
            if gate_iou >= 0.8:
                info["hard_label_gate_bypass_high_iou"] = True
                info["hard_label_gate_reason"] = "bypass_high_iou"
            else:
                info["hard_label_gate_pass"] = False
                info["hard_label_gate_reason"] = "label_semantic_gate_fail"
                info["reject_reason"] = "label_semantic_gate_fail"
                info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage1")
                return info

        tau = self._adaptive_assoc_dist_gate(track, obs)
        dist = self._assoc_world_distance_to_track_centroid(obs, track, predicted, keyframes)
        info["tau"] = float(tau)
        info["dist"] = float(dist)

        overlap_score = self._assoc_overlap_score(projected_box, obs)
        recent_overlap_score, recent_gap, recent_source, recent_box = self._recent_bbox_overlap_prior(track, frame, obs)
        info["recent_bbox_overlap_score"] = float(recent_overlap_score)
        info["recent_bbox_gap"] = None if recent_gap is None else int(recent_gap)
        if recent_box is not None:
            info["recent_bbox_xyxy"] = [float(v) for v in recent_box]
        effective_overlap_score, effective_tau, continuity_rescue_active, continuity_rescue_kind = self._carrier_stage1_continuity_rescue(
            track,
            obs,
            overlap_score=overlap_score,
            recent_overlap_score=recent_overlap_score,
            tau=tau,
            dist=dist,
        )
        info["overlap_score"] = float(overlap_score)
        info["effective_overlap_score"] = float(effective_overlap_score)
        info["effective_tau"] = float(effective_tau)
        info["continuity_rescue_active"] = bool(continuity_rescue_active)
        info["continuity_rescue_kind"] = continuity_rescue_kind
        if continuity_rescue_kind == "recent_bbox_overlap":
            info["overlap_source"] = "projected_box+recent_bbox"
        elif continuity_rescue_kind == "recent_bbox_overlap+tau":
            info["overlap_source"] = "projected_box+recent_bbox"
        if self.assoc_require_positive_overlap and effective_overlap_score <= 0.0:
            info["zero_overlap_dist_ratio"] = float(dist / max(effective_tau, 1e-6))
            info["reject_reason"] = "overlap_zero"
            info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage1")
            return info

        if dist > effective_tau:
            info["reject_reason"] = "dist_fail"
            info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage1")
            return info

        parent_ctx = self._assoc_parent_context(track, obs, frame, keyframes)
        info.update(parent_ctx)
        dist_score = max(0.0, 1.0 - (dist / max(effective_tau, 1e-6)))
        score = (
            self.assoc_stage1_weight_iou * effective_overlap_score
            + self.assoc_stage1_weight_dist * dist_score
            + self.assoc_stage1_weight_label * label_score
            + self.assoc_stage1_weight_subtype * subtype_score
            + self.assoc_parent_weight_stage1 * float(parent_ctx["parent_context_score"])
            - self.assoc_parent_penalty_weight_stage1 * float(parent_ctx["parent_label_penalty"])
        )
        if parent_ctx["parent_hard_conflict_stage1"]:
            can_soften_conflict = (
                obs.role == "U"
                and continuity_rescue_active
                and str(parent_ctx.get("assoc_parent_source") or "") == "stable"
                and int(parent_ctx.get("assoc_parent_support") or 0) >= int(self.local_min_support_frames)
            )
            if not can_soften_conflict:
                info["reject_reason"] = "parent_conflict"
                info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage1")
                info["context_family"] = self._assoc_context_family(info, stage="stage1")
                return info
            info["parent_conflict_softened"] = True
        info["valid"] = True
        info["reject_reason"] = "ok"
        info["entered_hungarian"] = True
        info["score"] = float(score)
        info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage1")
        info["context_family"] = self._assoc_context_family(info, stage="stage1")
        return info

    def _pair_assoc_score_stage2(self, obs: NodeObservation, track: OnlineNodeTrack, frame, keyframes) -> tuple[Optional[float], str]:
        debug = self._pair_assoc_debug_stage2(obs, track, frame, keyframes)
        score = debug.get("score")
        if score is None:
            return None, str(debug.get("reject_reason") or "invalid")
        return float(score), "ok"

    @staticmethod
    def _score_reason_from_assoc_debug(candidate_debug: dict[str, Any]) -> tuple[Optional[float], str]:
        score = candidate_debug.get("score")
        if score is None:
            return None, str(candidate_debug.get("reject_reason") or "invalid")
        return float(score), "ok"

    def _pair_assoc_debug_stage2(self, obs: NodeObservation, track: OnlineNodeTrack, frame, keyframes) -> dict[str, Any]:
        predicted = self._cached_predicted_centroid(track, keyframes, obs.centroid_world.device, obs.centroid_world.dtype)
        label_score = track.label_soft_score(obs.label)
        subtype_score = track.semantic_subtype_soft_score(obs)
        parent_ctx = self._assoc_parent_context_default(track, obs)
        dominant_track_label = track.dominant_label()
        label_exact = (obs.label == dominant_track_label)
        label_sim = label_semantic_similarity(obs.label, dominant_track_label)
        info: dict[str, Any] = {
            "candidate_node_id": track.node_id,
            "projectable": False,
            "projected_box_xyxy": None,
            "overlap_score": 0.0,
            "dist": None,
            "tau": None,
            "label_soft_score": float(label_score),
            "obs_semantic_subtype": obs.semantic_subtype,
            "obs_semantic_subtype_conf": float(obs.semantic_subtype_conf),
            "track_semantic_subtype": track.semantic_subtype,
            "semantic_subtype_soft_score": float(subtype_score),
            "track_dominant_label": dominant_track_label,
            "label_exact_match": bool(label_exact),
            "label_semantic_similarity": float(label_sim),
            "hard_label_gate_iou": 0.0,
            "hard_label_gate_pass": True,
            "hard_label_gate_reason": "label_ok",
            "hard_label_gate_bypass_high_iou": False,
            "entered_hungarian": False,
            "valid": False,
            "reject_reason": "invalid",
            "score": None,
            **parent_ctx,
        }
        info = self._annotate_assoc_candidate_debug(track, info, stage="stage2")
        if predicted is None:
            info["reject_reason"] = "no_pred_centroid"
            info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage2")
            return info

        # --- Hard label / semantic gate (B2) for stage2 ---
        # Stage2 tracks typically lack projected boxes; compute IoU when possible.
        projected_box_s2 = self._projected_box_for_track(track, frame, keyframes, obs)
        gate_iou_s2 = self._assoc_overlap_score(projected_box_s2, obs) if projected_box_s2 is not None else 0.0
        info["hard_label_gate_iou"] = float(gate_iou_s2)
        if not label_exact and label_sim < 0.8:
            if gate_iou_s2 >= 0.8:
                info["hard_label_gate_bypass_high_iou"] = True
                info["hard_label_gate_reason"] = "bypass_high_iou"
            else:
                info["hard_label_gate_pass"] = False
                info["hard_label_gate_reason"] = "label_semantic_gate_fail"
                info["reject_reason"] = "label_semantic_gate_fail"
                info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage2")
                return info

        tau_base = self._adaptive_assoc_dist_gate(track, obs)
        tau_cap = float(self.assoc_birth_gate_cap_by_role.get(obs.role, tau_base))
        tau_birth = max(1e-6, min(0.6 * tau_base, tau_cap))
        dist = self._assoc_world_distance_to_track_centroid(obs, track, predicted, keyframes)
        info["tau"] = float(tau_birth)
        info["dist"] = float(dist)
        if dist > tau_birth:
            info["reject_reason"] = "birth_gate_fail"
            info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage2")
            return info

        parent_ctx = self._assoc_parent_context(track, obs, frame, keyframes)
        info.update(parent_ctx)
        dist_score = max(0.0, 1.0 - (dist / tau_birth))
        score = (
            self.assoc_stage2_weight_dist * dist_score
            + self.assoc_stage2_weight_label * label_score
            + self.assoc_stage2_weight_subtype * subtype_score
            + self.assoc_parent_weight_stage2 * float(parent_ctx["parent_context_score"])
            - self.assoc_parent_penalty_weight_stage2 * float(parent_ctx["parent_label_penalty"])
        )
        if parent_ctx["parent_hard_conflict_stage2"]:
            info["reject_reason"] = "parent_conflict"
            info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage2")
            info["context_family"] = self._assoc_context_family(info, stage="stage2")
            return info
        info["valid"] = True
        info["reject_reason"] = "ok"
        info["entered_hungarian"] = True
        info["score"] = float(score)
        info["gate_family"] = self._assoc_gate_family(info["reject_reason"], stage="stage2")
        info["context_family"] = self._assoc_context_family(info, stage="stage2")
        return info

    def _projectability_by_track(
        self,
        candidate_nodes: list[str],
        *,
        frame,
        keyframes,
        probe_obs: NodeObservation,
    ) -> tuple[dict[str, Optional[list[float]]], dict[str, bool]]:
        import time as _projectability_time

        profile = os.environ.get("FG_PROFILE_STAGES", "0") == "1"

        def _tick(name: str, t0: float) -> None:
            if profile:
                self._record_runtime_profile(
                    f"update_frame.associate.projectability.{name}",
                    _projectability_time.perf_counter() - t0,
                )

        def _t0() -> float:
            return _projectability_time.perf_counter() if profile else 0.0

        projected_box_by_track_id: dict[str, Optional[list[float]]] = {}
        has_projected_box_by_track_id: dict[str, bool] = {}
        if frame is None or _ENV_DISABLE_BATCH_PROJECTABILITY:
            _t = _t0()
            for node_id in candidate_nodes:
                track = self.node_tracks[node_id]
                projected_box = self._projected_box_for_track(track, frame, keyframes, probe_obs)
                projected_box_by_track_id[node_id] = projected_box
                has_projected_box_by_track_id[node_id] = projected_box is not None
            _tick("legacy_fallback" if frame is None else "per_track_fallback", _t)
            return projected_box_by_track_id, has_projected_box_by_track_id

        cache = getattr(self, "_assoc_frame_cache", None)
        projected_cache = None if cache is None else cache.get("projected_box")
        device = probe_obs.centroid_world.device
        dtype = probe_obs.centroid_world.dtype

        points_by_track: list[tuple[str, torch.Tensor]] = []
        _t = _t0()
        for node_id in candidate_nodes:
            if projected_cache is not None and node_id in projected_cache:
                projected_box = projected_cache[node_id]
                projected_box_by_track_id[node_id] = projected_box
                has_projected_box_by_track_id[node_id] = projected_box is not None
                continue
            track = self.node_tracks[node_id]
            predicted_points = self._projectability_points_world_for_track(
                track,
                keyframes,
                device=device,
                dtype=dtype,
            )
            if predicted_points is None or predicted_points.numel() == 0:
                projected_box_by_track_id[node_id] = None
                has_projected_box_by_track_id[node_id] = False
                if projected_cache is not None:
                    projected_cache[node_id] = None
                continue
            points_by_track.append((node_id, predicted_points))
        _tick("predict_points_world", _t)

        if not points_by_track:
            return projected_box_by_track_id, has_projected_box_by_track_id

        _t = _t0()
        point_segments: list[tuple[str, int, int]] = []
        all_points_parts: list[torch.Tensor] = []
        offset = 0
        for node_id, points_world in points_by_track:
            n = int(points_world.shape[0])
            if n <= 0:
                continue
            point_segments.append((node_id, offset, offset + n))
            all_points_parts.append(points_world)
            offset += n
        if not all_points_parts:
            return projected_box_by_track_id, has_projected_box_by_track_id

        all_points_world = torch.cat(all_points_parts, dim=0)
        points_frame = transform_points_frame(frame, all_points_world)
        valid = torch.isfinite(points_frame).all(dim=-1) & (points_frame[:, 2] > 1e-6)
        h, w = self._assoc_frame_hw_cached(frame)
        K = self._assoc_frame_intrinsics_cached(frame, device=points_frame.device, dtype=points_frame.dtype)
        if K is None:
            focal = float(max(h, w)) * 0.5
            K = torch.tensor(
                [[focal, 0.0, float(w) * 0.5], [0.0, focal, float(h) * 0.5], [0.0, 0.0, 1.0]],
                device=points_frame.device,
                dtype=points_frame.dtype,
            )
        safe_z = torch.where(valid, points_frame[:, 2], torch.ones_like(points_frame[:, 2]))
        u = K[0, 0] * (points_frame[:, 0] / safe_z) + K[0, 2]
        v = K[1, 1] * (points_frame[:, 1] / safe_z) + K[1, 2]
        uv = torch.stack(
            [
                u.clamp(0.0, float(w - 1)),
                v.clamp(0.0, float(h - 1)),
            ],
            dim=-1,
        )
        uv = torch.where(valid[:, None], uv, torch.zeros_like(uv))
        _tick("transform_project", _t)

        _t = _t0()
        uv_np = uv.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        _tick("cpu_sync", _t)

        _t = _t0()
        for node_id, start, end in point_segments:
            seg_valid = valid_np[start:end]
            if not bool(seg_valid.any()):
                projected_box = None
            else:
                seg_uv = uv_np[start:end][seg_valid]
                mins = seg_uv.min(axis=0)
                maxs = seg_uv.max(axis=0)
                x1 = float(mins[0])
                y1 = float(mins[1])
                x2 = float(maxs[0])
                y2 = float(maxs[1])
                projected_box = None if (x2 - x1) <= 1e-6 or (y2 - y1) <= 1e-6 else [x1, y1, x2, y2]
            projected_box_by_track_id[node_id] = projected_box
            has_projected_box_by_track_id[node_id] = projected_box is not None
            if projected_cache is not None:
                projected_cache[node_id] = projected_box
        _tick("bbox_reduce", _t)
        return projected_box_by_track_id, has_projected_box_by_track_id

    def _is_stage2_candidate_track(self, track: OnlineNodeTrack, *, has_projected_box: bool) -> bool:
        newborn_obs_cap = 2 if track.role == "C" else 1
        return track.obs_count <= newborn_obs_cap or float(track.bbox_diag_world_est) <= 0.0 or not has_projected_box

    def _hungarian_assign(self, score_matrix: torch.Tensor, valid_matrix: torch.Tensor) -> list[tuple[int, int]]:
        if score_matrix.numel() == 0:
            return []
        n_obs, n_nodes = score_matrix.shape
        if n_obs == 0 or n_nodes == 0:
            return []
        invalid_cost = 1e9
        cost = (-score_matrix).detach().cpu().numpy()
        valid = valid_matrix.detach().cpu().numpy().astype(bool)
        cost[~valid] = invalid_cost

        if linear_sum_assignment is not None:
            row_idx, col_idx = linear_sum_assignment(cost)
            pairs = []
            for r, c in zip(row_idx.tolist(), col_idx.tolist()):
                if r < n_obs and c < n_nodes and valid[r, c]:
                    pairs.append((r, c))
            return pairs

        # Greedy fallback only when scipy is unavailable.
        pairs = []
        used_r: set[int] = set()
        used_c: set[int] = set()
        flat = []
        for r in range(n_obs):
            for c in range(n_nodes):
                if valid[r, c]:
                    flat.append((float(cost[r, c]), r, c))
        for _, r, c in sorted(flat, key=lambda item: item[0]):
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
        return pairs

    def _remember_assoc_debug(self, frame_idx: int, debug: dict) -> None:
        self.frame_assoc_debug[frame_idx] = debug
        self._assoc_debug_order.append(frame_idx)
        while len(self._assoc_debug_order) > self._assoc_debug_order.maxlen:
            stale = self._assoc_debug_order.popleft()
            self.frame_assoc_debug.pop(stale, None)

    def _remember_extract_debug(self, frame_idx: int, debug: list[dict]) -> None:
        self.frame_extract_debug[frame_idx] = list(debug)
        self._extract_debug_order.append(frame_idx)
        while len(self._extract_debug_order) > self._extract_debug_order.maxlen:
            stale = self._extract_debug_order.popleft()
            self.frame_extract_debug.pop(stale, None)

    def associate_observations_to_nodes(self, observations: list[NodeObservation], frame_idx: int, frame, keyframes) -> None:
        import os as _os_assoc_prof
        _assoc_prof = _os_assoc_prof.environ.get("FG_PROFILE_STAGES", "0") == "1"
        if _assoc_prof:
            import time as _assoc_time
            _assoc_tt: dict[str, float] = defaultdict(float)

            def _assoc_t0() -> float:
                return _assoc_time.perf_counter()

            def _assoc_tick(name: str, t0: float) -> None:
                _assoc_tt[name] += _assoc_time.perf_counter() - t0

            def _assoc_count(name: str, count: int) -> None:
                self._record_runtime_profile(f"update_frame.associate.{name}", 0.0, count=max(0, int(count)))
        else:
            _assoc_tt = {}

            def _assoc_t0() -> float:
                return 0.0

            def _assoc_tick(name: str, t0: float) -> None:
                return None

            def _assoc_count(name: str, count: int) -> None:
                return None

        _t_assoc_debug = _assoc_t0()
        self._assoc_det_to_obs_context = {int(obs.det_idx): obs for obs in observations}
        # Step 1 + Step 4: per-frame geometry cache so stage1/stage2/projectability
        # all share the same projected_box / predicted_centroid / recent_bbox per
        # track.  Cleared below in `finally:` so no leak if this function raises.
        self._reset_assoc_frame_cache()
        try:
            debug = {
                "frame_idx": int(frame_idx),
                "num_observations_by_role": {"O": 0, "C": 0, "U": 0},
                "num_stage1_candidate_edges_by_role": {"O": 0, "C": 0, "U": 0},
                "num_stage1_matches_by_role": {"O": 0, "C": 0, "U": 0},
                "num_stage2_candidate_edges_by_role": {"O": 0, "C": 0, "U": 0},
                "num_stage2_matches_by_role": {"O": 0, "C": 0, "U": 0},
                "num_new_nodes_by_role": {"O": 0, "C": 0, "U": 0},
                "num_obs_dropped_low_points": int(self._last_obs_extract_stats.get("num_obs_dropped_low_points", 0)),
                "num_obs_depth_boxplot_changed": int(self._last_obs_extract_stats.get("num_obs_depth_boxplot_changed", 0)),
                "num_obs_dbscan_changed": int(self._last_obs_extract_stats.get("num_obs_dbscan_changed", 0)),
                "num_tracks_with_projectable_box_by_role": {"O": 0, "C": 0, "U": 0},
                "num_tracks_without_projectable_box_by_role": {"O": 0, "C": 0, "U": 0},
                "unmatched_observations": [],
                "observation_debug": [],
            }

            grouped_obs: dict[str, list[int]] = defaultdict(list)
            obs_debug: dict[int, dict] = {}
            for obs_idx, obs in enumerate(observations):
                grouped_obs[obs.role].append(obs_idx)
                debug["num_observations_by_role"][obs.role] = debug["num_observations_by_role"].get(obs.role, 0) + 1
                obs_debug[obs_idx] = {
                    "det_idx": int(obs.det_idx),
                    "label": obs.label,
                    "role": obs.role,
                    "matched_node_id": None,
                    "assigned_stage": None,
                    "created_new_node": False,
                    "unmatched_reason": None,
                    "stage1_candidates": [],
                    "stage2_candidates": [],
                    "final_status": "unmatched",
                }
            _assoc_tick("debug_build", _t_assoc_debug)

            assigned_obs: dict[int, str] = {}
            assigned_nodes: set[str] = set()
            unmatched_reason_by_obs: dict[int, str] = {}
            total_candidate_nodes = 0
            total_stage1_pairs = 0
            total_stage2_pairs = 0
            for role, obs_indices in grouped_obs.items():
                candidate_nodes = [node_id for node_id, track in self.node_tracks.items() if track.role == role]
                total_candidate_nodes += len(candidate_nodes)
                if not candidate_nodes:
                    for obs_idx in obs_indices:
                        unmatched_reason_by_obs[obs_idx] = "no_role_candidate"
                    continue

                probe_obs = observations[obs_indices[0]]
                _t = _assoc_t0()
                _, has_projected_box_by_track_id = self._projectability_by_track(
                    candidate_nodes,
                    frame=frame,
                    keyframes=keyframes,
                    probe_obs=probe_obs,
                )
                _assoc_tick("projectability", _t)
                stage1_track_ids: list[str] = []
                for node_id in candidate_nodes:
                    if not has_projected_box_by_track_id.get(node_id, False):
                        debug["num_tracks_without_projectable_box_by_role"][role] += 1
                    else:
                        debug["num_tracks_with_projectable_box_by_role"][role] += 1
                        stage1_track_ids.append(node_id)

                stage1_valid = torch.zeros((len(obs_indices), len(stage1_track_ids)), dtype=torch.bool)
                stage1_score = torch.full((len(obs_indices), len(stage1_track_ids)), -1e6, dtype=torch.float32)
                stage1_fail_reasons: dict[int, list[str]] = defaultdict(list)
                total_stage1_pairs += len(obs_indices) * len(stage1_track_ids)
                _t = _assoc_t0()
                for r, obs_idx in enumerate(obs_indices):
                    obs = observations[obs_idx]
                    for c, node_id in enumerate(stage1_track_ids):
                        candidate_debug = self._pair_assoc_debug_stage1(obs, self.node_tracks[node_id], frame, keyframes)
                        score, reason = self._score_reason_from_assoc_debug(candidate_debug)
                        candidate_debug["candidate_node_id"] = node_id
                        candidate_debug["valid"] = score is not None
                        candidate_debug["reject_reason"] = reason
                        if score is not None:
                            candidate_debug["score"] = float(score)
                        obs_debug[obs_idx]["stage1_candidates"].append(candidate_debug)
                        if score is None:
                            stage1_fail_reasons[obs_idx].append(reason)
                            continue
                        stage1_valid[r, c] = True
                        stage1_score[r, c] = float(score)
                        debug["num_stage1_candidate_edges_by_role"][role] += 1
                _assoc_tick("stage1_pair_score", _t)

                _t = _assoc_t0()
                if role == "C":
                    self._apply_stage1_carrier_geom_rerank(
                        obs_indices=obs_indices,
                        stage1_track_ids=stage1_track_ids,
                        stage1_valid=stage1_valid,
                        stage1_score=stage1_score,
                        obs_debug=obs_debug,
                    )

                # Multi-anchor weak assist: small additive bonus for stable
                # tracks with weak primary anchors and a supporting aux
                # anchor.  Bounded by multi_anchor_assist_max_bonus so it
                # cannot override primary geometry.
                self._apply_multi_anchor_assoc_assist(
                    obs_indices=obs_indices,
                    stage1_track_ids=stage1_track_ids,
                    stage1_valid=stage1_valid,
                    stage1_score=stage1_score,
                    obs_debug=obs_debug,
                    observations=observations,
                    frame_idx=frame_idx,
                )
                _assoc_tick("stage1_rerank_assist", _t)

                _t = _assoc_t0()
                for r, c in self._hungarian_assign(stage1_score, stage1_valid):
                    obs_idx = obs_indices[r]
                    node_id = stage1_track_ids[c]
                    if obs_idx in assigned_obs or node_id in assigned_nodes:
                        continue
                    assigned_obs[obs_idx] = node_id
                    assigned_nodes.add(node_id)
                    debug["num_stage1_matches_by_role"][role] += 1
                    obs_debug[obs_idx]["matched_node_id"] = node_id
                    obs_debug[obs_idx]["assigned_stage"] = "stage1"
                    obs_debug[obs_idx]["final_status"] = "matched_existing"
                _assoc_tick("hungarian", _t)

                _t = _assoc_t0()
                stage2_obs_indices = [idx for idx in obs_indices if idx not in assigned_obs]
                stage2_track_ids = []
                for node_id in candidate_nodes:
                    if node_id in assigned_nodes:
                        continue
                    track = self.node_tracks[node_id]
                    if self._is_stage2_candidate_track(
                        track,
                        has_projected_box=has_projected_box_by_track_id.get(node_id, False),
                    ):
                        stage2_track_ids.append(node_id)
                _assoc_tick("stage2_candidate_select", _t)

                stage2_valid = torch.zeros((len(stage2_obs_indices), len(stage2_track_ids)), dtype=torch.bool)
                stage2_score = torch.full((len(stage2_obs_indices), len(stage2_track_ids)), -1e6, dtype=torch.float32)
                stage2_fail_reasons: dict[int, list[str]] = defaultdict(list)
                total_stage2_pairs += len(stage2_obs_indices) * len(stage2_track_ids)
                _t = _assoc_t0()
                for r, obs_idx in enumerate(stage2_obs_indices):
                    obs = observations[obs_idx]
                    for c, node_id in enumerate(stage2_track_ids):
                        candidate_debug = self._pair_assoc_debug_stage2(obs, self.node_tracks[node_id], frame, keyframes)
                        score, reason = self._score_reason_from_assoc_debug(candidate_debug)
                        candidate_debug["candidate_node_id"] = node_id
                        candidate_debug["valid"] = score is not None
                        candidate_debug["reject_reason"] = reason
                        if score is not None:
                            candidate_debug["score"] = float(score)
                        obs_debug[obs_idx]["stage2_candidates"].append(candidate_debug)
                        if score is None:
                            stage2_fail_reasons[obs_idx].append(reason)
                            continue
                        stage2_valid[r, c] = True
                        stage2_score[r, c] = float(score)
                        debug["num_stage2_candidate_edges_by_role"][role] += 1
                _assoc_tick("stage2_pair_score", _t)

                _t = _assoc_t0()
                for r, c in self._hungarian_assign(stage2_score, stage2_valid):
                    obs_idx = stage2_obs_indices[r]
                    node_id = stage2_track_ids[c]
                    if obs_idx in assigned_obs or node_id in assigned_nodes:
                        continue
                    assigned_obs[obs_idx] = node_id
                    assigned_nodes.add(node_id)
                    debug["num_stage2_matches_by_role"][role] += 1
                    obs_debug[obs_idx]["matched_node_id"] = node_id
                    obs_debug[obs_idx]["assigned_stage"] = "stage2"
                    obs_debug[obs_idx]["final_status"] = "matched_existing"
                _assoc_tick("hungarian", _t)

                _t = _assoc_t0()
                for obs_idx in stage2_obs_indices:
                    if obs_idx in assigned_obs:
                        continue
                    stage1_reasons = stage1_fail_reasons.get(obs_idx, [])
                    stage2_reasons = stage2_fail_reasons.get(obs_idx, [])
                    if not stage1_track_ids and any(not has_projected_box_by_track_id.get(node_id, False) for node_id in candidate_nodes):
                        unmatched_reason_by_obs[obs_idx] = "all_stage1_no_projected_box"
                        obs_debug[obs_idx]["unmatched_reason"] = "all_stage1_no_projected_box"
                        continue
                    if not stage1_reasons and not stage2_reasons:
                        unmatched_reason_by_obs[obs_idx] = "hungarian_unmatched"
                        obs_debug[obs_idx]["unmatched_reason"] = "hungarian_unmatched"
                        continue
                    if stage1_reasons and all(reason == "overlap_zero" for reason in stage1_reasons):
                        unmatched_reason_by_obs[obs_idx] = "all_stage1_overlap_zero"
                        obs_debug[obs_idx]["unmatched_reason"] = "all_stage1_overlap_zero"
                        continue
                    if stage1_reasons and all(reason == "dist_fail" for reason in stage1_reasons):
                        unmatched_reason_by_obs[obs_idx] = "all_stage1_dist_fail"
                        obs_debug[obs_idx]["unmatched_reason"] = "all_stage1_dist_fail"
                        continue
                    if stage2_reasons and all(reason == "birth_gate_fail" for reason in stage2_reasons):
                        unmatched_reason_by_obs[obs_idx] = "stage2_birth_gate_fail"
                        obs_debug[obs_idx]["unmatched_reason"] = "stage2_birth_gate_fail"
                        continue
                    if stage2_reasons and all(reason == "parent_conflict" for reason in stage2_reasons):
                        unmatched_reason_by_obs[obs_idx] = "stage2_parent_conflict"
                        obs_debug[obs_idx]["unmatched_reason"] = "stage2_parent_conflict"
                        continue
                    unmatched_reason_by_obs[obs_idx] = "hungarian_unmatched"
                    obs_debug[obs_idx]["unmatched_reason"] = "hungarian_unmatched"
                _assoc_tick("debug_build", _t)

            _t = _assoc_t0()
            for obs_idx, obs in enumerate(observations):
                node_id = assigned_obs.get(obs_idx)
                if node_id is None:
                    node_id = self._new_node_id(obs.role)
                    self.node_tracks[node_id] = OnlineNodeTrack(
                        node_id=node_id,
                        label=obs.label,
                        role=obs.role,
                        label_counts={obs.label: 1},
                        enable_reanchor=bool(self.enable_reanchor),
                    )
                    debug["num_new_nodes_by_role"][obs.role] = debug["num_new_nodes_by_role"].get(obs.role, 0) + 1
                    obs_debug[obs_idx]["created_new_node"] = True
                    obs_debug[obs_idx]["assigned_stage"] = "new_birth"
                    obs_debug[obs_idx]["final_status"] = "new_birth"
                    obs_debug[obs_idx]["matched_node_id"] = node_id
            _assoc_tick("new_node_creation", _t)
            _t = _assoc_t0()
            for obs_idx, obs in enumerate(observations):
                node_id = obs_debug[obs_idx]["matched_node_id"] or assigned_obs.get(obs_idx)
                if node_id is None:
                    node_id = obs.matched_node_id
                obs.matched_node_id = node_id
                track = self.node_tracks[node_id]
                disable_fusion_now = self.graph_policy.should_disable_3d_fusion_for_observation(
                    obs.label,
                    frame_idx,
                    obs.det_idx,
                )
                track.disable_3d_fusion = bool(getattr(track, "disable_3d_fusion", False) or disable_fusion_now)
                track.update_observation(obs, frame_idx, frame)
                obs_debug[obs_idx]["matched_node_id"] = node_id
                obs_debug[obs_idx]["disable_3d_fusion"] = bool(getattr(track, "disable_3d_fusion", False))
                obs_debug[obs_idx]["disable_3d_fusion_policy_hit"] = bool(disable_fusion_now)
                self._mark_assoc_debug_outcome(
                    obs_debug[obs_idx]["stage1_candidates"],
                    matched_node_id=node_id,
                    assigned_stage=obs_debug[obs_idx].get("assigned_stage"),
                    stage="stage1",
                )
                self._mark_assoc_debug_outcome(
                    obs_debug[obs_idx]["stage2_candidates"],
                    matched_node_id=node_id,
                    assigned_stage=obs_debug[obs_idx].get("assigned_stage"),
                    stage="stage2",
                )

                if obs_idx not in assigned_obs:
                    reason = unmatched_reason_by_obs.get(obs_idx, "hungarian_unmatched")
                    debug["unmatched_observations"].append(
                        {
                            "det_idx": int(obs.det_idx),
                            "role": obs.role,
                            "label": obs.label,
                            "reason": reason,
                        }
                    )
                    obs_debug[obs_idx]["unmatched_reason"] = reason
            _assoc_tick("track_update", _t)

            _t = _assoc_t0()
            debug["observation_debug"] = [obs_debug[idx] for idx in range(len(observations))]
            try:
                self._record_assoc_observation_competition(
                    int(frame_idx),
                    debug["observation_debug"],
                )
            except Exception:
                pass
            self._remember_assoc_debug(frame_idx, debug)
            _assoc_tick("debug_build", _t)
            if _assoc_prof:
                for _name, _dt in _assoc_tt.items():
                    self._record_runtime_profile(f"update_frame.associate.{_name}", _dt)
                _assoc_count("frame_count", 1)
                _assoc_count("observation_count", len(observations))
                _assoc_count("candidate_track_count", total_candidate_nodes)
                _assoc_count("stage1_pair_count", total_stage1_pairs)
                _assoc_count("stage2_pair_count", total_stage2_pairs)
        finally:
            self._assoc_det_to_obs_context = None
            # Drop per-frame geometry cache so stale tensors don't leak.
            self._assoc_frame_cache = None

    def _parent_geom_consistency(self, child_obs: NodeObservation, parent_obs: Optional[NodeObservation]) -> float:
        if parent_obs is None:
            return 0.0
        scores: list[float] = []
        if child_obs.box_xyxy is not None and parent_obs.box_xyxy is not None:
            scores.append(box_overlap_ratio_xyxy(child_obs.box_xyxy, parent_obs.box_xyxy))
        if expanded_box_contains(child_obs.centroid_world, parent_obs.bbox3d_world, expansion=0.05):
            scores.append(1.0)
        else:
            parent_extent = bbox_size(parent_obs.bbox3d_world)
            dist = world_distance(child_obs.centroid_world, parent_obs.centroid_world)
            norm = max(float(torch.linalg.norm(parent_extent).item()), 1e-6)
            scores.append(max(0.0, 1.0 - min(1.0, dist / (norm + 1e-6))))
        return float(sum(scores) / max(1, len(scores)))

    def _local_edge_score(self, edge: dict, *, geom_consistency: float, owner_conflict_penalty: float, carrier_bypass_penalty: float) -> float:
        selected = 1.0 if edge.get("selected") else 0.0
        mask_contain = float(edge.get("mask_contain") or 0.0)
        contain = float(edge.get("contain") or 0.0)
        score = (
            self.local_weights["selected"] * selected
            + self.local_weights["mask"] * mask_contain
            + self.local_weights["box"] * contain
            + self.local_weights["vis"] * 1.0
            + self.local_weights["geom"] * geom_consistency
            - self.local_weights["bypass"] * carrier_bypass_penalty
            - self.local_weights["owner"] * owner_conflict_penalty
        )
        return max(0.0, score)

    def is_viable_preferred_parent(
        self,
        *,
        child_obs: NodeObservation,
        parent_track: OnlineNodeTrack,
        edge_obs: dict,
        posterior_state: Optional[LocalParentPosterior],
    ) -> bool:
        if child_obs.role == "U" and child_obs.semantic_owner_mode == "prefer_carrier" and parent_track.role != "C":
            return False

        strong_current = bool(edge_obs.get("selected")) or float(edge_obs.get("mask_contain") or 0.0) >= self.preferred_tau_mask or float(edge_obs.get("contain") or 0.0) >= self.preferred_tau_box
        if strong_current:
            return True
        if posterior_state is None:
            return False

        parent_id = str(parent_track.node_id)
        recent = posterior_state.recent_support_count(parent_id, window=self.local_history_size)
        if recent < self.preferred_min_recent_support:
            return False
        if posterior_state.recent_switch_count(window=self.local_history_size) > (self.local_max_switches + 1):
            return False
        return True

    def _has_recent_viable_preferred_carrier(self, child_id: str, posterior: LocalParentPosterior, *, exclude_parent_id: Optional[str] = None) -> bool:
        for parent_id in posterior.preferred_parent_ids:
            if parent_id == exclude_parent_id:
                continue
            parent_track = self.node_tracks.get(parent_id)
            if parent_track is None or parent_track.role != "C":
                continue
            recent = posterior.recent_support_count(parent_id, window=self.local_history_size)
            if recent >= self.preferred_min_recent_support:
                return True
        return False

    def _owner_conflict_penalty(self, child_id: str, parent_id: str) -> float:
        posterior = self.local_posteriors.get(child_id)
        if posterior is None or posterior.stable_parent_id is None or posterior.stable_parent_id == parent_id:
            return 0.0
        stable_support = posterior.support_count(posterior.stable_parent_id)
        return 1.0 if stable_support >= self.local_min_support_frames else 0.5

    def _submit_local_llava(self, child_id: str, posterior: LocalParentPosterior) -> None:
        if posterior.llava_requested:
            return
        child_track = self.node_tracks.get(child_id)
        if child_track is None:
            return
        candidate_parent_ids = [node_id for node_id in (posterior.top1_parent_id, posterior.top2_parent_id) if node_id is not None]
        candidate_parent_crops = {
            node_id: self.node_tracks[node_id].best_view_crop_meta
            for node_id in candidate_parent_ids
            if node_id in self.node_tracks
        }
        payload = {
            "child_node_id": child_id,
            "child_label": child_track.label,
            "child_role": child_track.role,
            "candidate_parent_ids": candidate_parent_ids,
            "candidate_parent_crop_meta": candidate_parent_crops,
            "best_view_frame": child_track.best_view_frame,
            "best_view_crop_meta": child_track.best_view_crop_meta,
            "margin": posterior.margin,
            "support_summary": posterior.top_support_summary(limit=2),
        }
        self.llava_scheduler.submit_local(child_id, payload)
        posterior.llava_requested = True

    def _current_local_graph_edge(self, child_id: str):
        for edge in self.graph.local_edges.values():
            if edge.dst_node_id == child_id:
                return edge
        return None

    # ------------------------------------------------------------------
    # Co-visibility / sibling-duplicate-risk helpers (node consolidation)
    # ------------------------------------------------------------------
    def _track_observed_frames_set(self, node_id: str) -> set[int]:
        track = self.node_tracks.get(node_id)
        if track is None:
            return set()
        try:
            return {int(f) for f in (track.observed_frames or [])}
        except Exception:
            return set()

    def node_covisibility_count(self, node_a: str, node_b: str) -> int:
        """Number of frames in which both nodes were observed.

        Co-visibility is a strong negative signal for "duplicate node":
        if two tracks were ever associated to observations in the same
        frame, they are guaranteed to be different physical objects.
        """
        if not node_a or not node_b or node_a == node_b:
            return 0
        a = self._track_observed_frames_set(node_a)
        if not a:
            return 0
        b = self._track_observed_frames_set(node_b)
        if not b:
            return 0
        return len(a & b)

    def nodes_are_covisible(self, node_a: str, node_b: str, min_count: int = 1) -> bool:
        return self.node_covisibility_count(node_a, node_b) >= int(max(1, min_count))

    @staticmethod
    def _labels_compatible(label_a: str, label_b: str) -> bool:
        """Loose label-equality test used for sibling-duplicate detection.

        Returns True if labels match exactly (case-insensitive) or if
        ``label_semantic_similarity`` reports ≥ 0.9 in either direction.
        """
        a = (label_a or "").strip().lower()
        b = (label_b or "").strip().lower()
        if not a or not b:
            return False
        if a == b:
            return True
        sim = max(label_semantic_similarity(a, b), label_semantic_similarity(b, a))
        return float(sim) >= 0.9

    def _track_dominant_label(self, node_id: str) -> str:
        track = self.node_tracks.get(node_id)
        if track is None:
            return ""
        try:
            return str(track.dominant_label() or track.label or "")
        except Exception:
            return str(getattr(track, "label", "") or "")

    def _stable_same_label_children_of_parent(
        self,
        parent_id: str,
        child_role: str,
        child_label: str,
        *,
        require_committed: bool = True,
    ) -> list[str]:
        """Children of ``parent_id`` whose role and label match the candidate.

        First version intentionally restricts to ``status == "committed"`` to
        avoid the gate firing on freshly-spawned tentative siblings (which
        may themselves be in flux).  ``require_committed=False`` widens to
        any current edge for diagnostics / future expansion.
        """
        if not parent_id or not child_role or not child_label:
            return []
        out: list[str] = []
        for edge in self.graph.local_edges.values():
            if str(edge.src_node_id) != str(parent_id):
                continue
            if require_committed and str(edge.status).lower() != "committed":
                continue
            sibling_id = str(edge.dst_node_id)
            sibling_track = self.node_tracks.get(sibling_id)
            if sibling_track is None:
                continue
            if str(getattr(sibling_track, "role", "") or "") != str(child_role):
                continue
            sib_label = self._track_dominant_label(sibling_id)
            if not self._labels_compatible(sib_label, child_label):
                continue
            out.append(sibling_id)
        return out

    @staticmethod
    def _bbox_world_minmax(track) -> Optional[tuple[list[float], list[float]]]:
        bbox = getattr(track, "bbox_world_est", None)
        if not isinstance(bbox, dict):
            return None
        bmin = bbox.get("min")
        bmax = bbox.get("max")
        if bmin is None or bmax is None:
            return None
        try:
            bmin = [float(x) for x in bmin]
            bmax = [float(x) for x in bmax]
        except Exception:
            return None
        if len(bmin) != 3 or len(bmax) != 3:
            return None
        return bmin, bmax

    @staticmethod
    def _centroid_distance(track_a, track_b) -> float:
        ca = getattr(track_a, "centroid_world_est", None)
        cb = getattr(track_b, "centroid_world_est", None)
        if not ca or not cb:
            return float("inf")
        try:
            ca = [float(x) for x in ca]
            cb = [float(x) for x in cb]
        except Exception:
            return float("inf")
        if len(ca) < 3 or len(cb) < 3:
            return float("inf")
        dx = ca[0] - cb[0]
        dy = ca[1] - cb[1]
        dz = ca[2] - cb[2]
        return float((dx * dx + dy * dy + dz * dz) ** 0.5)

    @classmethod
    def _bbox_iou_3d(cls, track_a, track_b) -> tuple[float, float]:
        """Return (iou, overlap_ratio) for two AABBs in world space.

        ``overlap_ratio`` is intersection volume divided by the smaller
        of the two box volumes (more lenient than IoU for size mismatch).
        Returns (0.0, 0.0) on missing data.
        """
        ba = cls._bbox_world_minmax(track_a)
        bb = cls._bbox_world_minmax(track_b)
        if ba is None or bb is None:
            return 0.0, 0.0
        amin, amax = ba
        bmin, bmax = bb
        inter = 1.0
        for k in range(3):
            lo = max(amin[k], bmin[k])
            hi = min(amax[k], bmax[k])
            d = hi - lo
            if d <= 0.0:
                return 0.0, 0.0
            inter *= d
        vol_a = max(1e-9, (amax[0] - amin[0]) * (amax[1] - amin[1]) * (amax[2] - amin[2]))
        vol_b = max(1e-9, (bmax[0] - bmin[0]) * (bmax[1] - bmin[1]) * (bmax[2] - bmin[2]))
        union = vol_a + vol_b - inter
        iou = inter / union if union > 1e-9 else 0.0
        overlap_ratio = inter / max(1e-9, min(vol_a, vol_b))
        return float(iou), float(overlap_ratio)

    def _node_quality_score(self, node_id: str) -> tuple[float, dict]:
        """Deterministic quality score combining geometry, observation, and
        edge-degree signals.  Returns ``(score, components_dict)`` for
        debug.  Higher is better.
        """
        track = self.node_tracks.get(node_id)
        components: dict = {}
        if track is None:
            return 0.0, components
        score = 0.0
        # Geometry source
        try:
            geom_source = str(track.assoc_geom_source() or "")
        except Exception:
            geom_source = ""
        components["geom_source"] = geom_source
        if "stable_anchor_fused" in geom_source:
            score += 3.5
        elif "stable_anchor" in geom_source:
            score += 3.0
        elif "provisional" in geom_source:
            score += 2.0
        elif "candidate" in geom_source:
            score += 1.5
        elif "world_bbox" in geom_source:
            score += 0.5
        elif "world_centroid" in geom_source:
            score += 0.2
        if bool(getattr(track, "stable_geom_ready", False)):
            score += 1.0
            components["stable_geom_ready"] = True
        # Anchor point counts.
        for attr, weight in (
            ("primary_anchor_num_points", 0.005),
            ("fused_num_points", 0.003),
            ("candidate_num_points", 0.002),
        ):
            v = float(getattr(track, attr, 0) or 0)
            score += weight * min(v, 1500.0)
            components[attr] = v
        # Observation counts (capped).
        obs_count = int(getattr(track, "obs_count", 0) or 0)
        visible_count = int(getattr(track, "visible_count", 0) or 0)
        score += 0.2 * min(obs_count, 10)
        score += 0.1 * min(visible_count, 10)
        components["obs_count"] = obs_count
        components["visible_count"] = visible_count
        # Best view score.
        best_view = float(getattr(track, "best_view_score", 0.0) or 0.0)
        score += min(best_view, 2.0)
        components["best_view_score"] = best_view
        # Edge degree (committed edges count more than tentative).
        comm_deg = 0
        tent_deg = 0
        for edge in self.graph.local_edges.values():
            if edge.src_node_id == node_id or edge.dst_node_id == node_id:
                if str(edge.status).lower() == "committed":
                    comm_deg += 1
                else:
                    tent_deg += 1
        for edge in self.graph.remote_edges.values():
            if edge.src_node_id == node_id or edge.dst_node_id == node_id:
                if str(edge.status).lower() == "committed":
                    comm_deg += 1
                else:
                    tent_deg += 1
        score += 2.0 * min(comm_deg, 4)
        score += 0.3 * min(tent_deg, 4)
        components["committed_edge_degree"] = comm_deg
        components["tentative_edge_degree"] = tent_deg
        # Has any committed edge -> small bonus floor so committed
        # endpoints never read as "low quality" purely from geometry noise.
        if comm_deg > 0:
            score += 1.5
        return float(score), components

    def _committed_local_parent_of(self, child_id: str) -> Optional[str]:
        for edge in self.graph.local_edges.values():
            if (
                str(edge.dst_node_id) == str(child_id)
                and str(edge.status).lower() == "committed"
            ):
                return str(edge.src_node_id)
        return None

    # ------------------------------------------------------------------
    # Parent-context helper: estimate a node's most likely local parent
    # from multiple sources (committed -> tentative -> posterior_stable
    # -> posterior_top1 -> latest_signal).  Read-only.
    # ------------------------------------------------------------------
    def _node_parent_context(self, node_id: str) -> dict:
        """Return a dict describing the most likely local parent of ``node_id``.

        Inspects (in order of decreasing trust): committed local edges,
        tentative local edges, posterior.stable_parent_id,
        posterior.top1_parent_id and the latest assignment signals.  Never
        mutates state.  Returns ``source="none"`` and empty parent ids
        when nothing usable is known.

        Returned fields::

            node_id, committed_parent_id, committed_edge_type,
            tentative_parent_id, tentative_edge_type,
            posterior_top1_parent_id, posterior_top1_edge_type,
            posterior_stable_parent_id,
            latest_signal_parent_id, latest_signal_edge_type,
            best_parent_id, best_edge_type, source, confidence
        """
        ctx: dict = {
            "node_id": str(node_id),
            "committed_parent_id": None,
            "committed_edge_type": None,
            "tentative_parent_id": None,
            "tentative_edge_type": None,
            "posterior_top1_parent_id": None,
            "posterior_top1_edge_type": None,
            "posterior_stable_parent_id": None,
            "latest_signal_parent_id": None,
            "latest_signal_edge_type": None,
            "best_parent_id": None,
            "best_edge_type": None,
            "source": "none",
            "confidence": 0.0,
        }
        track = self.node_tracks.get(node_id)
        if track is None:
            return ctx
        # Local edges (committed first, then tentative).
        for edge in self.graph.local_edges.values():
            if str(edge.dst_node_id) != str(node_id):
                continue
            status = str(getattr(edge, "status", "") or "").lower()
            if status == "committed" and ctx["committed_parent_id"] is None:
                ctx["committed_parent_id"] = str(edge.src_node_id)
                ctx["committed_edge_type"] = str(getattr(edge, "edge_type", "") or "")
            elif status == "tentative" and ctx["tentative_parent_id"] is None:
                ctx["tentative_parent_id"] = str(edge.src_node_id)
                ctx["tentative_edge_type"] = str(getattr(edge, "edge_type", "") or "")
        # Posterior.
        posterior = self.local_posteriors.get(node_id)
        if posterior is not None:
            stable_pid = getattr(posterior, "stable_parent_id", None)
            if stable_pid:
                ctx["posterior_stable_parent_id"] = str(stable_pid)
            top1_pid = getattr(posterior, "top1_parent_id", None)
            if top1_pid:
                ctx["posterior_top1_parent_id"] = str(top1_pid)
                top1_track = self.node_tracks.get(str(top1_pid))
                if top1_track is not None:
                    ctx["posterior_top1_edge_type"] = self._edge_type_for_roles(
                        getattr(top1_track, "role", ""),
                        getattr(track, "role", ""),
                    )
        # Latest assignment signal: pick strongest entry.
        sig_map = self._latest_local_assignment_signals.get(node_id, {})
        if sig_map:
            best_sig_pid = None
            best_sig_strength = -1.0
            best_sig_etype = None
            for pid, sig in sig_map.items():
                strength = max(
                    float(sig.get("strength", 0.0) or 0.0),
                    float(sig.get("edge_evidence_strength", 0.0) or 0.0),
                )
                if strength > best_sig_strength:
                    best_sig_strength = strength
                    best_sig_pid = pid
                    best_sig_etype = str(sig.get("edge_type") or "") or None
            if best_sig_pid is not None:
                ctx["latest_signal_parent_id"] = str(best_sig_pid)
                ctx["latest_signal_edge_type"] = best_sig_etype
        # Pick best.
        if ctx["committed_parent_id"]:
            ctx["best_parent_id"] = ctx["committed_parent_id"]
            ctx["best_edge_type"] = ctx["committed_edge_type"]
            ctx["source"] = "committed"
            ctx["confidence"] = 1.0
        elif ctx["tentative_parent_id"]:
            ctx["best_parent_id"] = ctx["tentative_parent_id"]
            ctx["best_edge_type"] = ctx["tentative_edge_type"]
            ctx["source"] = "tentative"
            ctx["confidence"] = 0.7
        elif ctx["posterior_stable_parent_id"]:
            ctx["best_parent_id"] = ctx["posterior_stable_parent_id"]
            ctx["source"] = "posterior_stable"
            ctx["confidence"] = 0.6
            ptrack = self.node_tracks.get(ctx["posterior_stable_parent_id"])
            if ptrack is not None:
                ctx["best_edge_type"] = self._edge_type_for_roles(
                    getattr(ptrack, "role", ""), getattr(track, "role", "")
                )
        elif ctx["posterior_top1_parent_id"]:
            ctx["best_parent_id"] = ctx["posterior_top1_parent_id"]
            ctx["best_edge_type"] = ctx["posterior_top1_edge_type"]
            ctx["source"] = "posterior_top1"
            ctx["confidence"] = 0.4
        elif ctx["latest_signal_parent_id"]:
            ctx["best_parent_id"] = ctx["latest_signal_parent_id"]
            ctx["best_edge_type"] = ctx["latest_signal_edge_type"]
            ctx["source"] = "latest_signal"
            ctx["confidence"] = 0.25
        return ctx

    def _lowshot_sibling_duplicate_risk(
        self,
        *,
        parent_id: str,
        child_id: str,
        edge_type: str,
        signal: dict,
    ) -> tuple[bool, dict]:
        """Return (risk, debug).

        Risk is True only when there exists a committed sibling under the
        same parent with the same role and (≥0.9 similar) label, the two
        tracks have **never** been co-visible, and the candidate child's
        quality is meaningfully weaker than the sibling's.  All other
        cases (no sibling / has co-visibility / candidate quality clearly
        higher) return False.
        """
        debug: dict = {
            "sibling_duplicate_risk": False,
            "duplicate_risk_parent_id": parent_id,
            "duplicate_risk_sibling_id": None,
            "sibling_covisibility_count": 0,
            "sibling_child_quality": 0.0,
            "current_child_quality": 0.0,
            "sibling_centroid_distance": float("inf"),
            "sibling_bbox_iou": 0.0,
            "sibling_bbox_overlap": 0.0,
            "duplicate_risk_action": "allow",
            "duplicate_risk_reason": "no_sibling",
        }
        child_track = self.node_tracks.get(child_id)
        parent_track = self.node_tracks.get(parent_id)
        if child_track is None or parent_track is None:
            debug["duplicate_risk_reason"] = "missing_track"
            return False, debug
        child_role = str(child_track.role or "")
        child_label = self._track_dominant_label(child_id)
        if not child_role or not child_label:
            debug["duplicate_risk_reason"] = "missing_role_or_label"
            return False, debug
        siblings = self._stable_same_label_children_of_parent(
            parent_id, child_role, child_label, require_committed=True
        )
        # Filter out the candidate itself (defensive: should never appear).
        siblings = [s for s in siblings if s != child_id]
        if not siblings:
            return False, debug
        # Skip if the candidate already has a committed link of its own:
        # this is not a low-shot path.
        if self._committed_local_parent_of(child_id) is not None:
            debug["duplicate_risk_reason"] = "child_already_committed"
            return False, debug
        my_quality, my_qc = self._node_quality_score(child_id)
        debug["current_child_quality"] = my_quality
        debug["current_child_quality_components"] = my_qc

        worst_offender = None
        worst_components: dict = {}
        for sib_id in siblings:
            covis = self.node_covisibility_count(child_id, sib_id)
            if covis >= 1:
                # Strong negative evidence: not a duplicate.
                continue
            sib_track = self.node_tracks.get(sib_id)
            if sib_track is None:
                continue
            sib_quality, sib_qc = self._node_quality_score(sib_id)
            centroid = self._centroid_distance(child_track, sib_track)
            iou, overlap = self._bbox_iou_3d(child_track, sib_track)
            # Quality gap (sibling stronger than candidate).
            quality_gap = float(sib_quality - my_quality)
            # Geometry close threshold per role (m).
            geom_thresh = {"O": 0.20, "C": 0.12, "U": 0.08}.get(child_role, 0.15)
            geom_close = (centroid <= geom_thresh) or (overlap >= 0.20) or (iou >= 0.10)
            # Block when:
            #   - sibling is meaningfully stronger (gap >= 1.5); OR
            #   - geometry overlaps significantly even if quality gap is small.
            block = False
            if quality_gap >= 1.5:
                block = True
            elif geom_close and quality_gap >= 0.0:
                block = True
            cand = {
                "sibling_id": sib_id,
                "covisibility_count": int(covis),
                "sibling_child_quality": float(sib_quality),
                "sibling_quality_components": sib_qc,
                "sibling_centroid_distance": float(centroid),
                "sibling_bbox_iou": float(iou),
                "sibling_bbox_overlap": float(overlap),
                "quality_gap": quality_gap,
                "geom_close": bool(geom_close),
                "block": bool(block),
            }
            if not block:
                continue
            # Pick the worst offender (highest quality_gap then closest geometry).
            if (
                worst_offender is None
                or quality_gap > worst_offender["quality_gap"]
                or (
                    quality_gap == worst_offender["quality_gap"]
                    and centroid < worst_offender["sibling_centroid_distance"]
                )
            ):
                worst_offender = cand
                worst_components = sib_qc

        if worst_offender is None:
            debug["duplicate_risk_reason"] = "no_blocking_sibling"
            return False, debug

        debug.update({
            "sibling_duplicate_risk": True,
            "duplicate_risk_sibling_id": worst_offender["sibling_id"],
            "sibling_covisibility_count": worst_offender["covisibility_count"],
            "sibling_child_quality": worst_offender["sibling_child_quality"],
            "sibling_quality_components": worst_components,
            "sibling_centroid_distance": worst_offender["sibling_centroid_distance"],
            "sibling_bbox_iou": worst_offender["sibling_bbox_iou"],
            "sibling_bbox_overlap": worst_offender["sibling_bbox_overlap"],
            "duplicate_risk_action": "block_lowshot",
            "duplicate_risk_reason": "sibling_stronger_no_covis",
        })
        return True, debug

    # ------------------------------------------------------------------
    # Node consolidation (merge / prune duplicate or orphan tracks)
    # ------------------------------------------------------------------
    def _committed_local_parents_of(self, node_id: str) -> list[str]:
        """All parents of ``node_id`` connected by committed local edges."""
        out: list[str] = []
        for edge in self.graph.local_edges.values():
            if (
                str(edge.dst_node_id) == str(node_id)
                and str(edge.status).lower() == "committed"
            ):
                out.append(str(edge.src_node_id))
        return out

    def _has_any_committed_edge(self, node_id: str) -> bool:
        for edge in self.graph.local_edges.values():
            if (
                str(edge.status).lower() == "committed"
                and (edge.src_node_id == node_id or edge.dst_node_id == node_id)
            ):
                return True
        for edge in self.graph.remote_edges.values():
            if (
                str(edge.status).lower() == "committed"
                and (edge.src_node_id == node_id or edge.dst_node_id == node_id)
            ):
                return True
        return False

    def _has_any_tentative_local_edge(self, node_id: str) -> bool:
        for edge in self.graph.local_edges.values():
            if (
                str(edge.status).lower() == "tentative"
                and (edge.src_node_id == node_id or edge.dst_node_id == node_id)
            ):
                return True
        return False

    def _has_any_remote_edge(self, node_id: str) -> bool:
        """True iff any remote edge (committed or tentative) touches the node."""
        for edge in self.graph.remote_edges.values():
            if edge.src_node_id == node_id or edge.dst_node_id == node_id:
                return True
        return False

    def _has_strong_remote_edge(self, node_id: str) -> bool:
        for edge in self.graph.remote_edges.values():
            if edge.src_node_id == node_id or edge.dst_node_id == node_id:
                if str(edge.status).lower() == "committed":
                    return True
                # A tentative remote edge with non-trivial support still
                # blocks orphan-U pruning.
                if int(getattr(edge, "support_count", 0) or 0) >= 2:
                    return True
        return False

    def _track_stale_frames(self, node_id: str) -> int:
        track = self.node_tracks.get(node_id)
        if track is None:
            return 0
        last = getattr(track, "last_seen_frame", None)
        if last is None:
            return 0
        latest_frame = max((int(f) for f in self._frame_order), default=int(last))
        return int(max(0, latest_frame - int(last)))

    def _record_parent_competition_loss(
        self,
        node_id: str,
        *,
        winner_id: Optional[str],
        parent_id: Optional[str],
        reason: str,
        margin: float,
        frame_idx: int,
    ) -> None:
        """Append one parent-competition loss to ``node_parent_competition_losses``.

        Bounded list of recent loss frames so the orphan-U pruner can
        consult counts and recency without unbounded growth.
        """
        if node_id is None:
            return
        rec = self.node_parent_competition_losses.setdefault(
            str(node_id),
            {
                "loss_count": 0,
                "loss_frames": [],
                "last_loss_frame": None,
                "last_winner_id": None,
                "last_parent_id": None,
                "last_reason": None,
                "last_margin": 0.0,
            },
        )
        rec["loss_count"] = int(rec.get("loss_count", 0)) + 1
        frames = rec.setdefault("loss_frames", [])
        if not frames or frames[-1] != int(frame_idx):
            frames.append(int(frame_idx))
            if len(frames) > 32:
                del frames[: len(frames) - 32]
        rec["last_loss_frame"] = int(frame_idx)
        rec["last_winner_id"] = winner_id
        rec["last_parent_id"] = parent_id
        rec["last_reason"] = str(reason or "")
        rec["last_margin"] = float(margin or 0.0)

    def _carrier_merge_safety_gate(
        self, survivor_id: str, loser_id: str, candidate: dict
    ) -> tuple[bool, dict]:
        """Role-level safety gate for ``role == "C"`` merges.

        C carriers are special: a wrong C merge contaminates downstream
        U-C local edges (the U0084-C0038 bug).  We therefore demand
        *all* of the following before allowing a competition-duplicate
        C-role merge to fire:

        1. centroid_distance <= 0.40 m (absolute cap, role-aware param);
        2. bbox_iou >= 0.10 OR bbox_overlap >= 0.20;
        3. loser quality strictly lower than survivor quality;
        4. loser has no committed local edge of its own;
        5. loser has no strong child edge / recent child observation
           (otherwise we risk ripping a real instance off its children).
        6. no co-visibility (already enforced earlier in the pipeline).

        Returns ``(passed, debug_dict)``.  ``debug_dict`` always has the
        per-condition booleans so the consolidation event can show why a
        C carrier merge was rejected.

        This is a role-level gate, not a category special case.
        """
        debug: dict = {
            "carrier_gate_pass": False,
            "centroid_ok": False,
            "overlap_ok": False,
            "loser_quality_lower": False,
            "loser_has_committed_edge": False,
            "loser_has_child_edges": False,
            "loser_recent_child_evidence": False,
            "reason": "",
        }
        centroid_distance = float(candidate.get("centroid_distance", 0.0))
        abs_cap = float(self.carrier_merge_centroid_abs_cap)
        debug["centroid_distance"] = centroid_distance
        debug["centroid_cap"] = float(abs_cap)
        debug["centroid_ok"] = bool(centroid_distance <= abs_cap)
        bbox_iou = float(candidate.get("bbox_iou", 0.0))
        bbox_overlap = float(candidate.get("bbox_overlap", 0.0))
        min_iou = float(self.carrier_merge_min_iou)
        min_overlap = float(self.carrier_merge_min_overlap)
        debug["bbox_iou"] = bbox_iou
        debug["bbox_overlap"] = bbox_overlap
        debug["overlap_ok"] = bool(bbox_iou >= min_iou or bbox_overlap >= min_overlap)
        # Quality direction.
        survivor_q = (
            candidate["quality_a"]
            if survivor_id == candidate.get("a")
            else candidate["quality_b"]
        )
        loser_q = (
            candidate["quality_a"]
            if loser_id == candidate.get("a")
            else candidate["quality_b"]
        )
        debug["survivor_quality"] = float(survivor_q)
        debug["loser_quality"] = float(loser_q)
        debug["loser_quality_lower"] = bool(float(loser_q) < float(survivor_q))
        # Committed / child edge checks on the loser.
        loser_committed = bool(self._has_any_committed_edge(loser_id))
        debug["loser_has_committed_edge"] = loser_committed
        # "Strong child edge": any local edge where loser is the source
        # (parent) and the edge is committed or has tentative evidence.
        loser_has_child_edges = False
        loser_recent_child_evidence = False
        try:
            for edge in self.graph.local_edges.values():
                if edge.src_node_id != loser_id:
                    continue
                status = str(getattr(edge, "status", "")).lower()
                if status == "committed":
                    loser_has_child_edges = True
                    break
                # Tentative child edge with non-trivial evidence counts.
                ev_cnt = int(getattr(edge, "evidence_count", 0) or 0)
                if ev_cnt >= 1:
                    loser_recent_child_evidence = True
        except Exception:
            pass
        debug["loser_has_child_edges"] = loser_has_child_edges
        debug["loser_recent_child_evidence"] = loser_recent_child_evidence
        # Final gate: every condition must pass.
        if not debug["centroid_ok"]:
            debug["reason"] = "centroid_too_far"
            return False, debug
        if not debug["overlap_ok"]:
            debug["reason"] = "carrier_overlap_too_low"
            return False, debug
        if not debug["loser_quality_lower"]:
            debug["reason"] = "loser_quality_not_lower"
            return False, debug
        if loser_committed:
            debug["reason"] = "loser_has_committed_edge"
            return False, debug
        if loser_has_child_edges:
            debug["reason"] = "loser_has_strong_child_edges"
            return False, debug
        # Recent child evidence is a soft signal: only block if the loser
        # also has *no* committed parent of its own (already true here)
        # AND has child evidence — otherwise it might still be a real
        # carrier.  Be conservative: block.
        if loser_recent_child_evidence:
            debug["reason"] = "loser_recent_child_evidence"
            return False, debug
        debug["carrier_gate_pass"] = True
        debug["reason"] = "ok"
        return True, debug

    def _replace_node_id_in_local_edges(self, old_id: str, new_id: str) -> dict:
        from .graph_commit import PersistentFunctionalGraph as _G
        out: dict = {}
        merged_count = 0
        dropped_self_loops = 0
        for key, edge in self.graph.local_edges.items():
            src = new_id if edge.src_node_id == old_id else edge.src_node_id
            dst = new_id if edge.dst_node_id == old_id else edge.dst_node_id
            if src == dst:
                dropped_self_loops += 1
                continue
            edge.src_node_id = src
            edge.dst_node_id = dst
            new_key = (src, dst, edge.edge_type)
            existing = out.get(new_key)
            if existing is None:
                out[new_key] = edge
                continue
            existing_rank = _G._status_rank(existing.status)
            incoming_rank = _G._status_rank(edge.status)
            existing_strength = _G._edge_strength_tuple(existing)
            incoming_strength = _G._edge_strength_tuple(edge)
            if incoming_rank > existing_rank or (
                incoming_rank == existing_rank and incoming_strength > existing_strength
            ):
                out[new_key] = edge
            merged_count += 1
        self.graph.local_edges = out
        return {"merged_count": merged_count, "dropped_self_loops": dropped_self_loops}

    def _replace_node_id_in_remote_edges(self, old_id: str, new_id: str) -> dict:
        from .graph_commit import PersistentFunctionalGraph as _G
        out: dict = {}
        merged = 0
        dropped = 0
        for old_key, edge in self.graph.remote_edges.items():
            src = new_id if edge.src_node_id == old_id else edge.src_node_id
            dst = new_id if edge.dst_node_id == old_id else edge.dst_node_id
            if src == dst:
                dropped += 1
                continue
            edge.src_node_id = src
            edge.dst_node_id = dst
            # Rebuild pair-specific dict key so node-id rewrites stay
            # consistent with ``_remote_graph_edge_key``.  Legacy/label-only
            # keys (no pair suffix) are kept as-is by ``_split_*``.
            label_key, _, _ = self._split_remote_graph_edge_key(old_key)
            new_key = self._remote_graph_edge_key(label_key, src, dst) if label_key != old_key else old_key
            existing = out.get(new_key)
            if existing is None:
                out[new_key] = edge
                continue
            existing_rank = _G._status_rank(existing.status)
            incoming_rank = _G._status_rank(edge.status)
            existing_strength = _G._edge_strength_tuple(existing)
            incoming_strength = _G._edge_strength_tuple(edge)
            if incoming_rank > existing_rank or (
                incoming_rank == existing_rank and incoming_strength > existing_strength
            ):
                out[new_key] = edge
            merged += 1
        self.graph.remote_edges = out
        return {"merged_count": merged, "dropped_self_loops": dropped}

    def _replace_node_id_in_posteriors(self, old_id: str, new_id: str) -> dict:
        merged = {
            "local_migrated": 0,
            "remote_migrated": 0,
            "rewritten_parent_child_ids": [],
        }
        if old_id in self.local_posteriors:
            old_post = self.local_posteriors.pop(old_id)
            existing = self.local_posteriors.get(new_id)
            if existing is None:
                old_post.child_node_id = new_id
                self.local_posteriors[new_id] = old_post
            else:
                for pid, score in old_post.candidate_parent_scores.items():
                    existing.candidate_parent_scores[pid] = (
                        existing.candidate_parent_scores.get(pid, 0.0) + float(score)
                    )
                for pid, frames in old_post.candidate_support_frames.items():
                    fr = existing.candidate_support_frames.setdefault(pid, [])
                    for f in frames:
                        if f not in fr:
                            fr.append(int(f))
                ranked = sorted(
                    existing.candidate_parent_scores.items(),
                    key=lambda kv: kv[1], reverse=True,
                )
                existing.top1_parent_id = ranked[0][0] if ranked else None
                existing.top2_parent_id = ranked[1][0] if len(ranked) > 1 else None
            merged["local_migrated"] = 1
        for child_id, posterior in list(self.local_posteriors.items()):
            parent_ref_rewritten = False
            for attr_dict in (posterior.candidate_parent_scores, posterior.candidate_support_frames):
                if old_id in attr_dict:
                    parent_ref_rewritten = True
                    val = attr_dict.pop(old_id)
                    if isinstance(val, list):
                        merged_list = attr_dict.setdefault(new_id, [])
                        for v in val:
                            if v not in merged_list:
                                merged_list.append(v)
                    else:
                        attr_dict[new_id] = float(attr_dict.get(new_id, 0.0)) + float(val)
            for attr_name in ("top1_parent_id", "top2_parent_id", "stable_parent_id"):
                if getattr(posterior, attr_name, None) == old_id:
                    parent_ref_rewritten = True
                    setattr(posterior, attr_name, new_id)
            posterior.preferred_parent_ids = {
                (new_id if pid == old_id else pid) for pid in posterior.preferred_parent_ids
            }
            posterior.fallback_parent_ids = {
                (new_id if pid == old_id else pid) for pid in posterior.fallback_parent_ids
            }
            posterior.recent_top1_history = [
                (new_id if pid == old_id else pid) for pid in posterior.recent_top1_history
            ]
            if parent_ref_rewritten and child_id not in merged["rewritten_parent_child_ids"]:
                merged["rewritten_parent_child_ids"].append(str(child_id))
        for post in self.remote_posteriors.values():
            def _rep(pair):
                if pair is None:
                    return None
                a, b = pair
                if a == old_id or b == old_id:
                    return (new_id if a == old_id else a, new_id if b == old_id else b)
                return pair
            new_pairs: dict = {}
            for pair, score in post.candidate_pairs.items():
                p = _rep(pair) or pair
                new_pairs[p] = float(new_pairs.get(p, 0.0)) + float(score)
            post.candidate_pairs = new_pairs
            new_frames: dict = {}
            for pair, frames in post.support_frames.items():
                p = _rep(pair) or pair
                fr = new_frames.setdefault(p, [])
                for f in frames:
                    if f not in fr:
                        fr.append(int(f))
            post.support_frames = new_frames
            post.top1_pair = _rep(post.top1_pair)
            post.top2_pair = _rep(post.top2_pair)
            post.stable_pair = _rep(post.stable_pair)
            post.recent_top1_history = [_rep(p) for p in post.recent_top1_history]
            merged["remote_migrated"] += 1
        return merged

    def _replace_node_id_in_latest_current_parent_claims(
        self, old_id: str, new_id: str
    ) -> dict:
        merged_children = 0
        merged_parent_signals = 0
        claims = getattr(self, "_latest_current_parent_claims", None)
        if not isinstance(claims, dict):
            return {"merged_children": 0, "merged_parent_signals": 0}
        out: dict[str, dict[str, dict]] = {}
        for child_id, parent_map in claims.items():
            new_child_id = new_id if child_id == old_id else child_id
            dst_parent_map = out.setdefault(new_child_id, {})
            if new_child_id != child_id:
                merged_children += 1
            if not isinstance(parent_map, dict):
                continue
            for parent_id, signal in parent_map.items():
                new_parent_id = new_id if parent_id == old_id else parent_id
                existing = dst_parent_map.get(new_parent_id)
                if existing is None:
                    dst_parent_map[new_parent_id] = dict(signal or {})
                    continue
                merged_parent_signals += 1
                existing["assignment_strength"] = max(
                    float(existing.get("assignment_strength", 0.0) or 0.0),
                    float((signal or {}).get("assignment_strength", 0.0) or 0.0),
                )
                existing["mask_contain"] = max(
                    float(existing.get("mask_contain", 0.0) or 0.0),
                    float((signal or {}).get("mask_contain", 0.0) or 0.0),
                )
                existing["contain"] = max(
                    float(existing.get("contain", 0.0) or 0.0),
                    float((signal or {}).get("contain", 0.0) or 0.0),
                )
                existing["selected"] = bool(existing.get("selected", False)) or bool(
                    (signal or {}).get("selected", False)
                )
                existing["strong_current"] = bool(existing.get("strong_current", False)) or bool(
                    (signal or {}).get("strong_current", False)
                )
        self._latest_current_parent_claims = out
        return {
            "merged_children": int(merged_children),
            "merged_parent_signals": int(merged_parent_signals),
        }

    def _replace_node_id_in_latest_local_assignment_signals(
        self, old_id: str, new_id: str
    ) -> dict:
        merged_children = 0
        merged_parent_signals = 0
        claims = getattr(self, "_latest_local_assignment_signals", None)
        if not isinstance(claims, dict):
            return {"merged_children": 0, "merged_parent_signals": 0}
        out: dict[str, dict[str, dict]] = {}
        for child_id, parent_map in claims.items():
            new_child_id = new_id if child_id == old_id else child_id
            dst_parent_map = out.setdefault(new_child_id, {})
            if new_child_id != child_id:
                merged_children += 1
            if not isinstance(parent_map, dict):
                continue
            for parent_id, signal in parent_map.items():
                new_parent_id = new_id if parent_id == old_id else parent_id
                existing = dst_parent_map.get(new_parent_id)
                if existing is None:
                    dst_parent_map[new_parent_id] = dict(signal or {})
                    continue
                merged_parent_signals += 1
                existing["strength"] = max(
                    float(existing.get("strength", 0.0) or 0.0),
                    float((signal or {}).get("strength", 0.0) or 0.0),
                )
                existing["edge_evidence_strength"] = max(
                    float(existing.get("edge_evidence_strength", 0.0) or 0.0),
                    float((signal or {}).get("edge_evidence_strength", 0.0) or 0.0),
                )
                existing["score"] = max(
                    float(existing.get("score", 0.0) or 0.0),
                    float((signal or {}).get("score", 0.0) or 0.0),
                )
                existing["selected"] = bool(existing.get("selected", False)) or bool(
                    (signal or {}).get("selected", False)
                )
                existing["strong_current"] = bool(existing.get("strong_current", False)) or bool(
                    (signal or {}).get("strong_current", False)
                )
        self._latest_local_assignment_signals = out
        return {
            "merged_children": int(merged_children),
            "merged_parent_signals": int(merged_parent_signals),
        }

    def _replace_node_id_in_cabinet_aggregator(self, old_id: str, new_id: str) -> dict:
        agg = getattr(self, "cabinet_aggregator", None)
        if agg is None:
            return {"member_to_cabinet": 0, "member_evidence": 0, "groups": 0, "pair_keys": 0}
        member_to_cabinet_updates = 0
        member_evidence_updates = 0
        group_updates = 0
        pair_key_updates = 0
        old_gid = agg.member_to_cabinet.pop(old_id, None)
        if old_gid is not None:
            existing_gid = agg.member_to_cabinet.get(new_id)
            if existing_gid is None:
                agg.member_to_cabinet[new_id] = old_gid
            elif existing_gid != old_gid:
                try:
                    agg._merge_groups(existing_gid, old_gid)
                except Exception:
                    pass
            member_to_cabinet_updates += 1
        old_ev = agg.member_evidence.pop(old_id, None)
        if old_ev is not None:
            existing_ev = agg.member_evidence.get(new_id)
            if existing_ev is None:
                old_ev.node_id = new_id
                agg.member_evidence[new_id] = old_ev
            else:
                existing_ev.seeded_by_box = bool(existing_ev.seeded_by_box) or bool(old_ev.seeded_by_box)
                existing_ev.support_frames = sorted(set(existing_ev.support_frames).union(old_ev.support_frames))
                existing_ev.last_seen_frame = max(
                    int(existing_ev.last_seen_frame or -1),
                    int(old_ev.last_seen_frame or -1),
                )
                existing_ev.last_overlap_support = max(
                    float(existing_ev.last_overlap_support or 0.0),
                    float(old_ev.last_overlap_support or 0.0),
                )
                existing_ev.last_touch_support = max(
                    float(existing_ev.last_touch_support or 0.0),
                    float(old_ev.last_touch_support or 0.0),
                )
                existing_ev.last_view_score = max(
                    float(existing_ev.last_view_score or 0.0),
                    float(old_ev.last_view_score or 0.0),
                )
            member_evidence_updates += 1

        def _rewrite_pair_key_set(keys: set[str]) -> set[str]:
            nonlocal pair_key_updates
            out: set[str] = set()
            for key in keys:
                try:
                    a, b = str(key).split("|", 1)
                except ValueError:
                    out.add(str(key))
                    continue
                if a == old_id:
                    a = new_id
                if b == old_id:
                    b = new_id
                if a == b:
                    pair_key_updates += 1
                    continue
                out.add("|".join(sorted([a, b])))
            return out

        for group in agg.cabinet_tracks.values():
            changed = False
            if old_id in group.member_node_ids:
                group.member_node_ids.discard(old_id)
                group.member_node_ids.add(new_id)
                changed = True
            if old_id in group.seeded_member_ids:
                group.seeded_member_ids.discard(old_id)
                group.seeded_member_ids.add(new_id)
                changed = True
            new_pending = _rewrite_pair_key_set(set(group.pending_plane_pairs))
            new_validated = _rewrite_pair_key_set(set(group.validated_plane_pairs))
            new_failed = _rewrite_pair_key_set(set(group.failed_plane_pairs))
            if new_pending != group.pending_plane_pairs:
                group.pending_plane_pairs = new_pending
                changed = True
            if new_validated != group.validated_plane_pairs:
                group.validated_plane_pairs = new_validated
                changed = True
            if new_failed != group.failed_plane_pairs:
                group.failed_plane_pairs = new_failed
                changed = True
            if changed:
                group_updates += 1
        return {
            "member_to_cabinet": int(member_to_cabinet_updates),
            "member_evidence": int(member_evidence_updates),
            "groups": int(group_updates),
            "pair_keys": int(pair_key_updates),
        }

    def _materialize_rewritten_parent_edges(
        self,
        old_parent_id: str,
        new_parent_id: str,
        child_ids: list[str],
        *,
        reason: str,
    ) -> dict:
        created_edges: list[dict] = []
        skipped: list[dict] = []
        parent_track = self.node_tracks.get(new_parent_id)
        if parent_track is None:
            return {"created": [], "skipped": [{"reason": "missing_new_parent"}]}
        for child_id in sorted({str(cid) for cid in (child_ids or []) if cid}):
            if child_id == new_parent_id:
                skipped.append({"child_id": child_id, "reason": "self_loop"})
                continue
            child_track = self.node_tracks.get(child_id)
            posterior = self.local_posteriors.get(child_id)
            if child_track is None or posterior is None:
                skipped.append({"child_id": child_id, "reason": "missing_child_or_posterior"})
                continue
            edge_type = self._edge_type_for_roles(parent_track.role, child_track.role)
            # Only synthesize missing O-C carrier edges; O-U edges are more
            # sensitive to carrier-parent arbitration and are left to the
            # normal commit pipeline.
            if edge_type != "O-C":
                skipped.append({"child_id": child_id, "reason": "edge_type_not_materialized", "edge_type": edge_type})
                continue
            if self._current_local_graph_edge(child_id) is not None:
                skipped.append({"child_id": child_id, "reason": "existing_graph_edge"})
                continue
            if (
                getattr(posterior, "stable_parent_id", None) != new_parent_id
                and getattr(posterior, "top1_parent_id", None) != new_parent_id
            ):
                skipped.append({"child_id": child_id, "reason": "new_parent_not_top1_or_stable"})
                continue
            support_frames = list(posterior.candidate_support_frames.get(new_parent_id, []) or [])
            support_count = int(len(support_frames))
            evidence_score = float(posterior.candidate_parent_scores.get(new_parent_id, 0.0) or 0.0)
            arbitration = self._should_accept_local_parent_commit(
                new_parent_id,
                child_id,
                edge_type,
                posterior,
                {
                    "source": "duplicate_ref_materialize",
                    "support_count": support_count,
                    "recent_support_count": posterior.recent_support_count(new_parent_id, window=3),
                    "margin": float(getattr(posterior, "margin", 0.0) or 0.0),
                    "latest_margin": float(getattr(posterior, "latest_evidence_margin", 0.0) or 0.0),
                    "evidence_score": evidence_score,
                },
            )
            if not arbitration.get("accept", False):
                skipped.append({
                    "child_id": child_id,
                    "reason": "parent_arbitration_blocked",
                    "arbitration": arbitration.get("record", {}),
                })
                continue
            self.graph.upsert_local_edge(
                new_parent_id,
                child_id,
                edge_type,
                relation_text=self._relation_text_for_instance_edge(new_parent_id, child_id, edge_type),
                committed_kf=int(getattr(posterior, "stable_since_kf", -1) or -1),
                support_count=support_count,
                evidence_score=evidence_score,
                status="tentative",
                first_seen_frame=int(support_frames[0]) if support_frames else -1,
                last_seen_frame=int(support_frames[-1]) if support_frames else -1,
                last_update_source="duplicate_ref_materialize",
                margin=float(getattr(posterior, "margin", 0.0) or 0.0),
                latest_margin=float(getattr(posterior, "latest_evidence_margin", 0.0) or 0.0),
                recent_support_count=posterior.recent_support_count(new_parent_id, window=3),
                switch_count=posterior.recent_switch_count(window=4),
                retention_policy="ttl",
            )
            created_edges.append(
                {
                    "parent_id": new_parent_id,
                    "child_id": child_id,
                    "edge_type": edge_type,
                    "support_count": support_count,
                    "evidence_score": evidence_score,
                    "reason": str(reason),
                }
            )
        return {"created": created_edges, "skipped": skipped}

    def _rehydrate_missing_oc_edges_from_posteriors(self, frame_idx: int) -> dict:
        created_edges: list[dict] = []
        skipped: list[dict] = []
        for child_id, posterior in self.local_posteriors.items():
            child_track = self.node_tracks.get(child_id)
            if child_track is None or getattr(child_track, "role", "") != "C":
                continue
            if self._current_local_graph_edge(child_id) is not None:
                continue
            active_parent = self._select_active_local_parent_candidate(child_id, posterior)
            parent_id = (
                str(active_parent["parent_id"])
                if active_parent is not None
                else (getattr(posterior, "stable_parent_id", None) or getattr(posterior, "top1_parent_id", None))
            )
            if parent_id is None:
                continue
            parent_track = self.node_tracks.get(parent_id)
            if parent_track is None:
                skipped.append({"child_id": child_id, "reason": "missing_parent_track"})
                continue
            edge_type = self._edge_type_for_roles(parent_track.role, child_track.role)
            if edge_type != "O-C":
                continue
            support_count = int(active_parent["support_count"]) if active_parent is not None else posterior.support_count(parent_id)
            recent_support_count = int(active_parent["recent_support_count"]) if active_parent is not None else posterior.recent_support_count(parent_id, window=3)
            switch_count = int(active_parent["switch_count"]) if active_parent is not None else posterior.recent_switch_count(window=4)
            if support_count < 2 or recent_support_count < 2 or switch_count > 1:
                skipped.append(
                    {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "reason": "insufficient_support",
                        "support_count": int(support_count),
                        "recent_support_count": int(recent_support_count),
                        "switch_count": int(switch_count),
                    }
                )
                continue
            if not (
                active_parent is not None
                or float(getattr(posterior, "margin", 0.0) or 0.0) >= float(self.tentative_margin_local)
                or getattr(posterior, "top2_parent_id", None) is None
            ):
                skipped.append(
                    {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "reason": "margin_below_tentative_threshold",
                        "margin": float(getattr(posterior, "margin", 0.0) or 0.0),
                    }
                )
                continue
            arbitration = self._should_accept_local_parent_commit(
                parent_id,
                child_id,
                edge_type,
                posterior,
                {
                        "source": "posterior_rehydrate",
                        "support_count": int(support_count),
                        "recent_support_count": int(recent_support_count),
                        "margin": float((active_parent or {}).get("margin", getattr(posterior, "margin", 0.0)) or 0.0),
                        "latest_margin": float((active_parent or {}).get("latest_margin", getattr(posterior, "latest_evidence_margin", 0.0)) or 0.0),
                        "evidence_score": float((active_parent or {}).get("evidence_score", posterior.candidate_parent_scores.get(parent_id, 0.0)) or 0.0),
                        "active_parent_reason": str((active_parent or {}).get("active_parent_reason") or ""),
                    },
                )
            self._record_local_parent_arbitration(frame_idx, arbitration)
            if not arbitration.get("accept", False):
                skipped.append(
                    {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "reason": "parent_arbitration_blocked",
                        "arbitration": arbitration.get("record", {}),
                    }
                )
                continue
            self._apply_parent_arbitration_prunes(arbitration, frame_idx=frame_idx)
            support_frames = list(posterior.candidate_support_frames.get(parent_id, []) or [])
            self.graph.upsert_local_edge(
                parent_id,
                child_id,
                edge_type,
                relation_text=self._relation_text_for_instance_edge(parent_id, child_id, edge_type),
                committed_kf=-1,
                support_count=int(support_count),
                evidence_score=float((active_parent or {}).get("evidence_score", posterior.candidate_parent_scores.get(parent_id, 0.0)) or 0.0),
                status="tentative",
                first_seen_frame=int(support_frames[0]) if support_frames else int(frame_idx),
                last_seen_frame=int(support_frames[-1]) if support_frames else int(frame_idx),
                last_update_source="posterior_rehydrate",
                margin=float((active_parent or {}).get("margin", getattr(posterior, "margin", 0.0)) or 0.0),
                latest_margin=float((active_parent or {}).get("latest_margin", getattr(posterior, "latest_evidence_margin", 0.0)) or 0.0),
                recent_support_count=int(recent_support_count),
                switch_count=int(switch_count),
                retention_policy="ttl",
            )
            created_edges.append(
                {
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "support_count": int(support_count),
                    "recent_support_count": int(recent_support_count),
                }
            )
        bucket = self.frame_assoc_debug.setdefault(int(frame_idx), {"frame_idx": int(frame_idx)})
        bucket["posterior_edge_rehydrate"] = {
            "created_edges": created_edges,
            "skipped": skipped,
        }
        return {"created": created_edges, "skipped": skipped}

    def _replace_node_id_in_hierarchy(self, old_id: str, new_id: str) -> None:
        h = self.graph.hierarchy
        h.parent_u = {
            (new_id if k == old_id else k): (new_id if v == old_id else v)
            for k, v in h.parent_u.items()
        }
        h.parent_c = {
            (new_id if k == old_id else k): (new_id if v == old_id else v)
            for k, v in h.parent_c.items()
        }
        h.chains_uco = [
            tuple((new_id if x == old_id else x) for x in chain) for chain in h.chains_uco
        ]
        h.direct_uo = [
            tuple((new_id if x == old_id else x) for x in pair) for pair in h.direct_uo
        ]

    def _merge_track_stats(
        self,
        survivor,
        loser,
        *,
        transfer_observed_frames: bool = True,
    ) -> dict:
        """Cheap stat merge.  Survivor's geometry / anchor state is kept.

        ``transfer_observed_frames`` controls whether the loser's
        ``observed_frames`` are migrated into the survivor (v3 I).  Low
        confidence merges should set this to ``False`` so that the
        survivor cannot inherit visibility for frames it never actually
        saw — that was the root cause of the U0084-C0038 contamination.

        Returns a small debug dict with the before/after observed_frames
        snapshots for audit logging.
        """
        survivor_observed_frames_before = list(getattr(survivor, "observed_frames", []) or [])
        loser_observed_frames = list(getattr(loser, "observed_frames", []) or [])
        survivor.obs_count = int(survivor.obs_count) + int(getattr(loser, "obs_count", 0))
        survivor.visible_count = int(survivor.visible_count) + int(getattr(loser, "visible_count", 0))
        for label, count in (loser.label_counts or {}).items():
            survivor.label_counts[label] = int(survivor.label_counts.get(label, 0)) + int(count)
        for sub, score in (getattr(loser, "semantic_subtype_scores", {}) or {}).items():
            survivor.semantic_subtype_scores[sub] = (
                float(survivor.semantic_subtype_scores.get(sub, 0.0)) + float(score)
            )
        try:
            survivor.label = survivor.dominant_label() or survivor.label
        except Exception:
            pass
        try:
            if hasattr(survivor, "dominant_semantic_subtype"):
                survivor.semantic_subtype = (
                    survivor.dominant_semantic_subtype() or survivor.semantic_subtype
                )
        except Exception:
            pass
        if transfer_observed_frames:
            merged_frames = list(survivor_observed_frames_before)
            merged_frames_set = set(merged_frames)
            transferred: list[int] = []
            for f in loser_observed_frames:
                if f not in merged_frames_set:
                    merged_frames.append(int(f))
                    merged_frames_set.add(int(f))
                    transferred.append(int(f))
            merged_frames.sort()
            cap = int(getattr(survivor, "observed_frame_cap", 256) or 256)
            if len(merged_frames) > cap:
                merged_frames = merged_frames[-cap:]
            survivor.observed_frames = merged_frames
            transfer_mode = "transferred"
        else:
            # Skip migration: survivor keeps its own observed_frames.
            transferred = []
            transfer_mode = "blocked_low_confidence"
        if getattr(loser, "first_seen_frame", None) is not None:
            if (
                getattr(survivor, "first_seen_frame", None) is None
                or int(loser.first_seen_frame) < int(survivor.first_seen_frame)
            ):
                survivor.first_seen_frame = int(loser.first_seen_frame)
        if getattr(loser, "last_seen_frame", None) is not None:
            if (
                getattr(survivor, "last_seen_frame", None) is None
                or int(loser.last_seen_frame) > int(survivor.last_seen_frame)
            ):
                survivor.last_seen_frame = int(loser.last_seen_frame)
        if getattr(loser, "last_seen_kf", None) is not None:
            if (
                getattr(survivor, "last_seen_kf", None) is None
                or int(loser.last_seen_kf) > int(survivor.last_seen_kf)
            ):
                survivor.last_seen_kf = int(loser.last_seen_kf)
        if float(getattr(loser, "best_view_score", 0.0) or 0.0) > float(getattr(survivor, "best_view_score", 0.0) or 0.0):
            survivor.best_view_score = float(loser.best_view_score)
            survivor.best_view_frame = loser.best_view_frame
            survivor.best_view_crop_meta = loser.best_view_crop_meta
        return {
            "survivor_observed_frames_before": survivor_observed_frames_before,
            "loser_observed_frames": loser_observed_frames,
            "survivor_observed_frames_after": list(getattr(survivor, "observed_frames", []) or []),
            "transferred_observed_frames": transferred,
            "observed_frame_transfer_mode": transfer_mode,
        }

    def _prune_node_track(self, node_id: str, *, reason: str) -> dict:
        if node_id not in self.node_tracks:
            return {"ok": False, "reason": "missing_track"}
        if self._has_any_committed_edge(node_id):
            return {"ok": False, "reason": "has_committed_edge"}
        new_local: dict = {}
        dropped_local = 0
        for key, edge in self.graph.local_edges.items():
            if edge.src_node_id == node_id or edge.dst_node_id == node_id:
                dropped_local += 1
                continue
            new_local[key] = edge
        self.graph.local_edges = new_local
        new_remote: dict = {}
        dropped_remote = 0
        for key, edge in self.graph.remote_edges.items():
            if edge.src_node_id == node_id or edge.dst_node_id == node_id:
                dropped_remote += 1
                continue
            new_remote[key] = edge
        self.graph.remote_edges = new_remote
        self.local_posteriors.pop(node_id, None)
        for posterior in self.local_posteriors.values():
            posterior.candidate_parent_scores.pop(node_id, None)
            posterior.candidate_support_frames.pop(node_id, None)
            posterior.preferred_parent_ids.discard(node_id)
            posterior.fallback_parent_ids.discard(node_id)
            if posterior.top1_parent_id == node_id:
                posterior.top1_parent_id = None
            if posterior.top2_parent_id == node_id:
                posterior.top2_parent_id = None
            if posterior.stable_parent_id == node_id:
                posterior.stable_parent_id = None
        for post in self.remote_posteriors.values():
            post.candidate_pairs = {
                p: s for p, s in post.candidate_pairs.items() if node_id not in p
            }
            post.support_frames = {
                p: s for p, s in post.support_frames.items() if node_id not in p
            }
            if post.top1_pair and node_id in post.top1_pair:
                post.top1_pair = None
            if post.top2_pair and node_id in post.top2_pair:
                post.top2_pair = None
            if post.stable_pair and node_id in post.stable_pair:
                post.stable_pair = None
        h = self.graph.hierarchy
        h.parent_u = {k: v for k, v in h.parent_u.items() if k != node_id and v != node_id}
        h.parent_c = {k: v for k, v in h.parent_c.items() if k != node_id and v != node_id}
        h.chains_uco = [c for c in h.chains_uco if node_id not in c]
        h.direct_uo = [p for p in h.direct_uo if node_id not in p]
        self.node_tracks.pop(node_id, None)
        return {
            "ok": True,
            "reason": reason,
            "node_id": node_id,
            "dropped_local_edges": dropped_local,
            "dropped_remote_edges": dropped_remote,
        }

    def _is_orphan_u_node(self, node_id: str) -> tuple[bool, dict]:
        """An "orphan-U" is a U-role track with no committed local-parent
        edge, no committed remote edge, no aggregate-cabinet membership and
        no strong tentative remote edge.  Used to gate competition-loser
        pruning so we never delete tracks that are still anchored.
        """
        track = self.node_tracks.get(node_id)
        info = {
            "role": getattr(track, "role", None) if track else None,
            "has_committed_edge": False,
            "has_strong_remote": False,
            "has_aggregate_parent": False,
            "has_any_tentative_local": False,
        }
        if track is None or getattr(track, "role", "") != "U":
            return False, info
        if self._has_any_committed_edge(node_id):
            info["has_committed_edge"] = True
            return False, info
        # Aggregate-cabinet membership counts as anchored.
        gid = self.cabinet_aggregator.member_to_cabinet.get(node_id)
        if gid is not None:
            info["has_aggregate_parent"] = True
            return False, info
        if self._has_strong_remote_edge(node_id):
            info["has_strong_remote"] = True
            return False, info
        info["has_any_tentative_local"] = self._has_any_tentative_local_edge(node_id)
        return True, info

    def _should_prune_competition_loser_u(
        self, node_id: str, frame_idx: int
    ) -> tuple[bool, dict]:
        debug: dict = {"node_id": node_id, "frame_idx": int(frame_idx)}
        is_orphan, orphan_info = self._is_orphan_u_node(node_id)
        debug["orphan_info"] = orphan_info
        if not is_orphan:
            debug["reject_reason"] = "not_orphan_u"
            return False, debug
        rec = self.node_parent_competition_losses.get(node_id) or {}
        loss_count = int(rec.get("loss_count", 0))
        last_loss_frame = rec.get("last_loss_frame")
        margin = float(rec.get("last_margin", 0.0) or 0.0)
        debug["loss_count"] = loss_count
        debug["last_loss_frame"] = last_loss_frame
        debug["last_margin"] = margin
        if loss_count < int(self.orphan_u_prune_min_loss_count):
            debug["reject_reason"] = "loss_count_too_small"
            return False, debug
        # Stale: last_seen_frame must be old enough.
        stale = int(self._track_stale_frames(node_id))
        debug["stale_frames"] = stale
        if stale < int(self.orphan_u_prune_stale_frames):
            debug["reject_reason"] = "not_stale_enough"
            return False, debug
        # Margin: best winner should be clearly better than loser.
        if margin < float(self.orphan_u_prune_min_margin):
            debug["reject_reason"] = "margin_too_small"
            return False, debug
        winner_id = rec.get("last_winner_id")
        debug["last_winner_id"] = winner_id
        # Quality gap: winner must outscore loser by >= quality_gap.
        loser_q, _ = self._node_quality_score(node_id)
        winner_q = 0.0
        if winner_id and winner_id in self.node_tracks:
            winner_q, _ = self._node_quality_score(winner_id)
        debug["loser_quality"] = float(loser_q)
        debug["winner_quality"] = float(winner_q)
        if (winner_q - loser_q) < float(self.orphan_u_prune_quality_gap):
            debug["reject_reason"] = "quality_gap_too_small"
            return False, debug
        # Optionally allow only when no remote edge at all.
        if (
            self.orphan_u_prune_allow_if_no_remote
            and self._has_any_remote_edge(node_id)
        ):
            debug["reject_reason"] = "has_remote_edge"
            return False, debug
        debug["accept"] = True
        return True, debug

    def _prune_orphan_competition_loser_u_nodes(
        self, frame_idx: int, *, dry_run: bool = False
    ) -> list[dict]:
        if not self.orphan_u_prune_enable:
            return []
        events: list[dict] = []
        # Snapshot keys first because pruning mutates the registry.
        for node_id in list(self.node_parent_competition_losses.keys()):
            if node_id not in self.node_tracks:
                continue
            ok, dbg = self._should_prune_competition_loser_u(node_id, frame_idx)
            if not ok:
                continue
            entry = {
                "frame_idx": int(frame_idx),
                "node_id": node_id,
                "winner_id": dbg.get("last_winner_id"),
                "loss_count": dbg.get("loss_count"),
                "last_margin": dbg.get("last_margin"),
                "stale_frames": dbg.get("stale_frames"),
                "loser_quality": dbg.get("loser_quality"),
                "winner_quality": dbg.get("winner_quality"),
                "dry_run": bool(dry_run),
                "debug": dbg,
            }
            if not dry_run:
                entry["prune_debug"] = self._prune_node_track(
                    node_id, reason="orphan_u_competition_loser"
                )
                # Drop the loss record after pruning to keep state tidy.
                self.node_parent_competition_losses.pop(node_id, None)
            events.append(entry)
            if len(self.orphan_u_prune_events) < 4096:
                self.orphan_u_prune_events.append(entry)
        return events

    @staticmethod
    def _assignment_competition_pair_key(a: str, b: str) -> tuple[str, str]:
        a, b = str(a), str(b)
        return (a, b) if a <= b else (b, a)

    def _record_assignment_competition_event(
        self,
        *,
        frame_idx: int,
        type: str,
        node_a: str,
        node_b: str,
        shared_parent_id: Optional[str] = None,
        shared_child_id: Optional[str] = None,
        source: str,
        strength: float = 1.0,
    ) -> None:
        if not node_a or not node_b or str(node_a) == str(node_b):
            return
        if node_a not in self.node_tracks or node_b not in self.node_tracks:
            return
        key = self._assignment_competition_pair_key(node_a, node_b)
        ev = {
            "frame_idx": int(frame_idx),
            "type": str(type),
            "node_a": key[0],
            "node_b": key[1],
            "shared_parent_id": shared_parent_id,
            "shared_child_id": shared_child_id,
            "source": str(source),
            "strength": float(strength),
        }
        if len(self.node_assignment_competition_events) < 4096:
            self.node_assignment_competition_events.append(ev)
        slot = self.node_assignment_competition_pair_counts.setdefault(
            key,
            {
                "competition_count": 0,
                "parent_competition_count": 0,
                "child_competition_count": 0,
                "competition_sources": [],
                "shared_parent_ids": [],
                "shared_child_ids": [],
                "local_posterior_refs": [],
                "arbitration_refs": [],
                "last_frame_idx": -1,
            },
        )
        slot["competition_count"] = int(slot.get("competition_count", 0)) + 1
        if str(type) == "parent_competition":
            slot["parent_competition_count"] = (
                int(slot.get("parent_competition_count", 0)) + 1
            )
        elif str(type) == "child_competition":
            slot["child_competition_count"] = (
                int(slot.get("child_competition_count", 0)) + 1
            )
        if source not in slot["competition_sources"]:
            slot["competition_sources"].append(str(source))
        if shared_parent_id and shared_parent_id not in slot["shared_parent_ids"]:
            slot["shared_parent_ids"].append(str(shared_parent_id))
        if shared_child_id and shared_child_id not in slot["shared_child_ids"]:
            slot["shared_child_ids"].append(str(shared_child_id))
        if str(source) == "local_posterior":
            ref = (int(frame_idx), shared_child_id, shared_parent_id)
            if ref not in slot["local_posterior_refs"]:
                slot["local_posterior_refs"].append(ref)
        if str(source) in (
            "unstable_sibling_arbitration",
            "sibling_duplicate_risk",
        ):
            ref = (int(frame_idx), shared_parent_id)
            if ref not in slot["arbitration_refs"]:
                slot["arbitration_refs"].append(ref)
        slot["last_frame_idx"] = int(frame_idx)

    def _scan_local_posteriors_for_competition(self, frame_idx: int) -> int:
        """Per-frame scan: any pair of parent ids that both appear in the
        same child posterior's ``candidate_parent_scores`` are recorded
        as a child_competition event (they compete to be the parent of
        the same child).  Returns the number of pairs recorded.
        """
        recorded = 0
        for child_id, posterior in self.local_posteriors.items():
            scores = getattr(posterior, "candidate_parent_scores", None) or {}
            ids = [pid for pid in scores.keys() if pid in self.node_tracks]
            if len(ids) < 2:
                continue
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pa, pb = ids[i], ids[j]
                    if pa == pb:
                        continue
                    self._record_assignment_competition_event(
                        frame_idx=int(frame_idx),
                        type="child_competition",
                        node_a=pa,
                        node_b=pb,
                        shared_child_id=str(child_id),
                        source="local_posterior",
                        strength=float(
                            min(
                                float(scores.get(pa, 0.0)),
                                float(scores.get(pb, 0.0)),
                            )
                        ),
                    )
                    recorded += 1
        return recorded

    def node_assignment_competition_count(self, node_a: str, node_b: str) -> int:
        if not node_a or not node_b or node_a == node_b:
            return 0
        slot = self.node_assignment_competition_pair_counts.get(
            self._assignment_competition_pair_key(node_a, node_b)
        )
        if not slot:
            return 0
        return int(slot.get("competition_count", 0))

    def nodes_have_assignment_competition(
        self, node_a: str, node_b: str
    ) -> tuple[bool, dict]:
        slot = self.node_assignment_competition_pair_counts.get(
            self._assignment_competition_pair_key(node_a, node_b)
        )
        if not slot:
            return False, {
                "competition_count": 0,
                "parent_competition_count": 0,
                "child_competition_count": 0,
                "competition_sources": [],
                "shared_parent_ids": [],
                "shared_child_ids": [],
                "local_posterior_refs": [],
                "arbitration_refs": [],
            }
        debug = {
            "competition_count": int(slot.get("competition_count", 0)),
            "parent_competition_count": int(slot.get("parent_competition_count", 0)),
            "child_competition_count": int(slot.get("child_competition_count", 0)),
            "competition_sources": list(slot.get("competition_sources", [])),
            "shared_parent_ids": list(slot.get("shared_parent_ids", [])),
            "shared_child_ids": list(slot.get("shared_child_ids", [])),
            "local_posterior_refs": list(slot.get("local_posterior_refs", [])),
            "arbitration_refs": list(slot.get("arbitration_refs", [])),
        }
        return debug["competition_count"] >= 1, debug

    def _redirect_competition_ledger(self, old_id: str, new_id: str) -> dict:
        """Rewrite competition ledger so refs to ``old_id`` point at
        ``new_id``.  Pair counts and event entries are merged into
        whatever (sorted) key the surviving pair generates.  Self-pairs
        (old_id <-> new_id and resulting (X, X)) are dropped.
        """
        merged_pairs = 0
        dropped_self = 0
        # Update events.
        for ev in self.node_assignment_competition_events:
            if ev.get("node_a") == old_id:
                ev["node_a"] = new_id
            if ev.get("node_b") == old_id:
                ev["node_b"] = new_id
        # Drop self-pair events.
        self.node_assignment_competition_events = [
            ev
            for ev in self.node_assignment_competition_events
            if ev.get("node_a") != ev.get("node_b")
        ]
        # Update pair counts.
        new_counts: Dict[tuple[str, str], dict] = {}
        for key, slot in self.node_assignment_competition_pair_counts.items():
            a, b = key
            if a == old_id:
                a = new_id
            if b == old_id:
                b = new_id
            if a == b:
                dropped_self += 1
                continue
            new_key = self._assignment_competition_pair_key(a, b)
            existing = new_counts.get(new_key)
            if existing is None:
                new_counts[new_key] = dict(slot)
                continue
            # Merge counters and dedup lists.
            existing["competition_count"] = int(
                existing.get("competition_count", 0)
            ) + int(slot.get("competition_count", 0))
            existing["parent_competition_count"] = int(
                existing.get("parent_competition_count", 0)
            ) + int(slot.get("parent_competition_count", 0))
            existing["child_competition_count"] = int(
                existing.get("child_competition_count", 0)
            ) + int(slot.get("child_competition_count", 0))
            for fld in (
                "competition_sources",
                "shared_parent_ids",
                "shared_child_ids",
                "local_posterior_refs",
                "arbitration_refs",
            ):
                lst = existing.setdefault(fld, [])
                for item in slot.get(fld, []) or []:
                    if item not in lst:
                        lst.append(item)
            existing["last_frame_idx"] = max(
                int(existing.get("last_frame_idx", -1)),
                int(slot.get("last_frame_idx", -1)),
            )
            merged_pairs += 1
        self.node_assignment_competition_pair_counts = new_counts
        return {
            "merged_pairs": int(merged_pairs),
            "dropped_self": int(dropped_self),
        }

    def _replace_node_references(
        self, old_id: str, new_id: str, *, reason: str
    ) -> dict:
        """Unified endpoint replacement covering: local edges, remote
        edges, posteriors, hierarchy, competition ledger, parent-
        competition-loss history, and node_tracks/local_posteriors
        cleanup.  Idempotent and safe to call after merge or absorb-
        prune.  Returns merged debug.
        """
        if old_id == new_id:
            return {"ok": False, "reason": "same_id", "old_id": old_id}
        local_dbg = self._replace_node_id_in_local_edges(old_id, new_id)
        remote_dbg = self._replace_node_id_in_remote_edges(old_id, new_id)
        post_dbg = self._replace_node_id_in_posteriors(old_id, new_id)
        self._replace_node_id_in_hierarchy(old_id, new_id)
        comp_dbg = self._redirect_competition_ledger(old_id, new_id)
        claims_dbg = self._replace_node_id_in_latest_current_parent_claims(old_id, new_id)
        assign_dbg = self._replace_node_id_in_latest_local_assignment_signals(old_id, new_id)
        cabinet_dbg = self._replace_node_id_in_cabinet_aggregator(old_id, new_id)
        materialized_dbg = self._materialize_rewritten_parent_edges(
            old_id,
            new_id,
            list(post_dbg.get("rewritten_parent_child_ids", []) or []),
            reason=str(reason),
        )
        # Parent-competition loss history rewrite (used by orphan-U pass).
        if isinstance(getattr(self, "node_parent_competition_losses", None), dict):
            old_losses = self.node_parent_competition_losses.pop(old_id, None)
            if old_losses is not None:
                merged = self.node_parent_competition_losses.setdefault(new_id, {})
                # Best-effort merge: numeric counters add, lists union.
                for k, v in (old_losses or {}).items():
                    if isinstance(v, (int, float)) and isinstance(
                        merged.get(k), (int, float)
                    ):
                        merged[k] = type(v)(merged[k] + v)
                    elif isinstance(v, list):
                        cur = merged.setdefault(k, [])
                        for item in v:
                            if item not in cur:
                                cur.append(item)
                    else:
                        merged.setdefault(k, v)
        # Track + posterior cleanup (idempotent).
        self.node_tracks.pop(old_id, None)
        self.local_posteriors.pop(old_id, None)
        return {
            "ok": True,
            "reason": str(reason),
            "old_id": old_id,
            "new_id": new_id,
            "local_edges": local_dbg,
            "remote_edges": remote_dbg,
            "posteriors": post_dbg,
            "competition_ledger": comp_dbg,
            "current_parent_claims": claims_dbg,
            "latest_local_assignment_signals": assign_dbg,
            "cabinet_aggregator": cabinet_dbg,
            "materialized_local_edges": materialized_dbg,
        }

    def _assert_no_dangling_node_refs(self) -> tuple[bool, dict]:
        node_ids = set(self.node_tracks.keys())
        problems: list[dict] = []
        for key, edge in list(self.graph.local_edges.items()):
            for endpoint in (edge.src_node_id, edge.dst_node_id):
                if endpoint not in node_ids:
                    problems.append(
                        {"kind": "local_edge", "key": key, "missing": endpoint}
                    )
        for key, edge in list(self.graph.remote_edges.items()):
            for endpoint in (edge.src_node_id, edge.dst_node_id):
                if endpoint not in node_ids:
                    problems.append(
                        {"kind": "remote_edge", "key": key, "missing": endpoint}
                    )
        for child_id, posterior in list(self.local_posteriors.items()):
            if child_id not in node_ids:
                problems.append({"kind": "posterior_child", "missing": child_id})
            for pid in list(
                getattr(posterior, "candidate_parent_scores", {}).keys()
            ):
                if pid not in node_ids:
                    problems.append(
                        {
                            "kind": "posterior_candidate_parent",
                            "child": child_id,
                            "missing": pid,
                        }
                    )
        return (len(problems) == 0), {"problems": problems}

    def _competition_duplicate_geometry_check(
        self, node_a: str, node_b: str
    ) -> tuple[bool, str, dict]:
        """Return ``(close, action_kind, debug)``.

        ``action_kind`` is ``"iou_merge"`` if 3D IoU > threshold,
        ``"distance_prune"`` if centroid_distance < min(half_diag_a,
        half_diag_b), otherwise ``""`` and ``close=False``.
        """
        ta = self.node_tracks.get(node_a)
        tb = self.node_tracks.get(node_b)
        if ta is None or tb is None:
            return False, "", {"reason": "missing_track"}
        ba = self._bbox_world_minmax(ta)
        bb = self._bbox_world_minmax(tb)
        if ba is None or bb is None:
            return False, "", {"reason": "missing_geometry"}
        iou, overlap = self._bbox_iou_3d(ta, tb)
        cd = self._centroid_distance(ta, tb)
        amin, amax = ba
        bmin, bmax = bb
        diag_a = (
            (amax[0] - amin[0]) ** 2
            + (amax[1] - amin[1]) ** 2
            + (amax[2] - amin[2]) ** 2
        ) ** 0.5
        diag_b = (
            (bmax[0] - bmin[0]) ** 2
            + (bmax[1] - bmin[1]) ** 2
            + (bmax[2] - bmin[2]) ** 2
        ) ** 0.5
        half_diag_a = 0.5 * diag_a
        half_diag_b = 0.5 * diag_b
        centroid_threshold = min(half_diag_a, half_diag_b)
        debug = {
            "bbox_iou": float(iou),
            "bbox_overlap": float(overlap),
            "centroid_distance": float(cd),
            "diag_a": float(diag_a),
            "diag_b": float(diag_b),
            "half_diag_a": float(half_diag_a),
            "half_diag_b": float(half_diag_b),
            "centroid_threshold": float(centroid_threshold),
        }
        if float(iou) > float(self.competition_duplicate_iou_thr):
            debug["match_rule"] = "iou_above_threshold"
            return True, "iou_merge", debug
        if cd < centroid_threshold and centroid_threshold > 0.0:
            debug["match_rule"] = "centroid_within_half_diag"
            return True, "distance_prune", debug
        debug["match_rule"] = "not_close"
        return False, "", debug

    def _choose_competition_duplicate_survivor(
        self, node_a: str, node_b: str
    ) -> tuple[str, str, dict]:
        ta = self.node_tracks.get(node_a)
        tb = self.node_tracks.get(node_b)
        if ta is None and tb is None:
            return node_a, node_b, {"reason": "both_missing"}
        if ta is None:
            return node_b, node_a, {"reason": "a_missing"}
        if tb is None:
            return node_a, node_b, {"reason": "b_missing"}
        qa, comp_a = self._node_quality_score(node_a)
        qb, comp_b = self._node_quality_score(node_b)
        commit_a = self._has_any_committed_edge(node_a)
        commit_b = self._has_any_committed_edge(node_b)
        stable_a = bool(getattr(ta, "stable_geom_ready", False))
        stable_b = bool(getattr(tb, "stable_geom_ready", False))
        obs_a = int(getattr(ta, "obs_count", 0) or 0)
        obs_b = int(getattr(tb, "obs_count", 0) or 0)
        first_a = int(getattr(ta, "first_seen_frame", 1 << 30) or (1 << 30))
        first_b = int(getattr(tb, "first_seen_frame", 1 << 30) or (1 << 30))
        debug = {
            "quality_a": float(qa),
            "quality_b": float(qb),
            "committed_a": bool(commit_a),
            "committed_b": bool(commit_b),
            "stable_a": bool(stable_a),
            "stable_b": bool(stable_b),
            "obs_a": obs_a,
            "obs_b": obs_b,
            "first_a": first_a,
            "first_b": first_b,
        }
        # Decision lattice (per spec F): committed > stable > quality >
        # obs_count > first_seen.
        if commit_a and not commit_b:
            survivor, loser, reason = node_a, node_b, "survivor_committed"
        elif commit_b and not commit_a:
            survivor, loser, reason = node_b, node_a, "survivor_committed"
        elif stable_a and not stable_b:
            survivor, loser, reason = node_a, node_b, "survivor_stable_geom"
        elif stable_b and not stable_a:
            survivor, loser, reason = node_b, node_a, "survivor_stable_geom"
        elif abs(qa - qb) >= 0.5:
            if qa >= qb:
                survivor, loser, reason = node_a, node_b, "survivor_quality_score"
            else:
                survivor, loser, reason = node_b, node_a, "survivor_quality_score"
        elif obs_a != obs_b:
            if obs_a > obs_b:
                survivor, loser, reason = node_a, node_b, "survivor_obs_count"
            else:
                survivor, loser, reason = node_b, node_a, "survivor_obs_count"
        else:
            if first_a <= first_b:
                survivor, loser, reason = node_a, node_b, "survivor_first_seen"
            else:
                survivor, loser, reason = node_b, node_a, "survivor_first_seen"
        debug["reason"] = reason
        return survivor, loser, debug

    def _competition_duplicate_carrier_gate_ok(
        self,
        survivor: str,
        loser: str,
        geom_debug: dict,
    ) -> tuple[bool, dict]:
        """Re-use the v3 carrier safety gate for C-role pairs to prevent
        regression of the U0084-C0038 false-merge bug.  If gate returns
        False, the consolidation refuses the merge.
        """
        ta = self.node_tracks.get(survivor)
        tb = self.node_tracks.get(loser)
        if ta is None or tb is None:
            return True, {"skipped": "missing_track"}
        if str(getattr(ta, "role", "")) != "C" or str(getattr(tb, "role", "")) != "C":
            return True, {"skipped": "role_not_C"}
        qa, _ = self._node_quality_score(survivor)
        qb, _ = self._node_quality_score(loser)
        cand = {
            "a": survivor,
            "b": loser,
            "role": "C",
            "quality_a": float(qa),
            "quality_b": float(qb),
            "bbox_iou": float(geom_debug.get("bbox_iou", 0.0)),
            "bbox_overlap": float(geom_debug.get("bbox_overlap", 0.0)),
            "centroid_distance": float(geom_debug.get("centroid_distance", 0.0)),
        }
        try:
            ok, dbg = self._carrier_merge_safety_gate(survivor, loser, cand)
        except Exception as exc:  # defensive: never break consolidation
            return True, {"skipped": "gate_exception", "error": str(exc)}
        return bool(ok), dict(dbg or {})

    def _apply_competition_duplicate_action(
        self,
        node_a: str,
        node_b: str,
        action_kind: str,
        geom_debug: dict,
        comp_debug: dict,
        *,
        frame_idx: int,
        dry_run: bool,
    ) -> dict:
        survivor, loser, sel_debug = self._choose_competition_duplicate_survivor(
            node_a, node_b
        )
        ta = self.node_tracks.get(survivor)
        tb = self.node_tracks.get(loser)
        role_a = str(getattr(ta, "role", "")) if ta else ""
        role_b = str(getattr(tb, "role", "")) if tb else ""
        label_a = self._track_dominant_label(survivor)
        label_b = self._track_dominant_label(loser)
        qa, _ = self._node_quality_score(survivor)
        qb, _ = self._node_quality_score(loser)
        # Cross-role pairs: only allow IoU>0.25 path; never distance-only.
        if role_a and role_b and role_a != role_b and action_kind == "distance_prune":
            return {
                "frame_idx": int(frame_idx),
                "action": "keep",
                "reason": "cross_role_distance_prune_blocked",
                "node_a": node_a,
                "node_b": node_b,
                "survivor": survivor,
                "loser": loser,
                "role_a": role_a,
                "role_b": role_b,
                "label_a": label_a,
                "label_b": label_b,
                "covisibility_count": 0,
                "bbox_iou": float(geom_debug.get("bbox_iou", 0.0)),
                "centroid_distance": float(geom_debug.get("centroid_distance", 0.0)),
                "centroid_threshold": float(geom_debug.get("centroid_threshold", 0.0)),
                "competition_debug": comp_debug,
                "quality_a": float(qa),
                "quality_b": float(qb),
                "select_debug": sel_debug,
                "dry_run": bool(dry_run),
            }
        # C-role carrier safety gate veto.
        gate_ok, gate_debug = self._competition_duplicate_carrier_gate_ok(
            survivor, loser, geom_debug
        )
        if not gate_ok:
            return {
                "frame_idx": int(frame_idx),
                "action": "keep",
                "reason": "carrier_safety_gate_veto",
                "node_a": node_a,
                "node_b": node_b,
                "survivor": survivor,
                "loser": loser,
                "role_a": role_a,
                "role_b": role_b,
                "label_a": label_a,
                "label_b": label_b,
                "covisibility_count": 0,
                "bbox_iou": float(geom_debug.get("bbox_iou", 0.0)),
                "centroid_distance": float(geom_debug.get("centroid_distance", 0.0)),
                "centroid_threshold": float(geom_debug.get("centroid_threshold", 0.0)),
                "competition_debug": comp_debug,
                "quality_a": float(qa),
                "quality_b": float(qb),
                "select_debug": sel_debug,
                "carrier_gate_debug": gate_debug,
                "dry_run": bool(dry_run),
            }
        if dry_run:
            return {
                "frame_idx": int(frame_idx),
                "action": "merge" if action_kind == "iou_merge" else "absorb_prune",
                "reason": (
                    "competition_duplicate_iou_merge"
                    if action_kind == "iou_merge"
                    else "competition_duplicate_distance_prune_worse"
                ),
                "node_a": node_a,
                "node_b": node_b,
                "survivor": survivor,
                "loser": loser,
                "role_a": role_a,
                "role_b": role_b,
                "label_a": label_a,
                "label_b": label_b,
                "covisibility_count": 0,
                "bbox_iou": float(geom_debug.get("bbox_iou", 0.0)),
                "centroid_distance": float(geom_debug.get("centroid_distance", 0.0)),
                "centroid_threshold": float(geom_debug.get("centroid_threshold", 0.0)),
                "competition_debug": comp_debug,
                "quality_a": float(qa),
                "quality_b": float(qb),
                "select_debug": sel_debug,
                "dry_run": True,
            }
        # Real mutation.
        if action_kind == "iou_merge":
            reason = "competition_duplicate_iou_merge"
            # IoU merge: full stat merge + observed_frames transfer.
            track_stats = self._merge_track_stats(
                ta, tb, transfer_observed_frames=True
            )
            replace_dbg = self._replace_node_references(
                loser, survivor, reason=reason
            )
            replace_dbg["track_stats"] = track_stats
        else:  # distance_prune (absorb_prune worse)
            reason = "competition_duplicate_distance_prune_worse"
            # Absorb-prune: preserve survivor geometry; do still merge
            # observation visibility so future passes see the union.
            track_stats = self._merge_track_stats(
                ta, tb, transfer_observed_frames=True
            )
            replace_dbg = self._replace_node_references(
                loser, survivor, reason=reason
            )
            replace_dbg["track_stats"] = track_stats
        ok_after, dangling = self._assert_no_dangling_node_refs()
        if not ok_after:
            self.competition_duplicate_summary["dangling_ref_errors"] = (
                int(self.competition_duplicate_summary.get("dangling_ref_errors", 0))
                + 1
            )
        return {
            "frame_idx": int(frame_idx),
            "action": "merge" if action_kind == "iou_merge" else "absorb_prune",
            "reason": reason,
            "node_a": node_a,
            "node_b": node_b,
            "survivor": survivor,
            "loser": loser,
            "role_a": role_a,
            "role_b": role_b,
            "label_a": label_a,
            "label_b": label_b,
            "covisibility_count": 0,
            "bbox_iou": float(geom_debug.get("bbox_iou", 0.0)),
            "centroid_distance": float(geom_debug.get("centroid_distance", 0.0)),
            "centroid_threshold": float(geom_debug.get("centroid_threshold", 0.0)),
            "competition_debug": comp_debug,
            "quality_a": float(qa),
            "quality_b": float(qb),
            "select_debug": sel_debug,
            "replace_refs_summary": replace_dbg,
            "dangling_after": (not ok_after),
            "dangling_problems": dangling.get("problems") if not ok_after else [],
            "dry_run": False,
        }

    def consolidate_competition_duplicates(
        self,
        frame_idx: int,
        *,
        dry_run: bool = False,
    ) -> dict:
        """Section H entry point.  Iterate every (sorted) node pair that
        has at least one assignment-competition event and apply the
        three-condition rule (no co-visibility, geometry close,
        competition>=1).  Returns summary + events.
        """
        events: list[dict] = []
        if not self.enable_competition_duplicate_consolidation:
            return {"enabled": False, "frame_idx": int(frame_idx), "events": []}
        # Snapshot keys; mutations during loop will rewrite the ledger.
        pair_keys = list(self.node_assignment_competition_pair_counts.keys())
        for key in pair_keys:
            slot = self.node_assignment_competition_pair_counts.get(key)
            if not slot:
                continue
            node_a, node_b = key
            # After previous merges, key may reference a deleted node.
            if node_a not in self.node_tracks or node_b not in self.node_tracks:
                continue
            comp_count = int(slot.get("competition_count", 0))
            if comp_count < 1:
                self.competition_duplicate_summary["rejected_no_competition"] += 1
                continue
            self.competition_duplicate_summary["candidates"] += 1
            aggregate_a = self._is_aggregate_cabinet_node(node_a)
            aggregate_b = self._is_aggregate_cabinet_node(node_b)
            if aggregate_a or aggregate_b:
                track_a = self.node_tracks.get(node_a)
                track_b = self.node_tracks.get(node_b)
                self.competition_duplicate_summary["rejected_aggregate_cabinet"] += 1
                events.append(
                    {
                        "frame_idx": int(frame_idx),
                        "action": "keep",
                        "reason": "aggregate_cabinet_duplicate_consolidation_blocked",
                        "node_a": node_a,
                        "node_b": node_b,
                        "role_a": str(getattr(track_a, "role", "")) if track_a else "",
                        "role_b": str(getattr(track_b, "role", "")) if track_b else "",
                        "label_a": self._track_dominant_label(node_a),
                        "label_b": self._track_dominant_label(node_b),
                        "origin_a": str(getattr(track_a, "origin", "")) if track_a else "",
                        "origin_b": str(getattr(track_b, "origin", "")) if track_b else "",
                        "aggregate_a": bool(aggregate_a),
                        "aggregate_b": bool(aggregate_b),
                        "competition_count": comp_count,
                        "competition_sources": list(slot.get("competition_sources", [])),
                    }
                )
                continue
            # Condition 1: no co-visibility.
            if self.node_covisibility_count(node_a, node_b) > 0:
                self.competition_duplicate_summary["rejected_covisible"] += 1
                events.append(
                    {
                        "frame_idx": int(frame_idx),
                        "action": "keep",
                        "reason": "covisible_hard_reject",
                        "node_a": node_a,
                        "node_b": node_b,
                        "competition_count": comp_count,
                    }
                )
                continue
            # Condition 2: geometry close.
            close, action_kind, geom_debug = (
                self._competition_duplicate_geometry_check(node_a, node_b)
            )
            if geom_debug.get("reason") == "missing_geometry":
                self.competition_duplicate_summary["rejected_missing_geometry"] += 1
                events.append(
                    {
                        "frame_idx": int(frame_idx),
                        "action": "keep",
                        "reason": "missing_geometry",
                        "node_a": node_a,
                        "node_b": node_b,
                        "competition_count": comp_count,
                    }
                )
                continue
            if not close:
                self.competition_duplicate_summary["rejected_not_close"] += 1
                events.append(
                    {
                        "frame_idx": int(frame_idx),
                        "action": "keep",
                        "reason": "not_spatially_close",
                        "node_a": node_a,
                        "node_b": node_b,
                        "bbox_iou": float(geom_debug.get("bbox_iou", 0.0)),
                        "centroid_distance": float(
                            geom_debug.get("centroid_distance", 0.0)
                        ),
                        "centroid_threshold": float(
                            geom_debug.get("centroid_threshold", 0.0)
                        ),
                        "competition_count": comp_count,
                    }
                )
                continue
            # Condition 3 already satisfied (comp_count >= 1).
            comp_debug = {
                "competition_count": comp_count,
                "parent_competition_count": int(
                    slot.get("parent_competition_count", 0)
                ),
                "child_competition_count": int(
                    slot.get("child_competition_count", 0)
                ),
                "competition_sources": list(slot.get("competition_sources", [])),
                "shared_parent_ids": list(slot.get("shared_parent_ids", [])),
                "shared_child_ids": list(slot.get("shared_child_ids", [])),
            }
            ev = self._apply_competition_duplicate_action(
                node_a,
                node_b,
                action_kind,
                geom_debug,
                comp_debug,
                frame_idx=int(frame_idx),
                dry_run=bool(dry_run),
            )
            events.append(ev)
            if ev.get("action") == "merge":
                self.competition_duplicate_summary["iou_merges"] += 1
            elif ev.get("action") == "absorb_prune":
                self.competition_duplicate_summary["distance_absorb_prunes"] += 1
            elif ev.get("reason") == "carrier_safety_gate_veto":
                self.competition_duplicate_summary["rejected_carrier_gate"] += 1
        # Persist events on bounded ring.
        for ev in events:
            if len(self.competition_duplicate_events) < 4096:
                self.competition_duplicate_events.append(ev)
        return {
            "enabled": True,
            "frame_idx": int(frame_idx),
            "events": events,
            "summary": dict(self.competition_duplicate_summary),
        }

    def consolidate_duplicate_nodes(
        self,
        frame_idx: int,
        *,
        dry_run: Optional[bool] = None,
    ) -> dict:
        """Run the consolidation passes that operate on duplicate / orphan
        nodes.  As of the latest revision, large-object multi-view
        duplicate logic (instance-envelope, observation-compatibility,
        duplicate-cluster) has been removed in favour of the simpler
        competition-based pass implemented by
        ``consolidate_competition_duplicates``.  Orphan-U pruning and
        the read-only sidecar audit remain.
        """
        if not self.enable_node_consolidation:
            return {
                "enabled": False,
                "frame_idx": int(frame_idx),
                "candidates": [],
                "actions": [],
            }
        if dry_run is None:
            dry_run = bool(self.node_consolidation_dry_run)
        # Stage 1: competition-based duplicate consolidation.
        comp_result = self.consolidate_competition_duplicates(
            int(frame_idx), dry_run=bool(dry_run)
        )
        comp_events = list(comp_result.get("events") or [])
        # Mirror competition events into the legacy node_consolidation
        # events ring so existing summary writers / sidecar consumers
        # still see merge / absorb_prune actions in a uniform shape.
        actions: list[dict] = []
        for ev in comp_events:
            entry = dict(ev)
            entry.setdefault("source", "competition_duplicate")
            actions.append(entry)
            if len(self.node_consolidation_events) < 4096:
                self.node_consolidation_events.append(entry)
        # Stage 2: orphan-U pruning (independent of duplicate scoring).
        orphan_events: list[dict] = []
        if self.orphan_u_prune_enable:
            orphan_events = self._prune_orphan_competition_loser_u_nodes(
                int(frame_idx), dry_run=bool(dry_run)
            )
        # Stage 3: dangling reference audit (best-effort).
        ok_after, dangling = self._assert_no_dangling_node_refs()
        return {
            "enabled": True,
            "frame_idx": int(frame_idx),
            "dry_run": bool(dry_run),
            "actions": actions,
            "competition_duplicate_summary": dict(
                self.competition_duplicate_summary
            ),
            "orphan_u_prune_events": orphan_events,
            "num_orphan_u_pruned": len(orphan_events),
            "dangling_refs_ok": bool(ok_after),
            "dangling_problems": dangling.get("problems", []) if not ok_after else [],
        }

    def _should_replace_tentative_local_edge(self, existing_edge, *, parent_id: str, support_count: int, recent_support_count: int, margin: float) -> bool:
        if existing_edge is None or getattr(existing_edge, "status", "committed") != "tentative":
            return True
        if existing_edge.src_node_id == parent_id:
            return True
        if recent_support_count > int(getattr(existing_edge, "recent_support_count", 0)):
            return True
        if support_count > int(getattr(existing_edge, "support_count", 0)):
            return True
        return float(margin) >= float(getattr(existing_edge, "margin", 0.0)) + self.tentative_local_replace_margin

    def _select_active_local_parent_candidate(
        self,
        child_id: str,
        posterior: LocalParentPosterior,
    ) -> Optional[dict]:
        """Choose the current parent used for tentative local-edge refresh.

        ``LocalParentPosterior.top1_parent_id`` is an accumulated-score winner.
        That is useful for stability, but it can become stale after a physical
        parent is split into a new node: the old parent remains top1 by history
        while the new parent owns all recent support.  The tentative graph needs
        to follow the active recent evidence so links do not disappear until the
        accumulated score catches up.
        """
        child_track = self.node_tracks.get(child_id)
        if child_track is None:
            return None

        top1_id = posterior.top1_parent_id
        top2_id = posterior.top2_parent_id
        candidate_scores = dict(getattr(posterior, "candidate_parent_scores", {}) or {})
        if top1_id is not None:
            candidate_scores.setdefault(top1_id, 0.0)
        if top2_id is not None:
            candidate_scores.setdefault(top2_id, 0.0)
        if not candidate_scores:
            return None

        top1_recent = posterior.recent_support_count(top1_id, window=3) if top1_id is not None else 0
        top1_score = float(candidate_scores.get(top1_id, 0.0) or 0.0) if top1_id is not None else 0.0
        switch_count = posterior.recent_switch_count(window=4)
        ranked: list[dict] = []

        for parent_id, score in candidate_scores.items():
            if parent_id is None:
                continue
            parent_id = str(parent_id)
            parent_track = self.node_tracks.get(parent_id)
            if parent_track is None:
                continue
            edge_type = self._edge_type_for_roles(parent_track.role, child_track.role)
            if edge_type is None:
                continue

            support_count = posterior.support_count(parent_id)
            recent_support_count = posterior.recent_support_count(parent_id, window=3)
            if support_count < 2 or recent_support_count < 2 or switch_count > (self.local_max_switches + 1):
                continue

            assignment_signal = self._latest_local_assignment_signals.get(child_id, {}).get(parent_id, {})
            assignment_strength = float(assignment_signal.get("strength", 0.0) or 0.0)
            parent_score = float(score or 0.0)
            other_scores = [
                float(v or 0.0)
                for pid, v in candidate_scores.items()
                if str(pid) != parent_id
            ]
            candidate_margin = parent_score - (max(other_scores) if other_scores else 0.0)
            is_raw_top1 = parent_id == top1_id

            if is_raw_top1:
                eligible = (
                    posterior.margin >= self.tentative_margin_local
                    or assignment_strength > 0.0
                    or top2_id is None
                )
                reason = "raw_top1"
            else:
                top1_stale = top1_recent < 2
                recent_overtakes_top1 = recent_support_count > top1_recent
                close_to_top1 = (
                    top1_id is None
                    or (top1_score - parent_score) <= max(self.local_stable_margin, self.tentative_margin_local * 2.0)
                )
                eligible = (
                    (top1_stale or recent_overtakes_top1)
                    and (close_to_top1 or assignment_strength > 0.0)
                )
                reason = "recent_active_fallback"
            if not eligible:
                continue

            if child_track.role == "U" and parent_track.role == "O":
                if posterior.owner_mode == "prefer_carrier" and posterior.has_viable_preferred_parent:
                    continue
                if self._has_recent_viable_preferred_carrier(child_id, posterior, exclude_parent_id=parent_id):
                    continue

            ranked.append(
                {
                    "parent_id": parent_id,
                    "edge_type": edge_type,
                    "support_count": int(support_count),
                    "recent_support_count": int(recent_support_count),
                    "switch_count": int(switch_count),
                    "assignment_strength": float(assignment_strength),
                    "evidence_score": float(parent_score),
                    "margin": float(candidate_margin),
                    "latest_margin": float(posterior.latest_evidence_margin),
                    "active_parent_reason": reason,
                    "is_raw_top1": bool(is_raw_top1),
                }
            )

        if not ranked:
            return None
        ranked.sort(
            key=lambda item: (
                int(item["recent_support_count"]),
                float(item["assignment_strength"]),
                float(item["evidence_score"]),
                int(item["support_count"]),
                1 if item["is_raw_top1"] else 0,
            ),
            reverse=True,
        )
        return ranked[0]

    def _edge_claim_payload(self, edge) -> dict:
        if edge is None:
            return {}
        return {
            "support_count": int(getattr(edge, "support_count", 0) or 0),
            "recent_support_count": int(getattr(edge, "recent_support_count", 0) or 0),
            "margin": float(getattr(edge, "margin", 0.0) or 0.0),
            "latest_margin": float(getattr(edge, "latest_margin", 0.0) or 0.0),
            "evidence_score": float(getattr(edge, "evidence_score", 0.0) or 0.0),
            "status": str(getattr(edge, "status", "") or ""),
        }

    def _current_parent_claim(self, child_id: str) -> dict:
        child_track = self.node_tracks.get(child_id)
        claim = {
            "child_id": child_id,
            "child_role": None if child_track is None else getattr(child_track, "role", None),
            "existing_parent_id": None,
            "existing_parent_role": None,
            "existing_edge_type": None,
            "source": None,
            "status": None,
            "support_count": 0,
            "recent_support_count": 0,
            "margin": 0.0,
            "latest_margin": 0.0,
            "evidence_score": 0.0,
            "is_aggregate_cabinet_parent": False,
            "has_current_strong_evidence": False,
        }
        gid = self.cabinet_aggregator.member_to_cabinet.get(child_id)
        group = self.cabinet_aggregator.cabinet_tracks.get(gid) if gid is not None else None
        if group is not None:
            claim.update(
                {
                    "existing_parent_id": group.cabinet_id,
                    "existing_parent_role": "O",
                    "existing_edge_type": "O-C",
                    "source": "aggregate_cabinet",
                    "status": "aggregate",
                    "support_count": len(group.support_frames),
                    "recent_support_count": min(3, len(group.support_frames)),
                    "evidence_score": float(len(group.seeded_member_ids)),
                    "is_aggregate_cabinet_parent": True,
                }
            )
            return claim

        edge = self._current_local_graph_edge(child_id)
        if edge is not None:
            parent_track = self.node_tracks.get(edge.src_node_id)
            signal = self._latest_current_parent_claims.get(child_id, {}).get(edge.src_node_id, {})
            claim.update(
                {
                    "existing_parent_id": edge.src_node_id,
                    "existing_parent_role": None if parent_track is None else getattr(parent_track, "role", None),
                    "existing_edge_type": edge.edge_type,
                    "source": "graph",
                    **self._edge_claim_payload(edge),
                    "has_current_strong_evidence": bool(signal.get("strong_current", False)),
                    "is_aggregate_cabinet_parent": bool(self._is_aggregate_cabinet_track(parent_track)),
                }
            )
            return claim

        posterior = self.local_posteriors.get(child_id)
        if posterior is not None and posterior.stable_parent_id is not None:
            parent_track = self.node_tracks.get(posterior.stable_parent_id)
            signal = self._latest_current_parent_claims.get(child_id, {}).get(posterior.stable_parent_id, {})
            claim.update(
                {
                    "existing_parent_id": posterior.stable_parent_id,
                    "existing_parent_role": None if parent_track is None else getattr(parent_track, "role", None),
                    "existing_edge_type": None if parent_track is None or child_track is None else self._edge_type_for_roles(parent_track.role, child_track.role),
                    "source": "posterior_stable",
                    "status": "stable",
                    "support_count": posterior.support_count(posterior.stable_parent_id),
                    "recent_support_count": posterior.recent_support_count(posterior.stable_parent_id, window=3),
                    "margin": float(posterior.margin),
                    "latest_margin": float(posterior.latest_evidence_margin),
                    "evidence_score": float(posterior.candidate_parent_scores.get(posterior.stable_parent_id, 0.0)),
                    "has_current_strong_evidence": bool(signal.get("strong_current", False)),
                }
            )
        return claim

    def _candidate_parent_claim(self, parent_id: str, child_id: str, edge_type: str, posterior: LocalParentPosterior, commit_payload: dict) -> dict:
        parent_track = self.node_tracks.get(parent_id)
        child_track = self.node_tracks.get(child_id)
        signal = self._latest_current_parent_claims.get(child_id, {}).get(parent_id, {})
        status = str(signal.get("assignment_status") or "").strip().lower()
        return {
            "parent_id": parent_id,
            "parent_role": None if parent_track is None else getattr(parent_track, "role", None),
            "child_id": child_id,
            "child_role": None if child_track is None else getattr(child_track, "role", None),
            "edge_type": edge_type,
            "source": commit_payload.get("source", "posterior"),
            "support_count": int(commit_payload.get("support_count", posterior.support_count(parent_id)) or 0),
            "recent_support_count": int(commit_payload.get("recent_support_count", posterior.recent_support_count(parent_id, window=3)) or 0),
            "margin": float(commit_payload.get("margin", posterior.margin) or 0.0),
            "latest_margin": float(commit_payload.get("latest_margin", posterior.latest_evidence_margin) or 0.0),
            "evidence_score": float(commit_payload.get("evidence_score", posterior.candidate_parent_scores.get(parent_id, 0.0)) or 0.0),
            "has_current_strong_evidence": bool(signal.get("strong_current", False)),
            "is_direct_observation_selected": bool(signal.get("selected", False) or status in {"confirmed", "confirmed_mask"}),
            "is_fallback_relation": bool(parent_id in posterior.fallback_parent_ids or posterior.stable_parent_source == "fallback"),
        }

    def _viable_carrier_parent_claim_for_unit(self, child_id: str, *, exclude_parent_id: Optional[str] = None) -> Optional[dict]:
        for parent_id, signal in self._latest_current_parent_claims.get(child_id, {}).items():
            parent_track = self.node_tracks.get(parent_id)
            if parent_id != exclude_parent_id and parent_track is not None and parent_track.role == "C" and bool(signal.get("strong_current", False)):
                return {"source": "observation_local_edge", "parent_id": parent_id, "reason": "current_strong_cu"}

        edge = self._current_local_graph_edge(child_id)
        if edge is not None and edge.src_node_id != exclude_parent_id and edge.edge_type == "C-U":
            parent_track = self.node_tracks.get(edge.src_node_id)
            if parent_track is not None and parent_track.role == "C":
                return {"source": "graph", "parent_id": edge.src_node_id, "reason": "committed_or_tentative_cu"}

        posterior = self.local_posteriors.get(child_id)
        if posterior is not None:
            for parent_id in [posterior.stable_parent_id, posterior.top1_parent_id]:
                if parent_id is None or parent_id == exclude_parent_id:
                    continue
                parent_track = self.node_tracks.get(parent_id)
                if parent_track is None or parent_track.role != "C":
                    continue
                if posterior.support_count(parent_id) >= self.preferred_min_recent_support and posterior.recent_support_count(parent_id, window=self.local_history_size) >= self.preferred_min_recent_support:
                    return {"source": "posterior", "parent_id": parent_id, "reason": "recent_stable_or_top1_cu"}
        return None

    def _resolve_parent_ownership_conflict(self, existing_claim: dict, candidate_claim: dict) -> dict:
        decision = {
            "accept": True,
            "decision": "accept",
            "reason": "no_conflict",
            "prune_cabinet_members": [],
        }
        child_id = str(candidate_claim.get("child_id") or "")
        child_track = self.node_tracks.get(child_id)
        parent_track = self.node_tracks.get(str(candidate_claim.get("parent_id") or ""))
        child_role = candidate_claim.get("child_role")
        edge_type = candidate_claim.get("edge_type")

        if (
            child_role == "C"
            and child_track is not None
            and getattr(child_track, "label", None) in self.graph_policy.cabinet_seed_carriers
            and bool(existing_claim.get("is_aggregate_cabinet_parent"))
            and edge_type == "O-C"
            and not self._is_aggregate_cabinet_track(parent_track)
        ):
            if bool(candidate_claim.get("has_current_strong_evidence")) or bool(candidate_claim.get("is_direct_observation_selected")):
                decision.update(
                    {
                        "accept": True,
                        "decision": "ordinary_parent_claim_wins_pruned_cabinet",
                        "reason": "ordinary_parent_claim_wins_pruned_cabinet",
                        "prune_cabinet_members": [child_id],
                    }
                )
            else:
                decision.update(
                    {
                        "accept": False,
                        "decision": "block",
                        "reason": "blocked_due_to_existing_parent_claim",
                    }
                )
            return decision

        if child_role == "U" and edge_type == "O-U":
            viable_carrier = self._viable_carrier_parent_claim_for_unit(child_id, exclude_parent_id=candidate_claim.get("parent_id"))
            if viable_carrier is not None:
                decision.update(
                    {
                        "accept": False,
                        "decision": "block",
                        "reason": "blocked_due_to_viable_carrier_parent",
                        "viable_carrier_claim": viable_carrier,
                    }
                )
                return decision

        return decision

    def _should_accept_local_parent_commit(
        self,
        parent_id: str,
        child_id: str,
        edge_type: str,
        posterior: LocalParentPosterior,
        commit_payload: dict,
    ) -> dict:
        existing_claim = self._current_parent_claim(child_id)
        candidate_claim = self._candidate_parent_claim(parent_id, child_id, edge_type, posterior, commit_payload)
        decision = self._resolve_parent_ownership_conflict(existing_claim, candidate_claim)
        record = {
            "edge_type": edge_type,
            "parent_id": parent_id,
            "child_id": child_id,
            "parent_role": candidate_claim.get("parent_role"),
            "child_role": candidate_claim.get("child_role"),
            "existing_claim_source": existing_claim.get("source"),
            "candidate_claim_source": candidate_claim.get("source"),
            "decision": decision.get("decision"),
            "reason": decision.get("reason"),
            "existing_claim": existing_claim,
            "candidate_claim": candidate_claim,
            "arbitration_decision": decision,
        }
        return {
            "accept": bool(decision.get("accept", True)),
            "record": record,
            "prune_cabinet_members": list(decision.get("prune_cabinet_members", [])),
        }

    def _record_local_parent_arbitration(self, frame_idx: int, result: dict) -> None:
        record = result.get("record", {})
        bucket = self.frame_assoc_debug.setdefault(int(frame_idx), {"frame_idx": int(frame_idx)}).setdefault(
            "local_parent_arbitration",
            {"accepted_edges": [], "blocked_edges": [], "pruned_cabinet_members_due_to_parent_claim": []},
        )
        if result.get("accept"):
            bucket["accepted_edges"].append(record)
        else:
            bucket["blocked_edges"].append(record)
        if result.get("prune_cabinet_members"):
            bucket["pruned_cabinet_members_due_to_parent_claim"].extend(list(result.get("prune_cabinet_members", [])))

    def _apply_parent_arbitration_prunes(self, result: dict, *, frame_idx: int) -> dict:
        member_ids = list(result.get("prune_cabinet_members", []))
        if not member_ids:
            return {"removed_members": [], "dropped_groups": []}
        prune_debug = self.cabinet_aggregator.prune_members(
            member_ids,
            frame_idx=frame_idx,
            reason="ordinary_parent_claim_wins",
            node_tracks=self.node_tracks,
        )
        for member_id in member_ids:
            track = self.node_tracks.get(member_id)
            if track is not None:
                track.cabinet_group_id = None
        self.graph.local_edges = {
            key: edge
            for key, edge in self.graph.local_edges.items()
            if not (
                edge.dst_node_id in set(member_ids)
                and (edge.relation_text == "aggregate_cabinet" or str(edge.src_node_id).startswith("O_CABINET_"))
            )
        }
        return prune_debug

    def _is_lowshot_unambiguous_local_edge(
        self,
        child_id: str,
        parent_id: str,
        posterior: "LocalParentPosterior",
        signal: dict,
    ) -> tuple[bool, dict]:
        """Decide whether a low-shot, unambiguous local edge should enter the
        graph as a tentative edge with retention_policy=until_contradicted.

        Returns (accepted, reason_payload) where reason_payload contains all
        fields used in the decision for debug/baseline dump consumption.
        """
        reason: dict = {
            "child_id": child_id,
            "parent_id": parent_id,
            "edge_type": str(signal.get("edge_type") or ""),
            "child_role": str(signal.get("child_role") or ""),
            "parent_role": str(signal.get("parent_role") or ""),
            "status_strength": max(
                float(signal.get("strength", 0.0)),
                float(signal.get("edge_evidence_strength", 0.0) or 0.0),
            ),
            "assignment_strength": float(signal.get("strength", 0.0)),
            "edge_evidence_strength": float(signal.get("edge_evidence_strength", 0.0) or 0.0),
            "num_candidate_parents": int(signal.get("num_candidate_parents", 0) or 0),
            "num_strong_parents": int(signal.get("num_strong_parents", 0) or 0),
            "strong_current": bool(signal.get("strong_current", False)),
            "has_competition": bool(signal.get("has_competition", False)),
            "competition_basis": str(signal.get("competition_basis") or "num_strong_parents"),
            "owner_conflict_penalty": float(signal.get("owner_conflict_penalty", 0.0)),
            "carrier_bypass_penalty": float(signal.get("carrier_bypass_penalty", 0.0)),
            "role_compatible": bool(signal.get("role_compatible", False)),
            "label_compatible": bool(signal.get("label_compatible", True)),
            "selected": bool(signal.get("selected", False)),
            "pass_thr": bool(signal.get("pass_thr", False)),
            "is_chosen_parent": bool(signal.get("is_chosen_parent", False)),
            "support_count": int(posterior.support_count(parent_id)),
            "top1_parent_id": posterior.top1_parent_id,
            "top2_parent_id": posterior.top2_parent_id,
            "rejected_reason": None,
            "accepted": False,
        }

        if not self.tentative_local_lowshot_enable:
            reason["rejected_reason"] = "lowshot_disabled"
            return False, reason

        edge_type = reason["edge_type"]
        if edge_type not in self.tentative_local_lowshot_edge_types:
            reason["rejected_reason"] = "edge_type_not_eligible"
            return False, reason

        valid_role_combos = {
            "O-C": ("O", "C"),
            "C-U": ("C", "U"),
            "O-U": ("O", "U"),
        }
        expected = valid_role_combos.get(edge_type)
        if expected is None or (reason["parent_role"], reason["child_role"]) != expected:
            reason["rejected_reason"] = "role_combo_invalid"
            return False, reason
        if not reason["role_compatible"]:
            reason["rejected_reason"] = "role_incompatible"
            return False, reason
        if not reason["label_compatible"]:
            reason["rejected_reason"] = "label_incompatible"
            return False, reason

        if posterior.top1_parent_id != parent_id:
            reason["rejected_reason"] = "not_top1"
            return False, reason
        if reason["support_count"] < 1:
            reason["rejected_reason"] = "no_support"
            return False, reason
        if edge_type == "O-U" and reason["support_count"] < 2:
            reason["rejected_reason"] = "ou_lowshot_requires_multiframe_support"
            return False, reason

        threshold = float(
            self.tentative_local_lowshot_min_status_strength_by_edge_type.get(edge_type, 1.0)
        )
        reason["min_status_strength_threshold"] = threshold
        if reason["status_strength"] < threshold:
            reason["rejected_reason"] = "weak_status_strength"
            return False, reason

        if self.tentative_local_lowshot_require_no_competition:
            num_strong_parents = int(signal.get("num_strong_parents", 0) or 0)
            if num_strong_parents > 1:
                reason["rejected_reason"] = "competition_present"
                return False, reason

        # Penalty gates: any non-trivial conflict disqualifies the edge.
        if reason["owner_conflict_penalty"] > 0.0:
            reason["rejected_reason"] = "owner_conflict"
            return False, reason
        if reason["carrier_bypass_penalty"] > 0.0:
            reason["rejected_reason"] = "carrier_bypass"
            return False, reason

        if child_id not in self.node_tracks or parent_id not in self.node_tracks:
            reason["rejected_reason"] = "missing_node_track"
            return False, reason

        # Sibling-duplicate-risk gate: avoid emitting a lowshot tentative
        # edge when the candidate child looks like a duplicate of an
        # existing committed sibling under the same parent (no co-visibility
        # + same role/label + sibling clearly stronger or geometry close).
        risk, risk_dbg = self._lowshot_sibling_duplicate_risk(
            parent_id=parent_id,
            child_id=child_id,
            edge_type=edge_type,
            signal=signal,
        )
        reason.update(risk_dbg)
        if risk:
            reason["rejected_reason"] = "sibling_duplicate_risk"
            return False, reason

        reason["accepted"] = True
        return True, reason

    # ------------------------------------------------------------------
    # Parent-level unstable-sibling arbitration over the low-shot tentative
    # path.  Operates on a list of low-shot candidates already accepted by
    # ``_is_lowshot_unambiguous_local_edge``; groups them by
    # (parent_id, edge_type, child_role, normalized_label) and (when the
    # parent has no committed same-label child) keeps only the candidate
    # with a clearly-best score.  Co-visibility between any two children
    # in a group bypasses the gate (real distinct objects).
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_label(label: str) -> str:
        return (label or "").strip().lower()

    def _unstable_sibling_group_key(self, candidate: dict) -> tuple:
        return (
            str(candidate.get("parent_id") or ""),
            str(candidate.get("edge_type") or ""),
            str(candidate.get("child_role") or ""),
            self._normalize_label(candidate.get("child_label") or ""),
        )

    def _score_unstable_sibling_candidate(self, candidate: dict) -> tuple[float, dict]:
        """Deterministic score for one low-shot candidate.

        Higher is better.  Returns (score, breakdown_dict).
        """
        signal = candidate.get("signal", {}) or {}
        edge_strength = float(candidate.get("edge_strength", 0.0) or 0.0)
        # Signal score: selected/pass_thr/contain/strong_current.
        sig_score = 0.0
        if signal.get("selected"):
            sig_score += 0.4
        if signal.get("pass_thr"):
            sig_score += 0.3
        if signal.get("is_chosen_parent"):
            sig_score += 0.2
        if signal.get("strong_current"):
            sig_score += 0.1
        # Cap for stability.
        sig_score = min(sig_score, 1.0)
        node_quality = float(candidate.get("node_quality", 0.0) or 0.0)
        # Normalise node_quality to roughly [0..1] over the typical 0..10 band.
        norm_quality = max(0.0, min(node_quality / 10.0, 1.0))
        support_count = int(candidate.get("support_count", 0) or 0)
        recent_support = int(candidate.get("recent_support_count", 0) or 0)
        margin = float(candidate.get("margin", 0.0) or 0.0)
        latest_margin = float(candidate.get("latest_margin", 0.0) or 0.0)
        switch_count = int(candidate.get("switch_count", 0) or 0)
        score = (
            2.0 * edge_strength
            + 1.0 * sig_score
            + 0.6 * norm_quality
            + 0.4 * min(support_count, 3)
            + 0.3 * min(recent_support, 3)
            + 0.3 * min(margin, 2.0)
            + 0.2 * min(latest_margin, 1.0)
            - 0.5 * float(switch_count)
        )
        breakdown = {
            "edge_strength": edge_strength,
            "signal_score": sig_score,
            "node_quality": node_quality,
            "support_count": support_count,
            "recent_support_count": recent_support,
            "margin": margin,
            "latest_margin": latest_margin,
            "switch_count": switch_count,
        }
        return float(score), breakdown

    def _group_has_internal_covisibility(self, candidates: list[dict]) -> bool:
        if len(candidates) <= 1:
            return False
        ids = [str(c.get("child_id") or "") for c in candidates]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if not ids[i] or not ids[j]:
                    continue
                if self.node_covisibility_count(ids[i], ids[j]) >= 1:
                    return True
        return False

    def _arbitrate_unstable_sibling_lowshot_group(
        self,
        group_key: tuple,
        candidates: list[dict],
    ) -> tuple[list[dict], list[dict], dict]:
        """Return ``(accepted, blocked, summary)`` for a single group.

        Single-candidate groups are always accepted.  Groups whose children
        are internally co-visible are always accepted.  Otherwise the
        winner-by-score rule applies when ``top1 - top2 >=
        unstable_sibling_best_margin`` (else all kept as a tie).
        """
        summary: dict = {
            "group_key": list(group_key),
            "group_size": int(len(candidates)),
            "accepted_child_ids": [],
            "blocked_child_ids": [],
            "best_child_id": None,
            "best_score": None,
            "second_score": None,
            "margin": None,
            "reason": None,
            "per_child_scores": [],
        }
        if len(candidates) <= 1:
            for c in candidates:
                c["unstable_sibling_action"] = "accept"
                c["unstable_sibling_reason"] = "single_candidate"
                c["unstable_sibling_blocked"] = False
            summary["reason"] = "single_candidate"
            summary["accepted_child_ids"] = [str(c.get("child_id")) for c in candidates]
            return list(candidates), [], summary

        # Per-child scoring.
        scored: list[tuple[float, dict, dict]] = []
        for c in candidates:
            score, breakdown = self._score_unstable_sibling_candidate(c)
            c["unstable_sibling_score"] = float(score)
            c["unstable_sibling_score_breakdown"] = breakdown
            scored.append((score, c, breakdown))
            summary["per_child_scores"].append({
                "child_id": str(c.get("child_id")),
                "score": float(score),
                "breakdown": breakdown,
            })
        scored.sort(key=lambda kv: kv[0], reverse=True)
        for rank, (score, c, _) in enumerate(scored):
            c["unstable_sibling_rank"] = int(rank)

        # Co-visibility protection: any internal covis -> accept all.
        if self._group_has_internal_covisibility(candidates):
            for c in candidates:
                c["unstable_sibling_action"] = "accept"
                c["unstable_sibling_reason"] = "covisible_children_keep_all"
                c["unstable_sibling_blocked"] = False
            summary["reason"] = "covisible_children_keep_all"
            summary["accepted_child_ids"] = [str(c.get("child_id")) for c in candidates]
            summary["best_child_id"] = str(scored[0][1].get("child_id"))
            summary["best_score"] = float(scored[0][0])
            summary["second_score"] = float(scored[1][0])
            summary["margin"] = float(scored[0][0] - scored[1][0])
            return list(candidates), [], summary

        # If parent already has a committed same-label child, the existing
        # ``_lowshot_sibling_duplicate_risk`` gate handles ranking.  Don't
        # double-block here; just keep all that survived stage-1.
        parent_id = group_key[0]
        child_role = group_key[2]
        sample_label = candidates[0].get("child_label") or ""
        committed_siblings = self._stable_same_label_children_of_parent(
            parent_id, child_role, sample_label, require_committed=True
        )
        committed_siblings = [
            s for s in committed_siblings if s not in {str(c.get("child_id")) for c in candidates}
        ]
        if committed_siblings:
            for c in candidates:
                c["unstable_sibling_action"] = "accept"
                c["unstable_sibling_reason"] = "committed_sibling_present_keep_all"
                c["unstable_sibling_blocked"] = False
            summary["reason"] = "committed_sibling_present_keep_all"
            summary["accepted_child_ids"] = [str(c.get("child_id")) for c in candidates]
            summary["best_child_id"] = str(scored[0][1].get("child_id"))
            summary["best_score"] = float(scored[0][0])
            summary["second_score"] = float(scored[1][0])
            summary["margin"] = float(scored[0][0] - scored[1][0])
            return list(candidates), [], summary

        # Tie / clear-winner decision.
        best_score = float(scored[0][0])
        second_score = float(scored[1][0])
        margin = best_score - second_score
        summary["best_child_id"] = str(scored[0][1].get("child_id"))
        summary["best_score"] = best_score
        summary["second_score"] = second_score
        summary["margin"] = float(margin)
        threshold = float(self.unstable_sibling_best_margin)
        if margin < threshold:
            for c in candidates:
                c["unstable_sibling_action"] = "accept"
                c["unstable_sibling_reason"] = "unstable_sibling_tie_keep_all"
                c["unstable_sibling_blocked"] = False
            summary["reason"] = "unstable_sibling_tie_keep_all"
            summary["accepted_child_ids"] = [str(c.get("child_id")) for c in candidates]
            return list(candidates), [], summary

        # Clear winner: accept top1, block the rest.
        winner = scored[0][1]
        winner["unstable_sibling_action"] = "accept"
        winner["unstable_sibling_reason"] = "unstable_sibling_winner"
        winner["unstable_sibling_blocked"] = False
        accepted = [winner]
        blocked: list[dict] = []
        for _, cand, _ in scored[1:]:
            cand["unstable_sibling_action"] = "block"
            cand["unstable_sibling_reason"] = "unstable_sibling_worse_than_best"
            cand["unstable_sibling_blocked"] = True
            blocked.append(cand)
        summary["reason"] = "unstable_sibling_winner"
        summary["accepted_child_ids"] = [str(c.get("child_id")) for c in accepted]
        summary["blocked_child_ids"] = [str(c.get("child_id")) for c in blocked]
        return accepted, blocked, summary

    def _refresh_tentative_local_edges(self, frame_idx: int) -> None:
        # Reset per-frame lowshot debug snapshot.
        self._latest_local_lowshot_debug = {}
        # Stage 1: per-child decision.  Normal-path edges are upserted
        # immediately; low-shot accepted candidates are *deferred* into
        # ``lowshot_candidates`` so they can be partitioned into
        # parent-level groups before any graph mutation happens.
        lowshot_candidates: list[dict] = []
        for child_id, posterior in self.local_posteriors.items():
            active_parent = self._select_active_local_parent_candidate(child_id, posterior)
            parent_id = str(active_parent["parent_id"]) if active_parent is not None else posterior.top1_parent_id
            if parent_id is None:
                continue
            parent_id = str(parent_id)
            child_track = self.node_tracks.get(child_id)
            parent_track = self.node_tracks.get(parent_id)
            if child_track is None or parent_track is None:
                continue
            edge_type = str(
                (active_parent or {}).get("edge_type")
                or self._edge_type_for_roles(parent_track.role, child_track.role)
                or ""
            )
            if not edge_type:
                continue

            support_count = int(active_parent["support_count"]) if active_parent is not None else posterior.support_count(parent_id)
            recent_support_count = int(active_parent["recent_support_count"]) if active_parent is not None else posterior.recent_support_count(parent_id, window=3)
            switch_count = int(active_parent["switch_count"]) if active_parent is not None else posterior.recent_switch_count(window=4)
            assignment_signal = self._latest_local_assignment_signals.get(child_id, {}).get(parent_id, {})
            assignment_strength = float(assignment_signal.get("strength", 0.0) or 0.0)
            if active_parent is not None:
                existing_edge = self._current_local_graph_edge(child_id)
                if not self._should_replace_tentative_local_edge(
                    existing_edge,
                    parent_id=parent_id,
                    support_count=support_count,
                    recent_support_count=recent_support_count,
                    margin=float(active_parent["margin"]),
                ):
                    continue

                support_frames = posterior.candidate_support_frames.get(parent_id, [])
                arbitration = self._should_accept_local_parent_commit(
                    parent_id,
                    child_id,
                    edge_type,
                    posterior,
                    {
                        "source": "posterior_tentative",
                        "support_count": support_count,
                        "recent_support_count": recent_support_count,
                        "margin": float(active_parent["margin"]),
                        "latest_margin": float(active_parent["latest_margin"]),
                        "evidence_score": float(active_parent["evidence_score"]),
                        "active_parent_reason": str(active_parent.get("active_parent_reason") or ""),
                        "raw_top1_parent_id": posterior.top1_parent_id,
                        "raw_top2_parent_id": posterior.top2_parent_id,
                    },
                )
                self._record_local_parent_arbitration(frame_idx, arbitration)
                if not arbitration["accept"]:
                    continue
                self._apply_parent_arbitration_prunes(arbitration, frame_idx=frame_idx)
                self.graph.upsert_local_edge(
                    parent_id,
                    child_id,
                    edge_type,
                    relation_text=self._relation_text_for_instance_edge(parent_id, child_id, edge_type),
                    committed_kf=-1,
                    support_count=support_count,
                    evidence_score=float(active_parent["evidence_score"]),
                    status="tentative",
                    first_seen_frame=int(support_frames[0]) if support_frames else int(frame_idx),
                    last_seen_frame=int(support_frames[-1]) if support_frames else int(frame_idx),
                    last_update_source="local_active_parent",
                    margin=float(active_parent["margin"]),
                    latest_margin=float(active_parent["latest_margin"]),
                    recent_support_count=recent_support_count,
                    switch_count=switch_count,
                    retention_policy="ttl",
                )
                continue

            # Normal multi-frame tentative path failed -> try low-shot unambiguous branch.
            if not self.tentative_local_lowshot_enable:
                continue
            signal = dict(assignment_signal) if assignment_signal else {}
            # Ensure required fields exist; fall back to track-derived values.
            signal.setdefault("edge_type", edge_type)
            signal.setdefault("child_role", child_track.role)
            signal.setdefault("parent_role", parent_track.role)
            signal.setdefault("child_label", child_track.label or "")
            signal.setdefault("parent_label", parent_track.label or "")
            signal.setdefault("strength", float(assignment_strength))
            signal.setdefault("edge_evidence_strength", float(assignment_strength))
            signal.setdefault("num_candidate_parents", 0)
            signal.setdefault("num_strong_parents", 0)
            signal.setdefault("strong_current", False)
            signal.setdefault("competition_basis", "num_strong_parents")
            signal.setdefault("has_competition", False)
            signal.setdefault("role_compatible", True)
            signal.setdefault("label_compatible", True)
            signal.setdefault("owner_conflict_penalty", 0.0)
            signal.setdefault("carrier_bypass_penalty", 0.0)

            accepted, reason = self._is_lowshot_unambiguous_local_edge(
                child_id, parent_id, posterior, signal
            )
            self._latest_local_lowshot_debug.setdefault(child_id, {})[parent_id] = reason
            if not accepted:
                if reason.get("rejected_reason") == "sibling_duplicate_risk":
                    # Bounded log so the regression summary can list every block.
                    if len(self.lowshot_sibling_block_events) < 4096:
                        self.lowshot_sibling_block_events.append({
                            "frame_idx": int(frame_idx),
                            "child_id": child_id,
                            "parent_id": parent_id,
                            "edge_type": edge_type,
                            "child_role": child_track.role,
                            "child_label": self._track_dominant_label(child_id),
                            "sibling_id": reason.get("duplicate_risk_sibling_id"),
                            "sibling_covisibility_count": reason.get("sibling_covisibility_count"),
                            "sibling_centroid_distance": reason.get("sibling_centroid_distance"),
                            "sibling_bbox_iou": reason.get("sibling_bbox_iou"),
                            "sibling_bbox_overlap": reason.get("sibling_bbox_overlap"),
                            "current_child_quality": reason.get("current_child_quality"),
                            "sibling_child_quality": reason.get("sibling_child_quality"),
                            "duplicate_risk_reason": reason.get("duplicate_risk_reason"),
                        })
                    # Record competition loss for orphan-U pruning.
                    cur_q = float(reason.get("current_child_quality") or 0.0)
                    sib_q = float(reason.get("sibling_child_quality") or 0.0)
                    self._record_parent_competition_loss(
                        child_id,
                        winner_id=reason.get("duplicate_risk_sibling_id"),
                        parent_id=parent_id,
                        reason="sibling_duplicate_risk",
                        margin=float(sib_q - cur_q),
                        frame_idx=int(frame_idx),
                    )
                    # Competition ledger: child_id and the duplicate
                    # sibling competed for the same parent.
                    sib_id = reason.get("duplicate_risk_sibling_id")
                    if sib_id and sib_id in self.node_tracks:
                        self._record_assignment_competition_event(
                            frame_idx=int(frame_idx),
                            type="parent_competition",
                            node_a=str(child_id),
                            node_b=str(sib_id),
                            shared_parent_id=str(parent_id),
                            source="sibling_duplicate_risk",
                            strength=float(abs(sib_q - cur_q)),
                        )
                continue

            # Defer upsert until parent-group arbitration has run.
            qa, _ = self._node_quality_score(child_id)
            lowshot_candidates.append({
                "parent_id": parent_id,
                "child_id": child_id,
                "edge_type": edge_type,
                "child_role": child_track.role,
                "child_label": self._track_dominant_label(child_id) or (child_track.label or ""),
                "parent_role": parent_track.role,
                "parent_label": self._track_dominant_label(parent_id) or (parent_track.label or ""),
                "posterior": posterior,
                "signal": signal,
                "lowshot_reason": reason,
                "edge_strength": max(
                    float(signal.get("strength", 0.0) or 0.0),
                    float(signal.get("edge_evidence_strength", 0.0) or 0.0),
                ),
                "node_quality": float(qa),
                "support_count": int(support_count),
                "recent_support_count": int(recent_support_count),
                "margin": float(posterior.margin),
                "latest_margin": float(posterior.latest_evidence_margin),
                "switch_count": int(switch_count),
                "support_frames": list(posterior.candidate_support_frames.get(parent_id, [])),
            })

        # Stage 2: parent-level unstable-sibling arbitration over the
        # accepted low-shot candidates.  When disabled, every candidate is
        # accepted unchanged.
        if not lowshot_candidates:
            return

        groups: dict[tuple, list[dict]] = {}
        for cand in lowshot_candidates:
            groups.setdefault(self._unstable_sibling_group_key(cand), []).append(cand)

        if not self.unstable_sibling_arbitration_enable:
            accepted_lowshot = list(lowshot_candidates)
        else:
            accepted_lowshot = []
            for gkey, cands in groups.items():
                accepted, blocked, summary = (
                    self._arbitrate_unstable_sibling_lowshot_group(gkey, cands)
                )
                accepted_lowshot.extend(accepted)
                if len(cands) > 1 or summary.get("reason") != "single_candidate":
                    event = {
                        "frame_idx": int(frame_idx),
                        "group_key": list(gkey),
                        "parent_id": gkey[0],
                        "edge_type": gkey[1],
                        "child_role": gkey[2],
                        "child_label": gkey[3],
                        **summary,
                    }
                    if len(self.unstable_sibling_arbitration_events) < 4096:
                        self.unstable_sibling_arbitration_events.append(event)
                    # Competition ledger (parent_competition): every pair
                    # of children inside the group competes for the same
                    # parent.
                    cand_ids = [
                        str(c.get("child_id"))
                        for c in cands
                        if c.get("child_id")
                    ]
                    cand_ids = [cid for cid in cand_ids if cid in self.node_tracks]
                    for i in range(len(cand_ids)):
                        for j in range(i + 1, len(cand_ids)):
                            self._record_assignment_competition_event(
                                frame_idx=int(frame_idx),
                                type="parent_competition",
                                node_a=cand_ids[i],
                                node_b=cand_ids[j],
                                shared_parent_id=str(gkey[0]),
                                source="unstable_sibling_arbitration",
                                strength=1.0,
                            )
                # Annotate per-child debug snapshot for blocked candidates.
                for cand in blocked:
                    dbg = self._latest_local_lowshot_debug.setdefault(
                        cand["child_id"], {}
                    ).setdefault(cand["parent_id"], dict(cand.get("lowshot_reason") or {}))
                    dbg["accepted"] = False
                    dbg["rejected_reason"] = "unstable_sibling_worse_than_best"
                    dbg["unstable_sibling_group_key"] = list(gkey)
                    dbg["unstable_sibling_group_size"] = int(len(cands))
                    dbg["unstable_sibling_arbitration_enabled"] = True
                    dbg["unstable_sibling_score"] = cand.get("unstable_sibling_score")
                    dbg["unstable_sibling_score_breakdown"] = cand.get(
                        "unstable_sibling_score_breakdown"
                    )
                    dbg["unstable_sibling_rank"] = cand.get("unstable_sibling_rank")
                    dbg["unstable_sibling_best_child_id"] = summary.get("best_child_id")
                    dbg["unstable_sibling_best_score"] = summary.get("best_score")
                    dbg["unstable_sibling_second_score"] = summary.get("second_score")
                    dbg["unstable_sibling_margin"] = summary.get("margin")
                    dbg["unstable_sibling_action"] = "block"
                    dbg["unstable_sibling_reason"] = (
                        "unstable_sibling_worse_than_best"
                    )
                    dbg["unstable_sibling_blocked"] = True
                    # Record competition loss (bounded recall list).
                    self._record_parent_competition_loss(
                        cand["child_id"],
                        winner_id=summary.get("best_child_id"),
                        parent_id=gkey[0],
                        reason="unstable_sibling_arbitration",
                        margin=float(summary.get("margin") or 0.0),
                        frame_idx=int(frame_idx),
                    )
                # Annotate accepted candidates too.
                for cand in accepted:
                    dbg = self._latest_local_lowshot_debug.setdefault(
                        cand["child_id"], {}
                    ).setdefault(cand["parent_id"], dict(cand.get("lowshot_reason") or {}))
                    dbg["unstable_sibling_group_key"] = list(gkey)
                    dbg["unstable_sibling_group_size"] = int(len(cands))
                    dbg["unstable_sibling_arbitration_enabled"] = True
                    dbg["unstable_sibling_score"] = cand.get("unstable_sibling_score")
                    dbg["unstable_sibling_score_breakdown"] = cand.get(
                        "unstable_sibling_score_breakdown"
                    )
                    dbg["unstable_sibling_rank"] = cand.get("unstable_sibling_rank")
                    dbg["unstable_sibling_best_child_id"] = summary.get("best_child_id")
                    dbg["unstable_sibling_best_score"] = summary.get("best_score")
                    dbg["unstable_sibling_second_score"] = summary.get("second_score")
                    dbg["unstable_sibling_margin"] = summary.get("margin")
                    dbg["unstable_sibling_action"] = cand.get("unstable_sibling_action")
                    dbg["unstable_sibling_reason"] = cand.get("unstable_sibling_reason")
                    dbg["unstable_sibling_blocked"] = False

        # Stage 3: actual upserts for accepted low-shot candidates.
        for cand in accepted_lowshot:
            child_id = cand["child_id"]
            parent_id = cand["parent_id"]
            edge_type = cand["edge_type"]
            posterior = cand["posterior"]
            signal = cand["signal"]
            reason = cand["lowshot_reason"]
            existing_edge = self._current_local_graph_edge(child_id)
            if existing_edge is not None:
                # Never overwrite committed edges with low-shot tentatives.
                if str(getattr(existing_edge, "status", "")).lower() == "committed":
                    continue
                # Same edge -> just refresh last_seen_frame via upsert below.
                same_edge = (
                    getattr(existing_edge, "src_node_id", None) == parent_id
                    and getattr(existing_edge, "dst_node_id", None) == child_id
                    and getattr(existing_edge, "edge_type", None) == edge_type
                )
                if not same_edge:
                    existing_strength = float(getattr(existing_edge, "margin", 0.0))
                    if existing_strength >= float(posterior.margin) and not signal.get("selected", False):
                        continue

            support_frames = cand.get("support_frames") or []
            self.graph.upsert_local_edge(
                parent_id,
                child_id,
                edge_type,
                relation_text=self._relation_text_for_instance_edge(parent_id, child_id, edge_type),
                committed_kf=-1,
                support_count=int(cand.get("support_count", 0) or 0),
                evidence_score=float(posterior.candidate_parent_scores.get(parent_id, 0.0)),
                status="tentative",
                first_seen_frame=int(support_frames[0]) if support_frames else int(frame_idx),
                last_seen_frame=int(support_frames[-1]) if support_frames else int(frame_idx),
                last_update_source="local_lowshot_unambiguous",
                margin=float(posterior.margin),
                latest_margin=float(posterior.latest_evidence_margin),
                recent_support_count=int(max(1, cand.get("recent_support_count", 0) or 0)),
                switch_count=int(cand.get("switch_count", 0) or 0),
                retention_policy=self.tentative_local_lowshot_retention_policy,
            )
            reason["created"] = True
            reason["last_update_source"] = "local_lowshot_unambiguous"
            reason["retention_policy"] = self.tentative_local_lowshot_retention_policy

    def update_local_posteriors(self, frame_idx: int, observations: list[NodeObservation], sam3_out) -> None:
        det = getattr(sam3_out, "det", None)
        stage = self._pick_local_stage(getattr(det, "local_rel_debug", {}) or {})
        edges = stage.get("edges", []) if isinstance(stage, dict) else []
        det_to_node = {obs.det_idx: obs.matched_node_id for obs in observations if obs.matched_node_id is not None}
        det_to_obs = {obs.det_idx: obs for obs in observations}
        self._latest_local_assignment_signals = {}
        self._latest_current_parent_claims = {}
        self._latest_current_parent_claims_frame_idx = int(frame_idx)

        candidate_edges: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        obs_by_node_id: dict[str, NodeObservation] = {}

        for edge in edges:
            child_idx = int(edge.get("child_idx", -1))
            parent_idx = int(edge.get("parent_idx", -1))
            child_obs = det_to_obs.get(child_idx)
            parent_obs = det_to_obs.get(parent_idx)
            child_id = det_to_node.get(child_idx)
            parent_id = det_to_node.get(parent_idx)
            if child_obs is None or parent_obs is None or child_id is None or parent_id is None or child_id == parent_id:
                continue

            child_track = self.node_tracks.get(child_id)
            parent_track = self.node_tracks.get(parent_id)
            if child_track is None or parent_track is None:
                continue
            obs_by_node_id[child_id] = child_obs
            if child_obs.role not in ("U", "C"):
                continue
            if child_obs.allowed_parent_labels and parent_track.label not in child_obs.allowed_parent_labels:
                continue
            if child_obs.role == "U" and edge.get("type") not in ("C-U", "O-U"):
                continue
            if child_obs.role == "C" and edge.get("type") != "O-C":
                continue
            if self._edge_type_for_roles(parent_track.role, child_track.role) is None:
                continue

            group = self._local_assignment_group(stage, child_idx)
            assignment = group.get(str(edge.get("type") or "")) if isinstance(group, dict) else None
            chosen_edge = self._chosen_edge_from_assignment(stage, child_idx, str(edge.get("type") or ""))
            assignment_strength = 0.0
            assignment_status = ""
            if isinstance(assignment, dict):
                assignment_status = str(assignment.get("status") or "")
                if chosen_edge is not None:
                    try:
                        if int(chosen_edge.get("parent_idx", -1)) == parent_idx:
                            assignment_strength = self._local_assignment_status_strength(assignment_status, chosen_edge)
                    except Exception:
                        assignment_strength = 0.0

            geom_consistency = self._parent_geom_consistency(child_obs, parent_obs)
            candidate_edges[child_id][parent_id] = {
                "edge": edge,
                "geom_consistency": geom_consistency,
                "assignment_strength": float(assignment_strength),
                "assignment_status": assignment_status,
            }

        evidence_by_child: dict[str, dict[str, float]] = defaultdict(dict)
        preferred_parent_ids_by_child: dict[str, set[str]] = defaultdict(set)
        fallback_parent_ids_by_child: dict[str, set[str]] = defaultdict(set)
        viable_preferred_ids_by_child: dict[str, set[str]] = defaultdict(set)
        for child_id, parent_entries in candidate_edges.items():
            child_obs = obs_by_node_id.get(child_id)
            if child_obs is None:
                continue

            if child_obs.role == "C":
                preferred_parent_ids_by_child[child_id].update(parent_entries.keys())

            # First pass: discover preferred/fallback sets and freeze viable preferred parents.
            for parent_id, info in parent_entries.items():
                parent_track = self.node_tracks.get(parent_id)
                if parent_track is None:
                    continue
                if child_obs.preferred_parent_labels and parent_track.label in child_obs.preferred_parent_labels:
                    preferred_parent_ids_by_child[child_id].add(parent_id)
                    posterior_state = self.local_posteriors.get(child_id)
                    if self.is_viable_preferred_parent(
                        child_obs=child_obs,
                        parent_track=parent_track,
                        edge_obs=info["edge"],
                        posterior_state=posterior_state,
                    ):
                        viable_preferred_ids_by_child[child_id].add(parent_id)
                if child_obs.fallback_parent_labels and parent_track.label in child_obs.fallback_parent_labels:
                    fallback_parent_ids_by_child[child_id].add(parent_id)

        # Second pass: score edges using the frozen preferred-parent sets.
        for child_id, parent_entries in candidate_edges.items():
            child_track = self.node_tracks.get(child_id)
            child_obs = obs_by_node_id.get(child_id)
            if child_track is None or child_obs is None:
                continue
            has_viable_preferred_parent = bool(viable_preferred_ids_by_child.get(child_id))
            for parent_id, info in parent_entries.items():
                parent_track = self.node_tracks.get(parent_id)
                if parent_track is None:
                    continue
                carrier_bypass_penalty = 0.0
                if child_track.role == "U" and parent_track.role == "O" and has_viable_preferred_parent:
                    carrier_bypass_penalty = 1.0
                owner_conflict_penalty = self._owner_conflict_penalty(child_id, parent_id)
                score = self._local_edge_score(
                    info["edge"],
                    geom_consistency=info["geom_consistency"],
                    owner_conflict_penalty=owner_conflict_penalty,
                    carrier_bypass_penalty=carrier_bypass_penalty,
                )
                evidence_by_child[child_id][parent_id] = max(evidence_by_child[child_id].get(parent_id, 0.0), score)
                edge_obs = info["edge"]
                pass_thr_val = bool(edge_obs.get("pass_thr"))
                selected_val = bool(edge_obs.get("selected"))
                mask_contain_val = float(edge_obs.get("mask_contain") or 0.0)
                contain_val = float(edge_obs.get("contain") or 0.0)
                assign_strength_val = float(info.get("assignment_strength", 0.0))
                # is_chosen_parent: assignment_strength is only non-zero for the
                # chosen parent of the assignment group (see candidate_edges build).
                is_chosen_parent = assign_strength_val > 0.0
                # Edge-specific strong-parent definition. Weak placeholder
                # candidates (selected=False, contain=0, pass_thr=False, not the
                # chosen parent) must NOT count as strong.
                strong_current = (
                    selected_val
                    or pass_thr_val
                    or mask_contain_val >= self.preferred_tau_mask
                    or contain_val >= self.preferred_tau_box
                    or (is_chosen_parent and assign_strength_val > 0.0)
                )
                # Edge-level evidence strength derived from per-edge 2D signals.
                # This is what the low-shot unambiguous gate compares against
                # its per-edge-type threshold. We deliberately do NOT rely
                # solely on assignment_strength here, because the SAM3
                # assignment dump may use eids that are not list indices,
                # which can occasionally make _chosen_edge_from_assignment
                # return None even when the per-edge 2D evidence is strong
                # (selected/pass_thr/contain/mask_contain). See edge_requirement.md.
                edge_evidence_strength = float(assign_strength_val)
                if selected_val or pass_thr_val:
                    edge_evidence_strength = max(edge_evidence_strength, 1.0)
                if mask_contain_val >= self.preferred_tau_mask:
                    edge_evidence_strength = max(edge_evidence_strength, 1.0)
                if contain_val >= self.preferred_tau_box:
                    edge_evidence_strength = max(edge_evidence_strength, 0.9)
                self._latest_local_assignment_signals.setdefault(child_id, {})[parent_id] = {
                    "strength": assign_strength_val,
                    "edge_evidence_strength": float(edge_evidence_strength),
                    "status": str(info.get("assignment_status") or ""),
                    "edge_type": str(edge_obs.get("type") or self._edge_type_for_roles(parent_track.role, child_track.role) or ""),
                    "child_role": str(child_track.role),
                    "parent_role": str(parent_track.role),
                    "child_label": str(child_track.label or ""),
                    "parent_label": str(parent_track.label or ""),
                    "selected": selected_val,
                    "pass_thr": pass_thr_val,
                    "is_chosen_parent": bool(is_chosen_parent),
                    "score": float(score),
                    "mask_contain": mask_contain_val,
                    "contain": contain_val,
                    "owner_conflict_penalty": float(owner_conflict_penalty),
                    "carrier_bypass_penalty": float(carrier_bypass_penalty),
                    "geom_consistency": float(info.get("geom_consistency", 0.0)),
                    "num_candidate_parents": int(len(parent_entries)),
                    # has_competition / num_strong_parents are filled in a
                    # post-pass below once strong_current is known for every
                    # parent of this child.
                    "has_competition": False,
                    "num_strong_parents": 0,
                    "strong_current": bool(strong_current),
                    "competition_basis": "num_strong_parents",
                    "role_compatible": self._edge_type_for_roles(parent_track.role, child_track.role) is not None,
                    "label_compatible": (not child_obs.allowed_parent_labels) or (parent_track.label in child_obs.allowed_parent_labels),
                    "frame_idx": int(frame_idx),
                }
                self._latest_current_parent_claims.setdefault(child_id, {})[parent_id] = {
                    "edge_type": str(edge_obs.get("type") or self._edge_type_for_roles(parent_track.role, child_track.role) or ""),
                    "selected": bool(edge_obs.get("selected")),
                    "assignment_status": str(info.get("assignment_status") or ""),
                    "assignment_strength": float(info.get("assignment_strength", 0.0)),
                    "strong_current": bool(strong_current),
                    "mask_contain": float(edge_obs.get("mask_contain") or 0.0),
                    "contain": float(edge_obs.get("contain") or 0.0),
                }

        # Per-child num_strong_parents back-fill: only true strong candidates
        # (selected / pass_thr / mask_contain / contain / chosen-parent) count
        # as competitors. Weak placeholder parents must NOT block low-shot
        # unambiguous edges.
        for child_id, parent_signals in self._latest_local_assignment_signals.items():
            num_strong = sum(
                1 for sig in parent_signals.values() if bool(sig.get("strong_current", False))
            )
            for sig in parent_signals.values():
                sig["num_strong_parents"] = int(num_strong)
                sig["has_competition"] = bool(num_strong > 1)

        for obs in observations:
            if obs.matched_node_id is None or obs.role not in ("U", "C"):
                continue
            child_id = obs.matched_node_id
            posterior = self.local_posteriors.setdefault(
                child_id,
                LocalParentPosterior(child_node_id=child_id, child_role=obs.role),
            )
            preferred_parent_ids = set(preferred_parent_ids_by_child.get(child_id, set())).intersection(evidence_by_child.get(child_id, {}).keys())
            fallback_parent_ids = set(fallback_parent_ids_by_child.get(child_id, set())).intersection(evidence_by_child.get(child_id, {}).keys())

            viable_preferred_ids = set(viable_preferred_ids_by_child.get(child_id, set()))
            if obs.role == "C":
                viable_preferred_ids = set(preferred_parent_ids)

            posterior.preferred_parent_ids = set(preferred_parent_ids)
            posterior.fallback_parent_ids = set(fallback_parent_ids)
            posterior.owner_mode = obs.semantic_owner_mode
            posterior.has_viable_preferred_parent = bool(viable_preferred_ids)

            if obs.role == "U" and obs.semantic_owner_mode == "prefer_carrier":
                if viable_preferred_ids:
                    active_parent_pool = viable_preferred_ids
                    posterior.drop_candidates(fallback_parent_ids)
                elif fallback_parent_ids:
                    active_parent_pool = fallback_parent_ids
                    posterior.drop_candidates(preferred_parent_ids)
                else:
                    active_parent_pool = preferred_parent_ids
            else:
                active_parent_pool = preferred_parent_ids if preferred_parent_ids else set(evidence_by_child.get(child_id, {}).keys())

            staged_evidence = {
                parent_id: score
                for parent_id, score in evidence_by_child.get(child_id, {}).items()
                if parent_id in active_parent_pool
            }
            posterior.update(
                frame_idx,
                staged_evidence,
                decay=self.local_decay,
                min_support_frames=self.local_min_support_frames,
                stable_margin=self.local_stable_margin,
                history_size=self.local_history_size,
                max_switches=self.local_max_switches,
                recent_consistency_frames=self.local_recent_consistency_frames,
            )
            if posterior.is_unresolved(
                margin_threshold=self.llava_margin_local,
                unresolved_frames=self.unresolved_frames_for_llava,
            ):
                self._submit_local_llava(child_id, posterior)
        # Competition ledger: any pair of parent ids that both appear in
        # the same child's posterior candidate list constitutes a
        # child_competition.  This drives competition-based duplicate
        # consolidation later in the frame.
        try:
            self._scan_local_posteriors_for_competition(int(frame_idx))
        except Exception:
            pass

    def _submit_remote_llava(self, relation_key: str, posterior: RemotePairPosterior) -> None:
        if posterior.llava_requested:
            return
        payload = {
            "relation_key": relation_key,
            "relation_text": posterior.relation_text,
            "src_label": posterior.src_label,
            "dst_label": posterior.dst_label,
            "top1_pair": list(posterior.top1_pair) if posterior.top1_pair is not None else None,
            "top2_pair": list(posterior.top2_pair) if posterior.top2_pair is not None else None,
            "best_view_frames": list(posterior.best_view_frames),
            "margin": posterior.margin,
        }
        self.llava_scheduler.submit_remote(relation_key, payload)
        posterior.llava_requested = True

    # ----------------------- Section D helpers -----------------------
    @staticmethod
    def _atlas_template_key(src_label: str, dst_label: str, relation_text: str) -> str:
        return f"{(src_label or '').strip().lower()}->{(dst_label or '').strip().lower()}:{(relation_text or '').strip().lower()}"

    def _update_remote_atlas_templates(self, frame_result: dict, frame_idx: int) -> None:
        """Section D2: extract remote-relation templates from
        ``frame_result`` (LLM/DeepSeek atlas / scene reasoning) and store
        them in ``self.remote_atlas_templates``.  Defensive: missing or
        malformed fields are silently skipped.
        """
        if not bool(getattr(self, "remote_atlas_enable", True)):
            return
        if not isinstance(frame_result, dict):
            return
        # Source 1: ``remote_relation_candidates`` (already populated by
        # the semantic pipeline when LLM/SAM3 enumerate remote links).
        for item in frame_result.get("remote_relation_candidates") or []:
            if not isinstance(item, dict):
                continue
            self._record_atlas_template(
                src_label=item.get("from_object"),
                dst_label=item.get("to_object"),
                relation_text=item.get("relation"),
                directional=bool(item.get("directional", True)),
                frame_idx=frame_idx,
                roles=item.get("roles"),
                distance_prior=item.get("distance_prior"),
            )
        # Source 2: ``scene_atlas`` / ``atlas`` blob.  Defensive: only
        # require ``remote_relations`` to be a list of dicts.
        for blob_key in ("scene_atlas", "atlas", "remote_atlas"):
            blob = frame_result.get(blob_key)
            if not isinstance(blob, dict):
                continue
            for item in blob.get("remote_relations") or []:
                if not isinstance(item, dict):
                    continue
                self._record_atlas_template(
                    src_label=item.get("src_label") or item.get("from_object"),
                    dst_label=item.get("dst_label") or item.get("to_object"),
                    relation_text=item.get("relation_text") or item.get("relation"),
                    directional=bool(item.get("directional", True)),
                    frame_idx=frame_idx,
                    roles=item.get("roles"),
                    distance_prior=item.get("distance_prior"),
                )

    def _record_atlas_template(
        self,
        *,
        src_label,
        dst_label,
        relation_text,
        directional: bool,
        frame_idx: int,
        roles=None,
        distance_prior=None,
    ) -> None:
        if not src_label or not dst_label or not relation_text:
            return
        try:
            src_l = str(src_label).strip().lower()
            dst_l = str(dst_label).strip().lower()
            rel_t = str(relation_text).strip().lower()
        except Exception:
            return
        if not src_l or not dst_l or not rel_t:
            return
        if self.graph_policy.should_block_remote_edge(src_l, dst_l):
            self.remote_atlas_summary["templates_blocked_by_policy"] = int(
                self.remote_atlas_summary.get("templates_blocked_by_policy", 0)
            ) + 1
            return
        key = self._atlas_template_key(src_l, dst_l, rel_t)
        tmpl = self.remote_atlas_templates.get(key)
        cap = int(self.remote_atlas_template_source_frames_cap or 8)
        if tmpl is None:
            if len(self.remote_atlas_templates) >= int(self.remote_atlas_template_max or 1024):
                # Bounded buffer: drop the oldest template by last_seen_frame.
                try:
                    oldest = min(
                        self.remote_atlas_templates.items(),
                        key=lambda kv: int(kv[1].get("last_seen_frame", 0) or 0),
                    )
                    self.remote_atlas_templates.pop(oldest[0], None)
                except ValueError:
                    pass
            tmpl = {
                "src_label": src_l,
                "dst_label": dst_l,
                "relation_text": rel_t,
                "directional": bool(directional),
                "source_frames": [int(frame_idx)],
                "support_count": 1,
                "source": "atlas",
                "last_seen_frame": int(frame_idx),
                "roles": list(roles) if isinstance(roles, (list, tuple)) else None,
                "distance_prior": dict(distance_prior) if isinstance(distance_prior, dict) else None,
            }
            self.remote_atlas_templates[key] = tmpl
        else:
            tmpl["support_count"] = int(tmpl.get("support_count", 0)) + 1
            sf = list(tmpl.get("source_frames") or [])
            if not sf or sf[-1] != int(frame_idx):
                sf.append(int(frame_idx))
                if len(sf) > cap:
                    del sf[: len(sf) - cap]
                tmpl["source_frames"] = sf
            tmpl["last_seen_frame"] = int(frame_idx)
            if isinstance(roles, (list, tuple)) and not tmpl.get("roles"):
                tmpl["roles"] = list(roles)
            if isinstance(distance_prior, dict) and not tmpl.get("distance_prior"):
                tmpl["distance_prior"] = dict(distance_prior)
        self.remote_atlas_summary["templates_seen"] = int(
            self.remote_atlas_summary.get("templates_seen", 0)
        ) + 1

    def _atlas_label_match(self, track_label: str, target_label: str) -> bool:
        if not track_label or not target_label:
            return False
        try:
            tl = str(track_label).strip().lower()
            ml = str(target_label).strip().lower()
        except Exception:
            return False
        if not tl or not ml:
            return False
        if tl == ml:
            return True
        # Light substring fallback so trivial label variants still match
        # ("microwave" ~ "microwave oven").  Conservative: require at
        # least 4 chars of overlap to avoid spurious matches.
        if len(ml) >= 4 and (ml in tl or tl in ml):
            return True
        return False

    def _track_centroid_world(self, track) -> Optional[list[float]]:
        if track is None:
            return None
        c = getattr(track, "centroid_world_est", None)
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            try:
                return [float(c[0]), float(c[1]), float(c[2])]
            except Exception:
                return None
        return None

    def _atlas_spatial_score(
        self,
        src_track,
        dst_track,
        template: dict,
    ) -> tuple[float, dict]:
        """Combine 3D distance, distance prior, recent co-visibility,
        endpoint maturity and best-view quality into a [0,1] score.
        """
        debug: dict = {}
        ca = self._track_centroid_world(src_track)
        cb = self._track_centroid_world(dst_track)
        if ca is None or cb is None:
            return 0.0, {"reason": "no_centroid"}
        d = (
            (ca[0] - cb[0]) ** 2
            + (ca[1] - cb[1]) ** 2
            + (ca[2] - cb[2]) ** 2
        ) ** 0.5
        debug["distance_m"] = float(d)
        prior = template.get("distance_prior") or {}
        prior_min = float(prior.get("min_m", 0.0) or 0.0)
        prior_max = float(prior.get("max_m", self.remote_atlas_distance_scale_default) or self.remote_atlas_distance_scale_default)
        prior_max = max(prior_max, prior_min + 0.1)
        debug["distance_prior"] = {"min_m": prior_min, "max_m": prior_max}
        # 1.0 inside [min,max], decays linearly to 0 at 2*max.
        if d <= prior_max:
            d_score = 1.0 if d >= prior_min else max(0.0, d / max(prior_min, 1e-3))
        else:
            d_score = max(0.0, 1.0 - (d - prior_max) / max(prior_max, 1e-3))
        debug["distance_score"] = float(d_score)
        # Recent co-visibility bonus: shared frames in the last 64
        # observed-frames window are a strong positive signal.
        try:
            fa = list(getattr(src_track, "observed_frames", []) or [])[-64:]
            fb = list(getattr(dst_track, "observed_frames", []) or [])[-64:]
        except Exception:
            fa, fb = [], []
        recent_covis = len(set(fa) & set(fb))
        debug["recent_covisibility"] = int(recent_covis)
        covis_score = min(1.0, float(recent_covis) / 4.0)
        # Endpoint maturity (re-use existing helper).
        try:
            mat_a = float(self._node_endpoint_maturity_score(getattr(src_track, "node_id", "")))
        except Exception:
            mat_a = 0.0
        try:
            mat_b = float(self._node_endpoint_maturity_score(getattr(dst_track, "node_id", "")))
        except Exception:
            mat_b = 0.0
        endpoint_maturity = 0.5 * mat_a + 0.5 * mat_b
        debug["endpoint_maturity"] = float(endpoint_maturity)
        debug["maturity_a"] = float(mat_a)
        debug["maturity_b"] = float(mat_b)
        # Best-view quality bonus.
        try:
            bvs_a = float(getattr(src_track, "best_view_score", 0.0) or 0.0)
            bvs_b = float(getattr(dst_track, "best_view_score", 0.0) or 0.0)
        except Exception:
            bvs_a, bvs_b = 0.0, 0.0
        bvs = min(1.0, 0.5 * bvs_a + 0.5 * bvs_b)
        debug["best_view_score"] = float(bvs)
        # Combine: distance dominates (0.55), then maturity (0.20),
        # covis (0.15), best_view (0.10).
        score = (
            0.55 * d_score
            + 0.20 * endpoint_maturity
            + 0.15 * covis_score
            + 0.10 * bvs
        )
        debug["score"] = float(max(0.0, min(1.0, score)))
        return float(max(0.0, min(1.0, score))), debug

    def _generate_remote_atlas_candidates(self, frame_idx: int) -> list[dict]:
        """Section D3: scan templates × node_tracks and emit remote
        candidates that pass the configured maturity / spatial gates.
        Bounded by ``remote_atlas_max_candidates_per_kf``.
        """
        if not bool(getattr(self, "remote_atlas_enable", True)):
            return []
        if not self.remote_atlas_templates:
            return []
        candidates: list[dict] = []
        min_support = int(self.remote_atlas_min_template_support or 1)
        min_maturity = float(self.remote_atlas_min_endpoint_maturity or 0.35)
        min_score = float(self.remote_atlas_min_spatial_score or 0.35)
        write_min = float(self.remote_atlas_write_min_score or 0.55)
        llava_min = float(self.remote_atlas_llava_min_score or 0.35)
        max_cands = int(self.remote_atlas_max_candidates_per_kf or 5)
        # Pre-build label-> [node_id] index for speed.
        label_index: dict[str, list[str]] = {}
        for nid, track in self.node_tracks.items():
            lab = str(getattr(track, "label", "") or "").strip().lower()
            if lab:
                label_index.setdefault(lab, []).append(nid)
        for tmpl_key, tmpl in self.remote_atlas_templates.items():
            if int(tmpl.get("support_count", 0) or 0) < min_support:
                continue
            src_l = str(tmpl.get("src_label") or "")
            dst_l = str(tmpl.get("dst_label") or "")
            rel_t = str(tmpl.get("relation_text") or "")
            if not src_l or not dst_l or not rel_t:
                continue
            if self.graph_policy.should_block_remote_edge(src_l, dst_l):
                continue
            # Match: prefer exact label, fall back to substring.
            srcs = label_index.get(src_l) or [
                nid for nid, t in self.node_tracks.items()
                if self._atlas_label_match(getattr(t, "label", ""), src_l)
            ]
            dsts = label_index.get(dst_l) or [
                nid for nid, t in self.node_tracks.items()
                if self._atlas_label_match(getattr(t, "label", ""), dst_l)
            ]
            for sa in srcs:
                ta = self.node_tracks.get(sa)
                if ta is None:
                    continue
                for sb in dsts:
                    if sa == sb:
                        continue
                    tb = self.node_tracks.get(sb)
                    if tb is None:
                        continue
                    score, dbg = self._atlas_spatial_score(ta, tb, tmpl)
                    endpoint_maturity = float(dbg.get("endpoint_maturity", 0.0))
                    if endpoint_maturity < min_maturity:
                        self.remote_atlas_summary["candidates_skipped_low_maturity"] = int(
                            self.remote_atlas_summary.get("candidates_skipped_low_maturity", 0)
                        ) + 1
                        continue
                    if score < min_score:
                        self.remote_atlas_summary["candidates_skipped_low_score"] = int(
                            self.remote_atlas_summary.get("candidates_skipped_low_score", 0)
                        ) + 1
                        continue
                    relation_key = self._remote_relation_key(src_l, rel_t, dst_l)
                    candidate = {
                        "frame_idx": int(frame_idx),
                        "template_key": tmpl_key,
                        "relation_key": relation_key,
                        "relation_text": rel_t,
                        "src_node_id": sa,
                        "dst_node_id": sb,
                        "src_label": str(getattr(ta, "label", "") or ""),
                        "dst_label": str(getattr(tb, "label", "") or ""),
                        "atlas_support_count": int(tmpl.get("support_count", 0) or 0),
                        "spatial_score": float(score),
                        "endpoint_maturity": float(endpoint_maturity),
                        "recent_covisibility": int(dbg.get("recent_covisibility", 0)),
                        "needs_llava": bool(score < write_min and score >= llava_min),
                        "should_write_tentative": bool(score >= write_min),
                        "source": "atlas_spatial",
                        "spatial_debug": dbg,
                    }
                    candidates.append(candidate)
        # Sort by score descending and cap.
        candidates.sort(key=lambda c: -float(c.get("spatial_score", 0.0)))
        if len(candidates) > max_cands:
            candidates = candidates[:max_cands]
        self.remote_atlas_summary["candidates_generated"] = int(
            self.remote_atlas_summary.get("candidates_generated", 0)
        ) + len(candidates)
        # Buffer recent for snapshot inspection.
        for c in candidates:
            if len(self.remote_atlas_candidates_recent) < 1024:
                self.remote_atlas_candidates_recent.append(c)
        return candidates

    def _upsert_remote_candidate_edge(
        self,
        candidate: dict,
        frame_idx: int,
        *,
        source: str,
        status: str = "tentative",
    ) -> bool:
        """Section D4: upsert a tentative remote graph edge from an
        atlas-driven candidate.  Returns True on success.
        """
        relation_key = candidate.get("relation_key") or ""
        src = candidate.get("src_node_id") or ""
        dst = candidate.get("dst_node_id") or ""
        relation_text = candidate.get("relation_text") or ""
        if not relation_key or not src or not dst or not relation_text:
            return False
        if self.graph_policy.should_block_remote_edge(
            str(candidate.get("src_label") or ""),
            str(candidate.get("dst_label") or ""),
        ):
            return False
        src_track = self.node_tracks.get(src)
        dst_track = self.node_tracks.get(dst)
        if src_track is not None and dst_track is not None:
            if self.graph_policy.should_block_remote_edge(
                str(getattr(src_track, "label", "") or ""),
                str(getattr(dst_track, "label", "") or ""),
            ):
                return False
        try:
            graph_edge_key = self._remote_graph_edge_key(relation_key, src, dst)
        except Exception:
            return False
        try:
            self.graph.upsert_remote_edge(
                graph_edge_key,
                src,
                dst,
                relation_text,
                committed_kf=-1,
                support_count=int(candidate.get("atlas_support_count", 1) or 1),
                evidence_score=float(candidate.get("spatial_score", 0.0)),
                status=status,
                first_seen_frame=int(frame_idx),
                last_seen_frame=int(frame_idx),
                last_update_source=source,
                margin=float(candidate.get("spatial_score", 0.0)),
                latest_margin=float(candidate.get("spatial_score", 0.0)),
                recent_support_count=int(candidate.get("atlas_support_count", 1) or 1),
                switch_count=0,
                retention_policy="ttl",
            )
            self.remote_atlas_summary["candidates_written_tentative"] = int(
                self.remote_atlas_summary.get("candidates_written_tentative", 0)
            ) + 1
            return True
        except Exception:
            return False

    def _schedule_remote_atlas_llava_checks(
        self,
        candidates: list[dict],
        frame_idx: int,
    ) -> dict:
        """Section D5: throttle + dedupe + submit LLAVA verification jobs
        for atlas remote candidates whose score sits in the
        ``[llava_min, write_min)`` band.
        """
        summary = {"checked": 0, "scheduled": 0, "skipped": 0}
        if not bool(getattr(self, "remote_atlas_enable", True)):
            return summary
        every = max(1, int(self.remote_atlas_llava_every_kf or 1))
        max_per_round = max(0, int(self.remote_atlas_llava_max_per_round or 0))
        if max_per_round == 0:
            return summary
        last = int(getattr(self, "_remote_atlas_llava_last_kf", -1))
        if last >= 0 and (int(frame_idx) - last) < every:
            return summary
        scheduled = 0
        for cand in candidates:
            if scheduled >= max_per_round:
                break
            score = float(cand.get("spatial_score", 0.0))
            if score >= float(self.remote_atlas_write_min_score or 0.55):
                continue  # already written tentative; no need to verify.
            if score < float(self.remote_atlas_llava_min_score or 0.35):
                continue
            relation_key = str(cand.get("relation_key") or "")
            src = str(cand.get("src_node_id") or "")
            dst = str(cand.get("dst_node_id") or "")
            if not relation_key or not src or not dst:
                continue
            dedupe_key = f"atlas_remote:{relation_key}:{src}:{dst}"
            summary["checked"] += 1
            event: dict = {
                "frame_idx": int(frame_idx),
                "relation_key": relation_key,
                "relation_text": cand.get("relation_text"),
                "src_node_id": src,
                "dst_node_id": dst,
                "spatial_score": float(score),
                "dedupe_key": dedupe_key,
                "scheduled": False,
                "reason": "",
            }
            if dedupe_key in self._remote_atlas_llava_dedupe:
                event["reason"] = "dedup"
                summary["skipped"] += 1
            else:
                src_track = self.node_tracks.get(src)
                dst_track = self.node_tracks.get(dst)
                payload = {
                    "relation_key": relation_key,
                    "relation_text": cand.get("relation_text"),
                    "src_label": cand.get("src_label"),
                    "dst_label": cand.get("dst_label"),
                    "src_node_id": src,
                    "dst_node_id": dst,
                    "src_best_view_frame": getattr(src_track, "best_view_frame", None),
                    "dst_best_view_frame": getattr(dst_track, "best_view_frame", None),
                    "spatial_debug": cand.get("spatial_debug"),
                    "source": "atlas_spatial",
                }
                try:
                    self.llava_scheduler.submit(
                        "remote_atlas", dedupe_key, payload
                    )
                    self._remote_atlas_llava_dedupe.add(dedupe_key)
                    event["scheduled"] = True
                    event["reason"] = "submitted"
                    summary["scheduled"] += 1
                    scheduled += 1
                    self.remote_atlas_summary["candidates_queued_llava"] = int(
                        self.remote_atlas_summary.get("candidates_queued_llava", 0)
                    ) + 1
                except Exception as exc:  # pragma: no cover - defensive
                    event["reason"] = f"submit_error:{type(exc).__name__}"
                    summary["skipped"] += 1
            if len(self.remote_atlas_llava_events) < 4096:
                self.remote_atlas_llava_events.append(event)
        if scheduled > 0:
            self._remote_atlas_llava_last_kf = int(frame_idx)
        return summary

    def _process_remote_atlas_pipeline(
        self, frame_result: dict, frame_idx: int
    ) -> dict:
        """One-shot: update templates, generate candidates, write
        tentative edges for high-score candidates, queue LLAVA for
        mid-score candidates.  Returns a small per-call summary.
        """
        summary = {
            "templates": len(self.remote_atlas_templates),
            "candidates": 0,
            "written_tentative": 0,
            "llava": {"checked": 0, "scheduled": 0, "skipped": 0},
        }
        if not bool(getattr(self, "remote_atlas_enable", True)):
            return summary
        try:
            self._update_remote_atlas_templates(frame_result, frame_idx)
        except Exception:
            return summary
        candidates = self._generate_remote_atlas_candidates(frame_idx)
        summary["candidates"] = len(candidates)
        summary["templates"] = len(self.remote_atlas_templates)
        for cand in candidates:
            if bool(cand.get("should_write_tentative")):
                ok = self._upsert_remote_candidate_edge(
                    cand, frame_idx, source="remote_atlas_spatial"
                )
                if ok:
                    summary["written_tentative"] += 1
        summary["llava"] = self._schedule_remote_atlas_llava_checks(candidates, frame_idx)
        return summary

    def _should_replace_tentative_remote_edge(self, existing_edge, *, pair: tuple[str, str], support_count: int, margin: float) -> bool:
        if existing_edge is None or getattr(existing_edge, "status", "committed") != "tentative":
            return True
        if existing_edge.src_node_id == pair[0] and existing_edge.dst_node_id == pair[1]:
            return True
        if support_count > int(getattr(existing_edge, "support_count", 0)):
            return True
        return float(margin) >= float(getattr(existing_edge, "margin", 0.0)) + self.tentative_remote_replace_margin

    def _node_endpoint_maturity_score(self, node_id: str) -> float:
        """Maturity score in [0,1] for a remote-edge endpoint.

        Combines obs_count, observed_frames span and a quality bonus when
        the node already participates in a committed local edge.  Used by
        the conservative tentative remote path.
        """
        track = self.node_tracks.get(node_id)
        if track is None:
            return 0.0
        obs_count = int(getattr(track, "obs_count", 0) or 0)
        frames = list(getattr(track, "observed_frames", []) or [])
        # 1.0 at obs_count >= 6
        obs_score = min(1.0, obs_count / 6.0)
        if frames:
            span = float(max(frames) - min(frames)) if len(frames) >= 2 else 0.0
            span_score = min(1.0, span / 8.0)
        else:
            span_score = 0.0
        committed_bonus = 0.2 if self._has_any_committed_edge(node_id) else 0.0
        return float(min(1.0, 0.55 * obs_score + 0.35 * span_score + committed_bonus))

    def _remote_pair_recent_covisibility(
        self, src_id: str, dst_id: str, *, window: int
    ) -> int:
        a = self.node_tracks.get(src_id)
        b = self.node_tracks.get(dst_id)
        if a is None or b is None:
            return 0
        fa = set(int(f) for f in (getattr(a, "observed_frames", []) or []))
        fb = set(int(f) for f in (getattr(b, "observed_frames", []) or []))
        latest = max((int(f) for f in self._frame_order), default=0)
        cutoff = latest - int(max(0, window))
        shared = [f for f in (fa & fb) if f >= cutoff]
        return int(len(shared))

    def _remote_pair_distance_score(self, src_id: str, dst_id: str) -> float:
        """Score in [0,1] favouring pairs that are within a plausible
        spatial range for a relation (not on top of each other, not
        absurdly far).  Returns 0.5 when bbox info is missing.
        """
        ta = self.node_tracks.get(src_id)
        tb = self.node_tracks.get(dst_id)
        if ta is None or tb is None:
            return 0.0
        d = float(self._centroid_distance(ta, tb))
        if d <= 0.0:
            return 0.5
        # Prefer 0.05m-3.0m; decay outside this range.
        if d < 0.05:
            return 0.4
        if d <= 3.0:
            return 1.0 - 0.5 * (d / 3.0)  # gentle 1.0 -> 0.5 range
        if d <= 6.0:
            return max(0.0, 0.5 - 0.5 * ((d - 3.0) / 3.0))
        return 0.0

    def _is_tentative_remote_pair_ready(
        self,
        relation_key: str,
        posterior,
        pair: tuple[str, str],
        frame_idx: int,
    ) -> tuple[bool, dict]:
        debug: dict = {
            "relation_key": relation_key,
            "pair": list(pair),
            "frame_idx": int(frame_idx),
        }
        support_count = int(posterior.support_count(pair))
        switch_count = int(posterior.switch_count(window=6))
        margin = float(posterior.margin)
        debug["support_count"] = support_count
        debug["switch_count"] = switch_count
        debug["margin"] = margin
        if support_count < int(self.remote_tentative_min_support_frames):
            debug["reject_reason"] = "support_too_small"
            return False, debug
        if switch_count > 2:
            debug["reject_reason"] = "too_many_switches"
            return False, debug
        if margin < float(self.remote_tentative_min_margin):
            debug["reject_reason"] = "margin_too_small"
            return False, debug
        src_id, dst_id = pair
        if src_id not in self.node_tracks or dst_id not in self.node_tracks:
            debug["reject_reason"] = "missing_endpoint"
            return False, debug
        mat_src = self._node_endpoint_maturity_score(src_id)
        mat_dst = self._node_endpoint_maturity_score(dst_id)
        debug["maturity_src"] = mat_src
        debug["maturity_dst"] = mat_dst
        if min(mat_src, mat_dst) < float(self.remote_tentative_min_endpoint_maturity):
            debug["reject_reason"] = "endpoint_maturity_too_low"
            return False, debug
        dscore = self._remote_pair_distance_score(src_id, dst_id)
        debug["distance_score"] = dscore
        if dscore < float(self.remote_tentative_min_distance_score):
            debug["reject_reason"] = "distance_score_too_low"
            return False, debug
        if self.remote_tentative_require_recent_covisibility:
            covis = self._remote_pair_recent_covisibility(
                src_id, dst_id,
                window=int(self.remote_tentative_recent_covis_window),
            )
            debug["recent_covisibility"] = covis
            if covis < 1:
                debug["reject_reason"] = "no_recent_covisibility"
                return False, debug
        debug["accept"] = True
        return True, debug

    def _refresh_tentative_remote_edges(self, frame_idx: int) -> None:
        # Reset per-frame summary so callers can introspect the latest pass.
        summ = self.remote_tentative_summary
        for k in (
            "considered", "accepted", "rejected",
            "rejected_support_too_small", "rejected_too_many_switches",
            "rejected_margin_too_small", "rejected_missing_endpoint",
            "rejected_endpoint_maturity_too_low",
            "rejected_distance_score_too_low", "rejected_no_recent_covisibility",
            "skipped_committed_existing", "skipped_blocked_by_policy", "tentative_replaced",
        ):
            summ[k] = 0
        if not self.remote_tentative_enable:
            return
        for relation_key, posterior in self.remote_posteriors.items():
            if self.graph_policy.should_block_remote_edge(posterior.src_label, posterior.dst_label):
                summ["skipped_blocked_by_policy"] = int(summ.get("skipped_blocked_by_policy", 0)) + 1
                continue
            pair = posterior.top1_pair
            if pair is None:
                continue
            summ["considered"] = int(summ.get("considered", 0)) + 1
            ready, dbg = self._is_tentative_remote_pair_ready(
                relation_key, posterior, pair, frame_idx
            )
            if not ready:
                summ["rejected"] = int(summ.get("rejected", 0)) + 1
                rk = "rejected_" + str(dbg.get("reject_reason", "unknown"))
                summ[rk] = int(summ.get(rk, 0)) + 1
                if len(self.remote_tentative_events) < 4096:
                    self.remote_tentative_events.append({
                        "frame_idx": int(frame_idx),
                        "action": "reject",
                        "relation_key": relation_key,
                        "pair": list(pair),
                        "debug": dbg,
                    })
                continue
            graph_edge_key = self._remote_graph_edge_key(relation_key, pair[0], pair[1])
            existing_edge = self.graph.remote_edges.get(graph_edge_key)
            # Conservative: never overwrite a committed remote edge.
            if existing_edge is not None and str(getattr(existing_edge, "status", "")).lower() == "committed":
                summ["skipped_committed_existing"] = int(summ.get("skipped_committed_existing", 0)) + 1
                if len(self.remote_tentative_events) < 4096:
                    self.remote_tentative_events.append({
                        "frame_idx": int(frame_idx),
                        "action": "skip_committed",
                        "relation_key": relation_key,
                        "graph_edge_key": graph_edge_key,
                        "pair": list(pair),
                        "debug": dbg,
                    })
                continue
            support_count = int(posterior.support_count(pair))
            if not self._should_replace_tentative_remote_edge(
                existing_edge,
                pair=pair,
                support_count=support_count,
                margin=posterior.margin,
            ):
                if len(self.remote_tentative_events) < 4096:
                    self.remote_tentative_events.append({
                        "frame_idx": int(frame_idx),
                        "action": "skip_no_replace",
                        "relation_key": relation_key,
                        "graph_edge_key": graph_edge_key,
                        "pair": list(pair),
                        "debug": dbg,
                    })
                continue
            support_frames = posterior.support_frames.get(pair, [])
            self.graph.upsert_remote_edge(
                graph_edge_key,
                pair[0],
                pair[1],
                posterior.relation_text,
                committed_kf=-1,
                support_count=support_count,
                evidence_score=float(posterior.candidate_pairs.get(pair, 0.0)),
                status="tentative",
                first_seen_frame=int(support_frames[0]) if support_frames else int(frame_idx),
                last_seen_frame=int(support_frames[-1]) if support_frames else int(frame_idx),
                last_update_source="remote_tentative_top1",
                margin=float(posterior.margin),
                latest_margin=float(posterior.margin),
                recent_support_count=support_count,
                switch_count=int(posterior.switch_count(window=6)),
                retention_policy="ttl",
            )
            summ["accepted"] = int(summ.get("accepted", 0)) + 1
            if existing_edge is not None:
                summ["tentative_replaced"] = int(summ.get("tentative_replaced", 0)) + 1
            if len(self.remote_tentative_events) < 4096:
                self.remote_tentative_events.append({
                    "frame_idx": int(frame_idx),
                    "action": "upsert",
                    "relation_key": relation_key,
                    "graph_edge_key": graph_edge_key,
                    "pair": list(pair),
                    "debug": dbg,
                })

    # ---------------------- Section C helpers ----------------------
    def _record_remote_2d_evidence(
        self,
        frame_idx: int,
        item: dict,
        det_to_node: dict,
    ) -> None:
        """Append a 2D-observed remote relation to the backlog ledger.

        ``item`` mirrors the ``observed_candidates`` payload coming out of
        SAM3 (``from_det_idx``/``to_det_idx``/``from_object``/``to_object``/
        ``relation``/optional ``score``/optional bbox info).  We always
        record ``src_label``/``dst_label``/``relation_text`` plus the
        currently associated node ids (which may be ``None`` if the
        endpoint detection has not been matched yet).  Resolution can
        happen on this same frame (``resolved=True``) or later via
        :func:`_resolve_remote_2d_evidence_backlog`.
        """
        try:
            src_label = normalize_label(item.get("from_object"))
            dst_label = normalize_label(item.get("to_object"))
            relation_text = str(item.get("relation") or "").strip()
        except Exception:
            return
        if not relation_text or not src_label or not dst_label:
            return
        if self.graph_policy.should_block_remote_edge(src_label, dst_label):
            return
        relation_key = self._remote_relation_key(src_label, relation_text, dst_label)
        try:
            src_det_idx_raw = item.get("from_det_idx", -1)
            dst_det_idx_raw = item.get("to_det_idx", -1)
            src_det_idx = int(src_det_idx_raw) if src_det_idx_raw is not None else -1
            dst_det_idx = int(dst_det_idx_raw) if dst_det_idx_raw is not None else -1
        except Exception:
            src_det_idx, dst_det_idx = -1, -1
        src_node_id = det_to_node.get(src_det_idx) if src_det_idx >= 0 else None
        dst_node_id = det_to_node.get(dst_det_idx) if dst_det_idx >= 0 else None

        # Defensive bbox extraction (may be absent in some pipelines).
        def _maybe_bbox(key: str):
            val = item.get(key)
            if isinstance(val, (list, tuple)) and len(val) >= 4:
                try:
                    return [float(v) for v in val[:4]]
                except Exception:
                    return None
            return None

        try:
            confidence = float(item.get("score", item.get("confidence", 0.0)) or 0.0)
        except Exception:
            confidence = 0.0

        evidence = {
            "frame_idx": int(frame_idx),
            "relation_key": relation_key,
            "relation_text": relation_text,
            "src_label": src_label,
            "dst_label": dst_label,
            "src_det_idx": int(src_det_idx),
            "dst_det_idx": int(dst_det_idx),
            "src_node_id": src_node_id,
            "dst_node_id": dst_node_id,
            "src_bbox_xyxy": _maybe_bbox("from_bbox") or _maybe_bbox("src_bbox") or _maybe_bbox("from_bbox_xyxy"),
            "dst_bbox_xyxy": _maybe_bbox("to_bbox") or _maybe_bbox("dst_bbox") or _maybe_bbox("to_bbox_xyxy"),
            "confidence": confidence,
            "source": "2d_remote_relation",
            "resolved": bool(src_node_id is not None and dst_node_id is not None),
            "last_resolve_attempt_frame": int(frame_idx),
        }
        bucket = self.remote_2d_evidence_ledger.setdefault(relation_key, [])
        bucket.append(evidence)
        if len(bucket) > self.remote_2d_evidence_max_per_key:
            del bucket[: len(bucket) - self.remote_2d_evidence_max_per_key]
        self.remote_2d_backlog_summary["remote_2d_evidence_seen"] = int(
            self.remote_2d_backlog_summary.get("remote_2d_evidence_seen", 0)
        ) + 1
        # Global cap to bound memory.
        total = sum(len(b) for b in self.remote_2d_evidence_ledger.values())
        if total > self.remote_2d_evidence_global_cap:
            # Drop the oldest entry in the largest bucket.
            largest_key = max(
                self.remote_2d_evidence_ledger.keys(),
                key=lambda k: len(self.remote_2d_evidence_ledger[k]),
            )
            self.remote_2d_evidence_ledger[largest_key].pop(0)

    def _node_track_label_match(self, track, label: str) -> bool:
        if track is None:
            return False
        track_label = str(getattr(track, "label", "") or "")
        return normalize_label(track_label) == normalize_label(label)

    def _is_track_mature_enough_for_backlog(self, track) -> bool:
        if track is None:
            return False
        if bool(getattr(track, "stable_geom_ready", False)):
            return True
        return int(getattr(track, "obs_count", 0) or 0) >= 2

    def _resolve_remote_2d_evidence_backlog(self, frame_idx: int) -> dict:
        """Walk the ledger and try to convert unresolved evidence into
        posterior updates.

        Returns a small per-call summary; aggregate counters live on
        ``self.remote_2d_backlog_summary``.
        """
        out = {"resolved": 0, "ambiguous": 0, "expired": 0, "written": 0}
        if not self.remote_2d_backlog_enable:
            self.remote_2d_backlog_summary["remote_2d_backlog_size"] = sum(
                len(b) for b in self.remote_2d_evidence_ledger.values()
            )
            return out
        ttl = int(self.remote_2d_evidence_ttl_frames)
        cutoff = int(frame_idx) - ttl

        def _emit_event(action: str, evidence: dict, reason: str = "") -> None:
            if len(self.remote_2d_backlog_events) >= 2048:
                return
            self.remote_2d_backlog_events.append({
                "frame_idx": int(frame_idx),
                "action": action,
                "relation_key": evidence.get("relation_key"),
                "evidence_frame_idx": int(evidence.get("frame_idx", -1)),
                "src_label": evidence.get("src_label"),
                "dst_label": evidence.get("dst_label"),
                "src_node_id": evidence.get("src_node_id"),
                "dst_node_id": evidence.get("dst_node_id"),
                "reason": reason,
            })

        for relation_key, bucket in list(self.remote_2d_evidence_ledger.items()):
            if not bucket:
                continue
            kept: list[dict] = []
            for evidence in bucket:
                if evidence.get("resolved"):
                    kept.append(evidence)
                    continue
                if int(evidence.get("frame_idx", 0)) < cutoff:
                    out["expired"] += 1
                    self.remote_2d_backlog_summary["remote_2d_backlog_expired"] = int(
                        self.remote_2d_backlog_summary.get("remote_2d_backlog_expired", 0)
                    ) + 1
                    _emit_event("expired", evidence, reason="ttl_exceeded")
                    continue

                src_id = evidence.get("src_node_id")
                dst_id = evidence.get("dst_node_id")
                src_label = evidence.get("src_label", "")
                dst_label = evidence.get("dst_label", "")
                if self.graph_policy.should_block_remote_edge(src_label, dst_label):
                    out["expired"] += 1
                    _emit_event("expired", evidence, reason="blocked_by_policy")
                    continue

                # If ids were captured but the underlying nodes have been
                # merged or removed, drop them and fall back to the label
                # search below.
                if src_id is not None and src_id not in self.node_tracks:
                    src_id = None
                if dst_id is not None and dst_id not in self.node_tracks:
                    dst_id = None

                src_candidates = [src_id] if src_id is not None else [
                    nid for nid, tr in self.node_tracks.items()
                    if self._node_track_label_match(tr, src_label)
                    and self._is_track_mature_enough_for_backlog(tr)
                ]
                dst_candidates = [dst_id] if dst_id is not None else [
                    nid for nid, tr in self.node_tracks.items()
                    if self._node_track_label_match(tr, dst_label)
                    and self._is_track_mature_enough_for_backlog(tr)
                ]

                # Eliminate self-loops and require disjoint endpoints.
                src_candidates = [s for s in src_candidates if s is not None]
                dst_candidates = [d for d in dst_candidates if d is not None]
                pairs = [(s, d) for s in src_candidates for d in dst_candidates if s != d]
                if not pairs:
                    evidence["last_resolve_attempt_frame"] = int(frame_idx)
                    kept.append(evidence)
                    continue
                if len(pairs) > 1:
                    out["ambiguous"] += 1
                    self.remote_2d_backlog_summary["remote_2d_backlog_ambiguous"] = int(
                        self.remote_2d_backlog_summary.get("remote_2d_backlog_ambiguous", 0)
                    ) + 1
                    evidence["last_resolve_attempt_frame"] = int(frame_idx)
                    _emit_event("ambiguous", evidence, reason="multiple_label_candidates")
                    kept.append(evidence)
                    continue

                # Unique pair found: inject into posterior.
                pair = pairs[0]
                relation_text = str(evidence.get("relation_text") or "")
                posterior = self.remote_posteriors.setdefault(
                    relation_key,
                    RemotePairPosterior(
                        relation_key=relation_key,
                        relation_text=relation_text,
                        src_label=src_label,
                        dst_label=dst_label,
                    ),
                )
                # Inject without going through ``update`` so we do not
                # double-decay existing candidates and so the support frame
                # reflects the *original* observation, not the resolution
                # attempt.
                injected_score = max(float(evidence.get("confidence", 0.0)), 0.5)
                posterior.candidate_pairs[pair] = posterior.candidate_pairs.get(pair, 0.0) + injected_score
                frames = posterior.support_frames.setdefault(pair, [])
                ev_frame = int(evidence.get("frame_idx", frame_idx))
                if not frames or frames[-1] != ev_frame:
                    frames.append(ev_frame)
                    frames.sort()
                # Re-rank top1/top2/margin.
                ranked = sorted(posterior.candidate_pairs.items(), key=lambda x: x[1], reverse=True)
                posterior.top1_pair = ranked[0][0] if ranked else None
                posterior.top2_pair = ranked[1][0] if len(ranked) > 1 else None
                top1 = ranked[0][1] if ranked else 0.0
                top2 = ranked[1][1] if len(ranked) > 1 else 0.0
                posterior.margin = float(top1 - top2)

                evidence["src_node_id"] = pair[0]
                evidence["dst_node_id"] = pair[1]
                evidence["resolved"] = True
                evidence["last_resolve_attempt_frame"] = int(frame_idx)
                out["resolved"] += 1
                self.remote_2d_backlog_summary["remote_2d_backlog_resolved"] = int(
                    self.remote_2d_backlog_summary.get("remote_2d_backlog_resolved", 0)
                ) + 1
                _emit_event("resolved", evidence, reason="unique_label_pair")
                kept.append(evidence)
            if kept:
                self.remote_2d_evidence_ledger[relation_key] = kept
            else:
                self.remote_2d_evidence_ledger.pop(relation_key, None)

        self.remote_2d_backlog_summary["remote_2d_backlog_size"] = sum(
            len(b) for b in self.remote_2d_evidence_ledger.values()
        )
        return out

    def update_remote_posteriors(self, frame_idx: int, observations: list[NodeObservation], sam3_out) -> None:
        remote_rel_2d = getattr(sam3_out, "remote_rel_2d", None) or {}
        observed_candidates = remote_rel_2d.get("observed_candidates", []) or []
        det_to_node = {obs.det_idx: obs.matched_node_id for obs in observations if obs.matched_node_id is not None}
        obs_by_node_id = {obs.matched_node_id: obs for obs in observations if obs.matched_node_id is not None}

        # Record every 2D candidate (resolved or not) into the backlog
        # ledger so unresolvable evidence can be retried in later frames.
        if self.remote_2d_backlog_enable:
            for item in observed_candidates:
                self._record_remote_2d_evidence(frame_idx, item, det_to_node)

        grouped_scores: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
        grouped_meta: dict[str, tuple[str, str, str]] = {}
        for item in observed_candidates:
            src_id = det_to_node.get(int(item.get("from_det_idx", -1)))
            dst_id = det_to_node.get(int(item.get("to_det_idx", -1)))
            if src_id is None or dst_id is None or src_id == dst_id:
                continue
            src_label = normalize_label(item.get("from_object"))
            dst_label = normalize_label(item.get("to_object"))
            relation_text = str(item.get("relation") or "").strip()
            if self.graph_policy.should_block_remote_edge(src_label, dst_label):
                continue
            relation_key = self._remote_relation_key(src_label, relation_text, dst_label)
            score_payload = self._score_remote_pair_candidate(
                item,
                src_id=src_id,
                dst_id=dst_id,
                obs_by_node_id=obs_by_node_id,
            )
            score = float(score_payload["total_score"])
            pair = (src_id, dst_id)
            grouped_scores[relation_key][pair] = max(grouped_scores[relation_key].get(pair, 0.0), score)
            grouped_meta[relation_key] = (relation_text, src_label, dst_label)

        for relation_key, pair_scores in grouped_scores.items():
            relation_text, src_label, dst_label = grouped_meta[relation_key]
            posterior = self.remote_posteriors.setdefault(
                relation_key,
                RemotePairPosterior(
                    relation_key=relation_key,
                    relation_text=relation_text,
                    src_label=src_label,
                    dst_label=dst_label,
                ),
            )
            posterior.update(
                frame_idx,
                pair_scores,
                decay=self.remote_decay,
                min_support_frames=self.remote_min_support_frames,
                stable_margin=self.remote_stable_margin,
            )
            if posterior.is_unresolved(
                margin_threshold=self.llava_margin_remote,
                unresolved_frames=self.unresolved_frames_for_llava,
            ):
                self._submit_remote_llava(relation_key, posterior)

    def _is_aggregate_cabinet_track(self, track: Optional[OnlineNodeTrack]) -> bool:
        if track is None:
            return False
        return str(getattr(track, "label", "") or "") == "cabinet" and str(getattr(track, "origin", "") or "") == "aggregate"

    def _is_aggregate_cabinet_node(self, node_id: Optional[str]) -> bool:
        if not node_id:
            return False
        if str(node_id).startswith("O_CABINET_"):
            return True
        return self._is_aggregate_cabinet_track(self.node_tracks.get(str(node_id)))

    def _is_non_cabinet_object_track(self, track: Optional[OnlineNodeTrack]) -> bool:
        if track is None or getattr(track, "role", None) != "O":
            return False
        if self._is_aggregate_cabinet_track(track):
            return False
        return not self.graph_policy.is_hint_only_object(str(getattr(track, "label", "") or ""))

    def _current_object_parent_of_carrier(self, node_id: str) -> Optional[str]:
        parent_candidate = self._graph_non_cabinet_object_parent_candidate(node_id)
        return None if parent_candidate is None else str(parent_candidate.get("parent_node_id"))

    def _graph_non_cabinet_object_parent_candidate(self, node_id: str) -> dict | None:
        for edge in self.graph.local_edges.values():
            if edge.edge_type != "O-C" or edge.dst_node_id != node_id:
                continue
            parent_track = self.node_tracks.get(edge.src_node_id)
            if not self._is_non_cabinet_object_track(parent_track):
                continue
            status = str(getattr(edge, "status", "") or "")
            return {
                "parent_node_id": edge.src_node_id,
                "parent_label": getattr(parent_track, "label", None),
                "parent_role": getattr(parent_track, "role", None),
                "source": "graph_committed_edge" if status == "committed" else "graph_tentative_edge",
                "support_count": int(getattr(edge, "support_count", 0)),
                "recent_support_count": int(getattr(edge, "recent_support_count", 0)),
                "switch_count": int(getattr(edge, "switch_count", 0)),
                "status": status,
                "edge_type": str(getattr(edge, "edge_type", "") or ""),
                "confidence": float(getattr(edge, "evidence_score", 0.0) or 0.0),
                "strong_current": False,
            }
        return None

    def _posterior_parent_is_confident(
        self,
        posterior: Optional[LocalParentPosterior],
        parent_id: Optional[str],
        *,
        source: str,
    ) -> dict | None:
        if posterior is None or parent_id is None:
            return None
        parent_track = self.node_tracks.get(parent_id)
        if not self._is_non_cabinet_object_track(parent_track):
            return None

        support_count = posterior.support_count(parent_id)
        recent_support_count = posterior.recent_support_count(parent_id, window=3)
        switch_count = posterior.recent_switch_count(window=4)
        min_support = max(2, self.local_min_support_frames - 1)
        if support_count < min_support or recent_support_count < 2 or switch_count > self.local_max_switches:
            return None

        if source == "stable":
            if posterior.stable_parent_id != parent_id:
                return None
            if posterior.top1_parent_id is not None and posterior.top1_parent_id != parent_id:
                return None
        else:
            if posterior.top1_parent_id != parent_id:
                return None
            if not (
                posterior.stable_parent_id == parent_id
                or posterior.top2_parent_id is None
                or posterior.margin >= self.tentative_margin_local
            ):
                return None

        return {
            "parent_node_id": parent_id,
            "parent_label": getattr(parent_track, "label", None),
            "parent_role": getattr(parent_track, "role", None),
            "source": f"posterior_{source}",
            "support_count": int(support_count),
            "recent_support_count": int(recent_support_count),
            "switch_count": int(switch_count),
            "status": "posterior",
            "edge_type": "O-C",
            "confidence": float(posterior.candidate_parent_scores.get(parent_id, 0.0) or 0.0),
            "strong_current": False,
        }

    def _node_level_non_cabinet_object_parent_claim_for_cabinet(self, node_id: str) -> dict | None:
        graph_candidate = self._graph_non_cabinet_object_parent_candidate(node_id)
        if graph_candidate is not None:
            return graph_candidate
        posterior = self.local_posteriors.get(node_id)
        stable_candidate = self._posterior_parent_is_confident(posterior, getattr(posterior, "stable_parent_id", None), source="stable")
        if stable_candidate is not None:
            return stable_candidate
        return self._posterior_parent_is_confident(posterior, getattr(posterior, "top1_parent_id", None), source="top1")

    def _observation_non_cabinet_object_parent_candidate(
        self,
        obs: Optional[NodeObservation],
        observations: list[NodeObservation],
        *,
        frame_idx: Optional[int] = None,
    ) -> dict | None:
        if obs is None or obs.matched_node_id is None or obs.role != "C":
            return None

        det_to_obs = {int(item.det_idx): item for item in observations}
        candidate_parent_obs: list[NodeObservation] = []
        seen_det_idxs: set[int] = set()

        strong_det_idxs = {int(det_idx) for det_idx in list(obs.strong_parent_det_idxs)}
        for det_idx in list(obs.strong_parent_det_idxs) + list(obs.local_parent_det_idxs):
            det_idx = int(det_idx)
            if det_idx == int(obs.det_idx) or det_idx in seen_det_idxs:
                continue
            seen_det_idxs.add(det_idx)
            parent_obs = det_to_obs.get(det_idx)
            if parent_obs is not None:
                candidate_parent_obs.append(parent_obs)

        for parent_obs in candidate_parent_obs:
            parent_node_id = parent_obs.matched_node_id
            if parent_node_id is None or parent_node_id == obs.matched_node_id:
                continue
            parent_track = self.node_tracks.get(parent_node_id)
            if not self._is_non_cabinet_object_track(parent_track):
                continue
            signal = {}
            if frame_idx is None or self._latest_current_parent_claims_frame_idx == int(frame_idx):
                signal = self._latest_current_parent_claims.get(obs.matched_node_id, {}).get(parent_node_id, {})
            status = str(signal.get("assignment_status") or "observation")
            selected = bool(signal.get("selected", False))
            strong_current = bool(
                selected
                or status in {"confirmed", "confirmed_mask"}
                or bool(signal.get("strong_current", False))
                or int(parent_obs.det_idx) in strong_det_idxs
            )
            if not strong_current:
                continue
            return {
                "parent_node_id": parent_node_id,
                "parent_label": getattr(parent_track, "label", None),
                "parent_role": getattr(parent_track, "role", None),
                "source": "observation_local_edge",
                "support_count": 0,
                "recent_support_count": 0,
                "switch_count": 0,
                "status": status,
                "parent_det_idx": int(parent_obs.det_idx),
                "edge_type": "O-C",
                "selected": selected,
                "strong_current": True,
                "confidence": float(signal.get("assignment_strength", 0.0) or 0.0),
            }
        return None

    def _current_signal_non_cabinet_object_parent_candidate(self, node_id: str, *, frame_idx: Optional[int] = None) -> dict | None:
        if frame_idx is not None and self._latest_current_parent_claims_frame_idx != int(frame_idx):
            return None
        for parent_id, signal in self._latest_current_parent_claims.get(node_id, {}).items():
            if str(signal.get("edge_type") or "") != "O-C":
                continue
            if not bool(signal.get("strong_current", False)):
                continue
            parent_track = self.node_tracks.get(parent_id)
            if not self._is_non_cabinet_object_track(parent_track):
                continue
            return {
                "parent_node_id": parent_id,
                "parent_label": getattr(parent_track, "label", None),
                "parent_role": getattr(parent_track, "role", None),
                "source": "current_strong_local_edge",
                "support_count": 0,
                "recent_support_count": 0,
                "switch_count": 0,
                "status": str(signal.get("assignment_status") or "current_strong"),
                "edge_type": str(signal.get("edge_type") or "O-C"),
                "selected": bool(signal.get("selected", False)),
                "strong_current": True,
                "mask_contain": float(signal.get("mask_contain", 0.0) or 0.0),
                "contain": float(signal.get("contain", 0.0) or 0.0),
                "assignment_strength": float(signal.get("assignment_strength", 0.0) or 0.0),
                "confidence": float(signal.get("assignment_strength", 0.0) or 0.0),
            }
        return None

    def _semantic_non_cabinet_object_parent_claim_for_cabinet(
        self,
        obs: Optional[NodeObservation],
        observations: list[NodeObservation],
    ) -> dict | None:
        if obs is None or obs.matched_node_id is None or obs.role != "C":
            return None
        labels: list[str] = []
        if str(obs.semantic_parent_role or "").strip().upper() == "O" and obs.semantic_parent_label:
            labels.append(obs.semantic_parent_label)
        if obs.semantic_owner_object_label:
            labels.append(obs.semantic_owner_object_label)
        if not labels:
            return None

        normalized_labels = [normalize_label(label) for label in labels if normalize_label(label)]
        if not normalized_labels:
            return None
        current_obs_by_label = {
            normalize_label(item.label): item
            for item in observations
            if item.role == "O" and item.matched_node_id is not None
        }
        for label in normalized_labels:
            parent_obs = current_obs_by_label.get(label)
            parent_id = None if parent_obs is None else parent_obs.matched_node_id
            if parent_id is None:
                for node_id, track in sorted(self.node_tracks.items()):
                    if normalize_label(getattr(track, "label", None)) == label and self._is_non_cabinet_object_track(track):
                        parent_id = node_id
                        break
            if parent_id is None or parent_id == obs.matched_node_id:
                continue
            parent_track = self.node_tracks.get(parent_id)
            if not self._is_non_cabinet_object_track(parent_track):
                continue
            return {
                "parent_node_id": parent_id,
                "parent_label": getattr(parent_track, "label", None),
                "parent_role": getattr(parent_track, "role", None),
                "source": "semantic_owner",
                "support_count": 0,
                "recent_support_count": 0,
                "switch_count": 0,
                "status": "semantic",
                "edge_type": "O-C",
                "confidence": float(getattr(obs, "semantic_subtype_conf", 0.0) or 0.0),
                "strong_current": False,
            }
        return None

    def _non_cabinet_object_parent_claim_for_cabinet_candidate(
        self,
        node_id: str,
        obs: Optional[NodeObservation],
        observations: Optional[list[NodeObservation]] = None,
        *,
        frame_idx: Optional[int] = None,
    ) -> dict | None:
        observations = observations or []
        observation_candidate = self._observation_non_cabinet_object_parent_candidate(obs, observations, frame_idx=frame_idx)
        if observation_candidate is not None:
            return observation_candidate
        current_candidate = self._current_signal_non_cabinet_object_parent_candidate(node_id, frame_idx=frame_idx)
        if current_candidate is not None:
            return current_candidate
        node_candidate = self._node_level_non_cabinet_object_parent_claim_for_cabinet(node_id)
        if node_candidate is not None:
            return node_candidate
        return self._semantic_non_cabinet_object_parent_claim_for_cabinet(obs, observations)

    def _is_available_for_cabinet_aggregation(
        self,
        node_id: str,
        obs: Optional[NodeObservation] = None,
        observations: Optional[list[NodeObservation]] = None,
        *,
        frame_idx: Optional[int] = None,
    ) -> dict:
        claim = self._non_cabinet_object_parent_claim_for_cabinet_candidate(
            node_id,
            obs,
            observations or [],
            frame_idx=frame_idx,
        )
        available = claim is None
        out = {
            "available": bool(available),
            "blocked": not bool(available),
            "reason": None if available else "skip_cabinet_due_to_non_cabinet_object_parent_claim",
            "parent_node_id": None,
            "parent_label": None,
            "parent_role": None,
            "source": None,
            "support": 0,
            "support_count": 0,
            "recent": 0,
            "recent_support_count": 0,
            "confidence": 0.0,
            "strong_current": False,
            "edge_type": None,
            "status": None,
        }
        if claim is not None:
            support = int(claim.get("support_count", 0) or 0)
            recent = int(claim.get("recent_support_count", 0) or 0)
            out.update(
                {
                    "parent_node_id": claim.get("parent_node_id"),
                    "parent_label": claim.get("parent_label"),
                    "parent_role": claim.get("parent_role"),
                    "source": claim.get("source"),
                    "support": support,
                    "support_count": support,
                    "recent": recent,
                    "recent_support_count": recent,
                    "confidence": float(claim.get("confidence", claim.get("assignment_strength", 0.0)) or 0.0),
                    "strong_current": bool(claim.get("strong_current", False)),
                    "edge_type": claim.get("edge_type"),
                    "status": claim.get("status"),
                }
            )
        return out

    def _carrier_is_available_for_cabinet(self, node_id: str) -> bool:
        return bool(self._is_available_for_cabinet_aggregation(node_id).get("available", True))

    def _prune_cabinet_members_with_non_cabinet_parent(
        self,
        frame_idx: int,
        observations: list[NodeObservation],
        *,
        reason: str,
    ) -> dict:
        member_obs_by_node_id = {
            str(obs.matched_node_id): obs
            for obs in observations
            if obs.matched_node_id is not None and obs.role == "C"
        }
        occupied_members: dict[str, dict] = {}
        for group in self.cabinet_aggregator.cabinet_tracks.values():
            for member_id in sorted(group.member_node_ids):
                availability = self._is_available_for_cabinet_aggregation(
                    member_id,
                    member_obs_by_node_id.get(member_id),
                    observations,
                    frame_idx=frame_idx,
                )
                if bool(availability.get("blocked", False)):
                    occupied_members[member_id] = availability

        if not occupied_members:
            return {"removed_members": [], "dropped_groups": []}

        prune_debug = self.cabinet_aggregator.prune_members(
            occupied_members.keys(),
            frame_idx=frame_idx,
            reason=reason,
            node_tracks=self.node_tracks,
        )
        removed_members = []
        for removed in prune_debug.get("removed_members", []):
            availability = occupied_members.get(str(removed.get("node_id")))
            enriched = dict(removed)
            if availability is not None:
                enriched.update(
                    {
                        "parent_node_id": availability.get("parent_node_id"),
                        "parent_label": availability.get("parent_label"),
                        "parent_role": availability.get("parent_role"),
                        "parent_source": availability.get("source"),
                        "source": availability.get("source"),
                        "support": int(availability.get("support", 0) or 0),
                        "support_count": int(availability.get("support_count", 0) or 0),
                        "recent": int(availability.get("recent", 0) or 0),
                        "recent_support_count": int(availability.get("recent_support_count", 0) or 0),
                        "confidence": float(availability.get("confidence", 0.0) or 0.0),
                        "strong_current": bool(availability.get("strong_current", False)),
                        "edge_type": availability.get("edge_type"),
                    }
                )
            removed_members.append(enriched)

        for member_id in sorted(occupied_members):
            member_track = self.node_tracks.get(member_id)
            if member_track is not None:
                member_track.cabinet_group_id = None

        return {
            "removed_members": removed_members,
            "dropped_groups": list(prune_debug.get("dropped_groups", [])),
        }

    def update_cabinet_groups(self, frame_idx: int, frame_result: dict, observations: list[NodeObservation]) -> None:
        assoc_debug = self.frame_assoc_debug.setdefault(int(frame_idx), {"frame_idx": int(frame_idx)})
        cabinet_debug = {
            "frame_idx": int(frame_idx),
            "scene_allows_cabinet": False,
            "carrier_candidates": [],
            "pruned_members": [],
        }
        if not self.graph_policy.should_enable_cabinet_aggregation():
            assoc_debug["cabinet_aggregation"] = cabinet_debug
            return
        scene_allows_cabinet = self.graph_policy.should_enable_cabinet_aggregation_for_frame(frame_result)
        cabinet_debug["scene_allows_cabinet"] = bool(scene_allows_cabinet)
        if not scene_allows_cabinet:
            prune_debug = self._prune_cabinet_members_with_non_cabinet_parent(
                frame_idx,
                observations,
                reason="non_cabinet_parent_claim_before_cabinet_update",
            )
            cabinet_debug["pruned_members"] = list(prune_debug.get("removed_members", []))
            self.cabinet_aggregator.last_update_debug = dict(cabinet_debug)
            assoc_debug["cabinet_aggregation"] = cabinet_debug
            return
        carrier_node_ids: list[str] = []
        seeded_node_ids: set[str] = set()
        carrier_obs_payloads: list[CabinetCarrierObservation] = []
        unavailable_member_ids: set[str] = set()
        temp_excluded_member_ids: set[str] = set()
        obs_by_node_id = {
            str(obs.matched_node_id): obs
            for obs in observations
            if obs.matched_node_id is not None
        }
        prune_before = self._prune_cabinet_members_with_non_cabinet_parent(
            frame_idx,
            observations,
            reason="non_cabinet_parent_claim_before_cabinet_update",
        )
        cabinet_debug["pruned_members_before_update"] = list(prune_before.get("removed_members", []))
        for obs in observations:
            if obs.matched_node_id is None:
                continue
            if obs.role != "C" or obs.label not in self.graph_policy.cabinet_seed_carriers:
                continue
            excluded_by_temp_policy = self.graph_policy.should_exclude_cabinet_aggregation_observation(
                frame_idx,
                obs.det_idx,
            )
            if excluded_by_temp_policy:
                temp_excluded_member_ids.add(str(obs.matched_node_id))
                cabinet_debug["carrier_candidates"].append(
                    {
                        "node_id": obs.matched_node_id,
                        "label": obs.label,
                        "det_idx": int(obs.det_idx),
                        "whether_existing_object_parent": False,
                        "existing_parent_node_id": None,
                        "existing_parent_label": None,
                        "existing_parent_role": None,
                        "existing_parent_source": None,
                        "existing_parent_support_count": 0,
                        "existing_parent_recent_support_count": 0,
                        "existing_parent_confidence": 0.0,
                        "existing_parent_strong_current": False,
                        "existing_parent_edge_type": None,
                        "eligible_for_cabinet": False,
                        "skip_reason": "temp_exclude_cabinet_aggregation_observation",
                        "cabinet_box_marked": bool(obs.cabinet_box_marked),
                    }
                )
                continue
            availability = self._is_available_for_cabinet_aggregation(
                obs.matched_node_id,
                obs,
                observations,
                frame_idx=frame_idx,
            )
            existing_parent_node_id = availability.get("parent_node_id")
            existing_parent_track = self.node_tracks.get(existing_parent_node_id) if existing_parent_node_id is not None else None
            eligible_for_cabinet = bool(availability.get("available", True))
            cabinet_debug["carrier_candidates"].append(
                {
                    "node_id": obs.matched_node_id,
                    "label": obs.label,
                    "det_idx": int(obs.det_idx),
                    "whether_existing_object_parent": bool(existing_parent_node_id is not None),
                    "existing_parent_node_id": existing_parent_node_id,
                    "existing_parent_label": None if existing_parent_track is None else getattr(existing_parent_track, "label", None),
                    "existing_parent_role": availability.get("parent_role"),
                    "existing_parent_source": availability.get("source"),
                    "existing_parent_support_count": int(availability.get("support_count", 0) or 0),
                    "existing_parent_recent_support_count": int(availability.get("recent_support_count", 0) or 0),
                    "existing_parent_confidence": float(availability.get("confidence", 0.0) or 0.0),
                    "existing_parent_strong_current": bool(availability.get("strong_current", False)),
                    "existing_parent_edge_type": availability.get("edge_type"),
                    "eligible_for_cabinet": bool(eligible_for_cabinet),
                    "skip_reason": None if eligible_for_cabinet else "skip_cabinet_due_to_non_cabinet_object_parent_claim",
                    "cabinet_box_marked": bool(obs.cabinet_box_marked),
                }
            )
            if not eligible_for_cabinet:
                unavailable_member_ids.add(obs.matched_node_id)
                continue
            carrier_node_ids.append(obs.matched_node_id)
            if obs.cabinet_box_marked:
                seeded_node_ids.add(obs.matched_node_id)
            carrier_obs_payloads.append(
                CabinetCarrierObservation(
                    node_id=obs.matched_node_id,
                    label=obs.label,
                    det_idx=int(obs.det_idx),
                    frame_idx=int(frame_idx),
                    box_xyxy=None if obs.box_xyxy is None else [float(v) for v in obs.box_xyxy],
                    view_score=float(obs.view_score),
                    cabinet_box_marked=bool(obs.cabinet_box_marked),
                    cabinet_box_scores=[float(v) for v in obs.cabinet_box_scores],
                    cabinet_box_ids=[int(v) for v in obs.cabinet_box_ids],
                    box_touches_border=bool(obs.box_touches_border),
                    centroid_world=obs.centroid_world.detach().cpu(),
                    bbox3d_world=(obs.bbox3d_world[0].detach().cpu(), obs.bbox3d_world[1].detach().cpu()),
                )
            )

        def _member_available(member_id: str) -> bool:
            if str(member_id) in temp_excluded_member_ids:
                return False
            return bool(
                self._is_available_for_cabinet_aggregation(
                    member_id,
                    obs_by_node_id.get(member_id),
                    observations,
                    frame_idx=frame_idx,
                ).get("available", True)
            )

        agg_debug = self.cabinet_aggregator.update_frame(
            frame_idx,
            carrier_node_ids,
            seeded_node_ids,
            self.node_tracks,
            carrier_obs_payloads=carrier_obs_payloads,
            scene_allows_cabinet=True,
            is_member_available=_member_available,
        )
        prune_after = self._prune_cabinet_members_with_non_cabinet_parent(
            frame_idx,
            observations,
            reason="non_cabinet_parent_claim_after_cabinet_update",
        )
        cabinet_debug["pruned_members_after_update"] = list(prune_after.get("removed_members", []))
        cabinet_debug["pruned_members"] = list(cabinet_debug["pruned_members_before_update"]) + list(
            cabinet_debug["pruned_members_after_update"]
        )
        cabinet_debug["temp_excluded_member_ids"] = sorted(temp_excluded_member_ids)
        cabinet_debug["unavailable_member_ids"] = sorted(set(unavailable_member_ids).union(temp_excluded_member_ids))
        cabinet_debug["pair_adjacency"] = list(agg_debug.get("pair_adjacency", []))
        cabinet_debug["components"] = list(agg_debug.get("components", []))
        cabinet_debug["global_group_merge_candidates"] = list(agg_debug.get("global_group_merge_candidates", []))
        cabinet_debug["merged_group_pairs"] = list(agg_debug.get("merged_group_pairs", []))
        cabinet_debug["merge_reason"] = agg_debug.get("merge_reason")
        assoc_debug["cabinet_aggregation"] = cabinet_debug

    def _cabinet_groups_debug(self) -> Dict[str, dict]:
        if not self.graph_policy.should_enable_cabinet_aggregation():
            return {}
        out: Dict[str, dict] = {}
        for gid, track in self.cabinet_aggregator.cabinet_tracks.items():
            out[gid] = {
                "cabinet_id": track.cabinet_id,
                "member_carriers": sorted(track.member_node_ids),
                "seeded_members": sorted(track.seeded_member_ids),
                "support_frames": list(track.support_frames),
                "centroid_world_est": track.centroid_world_est,
                "bbox_world_est": track.bbox_world_est,
                "last_seen_frame": track.last_seen_frame,
                "last_commit_kf": track.last_commit_kf,
            }
        return out

    def _cleanup_stale_cabinet_aggregate_nodes(self) -> dict:
        if not self.graph_policy.should_enable_cabinet_aggregation():
            return {"removed_nodes": []}
        valid_cabinet_ids = {
            track.cabinet_id
            for track in self.cabinet_aggregator.cabinet_tracks.values()
        }
        stale_ids = sorted(
            node_id
            for node_id, track in self.node_tracks.items()
            if (
                getattr(track, "origin", "standard") == "aggregate"
                or str(node_id).startswith("O_CABINET_")
            )
            and node_id not in valid_cabinet_ids
        )
        if not stale_ids:
            return {"removed_nodes": []}

        stale_set = set(stale_ids)
        for node_id in stale_ids:
            self.node_tracks.pop(node_id, None)
            self.local_posteriors.pop(node_id, None)
            self.remote_posteriors.pop(node_id, None)

        for child_id, posterior in list(self.local_posteriors.items()):
            posterior.drop_candidates(stale_set)
            posterior.preferred_parent_ids = set(pid for pid in posterior.preferred_parent_ids if pid not in stale_set)
            posterior.fallback_parent_ids = set(pid for pid in posterior.fallback_parent_ids if pid not in stale_set)
            if posterior.top1_parent_id in stale_set:
                posterior.top1_parent_id = None
            if posterior.top2_parent_id in stale_set:
                posterior.top2_parent_id = None
            if posterior.stable_parent_id in stale_set:
                posterior.stable_parent_id = None

        self.graph.local_edges = {
            key: edge
            for key, edge in self.graph.local_edges.items()
            if edge.src_node_id not in stale_set and edge.dst_node_id not in stale_set
        }
        self.graph.remote_edges = {
            key: edge
            for key, edge in self.graph.remote_edges.items()
            if edge.src_node_id not in stale_set and edge.dst_node_id not in stale_set
        }
        return {"removed_nodes": stale_ids}

    def _record_runtime_profile(self, name: str, seconds: float, count: int = 1) -> None:
        if not name:
            return
        self.runtime_profile_totals[name] = self.runtime_profile_totals.get(name, 0.0) + float(seconds)
        self.runtime_profile_counts[name] = self.runtime_profile_counts.get(name, 0) + int(count)

    def export_runtime_profile(self) -> dict:
        return {
            "totals_s": dict(self.runtime_profile_totals),
            "counts": dict(self.runtime_profile_counts),
        }

    def update_frame(
        self,
        frame_idx: int,
        frame,
        frame_result: dict,
        sam3_out,
        *,
        is_keyframe: bool,
        keyframes,
    ) -> None:
        del is_keyframe
        self._latest_keyframes = keyframes
        # Refresh the relation-text cache before any edge upserts so all
        # downstream tentative/committed edges can lookup oc/cu/ou/remote
        # relation strings extracted from the current frame_result.
        self._update_relation_text_cache_from_frame_result(frame_result)
        import os as _os_prof
        _prof = _os_prof.environ.get("FG_PROFILE_STAGES", "0") == "1"
        if _prof:
            import time as _t_prof
            _tt: dict[str, float] = {}
            _prof_verbose = _os_prof.environ.get("FG_PROFILE_VERBOSE", "0") == "1"
            def _tick(name: str, t0: float) -> None:
                _tt[name] = _tt.get(name, 0.0) + (_t_prof.perf_counter() - t0)
            _t = _t_prof.perf_counter()
            observations = self.extract_frame_observations(frame, frame_result, sam3_out); _tick("extract", _t)
            _t = _t_prof.perf_counter()
            self.associate_observations_to_nodes(observations, frame_idx, frame, keyframes); _tick("associate", _t)
            _t = _t_prof.perf_counter()
            self._remember_extract_debug(frame_idx, self._last_obs_extract_debug)
            self._remember_frame_observations(frame_idx, observations); _tick("remember", _t)
            _t = _t_prof.perf_counter()
            self.update_local_posteriors(frame_idx, observations, sam3_out); _tick("local_post", _t)
            _t = _t_prof.perf_counter()
            self._refresh_tentative_local_edges(frame_idx); _tick("local_edges", _t)
            _t = _t_prof.perf_counter()
            self.update_remote_posteriors(frame_idx, observations, sam3_out); _tick("remote_post", _t)
            _t = _t_prof.perf_counter()
            self._resolve_remote_2d_evidence_backlog(frame_idx); _tick("remote_backlog", _t)
            _t = _t_prof.perf_counter()
            self._refresh_tentative_remote_edges(frame_idx); _tick("remote_edges", _t)
            _t = _t_prof.perf_counter()
            try:
                self._process_remote_atlas_pipeline(frame_result, frame_idx)
            except Exception:
                pass
            _tick("remote_atlas", _t)
            _t = _t_prof.perf_counter()
            self.graph.prune_stale_tentative_edges(
                frame_idx,
                ttl_frames=max(self.tentative_local_ttl_frames, self.tentative_remote_ttl_frames),
            ); _tick("prune", _t)
            _t = _t_prof.perf_counter()
            self.update_cabinet_groups(frame_idx, frame_result, observations); _tick("cabinet", _t)
            _t = _t_prof.perf_counter()
            self._rehydrate_missing_oc_edges_from_posteriors(frame_idx); _tick("local_rehydrate", _t)
            _t = _t_prof.perf_counter()
            self._purge_disallowed_standard_nodes(); _tick("purge", _t)
            _t = _t_prof.perf_counter()
            self._shape_hierarchy(); _tick("shape", _t)
            for _name, _dt in _tt.items():
                self._record_runtime_profile(f"update_frame.{_name}", _dt)
            if _prof_verbose:
                _parts = "  ".join(f"{k}={v*1000:.0f}" for k, v in sorted(_tt.items(), key=lambda kv: -kv[1]))
                print(f"[FPS_PROF][update_frame F{frame_idx}] {_parts}")
            return
        observations = self.extract_frame_observations(frame, frame_result, sam3_out)
        self.associate_observations_to_nodes(observations, frame_idx, frame, keyframes)
        self._remember_extract_debug(frame_idx, self._last_obs_extract_debug)
        self._remember_frame_observations(frame_idx, observations)
        self.update_local_posteriors(frame_idx, observations, sam3_out)
        self._refresh_tentative_local_edges(frame_idx)
        self.update_remote_posteriors(frame_idx, observations, sam3_out)
        self._resolve_remote_2d_evidence_backlog(frame_idx)
        self._refresh_tentative_remote_edges(frame_idx)
        try:
            self._process_remote_atlas_pipeline(frame_result, frame_idx)
        except Exception:
            pass
        self.graph.prune_stale_tentative_edges(
            frame_idx,
            ttl_frames=max(self.tentative_local_ttl_frames, self.tentative_remote_ttl_frames),
        )
        self.update_cabinet_groups(frame_idx, frame_result, observations)
        self._rehydrate_missing_oc_edges_from_posteriors(frame_idx)
        self._purge_disallowed_standard_nodes()
        self._shape_hierarchy()
        # NOTE: intermediate JSON snapshots used to be written here every
        # frame, but that dominates runtime (~88s on a 39-frame / 15-kf
        # sequence with a 50-180MB graph) and only serves offline analysis.
        # Snapshots are now written in `commit_keyframe` and `flush_graph`.

    def _purge_disallowed_standard_nodes(self) -> None:
        disallowed = {
            node_id
            for node_id, track in self.node_tracks.items()
            if getattr(track, "origin", "standard") == "standard"
            and self.graph_policy.should_skip_standard_node(track.label)
        }
        has_remote_block_policy = bool(getattr(self.graph_policy, "remote_edge_block_labels", set()))
        if not disallowed and not has_remote_block_policy:
            return

        for node_id in disallowed:
            self.node_tracks.pop(node_id, None)
            self.local_posteriors.pop(node_id, None)

        for child_id, posterior in list(self.local_posteriors.items()):
            posterior.drop_candidates(disallowed)
            posterior.preferred_parent_ids = set(pid for pid in posterior.preferred_parent_ids if pid not in disallowed)
            posterior.fallback_parent_ids = set(pid for pid in posterior.fallback_parent_ids if pid not in disallowed)
            if posterior.top1_parent_id in disallowed:
                posterior.top1_parent_id = None
            if posterior.top2_parent_id in disallowed:
                posterior.top2_parent_id = None
            if posterior.stable_parent_id in disallowed:
                posterior.stable_parent_id = None

        for relation_key, posterior in list(self.remote_posteriors.items()):
            if (
                self.graph_policy.is_suppressed_object(posterior.src_label)
                or self.graph_policy.is_suppressed_object(posterior.dst_label)
                or self.graph_policy.should_block_remote_edge(posterior.src_label, posterior.dst_label)
            ):
                self.remote_posteriors.pop(relation_key, None)
                continue
            for pair in list(posterior.candidate_pairs.keys()):
                if pair[0] in disallowed or pair[1] in disallowed:
                    posterior.candidate_pairs.pop(pair, None)
                    posterior.support_frames.pop(pair, None)
            ranked = sorted(posterior.candidate_pairs.items(), key=lambda item: item[1], reverse=True)
            posterior.top1_pair = ranked[0][0] if ranked else None
            posterior.top2_pair = ranked[1][0] if len(ranked) > 1 else None
            if posterior.stable_pair is not None and (
                posterior.stable_pair[0] in disallowed or posterior.stable_pair[1] in disallowed
            ):
                posterior.stable_pair = None
            if not posterior.candidate_pairs and posterior.stable_pair is None:
                self.remote_posteriors.pop(relation_key, None)

        self.graph.local_edges = {
            key: edge
            for key, edge in self.graph.local_edges.items()
            if edge.src_node_id not in disallowed and edge.dst_node_id not in disallowed
        }
        self.graph.remote_edges = {
            key: edge
            for key, edge in self.graph.remote_edges.items()
            if edge.src_node_id not in disallowed
            and edge.dst_node_id not in disallowed
            and not self.graph_policy.should_block_remote_edge(
                str(getattr(self.node_tracks.get(edge.src_node_id), "label", "") or ""),
                str(getattr(self.node_tracks.get(edge.dst_node_id), "label", "") or ""),
            )
        }
        self._shape_hierarchy()

    def _shape_hierarchy(self) -> None:
        parent_u: dict[str, str] = {}
        parent_c: dict[str, str] = {}
        direct_uo: list[tuple[str, str]] = []
        chains_uco: list[tuple[str, str, str]] = []

        for edge in self.graph.local_edges.values():
            child_track = self.node_tracks.get(edge.dst_node_id)
            parent_track = self.node_tracks.get(edge.src_node_id)
            if child_track is None or parent_track is None:
                continue
            if child_track.role == "C" and parent_track.role == "O":
                parent_c[edge.dst_node_id] = edge.src_node_id
            elif child_track.role == "U":
                parent_u[edge.dst_node_id] = edge.src_node_id
                if parent_track.role == "O":
                    direct_uo.append((edge.dst_node_id, edge.src_node_id))

        for u_node_id, parent_id in parent_u.items():
            parent_track = self.node_tracks.get(parent_id)
            if parent_track is None or parent_track.role != "C":
                continue
            o_node_id = parent_c.get(parent_id)
            if o_node_id is not None:
                chains_uco.append((u_node_id, parent_id, o_node_id))

        self.graph.update_hierarchy(
            parent_u=parent_u,
            parent_c=parent_c,
            chains_uco=chains_uco,
            direct_uo=direct_uo,
            cabinet_groups=self._cabinet_groups_debug(),
        )

    def _commit_stable_graph_snapshot(
        self,
        kf_idx: int,
        *,
        frame_idx_for_debug: Optional[int] = None,
        observations: Optional[list[NodeObservation]] = None,
    ) -> None:
        for child_id, posterior in self.local_posteriors.items():
            parent_id = posterior.stable_parent_id
            if parent_id is None:
                continue
            child_track = self.node_tracks.get(child_id)
            parent_track = self.node_tracks.get(parent_id)
            if child_track is None or parent_track is None:
                continue
            if not posterior.is_commit_ready(
                min_support_frames=self.local_min_support_frames,
                stable_margin=self.local_stable_margin,
                max_switches=self.local_max_switches,
                recent_consistency_frames=self.local_recent_consistency_frames,
                unresolved_margin=self.llava_margin_local,
                unresolved_frames=self.unresolved_frames_for_llava,
            ):
                continue
            if child_track.role == "U" and parent_track.role == "O":
                if posterior.stable_parent_source != "fallback":
                    if posterior.owner_mode == "prefer_carrier":
                        continue
                elif self._has_recent_viable_preferred_carrier(child_id, posterior, exclude_parent_id=parent_id):
                    continue
            edge_type = self._edge_type_for_roles(parent_track.role, child_track.role)
            if edge_type is None:
                continue
            support_count = posterior.support_count(parent_id)
            evidence_score = float(posterior.candidate_parent_scores.get(parent_id, 0.0))
            support_frames = posterior.candidate_support_frames.get(parent_id, [])
            recent_support_count = posterior.recent_support_count(parent_id, window=3)
            arbitration = self._should_accept_local_parent_commit(
                parent_id,
                child_id,
                edge_type,
                posterior,
                {
                    "source": "posterior_commit",
                    "support_count": support_count,
                    "recent_support_count": recent_support_count,
                    "margin": float(posterior.margin),
                    "latest_margin": float(posterior.latest_evidence_margin),
                    "evidence_score": evidence_score,
                },
            )
            self._record_local_parent_arbitration(kf_idx, arbitration)
            if not arbitration["accept"]:
                continue
            self._apply_parent_arbitration_prunes(arbitration, frame_idx=kf_idx)
            self.graph.upsert_local_edge(
                parent_id,
                child_id,
                edge_type,
                relation_text=self._relation_text_for_instance_edge(parent_id, child_id, edge_type),
                committed_kf=kf_idx,
                support_count=support_count,
                evidence_score=evidence_score,
                status="committed",
                first_seen_frame=int(support_frames[0]) if support_frames else int(kf_idx),
                last_seen_frame=int(support_frames[-1]) if support_frames else int(kf_idx),
                last_update_source="stable_commit",
                margin=float(posterior.margin),
                latest_margin=float(posterior.latest_evidence_margin),
                recent_support_count=recent_support_count,
                switch_count=posterior.recent_switch_count(window=4),
            )
            posterior.mark_committed(kf_idx)

        for relation_key, posterior in self.remote_posteriors.items():
            if posterior.stable_pair is None:
                continue
            if self.graph_policy.should_block_remote_edge(posterior.src_label, posterior.dst_label):
                continue
            support_count = posterior.support_count(posterior.stable_pair)
            evidence_score = float(posterior.candidate_pairs.get(posterior.stable_pair, 0.0))
            support_frames = posterior.support_frames.get(posterior.stable_pair, [])
            stable_graph_edge_key = self._remote_graph_edge_key(
                relation_key,
                posterior.stable_pair[0],
                posterior.stable_pair[1],
            )
            self.graph.upsert_remote_edge(
                stable_graph_edge_key,
                posterior.stable_pair[0],
                posterior.stable_pair[1],
                posterior.relation_text,
                committed_kf=kf_idx,
                support_count=support_count,
                evidence_score=evidence_score,
                status="committed",
                first_seen_frame=int(support_frames[0]) if support_frames else int(kf_idx),
                last_seen_frame=int(support_frames[-1]) if support_frames else int(kf_idx),
                last_update_source="stable_commit",
                margin=float(posterior.margin),
                latest_margin=float(posterior.margin),
                recent_support_count=support_count,
                switch_count=posterior.switch_count(window=6),
            )

        if self.graph_policy.should_enable_cabinet_aggregation():
            cabinet_frame_idx = int(frame_idx_for_debug if frame_idx_for_debug is not None else kf_idx)
            cabinet_observations = list(observations or self.frame_observations.get(cabinet_frame_idx, []))
            prune_debug = self._prune_cabinet_members_with_non_cabinet_parent(
                cabinet_frame_idx,
                cabinet_observations,
                reason="non_cabinet_parent_claim_before_cabinet_commit",
            )
            obs_by_node_id = {
                str(obs.matched_node_id): obs
                for obs in cabinet_observations
                if obs.matched_node_id is not None
            }

            def _member_available(member_id: str) -> bool:
                return bool(
                    self._is_available_for_cabinet_aggregation(
                        member_id,
                        obs_by_node_id.get(member_id),
                        cabinet_observations,
                        frame_idx=cabinet_frame_idx,
                    ).get("available", True)
                )

            commit_debug = self.cabinet_aggregator.commit_keyframe(
                kf_idx,
                self.node_tracks,
                self.graph,
                can_commit_member=_member_available,
                relation_text_lookup=self._relation_text_for_instance_edge,
            )
            commit_debug["pruned_members_before_commit"] = list(prune_debug.get("removed_members", []))
            commit_debug["dropped_groups_before_commit"] = list(prune_debug.get("dropped_groups", []))
            commit_debug["stale_aggregate_cleanup"] = self._cleanup_stale_cabinet_aggregate_nodes()
            self.frame_assoc_debug.setdefault(cabinet_frame_idx, {"frame_idx": cabinet_frame_idx})["cabinet_commit"] = commit_debug
        self._shape_hierarchy()

    def commit_keyframe(self, kf_idx: int, frame, keyframes) -> None:
        self._latest_keyframes = keyframes
        import os as _os_prof
        _prof_kf = _os_prof.environ.get("FG_PROFILE_STAGES", "0") == "1"
        if _prof_kf:
            import time as _t_prof
            _ttkf: dict[str, float] = {}
            _prof_verbose = _os_prof.environ.get("FG_PROFILE_VERBOSE", "0") == "1"
            def _tick_kf(name: str, t0: float) -> None:
                _ttkf[name] = _ttkf.get(name, 0.0) + (_t_prof.perf_counter() - t0)
            _t_kf_total = _t_prof.perf_counter()
            _t0 = _t_prof.perf_counter()
        observations = self.frame_observations.get(frame.frame_id, [])
        by_node: dict[str, NodeObservation] = {}
        for obs in observations:
            if obs.matched_node_id is not None:
                by_node[obs.matched_node_id] = obs
        if _prof_kf:
            _tick_kf("prepare", _t0)

        for node_id, obs in by_node.items():
            track = self.node_tracks.get(node_id)
            if track is None:
                continue

            cleaned_obs = obs
            if _prof_kf:
                _t0 = _t_prof.perf_counter()
            if self.kf_geom_dbscan_enable and not obs.points_frame.is_cuda:
                points_frame_clean = dbscan_filter_points(
                    obs.points_frame,
                    eps=self.kf_geom_dbscan_eps,
                    min_samples=self.kf_geom_dbscan_min_samples,
                    min_keep_ratio=self.kf_geom_dbscan_min_keep_ratio,
                    pre_voxel_size=self.kf_geom_dbscan_pre_voxel,
                )
                if points_frame_clean.shape[0] >= self.min_points_per_obs:
                    points_world_clean = transform_points_world(frame, points_frame_clean)
                    cleaned_obs = NodeObservation(
                        det_idx=obs.det_idx,
                        label=obs.label,
                        raw_label=obs.raw_label,
                        role=obs.role,
                        score=obs.score,
                        mask_area=obs.mask_area,
                        box_xyxy=obs.box_xyxy,
                        box_touches_border=obs.box_touches_border,
                        num_points=int(points_frame_clean.shape[0]),
                        dbscan_filtered=obs.dbscan_filtered or (points_frame_clean.shape[0] < obs.points_frame.shape[0]),
                        points_frame=points_frame_clean,
                        centroid_frame=points_frame_clean.mean(dim=0),
                        bbox3d_frame=bbox_from_points(points_frame_clean),
                        points_world=points_world_clean,
                        centroid_world=points_world_clean.mean(dim=0),
                        bbox3d_world=bbox_from_points(points_world_clean),
                        bbox_diag_world=bbox_diag(bbox_from_points(points_world_clean)),
                        allowed_parent_labels=set(obs.allowed_parent_labels),
                        preferred_parent_labels=set(obs.preferred_parent_labels),
                        fallback_parent_labels=set(obs.fallback_parent_labels),
                        preferred_parent_role=obs.preferred_parent_role,
                        is_direct_unit=obs.is_direct_unit,
                        semantic_owner_mode=obs.semantic_owner_mode,
                        cabinet_box_marked=obs.cabinet_box_marked,
                        cabinet_box_ids=list(obs.cabinet_box_ids),
                        cabinet_box_scores=list(obs.cabinet_box_scores),
                        local_parent_det_idxs=list(obs.local_parent_det_idxs),
                        strong_parent_det_idxs=list(obs.strong_parent_det_idxs),
                        semantic_subtype=obs.semantic_subtype,
                        semantic_subtype_conf=float(obs.semantic_subtype_conf),
                        semantic_parent_label=obs.semantic_parent_label,
                        semantic_parent_role=obs.semantic_parent_role,
                        semantic_owner_object_label=obs.semantic_owner_object_label,
                        view_score=obs.view_score,
                        matched_node_id=obs.matched_node_id,
                    )
            if _prof_kf:
                _tick_kf("dbscan_clean", _t0); _t0 = _t_prof.perf_counter()

            track.update_bbox_world_est(cleaned_obs)
            track.consider_anchor_candidate(kf_idx, cleaned_obs)
            if _prof_kf:
                _tick_kf("bbox_and_candidate", _t0); _t0 = _t_prof.perf_counter()
            self._update_pre_stable_local_geometry(track, kf_idx, cleaned_obs, keyframes)
            if _prof_kf:
                _tick_kf("pre_stable_geom", _t0); _t0 = _t_prof.perf_counter()
            role = str(getattr(track, "role", "U") or "U")
            track.maybe_promote_candidate_anchor(
                min_obs_count=self._role_cfg_int(self.anchor_min_obs_count_by_role, role, self.anchor_min_obs_count),
                min_kf_support=self._role_cfg_int(self.anchor_min_kf_support_by_role, role, 1),
                keyframes=keyframes,
                allow_border_touched=(role == "C"),
                border_min_obs_count=self.anchor_carrier_border_promote_min_obs_count if role == "C" else None,
                border_min_points=max(
                    self._role_cfg_int(self.anchor_candidate_border_min_points_by_role, role, self.min_points_per_obs),
                    self.anchor_carrier_border_promote_min_points if role == "C" else 0,
                ),
                border_min_view_score=max(
                    self._role_cfg_float(self.anchor_candidate_border_min_view_score_by_role, role, 0.0),
                    self.anchor_carrier_border_promote_min_view_score if role == "C" else 0.0,
                ),
                min_points=self._role_cfg_int(self.anchor_candidate_min_points_by_role, role, self.min_points_per_obs),
                min_view_score=self._role_cfg_float(self.anchor_candidate_min_view_score_by_role, role, 0.0),
                center_dev_max=self._role_cfg_float(self.anchor_candidate_bbox_center_gate_by_role, role, 0.2),
                centroid_dev_max=self._role_cfg_float(self.anchor_candidate_centroid_gate_by_role, role, 0.2),
                created_frame=kf_idx,
            )
            if _prof_kf:
                _tick_kf("promote_anchor", _t0); _t0 = _t_prof.perf_counter()
            track.fuse_anchor_observation(
                kf_idx,
                keyframes,
                cleaned_obs,
                min_points=self._role_cfg_int(self.anchor_fusion_min_points_by_role, role, self.min_points_per_obs),
                min_view_score=self._role_cfg_float(self.anchor_fusion_min_view_score_by_role, role, 0.0),
                allow_border=(role == "C"),
                border_min_points=self._role_cfg_int(self.anchor_fusion_border_min_points_by_role, role, self.min_points_per_obs),
                border_min_view_score=self._role_cfg_float(self.anchor_fusion_border_min_view_score_by_role, role, 0.0),
            )
            if _prof_kf:
                _tick_kf("fuse_anchor", _t0); _t0 = _t_prof.perf_counter()
            # Auxiliary anchor side cache: collect multi-view coverage for
            # stable tracks.  Does not affect primary source of truth.
            if track.stable_geom_ready:
                track._add_or_update_auxiliary_anchor(kf_idx, cleaned_obs, keyframes)
                if _prof_kf:
                    _tick_kf("aux_anchor", _t0); _t0 = _t_prof.perf_counter()
                # Conservative runtime re-anchor: low-frequency corrective
                # mechanism.  Enabled by default; disable globally via
                # FG_DISABLE_REANCHOR=1 or per-track via track.enable_reanchor=False.
                if self.enable_reanchor:
                    track._try_runtime_reanchor(kf_idx, keyframes)
                    if _prof_kf:
                        _tick_kf("reanchor", _t0); _t0 = _t_prof.perf_counter()
            if (
                getattr(track, "role", None) == "C"
                and getattr(track, "label", None) in self.graph_policy.cabinet_seed_carriers
            ):
                plane_debug = track.update_plane_normal_cache(kf_idx, keyframes, self.graph_policy)
                self.frame_assoc_debug.setdefault(int(kf_idx), {"frame_idx": int(kf_idx)}).setdefault(
                    "plane_normal_updates", {}
                )[node_id] = plane_debug
                if _prof_kf:
                    _tick_kf("plane_normal", _t0); _t0 = _t_prof.perf_counter()

        if self.graph_policy.should_enable_cabinet_aggregation() and observations:
            frame_idx_for_debug = int(getattr(frame, "frame_id", kf_idx))
            revalidate_debug = self.cabinet_aggregator.revalidate_groups_with_plane_normals(
                self.node_tracks,
                frame_idx_for_debug,
                is_member_available=lambda member_id: bool(
                    self._is_available_for_cabinet_aggregation(
                        member_id,
                        by_node.get(member_id),
                        observations,
                        frame_idx=frame_idx_for_debug,
                    ).get("available", True)
                ),
            )
            self.frame_assoc_debug.setdefault(frame_idx_for_debug, {"frame_idx": frame_idx_for_debug})[
                "cabinet_plane_revalidate"
            ] = revalidate_debug
            prev_cab_debug = self.frame_assoc_debug.get(frame_idx_for_debug, {}).get("cabinet_aggregation", {})
            if bool(prev_cab_debug.get("scene_allows_cabinet", False)):
                scene_tags = sorted(self.graph_policy.cabinet_scene_prior_labels or {"cabinet"})
                self.update_cabinet_groups(
                    frame_idx_for_debug,
                    {"known_tags": scene_tags, "present": [{"object": tag} for tag in scene_tags]},
                    observations,
                )
                self.frame_assoc_debug.setdefault(frame_idx_for_debug, {"frame_idx": frame_idx_for_debug})[
                    "cabinet_aggregation"
                ]["rerun_after_plane_normal_cache"] = True

        if _prof_kf:
            _t0 = _t_prof.perf_counter()
        self._commit_stable_graph_snapshot(
            kf_idx,
            frame_idx_for_debug=int(getattr(frame, "frame_id", kf_idx)),
            observations=observations,
        )
        if _prof_kf:
            _tick_kf("commit_stable", _t0); _t0 = _t_prof.perf_counter()
        self._purge_disallowed_standard_nodes()
        if _prof_kf:
            _tick_kf("purge", _t0); _t0 = _t_prof.perf_counter()
        # Cheap delta log line (always on when output_path is set).
        self._append_graph_delta(kf_idx)
        # Periodic node consolidation pass: cheap when no duplicates exist.
        try:
            self._node_consolidation_kf_counter += 1
            every = max(1, int(self.node_consolidation_every_kf))
            if (
                self.enable_node_consolidation
                and (self._node_consolidation_kf_counter % every) == 0
            ):
                self.consolidate_duplicate_nodes(kf_idx)
        except Exception as exc:  # never break commit on consolidation
            try:
                import traceback as _tb
                self.node_consolidation_events.append({
                    "frame_idx": int(kf_idx),
                    "action": "error",
                    "reason": f"exception:{type(exc).__name__}",
                    "trace": _tb.format_exc()[-800:],
                })
            except Exception:
                pass
        if _prof_kf:
            _tick_kf("consolidation", _t0); _t0 = _t_prof.perf_counter()
        # Throttle intermediate snapshot writes and offload to the
        # background worker; see module docstring.
        if self.snapshot_every_kf > 0 and (kf_idx % self.snapshot_every_kf) == 0:
            self._save_snapshot()
        if _prof_kf:
            _tick_kf("maybe_save", _t0)
            _total = (_t_prof.perf_counter() - _t_kf_total) * 1000.0
            self._record_runtime_profile("commit_keyframe.total", _total / 1000.0)
            for _name, _dt in _ttkf.items():
                self._record_runtime_profile(f"commit_keyframe.{_name}", _dt)
            if _prof_verbose:
                _parts = "  ".join(f"{k}={v*1000:.0f}" for k, v in sorted(_ttkf.items(), key=lambda kv: -kv[1]))
                print(f"[FPS_PROF][commit_keyframe kf{kf_idx}] total={_total:.0f}ms  n={len(by_node)}  {_parts}")

    @staticmethod
    def _edge_strength_for_final_cleanup(edge) -> tuple[float, ...]:
        return (
            *PersistentFunctionalGraph._edge_strength_tuple(edge),
            float(getattr(edge, "last_seen_frame", -1) or -1),
            float(getattr(edge, "first_seen_frame", -1) or -1),
        )

    def _rebuild_local_edges_from_values(self, edges) -> None:
        rebuilt = {}
        for edge in edges:
            key = (edge.src_node_id, edge.dst_node_id, edge.edge_type)
            existing = rebuilt.get(key)
            if existing is None or self._edge_strength_for_final_cleanup(edge) > self._edge_strength_for_final_cleanup(existing):
                rebuilt[key] = edge
        self.graph.local_edges = rebuilt

    def _cleanup_final_local_edge_invariant(self, *, reason: str) -> dict:
        by_child: dict[str, list] = defaultdict(list)
        for edge in self.graph.local_edges.values():
            by_child[str(edge.dst_node_id)].append(edge)
        kept = []
        removed = []
        for child_id, edges in by_child.items():
            if len(edges) <= 1:
                kept.extend(edges)
                continue
            winner = max(edges, key=self._edge_strength_for_final_cleanup)
            kept.append(winner)
            for edge in edges:
                if edge is winner:
                    continue
                removed.append(
                    {
                        "src_node_id": edge.src_node_id,
                        "dst_node_id": edge.dst_node_id,
                        "edge_type": edge.edge_type,
                        "status": getattr(edge, "status", ""),
                        "support_count": int(getattr(edge, "support_count", 0) or 0),
                        "recent_support_count": int(getattr(edge, "recent_support_count", 0) or 0),
                        "margin": float(getattr(edge, "margin", 0.0) or 0.0),
                        "kept_src_node_id": winner.src_node_id,
                        "reason": reason,
                    }
                )
        if removed:
            self._rebuild_local_edges_from_values(kept)
        return {"action": "final_local_edge_invariant", "removed": removed, "removed_count": len(removed)}

    def _drop_final_nodes_by_policy(self) -> dict:
        drop_ids = {
            str(node_id)
            for node_id, track in self.node_tracks.items()
            if self.graph_policy.should_drop_final_label(str(getattr(track, "label", "") or ""))
        }
        if not drop_ids:
            return {"action": "final_drop_labels", "removed_node_ids": [], "removed_count": 0}

        removed_nodes = [
            {
                "node_id": node_id,
                "label": str(getattr(self.node_tracks[node_id], "label", "") or ""),
                "role": str(getattr(self.node_tracks[node_id], "role", "") or ""),
            }
            for node_id in sorted(drop_ids)
            if node_id in self.node_tracks
        ]
        for node_id in drop_ids:
            self.node_tracks.pop(node_id, None)
            self.local_posteriors.pop(node_id, None)

        for posterior in self.local_posteriors.values():
            for node_id in drop_ids:
                posterior.candidate_parent_scores.pop(node_id, None)
                posterior.candidate_support_frames.pop(node_id, None)
            ranked = sorted(
                posterior.candidate_parent_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            posterior.top1_parent_id = ranked[0][0] if ranked else None
            posterior.top2_parent_id = ranked[1][0] if len(ranked) > 1 else None

        self.graph.local_edges = {
            key: edge
            for key, edge in self.graph.local_edges.items()
            if edge.src_node_id not in drop_ids and edge.dst_node_id not in drop_ids
        }
        self.graph.remote_edges = {
            key: edge
            for key, edge in self.graph.remote_edges.items()
            if edge.src_node_id not in drop_ids and edge.dst_node_id not in drop_ids
        }
        return {
            "action": "final_drop_labels",
            "removed_node_ids": sorted(drop_ids),
            "removed_nodes": removed_nodes,
            "removed_count": len(drop_ids),
        }

    def _materialize_final_single_frame_cup_handle_edges(
        self,
        kf_idx: int,
        *,
        action: str = "final_single_frame_cup_handle",
        update_source: str = "final_single_frame_cup_handle",
    ) -> dict:
        def _is_cup_like(label: str) -> bool:
            norm = normalize_label(label)
            return norm in {"cup", "mug"} or norm.endswith(" cup")

        added = []
        for child_id, posterior in list(self.local_posteriors.items()):
            child_track = self.node_tracks.get(child_id)
            if child_track is None:
                continue
            if str(getattr(child_track, "role", "") or "") != "U":
                continue
            if normalize_label(getattr(child_track, "label", "")) != "handle":
                continue
            candidates = []
            for parent_id in set(posterior.candidate_support_frames) | set(posterior.candidate_parent_scores):
                parent_track = self.node_tracks.get(parent_id)
                if parent_track is None:
                    continue
                if not _is_cup_like(getattr(parent_track, "label", "")):
                    continue
                edge_type = self._edge_type_for_roles(parent_track.role, child_track.role)
                if edge_type != "O-U":
                    continue
                support_frames = list(posterior.candidate_support_frames.get(parent_id, []))
                support_count = int(posterior.support_count(parent_id))
                if support_count < 1 and not support_frames:
                    continue
                candidates.append(
                    (
                        support_count,
                        int(posterior.recent_support_count(parent_id, window=3)),
                        float(posterior.candidate_parent_scores.get(parent_id, 0.0)),
                        parent_id,
                        support_frames,
                    )
                )
            if not candidates:
                continue
            candidates.sort(reverse=True)
            support_count, recent_support_count, evidence_score, parent_id, support_frames = candidates[0]
            existing_edge = self._current_local_graph_edge(child_id)
            if existing_edge is not None:
                existing_parent = self.node_tracks.get(existing_edge.src_node_id)
                existing_parent_label = normalize_label(getattr(existing_parent, "label", "")) if existing_parent is not None else ""
                if _is_cup_like(existing_parent_label):
                    continue
                if str(getattr(existing_edge, "status", "") or "").lower() == "committed":
                    continue
                self.graph.local_edges.pop(
                    (existing_edge.src_node_id, existing_edge.dst_node_id, existing_edge.edge_type),
                    None,
                )
            first_seen = int(support_frames[0]) if support_frames else int(kf_idx)
            last_seen = int(support_frames[-1]) if support_frames else int(kf_idx)
            self.graph.upsert_local_edge(
                parent_id,
                child_id,
                "O-U",
                relation_text=self._relation_text_for_instance_edge(parent_id, child_id, "O-U"),
                committed_kf=-1,
                support_count=max(1, int(support_count)),
                evidence_score=float(evidence_score),
                status="tentative",
                first_seen_frame=first_seen,
                last_seen_frame=last_seen,
                last_update_source=update_source,
                margin=float(getattr(posterior, "margin", 0.0) or 0.0),
                latest_margin=float(getattr(posterior, "latest_evidence_margin", 0.0) or 0.0),
                recent_support_count=max(1, int(recent_support_count)),
                switch_count=posterior.recent_switch_count(window=4),
                retention_policy="until_contradicted",
            )
            added.append(
                {
                    "src_node_id": parent_id,
                    "dst_node_id": child_id,
                    "support_count": max(1, int(support_count)),
                    "support_frames": [int(f) for f in support_frames],
                }
            )
        return {"action": action, "added": added, "added_count": len(added)}

    def _materialize_final_bottle_cap_edges_from_posterior(self, kf_idx: int) -> dict:
        added = []
        for child_id, posterior in list(self.local_posteriors.items()):
            child_track = self.node_tracks.get(child_id)
            if not self._is_label_role(child_track, label="cap", role="U"):
                continue
            candidates = []
            for parent_id in set(posterior.candidate_support_frames) | set(posterior.candidate_parent_scores):
                parent_track = self.node_tracks.get(parent_id)
                if not self._is_label_role(parent_track, label="bottle", role="O"):
                    continue
                support_frames = list(posterior.candidate_support_frames.get(parent_id, []) or [])
                support_count = int(posterior.support_count(parent_id))
                if support_count < 1 and not support_frames:
                    continue
                candidates.append(
                    (
                        support_count,
                        int(posterior.recent_support_count(parent_id, window=3)),
                        float(posterior.candidate_parent_scores.get(parent_id, 0.0) or 0.0),
                        parent_id,
                        support_frames,
                    )
                )
            if not candidates:
                continue
            candidates.sort(reverse=True)
            support_count, recent_support_count, evidence_score, parent_id, support_frames = candidates[0]
            existing_edge = self._current_local_graph_edge(child_id)
            if existing_edge is not None:
                existing_parent = self.node_tracks.get(existing_edge.src_node_id)
                if self._is_label_role(existing_parent, label="bottle", role="O"):
                    continue
                if str(getattr(existing_edge, "status", "") or "").strip().lower() == "committed":
                    continue
                self.graph.local_edges.pop(
                    (existing_edge.src_node_id, existing_edge.dst_node_id, existing_edge.edge_type),
                    None,
                )
            first_seen = int(support_frames[0]) if support_frames else int(kf_idx)
            last_seen = int(support_frames[-1]) if support_frames else int(kf_idx)
            self.graph.upsert_local_edge(
                parent_id,
                child_id,
                "O-U",
                relation_text=self._relation_text_for_instance_edge(parent_id, child_id, "O-U"),
                committed_kf=-1,
                support_count=max(1, int(support_count)),
                evidence_score=float(evidence_score),
                status="tentative",
                first_seen_frame=first_seen,
                last_seen_frame=last_seen,
                last_update_source="temp_bottle_cap_from_posterior",
                margin=float(getattr(posterior, "margin", 0.0) or 0.0),
                latest_margin=float(getattr(posterior, "latest_evidence_margin", 0.0) or 0.0),
                recent_support_count=max(1, int(recent_support_count)),
                switch_count=posterior.recent_switch_count(window=4),
                retention_policy="until_contradicted",
            )
            added.append(
                {
                    "src_node_id": parent_id,
                    "dst_node_id": child_id,
                    "support_count": max(1, int(support_count)),
                    "support_frames": [int(f) for f in support_frames],
                }
            )
        return {
            "action": "final_bottle_cap_from_posterior",
            "added": added,
            "added_count": len(added),
        }

    def apply_realtime_temp_graph_adjustments(self, kf_idx: int) -> None:
        if not self.graph_policy.should_apply_realtime_final_adjustments():
            return None
        self._apply_temp_final_graph_adjustments(kf_idx, realtime=True)
        return None

    def _limit_final_door_drawer_unit_edges(self) -> dict:
        grouped: dict[str, list] = defaultdict(list)
        for edge in self.graph.local_edges.values():
            parent = self.node_tracks.get(edge.src_node_id)
            child = self.node_tracks.get(edge.dst_node_id)
            if parent is None or child is None:
                continue
            parent_label = normalize_label(getattr(parent, "label", ""))
            child_label = normalize_label(getattr(child, "label", ""))
            if parent_label in {"door", "drawer"} and child_label in {"handle", "knob"}:
                grouped[str(edge.src_node_id)].append(edge)

        remove_keys = set()
        removed = []
        for parent_id, edges in grouped.items():
            if len(edges) <= 1:
                continue
            winner = max(edges, key=self._edge_strength_for_final_cleanup)
            for edge in edges:
                if edge is winner:
                    continue
                key = (edge.src_node_id, edge.dst_node_id, edge.edge_type)
                remove_keys.add(key)
                removed.append(
                    {
                        "parent_node_id": parent_id,
                        "removed_child_id": edge.dst_node_id,
                        "kept_child_id": winner.dst_node_id,
                        "edge_type": edge.edge_type,
                        "status": getattr(edge, "status", ""),
                        "support_count": int(getattr(edge, "support_count", 0) or 0),
                        "recent_support_count": int(getattr(edge, "recent_support_count", 0) or 0),
                    }
                )
        if remove_keys:
            self.graph.local_edges = {
                key: edge for key, edge in self.graph.local_edges.items() if key not in remove_keys
            }
        return {"action": "final_limit_door_drawer_units", "removed": removed, "removed_count": len(removed)}

    def _limit_final_parent_child_label_edges(self, *, drop_orphan_nodes: bool = True) -> dict:
        grouped: dict[tuple[str, str], list] = defaultdict(list)
        for edge in self.graph.local_edges.values():
            parent = self.node_tracks.get(edge.src_node_id)
            child = self.node_tracks.get(edge.dst_node_id)
            if parent is None or child is None:
                continue
            parent_role = str(getattr(parent, "role", "") or "").upper()
            child_role = str(getattr(child, "role", "") or "").upper()
            if parent_role != "O" or child_role not in {"U", "C"}:
                continue
            parent_label = normalize_label(str(getattr(parent, "label", "") or ""))
            parent_origin = str(getattr(parent, "origin", "") or "")
            if parent_label == "drawer" and parent_origin == "temp_merged_drawer":
                continue
            child_label = normalize_label(str(getattr(child, "label", "") or ""))
            if not child_label:
                continue
            grouped[(str(edge.src_node_id), child_label)].append(edge)

        kept_edges = []
        removed = []
        removed_child_candidates: set[str] = set()

        def _rank(edge) -> tuple[float, ...]:
            return (
                float(getattr(edge, "support_count", 0) or 0),
                float(getattr(edge, "recent_support_count", 0) or 0),
                float(getattr(edge, "evidence_score", 0.0) or 0.0),
                float(getattr(edge, "latest_margin", 0.0) or 0.0),
                float(getattr(edge, "margin", 0.0) or 0.0),
                float(getattr(edge, "last_seen_frame", -1) or -1),
                float(getattr(edge, "first_seen_frame", -1) or -1),
            )

        grouped_keys = set()
        for (parent_id, child_label), edges in grouped.items():
            if len(edges) <= 1:
                continue
            winner = max(edges, key=_rank)
            grouped_keys.update((edge.src_node_id, edge.dst_node_id, edge.edge_type) for edge in edges)
            kept_edges.append(winner)
            for edge in edges:
                if edge is winner:
                    continue
                removed_child_candidates.add(str(edge.dst_node_id))
                removed.append(
                    {
                        "parent_node_id": parent_id,
                        "child_label": child_label,
                        "removed_child_id": str(edge.dst_node_id),
                        "kept_child_id": str(winner.dst_node_id),
                        "edge_type": str(edge.edge_type),
                        "support_count": int(getattr(edge, "support_count", 0) or 0),
                        "recent_support_count": int(getattr(edge, "recent_support_count", 0) or 0),
                        "kept_support_count": int(getattr(winner, "support_count", 0) or 0),
                    }
                )

        if not removed:
            return {
                "action": "final_parent_single_child_per_label",
                "removed": [],
                "removed_count": 0,
                "removed_orphan_child_ids": [],
            }

        rebuilt_edges = []
        for key, edge in self.graph.local_edges.items():
            if key in grouped_keys:
                continue
            rebuilt_edges.append(edge)
        rebuilt_edges.extend(kept_edges)
        self._rebuild_local_edges_from_values(rebuilt_edges)

        incident_node_ids = set()
        for edge in self.graph.local_edges.values():
            incident_node_ids.add(str(edge.src_node_id))
            incident_node_ids.add(str(edge.dst_node_id))
        for edge in self.graph.remote_edges.values():
            incident_node_ids.add(str(edge.src_node_id))
            incident_node_ids.add(str(edge.dst_node_id))
        orphan_child_ids = {
            node_id
            for node_id in removed_child_candidates
            if node_id not in incident_node_ids
            and str(getattr(self.node_tracks.get(node_id), "role", "") or "").upper() in {"U", "C"}
        }
        drop_event = (
            self._drop_final_node_ids(orphan_child_ids, reason="final_parent_single_child_per_label_drop_orphans")
            if drop_orphan_nodes
            else {"removed_node_ids": [], "removed_count": 0}
        )
        return {
            "action": "final_parent_single_child_per_label",
            "removed": removed,
            "removed_count": len(removed),
            "removed_orphan_child_ids": drop_event.get("removed_node_ids", []),
            "removed_orphan_child_count": int(drop_event.get("removed_count", 0) or 0),
        }

    def _edge_has_2d_link_support(self, edge) -> bool:
        child_id = str(getattr(edge, "dst_node_id", "") or "")
        parent_id = str(getattr(edge, "src_node_id", "") or "")
        if not child_id or not parent_id:
            return False
        posterior = self.local_posteriors.get(child_id)
        if posterior is None:
            return False
        support_frames = list(getattr(posterior, "candidate_support_frames", {}).get(parent_id, []) or [])
        if support_frames:
            return True
        support_count = 0
        try:
            support_count = int(posterior.support_count(parent_id))
        except Exception:
            support_count = 0
        if support_count > 0:
            return True
        try:
            score = float(getattr(posterior, "candidate_parent_scores", {}).get(parent_id, 0.0) or 0.0)
        except Exception:
            score = 0.0
        return score > 0.0 and int(getattr(edge, "support_count", 0) or 0) > 0

    def _drop_local_edges_without_2d_support(self, *, reason: str = "final_drop_edges_without_2d_link") -> dict:
        removed = []
        kept_edges = []
        for edge in self.graph.local_edges.values():
            if self._edge_has_2d_link_support(edge):
                kept_edges.append(edge)
                continue
            parent = self.node_tracks.get(getattr(edge, "src_node_id", ""))
            child = self.node_tracks.get(getattr(edge, "dst_node_id", ""))
            removed.append(
                {
                    "src_node_id": str(getattr(edge, "src_node_id", "") or ""),
                    "dst_node_id": str(getattr(edge, "dst_node_id", "") or ""),
                    "edge_type": str(getattr(edge, "edge_type", "") or ""),
                    "parent_label": str(getattr(parent, "label", "") or ""),
                    "child_label": str(getattr(child, "label", "") or ""),
                    "support_count": int(getattr(edge, "support_count", 0) or 0),
                    "last_update_source": str(getattr(edge, "last_update_source", "") or ""),
                }
            )
        if removed:
            self._rebuild_local_edges_from_values(kept_edges)
        return {"action": reason, "removed": removed, "removed_count": len(removed)}

    def _materialize_merged_drawer_unit_edges_from_posterior(self, kf_idx: int) -> dict:
        added = []
        for child_id, posterior in list(self.local_posteriors.items()):
            child_id_s = str(child_id)
            child_track = self.node_tracks.get(child_id_s)
            if child_track is None:
                continue
            child_label = normalize_label(str(getattr(child_track, "label", "") or ""))
            if child_label not in {"handle", "knob"}:
                continue
            if str(getattr(child_track, "role", "") or "").upper() != "U":
                continue
            candidate_parent_ids = set(getattr(posterior, "candidate_support_frames", {}) or {})
            candidate_parent_ids.update(set(getattr(posterior, "candidate_parent_scores", {}) or {}))
            for parent_id in sorted(str(pid) for pid in candidate_parent_ids):
                parent_track = self.node_tracks.get(parent_id)
                if parent_track is None:
                    continue
                parent_label = normalize_label(str(getattr(parent_track, "label", "") or ""))
                parent_origin = str(getattr(parent_track, "origin", "") or "")
                if parent_label != "drawer" or parent_origin != "temp_merged_drawer":
                    continue
                edge_type = self._edge_type_for_roles(
                    str(getattr(parent_track, "role", "") or ""),
                    str(getattr(child_track, "role", "") or ""),
                )
                if edge_type is None:
                    continue
                support_frames = list(getattr(posterior, "candidate_support_frames", {}).get(parent_id, []) or [])
                try:
                    support_count = int(posterior.support_count(parent_id))
                except Exception:
                    support_count = len(support_frames)
                if support_count < 1 and not support_frames:
                    continue
                key = (parent_id, child_id_s, edge_type)
                if key in self.graph.local_edges:
                    continue
                first_seen = int(support_frames[0]) if support_frames else int(kf_idx)
                last_seen = int(support_frames[-1]) if support_frames else int(kf_idx)
                try:
                    recent_support_count = int(posterior.recent_support_count(parent_id, window=3))
                except Exception:
                    recent_support_count = 0
                try:
                    switch_count = int(posterior.recent_switch_count(window=4))
                except Exception:
                    switch_count = 0
                try:
                    evidence_score = float(getattr(posterior, "candidate_parent_scores", {}).get(parent_id, 0.0) or 0.0)
                except Exception:
                    evidence_score = 0.0
                edge = PersistentEdge(
                    src_node_id=parent_id,
                    dst_node_id=child_id_s,
                    edge_type=edge_type,
                    relation_text=self._relation_text_for_instance_edge(parent_id, child_id_s, edge_type),
                    committed_kf=-1,
                    support_count=max(1, int(support_count)),
                    evidence_score=evidence_score,
                    status="tentative",
                    first_seen_frame=first_seen,
                    last_seen_frame=last_seen,
                    last_update_source="temp_merged_drawer_unit_posterior",
                    margin=float(getattr(posterior, "margin", 0.0) or 0.0),
                    latest_margin=float(getattr(posterior, "latest_evidence_margin", 0.0) or 0.0),
                    recent_support_count=max(1, int(recent_support_count)),
                    switch_count=switch_count,
                    retention_policy="until_contradicted",
                )
                self.graph.local_edges[key] = edge
                added.append(
                    {
                        "src_node_id": parent_id,
                        "dst_node_id": child_id_s,
                        "edge_type": edge_type,
                        "support_count": max(1, int(support_count)),
                        "support_frames": [int(f) for f in support_frames],
                    }
                )
        return {
            "action": "materialize_merged_drawer_unit_edges_from_posterior",
            "added": added,
            "added_count": len(added),
        }

    def _limit_merged_drawer_unit_edges(self, *, kf_idx: Optional[int] = None, drop_orphan_nodes: bool = True) -> dict:
        grouped: dict[str, list] = defaultdict(list)
        for edge in self.graph.local_edges.values():
            parent = self.node_tracks.get(edge.src_node_id)
            child = self.node_tracks.get(edge.dst_node_id)
            if parent is None or child is None:
                continue
            parent_label = normalize_label(str(getattr(parent, "label", "") or ""))
            parent_origin = str(getattr(parent, "origin", "") or "")
            child_label = normalize_label(str(getattr(child, "label", "") or ""))
            if parent_label == "drawer" and parent_origin == "temp_merged_drawer" and child_label in {"handle", "knob"}:
                grouped[str(edge.src_node_id)].append(edge)

        if not grouped:
            return {"action": "final_merged_drawer_two_units", "removed": [], "removed_count": 0}

        try:
            from mast3r_slam.functional_graph.ply_overlay import _resolve_node_world_position
        except Exception:
            _resolve_node_world_position = None

        keep_keys: set[tuple[str, str, str]] = set()
        remove_keys: set[tuple[str, str, str]] = set()
        removed = []
        removed_child_candidates: set[str] = set()

        def _track_max_observation_depth(track) -> Optional[float]:
            depths = []
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

        def _edge_covis_frames(edge, child) -> set[int]:
            child_id = str(getattr(edge, "dst_node_id", "") or "")
            parent_id = str(getattr(edge, "src_node_id", "") or "")
            posterior = self.local_posteriors.get(child_id)
            frames: set[int] = set()
            if posterior is not None:
                for value in list(getattr(posterior, "candidate_support_frames", {}).get(parent_id, []) or []):
                    try:
                        frame_idx = int(value)
                    except Exception:
                        frame_idx = -1
                    if frame_idx >= 0:
                        frames.add(frame_idx)
            if not frames and child is not None:
                frames = _track_observation_frames(child)
            return frames

        for parent_id, edges in grouped.items():
            parent_pos = None
            parent_track = self.node_tracks.get(parent_id)
            if parent_track is not None and _resolve_node_world_position is not None:
                try:
                    parent_pos = _resolve_node_world_position(parent_track, self._latest_keyframes)
                except Exception:
                    parent_pos = None

            positioned = []
            for edge in edges:
                child = self.node_tracks.get(edge.dst_node_id)
                pos = None
                if child is not None and _resolve_node_world_position is not None:
                    try:
                        pos = _resolve_node_world_position(child, self._latest_keyframes)
                    except Exception:
                        pos = None
                x = float(pos[0]) if pos is not None else 0.0
                display_depth = float(pos[2]) if pos is not None and len(pos) >= 3 else None
                obs_depth = _track_max_observation_depth(child) if child is not None else None
                depth = display_depth if display_depth is not None else (obs_depth if obs_depth is not None else 0.0)
                # Avoid invoking pose/T_WC CUDA ops in the realtime hot path.
                # For this temporary wrist policy, choose the candidate with
                # the largest displayed node depth within each drawer side.
                camera_distance = float(depth)
                side_x = _track_visual_side_x(child, x) if child is not None else x
                covis_frames = _edge_covis_frames(edge, child)
                positioned.append((side_x, camera_distance, depth, self._edge_strength_for_final_cleanup(edge), edge, covis_frames))

            positioned_sorted = sorted(positioned, key=lambda item: item[0])
            if len(positioned_sorted) <= 1:
                keep_edge_ids = {id(item[4]) for item in positioned_sorted}
            else:
                covisible_pairs = []
                for i, left in enumerate(positioned_sorted):
                    for right in positioned_sorted[i + 1 :]:
                        shared = set(left[5]) & set(right[5])
                        if not shared:
                            continue
                        side_gap = abs(float(right[0] - left[0]))
                        min_depth = min(float(left[1]), float(right[1]))
                        depth_score = float(left[1] + right[1])
                        strength_score = left[3] + right[3]
                        covisible_pairs.append((side_gap, min_depth, depth_score, strength_score, left, right, shared))
                if covisible_pairs:
                    _, _, _, _, left_item, right_item, _ = max(covisible_pairs, key=lambda item: item[:4])
                    keep_edge_ids = {id(left_item[4]), id(right_item[4])}
                else:
                    best_item = max(positioned_sorted, key=lambda item: (item[1], item[3]))
                    keep_edge_ids = {id(best_item[4])}
            keep_ids = {str(edge.dst_node_id) for _, _, _, _, edge, _ in positioned if id(edge) in keep_edge_ids}
            for edge in edges:
                key = (edge.src_node_id, edge.dst_node_id, edge.edge_type)
                if id(edge) in keep_edge_ids:
                    keep_keys.add(key)
                    continue
                remove_keys.add(key)
                removed_child_candidates.add(str(edge.dst_node_id))
                removed.append(
                    {
                        "parent_node_id": str(parent_id),
                        "removed_child_id": str(edge.dst_node_id),
                        "kept_child_ids": sorted(keep_ids),
                        "selection": "covisible_pair_max_side_gap_then_depth",
                        "removed_depth": next(
                            (float(item[2]) for item in positioned if item[4] is edge),
                            None,
                        ),
                        "candidate_depths": {
                            str(item[4].dst_node_id): float(item[2])
                            for item in positioned
                        },
                        "candidate_side_x": {
                            str(item[4].dst_node_id): float(item[0])
                            for item in positioned
                        },
                        "candidate_depth_scores": {
                            str(item[4].dst_node_id): float(item[1])
                            for item in positioned
                        },
                        "candidate_covis_frames": {
                            str(item[4].dst_node_id): sorted(int(v) for v in item[5])
                            for item in positioned
                        },
                        "support_count": int(getattr(edge, "support_count", 0) or 0),
                    }
                )

        if remove_keys:
            self.graph.local_edges = {
                key: edge for key, edge in self.graph.local_edges.items() if key not in remove_keys
            }

        incident_node_ids = set()
        for edge in self.graph.local_edges.values():
            incident_node_ids.add(str(edge.src_node_id))
            incident_node_ids.add(str(edge.dst_node_id))
        for edge in self.graph.remote_edges.values():
            incident_node_ids.add(str(edge.src_node_id))
            incident_node_ids.add(str(edge.dst_node_id))
        orphan_child_ids = {
            node_id
            for node_id in removed_child_candidates
            if node_id not in incident_node_ids
            and str(getattr(self.node_tracks.get(node_id), "role", "") or "").upper() in {"U", "C"}
        }
        drop_event = (
            self._drop_final_node_ids(orphan_child_ids, reason="final_merged_drawer_two_units_drop_orphans")
            if drop_orphan_nodes
            else {"removed_node_ids": [], "removed_count": 0}
        )
        return {
            "action": "final_merged_drawer_two_units",
            "removed": removed,
            "removed_count": len(removed),
            "removed_orphan_child_ids": drop_event.get("removed_node_ids", []),
            "removed_orphan_child_count": int(drop_event.get("removed_count", 0) or 0),
        }

    def _apply_final_point_cloud_centroid_node_positions(self) -> dict:
        try:
            from mast3r_slam.functional_graph.ply_overlay import _resolve_track_point_cloud_centroid_world
        except Exception:
            _resolve_track_point_cloud_centroid_world = None

        updated = []
        for node_id, track in self.node_tracks.items():
            centroid = None
            source = None
            if _resolve_track_point_cloud_centroid_world is not None:
                try:
                    point_centroid, point_source = _resolve_track_point_cloud_centroid_world(track, self._latest_keyframes)
                    if point_centroid is not None:
                        centroid = [float(v) for v in point_centroid.reshape(3).tolist()]
                        source = str(point_source or "point_cloud_centroid")
                except Exception:
                    centroid = None
                    source = None
            if centroid is None:
                weighted = []
                seen_frames = set()
                for record in list(getattr(track, "best_observations", []) or []) + list(getattr(track, "recent_observations", []) or []):
                    try:
                        frame_idx = int(record.get("frame_idx", -1))
                    except Exception:
                        frame_idx = -1
                    if frame_idx in seen_frames:
                        continue
                    seen_frames.add(frame_idx)
                    cw = record.get("centroid_world")
                    if not isinstance(cw, (list, tuple)) or len(cw) < 3:
                        continue
                    try:
                        weight = max(1.0, float(record.get("num_points", 0) or 0))
                        weighted.append(([float(cw[0]), float(cw[1]), float(cw[2])], weight))
                    except Exception:
                        continue
                if weighted:
                    weight_sum = sum(weight for _, weight in weighted)
                    centroid = [
                        sum(pos[i] * weight for pos, weight in weighted) / max(weight_sum, 1e-6)
                        for i in range(3)
                    ]
                    source = "observation_point_cloud_centroid"
            if centroid is None:
                try:
                    delattr(track, "temp_visual_centroid_world")
                except Exception:
                    pass
                continue
            setattr(track, "temp_visual_centroid_world", [float(v) for v in centroid])
            setattr(track, "temp_visual_centroid_source", str(source or "point_cloud_centroid"))
            updated.append(
                {
                    "node_id": str(node_id),
                    "label": str(getattr(track, "label", "") or ""),
                    "role": str(getattr(track, "role", "") or ""),
                    "source": str(source or "point_cloud_centroid"),
                }
            )
        return {
            "action": "final_node_point_cloud_centroid",
            "updated": updated,
            "updated_count": len(updated),
        }

    @staticmethod
    def _is_label_role(track: Optional[OnlineNodeTrack], *, label: str, role: str) -> bool:
        if track is None:
            return False
        return (
            normalize_label(str(getattr(track, "label", "") or "")) == normalize_label(label)
            and str(getattr(track, "role", "") or "").upper() == role.upper()
        )

    def _merge_final_drawer_nodes(self) -> dict:
        drawer_ids = sorted(
            str(node_id)
            for node_id, track in self.node_tracks.items()
            if self._is_label_role(track, label="drawer", role="O")
        )
        if len(drawer_ids) <= 1:
            return {"action": "final_merge_drawers", "survivor": drawer_ids[0] if drawer_ids else None, "removed": [], "removed_count": 0}

        def _drawer_rank(node_id: str) -> tuple[int, int, str]:
            track = self.node_tracks.get(node_id)
            return (
                int(getattr(track, "obs_count", 0) or 0),
                int(getattr(track, "visible_count", 0) or 0),
                str(node_id),
            )

        survivor = max(drawer_ids, key=_drawer_rank)
        removed_ids = [node_id for node_id in drawer_ids if node_id != survivor]
        merged_pos = None
        try:
            from mast3r_slam.functional_graph.ply_overlay import _resolve_node_world_position

            drawer_positions = []
            for node_id in drawer_ids:
                track = self.node_tracks.get(node_id)
                if track is None:
                    continue
                pos = _resolve_node_world_position(track, self._latest_keyframes)
                if pos is None:
                    continue
                drawer_positions.append(torch.as_tensor(pos, dtype=torch.float32).reshape(3))
            if drawer_positions:
                merged_pos = torch.stack(drawer_positions, dim=0).mean(dim=0).tolist()
        except Exception:
            merged_pos = None
        survivor_track = self.node_tracks.get(survivor)
        if survivor_track is not None:
            survivor_track.origin = "temp_merged_drawer"
            survivor_track.obs_count = int(getattr(survivor_track, "obs_count", 0) or 0) + sum(
                int(getattr(self.node_tracks.get(node_id), "obs_count", 0) or 0) for node_id in removed_ids
            )
            survivor_track.visible_count = int(getattr(survivor_track, "visible_count", 0) or 0) + sum(
                int(getattr(self.node_tracks.get(node_id), "visible_count", 0) or 0) for node_id in removed_ids
            )
            if merged_pos is not None:
                self._clear_track_geometry_for_display_position(survivor_track, [float(v) for v in merged_pos])

        rewritten_edges = []
        removed_edges = []
        removed_set = set(removed_ids)
        for edge in self.graph.local_edges.values():
            if edge.dst_node_id in removed_set:
                removed_edges.append((edge.src_node_id, edge.dst_node_id, edge.edge_type))
                continue
            if edge.src_node_id in removed_set:
                edge.src_node_id = survivor
                edge.last_update_source = "temp_merge_drawers_final"
            rewritten_edges.append(edge)
        self._rebuild_local_edges_from_values(rewritten_edges)

        self.graph.remote_edges = {
            key: edge
            for key, edge in self.graph.remote_edges.items()
            if edge.src_node_id not in removed_set and edge.dst_node_id not in removed_set
        }

        for posterior in self.local_posteriors.values():
            for removed_id in removed_ids:
                score = posterior.candidate_parent_scores.pop(removed_id, None)
                if score is not None:
                    posterior.candidate_parent_scores[survivor] = max(
                        float(posterior.candidate_parent_scores.get(survivor, 0.0) or 0.0),
                        float(score),
                    )
                frames = posterior.candidate_support_frames.pop(removed_id, None)
                if frames:
                    merged = list(posterior.candidate_support_frames.get(survivor, []) or [])
                    merged.extend(int(f) for f in frames)
                    posterior.candidate_support_frames[survivor] = sorted(set(merged))
                if posterior.top1_parent_id == removed_id:
                    posterior.top1_parent_id = survivor
                if posterior.top2_parent_id == removed_id:
                    posterior.top2_parent_id = survivor
                if posterior.stable_parent_id == removed_id:
                    posterior.stable_parent_id = survivor
                if removed_id in posterior.preferred_parent_ids:
                    posterior.preferred_parent_ids.discard(removed_id)
                    posterior.preferred_parent_ids.add(survivor)
                if removed_id in posterior.fallback_parent_ids:
                    posterior.fallback_parent_ids.discard(removed_id)
                    posterior.fallback_parent_ids.add(survivor)

        for node_id in removed_ids:
            self.node_tracks.pop(node_id, None)
            self.local_posteriors.pop(node_id, None)

        return {
            "action": "final_merge_drawers",
            "survivor": survivor,
            "removed": removed_ids,
            "removed_count": len(removed_ids),
            "removed_edges": [list(key) for key in removed_edges],
            "merged_display_pos": [float(v) for v in merged_pos] if merged_pos is not None else None,
        }

    def _drop_final_node_ids(self, drop_ids: set[str], *, reason: str) -> dict:
        if not drop_ids:
            return {"action": reason, "removed_node_ids": [], "removed_count": 0}
        drop_ids = {str(node_id) for node_id in drop_ids}
        removed_nodes = [
            {
                "node_id": node_id,
                "label": str(getattr(self.node_tracks.get(node_id), "label", "") or ""),
                "role": str(getattr(self.node_tracks.get(node_id), "role", "") or ""),
            }
            for node_id in sorted(drop_ids)
            if node_id in self.node_tracks
        ]
        for node_id in drop_ids:
            self.node_tracks.pop(node_id, None)
            self.local_posteriors.pop(node_id, None)
        for posterior in self.local_posteriors.values():
            for node_id in drop_ids:
                posterior.candidate_parent_scores.pop(node_id, None)
                posterior.candidate_support_frames.pop(node_id, None)
            ranked = sorted(posterior.candidate_parent_scores.items(), key=lambda item: item[1], reverse=True)
            posterior.top1_parent_id = ranked[0][0] if ranked else None
            posterior.top2_parent_id = ranked[1][0] if len(ranked) > 1 else None
            if posterior.stable_parent_id in drop_ids:
                posterior.stable_parent_id = None
                posterior.stable_since_kf = None
        self.graph.local_edges = {
            key: edge
            for key, edge in self.graph.local_edges.items()
            if edge.src_node_id not in drop_ids and edge.dst_node_id not in drop_ids
        }
        self.graph.remote_edges = {
            key: edge
            for key, edge in self.graph.remote_edges.items()
            if edge.src_node_id not in drop_ids and edge.dst_node_id not in drop_ids
        }
        return {
            "action": reason,
            "removed_node_ids": sorted(drop_ids),
            "removed_nodes": removed_nodes,
            "removed_count": len(drop_ids),
        }

    def _keep_single_cap_per_bottle_final(self, *, drop_unlinked_caps: bool = True) -> dict:
        grouped: dict[str, list] = defaultdict(list)
        for edge in self.graph.local_edges.values():
            parent = self.node_tracks.get(edge.src_node_id)
            child = self.node_tracks.get(edge.dst_node_id)
            if self._is_label_role(parent, label="bottle", role="O") and self._is_label_role(child, label="cap", role="U"):
                grouped[str(edge.src_node_id)].append(edge)

        keep_cap_ids: set[str] = set()
        remove_edge_keys: set[tuple[str, str, str]] = set()
        decisions = []
        for bottle_id, edges in grouped.items():
            winner = max(edges, key=self._edge_strength_for_final_cleanup)
            keep_cap_ids.add(str(winner.dst_node_id))
            removed_for_bottle = []
            for edge in edges:
                if edge is winner:
                    continue
                key = (edge.src_node_id, edge.dst_node_id, edge.edge_type)
                remove_edge_keys.add(key)
                removed_for_bottle.append(str(edge.dst_node_id))
            decisions.append(
                {
                    "bottle_id": bottle_id,
                    "kept_cap_id": str(winner.dst_node_id),
                    "removed_cap_edge_ids": sorted(set(removed_for_bottle)),
                }
            )

        if remove_edge_keys:
            self.graph.local_edges = {
                key: edge for key, edge in self.graph.local_edges.items() if key not in remove_edge_keys
            }

        if drop_unlinked_caps:
            all_cap_ids = {
                str(node_id)
                for node_id, track in self.node_tracks.items()
                if self._is_label_role(track, label="cap", role="U")
            }
            drop_cap_ids = all_cap_ids - keep_cap_ids
            drop_event = self._drop_final_node_ids(drop_cap_ids, reason="final_bottle_single_cap_drop_extra_caps")
        else:
            drop_event = {"removed_node_ids": [], "removed_count": 0}
        return {
            "action": "final_bottle_single_cap",
            "decisions": decisions,
            "removed_edge_count": len(remove_edge_keys),
            "removed_cap_node_count": int(drop_event.get("removed_count", 0) or 0),
            "removed_cap_node_ids": drop_event.get("removed_node_ids", []),
        }

    @staticmethod
    def _clear_track_geometry_for_display_position(track: OnlineNodeTrack, pos_world: list[float]) -> None:
        display_pos = [float(v) for v in pos_world]
        track.centroid_world_est = display_pos
        track.bbox_world_est = {
            "min": [float(display_pos[0] - 0.005), float(display_pos[1] - 0.005), float(display_pos[2] - 0.005)],
            "max": [float(display_pos[0] + 0.005), float(display_pos[1] + 0.005), float(display_pos[2] + 0.005)],
        }
        setattr(track, "temp_visual_centroid_world", display_pos)
        setattr(track, "temp_visual_centroid_source", "display_position")
        for attr in (
            "anchor_kf_id",
            "candidate_anchor_kf_id",
            "points_anchor",
            "centroid_anchor",
            "bbox_anchor",
            "points_fused_local",
            "centroid_fused_local",
            "bbox_fused_local",
            "candidate_points_frame",
            "candidate_centroid_frame",
            "candidate_bbox_frame",
            "provisional_points_frame",
            "provisional_centroid_frame",
            "provisional_bbox_frame",
        ):
            try:
                setattr(track, attr, None)
            except Exception:
                pass
        track.stable_geom_ready = False
        track.fused_geom_ready = False
        try:
            setattr(track, "_points_tensor_cache", {})
        except Exception:
            pass

    def _separate_final_bottle_caps_for_display(self) -> dict:
        min_distance = float(self.graph_policy.bottle_cap_min_display_distance())
        if min_distance <= 0.0:
            return {"action": "final_bottle_cap_display_distance", "adjusted": [], "adjusted_count": 0}
        try:
            from mast3r_slam.functional_graph.ply_overlay import _resolve_node_world_position
        except Exception:
            return {"action": "final_bottle_cap_display_distance", "adjusted": [], "adjusted_count": 0, "error": "resolve_import_failed"}

        keyframes = self._latest_keyframes
        adjusted = []
        bottle_cap_entries = []
        reference_vec = None
        reference_edge = None

        def _display_pos_tensor(track: OnlineNodeTrack) -> Optional[torch.Tensor]:
            pos = _resolve_node_world_position(track, keyframes)
            if pos is None:
                centroid = getattr(track, "centroid_world_est", None)
                if centroid is None:
                    return None
                try:
                    pos = torch.as_tensor(centroid, dtype=torch.float32).reshape(3)
                    if torch.isfinite(pos).all():
                        return pos
                except Exception:
                    return None
            return torch.as_tensor(pos, dtype=torch.float32).reshape(3)

        for edge in list(self.graph.local_edges.values()):
            bottle = self.node_tracks.get(edge.src_node_id)
            cap = self.node_tracks.get(edge.dst_node_id)
            if not (self._is_label_role(bottle, label="bottle", role="O") and self._is_label_role(cap, label="cap", role="U")):
                continue
            b = _display_pos_tensor(bottle)
            c = _display_pos_tensor(cap)
            if b is None or c is None:
                continue
            vec = c - b
            dist = float(torch.linalg.norm(vec).item())
            if dist > 1e-6:
                unit_vec = vec / max(dist, 1e-6)
            else:
                unit_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            bottle_cap_entries.append((edge, cap, b, c, dist, unit_vec))
            if dist >= min_distance:
                reference_vec = unit_vec
                reference_edge = (str(edge.src_node_id), str(edge.dst_node_id), dist)
        for edge, cap, b, c, dist, unit_vec in bottle_cap_entries:
            if dist >= min_distance:
                continue
            move_vec = reference_vec if reference_vec is not None else unit_vec
            new_pos = (b + move_vec * float(min_distance)).tolist()
            old_cap_pos = c.tolist()
            self._clear_track_geometry_for_display_position(cap, new_pos)
            adjusted.append(
                {
                    "bottle_id": str(edge.src_node_id),
                    "cap_id": str(edge.dst_node_id),
                    "old_distance": dist,
                    "new_distance": float(min_distance),
                    "old_pos": [float(v) for v in old_cap_pos],
                    "new_pos": [float(v) for v in new_pos],
                    "direction_source": "reference_bottle_cap" if reference_vec is not None else "self_bottle_cap",
                    "reference_edge": list(reference_edge) if reference_edge is not None else None,
                }
            )
        return {"action": "final_bottle_cap_display_distance", "adjusted": adjusted, "adjusted_count": len(adjusted)}

    def _apply_temp_final_graph_adjustments(self, kf_idx: int, *, realtime: bool = False) -> None:
        events = []
        drop_event = self._drop_final_nodes_by_policy()
        if drop_event.get("removed_count"):
            events.append(drop_event)
        if self.graph_policy.should_keep_single_frame_cup_handle():
            event = self._materialize_final_single_frame_cup_handle_edges(kf_idx)
            if event.get("added_count"):
                events.append(event)
        if self.graph_policy.should_merge_drawers_final():
            event = self._merge_final_drawer_nodes()
            if event.get("removed_count"):
                events.append(event)
            if self.graph_policy.should_limit_parent_child_label_final():
                event = self._materialize_merged_drawer_unit_edges_from_posterior(kf_idx)
                if event.get("added_count"):
                    events.append(event)
        if self.graph_policy.should_limit_door_drawer_units():
            event = self._limit_final_door_drawer_unit_edges()
            if event.get("removed_count"):
                events.append(event)
        if self.graph_policy.should_keep_single_cap_per_bottle_final():
            event = self._materialize_final_bottle_cap_edges_from_posterior(kf_idx)
            if event.get("added_count"):
                events.append(event)
            if not realtime:
                event = self._keep_single_cap_per_bottle_final(drop_unlinked_caps=not realtime)
                if event.get("removed_edge_count") or event.get("removed_cap_node_count"):
                    events.append(event)
        if self.graph_policy.should_apply_final_local_edge_invariant():
            event = self._drop_local_edges_without_2d_support(
                reason="realtime_drop_edges_without_2d_link" if realtime else "final_drop_edges_without_2d_link"
            )
            if event.get("removed_count"):
                events.append(event)
        if self.graph_policy.should_limit_parent_child_label_final():
            event = self._limit_final_parent_child_label_edges(drop_orphan_nodes=not realtime)
            if event.get("removed_count") or event.get("removed_orphan_child_count"):
                events.append(event)
            drawer_event = self._limit_merged_drawer_unit_edges(kf_idx=kf_idx, drop_orphan_nodes=not realtime)
            if drawer_event.get("removed_count") or drawer_event.get("removed_orphan_child_count"):
                events.append(drawer_event)
        if self.graph_policy.bottle_cap_min_display_distance() > 0.0:
            event = self._separate_final_bottle_caps_for_display()
            if event.get("adjusted_count"):
                events.append(event)
        if self.graph_policy.should_apply_final_local_edge_invariant():
            event = self._cleanup_final_local_edge_invariant(reason="final_export_invariant")
            if event.get("removed_count"):
                events.append(event)
        if self.graph_policy.should_use_point_cloud_centroid_node_positions():
            event = self._apply_final_point_cloud_centroid_node_positions()
            if event.get("updated_count"):
                events.append(event)
        if events:
            self.final_temp_cleanup_events.extend(events)
            self._shape_hierarchy()

    def flush_graph(self, kf_idx: int) -> None:
        # Force a final consolidation pass (non-dry-run) before freezing
        # the snapshot so duplicate nodes don't reach disk / overlay.
        try:
            if self.enable_node_consolidation:
                self.consolidate_duplicate_nodes(kf_idx, dry_run=False)
        except Exception:
            pass
        self._commit_stable_graph_snapshot(kf_idx)
        if self.graph_policy.should_enable_cabinet_aggregation():
            flush_prune_debug = self._prune_cabinet_members_with_non_cabinet_parent(
                kf_idx,
                self.frame_observations.get(kf_idx, []),
                reason="non_cabinet_parent_claim_before_cabinet_flush",
            )
            if flush_prune_debug.get("removed_members"):
                self.frame_assoc_debug.setdefault(int(kf_idx), {"frame_idx": int(kf_idx)})[
                    "cabinet_flush_prune"
                ] = flush_prune_debug
                self._cleanup_stale_cabinet_aggregate_nodes()
                self._shape_hierarchy()
        self._purge_disallowed_standard_nodes()
        self._apply_temp_final_graph_adjustments(kf_idx)
        self._append_graph_delta(kf_idx)
        # Drain any pending async snapshot first, then write the final
        # authoritative snapshot synchronously so the bytes are on disk
        # by the time this call returns.
        try:
            self.drain_snapshot_worker(timeout=30.0)
        except Exception:
            pass
        self._save_snapshot(include_eval_geometry=True, pretty=True, sync=True)
        self.shutdown_snapshot_worker(timeout=5.0)

    def to_dict(self, *, include_eval_geometry: bool = False) -> dict:
        cabinet_payload = self.cabinet_aggregator.to_dict()
        if not self.graph_policy.should_enable_cabinet_aggregation():
            cabinet_payload = {"cabinet_tracks": {}, "member_to_cabinet": {}}
        keyframes = self._latest_keyframes if include_eval_geometry else None
        return {
            "nodes": {
                node_id: track.to_dict(keyframes=keyframes, include_eval_geometry=include_eval_geometry)
                for node_id, track in self.node_tracks.items()
            },
            "local_posteriors": {child_id: posterior.to_dict() for child_id, posterior in self.local_posteriors.items()},
            "remote_posteriors": {key: posterior.to_dict() for key, posterior in self.remote_posteriors.items()},
            "graph": self.graph.to_dict(),
            "cabinet_aggregator": cabinet_payload,
            "llava_tasks": list(self.llava_scheduler.submitted_tasks),
            "association_debug_recent": {str(k): v for k, v in self.frame_assoc_debug.items()},
            "extraction_debug_recent": {str(k): v for k, v in self.frame_extract_debug.items()},
            "node_consolidation_events": list(self.node_consolidation_events),
            "lowshot_sibling_block_events": list(self.lowshot_sibling_block_events),
            "unstable_sibling_arbitration_events": list(self.unstable_sibling_arbitration_events),
            "orphan_u_prune_events": list(self.orphan_u_prune_events),
            "remote_tentative_events": list(self.remote_tentative_events),
            "remote_tentative_summary": dict(self.remote_tentative_summary),
            "remote_2d_evidence_ledger_recent": {
                k: list(v[-8:]) for k, v in list(self.remote_2d_evidence_ledger.items())
            },
            "remote_2d_backlog_events_recent": list(self.remote_2d_backlog_events[-128:]),
            "remote_2d_backlog_summary": dict(self.remote_2d_backlog_summary),
            "remote_atlas_templates_recent": {
                k: dict(v) for k, v in list(self.remote_atlas_templates.items())[-128:]
            },
            "remote_atlas_candidate_events_recent": list(
                self.remote_atlas_candidates_recent[-128:]
            ),
            "remote_atlas_llava_events_recent": list(
                self.remote_atlas_llava_events[-128:]
            ),
            "remote_atlas_summary": dict(self.remote_atlas_summary),
            "final_temp_cleanup_events": list(self.final_temp_cleanup_events),
            "node_parent_competition_losses_recent": {
                k: dict(v) for k, v in list(self.node_parent_competition_losses.items())[-256:]
            },
        }

    def save_snapshot(self) -> None:
        self._save_snapshot()

    def _snapshot_worker_loop(self) -> None:
        """Background writer. Consumes the latest pending payload and
        writes it to disk. Any earlier pending payload is dropped when a
        newer one arrives.
        """
        while True:
            self._snapshot_event.wait()
            with self._snapshot_lock:
                self._snapshot_event.clear()
                item = self._snapshot_pending
                self._snapshot_pending = None
                if item is None and self._snapshot_shutdown:
                    self._snapshot_worker_done.set()
                    return
            if item is None:
                continue
            self._snapshot_worker_done.clear()
            payload, pretty = item
            try:
                indent = 2 if pretty else None
                write_json_atomic(self.output_path, payload, indent=indent)
            except Exception as exc:  # pragma: no cover
                print(f"[fg-snapshot] write failed: {exc!r}")
            finally:
                with self._snapshot_lock:
                    if self._snapshot_pending is None:
                        self._snapshot_worker_done.set()

    def _enqueue_snapshot_async(self, payload: dict, *, pretty: bool) -> None:
        if self.output_path is None or self._snapshot_worker is None:
            return
        with self._snapshot_lock:
            # Latest-wins: drop any pending payload.
            self._snapshot_pending = (payload, pretty)
            self._snapshot_worker_done.clear()
            self._snapshot_event.set()

    def drain_snapshot_worker(self, timeout: Optional[float] = None) -> bool:
        """Block until any pending snapshot is flushed to disk. Returns
        True on success, False on timeout."""
        if self._snapshot_worker is None:
            return True
        return self._snapshot_worker_done.wait(timeout=timeout)

    def shutdown_snapshot_worker(self, timeout: Optional[float] = 10.0) -> None:
        if self._snapshot_worker is None:
            return
        with self._snapshot_lock:
            self._snapshot_shutdown = True
            self._snapshot_event.set()
        self._snapshot_worker.join(timeout=timeout)

    def _save_snapshot(
        self,
        *,
        include_eval_geometry: bool = False,
        pretty: bool = False,
        sync: bool = False,
    ) -> None:
        """Persist the current graph to JSON.

        By default the heavy work (JSON serialisation + fsync) is dispatched
        to a background worker thread with latest-wins coalescing, so the
        main loop never blocks on disk.  Pass ``sync=True`` for the final
        flush at shutdown to guarantee the bytes have reached disk.
        """
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        indent = 2 if pretty else None
        import os as _os_prof
        _prof = _os_prof.environ.get("FG_PROFILE_STAGES", "0") == "1"
        if _prof:
            import time as _t_prof
            _t0 = _t_prof.perf_counter()
        payload = self.to_dict(include_eval_geometry=include_eval_geometry)
        if _prof:
            import time as _t_prof
            _t1 = _t_prof.perf_counter()
        if sync or self._snapshot_worker is None:
            write_json_atomic(self.output_path, payload, indent=indent)
            if _prof:
                _t2 = _t_prof.perf_counter()
                print(
                    f"[FPS_PROF][save_snapshot] to_dict={(_t1-_t0)*1000.0:.1f}ms  "
                    f"write={(_t2-_t1)*1000.0:.1f}ms  pretty={pretty}  sync=True"
                )
            return
        self._enqueue_snapshot_async(payload, pretty=pretty)
        if _prof:
            _t2 = _t_prof.perf_counter()
            print(
                f"[FPS_PROF][save_snapshot] to_dict={(_t1-_t0)*1000.0:.1f}ms  "
                f"enqueue={(_t2-_t1)*1000.0:.1f}ms  pretty={pretty}  sync=False"
            )

    def _append_graph_delta(self, kf_idx: int) -> None:
        """Append a lightweight summary line to graph_delta.jsonl.

        Intentionally cheap: only per-keyframe aggregate counts, so that if
        the full JSON snapshot is missing (crash, OOM, SIGKILL) the user can
        still reconstruct a progress timeline.  No per-node data is written.
        """
        if self._delta_log_path is None:
            return
        try:
            n_nodes = len(self.node_tracks)
            n_stable = sum(1 for t in self.node_tracks.values() if getattr(t, "stable_geom_ready", False))
            n_local = len(self.graph.local_edges)
            n_remote = len(self.graph.remote_edges)
            entry = {
                "kf_idx": int(kf_idx),
                "n_nodes": int(n_nodes),
                "n_stable": int(n_stable),
                "n_local_edges": int(n_local),
                "n_remote_edges": int(n_remote),
            }
            import json as _json
            line = _json.dumps(entry, ensure_ascii=False)
            with open(self._delta_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # Delta log is best-effort; never fail the main loop on it.
            pass
