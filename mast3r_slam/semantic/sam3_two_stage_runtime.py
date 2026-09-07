"""Runtime wrapper for SAM3 two-stage detection integrated with SLAM.

This module extracts prompts from DeepSeek frame_result, runs stage-1 objects,
stage-2 carriers/units, and a final role-aware suppression pass. It reuses the
core logic from tools/bench_sam3_two_stage.py without CLI or visualization
concerns so main.py stays slim.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision.ops import nms as tv_nms
except Exception:  # torchvision may be unavailable; fall back to pure torch
    tv_nms = None

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


ROLE_RANK = {"U": 0, "C": 1, "O": 2}


LOCAL_LINK_CAPACITY_RULES: Dict[Tuple[str, str, str], int] = {
    ("bottle", "cap", "O-U"): 1,
    ("jar", "lid", "O-U"): 1,
}


@dataclass
class Sam3DetResult:
    boxes: Optional[torch.Tensor]
    scores: Optional[torch.Tensor]
    labels: Optional[List[str]]
    masks: Optional[torch.Tensor]
    per_prompt_stats: Optional[Dict[str, Dict[str, float]]] = None
    stage2_prompts: Optional[List[str]] = None
    selected_objects: Optional[List[str]] = None
    local_rel_debug: Optional[Dict[str, Any]] = None
    label_to_rank: Optional[Dict[str, int]] = None
    u_to_allowed_parents: Optional[Dict[str, Set[str]]] = None
    c_to_allowed_parents: Optional[Dict[str, Set[str]]] = None
    cabinet_hints: Optional[Dict[str, Any]] = None
    graph_policy_debug: Optional[Dict[str, Any]] = None
    sam3_processor_resolution: Optional[int] = None


def _dedup_keep_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving prompt order."""
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def uimg_to_pil(uimg: torch.Tensor | np.ndarray) -> Image.Image:
    """Convert a floating-point image in [0, 1] to an uint8 PIL image."""
    if isinstance(uimg, torch.Tensor):
        arr = uimg.detach().cpu().numpy()
    else:
        arr = np.asarray(uimg)
    img8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(img8)


def _policy_bool(policy: Optional[Any], method_name: str, label: str, fallback_attr: str) -> bool:
    if policy is not None:
        checker = getattr(policy, method_name, None)
        if callable(checker):
            return bool(checker(label))
        values = set(getattr(policy, fallback_attr, set()) or set())
        if values:
            return label in values
    return False


def extract_prompts_from_frame_result(
    frame_result: Dict[str, Any],
    policy: Optional[Any] = None,
) -> Tuple[
    List[str],
    List[str],
    List[str],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, int],
    Set[str],
    Dict[str, List[str]],
    Dict[str, Set[str]],
]:
    """Build O/C/U prompts and role priorities from a DeepSeek frame result.

    Returns:
        obj_prompts_standard: Standard object prompts.
        obj_prompts_hint_only: Object prompts used only as detection hints.
        carrier_prompts: Functional-carrier prompts.
        obj_to_carriers: Mapping from objects to carriers.
        carrier_to_units: Mapping from carriers to interaction units.
        obj_to_direct_units: Mapping from objects to direct interaction units.
        label_to_rank: Role priority per label (U=0 < C=1 < O=2).
        remote_candidate_objects: Objects retained as remote-relation candidates.
    """

    present = frame_result.get("present", []) or []

    obj_prompts_standard: List[str] = []
    obj_prompts_hint_only: List[str] = []
    carrier_prompts: List[str] = []
    obj_to_carriers: DefaultDict[str, List[str]] = DefaultDict(list)
    carrier_to_units: DefaultDict[str, List[str]] = DefaultDict(list)
    obj_to_direct_units: DefaultDict[str, List[str]] = DefaultDict(list)
    deferred_object_to_carriers: DefaultDict[str, List[str]] = DefaultDict(list)
    carrier_to_deferred_objects: DefaultDict[str, Set[str]] = DefaultDict(set)
    label_to_rank: Dict[str, int] = {}
    remote_candidate_objects: Set[str] = set()


    for entry in present:
        obj = str(entry.get("object", "")).strip().lower()
        if not obj:
            continue

        if _policy_bool(policy, "is_suppressed_object", obj, "suppress_objects"):
            continue

        is_hint_only = _policy_bool(policy, "is_hint_only_object", obj, "hint_only_objects")
        if not is_hint_only:
            obj_prompts_standard.append(obj)
            label_to_rank[obj] = min(label_to_rank.get(obj, ROLE_RANK["O"]), ROLE_RANK["O"])
        else:
            obj_prompts_hint_only.append(obj)
            label_to_rank[obj] = min(label_to_rank.get(obj, ROLE_RANK["O"]), ROLE_RANK["O"])


        for c in entry.get("functional_carriers", []) or []:
            carrier = str(c.get("carrier", "")).strip().lower()
            if carrier:
                if not is_hint_only:
                    obj_to_carriers[obj].append(carrier)
                else:
                    deferred_object_to_carriers[obj].append(carrier)
                    carrier_to_deferred_objects[carrier].add(obj)
                carrier_prompts.append(carrier)
                label_to_rank[carrier] = min(label_to_rank.get(carrier, ROLE_RANK["C"]), ROLE_RANK["C"])
            for u in c.get("interactive_units", []) or []:
                if isinstance(u, dict):
                    u_str = str(u.get("unit") or "").strip().lower()
                else:
                    u_str = str(u or "").strip().lower()
                if u_str:
                    carrier_to_units[carrier].append(u_str)
                    label_to_rank[u_str] = min(label_to_rank.get(u_str, ROLE_RANK["U"]), ROLE_RANK["U"])


        for u in entry.get("direct_interactive_units", []) or []:
            if isinstance(u, dict):
                u_str = str(u.get("unit") or "").strip().lower()
            else:
                u_str = str(u or "").strip().lower()
            if u_str:
                obj_to_direct_units[obj].append(u_str)
                label_to_rank[u_str] = min(label_to_rank.get(u_str, ROLE_RANK["U"]), ROLE_RANK["U"])

    obj_prompts_standard = _dedup_keep_order(obj_prompts_standard)
    obj_prompts_hint_only = _dedup_keep_order(obj_prompts_hint_only)
    carrier_prompts = _dedup_keep_order(carrier_prompts)
    for k in list(obj_to_carriers.keys()):
        obj_to_carriers[k] = _dedup_keep_order(obj_to_carriers[k])
    for k in list(carrier_to_units.keys()):
        carrier_to_units[k] = _dedup_keep_order(carrier_to_units[k])
    for k in list(obj_to_direct_units.keys()):
        obj_to_direct_units[k] = _dedup_keep_order(obj_to_direct_units[k])


    for remote in frame_result.get("remote_relation_candidates", []) or []:
        if isinstance(remote, dict):
            from_obj = remote.get("from_object")
            to_obj = remote.get("to_object")
            if not isinstance(from_obj, str) or not isinstance(to_obj, str):
                continue
            a = from_obj.strip().lower()
            b = to_obj.strip().lower()
            if not a or not b:
                continue
            if _policy_bool(policy, "is_suppressed_object", a, "suppress_objects"):
                continue
            if _policy_bool(policy, "is_suppressed_object", b, "suppress_objects"):
                continue
            if _policy_bool(policy, "is_hint_only_object", a, "hint_only_objects"):
                continue
            if _policy_bool(policy, "is_hint_only_object", b, "hint_only_objects"):
                continue
            remote_candidate_objects.add(a)
            remote_candidate_objects.add(b)

    return (
        obj_prompts_standard,
        obj_prompts_hint_only,
        carrier_prompts,
        obj_to_carriers,
        carrier_to_units,
        obj_to_direct_units,
        label_to_rank,
        remote_candidate_objects,
        {key: _dedup_keep_order(value) for key, value in deferred_object_to_carriers.items()},
        {key: set(value) for key, value in carrier_to_deferred_objects.items()},
    )


@torch.inference_mode()
def run_prompts_collect(
    processor: Sam3Processor,
    state: dict,
    prompts: Sequence[str],
) -> Tuple[dict, Dict[str, Dict[str, float]], Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]], Optional[torch.Tensor], float]:
    """Run detection for each prompt and accumulate boxes, scores, and masks."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    all_boxes: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []
    all_labels: List[str] = []
    all_masks: List[torch.Tensor] = []
    per_prompt_stats: Dict[str, Dict[str, float]] = {}


    for p in prompts:
        state = processor.set_text_prompt(p, state)
        boxes = state.get("boxes", state.get("box"))
        scores = state.get("scores", state.get("score"))
        masks = state.get("masks", state.get("mask"))


        n = int(scores.numel()) if isinstance(scores, torch.Tensor) else 0
        max_s = float(scores.max().item()) if n > 0 else 0.0
        per_prompt_stats[p] = {"num": n, "max_score": max_s}

        if n == 0:
            continue
        if boxes is not None:
            all_boxes.append(boxes)
        if scores is not None:
            all_scores.append(scores)
        all_labels.extend([p] * n)
        if masks is not None:
            all_masks.append(masks)

    if len(all_scores) == 0:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return state, per_prompt_stats, None, None, None, None, 0.0

    boxes_cat = torch.cat(all_boxes, dim=0) if len(all_boxes) > 0 else None
    scores_cat = torch.cat(all_scores, dim=0)
    masks_cat = torch.cat(all_masks, dim=0) if len(all_masks) > 0 else None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return state, per_prompt_stats, boxes_cat, scores_cat, all_labels, masks_cat, elapsed


def apply_nms(
    boxes: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    labels: Optional[List[str]],
    masks: Optional[torch.Tensor],
    iou_thr: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]], Optional[torch.Tensor]]:
    if boxes is None or scores is None or labels is None:
        return boxes, scores, labels, masks
    if boxes.numel() == 0:
        return boxes, scores, labels, masks


    if tv_nms is not None:
        keep = tv_nms(boxes, scores, iou_thr)
    else:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        _, order = scores.sort(descending=True)
        keep_list: List[int] = []
        while order.numel() > 0:
            i = int(order[0])
            keep_list.append(i)
            if order.numel() == 1:
                break
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            w = torch.clamp(xx2 - xx1 + 1, min=0)
            h = torch.clamp(yy2 - yy1 + 1, min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            mask = iou <= iou_thr
            order = order[1:][mask]
        keep = torch.tensor(keep_list, device=boxes.device)

    boxes_kept = boxes[keep]
    scores_kept = scores[keep]
    masks_kept = masks[keep] if masks is not None else None
    labels_kept = [labels[int(i)] for i in keep]
    return boxes_kept, scores_kept, labels_kept, masks_kept


def apply_role_aware_nms(
    boxes: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    labels: Optional[List[str]],
    masks: Optional[torch.Tensor],
    iou_thr: float,
    label_to_rank: Dict[str, int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]], Optional[torch.Tensor]]:
    if boxes is None or scores is None or labels is None or boxes.numel() == 0:
        return boxes, scores, labels, masks


    ranks = torch.tensor([label_to_rank.get(l, 3) for l in labels], device=scores.device, dtype=torch.float32)
    composite = ranks * 10.0 - scores
    order = torch.argsort(composite, descending=False)


    def role_of(idx: int) -> str:
        r = int(ranks[idx].item()) if idx < ranks.numel() else 3
        if r == ROLE_RANK["U"]:
            return "U"
        if r == ROLE_RANK["C"]:
            return "C"
        return "O"

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    keep: List[int] = []
    while order.numel() > 0:
        i = int(order[0])
        rest = order[1:]
        if rest.numel() == 0:
            keep.append(i)
            break

        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        w = torch.clamp(xx2 - xx1 + 1, min=0)
        h = torch.clamp(yy2 - yy1 + 1, min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)


        if role_of(i) == "O":
            rest_roles_o = torch.tensor([role_of(int(r)) == "O" for r in rest], device=boxes.device, dtype=torch.bool)
            same_score = torch.abs(scores[rest] - scores[i]) <= 1e-6
            tie_mask = (iou > iou_thr) & rest_roles_o & same_score
            if tie_mask.any():
                # Drop both the current and tied overlaps
                order = rest[(iou <= iou_thr) & (~tie_mask)]
                continue

        keep.append(i)
        order = rest[iou <= iou_thr]

    keep_t = torch.tensor(keep, device=boxes.device, dtype=torch.long)
    if keep_t.numel() == 0:
        return None, None, None, None
    boxes_kept = boxes[keep_t]
    scores_kept = scores[keep_t]
    masks_kept = masks[keep_t] if masks is not None else None
    labels_kept = [labels[int(i)] for i in keep_t]
    return boxes_kept, scores_kept, labels_kept, masks_kept


def filter_by_role_threshold(
    boxes: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    labels: Optional[List[str]],
    masks: Optional[torch.Tensor],
    label_to_rank: Dict[str, int],
    thr_o: float,
    thr_c: float,
    thr_u: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]], Optional[torch.Tensor]]:
    """Filter O/C/U detections using role-specific confidence thresholds."""
    if boxes is None or scores is None or labels is None:
        return boxes, scores, labels, masks
    if boxes.numel() == 0:
        return boxes, scores, labels, masks

    keep: List[int] = []
    for idx, (lab, sc) in enumerate(zip(labels, scores.tolist())):
        role = label_to_rank.get(lab, ROLE_RANK["O"])
        thr = thr_o if role == ROLE_RANK["O"] else (thr_c if role == ROLE_RANK["C"] else thr_u)
        if sc >= thr:
            keep.append(idx)

    if not keep:
        return None, None, None, None

    keep_t = torch.tensor(keep, device=boxes.device, dtype=torch.long)
    boxes_kept = boxes[keep_t]
    scores_kept = scores[keep_t]
    masks_kept = masks[keep_t] if masks is not None else None
    labels_kept = [labels[int(i)] for i in keep_t]
    return boxes_kept, scores_kept, labels_kept, masks_kept


def apply_same_role_coverage_suppression(
    boxes: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    labels: Optional[List[str]],
    masks: Optional[torch.Tensor],
    cover_thr: float,
    label_to_rank: Dict[str, int],
    roles: Sequence[str] = ("U", "C", "O"),
    force_u: Optional[Set[str]] = None,
    pad: float = 0.0,
    same_label_only: bool = True,
    log_prefix: str = "",
    emit_log: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]], Optional[torch.Tensor]]:
    """Suppress a lower-scoring box covered by another box of the same role."""
    if cover_thr is None or cover_thr <= 0:
        return boxes, scores, labels, masks
    if boxes is None or scores is None or labels is None or boxes.numel() == 0:
        return boxes, scores, labels, masks


    force_u = force_u or set()
    rank_to_role = {v: k for k, v in ROLE_RANK.items()}


    def label_role(label: str) -> str:
        if label in force_u:
            return "U"
        rank = label_to_rank.get(label, ROLE_RANK["O"])
        return rank_to_role.get(rank, "O")

    roles_set = set(roles)
    groups: Dict[Tuple[str, ...], List[int]] = {}

    for idx, lab in enumerate(labels):
        role = label_role(lab)
        if role not in roles_set:
            continue
        key: Tuple[str, ...] = (role, lab) if same_label_only else (role,)
        groups.setdefault(key, []).append(idx)

    if all(len(v) <= 1 for v in groups.values()):
        return boxes, scores, labels, masks

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1).clamp(min=0) * (y2 - y1 + 1).clamp(min=0)


    def coverage_i_by_j(i: int, j: int, pad_px: float) -> torch.Tensor:
        jx1 = x1[j] - pad_px
        jy1 = y1[j] - pad_px
        jx2 = x2[j] + pad_px
        jy2 = y2[j] + pad_px
        xx1 = torch.maximum(x1[i], jx1)
        yy1 = torch.maximum(y1[i], jy1)
        xx2 = torch.minimum(x2[i], jx2)
        yy2 = torch.minimum(y2[i], jy2)
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter_ij = w * h
        return inter_ij / areas[i].clamp(min=1e-9)


    suppressed: Set[int] = set()
    for key, inds in groups.items():
        role = key[0]
        if len(inds) <= 1:
            continue
        ordered = sorted(inds, key=lambda i: float(scores[i].item()), reverse=True)
        for a_pos, a in enumerate(ordered):
            if a in suppressed:
                continue
            for b in ordered[a_pos + 1 :]:
                if b in suppressed:
                    continue
                cov_a_by_b = coverage_i_by_j(a, b, pad)
                cov_b_by_a = coverage_i_by_j(b, a, pad)
                coverage = torch.maximum(cov_a_by_b, cov_b_by_a)
                if float(coverage.item()) >= cover_thr:
                    prefix = log_prefix or "[SAM3][cov-supp]"
                    if emit_log:
                        print(
                            f"{prefix} role={role} "
                            f"keep_idx={a} keep={labels[a]} score={float(scores[a]):.3f} "
                            f"drop_idx={b} drop={labels[b]} score={float(scores[b]):.3f} "
                            f"cov_a_by_b={float(cov_a_by_b):.3f} cov_b_by_a={float(cov_b_by_a):.3f} "
                            f"cov_max={float(coverage):.3f} thr={cover_thr:.3f} pad={pad:.2f} "
                            f"reason=same_role_covered>=thr"
                        )
                    suppressed.add(b)

    if not suppressed:
        return boxes, scores, labels, masks

    keep = [i for i in range(len(labels)) if i not in suppressed]
    keep_t = torch.tensor(keep, device=boxes.device, dtype=torch.long)
    boxes_kept = boxes[keep_t]
    scores_kept = scores[keep_t]
    labels_kept = [labels[i] for i in keep]
    masks_kept = masks[keep_t] if masks is not None else None
    return boxes_kept, scores_kept, labels_kept, masks_kept


def apply_u_coverage_suppression(
    boxes: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    labels: Optional[List[str]],
    masks: Optional[torch.Tensor],
    cover_thr: float,
    label_to_rank: Dict[str, int],
    force_u: Optional[Set[str]] = None,
    pad: float = 0.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]], Optional[torch.Tensor]]:
    """Compatibility wrapper that applies coverage suppression only to U nodes."""
    return apply_same_role_coverage_suppression(
        boxes,
        scores,
        labels,
        masks,
        cover_thr,
        label_to_rank,
        roles=("U",),
        force_u=force_u,
        pad=pad,
    )


def build_u_allowed_parents(
    obj_to_carriers: Dict[str, List[str]],
    carrier_to_units: Dict[str, List[str]],
    obj_to_direct_units: Dict[str, List[str]],
    policy: Optional[Any] = None,
) -> Dict[str, Set[str]]:
    """Build the allowed carrier and owning-object parents for each U node."""
    carrier_to_objs: DefaultDict[str, Set[str]] = DefaultDict(set)
    for obj, carriers in obj_to_carriers.items():
        for carrier in carriers:
            carrier_to_objs[carrier].add(obj)

    u_to_allowed: DefaultDict[str, Set[str]] = DefaultDict(set)

    for obj, units in obj_to_direct_units.items():
        if _policy_bool(policy, "is_hint_only_object", obj, "hint_only_objects"):
            continue
        for u in units:
            u_to_allowed[u].add(obj)

    for carrier, units in carrier_to_units.items():
        for u in units:
            u_to_allowed[u].add(carrier)
            for obj in carrier_to_objs.get(carrier, []):
                if _policy_bool(policy, "is_hint_only_object", obj, "hint_only_objects"):
                    continue
                u_to_allowed[u].add(obj)

    return {k: set(v) for k, v in u_to_allowed.items()}


def build_c_allowed_parents(obj_to_carriers: Dict[str, List[str]], policy: Optional[Any] = None) -> Dict[str, Set[str]]:
    """Build the allowed object parents for each C node."""
    c_to_allowed: DefaultDict[str, Set[str]] = DefaultDict(set)
    for obj, carriers in obj_to_carriers.items():
        if _policy_bool(policy, "is_hint_only_object", obj, "hint_only_objects"):
            continue
        for carrier in carriers:
            c_to_allowed[carrier].add(obj)
    return {k: set(v) for k, v in c_to_allowed.items()}


def build_local_rel_candidates_full(
    boxes: Optional[torch.Tensor],
    labels: Optional[List[str]],
    label_to_rank: Dict[str, int],
    force_u: Optional[Set[str]],
    u_to_allowed_parents: Dict[str, Set[str]],
    c_to_allowed_parents: Dict[str, Set[str]],
    contain_thr: float,
    pad: float,
    keep_below_thr: bool = True,
    masks: Optional[torch.Tensor] = None,
    mask_thr: float = 0.7,
    mask_margin: float = 0.1,
    mask_bin_thr: float = 0.5,
    policy: Optional[Any] = None,
    carrier_to_deferred_objects: Optional[Dict[str, Set[str]]] = None,
    same_type_drop_multi_parent_links: bool = False,
) -> Dict[str, Any]:
    """Build O-C, C-U, and O-U local-relation candidates."""

    MASK_THR_U = 0.2

    rel = {
        "contain_thr": contain_thr,
        "pad": pad,
        "mask_thr": mask_thr,
        "mask_margin": mask_margin,
        "mask_thr_u": MASK_THR_U,
        "edges": [],
        "by_child": {},
        "by_parent": {},
        "ocu_chains": [],
        "assignments": {},
        "cabinet_hints": {
            "carrier_hits": [],
        },
        "same_type_parent_prune_debug": {
            "enabled": False,
            "mode": "drop_all_same_type_conflicts",
            "conflict_children": [],
            "dropped_edges": [],
        },
        "capacity_prune_debug": {
            "enabled": False,
            "rules": [],
            "groups": [],
            "dropped_edge_count": 0,
        },
    }

    if boxes is None or labels is None or boxes.numel() == 0:
        return rel

    force_u = force_u or set()
    cabinet_seed_carriers = set(getattr(policy, "cabinet_seed_carriers", {"door", "drawer"}))
    cabinet_contain_thr = float(getattr(policy, "cabinet_carrier_contain_thr", contain_thr))
    carrier_to_deferred_objects = carrier_to_deferred_objects or {}

    def role_of(label: str) -> str:
        if label in force_u:
            return "U"
        rank = label_to_rank.get(label, ROLE_RANK["O"])
        if rank == ROLE_RANK["U"]:
            return "U"
        if rank == ROLE_RANK["C"]:
            return "C"
        return "O"

    roles = [role_of(l) for l in labels]
    idx_by_role: Dict[str, List[int]] = {"O": [], "C": [], "U": []}
    for idx, r in enumerate(roles):
        idx_by_role[r].append(idx)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1).clamp(min=0) * (y2 - y1 + 1).clamp(min=0)


    def contain(child_idx: int, parent_idx: int) -> float:
        px1 = x1[parent_idx] - pad
        py1 = y1[parent_idx] - pad
        px2 = x2[parent_idx] + pad
        py2 = y2[parent_idx] + pad
        xx1 = torch.maximum(x1[child_idx], px1)
        yy1 = torch.maximum(y1[child_idx], py1)
        xx2 = torch.minimum(x2[child_idx], px2)
        yy2 = torch.minimum(y2[child_idx], py2)
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        return float((inter / areas[child_idx].clamp(min=1e-9)).item())

    edges: List[Dict[str, Any]] = []

    def add_edge(
        edge_type: str,
        child_idx: int,
        parent_idx: int,
        contain_val: float,
        child_role: str,
        parent_role: str,
        pass_thr: bool,
    ) -> None:
        edges.append(
            {
                "type": edge_type,
                "child_idx": int(child_idx),
                "child_label": labels[child_idx],
                "child_role": child_role,
                "parent_idx": int(parent_idx),
                "parent_label": labels[parent_idx],
                "parent_role": parent_role,
                "contain": contain_val,
                "pass_thr": bool(pass_thr),
                "mask_contain": None,
                "selected": False,
            }
        )


    for c_idx in idx_by_role["C"]:
        c_label = labels[c_idx]
        allowed_parents = c_to_allowed_parents.get(c_label, set())
        if not allowed_parents:
            continue
        for o_idx in idx_by_role["O"]:
            if labels[o_idx] not in allowed_parents:
                continue
            contain_val = contain(c_idx, o_idx)

            if _policy_bool(policy, "is_hint_only_object", labels[o_idx], "hint_only_objects") and c_label in cabinet_seed_carriers:
                pass_hint = contain_val >= cabinet_contain_thr
                rel["cabinet_hints"]["carrier_hits"].append(
                    {
                        "child_idx_pre_drop": int(c_idx),
                        "child_label": c_label,
                        "cabinet_parent_idx_pre_drop": int(o_idx),
                        "cabinet_label": labels[o_idx],
                        "contain": float(contain_val),
                        "mask_contain": None,
                        "pass_thr": bool(pass_hint),
                    }
                )
                continue

            pass_thr = contain_val >= contain_thr
            if pass_thr or keep_below_thr:
                add_edge("O-C", c_idx, o_idx, contain_val, "C", "O", pass_thr)


    for u_idx in idx_by_role["U"]:
        u_label = labels[u_idx]
        allowed_parents_u = u_to_allowed_parents.get(u_label, set())
        if not allowed_parents_u:
            continue
        # Parent carriers
        for c_idx in idx_by_role["C"]:
            if labels[c_idx] not in allowed_parents_u:
                continue
            contain_val = contain(u_idx, c_idx)
            pass_thr = contain_val >= contain_thr
            if pass_thr or keep_below_thr:
                add_edge("C-U", u_idx, c_idx, contain_val, "U", "C", pass_thr)
        # Parent objects
        for o_idx in idx_by_role["O"]:
            if labels[o_idx] not in allowed_parents_u:
                continue
            if _policy_bool(policy, "is_hint_only_object", labels[o_idx], "hint_only_objects"):
                continue
            contain_val = contain(u_idx, o_idx)
            pass_thr = contain_val >= contain_thr
            if pass_thr or keep_below_thr:
                add_edge("O-U", u_idx, o_idx, contain_val, "U", "O", pass_thr)

    # Deferred object hints from carrier containment when deferred O boxes exist.
    if idx_by_role["O"] and carrier_to_deferred_objects:
        for c_idx in idx_by_role["C"]:
            c_label = labels[c_idx]
            deferred_owners = carrier_to_deferred_objects.get(c_label, set())
            if not deferred_owners:
                continue
            for o_idx in idx_by_role["O"]:
                o_label = labels[o_idx]
                if o_label not in deferred_owners:
                    continue
                contain_val = contain(c_idx, o_idx)
                pass_hint = contain_val >= cabinet_contain_thr
                rel["cabinet_hints"]["carrier_hits"].append(
                    {
                        "child_idx_pre_drop": int(c_idx),
                        "child_label": c_label,
                        "cabinet_parent_idx_pre_drop": int(o_idx),
                        "cabinet_label": o_label,
                        "contain": float(contain_val),
                        "mask_contain": None,
                        "pass_thr": bool(pass_hint),
                    }
                )

    rel["edges"] = edges

    if same_type_drop_multi_parent_links:
        conflict_children: list[dict[str, Any]] = []
        dropped_edges: list[dict[str, Any]] = []
        keep_flags = [True] * len(edges)

        by_child_type: DefaultDict[tuple[int, str], list[int]] = DefaultDict(list)
        for eid, edge in enumerate(edges):
            by_child_type[(int(edge["child_idx"]), str(edge["type"]))].append(eid)

        for (child_idx, edge_type), eids in by_child_type.items():
            pass_thr_count = sum(1 for eid in eids if bool(edges[eid].get("pass_thr", False)))
            if pass_thr_count <= 1:
                continue
            child_role = str(edges[eids[0]].get("child_role") or "")
            if child_role == "C" and edge_type != "O-C":
                continue
            if child_role == "U" and edge_type not in ("C-U", "O-U"):
                continue
            for eid in eids:
                keep_flags[eid] = False
                dropped_edges.append(
                    {
                        "eid_before_prune": int(eid),
                        "child_idx": int(edges[eid]["child_idx"]),
                        "child_label": edges[eid]["child_label"],
                        "child_role": edges[eid]["child_role"],
                        "edge_type": edges[eid]["type"],
                        "parent_idx": int(edges[eid]["parent_idx"]),
                        "parent_label": edges[eid]["parent_label"],
                        "pass_thr": bool(edges[eid].get("pass_thr", False)),
                    }
                )
            conflict_children.append(
                {
                    "child_idx": int(edges[eids[0]]["child_idx"]),
                    "child_label": edges[eids[0]]["child_label"],
                    "child_role": edges[eids[0]]["child_role"],
                    "edge_type": edge_type,
                    "num_pass_thr_before_drop": int(pass_thr_count),
                    "reason": "same_type_multi_parent_conflict",
                }
            )

        if conflict_children:
            edges = [edge for edge, keep in zip(edges, keep_flags) if keep]
            rel["edges"] = edges

        rel["same_type_parent_prune_debug"] = {
            "enabled": True,
            "mode": "drop_all_same_type_conflicts",
            "conflict_children": conflict_children,
            "dropped_edges": dropped_edges,
        }

    def _get_mask(idx: int) -> Optional[torch.Tensor]:
        if masks is None:
            return None
        m = masks[idx]

        if m.dim() == 3 and m.shape[0] == 1:
            m = m[0]
        elif m.dim() != 2:
            m = m.squeeze()
        if m.dtype != torch.bool:
            m = m > mask_bin_thr
        return m

    def _rect_from_box(idx: int, H: int, W: int) -> torch.Tensor:
        bx1, by1, bx2, by2 = [int(v) for v in boxes[idx].tolist()]
        bx1 = max(0, min(W - 1, bx1))
        bx2 = max(0, min(W - 1, bx2))
        by1 = max(0, min(H - 1, by1))
        by2 = max(0, min(H - 1, by2))
        rect = torch.zeros((H, W), dtype=torch.bool, device=boxes.device)
        rect[by1 : by2 + 1, bx1 : bx2 + 1] = True
        return rect

    def _dilate_bool(m: torch.Tensor, k: int = 3, it: int = 1) -> torch.Tensor:
        x = m.float()[None, None, ...]
        pad2 = k // 2
        for _ in range(it):
            x = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad2)
        return x[0, 0] > 0.5

    def _erode_bool(m: torch.Tensor, k: int = 3, it: int = 1) -> torch.Tensor:
        inv = (~m).float()[None, None, ...]
        pad2 = k // 2
        for _ in range(it):
            inv = F.max_pool2d(inv, kernel_size=k, stride=1, padding=pad2)
        return ~(inv[0, 0] > 0.5)

    def _close_bool(m: torch.Tensor, k: int = 3, it: int = 1) -> torch.Tensor:
        return _erode_bool(_dilate_bool(m, k=k, it=it), k=k, it=it)

    def mask_contain(child_idx: int, parent_idx: int, use_morph: bool = False) -> Optional[float]:
        cm = _get_mask(child_idx)
        pm = _get_mask(parent_idx)
        if cm is None or pm is None:
            return None

        denom = float(cm.sum().item())
        if denom <= 1e-6:
            return 0.0

        if use_morph:
            H, W = pm.shape[-2], pm.shape[-1]
            rect = _rect_from_box(parent_idx, H, W)
            pm_in = pm & rect

            pm_fix = _close_bool(pm_in, k=3, it=1) & rect
            inter1 = float((cm & pm_fix).sum().item())
            s1 = inter1 / denom

            if s1 < mask_thr:
                pm_fix2 = _dilate_bool(pm_in, k=3, it=2) & rect
                inter2 = float((cm & pm_fix2).sum().item())
                s2 = inter2 / denom
                return max(s1, s2)

            return s1

        inter = float((cm & pm).sum().item())
        return inter / denom

    def apply_parent_child_capacity_prune(local_edges: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        debug: Dict[str, Any] = {
            "enabled": True,
            "rules": [
                {
                    "parent_label": parent_label,
                    "child_label": child_label,
                    "edge_type": edge_type,
                    "max_children": int(max_children),
                }
                for (parent_label, child_label, edge_type), max_children in LOCAL_LINK_CAPACITY_RULES.items()
            ],
            "groups": [],
            "dropped_edge_count": 0,
        }

        grouped: DefaultDict[tuple[int, str, str, str], list[int]] = DefaultDict(list)
        for eid, edge in enumerate(local_edges):
            key_rule = (str(edge["parent_label"]), str(edge["child_label"]), str(edge["type"]))
            if key_rule not in LOCAL_LINK_CAPACITY_RULES:
                continue
            group_key = (int(edge["parent_idx"]), key_rule[0], key_rule[1], key_rule[2])
            grouped[group_key].append(eid)

        if not grouped:
            return local_edges, debug

        drop_eids: Set[int] = set()
        for group_key, eids in grouped.items():
            max_children = int(LOCAL_LINK_CAPACITY_RULES[(group_key[1], group_key[2], group_key[3])])
            pass_eids = [eid for eid in eids if bool(local_edges[eid].get("pass_thr", False))]
            group_debug: Dict[str, Any] = {
                "group_key": {
                    "parent_idx": int(group_key[0]),
                    "parent_label": group_key[1],
                    "child_label": group_key[2],
                    "edge_type": group_key[3],
                },
                "max_children": max_children,
                "candidate_eids": [int(eid) for eid in eids],
                "pass_eids": [int(eid) for eid in pass_eids],
            }

            if len(pass_eids) <= max_children:
                group_debug["decision"] = "within_capacity"
                debug["groups"].append(group_debug)
                continue

            if masks is None:
                for eid in eids:
                    drop_eids.add(eid)
                group_debug["decision"] = "drop_all_no_mask"
                debug["groups"].append(group_debug)
                continue

            scored: list[tuple[float, float, int]] = []
            ambiguous = False
            for eid in pass_eids:
                cidx = int(local_edges[eid]["child_idx"])
                pidx = int(local_edges[eid]["parent_idx"])
                score = mask_contain(cidx, pidx, use_morph=True)
                if score is None:
                    ambiguous = True
                    break
                local_edges[eid]["mask_contain"] = float(score)
                scored.append((float(score), float(local_edges[eid].get("contain", 0.0)), int(eid)))

            if ambiguous or not scored:
                for eid in eids:
                    drop_eids.add(eid)
                group_debug["decision"] = "drop_all_no_mask"
                debug["groups"].append(group_debug)
                continue

            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best_mask, _best_contain, best_eid = scored[0]
            second_mask = scored[1][0] if len(scored) > 1 else -1.0

            group_debug["best_mask"] = float(best_mask)
            group_debug["second_mask"] = float(second_mask)

            if best_mask >= MASK_THR_U and (best_mask - second_mask) >= mask_margin:
                for eid in eids:
                    if eid != best_eid:
                        drop_eids.add(eid)
                group_debug["decision"] = "keep_best_by_mask"
                group_debug["kept_eid"] = int(best_eid)
            else:
                for eid in eids:
                    drop_eids.add(eid)
                group_debug["decision"] = "drop_all_ambiguous"

            debug["groups"].append(group_debug)

        if not drop_eids:
            return local_edges, debug

        pruned_edges = [edge for eid, edge in enumerate(local_edges) if eid not in drop_eids]
        debug["dropped_edge_count"] = int(len(drop_eids))
        return pruned_edges, debug

    edges, capacity_prune_debug = apply_parent_child_capacity_prune(edges)
    rel["capacity_prune_debug"] = capacity_prune_debug
    rel["edges"] = edges

    by_child: Dict[int, List[int]] = {}
    by_parent: Dict[int, List[int]] = {}
    for eid, e in enumerate(edges):
        by_child.setdefault(e["child_idx"], []).append(eid)
        by_parent.setdefault(e["parent_idx"], []).append(eid)
    rel["by_child"] = {int(k): v for k, v in by_child.items()}
    rel["by_parent"] = {int(k): v for k, v in by_parent.items()}

    # ---------------------------
    # Decision layer (Step A / Step B)
    # ---------------------------
    def decide_one(child_idx: int, edge_type: str) -> Dict[str, Any]:
        cand_eids = [
            eid for eid in by_child.get(child_idx, [])
            if edges[eid]["type"] == edge_type and edges[eid].get("pass_thr", False)
        ]

        record: Dict[str, Any] = {
            "status": None,
            "edge_type": edge_type,
            "child_idx": int(child_idx),
            "candidate_eids": [int(x) for x in cand_eids],
            "candidates": [],
            "chosen_eid": None,
            "chosen_parent_idx": None,
            "thr_used": None,
            "margin_used": float(mask_margin),
        }

        # Step A: 0 / 1
        if len(cand_eids) == 0:
            record["status"] = "none"
            return record

        if len(cand_eids) == 1:
            chosen = cand_eids[0]
            edges[chosen]["selected"] = True
            record["status"] = "confirmed"
            record["chosen_eid"] = int(chosen)
            record["chosen_parent_idx"] = int(edges[chosen]["parent_idx"])
            return record

        # Step B: >=2, use mask containment first
        scored: List[Tuple[float, int]] = []
        for eid in cand_eids:
            pidx = int(edges[eid]["parent_idx"])
            s = mask_contain(child_idx, pidx, use_morph=True)
            if s is None:
                record["status"] = "ambiguous_no_mask"
                for eid2 in cand_eids:
                    edges[eid2]["selected"] = False
                    record["candidates"].append(
                        {
                            "eid": int(eid2),
                            "parent_idx": int(edges[eid2]["parent_idx"]),
                            "parent_label": edges[eid2]["parent_label"],
                            "contain": float(edges[eid2]["contain"]),
                            "mask_contain": None,
                        }
                    )
                return record
            edges[eid]["mask_contain"] = float(s)
            scored.append((float(s), int(eid)))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_s, best_eid = scored[0]
        second_s = scored[1][0] if len(scored) > 1 else -1.0

        thr_used = MASK_THR_U if edge_type in ("C-U", "O-U") else mask_thr
        record["thr_used"] = float(thr_used)

        for s, eid in scored:
            record["candidates"].append(
                {
                    "eid": int(eid),
                    "parent_idx": int(edges[eid]["parent_idx"]),
                    "parent_label": edges[eid]["parent_label"],
                    "contain": float(edges[eid]["contain"]),
                    "mask_contain": float(s),
                }
            )

        if best_s >= thr_used and (best_s - second_s) >= mask_margin:
            record["status"] = "confirmed_mask"
            record["chosen_eid"] = int(best_eid)
            record["chosen_parent_idx"] = int(edges[best_eid]["parent_idx"])
            for _, eid in scored:
                edges[eid]["selected"] = (eid == best_eid)
        else:
            record["status"] = "ambiguous"
            for _, eid in scored:
                edges[eid]["selected"] = False

        return record

    assignments: Dict[int, Dict[str, Any]] = {}
    for u_idx in idx_by_role["U"]:
        assignments[int(u_idx)] = {
            "C-U": decide_one(u_idx, "C-U"),
            "O-U": decide_one(u_idx, "O-U"),
        }
    for c_idx in idx_by_role["C"]:
        assignments.setdefault(int(c_idx), {})
        assignments[int(c_idx)]["O-C"] = decide_one(c_idx, "O-C")

    rel["assignments"] = assignments

    # ---------------------------
    # Step C: carrier-child mask recheck
    # When a carrier (door/drawer/cabinet door) has ≥2 *selected* unit
    # children (knob/handle), compute mask_contain for each such edge
    # and deselect those that fail the threshold.
    # ---------------------------
    _CARRIER_LABELS_C = {"door", "drawer", "cabinet door"}
    _UNIT_LABELS_C = {"knob", "handle"}
    carrier_child_recheck_debug: List[Dict[str, Any]] = []
    for parent_idx, eids in by_parent.items():
        p_label = labels[parent_idx].lower().strip()
        if p_label not in _CARRIER_LABELS_C:
            continue
        # Collect selected C-U edges to knob/handle children
        cu_selected: List[int] = []
        for eid in eids:
            e = edges[eid]
            if (
                e["type"] == "C-U"
                and e.get("selected", False)
                and e["child_label"].lower().strip() in _UNIT_LABELS_C
            ):
                cu_selected.append(eid)
        if len(cu_selected) < 2:
            continue
        # Compute mask_contain for each
        scored: List[Tuple[float, int]] = []
        any_none = False
        for eid in cu_selected:
            cidx = edges[eid]["child_idx"]
            s = mask_contain(cidx, parent_idx, use_morph=True)
            if s is None:
                any_none = True
                edges[eid]["mask_contain"] = None
                scored.append((0.0, eid))
            else:
                edges[eid]["mask_contain"] = float(s)
                scored.append((float(s), eid))

        if any_none:
            # Cannot compute mask for some; skip recheck for this parent
            carrier_child_recheck_debug.append({
                "parent_idx": int(parent_idx),
                "parent_label": p_label,
                "status": "skip_no_mask",
                "cu_selected_eids": [int(x) for x in cu_selected],
            })
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        to_remove = [(s, eid) for s, eid in scored if s < MASK_THR_U]
        # Keep at least one child even if all fail
        if len(to_remove) >= len(scored):
            to_remove = to_remove[1:]  # scored is sorted desc, keep best
        for s, eid in to_remove:
            edges[eid]["selected"] = False

        carrier_child_recheck_debug.append({
            "parent_idx": int(parent_idx),
            "parent_label": p_label,
            "status": "rechecked",
            "cu_selected_eids": [int(x) for x in cu_selected],
            "scores": [{"eid": int(eid), "mask_contain": float(s)} for s, eid in scored],
            "removed_eids": [int(eid) for s, eid in to_remove],
        })

    rel["carrier_child_recheck"] = carrier_child_recheck_debug


    ocu_chains: List[Dict[str, Any]] = []
    # Build lookup for faster access
    oc_edges_for_c: DefaultDict[int, List[Dict[str, Any]]] = DefaultDict(list)
    for e in edges:
        if e["type"] == "O-C" and e.get("pass_thr", False):
            oc_edges_for_c[e["child_idx"]].append(e)

    for e in edges:
        if e["type"] != "C-U" or not e.get("pass_thr", False):
            continue
        u_idx = e["child_idx"]
        c_idx = e["parent_idx"]
        contain_uc = e["contain"]
        for oc in oc_edges_for_c.get(c_idx, []):
            o_idx = oc["parent_idx"]
            contain_co = oc["contain"]
            ocu_chains.append(
                {
                    "o_idx": int(o_idx),
                    "o_label": labels[o_idx],
                    "c_idx": int(c_idx),
                    "c_label": labels[c_idx],
                    "u_idx": int(u_idx),
                    "u_label": labels[u_idx],
                    "contain_uc": contain_uc,
                    "contain_co": contain_co,
                    "min_contain": float(min(contain_uc, contain_co)),
                }
            )

    rel["ocu_chains"] = ocu_chains
    return rel


def filter_u_not_in_allowed_parents(
    boxes: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    labels: Optional[List[str]],
    masks: Optional[torch.Tensor],
    label_to_rank: Dict[str, int],
    u_to_allowed_parents: Dict[str, Set[str]],
    contain_thr: float,
    pad: float = 0.0,
    force_u: Optional[Set[str]] = None,
    collect_drops: Optional[List[Dict[str, Any]]] = None,
    drop_reason: str = "r1_not_in_allowed_parent",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]], Optional[torch.Tensor]]:
    """Drop U detections not contained in any allowed carrier/object box."""
    if contain_thr is None or contain_thr <= 0:
        return boxes, scores, labels, masks
    if boxes is None or scores is None or labels is None or boxes.numel() == 0:
        return boxes, scores, labels, masks

    force_u = force_u or set()

    def role_rank(label: str) -> int:
        if label in force_u:
            return ROLE_RANK["U"]
        return label_to_rank.get(label, ROLE_RANK["O"])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1).clamp(min=0) * (y2 - y1 + 1).clamp(min=0)

    def coverage_i_by_j(i: int, j: int) -> torch.Tensor:
        jx1 = x1[j] - pad
        jy1 = y1[j] - pad
        jx2 = x2[j] + pad
        jy2 = y2[j] + pad
        xx1 = torch.maximum(x1[i], jx1)
        yy1 = torch.maximum(y1[i], jy1)
        xx2 = torch.minimum(x2[i], jx2)
        yy2 = torch.minimum(y2[i], jy2)
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter_ij = w * h
        return inter_ij / areas[i].clamp(min=1e-9)

    suppressed: Set[int] = set()
    for idx, lab in enumerate(labels):
        if role_rank(lab) != ROLE_RANK["U"]:
            continue
        allowed_parents = u_to_allowed_parents.get(lab, set())
        drop_info: Dict[str, Any] = {}
        if not allowed_parents:
            suppressed.add(idx)
            drop_info["reason"] = "no_allowed_parents"
        else:
            candidates = [j for j, plab in enumerate(labels) if plab in allowed_parents and role_rank(plab) in (ROLE_RANK["C"], ROLE_RANK["O"])]
            if not candidates:
                suppressed.add(idx)
                drop_info["reason"] = "no_parent_detected"
            else:
                coverages = [coverage_i_by_j(idx, cand) for cand in candidates]
                max_cov = torch.stack(coverages).max()
                drop_info["max_cov"] = float(max_cov.item())
                if float(max_cov.item()) < contain_thr:
                    suppressed.add(idx)
                    drop_info["reason"] = "contain_below_thr"

        if collect_drops is not None and idx in suppressed:
            drop_info.update(
                {
                    "rule": drop_reason,
                    "idx": int(idx),
                    "label": lab,
                    "score": float(scores[idx].item()) if scores is not None else None,
                    "allowed_parents": sorted(allowed_parents) if allowed_parents else [],
                }
            )
            collect_drops.append(drop_info)

    if not suppressed:
        return boxes, scores, labels, masks

    keep = [i for i in range(len(labels)) if i not in suppressed]
    if not keep:
        return None, None, None, None

    keep_t = torch.tensor(keep, device=boxes.device, dtype=torch.long)
    boxes_kept = boxes[keep_t]
    scores_kept = scores[keep_t]
    labels_kept = [labels[i] for i in keep]
    masks_kept = masks[keep_t] if masks is not None else None
    return boxes_kept, scores_kept, labels_kept, masks_kept


def _mk_pf_prefix(frame_idx: Optional[int], stage: str) -> str:
    frame_str = "?" if frame_idx is None else f"{frame_idx:06d}"
    return f"[SAM3][f={frame_str}][stage={stage}]"


def _post_merge_postfilters(
    boxes: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    labels: Optional[List[str]],
    masks: Optional[torch.Tensor],
    *,
    frame_idx: Optional[int],
    label_to_rank: Dict[str, int],
    u_to_allowed_parents: Dict[str, Set[str]],
    c_to_allowed_parents: Dict[str, Set[str]],
    cover_thr: float,
    cover_pad: float,
    u_parent_contain_thr: float,
    force_u: Optional[Set[str]],
    img_hw: Optional[Tuple[int, int]] = None,
    debug_out: Optional[Dict[str, Any]] = None,
    u_low_score_thr: float = 0.7,
    u_small_area_ratio: float = 0.25,
    border_margin_px: float = 3.0,
    drop_parent_without_u: bool = True,
    remote_candidate_objects: Optional[Set[str]] = None,
    policy: Optional[Any] = None,
    carrier_to_deferred_objects: Optional[Dict[str, Set[str]]] = None,
    same_type_drop_multi_parent_links: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]], Optional[torch.Tensor]]:

    prefix = _mk_pf_prefix(frame_idx, "POST_MERGE")
    debug = debug_out if debug_out is not None else {}
    drops: List[Dict[str, Any]] = debug.setdefault("drops", [])
    rel_dbg = debug.setdefault("local_rel", {})
    remote_candidates = set(remote_candidate_objects or [])

    def role_rank_of(label: str) -> int:
        if force_u and label in force_u:
            return ROLE_RANK["U"]
        return label_to_rank.get(label, ROLE_RANK["O"])

    detected_objects: Set[str] = {lab for lab in (labels or []) if role_rank_of(lab) == ROLE_RANK["O"]}

    border_margin = 3.0 if border_margin_px is None else border_margin_px
    index_lineage: list[int] = list(range(len(labels))) if labels is not None else []


    def near_border(box: torch.Tensor) -> bool:
        if img_hw is None:
            return False
        h, w = img_hw
        return bool(
            float(box[0]) <= border_margin
            or float(box[1]) <= border_margin
            or float(box[2]) >= (w - 1 - border_margin)
            or float(box[3]) >= (h - 1 - border_margin)
        )

    def border_contact_count(box: torch.Tensor, margin_px: float = 3.0) -> int:
        if img_hw is None:
            return 0
        h, w = img_hw
        count = 0
        if float(box[0]) <= margin_px:
            count += 1
        if float(box[1]) <= margin_px:
            count += 1
        if float(box[2]) >= (w - 1 - margin_px):
            count += 1
        if float(box[3]) >= (h - 1 - margin_px):
            count += 1
        return count

    def box_area_ratio(box: torch.Tensor) -> float:
        if img_hw is None:
            return 0.0
        h, w = img_hw
        box_area = max(0.0, float(box[2] - box[0] + 1.0)) * max(0.0, float(box[3] - box[1] + 1.0))
        return box_area / max(1.0, float(h * w))


    def label_role(label: str) -> str:
        if force_u and label in force_u:
            return "U"
        rank = label_to_rank.get(label, ROLE_RANK["O"])
        if rank == ROLE_RANK["U"]:
            return "U"
        if rank == ROLE_RANK["C"]:
            return "C"
        return "O"


    def record_local_rel(tag: str, keep_below_thr: bool = True) -> None:
        rel_dbg[tag] = build_local_rel_candidates_full(
            boxes,
            labels,
            label_to_rank,
            force_u,
            u_to_allowed_parents,
            c_to_allowed_parents,
            contain_thr=u_parent_contain_thr,
            pad=cover_pad,
            keep_below_thr=keep_below_thr,
            masks=masks,
            policy=policy,
            carrier_to_deferred_objects=carrier_to_deferred_objects,
            same_type_drop_multi_parent_links=same_type_drop_multi_parent_links,
        )

    def area(idx: int) -> float:
        if boxes is None:
            return 0.0
        return float(((boxes[idx, 2] - boxes[idx, 0] + 1) * (boxes[idx, 3] - boxes[idx, 1] + 1)).clamp(min=0).item())


    def drop_indices(to_drop: Set[int], rule: str, extra: Optional[Dict[str, Any]] = None, log: bool = True) -> None:
        nonlocal boxes, scores, labels, masks, index_lineage
        if boxes is None or scores is None or labels is None or not to_drop:
            return
        if log:
            for idx in sorted(to_drop):
                drops.append(
                    {
                        "rule": rule,
                        "idx": int(idx),
                        "label": labels[idx],
                        "score": float(scores[idx].item()) if scores is not None else None,
                        **(extra or {}),
                    }
                )
        keep = [i for i in range(len(labels)) if i not in to_drop]
        if not keep:
            boxes = None
            scores = None
            labels = None
            masks = None
            index_lineage = []
            return
        keep_t = torch.tensor(keep, device=boxes.device, dtype=torch.long)
        boxes = boxes[keep_t]
        scores = scores[keep_t]
        labels = [labels[i] for i in keep]
        masks = masks[keep_t] if masks is not None else None
        index_lineage = [index_lineage[i] for i in keep]



    boxes, scores, labels, masks = apply_same_role_coverage_suppression(
        boxes,
        scores,
        labels,
        masks,
        cover_thr=cover_thr,
        label_to_rank=label_to_rank,
        roles=("O", "C"),
        force_u=force_u,
        pad=cover_pad,
        same_label_only=False,
        log_prefix=prefix + "[cov0_oc]",
        emit_log=False,
    )


    boxes, scores, labels, masks = apply_same_role_coverage_suppression(
        boxes,
        scores,
        labels,
        masks,
        cover_thr=cover_thr,
        label_to_rank=label_to_rank,
        roles=("U",),
        force_u=force_u,
        pad=cover_pad,
        same_label_only=True,
        log_prefix=prefix + "[cov0_u]",
        emit_log=False,
    )


    drop_r05: Set[int] = set()
    drop_info_r05: List[Dict[str, Any]] = []
    if boxes is not None and scores is not None and labels is not None and img_hw is not None:
        for idx, lab in enumerate(labels):
            role = label_role(lab)
            if role not in ("C", "U"):
                continue
            contact_count = border_contact_count(boxes[idx], margin_px=3.0)
            area_ratio = box_area_ratio(boxes[idx])
            if contact_count >= 2 and area_ratio < (1.0 / 25.0):
                drop_r05.add(idx)
                drop_info_r05.append(
                    {
                        "rule": "r0_tiny_cu_two_border",
                        "idx": int(idx),
                        "label": lab,
                        "score": float(scores[idx].item()) if scores is not None else None,
                        "border_contact_count": int(contact_count),
                        "area_ratio": float(area_ratio),
                    }
                )

    drop_indices(drop_r05, "r0_tiny_cu_two_border", log=False)
    drops.extend(drop_info_r05)


    record_local_rel("raw", keep_below_thr=True)

    if boxes is None or scores is None or labels is None:
        return boxes, scores, labels, masks


    rel_raw = rel_dbg.get("raw", {})
    edges_raw = rel_raw.get("edges", [])
    by_child_raw = rel_raw.get("by_child", {})
    drop_u_r1: Set[int] = set()
    drop_info_r1: List[Dict[str, Any]] = []
    for idx, lab in enumerate(labels or []):
        if label_role(lab) != "U":
            continue
        allowed_parents = u_to_allowed_parents.get(lab, set())
        child_edges = [edges_raw[eid] for eid in by_child_raw.get(idx, []) if edges_raw[eid].get("type", "").endswith("-U")]
        if not allowed_parents:
            drop_u_r1.add(idx)
            drop_info_r1.append(
                {
                    "rule": "r1_not_in_allowed_parent",
                    "idx": int(idx),
                    "label": lab,
                    "score": float(scores[idx].item()),
                    "reason": "no_allowed_parents",
                    "allowed_parents": [],
                }
            )
            continue
        if not child_edges:
            drop_u_r1.add(idx)
            drop_info_r1.append(
                {
                    "rule": "r1_not_in_allowed_parent",
                    "idx": int(idx),
                    "label": lab,
                    "score": float(scores[idx].item()),
                    "reason": "no_parent_detected",
                    "allowed_parents": sorted(allowed_parents),
                }
            )
            continue
        max_cov = max(float(e.get("contain", 0.0)) for e in child_edges)
        has_pass = any(bool(e.get("pass_thr")) for e in child_edges)
        if not has_pass:
            drop_u_r1.add(idx)
            drop_info_r1.append(
                {
                    "rule": "r1_not_in_allowed_parent",
                    "idx": int(idx),
                    "label": lab,
                    "score": float(scores[idx].item()),
                    "reason": "contain_below_thr",
                    "max_cov": max_cov,
                    "allowed_parents": sorted(allowed_parents),
                }
            )

    drop_indices(drop_u_r1, "r1_not_in_allowed_parent", log=False)
    drops.extend(drop_info_r1)
    record_local_rel("after_r1", keep_below_thr=True)

    if boxes is None or scores is None or labels is None:
        return boxes, scores, labels, masks


    if drop_parent_without_u:
        rel_after_r1 = rel_dbg.get("after_r1", {})
        edges_r1 = rel_after_r1.get("edges", [])
        parents_with_u: Set[int] = set()
        for e in edges_r1:
            if e.get("type", "").endswith("-U") and e.get("pass_thr", False):
                parents_with_u.add(int(e.get("parent_idx", -1)))

        drop_parents: Set[int] = set()
        for idx, lab in enumerate(labels):
            role = label_role(lab)
            if role not in ("O", "C") or idx in parents_with_u:
                continue

            # Guard: keep objects that appear in remote relation candidates.
            if role == "O" and lab in remote_candidates:
                continue

            # For carriers, keep only if an owning object is remote-candidate AND detected this frame.
            if role == "C":
                parents_for_c = c_to_allowed_parents.get(lab, set())
                protect_owners = remote_candidates.intersection(parents_for_c).intersection(detected_objects)
                if protect_owners:
                    continue

            drop_parents.add(idx)

        drop_indices(drop_parents, "r2_parent_without_u")
        record_local_rel("after_r2", keep_below_thr=True)

    if boxes is None or scores is None or labels is None:
        return boxes, scores, labels, masks


    rel_after_r2 = rel_dbg.get("after_r2", rel_dbg.get("after_r1", rel_dbg.get("raw", {})))
    edges_r2 = rel_after_r2.get("edges", [])
    by_parent_r2 = rel_after_r2.get("by_parent", {})
    by_child_r2 = rel_after_r2.get("by_child", {})

    def _median(vals: List[float]) -> float:
        if not vals:
            return 0.0
        v = sorted(vals)
        mid = len(v) // 2
        if len(v) % 2 == 1:
            return v[mid]
        return 0.5 * (v[mid - 1] + v[mid])

    drop_u_r3: Set[int] = set()
    for u_idx in by_child_r2.keys():
        if labels is None or u_idx >= len(labels):
            continue
        if label_role(labels[u_idx]) != "U":
            continue
        score_u = float(scores[u_idx].item()) if scores is not None else 0.0
        if score_u >= u_low_score_thr:
            continue

        child_edges = [edges_r2[eid] for eid in by_child_r2.get(u_idx, []) if edges_r2[eid].get("type", "").endswith("-U") and edges_r2[eid].get("pass_thr", False)]
        if not child_edges:
            continue

        c_edges = [e for e in child_edges if e.get("parent_role") == "C"]
        o_edges = [e for e in child_edges if e.get("parent_role") == "O"]
        if c_edges:
            best_edge = max(c_edges, key=lambda e: float(e.get("contain", 0.0)))
        elif o_edges:
            best_edge = max(o_edges, key=lambda e: float(e.get("contain", 0.0)))
        else:
            continue

        parent_idx = int(best_edge.get("parent_idx", -1))
        parent_edges = [edges_r2[eid] for eid in by_parent_r2.get(parent_idx, []) if edges_r2[eid].get("type", "").endswith("-U") and edges_r2[eid].get("pass_thr", False)]
        sibling_us = [edge.get("child_idx", -1) for edge in parent_edges]
        sibling_us = [sid for sid in sibling_us if 0 <= sid < len(labels) and label_role(labels[sid]) == "U"]
        if len(sibling_us) <= 1:
            continue  # unique candidate, do not drop

        high_score_areas = [area(sid) for sid in sibling_us if scores is not None and float(scores[sid].item()) > u_low_score_thr]
        if len(high_score_areas) <= 1:
            continue  # not enough confident siblings to form a robust median

        median_area = _median(high_score_areas)
        if median_area <= 0:
            continue

        if area(u_idx) < u_small_area_ratio * median_area:
            drop_u_r3.add(u_idx)
            drops.append(
                {
                    "rule": "r3_lowconf_small_u",
                    "idx": int(u_idx),
                    "label": labels[u_idx],
                    "score": score_u,
                    "parent_idx": int(parent_idx),
                    "median_high_area": float(median_area),
                    "area": float(area(u_idx)),
                }
            )

    drop_indices(drop_u_r3, "r3_lowconf_small_u", log=False)
    record_local_rel("after_r3", keep_below_thr=True)

    if boxes is None or scores is None or labels is None:
        return boxes, scores, labels, masks


    rel_after_r3 = rel_dbg.get("after_r3", rel_after_r2)
    edges_r3 = rel_after_r3.get("edges", [])
    by_child_r3 = rel_after_r3.get("by_child", {})
    by_parent_r3 = rel_after_r3.get("by_parent", {})
    drop_r4: Set[int] = set()
    for u_idx in by_child_r3.keys():
        if labels is None or u_idx >= len(labels):
            continue
        if label_role(labels[u_idx]) != "U":
            continue
        if not near_border(boxes[u_idx]):
            continue
        child_edges = [edges_r3[eid] for eid in by_child_r3.get(u_idx, []) if edges_r3[eid].get("type", "").endswith("-U") and edges_r3[eid].get("pass_thr", False)]
        if not child_edges:
            continue

        c_edges = [e for e in child_edges if e.get("parent_role") == "C"]
        o_edges = [e for e in child_edges if e.get("parent_role") == "O"]
        if c_edges:
            best_edge = max(c_edges, key=lambda e: float(e.get("contain", 0.0)))
        elif o_edges:
            best_edge = max(o_edges, key=lambda e: float(e.get("contain", 0.0)))
        else:
            continue

        parent_idx = int(best_edge.get("parent_idx", -1))
        parent_role = best_edge.get("parent_role", "")
        parent_u_edges = [eid for eid in by_parent_r3.get(parent_idx, []) if edges_r3[eid].get("type", "").endswith("-U") and edges_r3[eid].get("pass_thr", False)]
        if len(parent_u_edges) != 1:
            continue

        drop_r4.add(u_idx)
        drop_r4.add(parent_idx)
        if parent_role == "C":
            oc_edges = [e for e in edges_r3 if e.get("type") == "O-C" and e.get("pass_thr", False) and int(e.get("child_idx", -1)) == parent_idx]
            for oc in oc_edges:
                drop_r4.add(int(oc.get("parent_idx", -1)))

    drop_indices(drop_r4, "r4_border_single_u")
    record_local_rel("after_r4", keep_below_thr=True)

    if boxes is None or scores is None or labels is None:
        return boxes, scores, labels, masks

    # Final snapshot (alias) after rules
    record_local_rel("after_cov", keep_below_thr=True)

    hints_pre: list[dict[str, Any]] = []
    for stage_name in ("after_cov", "after_r4", "after_r3", "after_r2", "after_r1", "raw"):
        stage_hints = list((rel_dbg.get(stage_name, {}).get("cabinet_hints", {}) or {}).get("carrier_hits", []) or [])
        if stage_hints:
            hints_pre = stage_hints
            break
    remapped_hints: list[dict[str, Any]] = []
    if boxes is not None and scores is not None and labels is not None:
        drop_blocked = {
            idx
            for idx, lab in enumerate(labels)
            if _policy_bool(policy, "is_suppressed_object", lab, "suppress_objects")
            or _policy_bool(policy, "is_hint_only_object", lab, "hint_only_objects")
        }
        drop_indices(drop_blocked, "r_final_blocked_object")
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(index_lineage)}

        for hit in hints_pre:
            old_child_idx = int(hit.get("child_idx_pre_drop", -1))
            if old_child_idx not in old_to_new_idx:
                continue
            new_hit = dict(hit)
            new_hit["child_idx"] = int(old_to_new_idx[old_child_idx])
            new_hit["cabinet_parent_idx"] = -1
            remapped_hints.append(new_hit)
    rel_dbg.setdefault("after_cov", {})["cabinet_hints"] = {"carrier_hits": remapped_hints}

    return boxes, scores, labels, masks


def build_stage2_u_prompts(
    selected_labels: Set[str],
    obj_to_carriers: Dict[str, List[str]],
    carrier_to_units: Dict[str, List[str]],
    obj_to_direct_units: Dict[str, List[str]],
) -> List[str]:
    stage2_prompts: List[str] = []
    seen = set()

    for lab in selected_labels:
        # As object: direct units + carriers' units
        for u in obj_to_direct_units.get(lab, []):
            if u and u not in seen:
                stage2_prompts.append(u)
                seen.add(u)
        for carrier in obj_to_carriers.get(lab, []):
            for u in carrier_to_units.get(carrier, []):
                if u and u not in seen:
                    stage2_prompts.append(u)
                    seen.add(u)

        # As carrier: its own units
        for u in carrier_to_units.get(lab, []):
            if u and u not in seen:
                stage2_prompts.append(u)
                seen.add(u)

    return stage2_prompts


def _merge_optional_tensors(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if a is None:
        return b
    if b is None:
        return a
    return torch.cat([a, b], dim=0)


class Sam3TwoStageRuntime:
    def __init__(
        self,
        checkpoint_path: str,
        bpe_path: str,
        device: str = "cuda",
        confidence: float = 0.3,
        thr_o: float = 0.5,
        thr_c: float = 0.5,
        thr_u: float = 0.5,
        overlap_iou: float = 0.9,
        u_cover_thr: float = 0.95,
        u_cover_pad: float = 0.0,
        u_parent_contain_thr: float = 0.9,
        force_u: Optional[Set[str]] = None,
        drop_parent_without_u: bool = True,
        same_type_drop_multi_parent_links: bool = False,
        processor_resolution: int = 1008,
    ) -> None:

        self.device = device
        self.confidence = confidence
        self.thr_o = thr_o
        self.thr_c = thr_c
        self.thr_u = thr_u
        self.overlap_iou = overlap_iou
        self.u_cover_thr = u_cover_thr
        self.u_cover_pad = u_cover_pad
        self.u_parent_contain_thr = u_parent_contain_thr
        self.force_u = force_u or set()
        self.drop_parent_without_u = bool(drop_parent_without_u)
        self.same_type_drop_multi_parent_links = bool(same_type_drop_multi_parent_links)
        self.processor_resolution = int(processor_resolution or 1008)
        self._printed_processor_resolution = False


        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            load_from_HF=False,
            device=device,
            eval_mode=True,
            image_size=self.processor_resolution,
        )

        self.processor = Sam3Processor(
            self.model,
            resolution=self.processor_resolution,
            device=device,
            confidence_threshold=confidence,
        )

    @torch.inference_mode()
    def run(
        self,
        uimg: torch.Tensor | np.ndarray,
        frame_result: Dict[str, Any],
        frame_idx: Optional[int] = None,
        policy: Optional[Any] = None,
    ) -> Sam3DetResult:

        if not self._printed_processor_resolution:
            print(f"[SAM3] processor_resolution={self.processor_resolution}")
            self._printed_processor_resolution = True

        (
            obj_prompts_standard,
            obj_prompts_hint_only,
            carrier_prompts,
            obj_to_carriers,
            carrier_to_units,
            obj_to_direct_units,
            label_to_rank,
            remote_candidate_objects,
            deferred_object_to_carriers,
            carrier_to_deferred_objects,
        ) = extract_prompts_from_frame_result(frame_result, policy=policy)
        u_to_allowed_parents = build_u_allowed_parents(obj_to_carriers, carrier_to_units, obj_to_direct_units, policy=policy)
        c_to_allowed_parents = build_c_allowed_parents(obj_to_carriers, policy=policy)
        graph_policy_debug = {
            "suppress_objects": sorted(set(getattr(policy, "suppress_objects", set()))),
            "hint_only_objects": sorted(set(getattr(policy, "hint_only_objects", set()))),
            "deferred_object_to_carriers": {key: list(value) for key, value in deferred_object_to_carriers.items()},
        }

        if len(obj_prompts_standard) == 0 and len(obj_prompts_hint_only) == 0 and len(carrier_prompts) == 0:
            return Sam3DetResult(
                None,
                None,
                None,
                None,
                per_prompt_stats={},
                stage2_prompts=[],
                selected_objects=[],
                label_to_rank=label_to_rank,
                u_to_allowed_parents=u_to_allowed_parents,
                c_to_allowed_parents=c_to_allowed_parents,
                cabinet_hints={"carrier_hits": []},
                graph_policy_debug=graph_policy_debug,
                sam3_processor_resolution=self.processor_resolution,
            )

        pil_img = uimg_to_pil(uimg)
        img_hw: Optional[Tuple[int, int]] = None
        if isinstance(uimg, torch.Tensor):
            if uimg.dim() >= 2:
                img_hw = (int(uimg.shape[0]), int(uimg.shape[1]))
        else:
            if hasattr(uimg, "shape") and len(uimg.shape) >= 2:
                img_hw = (int(uimg.shape[0]), int(uimg.shape[1]))

        state = self.processor.set_image(pil_img)


        stage1_prompts = _dedup_keep_order(obj_prompts_standard + obj_prompts_hint_only + carrier_prompts)
        state, per_stats_s1, boxes_s1, scores_s1, labels_s1, masks_s1, _ = run_prompts_collect(
            self.processor, state, stage1_prompts
        )

        boxes_s1t, scores_s1t, labels_s1t, masks_s1t = filter_by_role_threshold(
            boxes_s1, scores_s1, labels_s1, masks_s1, label_to_rank, self.thr_o, self.thr_c, self.thr_u
        )
        boxes_s1n, scores_s1n, labels_s1n, masks_s1n = apply_role_aware_nms(
            boxes_s1t, scores_s1t, labels_s1t, masks_s1t, self.overlap_iou, label_to_rank
        )


        selected_labels: Set[str] = set(labels_s1n or [])
        selected_objs = sorted(selected_labels)


        stage2_prompts = build_stage2_u_prompts(
            selected_labels,
            obj_to_carriers,
            carrier_to_units,
            obj_to_direct_units,
        )

        boxes_s2n: Optional[torch.Tensor] = None
        scores_s2n: Optional[torch.Tensor] = None
        labels_s2n: Optional[List[str]] = None
        masks_s2n: Optional[torch.Tensor] = None
        per_stats_s2: Dict[str, Dict[str, float]] = {}


        if len(stage2_prompts) > 0:
            state, per_stats_s2, boxes_s2, scores_s2, labels_s2, masks_s2, _ = run_prompts_collect(
                self.processor, state, stage2_prompts
            )
            boxes_s2t, scores_s2t, labels_s2t, masks_s2t = filter_by_role_threshold(
                boxes_s2, scores_s2, labels_s2, masks_s2, label_to_rank, self.thr_o, self.thr_c, self.thr_u
            )

            boxes_s2n, scores_s2n, labels_s2n, masks_s2n = boxes_s2t, scores_s2t, labels_s2t, masks_s2t

        boxes_all = _merge_optional_tensors(boxes_s1n, boxes_s2n)
        scores_all = _merge_optional_tensors(scores_s1n, scores_s2n)
        labels_all_list: List[str] = []
        if labels_s1n is not None:
            labels_all_list.extend(labels_s1n)
        if labels_s2n is not None:
            labels_all_list.extend(labels_s2n)
        labels_all: Optional[List[str]] = labels_all_list if labels_all_list else None
        masks_all = _merge_optional_tensors(masks_s1n, masks_s2n)


        boxes_all, scores_all, labels_all, masks_all = filter_by_role_threshold(
            boxes_all, scores_all, labels_all, masks_all, label_to_rank, self.thr_o, self.thr_c, self.thr_u
        )

        debug_out: Dict[str, Any] = {}

        boxes_final, scores_final, labels_final, masks_final = _post_merge_postfilters(
            boxes_all,
            scores_all,
            labels_all,
            masks_all,
            frame_idx=frame_idx,
            label_to_rank=label_to_rank,
            u_to_allowed_parents=u_to_allowed_parents,
            c_to_allowed_parents=c_to_allowed_parents,
            cover_thr=self.u_cover_thr,
            cover_pad=self.u_cover_pad,
            u_parent_contain_thr=self.u_parent_contain_thr,
            force_u=self.force_u,
            img_hw=img_hw,
            debug_out=debug_out,
            drop_parent_without_u=self.drop_parent_without_u,
            remote_candidate_objects=remote_candidate_objects,
            policy=policy,
            carrier_to_deferred_objects=carrier_to_deferred_objects,
            same_type_drop_multi_parent_links=self.same_type_drop_multi_parent_links,
        )

        per_stats_final = {**per_stats_s1}
        per_stats_final.update(per_stats_s2)
        return Sam3DetResult(
            boxes_final,
            scores_final,
            labels_final,
            masks_final,
            per_prompt_stats=per_stats_final,
            stage2_prompts=stage2_prompts,
            selected_objects=selected_objs,
            local_rel_debug=debug_out,
            label_to_rank=label_to_rank,
            u_to_allowed_parents=u_to_allowed_parents,
            c_to_allowed_parents=c_to_allowed_parents,
            cabinet_hints=(debug_out.get("local_rel", {}).get("after_cov", {}).get("cabinet_hints") or {"carrier_hits": []}),
            graph_policy_debug=graph_policy_debug,
            sam3_processor_resolution=self.processor_resolution,
        )
