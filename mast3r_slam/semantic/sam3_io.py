from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_ROLE_NODE_COLORS = {
    "O": (0, 0, 255),      # red
    "C": (0, 255, 255),    # yellow
    "U": (255, 0, 0),      # blue
}

_EDGE_COLORS = {
    "O-C": (0, 255, 0),    # green
    "C-U": (255, 0, 255),  # purple
    "O-U": (0, 165, 255),  # orange
}

_LOCAL_REL_STAGE_GUIDE: List[Tuple[str, str, str]] = [
    ("raw", "S0", "Raw local-relation candidates before rule-based pruning."),
    ("after_r1", "S1", "Relations retained after the first rule stage."),
    ("after_r2", "S2", "Relations retained after the second rule stage."),
    ("after_r3", "S3", "Relations retained after the third rule stage."),
    ("after_r4", "S4", "Relations retained after the fourth rule stage."),
    ("after_cov", "S5", "Final local relations after coverage filtering."),
]


def _summarize_stage(stage_payload: Dict[str, Any]) -> Dict[str, Any]:
    edges = stage_payload.get("edges") or []
    type_counts: Dict[str, int] = {}
    selected = 0
    pass_thr = 0
    for e in edges:
        et = str(e.get("type") or "UNKNOWN")
        type_counts[et] = type_counts.get(et, 0) + 1
        if bool(e.get("selected")):
            selected += 1
        if bool(e.get("pass_thr")):
            pass_thr += 1

    assignments = stage_payload.get("assignments") or {}
    assignment_items = sum(len(v) for v in assignments.values() if isinstance(v, dict))
    return {
        "edges_total": len(edges),
        "edges_selected": selected,
        "edges_pass_thr": pass_thr,
        "edge_type_counts": type_counts,
        "assignment_children": len(assignments),
        "assignment_items": int(assignment_items),
    }


def _inject_local_rel_stage_intros(local_rel_debug: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
    local_rel = local_rel_debug.get("local_rel") if isinstance(local_rel_debug, dict) else None
    if not isinstance(local_rel, dict):
        return local_rel_debug, [], {}

    stage_legend: List[Dict[str, str]] = []
    stage_summaries: Dict[str, Dict[str, Any]] = {}
    rebuilt_local_rel: Dict[str, Any] = {}
    consumed: set[str] = set()

    for stage_key, stage_tag, stage_desc in _LOCAL_REL_STAGE_GUIDE:
        if stage_key not in local_rel:
            continue
        marker = f"[{stage_tag}] {stage_key}"
        stage_legend.append({"marker": marker, "stage": stage_key, "description": stage_desc})
        stage_payload = local_rel.get(stage_key) or {}
        stage_summary = _summarize_stage(stage_payload if isinstance(stage_payload, dict) else {})
        stage_summaries[stage_key] = stage_summary

        rebuilt_local_rel[f"__stage_intro__{stage_key}"] = {
            "marker": marker,
            "description": stage_desc,
            "how_to_read": "Inspect edges_selected/edges_pass_thr, then edge_type_counts.",
            "summary": stage_summary,
        }
        rebuilt_local_rel[stage_key] = stage_payload
        consumed.add(stage_key)

    # Keep any non-standard stage keys to preserve original debug information.
    for k, v in local_rel.items():
        if k in consumed:
            continue
        rebuilt_local_rel[k] = v

    rebuilt_debug = dict(local_rel_debug)
    rebuilt_debug["local_rel"] = rebuilt_local_rel
    return rebuilt_debug, stage_legend, stage_summaries


def _sam3_det_to_json(det_result, extra: Optional[dict] = None) -> dict:
    detections = []
    if det_result.boxes is not None and det_result.scores is not None and det_result.labels is not None:
        for idx, (box, score, label) in enumerate(zip(det_result.boxes, det_result.scores, det_result.labels)):
            detections.append(
                {
                    "idx": int(idx),
                    "label": label,
                    "score": float(score),
                    "box_xyxy": [float(x) for x in box.tolist()],
                }
            )

    local_rel_debug = det_result.local_rel_debug or {}
    local_rel_debug_readable, stage_legend, stage_summaries = _inject_local_rel_stage_intros(local_rel_debug)

    out = {
        "__debug_readme__": {
            "summary": "This file contains SAM3 detections and local-relation diagnostics.",
            "stage_legend": stage_legend,
            "note": "Each local_rel stage is preceded by its matching __stage_intro__* marker.",
        },
        "stage_summaries": stage_summaries,
        "detections": detections,
        "per_prompt_stats": getattr(det_result, "per_prompt_stats", None) or {},
        "selected_objects": getattr(det_result, "selected_objects", None) or [],
        "stage2_prompts": getattr(det_result, "stage2_prompts", None) or [],
        "local_rel_debug": local_rel_debug_readable,
        "sam3_processor": {
            "resolution": getattr(det_result, "sam3_processor_resolution", None),
        },
    }
    if extra:
        out.update(extra)
    return out


def _color_for_instance(label: str, idx: int):
    h = hashlib.md5(f"{label}-{idx}".encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)
    rng = np.random.RandomState(seed)
    r, g, b = (rng.rand(3) * 255).astype(np.uint8)
    return int(b), int(g), int(r)


def _darken_bgr(color: tuple[int, int, int], factor: float = 0.45) -> tuple[int, int, int]:
    fac = float(np.clip(factor, 0.05, 0.95))
    return tuple(int(np.clip(round(float(ch) * fac), 0, 255)) for ch in color)


def _det_text_style_for_box_color(box_color: tuple[int, int, int]) -> dict[str, Any]:
    text_color = _darken_bgr(box_color, factor=0.45)
    return {
        "font_scale": 0.3,
        "thickness": 1,
        "text_color": text_color,
    }


def _save_sam3_vis(out_path, uimg, det_result, remote_rel_2d: Optional[Dict[str, Any]] = None, alpha: float = 0.35, thr: float = 0.5) -> None:
    if isinstance(uimg, torch.Tensor):
        img_np = (uimg.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    else:
        arr = np.asarray(uimg)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        img_np = arr

    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    H, W = img_np.shape[:2]

    def _text_anchor_for_box(
        text: str,
        x1: int,
        y1: int,
        y2: int,
        font_scale: float = 0.3,
        thickness: int = 1,
    ) -> Tuple[int, int]:
        """Choose a visible text baseline: above box when possible, otherwise inside/below with clamping."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        margin = 4

        # Preferred: above the box.
        x = x1
        y = y1 - margin

        # If text would be outside top boundary, move baseline inside box first.
        if y - th < 0:
            y = y1 + th + margin
            # If box is too small or still out, try just below the box.
            if y >= H:
                y = y2 + th + margin

        # Final clamp to keep the full text region inside image bounds.
        x = max(0, min(W - tw - 1, x))
        y = max(th + 1, min(H - 1, y))
        return x, y

    if det_result.boxes is not None and det_result.scores is not None and det_result.labels is not None:
        masks = det_result.masks
        for idx, (box, score, label) in enumerate(zip(det_result.boxes, det_result.scores, det_result.labels)):
            color = _color_for_instance(label, idx)
            text_style = _det_text_style_for_box_color(color)

            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            text = f"{idx}:{label}:{float(score):.2f}"
            tx, ty = _text_anchor_for_box(
                text,
                x1,
                y1,
                y2,
                font_scale=float(text_style["font_scale"]),
                thickness=int(text_style["thickness"]),
            )
            cv2.putText(
                img_np,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(text_style["font_scale"]),
                tuple(int(v) for v in text_style["text_color"]),
                int(text_style["thickness"]),
                cv2.LINE_AA,
            )

            if masks is not None:
                m = masks[idx].detach().float()
                if m.ndim > 2:
                    m = m.squeeze()
                    if m.ndim > 2:
                        m = m[0]

                if (m.min() < 0) or (m.max() > 1.0):
                    m = torch.sigmoid(m)

                m_rs = F.interpolate(m[None, None, ...], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                m_bin = (m_rs > thr).cpu().numpy()

                overlay = np.zeros_like(img_np, dtype=np.float32)
                overlay[..., 0] = color[0]
                overlay[..., 1] = color[1]
                overlay[..., 2] = color[2]

                img_np[m_bin] = (
                    img_np[m_bin].astype(np.float32) * (1 - alpha) + overlay[m_bin] * alpha
                ).astype(np.uint8)

    # Draw 2D hierarchy graph overlay using bbox centers as node positions.
    local_rel_debug = det_result.local_rel_debug or {}
    local_rel = local_rel_debug.get("local_rel") or {}
    _, stage = _pick_local_rel_stage(local_rel)
    edges = (stage or {}).get("edges") or []

    if det_result.boxes is not None and len(edges) > 0:
        boxes_np = det_result.boxes.detach().cpu().numpy()
        num_boxes = int(boxes_np.shape[0])

        def _idx(v: Any) -> Optional[int]:
            if v is None:
                return None
            try:
                i = int(v)
            except (TypeError, ValueError):
                return None
            if i < 0 or i >= num_boxes:
                return None
            return i

        def _center(i: int) -> Tuple[int, int]:
            x1, y1, x2, y2 = [float(x) for x in boxes_np[i].tolist()]
            cx = int(round((x1 + x2) * 0.5))
            cy = int(round((y1 + y2) * 0.5))
            cx = max(0, min(W - 1, cx))
            cy = max(0, min(H - 1, cy))
            return cx, cy

        node_role: Dict[int, str] = {}
        for e in edges:
            ci = _idx(e.get("child_idx"))
            pi = _idx(e.get("parent_idx"))
            cr = str(e.get("child_role") or "").upper()
            pr = str(e.get("parent_role") or "").upper()
            if ci is not None and cr in _ROLE_NODE_COLORS:
                node_role.setdefault(ci, cr)
            if pi is not None and pr in _ROLE_NODE_COLORS:
                node_role.setdefault(pi, pr)

        if det_result.labels is not None and det_result.label_to_rank:
            rank_to_role = {0: "U", 1: "C", 2: "O"}
            for i, lab in enumerate(det_result.labels):
                if i in node_role:
                    continue
                rank = det_result.label_to_rank.get(lab)
                role = rank_to_role.get(rank)
                if role in _ROLE_NODE_COLORS:
                    node_role[i] = role

        selected_edges = [e for e in edges if bool(e.get("selected"))]
        pass_thr_edges = [e for e in edges if bool(e.get("pass_thr"))]
        draw_edges = selected_edges if len(selected_edges) > 0 else (pass_thr_edges if len(pass_thr_edges) > 0 else edges)

        for e in draw_edges:
            et = str(e.get("type") or "").upper()
            color = _EDGE_COLORS.get(et)
            if color is None:
                continue
            ci = _idx(e.get("child_idx"))
            pi = _idx(e.get("parent_idx"))
            if ci is None or pi is None:
                continue
            p0 = _center(pi)
            p1 = _center(ci)
            cv2.line(img_np, p0, p1, color, 2, cv2.LINE_AA)

        # Draw remote links in white for debugging cross-object relations.
        remote_edges = (remote_rel_2d or {}).get("observed_candidates") or []
        for re in remote_edges:
            src = _idx(re.get("from_det_idx"))
            dst = _idx(re.get("to_det_idx"))
            if src is None or dst is None:
                continue
            p0 = _center(src)
            p1 = _center(dst)
            cv2.line(img_np, p0, p1, (255, 255, 255), 2, cv2.LINE_AA)

        radius = max(3, int(round(min(H, W) * 0.006)))
        for idx, role in node_role.items():
            color = _ROLE_NODE_COLORS.get(role)
            if color is None:
                continue
            cxy = _center(idx)
            cv2.circle(img_np, cxy, radius, color, -1, cv2.LINE_AA)
            cv2.circle(img_np, cxy, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_np)


def _pick_local_rel_stage(local_rel: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    for key in ("after_cov", "after_r4", "after_r3", "after_r2", "after_r1", "raw"):
        if key in local_rel:
            return key, local_rel[key]
    return None, None


def _infer_child_role(idx: int, edges: List[Dict[str, Any]]) -> Optional[str]:
    for edge in edges:
        if edge.get("child_idx") == idx and edge.get("child_role"):
            return edge["child_role"]
    return None


def _summarize_candidates(
    assignment: Dict[str, Any],
    edges_by_eid: Dict[int, Dict[str, Any]],
    label_by_idx: Dict[int, str],
) -> List[Dict[str, Any]]:
    summarized: List[Dict[str, Any]] = []

    cand_list = assignment.get("candidates") or []
    cand_eids = assignment.get("candidate_eids") or []

    if cand_list:
        for cand in cand_list:
            parent_idx = cand.get("parent_idx")
            summarized.append(
                {
                    "eid": cand.get("eid"),
                    "parent_idx": parent_idx,
                    "parent_label": label_by_idx.get(parent_idx) or cand.get("parent_label"),
                    "contain": cand.get("contain"),
                    "mask_contain": cand.get("mask_contain"),
                }
            )
    elif cand_eids:
        for eid in cand_eids:
            edge = edges_by_eid.get(eid)
            if edge is None:
                continue
            parent_idx = edge.get("parent_idx")
            summarized.append(
                {
                    "eid": eid,
                    "parent_idx": parent_idx,
                    "parent_label": label_by_idx.get(parent_idx) or edge.get("parent_label"),
                    "contain": edge.get("contain"),
                    "mask_contain": edge.get("mask_contain"),
                }
            )

    return summarized


def _build_instances_view(det_result, detections: List[Dict[str, Any]]):
    local_rel_debug = det_result.local_rel_debug or {}
    local_rel = local_rel_debug.get("local_rel") or {}
    stage_name, stage = _pick_local_rel_stage(local_rel)
    if stage is None:
        return {}, stage_name

    edges = stage.get("edges") or []
    assignments = stage.get("assignments") or {}
    drops = local_rel_debug.get("drops") or []

    edges_by_eid = {eid: edge for eid, edge in enumerate(edges)}
    label_by_idx = {det["idx"]: det["label"] for det in detections if "idx" in det}

    instances: Dict[str, Any] = {}
    for det in detections:
        idx = det["idx"]
        instance: Dict[str, Any] = {
            "label": det["label"],
            "score": det["score"],
            "box_xyxy": det["box_xyxy"],
        }

        role = _infer_child_role(idx, edges)
        if role:
            instance["role"] = role

        idx_drops = [drop for drop in drops if drop.get("idx") == idx]
        if idx_drops:
            instance["drops"] = idx_drops

        child_assignments = assignments.get(str(idx)) or assignments.get(idx)
        if child_assignments:
            summarized_assignments: Dict[str, Any] = {}
            for edge_type, assignment in child_assignments.items():
                if not isinstance(assignment, dict):
                    continue

                summary: Dict[str, Any] = {
                    "status": assignment.get("status"),
                    "edge_type": assignment.get("edge_type") or edge_type,
                    "chosen_eid": assignment.get("chosen_eid"),
                }

                candidate_eids = assignment.get("candidate_eids") or []
                if candidate_eids:
                    summary["candidate_eids"] = candidate_eids

                chosen_parent_idx = assignment.get("chosen_parent_idx")
                if chosen_parent_idx is not None:
                    summary["chosen_parent_idx"] = chosen_parent_idx
                    summary["chosen_parent_label"] = label_by_idx.get(chosen_parent_idx)

                candidates = _summarize_candidates(assignment, edges_by_eid, label_by_idx)
                if candidates:
                    summary["candidates"] = candidates

                summarized_assignments[edge_type] = summary

            if summarized_assignments:
                instance["assignments"] = summarized_assignments

        instances[str(idx)] = instance

    return instances, stage_name
