from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mast3r_slam.semantic.io_utils import write_json_atomic
from mast3r_slam.semantic.sam3_two_stage_runtime import (
    ROLE_RANK,
    extract_prompts_from_frame_result,
)


@dataclass
class NodeAnchor:
    role: str
    label: str
    anchor_kf_idx: int
    anchor_score: float
    anchor_box_xyxy: Optional[List[float]]
    centroid_anchor: List[float]
    aabb_min_anchor: List[float]
    aabb_max_anchor: List[float]
    n_obs: int = 0


@dataclass
class EdgeAnchor:
    src: str
    dst: str
    edge_type: str
    relation: str
    anchor_kf_idx: int
    anchor_score: float
    src_box_xyxy: Optional[List[float]]
    dst_box_xyxy: Optional[List[float]]
    n_obs: int = 0
    evidence_sum: float = 0.0


class FSGAnchorFuser:
    def __init__(self, out_path):
        self.out_path = out_path
        self.nodes: Dict[str, NodeAnchor] = {}
        self.edges: Dict[Tuple[str, str, str], EdgeAnchor] = {}

    @staticmethod
    def _node_id(role: str, label: str) -> str:
        return f"{role}:{label}"

    @staticmethod
    def _role_of_label(label_to_rank: Dict[str, int], label: str) -> str:
        r = label_to_rank.get(label)
        if r is None:
            return "O"
        if r == ROLE_RANK["O"]:
            return "O"
        if r == ROLE_RANK["C"]:
            return "C"
        return "U"

    @staticmethod
    def _pick_best_det_per_label(det) -> Dict[str, int]:
        best: Dict[str, Tuple[float, int]] = {}
        if det is None or det.labels is None or det.scores is None:
            return {}
        for i, lab in enumerate(det.labels):
            if not isinstance(lab, str):
                continue
            score = float(det.scores[i])
            if lab not in best or score > best[lab][0]:
                best[lab] = (score, int(i))
        return {lab: idx for lab, (_, idx) in best.items()}

    @staticmethod
    def _mask_to_points(frame, mask_hw: torch.Tensor, max_points: int = 1024, conf_thr: float = 0.3):
        if frame.X_canon is None or frame.C is None:
            return None
        X = frame.X_canon
        C = frame.C
        if X.dim() == 3:
            X = X.reshape(-1, 3)
        if C.dim() == 3:
            C = C.reshape(-1, 1)
        mask = mask_hw.reshape(-1)
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return None
        conf = C[idx].reshape(-1)
        good = conf > conf_thr
        idx = idx[good]
        if idx.numel() == 0:
            return None
        if idx.numel() > max_points:
            perm = torch.randperm(idx.numel(), device=idx.device)[:max_points]
            idx = idx[perm]
        pts = X[idx]
        return pts, conf[good][: pts.shape[0]]

    @staticmethod
    def _view_score(det_score: float, mask_area: int, mean_conf: float, hw_area: int) -> float:
        area_norm = float(mask_area) / max(1, hw_area)
        return det_score * area_norm * mean_conf

    def update_keyframe(self, kf_idx: int, frame, frame_result: Dict[str, Any], sam3_out) -> None:
        if sam3_out is None or sam3_out.det is None:
            return
        det = sam3_out.det
        remote_rel_2d = sam3_out.remote_rel_2d or {}

        _, _, _, _, _, _, label_to_rank, _, _, _ = extract_prompts_from_frame_result(frame_result)
        best_det = self._pick_best_det_per_label(det)

        if det.masks is None or det.boxes is None:
            return

        if isinstance(frame.img_shape, torch.Tensor):
            hw = frame.img_shape.reshape(-1).tolist()
        else:
            hw = list(frame.img_shape)
        if len(hw) < 2:
            return
        H, W = int(hw[0]), int(hw[1])
        hw_area = H * W

        # Nodes
        for label, di in best_det.items():
            role = self._role_of_label(label_to_rank, label)
            node_id = self._node_id(role, label)

            mask_raw = det.masks[di].float()
            if mask_raw.dim() == 2:
                mask0 = mask_raw[None, None, ...]
            elif mask_raw.dim() == 3:
                mask0 = mask_raw[None, ...]
            elif mask_raw.dim() == 4:
                mask0 = mask_raw
            else:
                continue
            if mask0.shape[-2:] != (H, W):
                mask0 = F.interpolate(mask0, size=(H, W), mode="nearest")
            mask = mask0[0, 0] > 0.5
            pts_conf = self._mask_to_points(frame, mask)
            if pts_conf is None:
                continue
            pts, conf = pts_conf

            centroid = pts.mean(dim=0).detach().cpu().numpy()
            pmin = pts.min(dim=0).values.detach().cpu().numpy()
            pmax = pts.max(dim=0).values.detach().cpu().numpy()

            det_score = float(det.scores[di])
            mean_conf = float(conf.mean().item()) if conf.numel() else 0.0
            view_score = self._view_score(det_score, int(mask.sum().item()), mean_conf, hw_area)
            box = det.boxes[di].detach().cpu().tolist()

            if node_id not in self.nodes:
                self.nodes[node_id] = NodeAnchor(
                    role=role,
                    label=label,
                    anchor_kf_idx=kf_idx,
                    anchor_score=view_score,
                    anchor_box_xyxy=box,
                    centroid_anchor=centroid.tolist(),
                    aabb_min_anchor=pmin.tolist(),
                    aabb_max_anchor=pmax.tolist(),
                    n_obs=1,
                )
            else:
                node = self.nodes[node_id]
                node.n_obs += 1
                node.centroid_anchor = (
                    (np.array(node.centroid_anchor) * (node.n_obs - 1) + centroid) / node.n_obs
                ).tolist()
                node.aabb_min_anchor = np.minimum(np.array(node.aabb_min_anchor), pmin).tolist()
                node.aabb_max_anchor = np.maximum(np.array(node.aabb_max_anchor), pmax).tolist()
                if view_score > node.anchor_score:
                    node.anchor_kf_idx = kf_idx
                    node.anchor_score = view_score
                    node.anchor_box_xyxy = box

        # Local edges
        for e in self._iter_local_edges(frame_result):
            self._update_edge(e, kf_idx, best_det, det)

        # Remote edges: confirmed_visible only
        confirmed_vis = remote_rel_2d.get("confirmed_visible", [])
        rel_map = {
            (c.get("from_object"), c.get("to_object")): (c.get("relation") or "")
            for c in frame_result.get("remote_relation_candidates", [])
        }
        for ce in confirmed_vis:
            a, b = ce.get("from_object"), ce.get("to_object")
            if not a or not b:
                continue
            rel = rel_map.get((a, b), "")
            self._update_remote_edge(a, b, rel, ce.get("evidence_sum", 0.0), kf_idx, best_det, det)

        self.save()

    def _iter_local_edges(self, frame_result: Dict[str, Any]):
        for obj in frame_result.get("present", []) or []:
            o = obj.get("object")
            if not o:
                continue
            for fc in obj.get("functional_carriers", []) or []:
                c = fc.get("carrier")
                if not c:
                    continue
                oc_rel = fc.get("oc_relation", "")
                yield ("O", o, "C", c, "oc", oc_rel)

                for iu in fc.get("interactive_units", []) or []:
                    u = iu.get("unit") if isinstance(iu, dict) else iu
                    if not u:
                        continue
                    yield ("C", c, "U", u, "cu", iu.get("cu_relation", "") if isinstance(iu, dict) else "")
                    yield ("O", o, "U", u, "ou", iu.get("ou_relation", "") if isinstance(iu, dict) else "")

            for du in obj.get("direct_interactive_units", []) or []:
                u = du.get("unit") if isinstance(du, dict) else du
                if not u:
                    continue
                yield ("O", o, "U", u, "ou", du.get("ou_relation", "") if isinstance(du, dict) else "")

    def _update_edge(self, e, kf_idx, best_det, det):
        src_role, src_lab, dst_role, dst_lab, etype, rel = e
        src = self._node_id(src_role, src_lab)
        dst = self._node_id(dst_role, dst_lab)
        key = (src, dst, etype)
        si = best_det.get(src_lab)
        di = best_det.get(dst_lab)
        if si is None or di is None:
            return
        sscore = float(det.scores[si])
        dscore = float(det.scores[di])
        score = min(sscore, dscore)
        sbox = det.boxes[si].detach().cpu().tolist()
        dbox = det.boxes[di].detach().cpu().tolist()

        if key not in self.edges:
            self.edges[key] = EdgeAnchor(
                src=src,
                dst=dst,
                edge_type=etype,
                relation=rel,
                anchor_kf_idx=kf_idx,
                anchor_score=score,
                src_box_xyxy=sbox,
                dst_box_xyxy=dbox,
                n_obs=1,
            )
        else:
            ed = self.edges[key]
            ed.n_obs += 1
            if score > ed.anchor_score:
                ed.anchor_kf_idx = kf_idx
                ed.anchor_score = score
                ed.src_box_xyxy = sbox
                ed.dst_box_xyxy = dbox

    def _update_remote_edge(self, a, b, rel, evidence_sum, kf_idx, best_det, det):
        src = self._node_id("O", a)
        dst = self._node_id("O", b)
        key = (src, dst, "remote")
        si = best_det.get(a)
        di = best_det.get(b)
        if si is None or di is None:
            return
        score = min(float(det.scores[si]), float(det.scores[di]))
        sbox = det.boxes[si].detach().cpu().tolist()
        dbox = det.boxes[di].detach().cpu().tolist()

        if key not in self.edges:
            self.edges[key] = EdgeAnchor(
                src=src,
                dst=dst,
                edge_type="remote",
                relation=rel,
                anchor_kf_idx=kf_idx,
                anchor_score=score,
                src_box_xyxy=sbox,
                dst_box_xyxy=dbox,
                n_obs=1,
                evidence_sum=float(evidence_sum),
            )
        else:
            ed = self.edges[key]
            ed.n_obs += 1
            ed.evidence_sum = max(ed.evidence_sum, float(evidence_sum))
            if score > ed.anchor_score:
                ed.anchor_kf_idx = kf_idx
                ed.anchor_score = score
                ed.src_box_xyxy = sbox
                ed.dst_box_xyxy = dbox

    def save(self) -> None:
        out = {
            "version": 1,
            "nodes": {k: vars(v) for k, v in self.nodes.items()},
            "edges": [vars(v) for v in self.edges.values()],
        }
        write_json_atomic(self.out_path, out)
