from __future__ import annotations


from dataclasses import dataclass
from collections import deque
import pathlib
from typing import Any, Dict, Optional
import cv2
import numpy as np

from mast3r_slam.functional_graph.policy import FunctionalGraphPolicy
from mast3r_slam.functional_graph.geometry_utils import box_touches_image_border
from mast3r_slam.semantic.merge_semantics import (
    atlas_object_carrier_set,
    atlas_object_set,
    build_graph_frame_result,
    build_frame_result_final,
    remote_endpoints_from_pairs,
)
from mast3r_slam.semantic.io_utils import append_jsonl, write_json_atomic
from mast3r_slam.semantic.sam3_io import _sam3_det_to_json, _save_sam3_vis


@dataclass
class DeepSeekOut:

    frame_result: dict
    graph_frame_result: dict
    incremental: Optional[dict] = None


@dataclass
class Sam3Out:
    det: Any
    remote_rel_2d: Optional[Dict[str, Any]] = None
    debug_image_path: Optional[str] = None


def _filter_border_no_link_detections(det, img_h: int, img_w: int, margin_px: int = 2):
    """Remove detections at image border that have no parent/child links.

    Modifies ``det`` in place: removes boxes/scores/labels/masks entries
    and re-indexes all edge references in ``local_rel_debug``.
    """
    import torch

    boxes = getattr(det, "boxes", None)
    labels = getattr(det, "labels", None)
    if boxes is None or labels is None or len(labels) == 0:
        return

    n_det = len(labels)

    # Build linked-det-idx set from local_rel_debug edges (final stage).
    linked_idxs: set[int] = set()
    local_rel = getattr(det, "local_rel_debug", None) or {}
    # Try stages in order of preference: final → earlier
    for stage_name in ("after_cov", "after_r4", "after_r3", "after_r2", "after_r1", "raw"):
        stage_key = "local_rel" if isinstance(local_rel, dict) and "local_rel" in local_rel else None
        stage_data = (local_rel.get("local_rel", {}) if stage_key else local_rel).get(stage_name, {})
        edges = stage_data.get("edges", [])
        if edges:
            for edge in edges:
                if not edge.get("selected"):
                    continue
                ci = int(edge.get("child_idx", -1))
                pi = int(edge.get("parent_idx", -1))
                if ci >= 0:
                    linked_idxs.add(ci)
                if pi >= 0:
                    linked_idxs.add(pi)
            break  # use the first stage with edges

    # Determine which indices to keep.
    keep: list[int] = []
    for i in range(n_det):
        box = boxes[i].detach().cpu().tolist() if torch.is_tensor(boxes) else list(boxes[i])
        at_border = box_touches_image_border(box, img_h, img_w, margin_px=margin_px)
        if at_border and i not in linked_idxs:
            continue  # drop
        keep.append(i)

    if len(keep) == n_det:
        return  # nothing to filter

    # Build old→new index mapping.
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(keep)}

    # Filter parallel arrays.
    keep_t = torch.tensor(keep, dtype=torch.long)
    if torch.is_tensor(boxes):
        det.boxes = boxes[keep_t]
    if getattr(det, "scores", None) is not None and torch.is_tensor(det.scores):
        det.scores = det.scores[keep_t]
    if getattr(det, "masks", None) is not None and torch.is_tensor(det.masks):
        det.masks = det.masks[keep_t]
    det.labels = [labels[i] for i in keep]

    # Re-index edges in ALL stages of local_rel_debug.
    inner_rel = local_rel.get("local_rel", local_rel) if isinstance(local_rel, dict) else {}
    if isinstance(inner_rel, dict):
        for stage_name, stage_data in inner_rel.items():
            if not isinstance(stage_data, dict):
                continue
            new_edges = []
            for edge in stage_data.get("edges", []):
                ci = int(edge.get("child_idx", -1))
                pi = int(edge.get("parent_idx", -1))
                if ci in old_to_new and pi in old_to_new:
                    edge = dict(edge)
                    edge["child_idx"] = old_to_new[ci]
                    edge["parent_idx"] = old_to_new[pi]
                    new_edges.append(edge)
            stage_data["edges"] = new_edges
            # Re-index by_child / by_parent if present.
            for key in ("by_child", "by_parent"):
                mapping = stage_data.get(key)
                if isinstance(mapping, dict):
                    new_mapping = {}
                    for old_k, v in mapping.items():
                        new_k = old_to_new.get(int(old_k))
                        if new_k is not None:
                            new_mapping[new_k] = v
                    stage_data[key] = new_mapping
            # Re-index assignments.
            assignments = stage_data.get("assignments")
            if isinstance(assignments, dict):
                new_assignments = {}
                for old_k, v in assignments.items():
                    new_k = old_to_new.get(int(old_k))
                    if new_k is not None:
                        new_assignments[new_k] = v
                stage_data["assignments"] = new_assignments

    # Re-index cabinet_hints if present.
    cab_hints = getattr(det, "cabinet_hints", None)
    if isinstance(cab_hints, dict):
        new_hits = []
        for hit in cab_hints.get("carrier_hits", []):
            ci = int(hit.get("child_idx", -1))
            pi = int(hit.get("cabinet_parent_idx", -1))
            if ci in old_to_new:
                hit = dict(hit)
                hit["child_idx"] = old_to_new[ci]
                if pi in old_to_new:
                    hit["cabinet_parent_idx"] = old_to_new[pi]
                new_hits.append(hit)
        cab_hints["carrier_hits"] = new_hits


class SemanticPipeline:

    def __init__(
        self,
        args,
        rampp=None,
        deepseek=None,
        scene_lock=None,
        sam3_runtime=None,
        frames_out_dir: Optional[pathlib.Path] = None,
        scene_history_path: Optional[pathlib.Path] = None,
        scene_json_path: Optional[pathlib.Path] = None,
        atlas_json_path: Optional[pathlib.Path] = None,
        sam3_save_vis: bool = False,
        graph_policy: Optional[FunctionalGraphPolicy] = None,
    ):

        self.args = args
        self.rampp = rampp
        self.deepseek = deepseek
        self.scene_lock = scene_lock
        self.sam3_runtime = sam3_runtime


        self.frames_out_dir = frames_out_dir
        self.scene_history_path = scene_history_path
        self.scene_json_path = scene_json_path
        self.atlas_json_path = atlas_json_path
        self.sam3_save_vis = sam3_save_vis
        self.graph_policy = graph_policy or FunctionalGraphPolicy()


        self.scene_type = None
        self.global_atlas = None
        self.last_scene_result = None
        self.pre_atlas_notice_printed = False

        # Remote 2D relation tracking
        self.remote_win_size = int(getattr(args, "remote_win_size", 30))
        self.remote_confirm_k = int(getattr(args, "remote_confirm_k", 2))
        self.remote_ambig_delta = float(getattr(args, "remote_ambig_delta", 0.2))
        self._remote_frame_cache = deque(maxlen=self.remote_win_size)
        self._remote_edge_stats: Dict[tuple[str, str], dict] = {}
        self.fixed_atlas_source: Optional[str] = None

    @staticmethod
    def _norm_tag(s: str | None) -> str:

        return (s or "").strip().lower()

    @staticmethod
    def _canon_tag(s: str | None) -> str:
        s = (s or "").strip().lower()
        s = s.replace("_", " ").replace("-", " ")
        s = " ".join(s.split())
        return s

    @staticmethod
    def _map_tag_to_atlas(tag: str, atlas_objs: set[str]) -> str:
        t = SemanticPipeline._canon_tag(tag)
        if not t:
            return t

        room_prefixes = ("kitchen ", "bathroom ", "living room ", "bedroom ", "dining room ")
        for p in room_prefixes:
            if t.startswith(p):
                rest = t[len(p) :].strip()
                if rest in atlas_objs:
                    return rest

        compact = t.replace(" ", "")
        compact_map = {o.replace(" ", ""): o for o in atlas_objs}
        if compact in compact_map:
            return compact_map[compact]

        return t

    def set_fixed_atlas(self, atlas: dict, *, source: str = "") -> None:
        """Use a fixed atlas for every frame without RAM++/DeepSeek calls."""
        self.global_atlas = dict(atlas or {})
        self.scene_type = str(self.global_atlas.get("scene_type") or "fixed")
        self.last_scene_result = {
            "scene_type": self.scene_type,
            "confidence": 1.0,
            "reason": "fixed_semantic_atlas",
        }
        self.fixed_atlas_source = source or "inline"
        if self.scene_json_path is not None:
            write_json_atomic(
                self.scene_json_path,
                {
                    "scene_type": self.scene_type,
                    "confidence": 1.0,
                    "locked_at_frame": 0,
                    "source": "fixed_semantic_atlas",
                    "atlas_source": self.fixed_atlas_source,
                },
            )
        if self.atlas_json_path is not None:
            write_json_atomic(self.atlas_json_path, self.global_atlas)
        print(
            f"[Fixed semantic atlas] loaded {len(self.global_atlas.get('objects', []) or [])} objects "
            f"from {self.fixed_atlas_source}"
        )

    def run_rampp(self, frame_idx: int, img) -> Optional[dict]:

        if self.rampp is None:
            return None
        tags = self.rampp.infer(img)
        if frame_idx % self.args.rampp_print_every == 0:
            print("[RAM++]", tags["tags_en"][:10])
        if self.frames_out_dir is not None:

            self.frames_out_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.frames_out_dir / f"frame_{frame_idx:06d}_tags.json", tags)
        return tags

    def run_deepseek(self, frame_idx: int, tags: dict) -> DeepSeekOut:

        if self.deepseek is None:
            raise RuntimeError("DeepSeek runtime is not initialized")


        norm_tags_list = [self._norm_tag(t) for t in tags["tags_en"] if self._norm_tag(t)]
        norm_tags = set(norm_tags_list)


        if self.global_atlas is None:
            scene_result = self.deepseek.classify_scene(tags["tags_en"])
            self.last_scene_result = scene_result

            lock_info = self.scene_lock.update(scene_result.get("scene_type"), scene_result.get("confidence", 0.0))

            if self.scene_history_path is not None:

                append_jsonl(
                    self.scene_history_path,
                    {
                        "frame": frame_idx,
                        "scene_type": scene_result.get("scene_type"),
                        "confidence": scene_result.get("confidence"),
                        "locked": lock_info["locked"],
                        "streak": lock_info["streak"],
                        "reason": lock_info.get("reason"),
                        "source": "live",
                    },
                )

            if lock_info["locked"] and self.global_atlas is None:

                self.scene_type = lock_info.get("scene_type")
                self.global_atlas = self.deepseek.get_or_build_global_atlas(self.scene_type)
                if self.scene_json_path is not None:
                    write_json_atomic(
                        self.scene_json_path,
                        {
                            "scene_type": self.scene_type,
                            "confidence": scene_result.get("confidence"),
                            "locked_at_frame": frame_idx,
                        },
                    )
                if self.atlas_json_path is not None:
                    write_json_atomic(self.atlas_json_path, self.global_atlas)
                print(f"[DeepSeek scene locked] {self.scene_type}")
                print("[DeepSeek atlas] built/loaded")
        else:

            scene_switch_enabled = bool(getattr(self.args, "enable_scene_switch", False))
            scene_switch_disabled = bool(getattr(self.args, "disable_scene_switch", False))
            recheck_every = int(getattr(self.args, "scene_recheck_every", 0) or 0)
            should_recheck = (
                scene_switch_enabled
                and not scene_switch_disabled
                and recheck_every > 0
                and self.scene_lock is not None
                and frame_idx > 0
                and frame_idx % recheck_every == 0
            )
            switch_info = None
            source = "cached_last"
            if should_recheck:
                scene_result = self.deepseek.classify_scene(tags["tags_en"])
                self.last_scene_result = scene_result
                switch_info = self.scene_lock.consider_switch(
                    scene_result.get("scene_type"),
                    scene_result.get("confidence", 0.0),
                    frame_idx,
                )
                source = "recheck"
                if switch_info.get("switched"):
                    old_scene = switch_info.get("old_scene")
                    self.scene_type = switch_info.get("scene_type")
                    self.global_atlas = self.deepseek.get_or_build_global_atlas(self.scene_type)
                    self._remote_frame_cache.clear()
                    self._remote_edge_stats.clear()
                    if self.scene_json_path is not None:
                        write_json_atomic(
                            self.scene_json_path,
                            {
                                "scene_type": self.scene_type,
                                "confidence": scene_result.get("confidence"),
                                "switched_at_frame": frame_idx,
                                "previous_scene_type": old_scene,
                            },
                        )
                    if self.atlas_json_path is not None:
                        write_json_atomic(self.atlas_json_path, self.global_atlas)
                    print(f"[DeepSeek scene switched] {old_scene} -> {self.scene_type} at frame {frame_idx}")
            else:
                scene_result = self.last_scene_result or {
                    "scene_type": self.scene_type,
                    "confidence": 1.0,
                    "reason": "cached_last",
                }
            lock_info = {
                "locked": True,
                "scene_type": self.scene_type,
                "streak": 0 if switch_info is None else switch_info.get("switch_streak", 0),
                "reason": "cached_last" if switch_info is None else switch_info.get("reason"),
            }
            if self.scene_history_path is not None:

                append_jsonl(
                    self.scene_history_path,
                    {
                        "frame": frame_idx,
                        "scene_type": scene_result.get("scene_type"),
                        "confidence": scene_result.get("confidence"),
                        "locked": True,
                        "active_scene_type": self.scene_type,
                        "streak": lock_info["streak"],
                        "reason": lock_info.get("reason"),
                        "source": source,
                        "switch_info": switch_info,
                    },
                )

        incremental = None
        if self.global_atlas is not None:

            atlas_oc = {self._canon_tag(o) for o in atlas_object_carrier_set(self.global_atlas)}
            atlas_remote_endpoints = remote_endpoints_from_pairs(
                (self.global_atlas or {}).get("remote_relation_candidates", [])
            )
            mapped_norm_tags = {self._map_tag_to_atlas(t, atlas_oc) for t in norm_tags}
            unknown_tags = sorted(mapped_norm_tags - atlas_oc)
            if unknown_tags:
                incremental = self.deepseek.infer_incremental_uc(
                    uc_tags_en=unknown_tags,
                    atlas_remote_objects_en=sorted(atlas_remote_endpoints),
                    scene_type=self.scene_type or "unknown",
                )
            else:
                incremental = {"objects": [], "remote_relation_candidates": []}
        else:

            if self.args.pre_atlas_uc_mode == "all":
                if not self.pre_atlas_notice_printed:
                    print("[DeepSeek] pre-atlas UC enabled: generating UC before scene lock (no atlas dedup)")
                    self.pre_atlas_notice_printed = True

                cur_scene_type = "unknown"
                pre_atlas_tags = sorted(norm_tags)
                incremental = self.deepseek.infer_incremental_uc(
                    uc_tags_en=pre_atlas_tags,
                    atlas_remote_objects_en=[],
                    scene_type=cur_scene_type,
                )
            else:
                incremental = None


        frame_result = build_frame_result_final(self.global_atlas, norm_tags_list, incremental)
        graph_frame_result = build_graph_frame_result(
            frame_result,
            suppress_objects=self.graph_policy.suppress_objects,
            hint_only_objects=self.graph_policy.hint_only_objects,
        )

        if frame_idx % self.args.deepseek_print_every == 0:

            print(
                f"[DeepSeek frame] locked={self.global_atlas is not None} "
                f"unknown={len(frame_result['debug']['unknown_tags'])} "
                f"objects={len(frame_result['present'])} "
                f"http(scene/atlas/inc)={self.deepseek.http_stats['scene']}/"
                f"{self.deepseek.http_stats['atlas']}/{self.deepseek.http_stats['incremental']}"
            )

        if self.frames_out_dir is not None:

            self.frames_out_dir.mkdir(parents=True, exist_ok=True)
            inc_out = incremental if incremental is not None else {"objects": []}
            write_json_atomic(self.frames_out_dir / f"frame_{frame_idx:06d}_incremental.json", inc_out)
            write_json_atomic(self.frames_out_dir / f"frame_{frame_idx:06d}_frame_result.json", frame_result)
            write_json_atomic(self.frames_out_dir / f"frame_{frame_idx:06d}_frame_result_graph.json", graph_frame_result)

        return DeepSeekOut(frame_result=frame_result, graph_frame_result=graph_frame_result, incremental=incremental)

    def run_fixed_semantic(self, frame_idx: int) -> DeepSeekOut:
        """Build frame_result from the fixed atlas only, with no external semantic calls."""
        if self.global_atlas is None or self.fixed_atlas_source is None:
            raise RuntimeError("Fixed semantic atlas is not initialized")

        frame_result = build_frame_result_final(self.global_atlas, [], {"objects": [], "remote_relation_candidates": []})
        graph_frame_result = build_graph_frame_result(
            frame_result,
            suppress_objects=self.graph_policy.suppress_objects,
            hint_only_objects=self.graph_policy.hint_only_objects,
        )
        frame_result.setdefault("debug", {})["source"] = "fixed_semantic_atlas"
        frame_result["debug"]["atlas_source"] = self.fixed_atlas_source
        graph_frame_result.setdefault("debug", {})["source"] = "fixed_semantic_atlas"
        graph_frame_result["debug"]["atlas_source"] = self.fixed_atlas_source

        if frame_idx % self.args.deepseek_print_every == 0:
            print(
                f"[Fixed semantic frame] objects={len(frame_result.get('present', []) or [])} "
                f"remote={len(frame_result.get('remote_relation_candidates', []) or [])}"
            )

        if self.frames_out_dir is not None:
            self.frames_out_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.frames_out_dir / f"frame_{frame_idx:06d}_incremental.json", {"objects": []})
            write_json_atomic(self.frames_out_dir / f"frame_{frame_idx:06d}_frame_result.json", frame_result)
            write_json_atomic(self.frames_out_dir / f"frame_{frame_idx:06d}_frame_result_graph.json", graph_frame_result)

        return DeepSeekOut(frame_result=frame_result, graph_frame_result=graph_frame_result, incremental={"objects": []})

    def run_sam3(self, frame_idx: int, frame_uimg, frame_result: dict, graph_frame_result: Optional[dict] = None) -> Optional[Sam3Out]:

        if self.sam3_runtime is None:
            return None
        det = self.sam3_runtime.run(frame_uimg, frame_result, frame_idx=frame_idx, policy=self.graph_policy)

        if det is not None and not bool(getattr(self.args, "sam3_keep_border_no_link_detections", False)):
            import torch as _torch
            if hasattr(frame_uimg, "shape"):
                if _torch.is_tensor(frame_uimg):
                    _h, _w = int(frame_uimg.shape[0]), int(frame_uimg.shape[1])
                else:
                    _h, _w = int(frame_uimg.shape[0]), int(frame_uimg.shape[1])
                _filter_border_no_link_detections(det, _h, _w, margin_px=2)
        remote_frame_result = graph_frame_result if isinstance(graph_frame_result, dict) else frame_result
        remote_rel_2d = self._build_remote_rel_2d(frame_idx, frame_uimg, det, remote_frame_result) if det else None
        debug_image_path = None
        if self.frames_out_dir is not None:

            self.frames_out_dir.mkdir(parents=True, exist_ok=True)
            rgb_path = self.frames_out_dir / f"frame_{frame_idx:06d}_rgb.png"
            self._save_rgb_frame(rgb_path, frame_uimg)
            debug_image_path = str(rgb_path)
            det_path = self.frames_out_dir / f"frame_{frame_idx:06d}_sam3_det.json"
            write_json_atomic(det_path, _sam3_det_to_json(det, extra={"remote_rel_2d": remote_rel_2d}))
            if self.sam3_save_vis:

                vis_path = self.frames_out_dir / f"frame_{frame_idx:06d}_sam3_det.png"
                _save_sam3_vis(vis_path, frame_uimg, det, remote_rel_2d=remote_rel_2d)
                debug_image_path = str(vis_path)
        return Sam3Out(det=det, remote_rel_2d=remote_rel_2d, debug_image_path=debug_image_path)

    @staticmethod
    def _save_rgb_frame(path: pathlib.Path, frame_uimg) -> None:
        if hasattr(frame_uimg, "detach"):
            img_np = (frame_uimg.detach().clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
        else:
            arr = np.asarray(frame_uimg)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            img_np = arr
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    def _build_remote_rel_2d(self, frame_idx: int, frame_uimg, det, frame_result: dict) -> dict:
        obj_set = {self._canon_tag(e.get("object")) for e in frame_result.get("present", []) if e.get("object")}

        label_best: Dict[str, dict] = {}
        labels = det.labels or []
        scores = det.scores
        boxes = det.boxes
        for idx, label in enumerate(labels):
            if not isinstance(label, str):
                continue
            name = self._canon_tag(label)
            if not name:
                continue
            score = float(scores[idx].item()) if scores is not None else 0.0
            if name not in label_best or score > label_best[name]["score"]:
                label_best[name] = {
                    "idx": int(idx),
                    "score": score,
                    "box_xyxy": boxes[idx].tolist() if boxes is not None else None,
                }

        observed_candidates = []
        allowed_edge_keys: set[tuple[str, str]] = set()
        for rel in frame_result.get("remote_relation_candidates", []) or []:
            a = self._canon_tag(rel.get("from_object"))
            b = self._canon_tag(rel.get("to_object"))
            if not a or not b:
                continue
            allowed_edge_keys.add((a, b))
            if a not in obj_set or b not in obj_set:
                continue
            if a not in label_best or b not in label_best:
                continue
            a_score = label_best[a]["score"]
            b_score = label_best[b]["score"]
            evidence = min(a_score, b_score)
            observed_candidates.append(
                {
                    "from_object": a,
                    "to_object": b,
                    "relation": (rel.get("relation") or "").strip(),
                    "evidence": evidence,
                    "from_score": a_score,
                    "to_score": b_score,
                    "from_det_idx": label_best[a]["idx"],
                    "to_det_idx": label_best[b]["idx"],
                }
            )

        # purge edges no longer allowed by current frame_result
        for key in list(self._remote_edge_stats.keys()):
            if key not in allowed_edge_keys:
                self._remote_edge_stats.pop(key, None)

        # update sliding window stats
        win_start = frame_idx - self.remote_win_size + 1
        for rel in observed_candidates:
            key = (rel["from_object"], rel["to_object"])
            if key not in self._remote_edge_stats:
                self._remote_edge_stats[key] = {
                    "frames": deque(),
                    "scores": deque(),
                    "score_sum": 0.0,
                }
            stats = self._remote_edge_stats[key]
            if len(stats["frames"]) == 0 or stats["frames"][-1] != frame_idx:
                stats["frames"].append(frame_idx)
                stats["scores"].append(rel["evidence"])
                stats["score_sum"] += rel["evidence"]

        # prune old frames
        for stats in self._remote_edge_stats.values():
            while stats["frames"] and stats["frames"][0] < win_start:
                stats["frames"].popleft()
                stats["score_sum"] -= stats["scores"].popleft()

        confirmed_all = []
        confirmed_visible = []
        for key, stats in self._remote_edge_stats.items():
            if len(stats["frames"]) >= self.remote_confirm_k:
                confirmed_all.append(
                    {
                        "from_object": key[0],
                        "to_object": key[1],
                        "evidence_sum": stats["score_sum"],
                        "seen_frames": list(stats["frames"]),
                    }
                )
                if key[0] in label_best and key[1] in label_best:
                    confirmed_visible.append(
                        {
                            "from_object": key[0],
                            "to_object": key[1],
                            "evidence_sum": stats["score_sum"],
                            "seen_frames": list(stats["frames"]),
                        }
                    )

        # update frame cache for potential future disambiguation
        self._remote_frame_cache.append(
            {"frame_idx": frame_idx, "label_best": label_best}
        )

        return {
            "observed_candidates": observed_candidates,
            "confirmed_visible": confirmed_visible,
            "confirmed_all": confirmed_all,
        }
