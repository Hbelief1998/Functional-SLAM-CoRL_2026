"""Node-aware place recognition helper module.

This is a lightweight, pure-Python helper that uses stable functional-graph
nodes (O/C roles) to compute a small "node signature" per frame/keyframe and
to rerank/filter retrieval candidates produced by the MASt3R retrieval
database before they enter the geometric `FactorGraph` verification step.

Design constraints:
- No CUDA / torch tensors stored in signatures (must be JSON-serializable so
  they can be exchanged through a multiprocessing.Manager between the main
  process and the backend process).
- All public functions degrade gracefully when there is not enough node
  information (return original candidates / empty signature / score=0).
- The module never accepts/rejects a candidate by itself; it only produces a
  reordered/filtered candidate list. The geometric check inside
  ``FactorGraph.add_factors`` remains the final gate.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .types import normalize_label


_DEFAULT_USE_ROLES = ("O", "C")
_DEFAULT_EXCLUDE_ROLES = ("U",)
_DEFAULT_ALLOWED_GEOM_SOURCES = (
    "stable_anchor_fused",
    "stable_anchor",
    "world_bbox",
    "world_centroid",
)


def _coerce_roles(value: Optional[Iterable[str]], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return tuple(default)
    out = []
    for r in value:
        if not r:
            continue
        s = str(r).strip().upper()
        if s:
            out.append(s)
    return tuple(out) if out else tuple(default)


def _coerce_str_set(value: Optional[Iterable[str]], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return tuple(default)
    out = []
    for r in value:
        if not r:
            continue
        s = str(r).strip()
        if s:
            out.append(s)
    return tuple(out) if out else tuple(default)


def _node_passes_filters(track, cfg: Dict[str, Any]) -> bool:
    """Decide whether a stable-track-backed observation should be kept."""
    use_roles = _coerce_roles(cfg.get("use_roles"), _DEFAULT_USE_ROLES)
    exclude_roles = _coerce_roles(cfg.get("exclude_roles"), _DEFAULT_EXCLUDE_ROLES)
    require_stable_geom = bool(cfg.get("require_stable_geom", True))
    allowed_geom_sources = _coerce_str_set(
        cfg.get("allowed_geom_sources"), _DEFAULT_ALLOWED_GEOM_SOURCES
    )
    min_obs = int(cfg.get("min_track_obs_count", 3) or 0)
    min_vis = int(cfg.get("min_track_visible_count", 2) or 0)

    role = str(getattr(track, "role", "") or "").strip().upper()
    if role in exclude_roles:
        return False
    if role not in use_roles:
        return False

    if int(getattr(track, "obs_count", 0) or 0) < min_obs:
        return False
    if int(getattr(track, "visible_count", 0) or 0) < min_vis:
        return False

    if require_stable_geom:
        stable_ready = bool(getattr(track, "stable_geom_ready", False))
        geom_source = ""
        getter = getattr(track, "assoc_geom_source", None)
        if callable(getter):
            try:
                geom_source = str(getter() or "")
            except Exception:
                geom_source = ""
        if not stable_ready and geom_source not in allowed_geom_sources:
            return False
    return True


def build_signature_from_online_state(
    online_state,
    frame_idx: int,
    *,
    kf_idx: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
    allow_unmatched_label_only: bool = True,
) -> Dict[str, Any]:
    """Build a JSON-serializable node signature for a frame.

    Returns an empty signature (all collections empty) if node info is
    missing or the configuration excludes everything. Never raises.
    """
    cfg = dict(cfg or {})
    sig: Dict[str, Any] = {
        "frame_idx": int(frame_idx) if frame_idx is not None else None,
        "kf_idx": int(kf_idx) if kf_idx is not None else None,
        "nodes": [],
        "label_counts": {},
        "role_counts": {},
        "stable_node_ids": [],
    }
    if online_state is None:
        return sig

    use_roles = _coerce_roles(cfg.get("use_roles"), _DEFAULT_USE_ROLES)
    exclude_roles = _coerce_roles(cfg.get("exclude_roles"), _DEFAULT_EXCLUDE_ROLES)
    min_view_score = float(cfg.get("min_view_score", 0.0) or 0.0)

    frame_obs_map = getattr(online_state, "frame_observations", None)
    if frame_obs_map is None:
        return sig
    observations = frame_obs_map.get(int(frame_idx)) if hasattr(frame_obs_map, "get") else None
    if not observations:
        return sig

    node_tracks = getattr(online_state, "node_tracks", {}) or {}

    label_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    stable_ids: List[str] = []
    nodes_out: List[Dict[str, Any]] = []
    seen_node_ids: set[str] = set()

    for obs in observations:
        try:
            obs_role = str(getattr(obs, "role", "") or "").strip().upper()
            if obs_role in exclude_roles:
                continue
            view_score = float(getattr(obs, "view_score", 0.0) or 0.0)
            if view_score < min_view_score:
                continue

            matched_id = getattr(obs, "matched_node_id", None)
            track = node_tracks.get(matched_id) if matched_id else None

            if track is not None:
                if not _node_passes_filters(track, cfg):
                    continue
                role = str(getattr(track, "role", obs_role) or obs_role).strip().upper()
                if role not in use_roles:
                    continue
                label = normalize_label(getattr(track, "label", "") or getattr(obs, "label", ""))
                geom_source = ""
                getter = getattr(track, "assoc_geom_source", None)
                if callable(getter):
                    try:
                        geom_source = str(getter() or "")
                    except Exception:
                        geom_source = ""
                node_id = str(matched_id)
                if node_id in seen_node_ids:
                    continue
                seen_node_ids.add(node_id)
                nodes_out.append({
                    "node_id": node_id,
                    "label": label,
                    "role": role,
                    "view_score": view_score,
                    "stable": bool(getattr(track, "stable_geom_ready", False)),
                    "geom_source": geom_source,
                    "obs_count": int(getattr(track, "obs_count", 0) or 0),
                    "visible_count": int(getattr(track, "visible_count", 0) or 0),
                })
                stable_ids.append(node_id)
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1
                if role:
                    role_counts[role] = role_counts.get(role, 0) + 1
            else:
                # Unmatched observation: only used for query-side label-only matching.
                if not allow_unmatched_label_only:
                    continue
                if obs_role not in use_roles:
                    continue
                label = normalize_label(getattr(obs, "label", ""))
                if not label:
                    continue
                nodes_out.append({
                    "node_id": None,
                    "label": label,
                    "role": obs_role,
                    "view_score": view_score,
                    "stable": False,
                    "geom_source": "unmatched",
                    "obs_count": 0,
                    "visible_count": 0,
                })
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1
                if obs_role:
                    role_counts[obs_role] = role_counts.get(obs_role, 0) + 1
        except Exception:
            # Skip malformed observations without breaking signature build.
            continue

    sig["nodes"] = nodes_out
    sig["label_counts"] = label_counts
    sig["role_counts"] = role_counts
    sig["stable_node_ids"] = sorted(stable_ids)
    return sig


def _multiset_overlap(a: Dict[str, int], b: Dict[str, int]) -> Tuple[int, int]:
    """Return (intersection_size, union_size) of two label/role multisets."""
    inter = 0
    union = 0
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        ca = int(a.get(k, 0))
        cb = int(b.get(k, 0))
        inter += min(ca, cb)
        union += max(ca, cb)
    return inter, union


def score_signature_pair(
    query_sig: Dict[str, Any],
    cand_sig: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute a similarity score between two node signatures.

    Always returns a dict with the same shape; never raises.
    """
    cfg = dict(cfg or {})
    w_nid = float(cfg.get("weight_node_id_overlap", 0.55) or 0.0)
    w_lab = float(cfg.get("weight_label_overlap", 0.30) or 0.0)
    w_role = float(cfg.get("weight_role_overlap", 0.10) or 0.0)
    w_cnt = float(cfg.get("weight_count_coverage", 0.05) or 0.0)
    bonus_o = float(cfg.get("object_role_bonus", 1.2) or 1.0)
    bonus_c = float(cfg.get("carrier_role_bonus", 0.8) or 1.0)

    q_nodes = list((query_sig or {}).get("nodes") or [])
    c_nodes = list((cand_sig or {}).get("nodes") or [])
    q_count = len(q_nodes)
    c_count = len(c_nodes)
    insufficient = (q_count == 0 or c_count == 0)

    out = {
        "score": 0.0,
        "shared_node_ids": [],
        "shared_labels": [],
        "node_id_score": 0.0,
        "label_score": 0.0,
        "role_score": 0.0,
        "count_score": 0.0,
        "query_node_count": q_count,
        "cand_node_count": c_count,
        "insufficient_node_info": bool(insufficient),
    }
    if insufficient:
        return out

    # node_id Jaccard with O/C role bonuses
    q_ids = {n.get("node_id"): n for n in q_nodes if n.get("node_id")}
    c_ids = {n.get("node_id"): n for n in c_nodes if n.get("node_id")}
    shared_ids = sorted(set(q_ids.keys()) & set(c_ids.keys()))
    union_ids = set(q_ids.keys()) | set(c_ids.keys())
    if union_ids:
        weighted_inter = 0.0
        for nid in shared_ids:
            role = str((q_ids[nid].get("role") or c_ids[nid].get("role") or "")).upper()
            bonus = bonus_o if role == "O" else (bonus_c if role == "C" else 1.0)
            weighted_inter += bonus
        weighted_union = 0.0
        for nid in union_ids:
            n = q_ids.get(nid) or c_ids.get(nid)
            role = str((n.get("role") if n else "") or "").upper()
            bonus = bonus_o if role == "O" else (bonus_c if role == "C" else 1.0)
            weighted_union += bonus
        node_id_score = weighted_inter / weighted_union if weighted_union > 0 else 0.0
    else:
        node_id_score = 0.0

    # label multiset overlap
    q_labels = dict((query_sig or {}).get("label_counts") or {})
    c_labels = dict((cand_sig or {}).get("label_counts") or {})
    inter_lab, union_lab = _multiset_overlap(q_labels, c_labels)
    label_score = (inter_lab / union_lab) if union_lab > 0 else 0.0
    shared_labels = sorted([k for k in (set(q_labels.keys()) & set(c_labels.keys()))])

    # role count similarity (1 - normalized L1 distance)
    q_roles = dict((query_sig or {}).get("role_counts") or {})
    c_roles = dict((cand_sig or {}).get("role_counts") or {})
    role_keys = set(q_roles.keys()) | set(c_roles.keys())
    if role_keys:
        diff = 0
        total = 0
        for k in role_keys:
            qv = int(q_roles.get(k, 0))
            cv = int(c_roles.get(k, 0))
            diff += abs(qv - cv)
            total += qv + cv
        role_score = 1.0 - (diff / total) if total > 0 else 0.0
    else:
        role_score = 0.0

    # count coverage: shared visible nodes (label+role) normalized by min size
    denom = max(1, min(q_count, c_count))
    count_score = min(1.0, (inter_lab / denom))

    # final score: weighted sum, clamped to [0, 1]
    total_w = w_nid + w_lab + w_role + w_cnt
    if total_w <= 0:
        score = 0.0
    else:
        score = (
            w_nid * node_id_score
            + w_lab * label_score
            + w_role * role_score
            + w_cnt * count_score
        ) / total_w
    score = max(0.0, min(1.0, float(score)))

    out.update({
        "score": score,
        "shared_node_ids": shared_ids,
        "shared_labels": shared_labels,
        "node_id_score": float(node_id_score),
        "label_score": float(label_score),
        "role_score": float(role_score),
        "count_score": float(count_score),
    })
    return out


def rerank_candidates_by_nodes(
    candidate_indices: List[int],
    *,
    query_signature: Optional[Dict[str, Any]],
    candidate_signatures: Optional[Dict[int, Dict[str, Any]]],
    cfg: Optional[Dict[str, Any]] = None,
    mode: str = "loop",
) -> Tuple[List[int], Dict[str, Any]]:
    """Rerank/filter retrieval candidates using node signatures.

    Returns (reranked_candidate_indices, debug_payload). On any insufficient
    information, returns the original list unchanged with debug noting fallback.
    """
    cfg = dict(cfg or {})
    candidate_signatures = dict(candidate_signatures or {})
    original = [int(c) for c in (candidate_indices or [])]

    debug: Dict[str, Any] = {
        "mode": mode,
        "original_candidates": list(original),
        "final_candidates": list(original),
        "query_node_count": int(len((query_signature or {}).get("nodes") or [])),
        "candidate_scores": [],
        "fallback_used": False,
        "best_candidate": None,
        "best_score": 0.0,
    }

    if not original:
        return [], debug

    if mode == "reloc":
        min_query_nodes = int(cfg.get("reloc_min_query_nodes", 2) or 0)
        min_score = float(cfg.get("reloc_min_score", 0.20) or 0.0)
        top_k = int(cfg.get("reloc_top_k", 3) or 0)
        do_filter = bool(cfg.get("reloc_filter", True))
        fallback = bool(cfg.get("reloc_fallback_to_original", True))
    else:
        min_query_nodes = int(cfg.get("loop_min_query_nodes", 2) or 0)
        min_score = float(cfg.get("loop_min_score", 0.12) or 0.0)
        top_k = int(cfg.get("loop_top_k", 6) or 0)
        do_filter = bool(cfg.get("loop_filter", True))
        fallback = bool(cfg.get("loop_fallback_to_original", True))

    q_nodes = list((query_signature or {}).get("nodes") or [])
    if len(q_nodes) < max(1, min_query_nodes):
        debug["fallback_used"] = True
        debug["fallback_reason"] = "insufficient_query_nodes"
        return list(original), debug

    scored: List[Dict[str, Any]] = []
    for cidx in original:
        c_sig = candidate_signatures.get(int(cidx)) or candidate_signatures.get(cidx)
        if not c_sig or not (c_sig.get("nodes") or []):
            entry = {
                "candidate_idx": int(cidx),
                "score": 0.0,
                "shared_node_ids": [],
                "shared_labels": [],
                "missing_candidate_signature": True,
            }
            scored.append(entry)
            continue
        s = score_signature_pair(query_signature, c_sig, cfg)
        entry = {
            "candidate_idx": int(cidx),
            "score": float(s.get("score", 0.0)),
            "shared_node_ids": list(s.get("shared_node_ids") or []),
            "shared_labels": list(s.get("shared_labels") or []),
            "node_id_score": float(s.get("node_id_score", 0.0)),
            "label_score": float(s.get("label_score", 0.0)),
            "missing_candidate_signature": False,
        }
        scored.append(entry)

    # Sort by score descending; preserve original order for ties via index.
    order_index = {c: i for i, c in enumerate(original)}
    scored.sort(key=lambda e: (-float(e["score"]), order_index.get(int(e["candidate_idx"]), 0)))

    kept: List[Dict[str, Any]] = []
    for entry in scored:
        if not do_filter:
            entry["kept"] = True
            entry["reason"] = "filter_disabled"
            kept.append(entry)
            continue
        if entry.get("missing_candidate_signature"):
            if mode == "loop":
                entry["kept"] = True
                entry["reason"] = "missing_signature_keep_conservative"
                kept.append(entry)
            else:
                entry["kept"] = False
                entry["reason"] = "missing_signature_drop"
            continue
        if float(entry["score"]) >= min_score:
            entry["kept"] = True
            entry["reason"] = "score_pass"
            kept.append(entry)
        else:
            entry["kept"] = False
            entry["reason"] = "score_below_threshold"

    if top_k > 0 and len(kept) > top_k:
        dropped = kept[top_k:]
        kept = kept[:top_k]
        for d in dropped:
            d["kept"] = False
            d["reason"] = "exceeds_top_k"
            scored_idx = next((i for i, e in enumerate(scored) if e is d), None)
            del scored_idx  # entry already in scored list

    final_candidates = [int(e["candidate_idx"]) for e in kept]

    if not final_candidates and fallback:
        debug["fallback_used"] = True
        debug["fallback_reason"] = "all_filtered_fallback"
        debug["candidate_scores"] = scored
        debug["final_candidates"] = list(original)
        debug["best_candidate"] = scored[0]["candidate_idx"] if scored else None
        debug["best_score"] = float(scored[0]["score"]) if scored else 0.0
        return list(original), debug

    debug["candidate_scores"] = scored
    debug["final_candidates"] = list(final_candidates)
    debug["best_candidate"] = scored[0]["candidate_idx"] if scored else None
    debug["best_score"] = float(scored[0]["score"]) if scored else 0.0
    return final_candidates, debug


# ---------------------------------------------------------------------------
# v2 helpers: functional-graph aware place recognition (SUPPLEMENT-only).
# ---------------------------------------------------------------------------


def _online_visible_node_ids(online_state, frame_idx: int) -> set:
    """Return the set of stable matched node IDs visible in the given frame."""
    out: set = set()
    if online_state is None:
        return out
    fmap = getattr(online_state, "frame_observations", None)
    if fmap is None:
        return out
    obs_list = fmap.get(int(frame_idx)) if hasattr(fmap, "get") else None
    if not obs_list:
        return out
    for obs in obs_list:
        nid = getattr(obs, "matched_node_id", None)
        if nid:
            out.add(str(nid))
    return out


def _safe_chain_label(chain_ids, node_tracks) -> str:
    parts = []
    for nid in chain_ids:
        tr = (node_tracks or {}).get(nid)
        lbl = normalize_label(getattr(tr, "label", "") or "") if tr is not None else ""
        parts.append(lbl or "?")
    return "|".join(parts)


def build_functional_place_signature(
    online_state,
    frame_idx: int,
    *,
    kf_idx=None,
    cfg=None,
):
    """Build a v2 functional-place signature using the persistent functional graph.

    The signature contains:
      - node_ids: stable matched node IDs visible in this frame (O/C and U if linked through chain)
      - chain_ids: list of chains (tuple of node_ids) where ALL endpoints are visible
      - chain_labels: same chains, but as label tuples (for IDF label matching)
      - direct_uo: list of (U,O) pairs where both nodes visible
      - remote_relations: list of remote edge keys whose endpoints are visible (optional)
      - centroids: dict node_id -> [x,y,z] (world) for spatial sanity
      - frame_idx, kf_idx
    All fields are JSON-serializable. Never raises.
    """
    cfg = dict(cfg or {})
    sig = {
        "frame_idx": int(frame_idx) if frame_idx is not None else None,
        "kf_idx": int(kf_idx) if kf_idx is not None else None,
        "node_ids": [],
        "node_labels": {},
        "node_roles": {},
        "chain_ids": [],
        "chain_labels": [],
        "direct_uo": [],
        "direct_labels": [],
        "remote_relations": [],
        "centroids": {},
        "label_counts": {},
        "tokens": [],
        "strong_tokens": [],
        "weak_tokens": [],
        "full_chain_ids": [],
        "semantic_graphlets": [],
        "functional_edges": [],
    }
    if online_state is None:
        return sig

    use_chains = bool(cfg.get("use_functional_chains", True))
    use_direct = bool(cfg.get("use_direct_uo", True))
    use_remote = bool(cfg.get("use_remote_relations", False))

    visible = _online_visible_node_ids(online_state, frame_idx)
    if not visible:
        return sig

    node_tracks = getattr(online_state, "node_tracks", {}) or {}
    sig_nodes: list = []
    label_counts: dict = {}
    centroids: dict = {}
    node_labels: dict = {}
    node_roles: dict = {}
    for nid in sorted(visible):
        tr = node_tracks.get(nid)
        if tr is None:
            continue
        role = str(getattr(tr, "role", "") or "").strip().upper()
        label = normalize_label(getattr(tr, "label", "") or "")
        sig_nodes.append(nid)
        node_labels[nid] = label
        node_roles[nid] = role
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
        c = getattr(tr, "centroid_world_est", None)
        if c is not None:
            try:
                xyz = [float(x) for x in list(c)[:3]]
                if len(xyz) == 3:
                    centroids[nid] = xyz
            except Exception:
                pass
    sig["node_ids"] = sig_nodes
    sig["node_labels"] = node_labels
    sig["node_roles"] = node_roles
    sig["centroids"] = centroids
    sig["label_counts"] = label_counts

    graph = getattr(online_state, "graph", None)
    if graph is None:
        return sig

    if use_chains:
        try:
            chains_uco = list(getattr(graph.hierarchy, "chains_uco", []) or [])
        except Exception:
            chains_uco = []
        for chain in chains_uco:
            try:
                chain_t = tuple(str(x) for x in chain)
            except Exception:
                continue
            if len(chain_t) < 2:
                continue
            # Require at least 2 endpoints visible (typically U and O)
            vis_count = sum(1 for nid in chain_t if nid in visible)
            if vis_count < 2:
                continue
            sig["chain_ids"].append(list(chain_t))
            sig["chain_labels"].append([
                node_labels.get(nid) or normalize_label(getattr(node_tracks.get(nid), "label", "") or "")
                for nid in chain_t
            ])
            if all(nid in visible for nid in chain_t):
                sig["full_chain_ids"].append(list(chain_t))

    if use_direct:
        try:
            direct_uo = list(getattr(graph.hierarchy, "direct_uo", []) or [])
        except Exception:
            direct_uo = []
        for pair in direct_uo:
            try:
                u, o = str(pair[0]), str(pair[1])
            except Exception:
                continue
            if u in visible and o in visible:
                sig["direct_uo"].append([u, o])
                sig["direct_labels"].append([
                    node_labels.get(u) or "",
                    node_labels.get(o) or "",
                ])

    if use_remote:
        try:
            remote_edges = dict(getattr(graph, "remote_edges", {}) or {})
        except Exception:
            remote_edges = {}
        for key, edge in remote_edges.items():
            try:
                src = str(getattr(edge, "src_node_id", ""))
                dst = str(getattr(edge, "dst_node_id", ""))
            except Exception:
                continue
            if src and dst and src in visible and dst in visible:
                sig["remote_relations"].append({
                    "key": str(key),
                    "src": src,
                    "dst": dst,
                    "relation_text": str(getattr(edge, "relation_text", "") or ""),
                })

    # ----- v3: functional tokens for IDF-based scoring -----
    strong_tokens: list = []
    weak_tokens: list = []

    # Node tokens: only stable visible O/C nodes (skip U).
    for nid in sig_nodes:
        role = (node_roles.get(nid) or "").upper()
        if role in ("O", "C"):
            strong_tokens.append(f"node:{nid}")

    # Chain tokens: full chain visible -> strong id token + weak label token.
    for chain_t, chain_lbls in zip(sig["chain_ids"], sig["chain_labels"]):
        chain_t_tup = tuple(chain_t)
        if all(nid in visible for nid in chain_t_tup):
            strong_tokens.append("chain_id:" + ">".join(chain_t_tup))
        # label token always emitted as weak when chain entry exists
        lbls = [str(l or "?") for l in chain_lbls]
        weak_tokens.append("chain_label:" + ">".join(lbls))

    # Direct U-O tokens.
    for pair, lbl_pair in zip(sig["direct_uo"], sig["direct_labels"]):
        try:
            u, o = str(pair[0]), str(pair[1])
        except Exception:
            continue
        strong_tokens.append(f"direct_id:{u}>{o}")
        ul, ol = str(lbl_pair[0] or "?"), str(lbl_pair[1] or "?")
        weak_tokens.append(f"direct_label:{ul}>{ol}")

    # Remote relation tokens (weak, only when enabled).
    if use_remote:
        for rel in sig["remote_relations"]:
            src_lbl = node_labels.get(rel.get("src", ""), "") or "?"
            dst_lbl = node_labels.get(rel.get("dst", ""), "") or "?"
            rtxt = (rel.get("relation_text") or "").strip().lower()
            weak_tokens.append(f"remote_label:{src_lbl}>{dst_lbl}:{rtxt}")

    # Spatial bin tokens along functional edges (weak).
    spatial_bin_m = float(cfg.get("spatial_bin_size_m", 0.5) or 0.5)
    if spatial_bin_m > 0.0:
        # along chains: pairwise edges (a,b) within chain
        edges_for_spatial = []
        for chain_t, chain_lbls in zip(sig["chain_ids"], sig["chain_labels"]):
            for i in range(len(chain_t) - 1):
                edges_for_spatial.append(
                    ("chain", chain_t[i], chain_t[i + 1], chain_lbls[i], chain_lbls[i + 1])
                )
        for pair, lbl_pair in zip(sig["direct_uo"], sig["direct_labels"]):
            edges_for_spatial.append(
                ("direct", pair[0], pair[1], lbl_pair[0], lbl_pair[1])
            )
        for etype, a, b, la, lb in edges_for_spatial:
            ca = centroids.get(a)
            cb = centroids.get(b)
            if ca is None or cb is None:
                continue
            try:
                dx = float(ca[0]) - float(cb[0])
                dy = float(ca[1]) - float(cb[1])
                dz = float(ca[2]) - float(cb[2])
                d = (dx * dx + dy * dy + dz * dz) ** 0.5
            except Exception:
                continue
            db = int(d / spatial_bin_m)
            la_s = str(la or "?")
            lb_s = str(lb or "?")
            weak_tokens.append(f"spatial_bin:{etype}:{la_s}>{lb_s}:{db}")

    # de-dup but preserve order
    def _dedup(lst):
        seen = set()
        out = []
        for x in lst:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    strong_tokens = _dedup(strong_tokens)
    weak_tokens = _dedup(weak_tokens)
    sig["strong_tokens"] = strong_tokens
    sig["weak_tokens"] = weak_tokens
    sig["tokens"] = _dedup(strong_tokens + weak_tokens)
    sig["functional_edges"] = build_functional_edge_set(sig)
    graphlets, graphlet_debug = build_semantic_graphlets(sig, cfg)
    sig["semantic_graphlets"] = graphlets
    sig["semantic_graphlet_debug"] = graphlet_debug

    return sig


def _valid_xyz(value) -> Optional[List[float]]:
    try:
        xyz = [float(x) for x in list(value)[:3]]
    except Exception:
        return None
    if len(xyz) != 3:
        return None
    return xyz


def _euclidean(a: List[float], b: List[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _triangle_shape(points: List[List[float]]) -> List[float]:
    if len(points) != 3:
        return [0.0, 0.0, 0.0]
    d01 = _euclidean(points[0], points[1])
    d02 = _euclidean(points[0], points[2])
    d12 = _euclidean(points[1], points[2])
    vals = sorted([d01, d02, d12])
    denom = sum(vals) + 1e-9
    return [float(v / denom) for v in vals]


def build_functional_edge_set(signature: dict) -> list[dict]:
    """Convert full U-C-O chains and direct U-O pairs into directed edges."""
    sig = dict(signature or {})
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(src, dst, edge_type):
        src_s, dst_s, typ_s = str(src), str(dst), str(edge_type)
        if not src_s or not dst_s:
            return
        key = (src_s, dst_s, typ_s)
        if key in seen:
            return
        seen.add(key)
        out.append({"src": src_s, "dst": dst_s, "type": typ_s})

    chains = sig.get("full_chain_ids") or []
    if not chains:
        # Older signatures only have chain_ids. Treat 3-node chains as full
        # chains when full_chain_ids has not yet been populated.
        chains = [ch for ch in (sig.get("chain_ids") or []) if len(ch) >= 3]
    for chain in chains:
        try:
            chain_t = [str(x) for x in chain]
        except Exception:
            continue
        if len(chain_t) >= 3:
            u, c, o = chain_t[0], chain_t[1], chain_t[2]
            _add(u, c, "U-C")
            _add(c, o, "C-O")

    for pair in sig.get("direct_uo") or []:
        try:
            u, o = str(pair[0]), str(pair[1])
        except Exception:
            continue
        _add(u, o, "U-O")
    return out


def build_semantic_graphlets(signature: dict, cfg: dict) -> tuple[list[dict], dict]:
    """Build nearest-two semantic graphlets from stable O/C/U nodes."""
    sig = dict(signature or {})
    cfg = dict(cfg or {})
    include_u = bool(cfg.get("object_topology_include_u", True))
    max_graphlets = int(cfg.get("object_topology_max_graphlets_per_kf", 64) or 0)
    node_ids = [str(n) for n in (sig.get("node_ids") or [])]
    labels = dict(sig.get("node_labels") or {})
    roles = {str(k): str(v or "").upper() for k, v in dict(sig.get("node_roles") or {}).items()}
    centroids_raw = dict(sig.get("centroids") or {})
    centroids = {str(k): _valid_xyz(v) for k, v in centroids_raw.items()}
    centroids = {k: v for k, v in centroids.items() if v is not None}

    connected_u: set[str] = set()
    for chain in (sig.get("full_chain_ids") or []):
        if chain:
            connected_u.add(str(chain[0]))
    for chain in (sig.get("chain_ids") or []):
        if len(chain) >= 3 and all(str(x) in centroids for x in chain[:3]):
            connected_u.add(str(chain[0]))
    for pair in (sig.get("direct_uo") or []):
        if pair:
            connected_u.add(str(pair[0]))

    eligible: list[str] = []
    skipped = {"no_centroid": 0, "role": 0, "unconnected_u": 0}
    for nid in node_ids:
        role = roles.get(nid, "")
        if nid not in centroids:
            skipped["no_centroid"] += 1
            continue
        if role in {"O", "C"}:
            eligible.append(nid)
        elif role == "U" and include_u:
            if nid in connected_u:
                eligible.append(nid)
            else:
                skipped["unconnected_u"] += 1
        else:
            skipped["role"] += 1

    if len(eligible) < 3:
        return [], {
            "eligible_node_count": len(eligible),
            "graphlet_count": 0,
            "skipped": skipped,
            "reason": "fewer_than_three_nodes",
        }

    graphlets: list[dict] = []
    seen_triples: set[tuple[str, str, str]] = set()
    for nid in eligible:
        neighbors = [other for other in eligible if other != nid]
        neighbors.sort(key=lambda other: (_euclidean(centroids[nid], centroids[other]), other))
        if len(neighbors) < 2:
            continue
        triple = [nid, neighbors[0], neighbors[1]]
        key = tuple(sorted(triple))
        if key in seen_triples:
            continue
        seen_triples.add(key)
        points = [centroids[x] for x in triple]
        graphlets.append({
            "node_ids": list(triple),
            "roles": [roles.get(x, "") for x in triple],
            "labels": [normalize_label(labels.get(x, "") or "") for x in triple],
            "centroids": {x: list(centroids[x]) for x in triple},
            "shape": _triangle_shape(points),
            "source": "nearest2",
        })
        if max_graphlets > 0 and len(graphlets) >= max_graphlets:
            break

    return graphlets, {
        "eligible_node_count": len(eligible),
        "graphlet_count": len(graphlets),
        "skipped": skipped,
        "max_graphlets": max_graphlets,
    }


def _graphlet_nodes(graphlet: dict) -> list[dict]:
    ids = [str(x) for x in (graphlet or {}).get("node_ids") or []]
    roles = list((graphlet or {}).get("roles") or [])
    labels = list((graphlet or {}).get("labels") or [])
    out = []
    for i, nid in enumerate(ids):
        out.append({
            "node_id": nid,
            "role": str(roles[i] if i < len(roles) else "").upper(),
            "label": normalize_label(labels[i] if i < len(labels) else ""),
        })
    return out


def score_semantic_graphlet_pair(gq: dict, gc: dict, cfg: dict = None) -> dict:
    """Score two 3-node graphlets using semantic compatibility and shape."""
    q_nodes = _graphlet_nodes(gq)
    c_nodes = _graphlet_nodes(gc)
    if len(q_nodes) != 3 or len(c_nodes) != 3:
        return {"score": 0.0, "semantic_score": 0.0, "shape_score": 0.0, "mapping": {}}

    best_sem = 0.0
    best_mapping: dict[str, str] = {}
    for perm in permutations(range(3)):
        compat = 0.0
        mapping: dict[str, str] = {}
        for qi, ci in enumerate(perm):
            qn, cn = q_nodes[qi], c_nodes[ci]
            if qn["node_id"] and qn["node_id"] == cn["node_id"]:
                s = 1.0
            elif qn["role"] == cn["role"] and qn["label"] and qn["label"] == cn["label"]:
                s = 1.0
            else:
                s = 0.0
            compat += s
            mapping[qn["node_id"]] = cn["node_id"]
        sem = compat / 3.0
        if sem > best_sem:
            best_sem = sem
            best_mapping = mapping

    q_shape = [float(x) for x in ((gq or {}).get("shape") or [0.0, 0.0, 0.0])[:3]]
    c_shape = [float(x) for x in ((gc or {}).get("shape") or [0.0, 0.0, 0.0])[:3]]
    if len(q_shape) != 3 or len(c_shape) != 3:
        shape_score = 0.0
    else:
        l1 = sum(abs(q_shape[i] - c_shape[i]) for i in range(3))
        shape_score = float(1.0 / (1.0 + l1))
    score = float(best_sem * shape_score)
    return {
        "score": score,
        "semantic_score": float(best_sem),
        "shape_score": shape_score,
        "mapping": best_mapping,
    }


def _ensure_topology_fields(signature: dict, cfg: dict = None) -> dict:
    sig = dict(signature or {})
    if "functional_edges" not in sig:
        sig["functional_edges"] = build_functional_edge_set(sig)
    if "semantic_graphlets" not in sig:
        graphlets, dbg = build_semantic_graphlets(sig, cfg or {})
        sig["semantic_graphlets"] = graphlets
        sig["semantic_graphlet_debug"] = dbg
    return sig


def score_graphlet_topology_pair(query_sig: dict, cand_sig: dict, cfg: dict = None) -> dict:
    """Compute object-topology graphlet score S_G."""
    cfg = dict(cfg or {})
    q = _ensure_topology_fields(query_sig, cfg)
    c = _ensure_topology_fields(cand_sig, cfg)
    q_graphlets = list(q.get("semantic_graphlets") or [])
    c_graphlets = list(c.get("semantic_graphlets") or [])
    compare_limit = int(cfg.get("object_topology_compare_graphlets_per_kf", 24) or 0)
    if compare_limit > 0:
        q_graphlets = q_graphlets[:compare_limit]
        c_graphlets = c_graphlets[:compare_limit]
    out = {
        "score": 0.0,
        "matched_graphlets": [],
        "query_graphlet_count": len(q_graphlets),
        "candidate_graphlet_count": len(c_graphlets),
    }
    if not q_graphlets or not c_graphlets:
        return out

    matches = []
    for gq in q_graphlets:
        best = None
        for gc in c_graphlets:
            s = score_semantic_graphlet_pair(gq, gc, cfg)
            if best is None or float(s.get("score", 0.0)) > float(best.get("graphlet_score", 0.0)):
                best = {
                    "query_node_ids": [str(x) for x in (gq.get("node_ids") or [])],
                    "candidate_node_ids": [str(x) for x in (gc.get("node_ids") or [])],
                    "graphlet_score": float(s.get("score", 0.0)),
                    "semantic_score": float(s.get("semantic_score", 0.0)),
                    "shape_score": float(s.get("shape_score", 0.0)),
                    "mapping": dict(s.get("mapping") or {}),
                }
        if best is not None:
            matches.append(best)
    score = sum(float(m.get("graphlet_score", 0.0)) for m in matches) / max(1, len(q_graphlets))
    out["score"] = float(score)
    out["matched_graphlets"] = matches
    return out


def score_functional_consistency_on_matches(
    *,
    query_sig: dict,
    cand_sig: dict,
    matched_graphlets: list[dict],
    cfg: dict = None,
) -> dict:
    """Score preservation of directed functional edges under graphlet matches."""
    q = _ensure_topology_fields(query_sig, cfg or {})
    c = _ensure_topology_fields(cand_sig, cfg or {})
    q_edges = {(str(e.get("src")), str(e.get("dst")), str(e.get("type"))) for e in (q.get("functional_edges") or [])}
    c_edges = {(str(e.get("src")), str(e.get("dst")), str(e.get("type"))) for e in (c.get("functional_edges") or [])}
    details = []
    weighted_sum = 0.0
    weight_sum = 0.0
    func_match_count = 0
    func_query_edge_count = 0
    for m in matched_graphlets or []:
        q_ids = {str(x) for x in (m.get("query_node_ids") or [])}
        if not q_ids:
            gq = m.get("query_graphlet") or {}
            q_ids = {str(x) for x in (gq.get("node_ids") or [])}
        mapping = {str(k): str(v) for k, v in dict(m.get("mapping") or {}).items()}
        local_q_edges = [e for e in q_edges if e[0] in q_ids and e[1] in q_ids]
        if not local_q_edges:
            continue
        preserved = 0
        for src, dst, typ in local_q_edges:
            ms, md = mapping.get(src), mapping.get(dst)
            if ms is not None and md is not None and (ms, md, typ) in c_edges:
                preserved += 1
        psi = preserved / max(1, len(local_q_edges))
        w = max(0.0, float(m.get("graphlet_score", 0.0)))
        weighted_sum += w * psi
        weight_sum += w
        func_match_count += 1
        func_query_edge_count += len(local_q_edges)
        details.append({
            "query_node_ids": sorted(q_ids),
            "query_edge_count": len(local_q_edges),
            "preserved_count": int(preserved),
            "score": float(psi),
            "weight": float(w),
        })
    score = (weighted_sum / weight_sum) if weight_sum > 0 else 0.0
    return {
        "score": float(score),
        "functional_match_count": int(func_match_count),
        "functional_query_edge_count": int(func_query_edge_count),
        "details": details,
    }


def retrieve_object_topology_candidates(
    *,
    query_signature: dict,
    all_kf_signatures: dict,
    existing_candidates: list[int],
    cfg: dict,
    query_kf_idx: int | None = None,
) -> tuple[list[int], dict]:
    """Retrieve candidates by object-topology graphlet score S_G."""
    cfg = dict(cfg or {})
    existing = {int(x) for x in (existing_candidates or [])}
    top_k = int(cfg.get("object_topology_top_k", 3) or 0)
    min_gap = int(cfg.get("loop_min_temporal_gap", 10) or 0)
    q_kf = int(query_kf_idx) if query_kf_idx is not None else None
    q_sig = _ensure_topology_fields(query_signature, cfg)
    debug = {
        "query_kf_idx": q_kf,
        "candidate_scores": [],
        "rejected": [],
        "supplement": [],
    }
    scored = []
    for cand_idx, cand_sig in (all_kf_signatures or {}).items():
        try:
            ci = int(cand_idx)
        except Exception:
            continue
        if q_kf is not None and ci == q_kf:
            continue
        if q_kf is not None and min_gap > 0 and abs(ci - q_kf) < min_gap:
            debug["rejected"].append({"candidate_idx": ci, "reason": "temporal_gap"})
            continue
        comp = score_graphlet_topology_pair(q_sig, cand_sig, cfg)
        entry = {
            "candidate_idx": ci,
            "topology_score": float(comp.get("score", 0.0)),
            "matched_graphlet_count": len(comp.get("matched_graphlets") or []),
            "query_graphlet_count": int(comp.get("query_graphlet_count", 0)),
            "candidate_graphlet_count": int(comp.get("candidate_graphlet_count", 0)),
            "already_visual": ci in existing,
        }
        scored.append((entry["topology_score"], ci, entry))
    scored.sort(key=lambda x: (-float(x[0]), x[1]))
    debug["candidate_scores"] = [e for _, _, e in scored]
    accepted = [e for score, _, e in scored if score > 0.0 and not e.get("already_visual")]
    if top_k > 0:
        accepted = accepted[:top_k]
    debug["supplement"] = [int(e["candidate_idx"]) for e in accepted]
    return list(debug["supplement"]), debug


def retrieve_functionalized_topology_candidates(
    *,
    query_signature: dict,
    all_kf_signatures: dict,
    existing_candidates: list[int],
    cfg: dict,
    query_kf_idx: int | None = None,
) -> tuple[list[int], dict]:
    """Retrieve by S_G * (1 + S_F); S_F is only a ranking boost."""
    cfg = dict(cfg or {})
    existing = {int(x) for x in (existing_candidates or [])}
    top_k = int(cfg.get("functionalized_topology_top_k", 3) or 0)
    min_gap = int(cfg.get("loop_min_temporal_gap", 10) or 0)
    q_kf = int(query_kf_idx) if query_kf_idx is not None else None
    q_sig = _ensure_topology_fields(query_signature, cfg)
    debug = {
        "query_kf_idx": q_kf,
        "candidate_scores": [],
        "rejected": [],
        "supplement": [],
    }
    scored = []
    for cand_idx, cand_sig in (all_kf_signatures or {}).items():
        try:
            ci = int(cand_idx)
        except Exception:
            continue
        if q_kf is not None and ci == q_kf:
            continue
        if q_kf is not None and min_gap > 0 and abs(ci - q_kf) < min_gap:
            debug["rejected"].append({"candidate_idx": ci, "reason": "temporal_gap"})
            continue
        topo = score_graphlet_topology_pair(q_sig, cand_sig, cfg)
        fcomp = score_functional_consistency_on_matches(
            query_sig=q_sig,
            cand_sig=cand_sig,
            matched_graphlets=list(topo.get("matched_graphlets") or []),
            cfg=cfg,
        )
        topo_score = float(topo.get("score", 0.0))
        f_score = float(fcomp.get("score", 0.0))
        combined = topo_score * (1.0 + f_score)
        entry = {
            "candidate_idx": ci,
            "topology_score": topo_score,
            "functional_consistency_score": f_score,
            "combined_score": float(combined),
            "matched_graphlet_count": len(topo.get("matched_graphlets") or []),
            "functional_match_count": int(fcomp.get("functional_match_count", 0)),
            "functional_query_edge_count": int(fcomp.get("functional_query_edge_count", 0)),
            "already_visual": ci in existing,
        }
        scored.append((combined, ci, entry))
    scored.sort(key=lambda x: (-float(x[0]), x[1]))
    debug["candidate_scores"] = [e for _, _, e in scored]
    accepted = [e for score, _, e in scored if score > 0.0 and not e.get("already_visual")]
    if top_k > 0:
        accepted = accepted[:top_k]
    debug["supplement"] = [int(e["candidate_idx"]) for e in accepted]
    return list(debug["supplement"]), debug


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _extract_supplement_semantic_score(
    candidate_idx: int,
    sources: list[str],
    supp_debug: dict,
    topology_debug: dict,
    ftopo_debug: dict,
) -> tuple[float, dict]:
    """Return the best semantic/topology score for a supplement candidate."""
    cand_i = int(candidate_idx)
    source_set = {str(s) for s in (sources or [])}
    per_source: dict[str, float] = {}

    if not source_set or "functional" in source_set:
        for entry in (supp_debug or {}).get("candidate_scores", []) or []:
            try:
                if int(entry.get("candidate_idx", -1)) != cand_i:
                    continue
            except Exception:
                continue
            score = max(
                float(entry.get("score", 0.0) or 0.0),
                float(entry.get("token_score", 0.0) or 0.0),
            )
            per_source["functional"] = max(per_source.get("functional", 0.0), score)

    if not source_set or "object_topology" in source_set:
        for entry in (topology_debug or {}).get("candidate_scores", []) or []:
            try:
                if int(entry.get("candidate_idx", -1)) != cand_i:
                    continue
            except Exception:
                continue
            score = float(entry.get("topology_score", 0.0) or 0.0)
            per_source["object_topology"] = max(per_source.get("object_topology", 0.0), score)

    if not source_set or "functionalized_topology" in source_set:
        for entry in (ftopo_debug or {}).get("candidate_scores", []) or []:
            try:
                if int(entry.get("candidate_idx", -1)) != cand_i:
                    continue
            except Exception:
                continue
            score = max(
                float(entry.get("combined_score", 0.0) or 0.0),
                float(entry.get("topology_score", 0.0) or 0.0),
            )
            per_source["functionalized_topology"] = max(
                per_source.get("functionalized_topology", 0.0), score
            )

    best_source = None
    best_score = 0.0
    for source, score in per_source.items():
        if float(score) > best_score:
            best_source = source
            best_score = float(score)
    return _clamp01(best_score), {
        "candidate_idx": cand_i,
        "sources": list(sources or []),
        "per_source_scores": per_source,
        "best_source": best_source,
        "best_score_raw": float(best_score),
        "best_score": _clamp01(best_score),
    }


def compute_topology_functional_supplement_utility(
    *,
    candidate_idx: int,
    current_idx: int,
    sources: list[str],
    semantic_score: float,
    edge_infos: list[dict],
    node_cfg: dict,
    edge_exists: bool = False,
) -> dict:
    """Compute utility for one geometry-auditioned supplement candidate."""
    cfg = dict(node_cfg or {})
    infos = list(edge_infos or [])
    valid = any(bool(e.get("valid", False)) for e in infos)
    match_frac_min = 0.0
    q_mean = 0.0
    for e in infos:
        try:
            match_frac_min = max(match_frac_min, float(e.get("match_frac_min", 0.0) or 0.0))
        except Exception:
            pass
        try:
            q_mean = max(q_mean, float(e.get("q_mean", 0.0) or 0.0))
        except Exception:
            pass

    audition_min = float(cfg.get("topology_functional_audition_min_match_frac", 0.05) or 0.05)
    select_min = float(cfg.get("topology_functional_select_min_match_frac", 0.08) or 0.08)
    min_utility = float(cfg.get("topology_functional_select_min_utility", 0.35) or 0.35)
    temporal_gap_norm = float(cfg.get("topology_functional_temporal_gap_norm", 50) or 50)

    geom_score = _clamp01((match_frac_min - audition_min) / max(1e-6, 0.20 - audition_min))
    temporal_gap = abs(int(current_idx) - int(candidate_idx))
    temporal_score = _clamp01(temporal_gap / max(1e-6, temporal_gap_norm))

    source_set = {str(s) for s in (sources or [])}
    if "functionalized_topology" in source_set:
        novelty_score = 1.0
    elif "object_topology" in source_set:
        novelty_score = 0.85
    elif "functional" in source_set:
        novelty_score = 0.75
    else:
        novelty_score = 0.5

    w_geom = float(cfg.get("topology_functional_utility_weight_geom", 0.45) or 0.0)
    w_sem = float(cfg.get("topology_functional_utility_weight_semantic", 0.25) or 0.0)
    w_tmp = float(cfg.get("topology_functional_utility_weight_temporal", 0.20) or 0.0)
    w_nov = float(cfg.get("topology_functional_utility_weight_novelty", 0.10) or 0.0)
    utility = (
        w_geom * geom_score
        + w_sem * _clamp01(float(semantic_score or 0.0))
        + w_tmp * temporal_score
        + w_nov * novelty_score
    )

    selected = True
    reject_reason = None
    if edge_exists:
        selected = False
        reject_reason = "edge_already_exists"
    elif not valid:
        selected = False
        reject_reason = "invalid_geometry"
    elif match_frac_min < select_min:
        selected = False
        reject_reason = "match_frac_below_select_threshold"
    elif utility < min_utility:
        selected = False
        reject_reason = "utility_below_threshold"

    return {
        "candidate_idx": int(candidate_idx),
        "sources": list(sources or []),
        "semantic_score": _clamp01(float(semantic_score or 0.0)),
        "match_frac_min": float(match_frac_min),
        "q_mean": float(q_mean),
        "geom_score": float(geom_score),
        "temporal_gap": int(temporal_gap),
        "temporal_score": float(temporal_score),
        "novelty_score": float(novelty_score),
        "utility": float(utility),
        "selected": bool(selected),
        "reject_reason": reject_reason,
    }


def build_strengthen_anchor_pool(
    original_anchor_pool: list[int],
    accepted_supplement_anchor_indices: list[int],
    node_cfg: dict,
) -> tuple[list[int], list[int]]:
    """Build v4 strengthening anchors with supplement anchors opt-in only."""
    pool = [int(c) for c in (original_anchor_pool or [])]
    supplement_used: list[int] = []
    if bool((node_cfg or {}).get("functional_strengthen_supplement_anchors", False)):
        pool = union_keep_order(pool, [int(c) for c in (accepted_supplement_anchor_indices or [])])
        supp_set = {int(c) for c in (accepted_supplement_anchor_indices or [])}
        supplement_used = [int(c) for c in pool if int(c) in supp_set]
    seen = set()
    deduped = []
    for c in pool:
        c_i = int(c)
        if c_i in seen:
            continue
        seen.add(c_i)
        deduped.append(c_i)
    return deduped, supplement_used


def _normalize_chain_id(chain) -> tuple:
    try:
        return tuple(str(x) for x in chain)
    except Exception:
        return tuple()


def _label_chain_idf_score(q_chains_lbl, c_chains_lbl) -> float:
    """Compute IDF-weighted overlap of chain-label tuples."""
    if not q_chains_lbl or not c_chains_lbl:
        return 0.0
    q_set = {tuple(c) for c in q_chains_lbl}
    c_set = {tuple(c) for c in c_chains_lbl}
    inter = q_set & c_set
    if not inter:
        return 0.0
    # Reward longer / more distinctive chains.
    score = 0.0
    for tup in inter:
        # IDF heuristic: longer chain = more distinctive.
        score += min(1.0, 0.4 + 0.2 * (len(tup) - 1))
    denom = max(1, len(q_set | c_set))
    return min(1.0, score / denom)


def _spatial_score(query_centroids, cand_centroids, shared_node_ids, max_dist_m: float) -> float:
    if not shared_node_ids or max_dist_m <= 0:
        return 0.0
    diffs = []
    for nid in shared_node_ids:
        qc = query_centroids.get(nid)
        cc = cand_centroids.get(nid)
        if qc is None or cc is None or len(qc) < 3 or len(cc) < 3:
            continue
        dx = float(qc[0]) - float(cc[0])
        dy = float(qc[1]) - float(cc[1])
        dz = float(qc[2]) - float(cc[2])
        diffs.append((dx * dx + dy * dy + dz * dz) ** 0.5)
    if not diffs:
        return 0.0
    avg = sum(diffs) / len(diffs)
    if avg >= max_dist_m:
        return 0.0
    return float(max(0.0, 1.0 - avg / max_dist_m))


def score_functional_place_pair(query_sig, cand_sig, cfg=None):
    """Score two v2 functional-place signatures.

    Returns a dict containing component scores plus a final weighted score.
    Never raises.
    """
    cfg = dict(cfg or {})
    q = dict(query_sig or {})
    c = dict(cand_sig or {})

    w_nid = float(cfg.get("weight_node_id", 0.35) or 0.0)
    w_chain = float(cfg.get("weight_chain_id", 0.30) or 0.0)
    w_lbl = float(cfg.get("weight_chain_label_idf", 0.15) or 0.0)
    w_direct = float(cfg.get("weight_direct", 0.08) or 0.0)
    w_spatial = float(cfg.get("weight_spatial", 0.07) or 0.0)
    w_remote = float(cfg.get("weight_remote", 0.05) or 0.0)
    use_remote = bool(cfg.get("use_remote_relations", False))
    spatial_max = float(cfg.get("spatial_distance_max_m", cfg.get("acceptance", {}).get("spatial_distance_max_m", 5.0)) or 0.0)

    q_nodes = set(q.get("node_ids") or [])
    c_nodes = set(c.get("node_ids") or [])
    shared_node_ids = sorted(q_nodes & c_nodes)
    union_nodes = q_nodes | c_nodes
    node_id_score = (len(shared_node_ids) / len(union_nodes)) if union_nodes else 0.0

    q_chains = {tuple(_normalize_chain_id(ch)) for ch in (q.get("chain_ids") or [])}
    c_chains = {tuple(_normalize_chain_id(ch)) for ch in (c.get("chain_ids") or [])}
    shared_chains = sorted(q_chains & c_chains)
    union_chains = q_chains | c_chains
    chain_id_score = (len(shared_chains) / len(union_chains)) if union_chains else 0.0

    chain_label_score = _label_chain_idf_score(q.get("chain_labels") or [], c.get("chain_labels") or [])

    q_direct = {tuple(p) for p in (q.get("direct_uo") or [])}
    c_direct = {tuple(p) for p in (c.get("direct_uo") or [])}
    shared_direct = sorted(q_direct & c_direct)
    union_direct = q_direct | c_direct
    direct_score = (len(shared_direct) / len(union_direct)) if union_direct else 0.0

    spatial_score = _spatial_score(
        q.get("centroids") or {}, c.get("centroids") or {}, shared_node_ids, spatial_max
    )

    remote_score = 0.0
    if use_remote:
        q_rem = {(r.get("src"), r.get("dst")) for r in (q.get("remote_relations") or []) if isinstance(r, dict)}
        c_rem = {(r.get("src"), r.get("dst")) for r in (c.get("remote_relations") or []) if isinstance(r, dict)}
        union_rem = q_rem | c_rem
        if union_rem:
            remote_score = len(q_rem & c_rem) / len(union_rem)

    total_w = w_nid + w_chain + w_lbl + w_direct + w_spatial + (w_remote if use_remote else 0.0)
    if total_w <= 0:
        score = 0.0
    else:
        score = (
            w_nid * node_id_score
            + w_chain * chain_id_score
            + w_lbl * chain_label_score
            + w_direct * direct_score
            + w_spatial * spatial_score
            + (w_remote * remote_score if use_remote else 0.0)
        ) / total_w
    score = max(0.0, min(1.0, float(score)))

    return {
        "score": score,
        "node_id_score": node_id_score,
        "chain_id_score": chain_id_score,
        "chain_label_score": chain_label_score,
        "direct_score": direct_score,
        "spatial_score": spatial_score,
        "remote_score": remote_score,
        "shared_node_ids": shared_node_ids,
        "shared_chain_ids": [list(ch) for ch in shared_chains],
        "shared_direct": [list(p) for p in shared_direct],
    }


def score_candidate_against_query(query_signature: dict, candidate_signature: dict, cfg=None) -> dict:
    """Score an already-verified original retrieval anchor with functional evidence.

    This is intentionally a thin wrapper over ``score_functional_place_pair``.
    It does not accept a candidate on weak label-only overlap: a candidate must
    share enough O/C node ids, an exact functional chain id, or a strong O-node
    token plus the configured score threshold.
    """
    cfg = dict(cfg or {})
    q = dict(query_signature or {})
    c = dict(candidate_signature or {})
    comp = score_functional_place_pair(q, c, cfg)
    score = float(comp.get("score", 0.0) or 0.0)
    min_score = float(cfg.get("functional_original_anchor_min_score", 0.25) or 0.0)
    min_shared_oc = int(cfg.get("functional_original_anchor_min_shared_oc_nodes", 2) or 0)
    min_shared_chains = int(cfg.get("functional_original_anchor_min_shared_chain_ids", 1) or 0)

    q_roles = dict(q.get("node_roles") or {})
    c_roles = dict(c.get("node_roles") or {})
    shared_node_ids = list(comp.get("shared_node_ids") or [])
    shared_oc_node_ids = []
    for node_id in shared_node_ids:
        role = str(q_roles.get(node_id) or c_roles.get(node_id) or "").upper()
        if role in {"O", "C"}:
            shared_oc_node_ids.append(node_id)

    shared_chain_ids = [list(ch) for ch in (comp.get("shared_chain_ids") or [])]
    q_strong = set(q.get("strong_tokens") or [])
    c_strong = set(c.get("strong_tokens") or [])
    shared_strong_tokens = sorted(q_strong & c_strong)
    shared_o_node_tokens = [tok for tok in shared_strong_tokens if str(tok).startswith("node:O")]

    has_functional_support = False
    support_reason = "insufficient_functional_support"
    if score >= min_score and len(shared_oc_node_ids) >= min_shared_oc:
        has_functional_support = True
        support_reason = "shared_oc_nodes"
    elif len(shared_chain_ids) >= min_shared_chains:
        has_functional_support = True
        support_reason = "shared_chain_ids"
    elif score >= min_score and shared_o_node_tokens:
        has_functional_support = True
        support_reason = "shared_o_node_token"

    out = dict(comp)
    out.update({
        "score": score,
        "shared_node_ids": shared_node_ids,
        "shared_oc_node_ids": shared_oc_node_ids,
        "shared_chain_ids": shared_chain_ids,
        "shared_strong_tokens": shared_strong_tokens,
        "shared_o_node_tokens": shared_o_node_tokens,
        "has_functional_support": bool(has_functional_support),
        "support_reason": support_reason,
    })
    return out


def _accepts_supplement(comp, cfg) -> bool:
    """v2 acceptance gate for adding a candidate as functional supplement."""
    acc = dict((cfg or {}).get("acceptance") or {})
    min_shared_nodes = int(acc.get("min_shared_node_ids", 2) or 0)
    min_shared_chains = int(acc.get("min_shared_chain_ids", 1) or 0)
    chain_lbl_thr = float(acc.get("chain_label_score_threshold", 0.5) or 0.0)
    spatial_max = float((cfg or {}).get("spatial_distance_max_m", acc.get("spatial_distance_max_m", 5.0)) or 0.0)
    min_score = float(acc.get("min_score", 0.0) or 0.0)

    n_shared_nodes = len(comp.get("shared_node_ids") or [])
    n_shared_chains = len(comp.get("shared_chain_ids") or [])
    chain_lbl = float(comp.get("chain_label_score", 0.0))
    spatial = float(comp.get("spatial_score", 0.0))
    score = float(comp.get("score", 0.0))

    # Hard score gate (when configured).
    if min_score > 0.0 and score < min_score:
        return False

    if n_shared_nodes >= min_shared_nodes:
        return True
    if n_shared_chains >= min_shared_chains:
        return True
    if chain_lbl >= chain_lbl_thr and spatial > 0.0 and spatial_max > 0.0:
        return True
    return False


def is_query_signature_fresh_for_frame(query_signature, frame_id) -> bool:
    """Check that the published query signature was built for the given frame_id."""
    if not query_signature or frame_id is None:
        return False
    try:
        sig_frame = query_signature.get("frame_idx")
        return sig_frame is not None and int(sig_frame) == int(frame_id)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# v3 helpers: token IDF based scoring + support-window generation.
# ---------------------------------------------------------------------------


def build_token_document_frequency(all_kf_signatures: dict) -> dict:
    """Build a token -> document frequency map from all keyframe signatures."""
    df: dict = {}
    if not all_kf_signatures:
        return df
    for sig in all_kf_signatures.values():
        if not sig:
            continue
        # Use the union of strong+weak tokens (deduplicated per-document).
        toks = set()
        toks.update(sig.get("strong_tokens") or [])
        toks.update(sig.get("weak_tokens") or [])
        if not toks:
            toks.update(sig.get("tokens") or [])
        for t in toks:
            df[t] = df.get(t, 0) + 1
    return df


def token_idf(token: str, df: dict, total_docs: int, cfg=None) -> float:
    """Normalized IDF in [0, 1] with a heavy down-weight for too-common tokens."""
    cfg = dict(cfg or {})
    if total_docs is None or total_docs <= 0:
        return 1.0
    common_clip = float(cfg.get("token_idf_common_clip", 0.15) or 0.0)
    common_clip_min_docs = int(cfg.get("token_idf_common_clip_min_docs", 10) or 0)
    min_df = max(1, int(cfg.get("token_idf_min_df", 1) or 1))
    df_t = int((df or {}).get(token, 0))
    if df_t < min_df:
        return 1.0
    frac = df_t / float(total_docs)
    if (
        common_clip > 0.0
        and total_docs >= max(1, common_clip_min_docs)
        and frac >= common_clip
    ):
        return 0.05
    import math as _math
    denom = _math.log(1.0 + total_docs)
    if denom <= 0.0:
        return 1.0
    return float(_math.log((1.0 + total_docs) / (1.0 + df_t)) / denom)


def score_functional_tokens(
    query_sig: dict,
    cand_sig: dict,
    df: dict,
    total_docs: int,
    cfg=None,
) -> dict:
    """Compute IDF-weighted token similarity between two signatures.

    Returns a dict with keys: token_score, shared_tokens, shared_strong_tokens,
    shared_weak_tokens, shared_o_node_tokens, strong_token_count, has_o_node,
    reject_reason.
    """
    cfg = dict(cfg or {})
    q = dict(query_sig or {})
    c = dict(cand_sig or {})
    q_strong = set(q.get("strong_tokens") or [])
    c_strong = set(c.get("strong_tokens") or [])
    q_weak = set(q.get("weak_tokens") or [])
    c_weak = set(c.get("weak_tokens") or [])

    shared_strong = sorted(q_strong & c_strong)
    shared_weak = sorted(q_weak & c_weak)
    shared_tokens = sorted(set(shared_strong) | set(shared_weak))

    strong_boost = float(cfg.get("token_strong_boost", 1.5) or 1.0)
    score_sum = 0.0
    for t in shared_strong:
        score_sum += strong_boost * token_idf(t, df, total_docs, cfg)
    for t in shared_weak:
        score_sum += token_idf(t, df, total_docs, cfg)

    # Normalize by number of shared tokens with a small floor so that
    # 1 strong rare token already gives a meaningful score.
    denom = max(1, len(shared_tokens))
    token_score = float(min(1.0, score_sum / denom))

    shared_o_node_tokens = [t for t in shared_strong if t.startswith("node:O")]
    has_o_node = bool(shared_o_node_tokens)

    return {
        "token_score": token_score,
        "shared_tokens": shared_tokens,
        "shared_strong_tokens": shared_strong,
        "shared_weak_tokens": shared_weak,
        "shared_o_node_tokens": shared_o_node_tokens,
        "strong_token_count": len(shared_strong),
        "has_o_node": has_o_node,
        "reject_reason": None,
    }


def _accepts_token_supplement(comp: dict, cfg: dict) -> tuple:
    """v3 acceptance gate using token IDF.

    Returns (accept, reject_reason).
    """
    min_strong = int((cfg or {}).get("functional_loop_min_strong_token_count", 1) or 0)
    min_score = float((cfg or {}).get("functional_loop_min_token_idf_score", 0.5) or 0.0)
    require_o = bool((cfg or {}).get("functional_loop_require_o_node", True))

    if int(comp.get("strong_token_count", 0)) < min_strong:
        return False, "insufficient_strong_tokens"
    if float(comp.get("token_score", 0.0)) < min_score:
        return False, "token_score_below_threshold"
    if require_o and not bool(comp.get("has_o_node", False)):
        return False, "missing_o_node_token"
    return True, None


def support_window_candidates(
    anchor: int,
    *,
    current_kf: int,
    window: int,
    n_keyframes: int,
    excluded: set = None,
    min_neighbor_gap: int = 0,
) -> list:
    """Return ordered support window indices around an accepted anchor.

    Excludes:
      - current keyframe;
      - invalid (negative) indices and indices >= n_keyframes;
      - duplicates;
      - indices in ``excluded`` (e.g., already-issued original retrievals or
        already-accepted anchors);
      - immediate temporal neighbors of current keyframe within
        ``min_neighbor_gap``.
    """
    excluded = set(excluded or [])
    out: list = []
    seen: set = set()
    if window <= 0:
        return out
    # interleave +/- to bias towards closer neighbors first
    for d in range(1, int(window) + 1):
        for s in (anchor - d, anchor + d):
            if s == int(current_kf):
                continue
            if s < 0:
                continue
            if n_keyframes is not None and n_keyframes > 0 and s >= int(n_keyframes):
                continue
            if min_neighbor_gap > 0 and abs(s - int(current_kf)) <= int(min_neighbor_gap):
                continue
            if s in excluded:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(int(s))
    return out





def retrieve_functional_candidates(
    *,
    query_kf_idx,
    query_signature,
    all_kf_signatures,
    existing_candidates=None,
    cfg=None,
    mode: str = "loop",
    precomputed_df: dict | None = None,
    total_docs: int | None = None,
):
    """Return (supplement_indices, debug) for candidates beyond the existing set.

    Never modifies or removes ``existing_candidates``. Always JSON-serializable.
    """
    cfg = dict(cfg or {})
    existing = set(int(x) for x in (existing_candidates or []))
    debug = {
        "mode": mode,
        "query_kf_idx": int(query_kf_idx) if query_kf_idx is not None else None,
        "supplement": [],
        "candidate_scores": [],
        "rejected": [],
        "fallback_reason": None,
        "use_token_idf": False,
        "total_docs": 0,
    }

    if not query_signature or not (
        query_signature.get("node_ids")
        or query_signature.get("chain_ids")
        or query_signature.get("strong_tokens")
        or query_signature.get("weak_tokens")
    ):
        debug["fallback_reason"] = "empty_query_signature"
        return [], debug

    if mode == "loop":
        add_top_k = int(cfg.get("loop_add_top_k", 2) or 0)
        legacy_min_gap = int(cfg.get("loop_min_temporal_gap", 10) or 0)
        v3_min_gap = int(cfg.get("functional_loop_min_temporal_gap", legacy_min_gap) or 0)
    else:
        add_top_k = int(cfg.get("reloc_add_top_k", 3) or 0)
        legacy_min_gap = 0
        v3_min_gap = 0

    use_token_idf = bool(cfg.get("use_functional_token_idf", False))
    df: dict = {}
    docs_n = 0
    if use_token_idf:
        if precomputed_df is not None and total_docs is not None:
            df = dict(precomputed_df or {})
            docs_n = int(total_docs or 0)
        else:
            df = build_token_document_frequency(all_kf_signatures or {})
            docs_n = len(all_kf_signatures or {})
        # Be conservative when the corpus is too small for stable IDF.
        min_corpus = int(cfg.get("token_idf_min_corpus", 5) or 0)
        if docs_n < max(1, min_corpus):
            use_token_idf = False
            debug["fallback_reason"] = "small_corpus_use_legacy"
    debug["use_token_idf"] = bool(use_token_idf)
    debug["total_docs"] = int(docs_n)

    min_gap = v3_min_gap if use_token_idf else legacy_min_gap

    q_kf = int(query_kf_idx) if query_kf_idx is not None else None

    scored = []
    for cand_idx, c_sig in (all_kf_signatures or {}).items():
        try:
            ci = int(cand_idx)
        except Exception:
            continue
        if q_kf is not None and ci == q_kf:
            continue
        if ci in existing:
            continue
        if mode == "loop" and q_kf is not None and min_gap > 0 and abs(ci - q_kf) < min_gap:
            debug["rejected"].append({"candidate_idx": ci, "reason": "temporal_gap"})
            continue
        comp = score_functional_place_pair(query_signature, c_sig, cfg)
        entry = {
            "candidate_idx": ci,
            "score": float(comp.get("score", 0.0)),
            "node_id_score": float(comp.get("node_id_score", 0.0)),
            "chain_id_score": float(comp.get("chain_id_score", 0.0)),
            "chain_label_score": float(comp.get("chain_label_score", 0.0)),
            "spatial_score": float(comp.get("spatial_score", 0.0)),
            "direct_score": float(comp.get("direct_score", 0.0)),
            "remote_score": float(comp.get("remote_score", 0.0)),
            "shared_node_ids": list(comp.get("shared_node_ids") or []),
            "shared_chain_ids": [list(ch) for ch in (comp.get("shared_chain_ids") or [])],
        }
        if use_token_idf:
            tcomp = score_functional_tokens(query_signature, c_sig, df, docs_n, cfg)
            accept, reject_reason = _accepts_token_supplement(tcomp, cfg)
            entry.update({
                "token_score": float(tcomp.get("token_score", 0.0)),
                "shared_tokens": list(tcomp.get("shared_tokens") or []),
                "shared_strong_tokens": list(tcomp.get("shared_strong_tokens") or []),
                "shared_weak_tokens": list(tcomp.get("shared_weak_tokens") or []),
                "shared_o_node_tokens": list(tcomp.get("shared_o_node_tokens") or []),
                "strong_token_count": int(tcomp.get("strong_token_count", 0)),
                "has_o_node": bool(tcomp.get("has_o_node", False)),
                "accepted": bool(accept),
                "accept_reason": "idf_strong_tokens" if accept else None,
                "reject_reason": reject_reason,
            })
            if not accept:
                debug["rejected"].append({
                    "candidate_idx": ci,
                    "reason": reject_reason or "below_token_acceptance",
                    "token_score": entry["token_score"],
                    "strong_token_count": entry["strong_token_count"],
                })
        else:
            accept = _accepts_supplement(comp, cfg)
            entry["accepted"] = bool(accept)
            if not accept:
                debug["rejected"].append({
                    "candidate_idx": ci,
                    "reason": "below_acceptance",
                    "score": entry["score"],
                })
        scored.append(entry)
    debug["candidate_scores"] = scored

    accepted = [e for e in scored if e.get("accepted")]
    if use_token_idf:
        accepted.sort(key=lambda e: -float(e.get("token_score", 0.0)))
    else:
        accepted.sort(key=lambda e: -float(e.get("score", 0.0)))
    if add_top_k > 0:
        accepted = accepted[:add_top_k]
    supplement_indices = [int(e["candidate_idx"]) for e in accepted]
    debug["supplement"] = supplement_indices
    return supplement_indices, debug


def order_candidates_for_reloc(
    *,
    original_candidates,
    supplement_candidates,
    query_signature,
    candidate_signatures,
    cfg=None,
):
    """Order candidates for sequential reloc verification.

    Strategy: high-score functional supplements first, then originals in
    their original order, then remaining lower-score supplements (if any).
    Never drops any candidate; always preserves originals.
    """
    cfg = dict(cfg or {})
    originals = [int(c) for c in (original_candidates or [])]
    supplements = [int(c) for c in (supplement_candidates or [])]

    use_token_idf = bool(cfg.get("use_functional_token_idf", False))
    df: dict = {}
    total_docs = 0
    if use_token_idf and candidate_signatures:
        df = build_token_document_frequency(candidate_signatures or {})
        total_docs = len(candidate_signatures or {})
        if total_docs < int(cfg.get("token_idf_min_corpus", 5) or 0):
            use_token_idf = False

    score_map: dict = {}
    if query_signature and candidate_signatures:
        for cidx in set(originals) | set(supplements):
            c_sig = (candidate_signatures or {}).get(int(cidx)) or (candidate_signatures or {}).get(cidx)
            if not c_sig:
                continue
            if use_token_idf:
                tcomp = score_functional_tokens(query_signature, c_sig, df, total_docs, cfg)
                score_map[int(cidx)] = float(tcomp.get("token_score", 0.0))
            else:
                comp = score_functional_place_pair(query_signature, c_sig, cfg)
                score_map[int(cidx)] = float(comp.get("score", 0.0))

    high_thr = float(cfg.get("reloc_high_score_threshold", 0.4) or 0.0)
    high_supplement = [c for c in supplements if score_map.get(c, 0.0) >= high_thr]
    high_supplement.sort(key=lambda c: -score_map.get(c, 0.0))
    low_supplement = [c for c in supplements if c not in high_supplement]
    low_supplement.sort(key=lambda c: -score_map.get(c, 0.0))

    seen: set = set()
    ordered: list = []
    for c in high_supplement + originals + low_supplement:
        if c in seen:
            continue
        seen.add(c)
        ordered.append(c)
    return ordered


def union_keep_order(originals, supplements):
    """Return originals followed by supplements not already in originals."""
    seen = set()
    out = []
    for c in (originals or []):
        ci = int(c)
        if ci in seen:
            continue
        seen.add(ci)
        out.append(ci)
    for c in (supplements or []):
        ci = int(c)
        if ci in seen:
            continue
        seen.add(ci)
        out.append(ci)
    return out
