"""0108-1050-semantic-restart"""



import argparse
import datetime
import json
import os
import pathlib
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
from mast3r_slam.functional_graph.realtime_viz import publish_functional_graph_viz_snapshot
import torch.multiprocessing as mp


class _LocalSharedValue:
    def __init__(self, value):
        self.value = value


class _LocalManager:
    """Small manager substitute for no-viz thread-backend evaluation.

    This avoids ``mp.Manager()`` startup stalls on servers where process
    manager initialization is unreliable.  It is only selected by an explicit
    environment variable and should not be used with multiprocessing backends.
    """

    def RLock(self):
        return threading.RLock()

    def Value(self, _typecode, value):
        return _LocalSharedValue(value)

    def list(self):
        return []

    def dict(self):
        return {}

    def Queue(self):
        return queue.Queue()


def _ensure_node_place_recognition_imports():
    """Import functional graph retrieval helpers only when node-place logic runs."""
    global build_signature_from_online_state
    global build_functional_place_signature
    global rerank_candidates_by_nodes
    global retrieve_functional_candidates
    global retrieve_object_topology_candidates
    global retrieve_functionalized_topology_candidates
    global _extract_supplement_semantic_score
    global compute_topology_functional_supplement_utility
    global build_strengthen_anchor_pool
    global score_candidate_against_query
    global order_candidates_for_reloc
    global is_query_signature_fresh_for_frame
    global union_keep_order
    global support_window_candidates

    if "rerank_candidates_by_nodes" in globals():
        return

    from mast3r_slam.functional_graph.node_place_recognition import (
        build_signature_from_online_state,
        build_functional_place_signature,
        rerank_candidates_by_nodes,
        retrieve_functional_candidates,
        retrieve_object_topology_candidates,
        retrieve_functionalized_topology_candidates,
        _extract_supplement_semantic_score,
        compute_topology_functional_supplement_utility,
        build_strengthen_anchor_pool,
        score_candidate_against_query,
        order_candidates_for_reloc,
        is_query_signature_fresh_for_frame,
        union_keep_order,
        support_window_candidates,
    )


def _write_json_atomic(path, payload):
    from mast3r_slam.semantic.io_utils import write_json_atomic

    write_json_atomic(path, payload)


def union_keep_order(*seqs):
    out = []
    seen = set()
    for seq in seqs:
        for item in seq or []:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def _make_shared_manager(args):
    use_local = os.environ.get("MAST3R_SLAM_LOCAL_MANAGER", "0") == "1"
    use_viz_thread = os.environ.get("MAST3R_SLAM_VIZ_THREAD", "0") == "1"
    if use_local:
        if not args.no_viz and not use_viz_thread:
            raise RuntimeError("MAST3R_SLAM_LOCAL_MANAGER=1 requires --no-viz or MAST3R_SLAM_VIZ_THREAD=1")
        if os.environ.get("MAST3R_SLAM_BACKEND_THREAD", "0") != "1":
            raise RuntimeError("MAST3R_SLAM_LOCAL_MANAGER=1 requires MAST3R_SLAM_BACKEND_THREAD=1")
        print("[INFO] using local manager for thread-backend evaluation")
        return _LocalManager()
    return mp.Manager()


def _edge_exists(factor_graph, a: int, b: int) -> bool:
    """Return True if (a,b) or (b,a) already exists in the factor graph.

    Treats edges as undirected even though they are stored as directed pairs.
    """
    try:
        ii = factor_graph.ii
        jj = factor_graph.jj
        if ii is None or jj is None or ii.numel() == 0:
            return False
        ii_cpu = ii.detach().cpu().tolist()
        jj_cpu = jj.detach().cpu().tolist()
        a_i, b_i = int(a), int(b)
        for u, v in zip(ii_cpu, jj_cpu):
            if (u == a_i and v == b_i) or (u == b_i and v == a_i):
                return True
        return False
    except Exception:
        return False


def _factor_edge_count(factor_graph) -> int:
    try:
        ii = factor_graph.ii
        if ii is None:
            return 0
        return int(ii.numel())
    except Exception:
        return 0


def _new_factor_edges_since(factor_graph, start_count: int) -> list[dict]:
    try:
        ii = factor_graph.ii
        jj = factor_graph.jj
        if ii is None or jj is None:
            return []
        ii_cpu = ii.detach().cpu().tolist()
        jj_cpu = jj.detach().cpu().tolist()
        start = max(0, int(start_count))
        out = []
        for k in range(start, min(len(ii_cpu), len(jj_cpu))):
            out.append({"ii": int(ii_cpu[k]), "jj": int(jj_cpu[k])})
        return out
    except Exception:
        return []


def _is_topology_functional_loop_deficit(
    *,
    current_idx: int,
    accepted_original_anchors: list[int],
    factor_graph,
    node_cfg: dict,
) -> tuple[bool, dict]:
    """Decide whether safe topology/functional supplement should run."""
    cfg = dict(node_cfg or {})
    deficit_only = bool(cfg.get("topology_functional_supplement_deficit_only", False))
    min_gap = int(cfg.get("topology_functional_deficit_min_temporal_gap", 15) or 0)
    anchors = [int(a) for a in (accepted_original_anchors or [])]
    has_long_range_original = any(abs(int(current_idx) - int(a)) >= min_gap for a in anchors)
    should_run = True
    if deficit_only:
        should_run = not has_long_range_original
    return bool(should_run), {
        "deficit_only": bool(deficit_only),
        "accepted_original_anchors": list(anchors),
        "min_temporal_gap": int(min_gap),
        "has_long_range_original": bool(has_long_range_original),
        "should_run_supplement": bool(should_run),
    }



def _node_place_jsonl_write(payload):
    """Best-effort append of one JSONL event to the configured path.

    Path resolution order:
      1. ``config["node_place_recognition"]["jsonl_path"]`` (set in main process).
      2. ``logs/node_place_debug/<save_as>_node_place.jsonl`` fallback.
    Failures must NEVER crash SLAM.
    """
    try:
        npr = (config.get("node_place_recognition") or {}) if isinstance(config, dict) else {}
        if not bool(npr.get("jsonl_debug_enabled", True)):
            return
        path = npr.get("jsonl_path")
        if not path:
            return
        import json as _json
        import os as _os
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Silent: never crash the SLAM loop on debug write errors.
        pass





def _write_baseline_rgb_frame_copy(baseline_dumper, frame_idx: int, frame) -> str | None:
    uimg = getattr(frame, "uimg", None)
    if uimg is None or not torch.is_tensor(uimg) or uimg.ndim != 3 or uimg.shape[-1] != 3:
        return None
    rgb_u8 = (uimg.detach().clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    out_path = pathlib.Path(baseline_dumper.baseline_dir) / "frames" / f"frame_{int(frame_idx):06d}_rgb.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(out_path), bgr_u8):
        return None
    return str(out_path)


def relocalization(frame, keyframes, factor_graph, retrieval_database, states=None):


    with keyframes.lock:

        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        original_retrieval_inds = list(retrieval_inds)
        node_cfg = config.get("node_place_recognition", {}) or {}
        if node_cfg.get("enabled", False):
            _ensure_node_place_recognition_imports()

        # ---- v1 legacy reloc rerank/filter (off by default in v2 config) ----
        if (
            states is not None
            and node_cfg.get("enabled", False)
            and node_cfg.get("rerank_reloc", False)
            and node_cfg.get("reloc_filter", False)
            and retrieval_inds
        ):
            try:
                query_sig = states.get_current_node_query_signature() or {}
                cand_sigs = states.get_kf_node_signatures(list(retrieval_inds))
                reranked, node_debug = rerank_candidates_by_nodes(
                    list(retrieval_inds),
                    query_signature=query_sig,
                    candidate_signatures=cand_sigs,
                    cfg=node_cfg,
                    mode="reloc",
                )
                retrieval_inds = reranked
                states.append_node_place_debug({
                    "type": "reloc_legacy",
                    "frame_id": int(getattr(frame, "frame_id", -1)),
                    "original_candidates": list(original_retrieval_inds),
                    "final_candidates": list(retrieval_inds),
                    "debug": node_debug,
                })
                if node_cfg.get("debug", False):
                    print(
                        f"[NODE_PLACE][reloc_legacy] frame={getattr(frame, 'frame_id', -1)} "
                        f"original={original_retrieval_inds} final={list(retrieval_inds)} "
                    )
            except Exception as exc:
                print(f"[NODE_PLACE][reloc_legacy] ERROR: {type(exc).__name__}: {exc}")
                retrieval_inds = list(original_retrieval_inds)

        # ---- v2 functional supplement + sequential verification ----
        supplement_inds: list = []
        stale_query = False
        if (
            states is not None
            and node_cfg.get("enabled", False)
            and node_cfg.get("reloc_add_functional_candidates", False)
        ):
            try:
                query_sig = states.get_current_node_query_signature() or {}
                require_match = bool(node_cfg.get("reloc_require_frame_id_match", True))
                fid = int(getattr(frame, "frame_id", -1))
                if require_match and not is_query_signature_fresh_for_frame(query_sig, fid):
                    stale_query = True
                else:
                    all_sigs = states.get_all_kf_node_signatures()
                    supplement_inds, _supp_debug = retrieve_functional_candidates(
                        query_kf_idx=None,
                        query_signature=query_sig,
                        all_kf_signatures=all_sigs,
                        existing_candidates=list(retrieval_inds),
                        cfg=node_cfg,
                        mode="reloc",
                    )
            except Exception as exc:
                print(f"[NODE_PLACE][reloc] ERROR: {type(exc).__name__}: {exc}")

        # Build candidate order: high-score supplement first, then originals,
        # then remaining supplements. Originals are never dropped.
        if supplement_inds and not stale_query:
            try:
                query_sig = states.get_current_node_query_signature() or {} if states is not None else {}
                cand_sigs = states.get_kf_node_signatures(
                    list(set(list(retrieval_inds) + list(supplement_inds)))
                ) if states is not None else {}
                candidate_order = order_candidates_for_reloc(
                    original_candidates=list(retrieval_inds),
                    supplement_candidates=list(supplement_inds),
                    query_signature=query_sig,
                    candidate_signatures=cand_sigs,
                    cfg=node_cfg,
                )
            except Exception:
                candidate_order = union_keep_order(list(retrieval_inds), list(supplement_inds))
        else:
            candidate_order = union_keep_order(list(retrieval_inds), list(supplement_inds))

        successful_loop_closure = False
        accepted_candidate = None
        geometry_results: list = []

        if candidate_order:

            keyframes.append(frame)
            n_kf = len(keyframes)
            print(
                "RELOCALIZING (sequential) against kf",
                n_kf - 1,
                "candidates=",
                candidate_order,
            )
            for cand_idx in candidate_order:
                ok = factor_graph.add_factors(
                    [n_kf - 1],
                    [int(cand_idx)],
                    config["reloc"]["min_match_frac"],
                    is_reloc=config["reloc"]["strict"],
                )
                geometry_results.append({
                    "candidate_idx": int(cand_idx),
                    "passed": bool(ok),
                })
                if ok:
                    accepted_candidate = int(cand_idx)
                    successful_loop_closure = True
                    break

            if successful_loop_closure:
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print(f"Success! Relocalized via candidate kf={accepted_candidate}")

                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[accepted_candidate].clone()
            else:

                keyframes.pop_last()
                print("Failed to relocalize")

        # Emit debug regardless of success/failure.
        if states is not None and node_cfg.get("enabled", False):
            try:
                debug_payload = {
                    "type": "reloc",
                    "frame_id": int(getattr(frame, "frame_id", -1)),
                    "original_retrieval": list(original_retrieval_inds),
                    "functional_supplement": list(supplement_inds),
                    "candidate_order": list(candidate_order),
                    "geometry_results": geometry_results,
                    "accepted_candidate": accepted_candidate,
                    "stale_query_signature": bool(stale_query),
                }
                states.append_node_place_debug(debug_payload)
                _node_place_jsonl_write(debug_payload)
                if node_cfg.get("debug", False):
                    print(
                        f"[NODE_PLACE][reloc] frame={getattr(frame, 'frame_id', -1)} "
                        f"original={original_retrieval_inds} supplement={supplement_inds} "
                        f"order={candidate_order} accepted={accepted_candidate} "
                        f"results={geometry_results} stale_query={stale_query}"
                    )
            except Exception:
                pass

        if successful_loop_closure:

            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):

    try:
        _run_backend_impl(cfg, model, states, keyframes, K)
    except BaseException as exc:  # noqa: BLE001 - we want to log everything
        import traceback as _tb
        import sys as _sys
        _sys.stderr.write(
            f"[BACKEND] FATAL: {type(exc).__name__}: {exc}\n"
        )
        _tb.print_exc()
        _sys.stderr.flush()

        raise


class BackendThread(threading.Thread):
    def __init__(self, cfg, model, states, keyframes, K):
        super().__init__(daemon=True)
        self._args = (cfg, model, states, keyframes, K)
        self.exception = None
        self.exitcode = None

    def run(self):
        try:
            run_backend(*self._args)
            self.exitcode = 0
        except BaseException as exc:  # noqa: BLE001
            self.exception = exc
            self.exitcode = 1


def _run_backend_impl(cfg, model, states, keyframes, K):
    set_global_config(cfg)
    backend_profile_enabled = os.environ.get("FG_PROFILE_STAGES", "0") == "1"

    def _backend_profile(name: str, seconds: float, *, kf_idx: int | None = None, count: int = 1) -> None:
        if not backend_profile_enabled:
            return
        try:
            payload = {"name": name, "seconds": float(seconds), "count": int(count)}
            if kf_idx is not None:
                payload["kf_idx"] = int(kf_idx)
            states.append_runtime_profile_event(payload)
        except Exception:
            pass

    device = keyframes.device

    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:

        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:

            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database, states)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:

            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue


        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]

        _t_backend_kf = time.perf_counter() if backend_profile_enabled else 0.0
        _t_retrieval = time.perf_counter() if backend_profile_enabled else 0.0
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        if backend_profile_enabled:
            _backend_profile("backend.retrieval_database", time.perf_counter() - _t_retrieval, kf_idx=idx)
        # Node-aware rerank/filter for loop-closure candidates (no-op when disabled).
        _t_functional_loop = time.perf_counter() if backend_profile_enabled else 0.0
        original_retrieval_inds = list(retrieval_inds)
        node_cfg = config.get("node_place_recognition", {}) or {}
        if node_cfg.get("enabled", False):
            _ensure_node_place_recognition_imports()
        # ---- v1 legacy rerank/filter path (off by default in v2 config) ----
        if (
            node_cfg.get("enabled", False)
            and node_cfg.get("rerank_loop", False)
            and node_cfg.get("loop_filter", False)
            and retrieval_inds
        ):
            try:
                query_sig = states.get_kf_node_signature(idx) or {}
                cand_sigs = states.get_kf_node_signatures(list(retrieval_inds))
                reranked, node_debug = rerank_candidates_by_nodes(
                    list(retrieval_inds),
                    query_signature=query_sig,
                    candidate_signatures=cand_sigs,
                    cfg=node_cfg,
                    mode="loop",
                )
                retrieval_inds = reranked
                filtered = sorted(set(original_retrieval_inds) - set(retrieval_inds))
                states.append_node_place_debug({
                    "type": "loop_legacy",
                    "kf_idx": int(idx),
                    "original_candidates": list(original_retrieval_inds),
                    "final_candidates": list(retrieval_inds),
                    "debug": node_debug,
                })
                if node_cfg.get("debug", False):
                    print(
                        f"[NODE_PLACE][loop_legacy] kf={idx} "
                        f"original={original_retrieval_inds} final={list(retrieval_inds)} "
                        f"filtered={filtered} "
                        f"best={node_debug.get('best_candidate')} "
                        f"best_score={node_debug.get('best_score')}"
                    )
            except Exception as exc:
                print(f"[NODE_PLACE][loop_legacy] ERROR: {type(exc).__name__}: {exc}")
                retrieval_inds = list(original_retrieval_inds)

        # ---- v2 functional supplement + topology supplement proposal paths ----
        # These streams propose additional loop candidates only. Original
        # MASt3R retrieval candidates are kept intact and verified first.
        supplement_inds: list = []
        topology_inds: list = []
        ftopo_inds: list = []
        supp_debug: dict = {}
        topology_debug: dict = {}
        ftopo_debug: dict = {}
        verify_individually = bool(node_cfg.get("functional_loop_verify_individually", False))
        has_supplement_streams = bool(
            node_cfg.get("loop_add_functional_candidates", False)
            or node_cfg.get("object_topology_enabled", False)
            or node_cfg.get("functionalized_topology_enabled", False)
        )
        deficit_gate_retrieval = (
            bool(node_cfg.get("functional_loop_deficit_gate_retrieval", False))
            and os.environ.get("FG_DISABLE_LOOP_DEFICIT_GATE", "0") != "1"
            and verify_individually
        )
        supplement_retrieval_skipped = False
        supplement_retrieval_skip_reason = None
        query_sig = {}
        all_sigs = {}
        if (
            node_cfg.get("enabled", False)
            and has_supplement_streams
            and not deficit_gate_retrieval
        ):
            try:
                _t_loop_sig = time.perf_counter() if backend_profile_enabled else 0.0
                query_sig = states.get_kf_node_signature(idx) or {}
                all_sigs = states.get_all_kf_node_signatures()
                if backend_profile_enabled:
                    _backend_profile(
                        "backend.functional_loop.signature_fetch",
                        time.perf_counter() - _t_loop_sig,
                        kf_idx=idx,
                    )
                if node_cfg.get("loop_add_functional_candidates", False):
                    _t = time.perf_counter() if backend_profile_enabled else 0.0
                    supplement_inds, supp_debug = retrieve_functional_candidates(
                        query_kf_idx=idx,
                        query_signature=query_sig,
                        all_kf_signatures=all_sigs,
                        existing_candidates=list(retrieval_inds),
                        cfg=node_cfg,
                        mode="loop",
                    )
                    if backend_profile_enabled:
                        _backend_profile(
                            "backend.functional_loop.functional_candidate_retrieve",
                            time.perf_counter() - _t,
                            kf_idx=idx,
                        )
                        _backend_profile(
                            "backend.functional_loop.functional_candidate_count",
                            0.0,
                            kf_idx=idx,
                            count=len(supplement_inds or []),
                        )
                if node_cfg.get("object_topology_enabled", False):
                    _t = time.perf_counter() if backend_profile_enabled else 0.0
                    topology_inds, topology_debug = retrieve_object_topology_candidates(
                        query_signature=query_sig,
                        all_kf_signatures=all_sigs,
                        existing_candidates=list(retrieval_inds),
                        cfg=node_cfg,
                        query_kf_idx=int(idx),
                    )
                    if backend_profile_enabled:
                        _backend_profile(
                            "backend.functional_loop.topology_candidate_retrieve",
                            time.perf_counter() - _t,
                            kf_idx=idx,
                        )
                        _backend_profile(
                            "backend.functional_loop.topology_candidate_count",
                            0.0,
                            kf_idx=idx,
                            count=len(topology_inds or []),
                        )
                if node_cfg.get("functionalized_topology_enabled", False):
                    _t = time.perf_counter() if backend_profile_enabled else 0.0
                    ftopo_inds, ftopo_debug = retrieve_functionalized_topology_candidates(
                        query_signature=query_sig,
                        all_kf_signatures=all_sigs,
                        existing_candidates=list(retrieval_inds),
                        cfg=node_cfg,
                        query_kf_idx=int(idx),
                    )
                    if backend_profile_enabled:
                        _backend_profile(
                            "backend.functional_loop.ftopo_candidate_retrieve",
                            time.perf_counter() - _t,
                            kf_idx=idx,
                        )
                        _backend_profile(
                            "backend.functional_loop.ftopo_candidate_count",
                            0.0,
                            kf_idx=idx,
                            count=len(ftopo_inds or []),
                        )
                if verify_individually:
                    # Keep retrieval_inds untouched; supplements are verified
                    # one-by-one by MASt3R geometry after the base call.
                    if node_cfg.get("debug", False):
                        print(
                            f"[NODE_PLACE][loop+] kf={idx} original={original_retrieval_inds} "
                            f"functional={supplement_inds} topology={topology_inds} "
                            f"functionalized={ftopo_inds}"
                        )
                else:
                    # Legacy v2 batch path: only append functional supplements.
                    final_with_supp = union_keep_order(list(retrieval_inds), supplement_inds)
                    retrieval_inds = final_with_supp
                    debug_payload = {
                        "type": "loop",
                        "kf_idx": int(idx),
                        "frame_id": int(getattr(frame, "frame_id", -1)),
                        "original_retrieval": list(original_retrieval_inds),
                        "functional_supplement": list(supplement_inds),
                        "final_candidates": list(retrieval_inds),
                        "candidate_scores": supp_debug.get("candidate_scores", []),
                        "rejected": supp_debug.get("rejected", []),
                        "fallback_reason": supp_debug.get("fallback_reason"),
                    }
                    states.append_node_place_debug(debug_payload)
                    _node_place_jsonl_write(debug_payload)
                    if node_cfg.get("debug", False):
                        best = None
                        if supp_debug.get("candidate_scores"):
                            best_e = max(supp_debug["candidate_scores"], key=lambda e: e.get("score", 0.0))
                            best = (best_e.get("candidate_idx"), round(float(best_e.get("score", 0.0)), 4))
                        print(
                            f"[NODE_PLACE][loop] kf={idx} original={original_retrieval_inds} "
                            f"supplement={supplement_inds} final={list(retrieval_inds)} best={best}"
                        )
            except Exception as exc:
                print(f"[NODE_PLACE][loop] ERROR: {type(exc).__name__}: {exc}")
        kf_idx += list(retrieval_inds)

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        base_edge_start = _factor_edge_count(factor_graph)
        base_new_edges: list = []
        accepted_original_anchors: list = []
        if kf_idx:

            _t = time.perf_counter() if backend_profile_enabled else 0.0
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )
            if backend_profile_enabled:
                _backend_profile(
                    "backend.functional_loop.original_retrieval_factor_add",
                    time.perf_counter() - _t,
                    kf_idx=idx,
                )
                _backend_profile(
                    "backend.functional_loop.original_retrieval_factor_candidate_count",
                    0.0,
                    kf_idx=idx,
                    count=len(kf_idx),
                )
            base_new_edges = _new_factor_edges_since(factor_graph, base_edge_start)
            original_set = set(int(c) for c in original_retrieval_inds)
            seen_original_anchor = set()
            for edge in base_new_edges:
                a_i = int(edge.get("ii", -1))
                b_i = int(edge.get("jj", -1))
                anchor_i = None
                if b_i == int(idx) and a_i in original_set:
                    anchor_i = a_i
                elif a_i == int(idx) and b_i in original_set:
                    anchor_i = b_i
                if anchor_i is not None and anchor_i not in seen_original_anchor:
                    seen_original_anchor.add(anchor_i)
                    accepted_original_anchors.append(anchor_i)

        _t_deficit = time.perf_counter() if backend_profile_enabled else 0.0
        should_run_supplement, deficit_debug = _is_topology_functional_loop_deficit(
            current_idx=int(idx),
            accepted_original_anchors=accepted_original_anchors,
            factor_graph=factor_graph,
            node_cfg=node_cfg,
        )
        if backend_profile_enabled:
            _backend_profile(
                "backend.functional_loop.deficit_check",
                time.perf_counter() - _t_deficit,
                kf_idx=idx,
            )

        if (
            node_cfg.get("enabled", False)
            and has_supplement_streams
            and deficit_gate_retrieval
        ):
            deficit_only = bool(node_cfg.get("topology_functional_supplement_deficit_only", False))
            should_fetch_supplement = (not deficit_only) or bool(should_run_supplement)
            if not should_fetch_supplement:
                supplement_retrieval_skipped = True
                supplement_retrieval_skip_reason = "no_loop_deficit"
                if backend_profile_enabled:
                    _backend_profile(
                        "backend.functional_loop.supplement_retrieval_skipped",
                        0.0,
                        kf_idx=idx,
                        count=1,
                    )
            else:
                try:
                    _t_loop_sig = time.perf_counter() if backend_profile_enabled else 0.0
                    query_sig = states.get_kf_node_signature(idx) or {}
                    if backend_profile_enabled:
                        _backend_profile(
                            "backend.functional_loop.get_query_sig",
                            time.perf_counter() - _t_loop_sig,
                            kf_idx=idx,
                        )
                    _t_loop_sig = time.perf_counter() if backend_profile_enabled else 0.0
                    all_sigs = states.get_all_kf_node_signatures()
                    if backend_profile_enabled:
                        elapsed = time.perf_counter() - _t_loop_sig
                        _backend_profile("backend.functional_loop.get_all_sigs", elapsed, kf_idx=idx)
                        _backend_profile("backend.functional_loop.signature_fetch", elapsed, kf_idx=idx)
                    if node_cfg.get("loop_add_functional_candidates", False):
                        _t = time.perf_counter() if backend_profile_enabled else 0.0
                        supplement_inds, supp_debug = retrieve_functional_candidates(
                            query_kf_idx=idx,
                            query_signature=query_sig,
                            all_kf_signatures=all_sigs,
                            existing_candidates=list(retrieval_inds),
                            cfg=node_cfg,
                            mode="loop",
                        )
                        if backend_profile_enabled:
                            _backend_profile(
                                "backend.functional_loop.functional_candidate_retrieve",
                                time.perf_counter() - _t,
                                kf_idx=idx,
                            )
                            _backend_profile(
                                "backend.functional_loop.functional_candidate_count",
                                0.0,
                                kf_idx=idx,
                                count=len(supplement_inds or []),
                            )
                    if node_cfg.get("object_topology_enabled", False):
                        _t = time.perf_counter() if backend_profile_enabled else 0.0
                        topology_inds, topology_debug = retrieve_object_topology_candidates(
                            query_signature=query_sig,
                            all_kf_signatures=all_sigs,
                            existing_candidates=list(retrieval_inds),
                            cfg=node_cfg,
                            query_kf_idx=int(idx),
                        )
                        if backend_profile_enabled:
                            _backend_profile(
                                "backend.functional_loop.topology_candidate_retrieve",
                                time.perf_counter() - _t,
                                kf_idx=idx,
                            )
                            _backend_profile(
                                "backend.functional_loop.topology_candidate_count",
                                0.0,
                                kf_idx=idx,
                                count=len(topology_inds or []),
                            )
                    if node_cfg.get("functionalized_topology_enabled", False):
                        _t = time.perf_counter() if backend_profile_enabled else 0.0
                        ftopo_inds, ftopo_debug = retrieve_functionalized_topology_candidates(
                            query_signature=query_sig,
                            all_kf_signatures=all_sigs,
                            existing_candidates=list(retrieval_inds),
                            cfg=node_cfg,
                            query_kf_idx=int(idx),
                        )
                        if backend_profile_enabled:
                            _backend_profile(
                                "backend.functional_loop.ftopo_candidate_retrieve",
                                time.perf_counter() - _t,
                                kf_idx=idx,
                            )
                            _backend_profile(
                                "backend.functional_loop.ftopo_candidate_count",
                                0.0,
                                kf_idx=idx,
                                count=len(ftopo_inds or []),
                            )
                except Exception as exc:
                    print(f"[NODE_PLACE][loop-deferred] ERROR: {type(exc).__name__}: {exc}")

        # ---- Geometry verification for OR supplement pool ----
        _t_pool = time.perf_counter() if backend_profile_enabled else 0.0
        supplement_pool: list = []
        supplement_source_map: dict[int, list[str]] = {}
        for _src, _vals in (
            ("functional", supplement_inds),
            ("object_topology", topology_inds),
            ("functionalized_topology", ftopo_inds),
        ):
            for _cand in _vals or []:
                try:
                    _ci = int(_cand)
                except Exception:
                    continue
                supplement_source_map.setdefault(_ci, [])
                if _src not in supplement_source_map[_ci]:
                    supplement_source_map[_ci].append(_src)
                if _ci not in supplement_pool:
                    supplement_pool.append(_ci)

        original_set = set(int(c) for c in original_retrieval_inds)
        clean_supplement_pool: list[int] = []
        for cand in supplement_pool:
            cand_i = int(cand)
            if cand_i == int(idx) or cand_i < 0:
                continue
            if cand_i in original_set:
                continue
            if cand_i == int(idx) - 1:
                continue
            if _edge_exists(factor_graph, cand_i, int(idx)):
                continue
            if cand_i not in clean_supplement_pool:
                clean_supplement_pool.append(cand_i)
        supplement_pool = clean_supplement_pool
        if backend_profile_enabled:
            _backend_profile(
                "backend.functional_loop.supplement_pool_build",
                time.perf_counter() - _t_pool,
                kf_idx=idx,
            )
            _backend_profile(
                "backend.functional_loop.supplement_pool_count",
                0.0,
                kf_idx=idx,
                count=len(supplement_pool),
            )

        accepted_supplement_anchors: list[dict] = []
        accepted_supplement_anchor_indices: list[int] = []
        supplement_geometry_results: list[dict] = []
        supplement_audition_results: list[dict] = []
        selected_supplement_edges: list[int] = []
        if (
            node_cfg.get("enabled", False)
            and bool(node_cfg.get("topology_functional_supplement_enabled", False) or verify_individually)
            and bool(should_run_supplement)
            and supplement_pool
        ):
            max_supp_edges = int(node_cfg.get("topology_functional_max_supplement_edges_per_kf", 1) or 0)
            audition_enabled = bool(node_cfg.get("topology_functional_audition_enabled", False))
            audition_thr = float(
                node_cfg.get(
                    "topology_functional_audition_min_match_frac",
                    config["local_opt"]["min_match_frac"],
                )
                or config["local_opt"]["min_match_frac"]
            )
            select_thr = float(
                node_cfg.get(
                    "topology_functional_select_min_match_frac",
                    config["local_opt"]["min_match_frac"],
                )
                or config["local_opt"]["min_match_frac"]
            )
            if audition_enabled and hasattr(factor_graph, "evaluate_factors"):
                for cand_i in supplement_pool:
                    sources = list(supplement_source_map.get(int(cand_i), []))
                    try:
                        _t = time.perf_counter() if backend_profile_enabled else 0.0
                        edge_infos = factor_graph.evaluate_factors(
                            [int(cand_i)], [int(idx)], audition_thr, is_reloc=False,
                        )
                        if backend_profile_enabled:
                            _backend_profile(
                                "backend.functional_loop.supplement_audition_geometry",
                                time.perf_counter() - _t,
                                kf_idx=idx,
                            )
                    except Exception as _e:
                        edge_infos = []
                        semantic_score, semantic_debug = _extract_supplement_semantic_score(
                            int(cand_i), sources, supp_debug, topology_debug, ftopo_debug,
                        )
                        util = {
                            "candidate_idx": int(cand_i),
                            "sources": sources,
                            "semantic_score": float(semantic_score),
                            "selected": False,
                            "reject_reason": f"audition_error:{type(_e).__name__}",
                            "edge_infos": [],
                            "semantic_debug": semantic_debug,
                        }
                        supplement_audition_results.append(util)
                        continue
                    _t = time.perf_counter() if backend_profile_enabled else 0.0
                    semantic_score, semantic_debug = _extract_supplement_semantic_score(
                        int(cand_i), sources, supp_debug, topology_debug, ftopo_debug,
                    )
                    util = compute_topology_functional_supplement_utility(
                        candidate_idx=int(cand_i),
                        current_idx=int(idx),
                        sources=sources,
                        semantic_score=semantic_score,
                        edge_infos=list(edge_infos or []),
                        node_cfg=node_cfg,
                        edge_exists=_edge_exists(factor_graph, int(cand_i), int(idx)),
                    )
                    util["edge_infos"] = list(edge_infos or [])
                    util["semantic_debug"] = semantic_debug
                    supplement_audition_results.append(util)
                    if backend_profile_enabled:
                        _backend_profile(
                            "backend.functional_loop.supplement_score",
                            time.perf_counter() - _t,
                            kf_idx=idx,
                        )

                selected = [r for r in supplement_audition_results if r.get("selected")]
                selected.sort(
                    key=lambda r: (
                        -float(r.get("utility", 0.0)),
                        -float(r.get("match_frac_min", 0.0)),
                        -float(r.get("semantic_score", 0.0)),
                        -int(r.get("temporal_gap", 0)),
                    )
                )
                if max_supp_edges > 0:
                    selected = selected[:max_supp_edges]
                selected_supplement_edges = [int(r["candidate_idx"]) for r in selected]
            else:
                selected_supplement_edges = list(supplement_pool)
                if max_supp_edges > 0:
                    selected_supplement_edges = selected_supplement_edges[:max_supp_edges]

            for cand_i in selected_supplement_edges:
                before = _factor_edge_count(factor_graph)
                try:
                    _t = time.perf_counter() if backend_profile_enabled else 0.0
                    if hasattr(factor_graph, "add_factors_with_info"):
                        ok, edge_infos = factor_graph.add_factors_with_info(
                            [int(cand_i)], [int(idx)], select_thr, is_reloc=False,
                        )
                    else:
                        ok = factor_graph.add_factors([int(cand_i)], [int(idx)], select_thr)
                        edge_infos = _new_factor_edges_since(factor_graph, before)
                    ok_b = bool(ok)
                    if backend_profile_enabled:
                        _backend_profile(
                            "backend.functional_loop.supplement_add_factor",
                            time.perf_counter() - _t,
                            kf_idx=idx,
                        )
                except Exception as _e:
                    entry = {
                        "candidate_idx": int(cand_i),
                        "sources": list(supplement_source_map.get(int(cand_i), [])),
                        "added": False,
                        "passed": False,
                        "threshold": select_thr,
                        "edge_infos": [],
                        "error": f"{type(_e).__name__}: {_e}",
                    }
                    supplement_geometry_results.append(entry)
                    continue
                if not edge_infos:
                    edge_infos = _new_factor_edges_since(factor_graph, before)
                entry = {
                    "candidate_idx": int(cand_i),
                    "sources": list(supplement_source_map.get(int(cand_i), [])),
                    "added": bool(ok_b),
                    "passed": bool(ok_b),
                    "threshold": select_thr,
                    "edge_infos": list(edge_infos or []),
                }
                supplement_geometry_results.append(entry)
                if ok_b:
                    accepted_supplement_anchor_indices.append(int(cand_i))
                    accepted_supplement_anchors.append(entry)

        # ---- v4 functional/topology strengthening of geometry-accepted anchors ----
        functional_original_anchor_scores: list = []
        functional_original_selected_anchors: list = []
        functional_original_support_accepted: list = []
        functional_original_geometry_results: list = []
        strengthen_anchor_entries: list[dict] = []
        strengthen_supplement_anchors: list[int] = []
        if (
            node_cfg.get("enabled", False)
            and node_cfg.get("functional_strengthen_original_retrieval", False)
        ):
            try:
                _t_strengthen_score = time.perf_counter() if backend_profile_enabled else 0.0
                require_geometry_pass = bool(
                    node_cfg.get("functional_original_anchor_require_geometry_pass", True)
                )
                original_anchor_pool = (
                    list(accepted_original_anchors)
                    if require_geometry_pass
                    else list(original_retrieval_inds)
                )
                original_anchor_pool, strengthen_supplement_anchors = build_strengthen_anchor_pool(
                    original_anchor_pool,
                    accepted_supplement_anchor_indices,
                    node_cfg,
                )
                if original_anchor_pool:
                    query_sig = states.get_kf_node_signature(idx) or {}
                    cand_sigs = states.get_kf_node_signatures(original_anchor_pool)
                    if (
                        deficit_gate_retrieval
                        and bool(node_cfg.get("functional_loop_preserve_original_strengthening_when_no_deficit", True))
                    ):
                        if node_cfg.get("object_topology_enabled", False) and not topology_debug:
                            _, topology_debug = retrieve_object_topology_candidates(
                                query_signature=query_sig,
                                all_kf_signatures=cand_sigs,
                                existing_candidates=list(retrieval_inds),
                                cfg=node_cfg,
                                query_kf_idx=int(idx),
                            )
                            topology_debug["small_anchor_pool_only"] = True
                        if node_cfg.get("functionalized_topology_enabled", False) and not ftopo_debug:
                            _, ftopo_debug = retrieve_functionalized_topology_candidates(
                                query_signature=query_sig,
                                all_kf_signatures=cand_sigs,
                                existing_candidates=list(retrieval_inds),
                                cfg=node_cfg,
                                query_kf_idx=int(idx),
                            )
                            ftopo_debug["small_anchor_pool_only"] = True
                    for cand in original_anchor_pool:
                        cand_sig = cand_sigs.get(int(cand)) or cand_sigs.get(cand) or {}
                        score_dbg = score_candidate_against_query(query_sig, cand_sig, node_cfg)
                        score_entry = {
                            "candidate_idx": int(cand),
                            "score": float(score_dbg.get("score", 0.0)),
                            "has_functional_support": bool(score_dbg.get("has_functional_support", False)),
                            "support_reason": str(score_dbg.get("support_reason", "")),
                            "shared_node_ids": list(score_dbg.get("shared_node_ids") or []),
                            "shared_oc_node_ids": list(score_dbg.get("shared_oc_node_ids") or []),
                            "shared_chain_ids": [list(ch) for ch in (score_dbg.get("shared_chain_ids") or [])],
                            "shared_strong_tokens": list(score_dbg.get("shared_strong_tokens") or []),
                        }
                        topo_score = 0.0
                        for e in topology_debug.get("candidate_scores", []) if topology_debug else []:
                            if int(e.get("candidate_idx", -1)) == int(cand):
                                topo_score = max(topo_score, float(e.get("topology_score", 0.0)))
                        ftopo_score = 0.0
                        fcons_score = 0.0
                        for e in ftopo_debug.get("candidate_scores", []) if ftopo_debug else []:
                            if int(e.get("candidate_idx", -1)) == int(cand):
                                ftopo_score = max(ftopo_score, float(e.get("combined_score", 0.0)))
                                fcons_score = max(fcons_score, float(e.get("functional_consistency_score", 0.0)))
                        min_anchor_score = float(node_cfg.get("functional_original_anchor_min_score", 0.25) or 0.0)
                        has_topology_support = topo_score >= min_anchor_score or ftopo_score >= min_anchor_score
                        score_entry.update({
                            "topology_score": float(topo_score),
                            "functionalized_topology_score": float(ftopo_score),
                            "functional_consistency_score": float(fcons_score),
                            "has_topology_support": bool(has_topology_support),
                            "anchor_source": (
                                "supplement_anchor"
                                if int(cand) in set(accepted_supplement_anchor_indices)
                                else "original_anchor"
                            ),
                        })
                        functional_original_anchor_scores.append(score_entry)
                    eligible = [
                        e for e in functional_original_anchor_scores
                        if e.get("has_functional_support") or e.get("has_topology_support")
                    ]
                    order_index = {int(c): i for i, c in enumerate(original_anchor_pool)}
                    eligible.sort(
                        key=lambda e: (
                            -max(
                                float(e.get("score", 0.0)),
                                float(e.get("topology_score", 0.0)),
                                float(e.get("functionalized_topology_score", 0.0)),
                            ),
                            order_index.get(int(e.get("candidate_idx", -1)), 0),
                        )
                    )
                    max_orig_anchor = int(node_cfg.get("functional_original_max_anchors_per_kf", 2) or 0)
                    if max_orig_anchor > 0:
                        eligible = eligible[:max_orig_anchor]
                    functional_original_selected_anchors = [int(e["candidate_idx"]) for e in eligible]
                    strengthen_anchor_entries = list(eligible)
                if backend_profile_enabled:
                    _backend_profile(
                        "backend.functional_loop.strengthen_score",
                        time.perf_counter() - _t_strengthen_score,
                        kf_idx=idx,
                    )

                _t_strengthen_support = time.perf_counter() if backend_profile_enabled else 0.0
                _strengthen_add_count = 0
                if original_anchor_pool:
                    support_thr = float(node_cfg.get("functional_original_support_min_match_frac", 0.06) or 0.06)
                    window = int(node_cfg.get("functional_original_expand_window", 2) or 0)
                    max_support = int(node_cfg.get("functional_original_max_support_edges_per_anchor", 2) or 0)
                    n_kf = len(keyframes)
                    excluded_base = set(int(c) for c in original_retrieval_inds)
                    excluded_base.update(int(c) for c in accepted_supplement_anchor_indices)
                    excluded_base.add(int(idx))
                    excluded_base.add(int(idx) - 1)
                    for anchor in functional_original_selected_anchors:
                        support_cands = support_window_candidates(
                            int(anchor),
                            current_kf=int(idx),
                            window=window,
                            n_keyframes=n_kf,
                            excluded=excluded_base,
                            min_neighbor_gap=1,
                        )
                        accepted_for_anchor = 0
                        for support in support_cands:
                            if max_support > 0 and accepted_for_anchor >= max_support:
                                break
                            support_i = int(support)
                            if _edge_exists(factor_graph, support_i, int(idx)):
                                functional_original_geometry_results.append({
                                    "candidate_idx": support_i,
                                    "type": "strengthened_support",
                                    "anchor": int(anchor),
                                    "passed": False,
                                    "threshold": support_thr,
                                    "skipped": "edge_already_exists",
                                })
                                continue
                            try:
                                _t = time.perf_counter() if backend_profile_enabled else 0.0
                                ok_s = factor_graph.add_factors(
                                    [support_i], [int(idx)], support_thr,
                                )
                                ok_s_b = bool(ok_s)
                                if backend_profile_enabled:
                                    _backend_profile(
                                        "backend.functional_loop.strengthen_add_factor",
                                        time.perf_counter() - _t,
                                        kf_idx=idx,
                                    )
                                    _strengthen_add_count += 1
                            except Exception as _e:
                                functional_original_geometry_results.append({
                                    "candidate_idx": support_i,
                                    "type": "strengthened_support",
                                    "anchor": int(anchor),
                                    "passed": False,
                                    "threshold": support_thr,
                                    "error": f"{type(_e).__name__}: {_e}",
                                })
                                continue
                            functional_original_geometry_results.append({
                                "candidate_idx": support_i,
                                "type": "strengthened_support",
                                "anchor": int(anchor),
                                "passed": ok_s_b,
                                "threshold": support_thr,
                            })
                            if ok_s_b:
                                accepted_for_anchor += 1
                                if support_i not in functional_original_support_accepted:
                                    functional_original_support_accepted.append(support_i)
                if backend_profile_enabled:
                    _backend_profile(
                        "backend.functional_loop.strengthen_support_total",
                        time.perf_counter() - _t_strengthen_support,
                        kf_idx=idx,
                    )
                    _backend_profile(
                        "backend.functional_loop.strengthen_add_factor_count",
                        0.0,
                        kf_idx=idx,
                        count=_strengthen_add_count,
                    )
            except Exception as exc:
                print(f"[NODE_PLACE][loop-v4] ERROR: {type(exc).__name__}: {exc}")

        if node_cfg.get("enabled", False):
            try:
                _t = time.perf_counter() if backend_profile_enabled else 0.0
                debug_payload_loop = {
                    "type": "loop",
                    "mode": "safe_v4_topology_functional_audition",
                    "kf_idx": int(idx),
                    "frame_id": int(getattr(frame, "frame_id", -1)),
                    "original_retrieval": list(original_retrieval_inds),
                    "base_new_edges": list(base_new_edges),
                    "accepted_original_anchors": list(accepted_original_anchors),
                    "deficit_debug": dict(deficit_debug or {}),
                    "supplement_retrieval_skipped": bool(supplement_retrieval_skipped),
                    "supplement_retrieval_skip_reason": supplement_retrieval_skip_reason,
                    "deficit_gate_retrieval": bool(deficit_gate_retrieval),
                    "functional_candidates": list(supplement_inds),
                    "object_topology_candidates": list(topology_inds),
                    "functionalized_topology_candidates": list(ftopo_inds),
                    "supplement_pool": list(supplement_pool),
                    "supplement_audition_results": list(supplement_audition_results),
                    "selected_supplement_edges": list(selected_supplement_edges),
                    "accepted_supplement_anchors": list(accepted_supplement_anchors),
                    "supplement_geometry_results": list(supplement_geometry_results),
                    "functional_debug": dict(supp_debug or {}),
                    "topology_debug": dict(topology_debug or {}),
                    "functionalized_topology_debug": dict(ftopo_debug or {}),
                    "functional_original_anchor_scores": list(functional_original_anchor_scores),
                    "functional_original_selected_anchors": list(functional_original_selected_anchors),
                    "functional_original_support_accepted": list(functional_original_support_accepted),
                    "functional_original_geometry_results": list(functional_original_geometry_results),
                    "strengthen_original_anchors": [
                        int(c) for c in functional_original_selected_anchors
                        if int(c) not in set(accepted_supplement_anchor_indices)
                    ],
                    "functional_strengthen_supplement_anchors": bool(
                        node_cfg.get("functional_strengthen_supplement_anchors", False)
                    ),
                    "strengthen_supplement_anchors": list(strengthen_supplement_anchors),
                    "strengthen_anchors": list(functional_original_selected_anchors),
                    "strengthen_anchor_entries": list(strengthen_anchor_entries),
                    "strengthened_support_edges": list(functional_original_geometry_results),
                    "edge_count_before": int(base_edge_start),
                    "edge_count_after": int(_factor_edge_count(factor_graph)),
                }
                states.append_node_place_debug(debug_payload_loop)
                _node_place_jsonl_write(debug_payload_loop)
                if node_cfg.get("debug", False):
                    print(
                        f"[NODE_PLACE][loop+] kf={idx} "
                        f"functional={supplement_inds} topology={topology_inds} "
                        f"ftopo={ftopo_inds} accepted_supp={accepted_supplement_anchor_indices} "
                        f"strengthen={functional_original_selected_anchors}"
                    )
                if backend_profile_enabled:
                    _backend_profile(
                        "backend.functional_loop.debug_build",
                        time.perf_counter() - _t,
                        kf_idx=idx,
                    )
            except Exception:
                pass

        if backend_profile_enabled:
            _backend_profile(
                "backend.functional_loop_detection",
                time.perf_counter() - _t_functional_loop,
                kf_idx=idx,
            )

        with states.lock:

            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()


        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:

            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)
        if backend_profile_enabled:
            _backend_profile("backend.keyframe_total", time.perf_counter() - _t_backend_kf, kf_idx=idx)


def infer_debug_dump_sequence_name(dataset_path: str) -> str:
    path = pathlib.Path(dataset_path)
    parts = list(path.parts)
    if len(parts) >= 3 and parts[-1] in {"raw_rgb", "rgb", "rgb_SLAM", "rgb_SLAM2"}:
        return "/".join(parts[-3:-1])
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return str(path)


if __name__ == "__main__":



    mp.set_start_method(os.environ.get("MAST3R_SLAM_MP_START_METHOD", "spawn"))

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.set_grad_enabled(False)

    device = "cuda:0"

    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to an RGB sequence directory.")
    parser.add_argument("--config", default="config/fungraph_eval_node_place.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument(
        "--intrinsics-mode",
        choices=["config", "calib", "no-calib"],
        default="config",
        help=(
            "Camera intrinsics handling. 'config' preserves the config file, "
            "'calib' requires --calib and uses calibrated optimization, "
            "'no-calib' forces ray-based optimization even if the config inherits "
            "a calibrated setup."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument(
        "--fg-temp-localize-after-max-frames",
        action="store_true",
        help=(
            "Temporary realtime mode: when --max-frames N is provided, process the full "
            "dataset but freeze reconstruction/semantic/functional-graph updates after "
            "frame N-1; later frames update only the tracked camera pose."
        ),
    )
    parser.add_argument(
        "--fg-temp-freeze-map-after-frame",
        type=int,
        default=-1,
        help=(
            "Temporary realtime mode: freeze reconstruction/semantic/functional-graph "
            "updates from this 0-based frame index onward while continuing localization."
        ),
    )
    parser.add_argument("--enable-rampp", action="store_true")
    parser.add_argument("--rampp-ckpt", default="checkpoints/ram_plus_swin_large_14m.pth")
    parser.add_argument("--rampp-device", default="cuda")
    parser.add_argument("--rampp-print-every", type=int, default=3)
    parser.add_argument("--enable-deepseek", action="store_true")
    parser.add_argument(
        "--fixed-semantic-atlas-json",
        default="",
        help=(
            "Temporary/debug mode: load a fixed semantic atlas JSON and use it "
            "to build per-frame frame_result without RAM++ or DeepSeek."
        ),
    )
    parser.add_argument(
        "--fixed-semantic-atlas-inline",
        default="",
        help=(
            "Temporary/debug mode: inline JSON payload for a fixed semantic atlas. "
            "Takes precedence over --fixed-semantic-atlas-json."
        ),
    )
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--deepseek-model", default=os.environ.get("DEEPSEEK_MODEL", ""))
    parser.add_argument("--deepseek-timeout", type=int, default=60)
    parser.add_argument("--deepseek-retries", type=int, default=2)
    parser.add_argument("--deepseek-cache-dir", default="logs/deepseek_cache")
    parser.add_argument("--deepseek-print-every", type=int, default=3)
    parser.add_argument("--scene-lock-m", type=int, default=2)
    parser.add_argument("--scene-lock-thresh", type=float, default=0.8)
    parser.add_argument(
        "--enable-scene-switch",
        action="store_true",
        help=(
            "Enable low-frequency scene rechecking and hysteresis-based scene switching after the initial "
            "scene is locked. Disabled by default to avoid extra DeepSeek calls on single-room sequences."
        ),
    )
    parser.add_argument("--scene-recheck-every", type=int, default=10)
    parser.add_argument("--scene-switch-m", type=int, default=2)
    parser.add_argument("--scene-switch-thresh", type=float, default=0.85)
    parser.add_argument("--scene-switch-cooldown", type=int, default=20)
    parser.add_argument(
        "--disable-scene-switch",
        action="store_true",
        help="Compatibility flag: force-disable scene switch even if --enable-scene-switch is provided.",
    )
    parser.add_argument(
        "--semantic-keyframes-only-after-scene-lock",
        action="store_true",
        help=(
            "After DeepSeek has locked/built the scene atlas, run RAM++/DeepSeek/SAM3 "
            "only on frames selected as SLAM keyframes. Default preserves per-frame semantics."
        ),
    )
    parser.add_argument(
        "--functional-graph-keyframes-only-after-scene-lock",
        action="store_true",
        help=(
            "After the scene atlas is available, update the online functional graph only "
            "for SLAM keyframes. Default preserves per-frame functional graph updates."
        ),
    )
    parser.add_argument("--pre-atlas-uc-mode", choices=["all", "none"], default="all")
    parser.add_argument(
        "--semantic-out-dir",
        default="",
        help="If set, semantic outputs (scene/atlas/frames/run_args) will be saved under this directory.",
    )
    parser.add_argument(
        "--disable-intermediate-outputs",
        action="store_true",
        help=(
            "Disable per-frame/debug outputs for FPS benchmarking while preserving final "
            "online_functional_graph.json and final reconstruction PLY."
        ),
    )
    parser.add_argument(
        "--fps-profile-output",
        default="",
        help="If set, write final Functional-SLAM FPS and module timing summary to this JSON path.",
    )
    parser.add_argument("--enable-sam3", action="store_true")
    parser.add_argument("--sam3-ckpt", default="checkpoints/sam3.pt")
    parser.add_argument("--sam3-bpe-path", default="sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--sam3-confidence", type=float, default=0.3)
    parser.add_argument("--sam3-overlap-iou", type=float, default=0.9)
    parser.add_argument("--sam3-thr-o", type=float, default=0.5)
    parser.add_argument("--sam3-thr-c", type=float, default=0.5)
    parser.add_argument("--sam3-thr-u", type=float, default=0.5)
    parser.add_argument("--sam3-u-cover-thr", type=float, default=0.95)
    parser.add_argument("--sam3-u-cover-pad", type=float, default=0.0)
    parser.add_argument("--sam3-u-parent-contain-thr", type=float, default=0.9)
    parser.add_argument("--sam3-force-u", default="")
    parser.add_argument("--sam3-save-vis", action="store_true")
    parser.add_argument(
        "--sam3-keep-border-no-link-detections",
        action="store_true",
        help=(
            "Keep SAM3 detections that touch the image border even when they have no "
            "selected local O/C/U links. Default preserves the existing border cleanup."
        ),
    )
    parser.add_argument(
        "--sam3-keep-parent-without-u",
        action="store_true",
        help=(
            "Keep O/C detections even if no selected U child is detected. Default preserves "
            "the existing r2_parent_without_u cleanup."
        ),
    )
    parser.add_argument(
        "--sam3-processor-resolution",
        type=int,
        default=1008,
        help="Internal square resolution used by Sam3Processor before the SAM3 image backbone. Default 1008 preserves original behavior. Suggested experiments: 896, 784, 672, 560.",
    )
    parser.add_argument(
        "--sam3-same-type-drop-multi-parent-links",
        action="store_true",
        help="If enabled, drop all same-type local links for a child when that child has multiple parents within O-C / C-U / O-U respectively.",
    )
    parser.add_argument("--fg-overlay-scene-keep-ratio", type=float, default=0.50)
    parser.add_argument("--fg-overlay-clear-scene-near-graph", action="store_true")
    parser.add_argument("--fg-overlay-clear-node-radius-scale", type=float, default=1.8)
    parser.add_argument("--fg-overlay-clear-edge-radius-scale", type=float, default=1.6)
    parser.add_argument("--fg-overlay-export-focus", action="store_true")
    parser.add_argument("--fg-obs-depth-boxplot-enable", dest="fg_obs_depth_boxplot_enable", action="store_true")
    parser.add_argument("--fg-obs-depth-boxplot-disable", dest="fg_obs_depth_boxplot_enable", action="store_false")
    parser.add_argument("--fg-obs-depth-boxplot-iqr-scale", type=float, default=1.5)
    parser.add_argument("--fg-obs-depth-boxplot-near-iqr-scale", type=float, default=None)
    parser.add_argument("--fg-obs-depth-boxplot-min-keep-ratio", type=float, default=0.35)
    parser.add_argument("--fg-obs-dbscan-enable", dest="fg_obs_dbscan_enable", action="store_true")
    parser.add_argument("--fg-obs-dbscan-disable", dest="fg_obs_dbscan_enable", action="store_false")
    parser.add_argument("--fg-kf-geom-dbscan-enable", dest="fg_kf_geom_dbscan_enable", action="store_true")
    parser.add_argument("--fg-kf-geom-dbscan-disable", dest="fg_kf_geom_dbscan_enable", action="store_false")
    parser.add_argument("--fg-kf-geom-dbscan-eps", type=float, default=0.025)
    parser.add_argument("--fg-kf-geom-dbscan-min-samples", type=int, default=10)
    parser.add_argument("--fg-kf-geom-dbscan-min-keep-ratio", type=float, default=0.4)
    parser.add_argument(
        "--fg-keep-border-no-link-observations",
        action="store_true",
        help=(
            "Keep detections that touch the image border even when no local functional "
            "link selected them. Default preserves the existing extraction cleanup."
        ),
    )
    parser.add_argument("--fg-enable-cabinet-aggregation", dest="fg_enable_cabinet_aggregation", action="store_true")
    parser.add_argument("--fg-disable-cabinet-aggregation", dest="fg_enable_cabinet_aggregation", action="store_false")
    parser.add_argument("--fg-cabinet-output", dest="fg_cabinet_output", action="store_true")
    parser.add_argument("--fg-no-cabinet-output", dest="fg_cabinet_output", action="store_false")
    parser.add_argument(
        "--fg-suppress-objects",
        default="stove",
        help="Comma-separated object labels to suppress from the functional graph. Default preserves legacy behavior.",
    )
    parser.add_argument(
        "--fg-hint-only-objects",
        default="cabinet",
        help="Comma-separated object labels used only as detection hints. Default preserves legacy behavior.",
    )
    parser.add_argument("--fg-temp-no-door-fusion", action="store_true")
    parser.add_argument(
        "--fg-temp-disable-fusion-observations",
        default="",
        help="Temporary comma-separated frame:det_idx selectors whose matched tracks should not use 3D fusion.",
    )
    parser.add_argument("--fg-temp-drop-lid-nodes", action="store_true")
    parser.add_argument("--fg-temp-drop-overlap-lid-with-cap", action="store_true")
    parser.add_argument("--fg-temp-cap-lid-overlap-iou", type=float, default=0.9)
    parser.add_argument("--fg-temp-remote-tentative-color", action="store_true")
    parser.add_argument("--fg-temp-block-remote-labels", default="")
    parser.add_argument("--fg-temp-final-local-edge-invariant", action="store_true")
    parser.add_argument("--fg-temp-drop-final-labels", default="")
    parser.add_argument("--fg-temp-drop-border-u-nodes", action="store_true")
    parser.add_argument("--fg-temp-keep-single-frame-cup-handle", action="store_true")
    parser.add_argument("--fg-temp-limit-door-drawer-units", action="store_true")
    parser.add_argument(
        "--fg-temp-merge-drawers-final",
        action="store_true",
        help="Temporary: before final export, merge all drawer nodes into one drawer node and redirect drawer-unit edges.",
    )
    parser.add_argument(
        "--fg-temp-bottle-single-cap-final",
        action="store_true",
        help="Temporary: before final export, keep only the strongest cap linked to each bottle and remove other cap nodes.",
    )
    parser.add_argument(
        "--fg-temp-bottle-cap-min-distance",
        type=float,
        default=0.0,
        help="Temporary: before final export, push cap nodes away from their bottle to at least this display distance in meters.",
    )
    parser.add_argument(
        "--fg-temp-parent-single-child-per-label-final",
        action="store_true",
        help="Temporary: before final export, keep only one same-label child part edge per parent, ranked by 2D link support.",
    )
    parser.add_argument(
        "--fg-temp-node-pos-point-centroid",
        action="store_true",
        help="Temporary: place graph node markers at observed point-cloud centroids for final overlay/realtime snapshots.",
    )
    parser.add_argument(
        "--fg-temp-realtime-final-adjustments",
        action="store_true",
        help="Apply final-export temporary graph filters to realtime visualization snapshots only.",
    )
    parser.add_argument(
        "--fg-temp-cabinet-iou-only-aggregation",
        action="store_true",
        help="Temporary: aggregate cabinet carrier members only when their 3D IoU is greater than zero.",
    )
    parser.add_argument(
        "--fg-temp-exclude-cabinet-aggregation-observations",
        default="",
        help="Temporary comma-separated frame:det_idx selectors to exclude from cabinet aggregation.",
    )
    parser.add_argument("--fg-temp-stage1-min-proj-iou", type=float, default=0.0)
    parser.add_argument("--debug-dump-baseline-dir", default="")
    parser.add_argument("--debug-dump-gt-file", default="")
    parser.add_argument("--debug-dump-sequence-name", default="")
    parser.add_argument("--assoc-debug-root", default="logs/assoc_debug")
    parser.add_argument("--assoc-debug-run-name", default="")
    parser.add_argument("--assoc-export-2d-overlay", action="store_true")
    parser.add_argument("--assoc-export-progressive-ply", action="store_true")
    parser.add_argument(
        "--semantic-tracker-overlap",
        dest="semantic_tracker_overlap",
        action="store_true",
        help="Run RAM++ -> DeepSeek in a worker thread overlapped with Tracker; SAM3 still joins before semantic frame_result is needed.",
    )
    parser.add_argument(
        "--no-semantic-tracker-overlap",
        dest="semantic_tracker_overlap",
        action="store_false",
    )
    parser.set_defaults(
        fg_enable_cabinet_aggregation=True,
        fg_cabinet_output=True,
        fg_obs_depth_boxplot_enable=True,
        fg_obs_dbscan_enable=True,
        fg_kf_geom_dbscan_enable=True,
        semantic_tracker_overlap=True,
    )

    args = parser.parse_args()
    disable_intermediate_outputs = bool(args.disable_intermediate_outputs) or bool(
        int(os.environ.get("FG_NO_IO_DUMP", "0") or "0")
    )
    if args.fps_profile_output:
        os.environ["FG_PROFILE_STAGES"] = "1"

    def _parse_label_csv(text: str) -> set[str]:
        return {part.strip().lower() for part in str(text or "").split(",") if part.strip()}

    fixed_semantic_atlas = None
    if args.fixed_semantic_atlas_inline:
        fixed_semantic_atlas = json.loads(args.fixed_semantic_atlas_inline)
    elif args.fixed_semantic_atlas_json:
        with open(args.fixed_semantic_atlas_json, "r", encoding="utf-8") as f:
            fixed_semantic_atlas = json.load(f)
    if fixed_semantic_atlas is not None:
        if args.enable_rampp or args.enable_deepseek:
            raise RuntimeError(
                "Fixed semantic atlas mode is mutually exclusive with --enable-rampp/--enable-deepseek."
            )

    rampp = None
    if args.enable_rampp:

        from mast3r_slam.semantic.rampp_runtime import RamppTagger

        rampp = RamppTagger(args.rampp_ckpt, device=args.rampp_device)

    deepseek = None
    scene_lock = None
    semantic_out_dir = None
    scene_history_path = None
    frames_out_dir = None
    scene_json_path = None
    atlas_json_path = None
    run_args_path = None
    if args.enable_deepseek or fixed_semantic_atlas is not None:

        if args.enable_deepseek and not args.enable_rampp:
            raise RuntimeError("DeepSeek requires RAM++ tags; enable --enable-rampp as well.")
        if args.enable_deepseek:
            if not args.deepseek_model:
                raise RuntimeError(
                    "DeepSeek is enabled but no model deployment was provided. "
                    "Set DEEPSEEK_MODEL or pass --deepseek-model."
                )
            from mast3r_slam.semantic.deepseek_runtime import DeepSeekRuntime
            from mast3r_slam.semantic.scene_lock import SceneLock


            deepseek = DeepSeekRuntime(
                api_key=None,
                base_url=args.deepseek_base_url,
                model=args.deepseek_model,
                timeout_sec=args.deepseek_timeout,
                max_retries=args.deepseek_retries,
                cache_dir=args.deepseek_cache_dir,
            )
            scene_lock = SceneLock(
                m=args.scene_lock_m,
                conf_thresh=args.scene_lock_thresh,
                switch_m=args.scene_switch_m,
                switch_conf_thresh=args.scene_switch_thresh,
                switch_cooldown_frames=args.scene_switch_cooldown,
            )

        if args.semantic_out_dir:
            semantic_out_dir = pathlib.Path(args.semantic_out_dir)
        else:
            semantic_out_dir = pathlib.Path("logs/semantic_runs") / f"{args.save_as}_{datetime_now}"

        semantic_out_dir.mkdir(parents=True, exist_ok=True)
        frames_out_dir = semantic_out_dir / "frames"
        scene_history_path = semantic_out_dir / "scene_history.jsonl"
        scene_json_path = semantic_out_dir / "scene.json"
        atlas_json_path = semantic_out_dir / "global_atlas.json"
        run_args_path = semantic_out_dir / "run_args.json"
        if not disable_intermediate_outputs:
            _write_json_atomic(run_args_path, vars(args))

    # Disable per-frame/debug dumps for FPS benchmarking.  Unlike the legacy
    # FG_NO_IO_DUMP path, this keeps semantic_out_dir so the final authoritative
    # online_functional_graph.json can still be flushed at shutdown.
    if disable_intermediate_outputs:
        frames_out_dir = None
        scene_history_path = None
        scene_json_path = None
        atlas_json_path = None
        print("[FPS] intermediate 2D/3D/debug outputs disabled; final JSON/PLY retained")

    sam3_runtime = None
    sam3_save_vis = args.sam3_save_vis and not disable_intermediate_outputs
    if args.enable_sam3:
        if not args.enable_deepseek and fixed_semantic_atlas is None:
            raise RuntimeError(
                "SAM3 requires DeepSeek outputs or --fixed-semantic-atlas-json/--fixed-semantic-atlas-inline."
            )
        if not args.sam3_ckpt or not args.sam3_bpe_path:
            raise RuntimeError("SAM3 enabled but checkpoint or BPE path is missing.")

        from mast3r_slam.semantic.sam3_two_stage_runtime import Sam3TwoStageRuntime

        force_u = set([x.strip() for x in args.sam3_force_u.split(",") if x.strip()])
        sam3_runtime = Sam3TwoStageRuntime(
            checkpoint_path=args.sam3_ckpt,
            bpe_path=args.sam3_bpe_path,
            device=args.sam3_device,
            confidence=args.sam3_confidence,
            thr_o=args.sam3_thr_o,
            thr_c=args.sam3_thr_c,
            thr_u=args.sam3_thr_u,
            overlap_iou=args.sam3_overlap_iou,
            u_cover_thr=args.sam3_u_cover_thr,
            u_cover_pad=args.sam3_u_cover_pad,
            u_parent_contain_thr=args.sam3_u_parent_contain_thr,
            force_u=force_u,
            drop_parent_without_u=not args.sam3_keep_parent_without_u,
            same_type_drop_multi_parent_links=args.sam3_same_type_drop_multi_parent_links,
            processor_resolution=args.sam3_processor_resolution,
        )


    load_config(args.config)
    if args.intrinsics_mode == "no-calib":
        config["use_calib"] = False
        args.calib = ""
    elif args.intrinsics_mode == "calib":
        config["use_calib"] = True
        if not args.calib:
            raise RuntimeError("--intrinsics-mode calib requires --calib")
    print(args.dataset)
    print(config)


    manager = _make_shared_manager(args)
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)
    functional_graph_viz_state = None
    if not args.no_viz:
        functional_graph_viz_state = manager.dict()
        functional_graph_viz_state["version"] = 0
        functional_graph_viz_state["snapshot"] = {
            "kf_idx": -1,
            "nodes": [],
            "edges": [],
            "summary": {"n_nodes": 0, "n_edges": 0},
        }


    dataset = load_dataset(args.dataset)
    temp_freeze_map_after_frame = int(getattr(args, "fg_temp_freeze_map_after_frame", -1) or -1)
    if bool(getattr(args, "fg_temp_localize_after_max_frames", False)) and int(args.max_frames) > 0:
        temp_freeze_map_after_frame = int(args.max_frames)
        print(
            "[TEMP] full-sequence localization enabled: "
            f"freeze reconstruction/semantic/functional graph from frame {temp_freeze_map_after_frame}"
        )
    else:
        dataset.truncate(args.max_frames)
    if temp_freeze_map_after_frame >= 0:
        os.environ["MAST3R_SLAM_VIZ_HIDE_CURRENT_AFTER_FRAME"] = str(temp_freeze_map_after_frame)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]


    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )


    # The old fixed 512-slot keyframe buffer over-allocates several GiB on
    # short FunGraph sequences.  Keep the historical upper bound, but size the
    # buffer to the loaded sequence so full-scene evaluation can coexist with
    # semantic model workers on 24GB GPUs.
    keyframe_buffer = min(512, max(32, len(dataset) + 8))
    keyframe_buffer_cap = os.environ.get("MAST3R_SLAM_KEYFRAME_BUFFER_CAP")
    if keyframe_buffer_cap:
        try:
            keyframe_buffer = min(keyframe_buffer, max(32, int(keyframe_buffer_cap)))
        except ValueError:
            print(f"[WARN] invalid MAST3R_SLAM_KEYFRAME_BUFFER_CAP={keyframe_buffer_cap!r}; ignoring")
    print(f"SharedKeyframes buffer={keyframe_buffer} for dataset_frames={len(dataset)}")
    keyframes = SharedKeyframes(manager, h, w, buffer=keyframe_buffer)
    states = SharedStates(manager, h, w)

    viz_thread_enabled = os.environ.get("MAST3R_SLAM_VIZ_THREAD", "0") == "1"


    if not args.no_viz:
        if viz_thread_enabled:
            if os.environ.get("MAST3R_SLAM_BACKEND_THREAD", "0") != "1":
                raise RuntimeError("MAST3R_SLAM_VIZ_THREAD=1 requires MAST3R_SLAM_BACKEND_THREAD=1")
            viz = threading.Thread(
                target=run_visualization,
                args=(config, states, keyframes, main2viz, viz2main, functional_graph_viz_state),
                name="mast3r-slam-viz",
            )
            print("[INFO] using visualization thread")
        else:
            viz = mp.Process(
                target=run_visualization,
                args=(config, states, keyframes, main2viz, viz2main, functional_graph_viz_state),
            )
        viz.start()


    model = load_mast3r(device=device)
    model.share_memory()


    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:

        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)


    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()


    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()


    # Configure node-place JSONL debug path before the backend process is
    # forked/spawned so it sees the path through the shared `config` dict.
    try:
        npr_cfg = config.setdefault("node_place_recognition", {}) if isinstance(config, dict) else None
        if npr_cfg is not None and disable_intermediate_outputs:
            npr_cfg["jsonl_debug_enabled"] = False
            npr_cfg["jsonl_path"] = ""
        if npr_cfg is not None and bool(npr_cfg.get("jsonl_debug_enabled", True)):
            if semantic_out_dir is not None:
                npr_cfg["jsonl_path"] = str(pathlib.Path(semantic_out_dir) / "node_place_debug.jsonl")
            else:
                fallback_dir = pathlib.Path("logs/node_place_debug")
                fallback_dir.mkdir(parents=True, exist_ok=True)
                save_tag = str(args.save_as).replace("/", "_") if getattr(args, "save_as", None) else "default"
                npr_cfg["jsonl_path"] = str(fallback_dir / f"{save_tag}_node_place.jsonl")
    except Exception:
        pass

    backend_thread_enabled = os.environ.get("MAST3R_SLAM_BACKEND_THREAD", "0") == "1"
    if backend_thread_enabled:
        backend = BackendThread(config, model, states, keyframes, K)
    else:
        backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    def _wait_backend(reason: str) -> None:
        """Wait for a backend task and fail fast if the backend terminates.

        In single-thread mode the main thread polls ``global_optimizer_tasks``
        or ``reloc_sem``. Without this watchdog, a backend failure would leave
        those states uncleared and make the main thread wait indefinitely.
        """
        while config["single_thread"]:
            with states.lock:
                if reason == "global":
                    if len(states.global_optimizer_tasks) == 0:
                        return
                elif reason == "reloc":
                    if states.reloc_sem.value == 0:
                        return
                else:
                    return
            backend_exc = getattr(backend, "exception", None)
            if backend_exc is not None:
                raise RuntimeError(
                    f"Backend thread died while main was waiting on '{reason}'."
                ) from backend_exc
            if not backend.is_alive():
                exitcode = backend.exitcode
                raise RuntimeError(
                    f"Backend process died (exitcode={exitcode}) while main was waiting on "
                    f"'{reason}'. Check stderr for traceback. Aborting to avoid deadlock."
                )
            time.sleep(0.01)

    deepseek_pool = ThreadPoolExecutor(max_workers=1)
    semantic_pool = deepseek_pool  # alias for the new overlap path; same single worker.
    from mast3r_slam.functional_graph import (
        FunctionalGraphPolicy,
        OnlineFunctionalState,
        export_functional_graph_overlay_ply,
    )
    from mast3r_slam.functional_graph.debug_assoc_vis import export_assoc_overlay_image
    from mast3r_slam.functional_graph.debug_baseline_io import BaselineDebugDumper
    from mast3r_slam.semantic.fsg_anchor_fuser import FSGAnchorFuser
    from mast3r_slam.semantic.semantic_pipeline import SemanticPipeline
    from mast3r_slam.functional_graph.ply_overlay import export_overlay_edge_sidecar
    from mast3r_slam.functional_graph.ply_overlay import (
        OverlayVizConfig,
        export_progressive_functional_graph_overlay_pred_world,
    )
    if args.enable_sam3 or (config.get("node_place_recognition", {}) or {}).get("enabled", False):
        _ensure_node_place_recognition_imports()

    graph_policy = FunctionalGraphPolicy(
        suppress_objects=_parse_label_csv(args.fg_suppress_objects),
        hint_only_objects=_parse_label_csv(args.fg_hint_only_objects),
        enable_cabinet_aggregation=bool(args.fg_enable_cabinet_aggregation),
        aggregate_output_objects={"cabinet"} if bool(args.fg_cabinet_output) else set(),
        aggregate_candidate_objects={"cabinet"},
        disable_fusion_labels={"door"} if bool(args.fg_temp_no_door_fusion) else set(),
        temp_disable_fusion_observations=_parse_label_csv(args.fg_temp_disable_fusion_observations),
        drop_lid_nodes=bool(args.fg_temp_drop_lid_nodes),
        drop_lid_when_overlaps_cap=bool(args.fg_temp_drop_overlap_lid_with_cap),
        cap_lid_overlap_iou_thr=float(args.fg_temp_cap_lid_overlap_iou),
        remote_edge_block_labels=_parse_label_csv(args.fg_temp_block_remote_labels),
        temp_final_local_edge_invariant=bool(args.fg_temp_final_local_edge_invariant),
        temp_final_drop_labels=_parse_label_csv(args.fg_temp_drop_final_labels),
        temp_drop_border_u_nodes=bool(args.fg_temp_drop_border_u_nodes),
        temp_keep_single_frame_cup_handle=bool(args.fg_temp_keep_single_frame_cup_handle),
        temp_limit_door_drawer_units=bool(args.fg_temp_limit_door_drawer_units),
        temp_merge_drawers_final=bool(args.fg_temp_merge_drawers_final),
        temp_bottle_single_cap_final=bool(args.fg_temp_bottle_single_cap_final),
        temp_bottle_cap_min_distance=float(args.fg_temp_bottle_cap_min_distance),
        temp_parent_single_child_per_label_final=bool(args.fg_temp_parent_single_child_per_label_final),
        temp_node_pos_point_centroid=bool(args.fg_temp_node_pos_point_centroid),
        temp_realtime_final_adjustments=bool(args.fg_temp_realtime_final_adjustments),
        temp_cabinet_iou_only_aggregation=bool(args.fg_temp_cabinet_iou_only_aggregation),
        temp_exclude_cabinet_aggregation_observations=_parse_label_csv(
            args.fg_temp_exclude_cabinet_aggregation_observations
        ),
        temp_stage1_min_projected_iou=float(args.fg_temp_stage1_min_proj_iou),
        cabinet_scene_prior_labels={"cabinet"},
        cabinet_seed_carriers={"door", "drawer"},
        cabinet_unit_labels={"handle", "knob"},
    )
    semantic = SemanticPipeline(
        args=args,
        rampp=rampp,
        deepseek=deepseek,
        scene_lock=scene_lock,
        sam3_runtime=sam3_runtime,
        frames_out_dir=frames_out_dir,
        scene_history_path=scene_history_path,
        scene_json_path=scene_json_path,
        atlas_json_path=atlas_json_path,
        sam3_save_vis=sam3_save_vis,
        graph_policy=graph_policy,
    )
    if fixed_semantic_atlas is not None:
        semantic.set_fixed_atlas(fixed_semantic_atlas, source=args.fixed_semantic_atlas_json or "inline")
    fsg_fuser = None
    if semantic_out_dir is not None and not graph_policy.hint_only_objects and not disable_intermediate_outputs:
        fsg_fuser = FSGAnchorFuser(semantic_out_dir / "fsg_anchor.json")

    online_state = None
    if sam3_runtime is not None:
        online_state = OnlineFunctionalState(
            output_path=(semantic_out_dir / "online_functional_graph.json") if semantic_out_dir is not None else None,
            graph_policy=graph_policy,
            obs_depth_boxplot_enable=args.fg_obs_depth_boxplot_enable,
            obs_depth_boxplot_iqr_scale=args.fg_obs_depth_boxplot_iqr_scale,
            obs_depth_boxplot_near_iqr_scale=args.fg_obs_depth_boxplot_near_iqr_scale,
            obs_depth_boxplot_min_keep_ratio=args.fg_obs_depth_boxplot_min_keep_ratio,
            obs_dbscan_enable=args.fg_obs_dbscan_enable,
            kf_geom_dbscan_enable=args.fg_kf_geom_dbscan_enable,
            kf_geom_dbscan_eps=args.fg_kf_geom_dbscan_eps,
            kf_geom_dbscan_min_samples=args.fg_kf_geom_dbscan_min_samples,
            kf_geom_dbscan_min_keep_ratio=args.fg_kf_geom_dbscan_min_keep_ratio,
            save_intermediate_snapshots=not disable_intermediate_outputs,
            enable_delta_log=not disable_intermediate_outputs,
            keep_border_no_link_observations=args.fg_keep_border_no_link_observations,
        )
    # ---- Graceful shutdown: SIGTERM/SIGINT -> flush the graph once.
    #      Without this, a `kill <pid>` (systemd stop, timeout, Ctrl-C)
    #      leaves the online graph JSON un-written since nothing else
    #      persists it mid-run.  The handler uses a one-shot latch so
    #      two signals in a row still let the process abort.
    if online_state is not None:
        import signal as _signal
        _shutdown_latch = {"fired": False, "last_kf": -1}
        def _flush_on_signal(signum, _frame):  # type: ignore[no-untyped-def]
            if _shutdown_latch["fired"]:
                # Second signal: restore default handler and re-raise.
                _signal.signal(signum, _signal.SIG_DFL)
                os.kill(os.getpid(), signum)
                return
            _shutdown_latch["fired"] = True
            print(f"[fg-shutdown] received signal {signum}, flushing graph...", flush=True)
            try:
                kf = _shutdown_latch["last_kf"]
                if kf < 0:
                    kf = 0
                online_state.flush_graph(kf)
            except Exception as _exc:
                print(f"[fg-shutdown] flush failed: {_exc!r}", flush=True)
            # Tell the main loop to exit cleanly.
            try:
                states.set_mode(Mode.TERMINATED)
            except Exception:
                pass
        try:
            _signal.signal(_signal.SIGINT, _flush_on_signal)
            _signal.signal(_signal.SIGTERM, _flush_on_signal)
        except (ValueError, OSError):
            # Not in main thread (e.g. unit tests); skip.
            pass
    baseline_dumper = None
    if args.debug_dump_baseline_dir and not disable_intermediate_outputs:
        baseline_dumper = BaselineDebugDumper(args.debug_dump_baseline_dir, graph_policy)
    assoc_debug_run_name = args.assoc_debug_run_name or args.save_as
    assoc_debug_root = pathlib.Path(args.assoc_debug_root) / assoc_debug_run_name
    assoc_vis_dir = assoc_debug_root / "vis_2d"
    assoc_ply_progress_dir = assoc_debug_root / "ply_progress"
    assoc_manifest_path = assoc_debug_root / "vis_manifest.json"
    assoc_manifest = {
        "run_name": assoc_debug_run_name,
        "outputs": [],
    }
    if args.assoc_export_2d_overlay:
        assoc_manifest["outputs"].append(
            {
                "path_template": str(assoc_vis_dir / "frame_XXXXXX_assoc_overlay.png"),
                "world_mode": "image_plane",
                "scene_source": "real_rgb_or_saved_sam3_vis",
                "node_source": "online_state_current_frame_assoc",
                "uses_transform": False,
            }
        )
    if args.assoc_export_progressive_ply:
        assoc_manifest["outputs"].append(
            {
                "path_template": str(assoc_ply_progress_dir / "assoc_overlay_frame_XXXXXX.ply"),
                "world_mode": "pred_world_progressive",
                "scene_source": "pred_world_reconstruction_from_keyframes",
                "node_source": "online_state_pred_world_nodes",
                "uses_transform": False,
            }
        )
    if assoc_manifest["outputs"]:
        _write_json_atomic(assoc_manifest_path, assoc_manifest)

    i = 0
    fps_timer = time.time()

    frames = []
    processed_timestamps = []
    run_wall_start = None
    run_wall_end = None

    # [FPS_PROF] per-stage timing accumulators (populated only when
    # FG_PROFILE_STAGES=1 to keep production output untouched).
    import os as _os_prof
    _profile_stages = _os_prof.environ.get("FG_PROFILE_STAGES", "0") == "1"
    _profile_verbose = _os_prof.environ.get("FG_PROFILE_VERBOSE", "0") == "1"
    _stage_totals: dict[str, float] = {}
    _stage_counts: dict[str, int] = {}

    def _stage_end(_name: str, _t0: float) -> None:
        if not _profile_stages:
            return
        _dt = time.perf_counter() - _t0
        _stage_totals[_name] = _stage_totals.get(_name, 0.0) + _dt
        _stage_counts[_name] = _stage_counts.get(_name, 0) + 1
        if _profile_verbose:
            print(f"[FPS_PROF][F{i}] {_name}: {_dt*1000.0:.1f} ms")

    def _stage_add(_name: str, _dt: float) -> None:
        # Record a duration measured outside the main thread (e.g., reported by
        # the semantic worker).  Mirrors ``_stage_end`` but lets the caller
        # supply the elapsed time directly.
        if not _profile_stages:
            return
        _stage_totals[_name] = _stage_totals.get(_name, 0.0) + float(_dt)
        _stage_counts[_name] = _stage_counts.get(_name, 0) + 1
        if _profile_verbose:
            print(f"[FPS_PROF][F{i}] {_name}: {float(_dt)*1000.0:.1f} ms")

    def _run_semantic_pre_sam3(frame_idx: int, img_for_semantic):
        """Run same-frame semantic pre-processing (RAM++ -> DeepSeek) in a
        worker thread.  SAM3 stays in the main thread because it needs
        frame.uimg and the GPU; this helper only covers the steps that can
        overlap with Tracker.  All inner timings are measured here so the
        main thread can fold them into the per-stage summary via
        ``_stage_add``.
        """
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        tags_local = None
        ds_out_local = None

        if semantic.rampp is not None:
            t0 = time.perf_counter()
            tags_local = semantic.run_rampp(frame_idx, img_for_semantic)
            timings["semantic_rampp"] = time.perf_counter() - t0

        if semantic.deepseek is not None:
            if tags_local is None:
                raise RuntimeError(
                    "DeepSeek enabled but RAM++ tags are missing for this frame."
                )
            t0 = time.perf_counter()
            ds_out_local = semantic.run_deepseek(frame_idx, tags_local)
            timings["semantic_deepseek"] = time.perf_counter() - t0

        timings["semantic_pre_total"] = time.perf_counter() - t_total
        return {
            "tags": tags_local,
            "ds_out": ds_out_local,
            "timings": timings,
        }

    def _run_deepseek_timed(frame_idx: int, tags_local):
        # Used by the legacy --no-semantic-tracker-overlap path so we still get
        # a real ``semantic_deepseek`` timing in the summary.
        t0 = time.perf_counter()
        out = semantic.run_deepseek(frame_idx, tags_local)
        return out, {"semantic_deepseek": time.perf_counter() - t0}

    def _semantic_scene_is_locked() -> bool:
        # The global atlas is created exactly when the scene becomes usable for
        # per-frame graph extraction; checking it avoids relying on SceneLock
        # internals and naturally follows any scene-switch atlas rebuild.
        return getattr(semantic, "global_atlas", None) is not None

    def _drain_latest_viz_msg(current_msg: WindowMsg) -> WindowMsg:
        # The GUI can enqueue several states while a frame is running.  Use the
        # newest state only, otherwise pause/unpause can lag behind stale
        # messages and appear to be ignored.
        latest = current_msg
        while True:
            msg_local = try_get_msg(viz2main)
            if msg_local is None:
                break
            latest = msg_local
        return latest

    while True:

        mode = states.get_mode()
        last_msg = _drain_latest_viz_msg(last_msg)
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break


        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()


        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break


        _t_frame_begin = time.perf_counter() if _profile_stages else 0.0
        _t0 = time.perf_counter() if _profile_stages else 0.0
        if run_wall_start is None:
            run_wall_start = time.perf_counter()
        timestamp, img = dataset[i]
        processed_timestamps.append(float(timestamp))
        if save_frames:
            frames.append(img)
        _stage_end("dataset_read", _t0)
        temp_map_frozen = bool(temp_freeze_map_after_frame >= 0 and i >= temp_freeze_map_after_frame)
        if temp_map_frozen and i == temp_freeze_map_after_frame:
            print(
                "[TEMP] freezing reconstruction/semantic/functional graph; "
                "subsequent frames update camera localization only"
            )

        # --- Semantic pre-processing scheduling -----------------------
        # Two flavors share the rest of the loop:
        #   * overlap path (default): RAM++ -> DeepSeek run in a single
        #     background worker concurrently with Tracker; we join before
        #     SAM3 since SAM3 consumes frame_result.
        #   * legacy path: RAM++ runs synchronously, then DeepSeek is
        #     forked while Tracker proceeds (kept for A/B comparison).
        scene_locked_before_semantic = _semantic_scene_is_locked()
        defer_semantic_until_keyframe = bool(
            args.semantic_keyframes_only_after_scene_lock and scene_locked_before_semantic
        )
        semantic_future = None
        deepseek_future = None
        tags = None
        ds_out = None
        if (
            not temp_map_frozen
            and args.semantic_tracker_overlap
            and args.enable_rampp
            and not defer_semantic_until_keyframe
        ):
            img_for_semantic = img.copy() if hasattr(img, "copy") else img
            semantic_future = semantic_pool.submit(
                _run_semantic_pre_sam3, i, img_for_semantic
            )
            print(f"[F{i}] semantic pre-SAM3 fork")
        elif not temp_map_frozen and args.enable_rampp and not defer_semantic_until_keyframe:
            # Legacy synchronous RAM++ path.
            _t0 = time.perf_counter() if _profile_stages else 0.0
            tags = semantic.run_rampp(i, img)
            _stage_end("rampp", _t0)
            print(f"[F{i}] RAM++ done")
            if deepseek is not None:
                if tags is None:
                    raise RuntimeError(
                        "DeepSeek enabled but RAM++ tags are missing for this frame."
                    )
                deepseek_future = semantic_pool.submit(_run_deepseek_timed, i, tags)
                print(f"[F{i}] DeepSeek fork")


        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )

        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)
        if K is not None:
            frame.K = K

        add_new_kf = False
        _t0_track = time.perf_counter() if _profile_stages else 0.0
        if mode == Mode.INIT:

            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)

            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            _wait_backend("global")

        elif mode == Mode.TRACKING:

            add_new_kf, match_info, try_reloc = tracker.track(
                frame,
                update_keyframe_pointmap=not temp_map_frozen,
            )
            if temp_map_frozen:
                add_new_kf = False
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)
            _stage_end("tracker_track", _t0_track)
            print(f"[F{i}] track done")

        elif mode == Mode.RELOC:

            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            _wait_backend("reloc")
            _stage_end("tracker_reloc", _t0_track)
            print(f"[F{i}] track done")

        else:
            raise Exception("Invalid mode")

        current_frame_is_keyframe = bool(add_new_kf or mode == Mode.INIT)
        frame_result = None
        graph_frame_result = None
        if semantic_future is not None:
            _t0 = time.perf_counter() if _profile_stages else 0.0
            sem_out = semantic_future.result()
            _stage_end("semantic_wait", _t0)
            for _name, _dt in sem_out.get("timings", {}).items():
                _stage_add(_name, _dt)
            tags = sem_out.get("tags")
            ds_out = sem_out.get("ds_out")
            if ds_out is not None:
                frame_result = ds_out.frame_result
                graph_frame_result = ds_out.graph_frame_result
            print(f"[F{i}] semantic pre-SAM3 join")
        elif deepseek_future is not None:
            _t0 = time.perf_counter() if _profile_stages else 0.0
            ds_out, _ds_timings = deepseek_future.result()
            _stage_end("deepseek_wait", _t0)
            for _name, _dt in _ds_timings.items():
                _stage_add(_name, _dt)
            frame_result = ds_out.frame_result
            graph_frame_result = ds_out.graph_frame_result
            print(f"[F{i}] DeepSeek join")
        elif (
            not temp_map_frozen
            and defer_semantic_until_keyframe
            and args.enable_rampp
            and current_frame_is_keyframe
        ):
            sem_out = _run_semantic_pre_sam3(i, img)
            for _name, _dt in sem_out.get("timings", {}).items():
                _stage_add(_name, _dt)
            tags = sem_out.get("tags")
            ds_out = sem_out.get("ds_out")
            if ds_out is not None:
                frame_result = ds_out.frame_result
                graph_frame_result = ds_out.graph_frame_result
            print(f"[F{i}] semantic keyframe-only pre-SAM3 done")
        elif defer_semantic_until_keyframe and _profile_verbose:
            print(f"[F{i}] semantic skipped after scene lock (non-keyframe)")
        elif not temp_map_frozen and fixed_semantic_atlas is not None:
            _t0 = time.perf_counter() if _profile_stages else 0.0
            ds_out = semantic.run_fixed_semantic(i)
            _stage_end("semantic_fixed", _t0)
            frame_result = ds_out.frame_result
            graph_frame_result = ds_out.graph_frame_result

        sam3_out = None
        if sam3_runtime is not None and frame_result is not None:
            _t0 = time.perf_counter() if _profile_stages else 0.0
            sam3_out = semantic.run_sam3(i, frame.uimg, frame_result, graph_frame_result=graph_frame_result)
            _stage_end("sam3", _t0)
            print(f"[F{i}] SAM3 done")

        fg_update_allowed = (not temp_map_frozen) and not (
            bool(args.functional_graph_keyframes_only_after_scene_lock)
            and _semantic_scene_is_locked()
            and not current_frame_is_keyframe
        )
        if online_state is not None and frame_result is not None and sam3_out is not None and fg_update_allowed:
            _t0 = time.perf_counter() if _profile_stages else 0.0
            online_state.update_frame(
                frame_idx=i,
                frame=frame,
                frame_result=frame_result,
                sam3_out=sam3_out,
                is_keyframe=current_frame_is_keyframe,
                keyframes=keyframes,
            )
            _stage_end("online_state_update", _t0)
            # Publish a current-frame node query signature for relocalization
            # rerank in the backend (best-effort; safe when disabled).
            try:
                _node_cfg = config.get("node_place_recognition", {}) or {}
                if _node_cfg.get("enabled", False):
                    _query_sig = build_signature_from_online_state(
                        online_state,
                        i,
                        kf_idx=None,
                        cfg=_node_cfg,
                        allow_unmatched_label_only=True,
                    )
                    # Merge in v2 functional-place fields.
                    try:
                        _v2_sig = build_functional_place_signature(
                            online_state, i, kf_idx=None, cfg=_node_cfg,
                        )
                        if _v2_sig:
                            _query_sig.update(_v2_sig)
                            # Preserve a v2-aware label_counts merge (v2 wins).
                            if _v2_sig.get("label_counts"):
                                _query_sig["label_counts"] = _v2_sig["label_counts"]
                    except Exception:
                        pass
                    states.set_current_node_query_signature(_query_sig)
            except Exception as _np_exc:
                print(f"[NODE_PLACE][query] ERROR: {type(_np_exc).__name__}: {_np_exc}")
            if args.assoc_export_2d_overlay:
                base_image_path = None
                if frames_out_dir is not None:
                    candidate = frames_out_dir / f"frame_{i:06d}_sam3_det.png"
                    if candidate.exists():
                        base_image_path = candidate
                overlay_path = assoc_vis_dir / f"frame_{i:06d}_assoc_overlay.png"
                export_assoc_overlay_image(
                    output_path=overlay_path,
                    det=sam3_out.det,
                    frame_assoc_debug=online_state.frame_assoc_debug.get(i),
                    frame_extract_debug=online_state.frame_extract_debug.get(i),
                    frame_uimg=frame.uimg,
                    base_image_path=base_image_path,
                )
        elif online_state is not None and frame_result is not None and sam3_out is not None and not fg_update_allowed:
            _stage_add("online_state_update_skipped_after_scene_lock", 0.0)
            if _profile_verbose:
                print(f"[F{i}] online functional graph update skipped after scene lock (non-keyframe)")

        new_kf_idx = None
        if add_new_kf:

            keyframes.append(frame)
            new_kf_idx = len(keyframes) - 1
            # Publish a keyframe node signature BEFORE queueing global
            # optimization so the backend's loop-closure rerank can use it.
            try:
                _node_cfg = config.get("node_place_recognition", {}) or {}
                if online_state is not None and _node_cfg.get("enabled", False):
                    _kf_sig = build_signature_from_online_state(
                        online_state,
                        i,
                        kf_idx=new_kf_idx,
                        cfg=_node_cfg,
                        allow_unmatched_label_only=False,
                    )
                    try:
                        _v2_sig = build_functional_place_signature(
                            online_state, i, kf_idx=new_kf_idx, cfg=_node_cfg,
                        )
                        if _v2_sig:
                            _kf_sig.update(_v2_sig)
                            if _v2_sig.get("label_counts"):
                                _kf_sig["label_counts"] = _v2_sig["label_counts"]
                    except Exception:
                        pass
                    states.set_kf_node_signature(new_kf_idx, _kf_sig)
            except Exception as _np_exc:
                print(f"[NODE_PLACE][kf_sig] ERROR: {type(_np_exc).__name__}: {_np_exc}")
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            _wait_backend("global")

        if mode == Mode.INIT:
            new_kf_idx = 0

        if online_state is not None and new_kf_idx is not None and frame_result is not None and sam3_out is not None:
            _t0 = time.perf_counter() if _profile_stages else 0.0
            online_state.commit_keyframe(
                kf_idx=new_kf_idx,
                frame=frame,
                keyframes=keyframes,
            )
            _stage_end("online_state_commit", _t0)
            if not args.no_viz:
                online_state.apply_realtime_temp_graph_adjustments(new_kf_idx)
            publish_functional_graph_viz_snapshot(
                functional_graph_viz_state,
                online_state,
                keyframes,
                kf_idx=new_kf_idx,
            )
            try:
                _shutdown_latch["last_kf"] = int(new_kf_idx)
            except NameError:
                pass

        if fsg_fuser is not None and new_kf_idx is not None and frame_result is not None and sam3_out is not None:
            fsg_fuser.update_keyframe(new_kf_idx, frame, frame_result, sam3_out)
        if baseline_dumper is not None and frame_result is not None and sam3_out is not None:
            vis2d_image_path = getattr(sam3_out, "debug_image_path", None)
            rgb_image_path = _write_baseline_rgb_frame_copy(baseline_dumper, i, frame)
            baseline_dumper.record_frame(
                frame_idx=i,
                frame=frame,
                timestamp=float(timestamp),
                frame_result=frame_result,
                sam3_out=sam3_out,
                is_keyframe=new_kf_idx is not None,
                keyframe_slot=new_kf_idx,
                frame_assoc_debug=online_state.frame_assoc_debug.get(i) if online_state is not None else None,
                frame_extract_debug=online_state.frame_extract_debug.get(i) if online_state is not None else None,
                vis2d_image_path=vis2d_image_path,
                rgb_image_path=rgb_image_path,
                debug_image_path=vis2d_image_path,
            )
        if (
            args.assoc_export_progressive_ply
            and online_state is not None
        ):
            _t0 = time.perf_counter() if _profile_stages else 0.0
            export_progressive_functional_graph_overlay_pred_world(
                output_dir=assoc_ply_progress_dir,
                frame_idx=i,
                online_state=online_state,
                keyframes=keyframes,
                viz_config=OverlayVizConfig(
                    scene_keep_ratio=args.fg_overlay_scene_keep_ratio,
                    clear_scene_near_graph=args.fg_overlay_clear_scene_near_graph,
                    clear_node_radius_scale=args.fg_overlay_clear_node_radius_scale,
                    clear_edge_radius_scale=args.fg_overlay_clear_edge_radius_scale,
                    remote_edges_tentative_style=args.fg_temp_remote_tentative_color,
                    export_mode="focus" if args.fg_overlay_export_focus else "balanced",
                ),
                c_conf_threshold=float(last_msg.C_conf_threshold),
            )
            _stage_end("progressive_ply", _t0)
        if _profile_stages:
            _stage_end("frame_total", _t_frame_begin)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    run_wall_end = time.perf_counter()
    processed_frame_count = int(i)
    run_wall_s = (
        float(run_wall_end - run_wall_start)
        if run_wall_start is not None and run_wall_end is not None
        else 0.0
    )
    functional_slam_fps = (
        float(processed_frame_count) / run_wall_s if run_wall_s > 0.0 else 0.0
    )

    # [FPS_PROF] per-stage summary
    if _profile_stages and _stage_totals:
        print("[FPS_PROF][SUMMARY] per-stage totals (stage: total_s  avg_ms  count)")
        for _n in sorted(_stage_totals, key=lambda k: -_stage_totals[k]):
            _tot = _stage_totals[_n]
            _cnt = max(1, _stage_counts.get(_n, 1))
            print(f"[FPS_PROF][SUMMARY]   {_n}: {_tot:.2f}s  {_tot*1000.0/_cnt:.1f}ms  n={_cnt}")


    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        # The final TUM trajectory is required for ATE/recalibration reports and
        # is not a per-frame debug artifact, so keep it even in FPS/no-dump mode.
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        if not disable_intermediate_outputs:
            eval.save_keyframes(
                save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
            )
    if online_state is not None:
        last_kf_idx = len(keyframes) - 1
        if last_kf_idx >= 0:
            online_state.flush_graph(last_kf_idx)
            publish_functional_graph_viz_snapshot(
                functional_graph_viz_state,
                online_state,
                keyframes,
                kf_idx=last_kf_idx,
            )
        else:
            online_state.save_snapshot()

        if dataset.save_results:
            sequence_prefix = infer_debug_dump_sequence_name(args.dataset).replace("/", "_")
            overlay_ply = save_dir / f"{sequence_prefix}_functional_graph_overlay.ply"
            balanced_cfg = OverlayVizConfig(
                scene_keep_ratio=args.fg_overlay_scene_keep_ratio,
                clear_scene_near_graph=args.fg_overlay_clear_scene_near_graph,
                clear_node_radius_scale=args.fg_overlay_clear_node_radius_scale,
                clear_edge_radius_scale=args.fg_overlay_clear_edge_radius_scale,
                remote_edges_tentative_style=args.fg_temp_remote_tentative_color,
                export_mode="balanced",
            )
            export_functional_graph_overlay_ply(
                save_dir / f"{seq_name}.ply",
                overlay_ply,
                online_state,
                keyframes,
                viz_config=balanced_cfg,
            )
            print(f"functional graph overlay ply: {overlay_ply}")

        if dataset.save_results and not disable_intermediate_outputs:
            try:
                sidecar_path = save_dir / "functional_graph_overlay_edges.json"
                export_overlay_edge_sidecar(
                    sidecar_path,
                    online_state,
                    keyframes,
                    viz_config=balanced_cfg,
                )
                print(f"functional graph overlay edge sidecar: {sidecar_path}")
            except Exception as e:  # pragma: no cover - never fail the pipeline
                print(f"[warn] overlay edge sidecar failed: {e}")

            # ---- Section G: dedicated debug-summary sidecars ----------
            try:
                import json as _json_g
                g_payloads = {
                    "node_observation_consolidation_summary.json": {
                        "node_consolidation_events_recent": list(getattr(online_state, "node_consolidation_events", []))[-256:],
                    },
                    "remote_2d_backlog_summary.json": {
                        "remote_2d_backlog_summary": dict(getattr(online_state, "remote_2d_backlog_summary", {})),
                        "remote_2d_backlog_events_recent": list(getattr(online_state, "remote_2d_backlog_events", []))[-256:],
                        "remote_2d_evidence_ledger_recent": {
                            k: list(v[-8:]) for k, v in list(getattr(online_state, "remote_2d_evidence_ledger", {}).items())
                        },
                    },
                    "remote_atlas_candidate_summary.json": {
                        "remote_atlas_summary": dict(getattr(online_state, "remote_atlas_summary", {})),
                        "remote_atlas_templates_recent": {
                            k: dict(v) for k, v in list(getattr(online_state, "remote_atlas_templates", {}).items())[-256:]
                        },
                        "remote_atlas_candidate_events_recent": list(getattr(online_state, "remote_atlas_candidates_recent", []))[-256:],
                    },
                    "remote_atlas_llava_summary.json": {
                        "remote_atlas_summary": dict(getattr(online_state, "remote_atlas_summary", {})),
                        "remote_atlas_llava_events_recent": list(getattr(online_state, "remote_atlas_llava_events", []))[-512:],
                    },
                }
                for fname, payload in g_payloads.items():
                    p = save_dir / fname
                    with open(p, "w", encoding="utf-8") as fh:
                        _json_g.dump(payload, fh, indent=2, default=str)
                    print(f"functional graph debug summary: {p}")
            except Exception as e:  # pragma: no cover - never fail the pipeline
                print(f"[warn] debug-summary sidecars failed: {e}")

            if args.fg_overlay_export_focus:
                focus_overlay_ply = save_dir / f"{sequence_prefix}_functional_graph_overlay_focus.ply"
                focus_cfg = OverlayVizConfig(
                    scene_keep_ratio=0.06,
                    clear_scene_near_graph=True,
                    clear_node_radius_scale=2.0,
                    clear_edge_radius_scale=1.8,
                    remote_edges_tentative_style=args.fg_temp_remote_tentative_color,
                    export_mode="focus",
                )
                export_functional_graph_overlay_ply(
                    save_dir / f"{seq_name}.ply",
                    focus_overlay_ply,
                    online_state,
                    keyframes,
                    viz_config=focus_cfg,
                )
                print(f"functional graph overlay focus ply: {focus_overlay_ply}")

    def _aggregate_profile_events(events: list[dict]) -> tuple[dict[str, float], dict[str, int]]:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for event in events:
            name = str(event.get("name") or "")
            if not name:
                continue
            totals[name] = totals.get(name, 0.0) + float(event.get("seconds", 0.0) or 0.0)
            counts[name] = counts.get(name, 0) + int(event.get("count", 1) or 1)
        return totals, counts

    def _sum_profile_names(totals: dict[str, float], names: list[str]) -> float:
        return float(sum(float(totals.get(name, 0.0) or 0.0) for name in names))

    fps_profile_payload = None
    if args.fps_profile_output:
        raw_totals = dict(_stage_totals)
        raw_counts = dict(_stage_counts)
        if online_state is not None:
            online_profile = online_state.export_runtime_profile()
            for name, value in online_profile.get("totals_s", {}).items():
                raw_totals[name] = raw_totals.get(name, 0.0) + float(value)
            for name, value in online_profile.get("counts", {}).items():
                raw_counts[name] = raw_counts.get(name, 0) + int(value)
        try:
            backend_totals, backend_counts = _aggregate_profile_events(states.get_runtime_profile_events())
        except Exception:
            backend_totals, backend_counts = {}, {}
        for name, value in backend_totals.items():
            raw_totals[name] = raw_totals.get(name, 0.0) + float(value)
        for name, value in backend_counts.items():
            raw_counts[name] = raw_counts.get(name, 0) + int(value)

        semantic_total = _sum_profile_names(
            raw_totals,
            ["semantic_rampp", "semantic_deepseek", "sam3"],
        )
        anchor_total = _sum_profile_names(
            raw_totals,
            [
                "commit_keyframe.bbox_and_candidate",
                "commit_keyframe.pre_stable_geom",
                "commit_keyframe.promote_anchor",
                "commit_keyframe.fuse_anchor",
                "commit_keyframe.aux_anchor",
                "commit_keyframe.reanchor",
                "commit_keyframe.plane_normal",
            ],
        )
        node_assoc_total = _sum_profile_names(raw_totals, ["update_frame.associate"])
        edge_assoc_total = _sum_profile_names(
            raw_totals,
            [
                "update_frame.local_post",
                "update_frame.local_edges",
                "update_frame.remote_post",
                "update_frame.remote_backlog",
                "update_frame.remote_edges",
                "update_frame.remote_atlas",
                "update_frame.cabinet",
                "update_frame.local_rehydrate",
                "update_frame.purge",
                "update_frame.shape",
                "commit_keyframe.commit_stable",
                "commit_keyframe.consolidation",
            ],
        )
        functional_loop_total = _sum_profile_names(
            raw_totals,
            ["backend.functional_loop_detection"],
        )

        def _module(total_s: float, source_names: list[str]) -> dict:
            denom = max(1, processed_frame_count)
            return {
                "total_s": float(total_s),
                "avg_ms_per_frame": float(total_s) * 1000.0 / denom,
                "source_event_count": int(sum(raw_counts.get(n, 0) for n in source_names)),
                "source_stages": list(source_names),
            }

        def _breakdown(source_names: list[str]) -> dict:
            return {name: _module(float(raw_totals.get(name, 0.0) or 0.0), [name]) for name in source_names}

        output_paths = {}
        if semantic_out_dir is not None:
            output_paths["final_graph_json"] = str(pathlib.Path(semantic_out_dir) / "online_functional_graph.json")
        if dataset.save_results:
            output_paths["final_reconstruction_ply"] = str(save_dir / f"{seq_name}.ply")
            if online_state is not None:
                sequence_prefix = infer_debug_dump_sequence_name(args.dataset).replace("/", "_")
                output_paths["final_functional_graph_overlay_ply"] = str(
                    save_dir / f"{sequence_prefix}_functional_graph_overlay.ply"
                )
        output_paths["fps_profile_json"] = str(pathlib.Path(args.fps_profile_output))

        fps_profile_payload = {
            "fps_label": (
                "Functional-SLAM-FPS"
                if (args.enable_rampp or args.enable_deepseek or args.enable_sam3)
                else "Functional-SLAM-FPS"
            ),
            "sequence": infer_debug_dump_sequence_name(args.dataset),
            "dataset": str(args.dataset),
            "config": str(args.config),
            "processed_frames": int(processed_frame_count),
            "wall_time_s": float(run_wall_s),
            "system_fps": float(functional_slam_fps),
            "functional_slam_fps": float(functional_slam_fps),
            "disable_intermediate_outputs": bool(disable_intermediate_outputs),
            "modules": {
                "semantic_information_deepseek_rampp_sam3": _module(
                    semantic_total,
                    ["semantic_rampp", "semantic_deepseek", "sam3"],
                ),
                "anchor": _module(
                    anchor_total,
                    [
                        "commit_keyframe.bbox_and_candidate",
                        "commit_keyframe.pre_stable_geom",
                        "commit_keyframe.promote_anchor",
                        "commit_keyframe.fuse_anchor",
                        "commit_keyframe.aux_anchor",
                        "commit_keyframe.reanchor",
                        "commit_keyframe.plane_normal",
                    ],
                ),
                "node_association": _module(node_assoc_total, ["update_frame.associate"]),
                "edge_association": _module(
                    edge_assoc_total,
                    [
                        "update_frame.local_post",
                        "update_frame.local_edges",
                        "update_frame.remote_post",
                        "update_frame.remote_backlog",
                        "update_frame.remote_edges",
                        "update_frame.remote_atlas",
                        "update_frame.cabinet",
                        "update_frame.local_rehydrate",
                        "update_frame.purge",
                        "update_frame.shape",
                        "commit_keyframe.commit_stable",
                        "commit_keyframe.consolidation",
                    ],
                ),
                "functional_graph_loop_detection": _module(
                    functional_loop_total,
                    ["backend.functional_loop_detection"],
                ),
            },
            "module_breakdown": {
                "node_association": _breakdown(
                    [
                        "update_frame.associate.projectability",
                        "update_frame.associate.projectability.predict_points_world",
                        "update_frame.associate.projectability.transform_project",
                        "update_frame.associate.projectability.cpu_sync",
                        "update_frame.associate.projectability.bbox_reduce",
                        "update_frame.associate.stage1_pair_score",
                        "update_frame.associate.stage1_rerank_assist",
                        "update_frame.associate.hungarian",
                        "update_frame.associate.stage2_candidate_select",
                        "update_frame.associate.stage2_pair_score",
                        "update_frame.associate.new_node_creation",
                        "update_frame.associate.track_update",
                        "update_frame.associate.debug_build",
                    ]
                ),
                "functional_graph_loop_detection": _breakdown(
                    [
                        "backend.functional_loop.signature_fetch",
                        "backend.functional_loop.deficit_check",
                        "backend.functional_loop.get_query_sig",
                        "backend.functional_loop.get_all_sigs",
                        "backend.functional_loop.functional_candidate_retrieve",
                        "backend.functional_loop.topology_candidate_retrieve",
                        "backend.functional_loop.ftopo_candidate_retrieve",
                        "backend.functional_loop.original_retrieval_factor_add",
                        "backend.functional_loop.supplement_pool_build",
                        "backend.functional_loop.supplement_audition_geometry",
                        "backend.functional_loop.supplement_score",
                        "backend.functional_loop.supplement_add_factor",
                        "backend.functional_loop.strengthen_score",
                        "backend.functional_loop.strengthen_support_total",
                        "backend.functional_loop.strengthen_add_factor",
                        "backend.functional_loop.debug_build",
                    ]
                ),
            },
            "profile_diagnostics": {
                "node_association": {
                    "profiled_frames": int(raw_counts.get("update_frame.associate.frame_count", 0)),
                    "observations_total": int(raw_counts.get("update_frame.associate.observation_count", 0)),
                    "candidate_tracks_total": int(raw_counts.get("update_frame.associate.candidate_track_count", 0)),
                    "stage1_pairs_total": int(raw_counts.get("update_frame.associate.stage1_pair_count", 0)),
                    "stage2_pairs_total": int(raw_counts.get("update_frame.associate.stage2_pair_count", 0)),
                },
                "functional_graph_loop_detection": {
                    "functional_candidates_total": int(raw_counts.get("backend.functional_loop.functional_candidate_count", 0)),
                    "topology_candidates_total": int(raw_counts.get("backend.functional_loop.topology_candidate_count", 0)),
                    "functionalized_topology_candidates_total": int(raw_counts.get("backend.functional_loop.ftopo_candidate_count", 0)),
                    "supplement_pool_total": int(raw_counts.get("backend.functional_loop.supplement_pool_count", 0)),
                    "supplement_retrieval_skipped_total": int(
                        raw_counts.get("backend.functional_loop.supplement_retrieval_skipped", 0)
                    ),
                    "original_retrieval_factor_candidates_total": int(
                        raw_counts.get("backend.functional_loop.original_retrieval_factor_candidate_count", 0)
                    ),
                    "strengthen_add_factor_calls_total": int(raw_counts.get("backend.functional_loop.strengthen_add_factor_count", 0)),
                },
            },
            "raw_stage_totals_s": raw_totals,
            "raw_stage_counts": raw_counts,
            "output_paths": output_paths,
        }
        _write_json_atomic(pathlib.Path(args.fps_profile_output), fps_profile_payload)
        print(f"[FPS] {fps_profile_payload['fps_label']}: {functional_slam_fps:.3f} ({processed_frame_count} frames / {run_wall_s:.3f}s)")
        for name, info in fps_profile_payload["modules"].items():
            print(f"[FPS] {name}: {info['avg_ms_per_frame']:.2f} ms/frame")
        print(f"[FPS] profile json: {args.fps_profile_output}")

    if baseline_dumper is not None:
        baseline_meta = baseline_dumper.finalize(
            sequence_name=args.debug_dump_sequence_name or infer_debug_dump_sequence_name(args.dataset),
            dataset_path=args.dataset,
            gt_file=(args.debug_dump_gt_file or None),
        )
        print(f"baseline pose-frozen dump: {args.debug_dump_baseline_dir}")
        print(f"baseline meta frames={baseline_meta.num_frames} keyframes={len(baseline_meta.keyframe_indices)}")

    if save_frames:

        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")

    if backend_thread_enabled:
        backend.join(timeout=5.0)
        if backend.is_alive():
            print("[WARN] backend thread did not exit within 5s; continuing shutdown")
    else:
        backend.join()
    if not args.no_viz:
        viz.join()
    deepseek_pool.shutdown(wait=True)
