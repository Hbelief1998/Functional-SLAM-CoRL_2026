from __future__ import annotations



from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple


def _norm(s: str | None) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip().lower()


def _unit_name(unit: object) -> str:
    if isinstance(unit, dict):
        return _norm(unit.get("unit"))
    if isinstance(unit, str):
        return _norm(unit)
    return ""


def _unit_dict(unit: object, is_direct: bool) -> Optional[dict]:
    name = _unit_name(unit)
    if not name:
        return None
    if isinstance(unit, dict):
        out = {"unit": name}
        if is_direct:
            out["ou_relation"] = (unit.get("ou_relation") or "").strip()
        else:
            out["cu_relation"] = (unit.get("cu_relation") or "").strip()
            out["ou_relation"] = (unit.get("ou_relation") or "").strip()
        return out
    if is_direct:
        return {"unit": name, "ou_relation": ""}
    return {"unit": name, "cu_relation": "", "ou_relation": ""}


def _pick_relation(old: str, new: str) -> str:
    old = (old or "").strip()
    new = (new or "").strip()
    if not old:
        return new
    if not new:
        return old
    if old == "part of" and new != "part of":
        return new
    if new == "part of" and old != "part of":
        return old
    return new if len(new) > len(old) else old


def atlas_object_set(global_atlas: dict) -> Set[str]:

    objs = global_atlas.get("objects", []) if global_atlas else []
    return {_norm(o.get("object")) for o in objs if o.get("object")}


def atlas_object_carrier_set(global_atlas: dict) -> Set[str]:
    objs = global_atlas.get("objects", []) if global_atlas else []
    out: Set[str] = set()
    for obj in objs:
        on = _norm(obj.get("object"))
        if on:
            out.add(on)
        for c in obj.get("functional_carriers", []) or []:
            cn = _norm(c.get("carrier"))
            if cn:
                out.add(cn)
    return out


def remote_endpoints_from_pairs(remote_pairs: List[dict]) -> Set[str]:
    """Extract normalized endpoint labels from remote-relation candidates."""
    out: Set[str] = set()
    for rel in remote_pairs or []:
        a = _norm(rel.get("from_object"))
        b = _norm(rel.get("to_object"))
        if a:
            out.add(a)
        if b:
            out.add(b)
    return out


def atlas_lookup_uc(global_atlas: dict, object_name: str) -> Optional[dict]:

    if not global_atlas:
        return None
    target = _norm(object_name)
    for obj in global_atlas.get("objects", []):
        if _norm(obj.get("object")) == target:
            return {
                "object": target,
                "functional_carriers": obj.get("functional_carriers", []) or [],
                "direct_interactive_units": obj.get("direct_interactive_units", []) or [],
            }
    return None


def _merge_carriers(dst: Dict[str, dict], carriers: List[dict]) -> None:

    for c in carriers or []:
        name = _norm(c.get("carrier"))
        if not name:
            continue
        oc_relation = _pick_relation(c.get("oc_relation") or "", "part of")
        if name not in dst:
            dst[name] = {"carrier": name, "oc_relation": oc_relation, "units": {}}
        else:
            dst[name]["oc_relation"] = _pick_relation(dst[name].get("oc_relation", ""), oc_relation)

        unit_map = dst[name]["units"]
        for u in c.get("interactive_units", []) or []:
            unit_dict = _unit_dict(u, is_direct=False)
            if not unit_dict:
                continue
            unit_name = unit_dict["unit"]
            if unit_name not in unit_map:
                unit_map[unit_name] = unit_dict
            else:
                unit_map[unit_name]["cu_relation"] = _pick_relation(
                    unit_map[unit_name].get("cu_relation", ""), unit_dict.get("cu_relation", "")
                )
                unit_map[unit_name]["ou_relation"] = _pick_relation(
                    unit_map[unit_name].get("ou_relation", ""), unit_dict.get("ou_relation", "")
                )


def _carriers_to_list(carrier_map: Dict[str, dict]) -> List[dict]:

    out: List[dict] = []
    for name, payload in sorted(carrier_map.items()):
        units = [payload["units"][u] for u in sorted(payload["units"].keys())]
        out.append(
            {
                "carrier": name,
                "oc_relation": payload.get("oc_relation", "") or "part of",
                "interactive_units": units,
            }
        )
    return out


def _merge_object(base: dict, incoming: dict) -> dict:

    direct_map: Dict[str, dict] = {}
    for raw_unit in (base.get("direct_interactive_units", []) or []) + (incoming.get("direct_interactive_units", []) or []):
        unit_dict = _unit_dict(raw_unit, is_direct=True)
        if not unit_dict:
            continue
        unit_name = unit_dict["unit"]
        if unit_name not in direct_map:
            direct_map[unit_name] = unit_dict
        else:
            direct_map[unit_name]["ou_relation"] = _pick_relation(
                direct_map[unit_name].get("ou_relation", ""), unit_dict.get("ou_relation", "")
            )

    carrier_map: Dict[str, dict] = {}
    _merge_carriers(carrier_map, base.get("functional_carriers", []) or [])
    _merge_carriers(carrier_map, incoming.get("functional_carriers", []) or [])

    return {
        "object": base.get("object") or incoming.get("object"),
        "functional_carriers": _carriers_to_list(carrier_map),
        "direct_interactive_units": [direct_map[u] for u in sorted(direct_map.keys())],
    }


def _merge_present(entries: List[dict]) -> List[dict]:

    merged: Dict[str, dict] = {}
    for entry in entries:
        name = entry.get("object")
        if not name:
            continue
        if name not in merged:
            merged[name] = entry
        else:
            merged[name] = _merge_object(merged[name], entry)
    return list(merged.values())


def _normalize_roles(present: List[dict]) -> List[dict]:
    # If a label appears with multiple roles, keep only its lowest-level role (O > C > U).
    ROLE_VAL = {"O": 2, "C": 1, "U": 0}
    min_role: Dict[str, int] = {}

    # Pass 1: compute minimal role per label
    for entry in present:
        obj = _norm(entry.get("object"))
        if obj:
            min_role[obj] = min(min_role.get(obj, ROLE_VAL["O"]), ROLE_VAL["O"])
        for c in entry.get("functional_carriers", []) or []:
            name = _norm(c.get("carrier"))
            if name:
                min_role[name] = min(min_role.get(name, ROLE_VAL["C"]), ROLE_VAL["C"])
            for u in c.get("interactive_units", []) or []:
                u_norm = _unit_name(u)
                if u_norm:
                    min_role[u_norm] = min(min_role.get(u_norm, ROLE_VAL["U"]), ROLE_VAL["U"])
        for u in entry.get("direct_interactive_units", []) or []:
            u_norm = _unit_name(u)
            if u_norm:
                min_role[u_norm] = min(min_role.get(u_norm, ROLE_VAL["U"]), ROLE_VAL["U"])

    # Pass 2: rebuild present keeping only entries matching their minimal role
    normalized: List[dict] = []
    for entry in present:
        obj = _norm(entry.get("object"))
        if not obj:
            continue
        if min_role.get(obj, ROLE_VAL["O"]) != ROLE_VAL["O"]:
            # This label plays a lower role elsewhere; drop object entry
            continue

        carriers_filtered = []
        for c in entry.get("functional_carriers", []) or []:
            name = _norm(c.get("carrier"))
            if not name or min_role.get(name, ROLE_VAL["C"]) != ROLE_VAL["C"]:
                continue
            units_filtered = []
            for u in c.get("interactive_units", []) or []:
                u_name = _unit_name(u)
                if not u_name or min_role.get(u_name, ROLE_VAL["U"]) != ROLE_VAL["U"]:
                    continue
                unit_dict = _unit_dict(u, is_direct=False)
                if unit_dict:
                    units_filtered.append(unit_dict)
            carriers_filtered.append(
                {
                    "carrier": name,
                    "oc_relation": c.get("oc_relation") or "part of",
                    "interactive_units": units_filtered,
                }
            )

        direct_units_filtered = []
        for u in entry.get("direct_interactive_units", []) or []:
            u_name = _unit_name(u)
            if not u_name or min_role.get(u_name, ROLE_VAL["U"]) != ROLE_VAL["U"]:
                continue
            unit_dict = _unit_dict(u, is_direct=True)
            if unit_dict:
                direct_units_filtered.append(unit_dict)

        normalized.append(
            {
                "object": obj,
                "functional_carriers": carriers_filtered,
                "direct_interactive_units": direct_units_filtered,
            }
        )

    return normalized


def _merge_relations(rel_lists: List[List[dict]]) -> List[dict]:
    # Merge remote relation candidates, dedup by directed (from,to).
    # Preserve 'relation' field; prefer non-empty relation when duplicates occur.
    rel_map: Dict[Tuple[str, str], str] = {}
    order: List[Tuple[str, str]] = []

    for lst in rel_lists:
        for rel in lst or []:
            a = _norm(rel.get("from_object"))
            b = _norm(rel.get("to_object"))
            if not a or not b:
                continue
            key = (a, b)
            rel_text = (rel.get("relation") or "").strip()

            if key not in rel_map:
                rel_map[key] = rel_text
                order.append(key)
            else:
                if not rel_map[key] and rel_text:
                    rel_map[key] = rel_text

    out: List[dict] = []
    for a, b in order:
        out.append({"from_object": a, "to_object": b, "relation": rel_map[(a, b)]})
    return out


def build_frame_result_final(global_atlas: dict | None, rampp_tags_en: List[str], incremental: dict | None) -> dict:
    """
    Frame-level semantic output.

    Mode (desired): global_atlas ALL + unknown_tags (observed tags not covered by atlas)

    - known_tags: all objects in global_atlas (full list, not per-frame intersection)
    - unknown_tags: observed tags that are not in atlas
    - present: ALL atlas UC entries + incremental UC entries (merged/deduped)
    - remote_relation_candidates: merged from incremental + atlas (deduped)
    - observed_tags: tags detected in current frame (for debugging/logging)
    """
    atlas_objs = atlas_object_set(global_atlas) if global_atlas else set()

    # Observed tags in this frame (keep it for debugging)
    tags_set = {_norm(t) for t in (rampp_tags_en or []) if t}
    observed_tags = sorted(tags_set)

    # Case-1 change:
    # known_tags should represent the FULL atlas object list (not just per-frame known tags)
    known_tags = sorted(atlas_objs)

    # unknown_tags are the observed tags not covered by the atlas
    unknown_tags = sorted(tags_set - atlas_objs)

    # Case-1 change:
    # present should include ALL atlas object UC entries (not only those observed in current frame)
    present_entries: List[dict] = []
    if global_atlas:
        for obj_name in known_tags:
            entry = atlas_lookup_uc(global_atlas, obj_name)
            if entry:
                present_entries.append(entry)

    # Append incremental UC outputs (usually only for unknown tags, but keep generic)
    if incremental:
        present_entries.extend(incremental.get("objects", []) or [])

    # Merge/deduplicate objects
    present = _merge_present(present_entries)
    present = _normalize_roles(present)

    # Merge/deduplicate remote relations (atlas + incremental)
    remote_relation_candidates = _merge_relations([
        incremental.get("remote_relation_candidates", []) if incremental else [],
        global_atlas.get("remote_relation_candidates", []) if global_atlas else [],
    ])

    present_obj_set = {_norm(e.get("object")) for e in present if _norm(e.get("object"))}
    remote_relation_candidates = [
        r
        for r in remote_relation_candidates
        if _norm(r.get("from_object")) in present_obj_set and _norm(r.get("to_object")) in present_obj_set
    ]

    # --- build debug info (KEEP BACKWARD COMPAT) ---
    debug = {

        "observed_tags": observed_tags,


        "known_tags": known_tags,


        "unknown_tags": unknown_tags,

        "num_atlas_objects": len(known_tags),
        "num_observed_tags": len(observed_tags),
        "num_unknown_tags": len(unknown_tags),
    }


    result = {
        "present": present,
        "remote_relation_candidates": remote_relation_candidates,
        "debug": debug,
    }


    result["known_tags"] = known_tags
    result["unknown_tags"] = unknown_tags
    result["observed_tags"] = observed_tags

    return result


def build_graph_frame_result(
    frame_result: dict,
    *,
    suppress_objects: set[str],
    hint_only_objects: set[str],
) -> dict:
    """Build a graph-facing frame_result.

    This helper keeps the raw semantic output untouched while producing a trimmed
    view used by online graph logic and remote filtering.
    """

    src = deepcopy(frame_result or {})
    blocked = set(suppress_objects) | set(hint_only_objects)

    present = src.get("present", []) or []
    src["present"] = [entry for entry in present if _norm(entry.get("object")) not in blocked]

    remote = src.get("remote_relation_candidates", []) or []
    src["remote_relation_candidates"] = [
        rel
        for rel in remote
        if _norm(rel.get("from_object")) not in blocked and _norm(rel.get("to_object")) not in blocked
    ]

    known_tags = src.get("known_tags")
    if isinstance(known_tags, list):
        src["known_tags"] = [tag for tag in known_tags if _norm(tag) not in blocked]
    observed_tags = src.get("observed_tags")
    if isinstance(observed_tags, list):
        src["observed_tags"] = [tag for tag in observed_tags if _norm(tag) not in blocked]
    unknown_tags = src.get("unknown_tags")
    if isinstance(unknown_tags, list):
        src["unknown_tags"] = [tag for tag in unknown_tags if _norm(tag) not in blocked]

    debug = src.get("debug")
    if isinstance(debug, dict):
        for key in ("known_tags", "unknown_tags", "observed_tags"):
            if isinstance(debug.get(key), list):
                debug[key] = [tag for tag in debug[key] if _norm(tag) not in blocked]
        if isinstance(debug.get("num_atlas_objects"), int):
            debug["num_atlas_objects"] = len(debug.get("known_tags", []))
        if isinstance(debug.get("num_observed_tags"), int):
            debug["num_observed_tags"] = len(debug.get("observed_tags", []))
        if isinstance(debug.get("num_unknown_tags"), int):
            debug["num_unknown_tags"] = len(debug.get("unknown_tags", []))

    return src
