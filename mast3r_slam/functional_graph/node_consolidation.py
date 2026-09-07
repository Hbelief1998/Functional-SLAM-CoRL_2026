"""Node consolidation helpers (Section A/H of node&&edge_requirement.md).

This module hosts utilities that *used* to live as hard-coded tables
inside ``online_state.py``.  Moving them here lets the production code
stay free of category-specific rules while still allowing optional
alias / synonym tables to be supplied via config or runtime policy.

Design contract (per requirement A):

* Default alias groups are *empty*.  ``label_semantic_similarity``
  therefore reduces to a strict equality check unless the caller
  explicitly passes ``alias_groups``.
* A small *legacy* alias table is kept for backward-compatibility with
  call-sites that still use the global ``label_semantic_similarity``
  helper from ``online_state``.  New code should pass an explicit
  ``alias_groups`` argument loaded via ``load_label_alias_groups``.
* ``load_label_alias_groups`` accepts a dict / mapping that may contain
  either ``alias_groups`` (a list of lists/sets of strings) or
  ``label_aliases`` (a mapping ``canonical -> list[synonyms]``).  Unknown
  shapes are ignored gracefully.
"""

from __future__ import annotations

from typing import Iterable, List, Optional


# ---------------------------------------------------------------------------
# Default (production) alias table
# ---------------------------------------------------------------------------
# IMPORTANT: per requirement A, the production default is *empty*.  Code that
# still wants the legacy table must opt in by reading ``LEGACY_ALIAS_GROUPS``
# explicitly (see ``label_semantic_similarity_with_legacy_default`` below).
DEFAULT_ALIAS_GROUPS: List[frozenset[str]] = []

# Kept for backward compatibility with callers / tests that referenced the
# previous online_state module-level constant.  Do not extend this table for
# new categories — register synonyms via config instead.
LEGACY_ALIAS_GROUPS: List[frozenset[str]] = [
    frozenset({"door", "cabinet door"}),
    frozenset({"knob", "dial"}),
    frozenset({"handle", "lever", "door handle", "cabinet door handle"}),
    frozenset({"drawer", "cabinet drawer"}),
    frozenset({"faucet", "tap"}),
    frozenset({"stove", "cooktop", "burner"}),
    frozenset({"switch", "light switch"}),
    frozenset({"cabinet", "cupboard"}),
]


def _coerce_groups(groups: Optional[Iterable]) -> List[frozenset[str]]:
    """Normalise an iterable of label-groups into ``list[frozenset[str]]``.

    Any non-iterable, empty, or string-only group is dropped silently.
    Each label is lower-cased and stripped before being added so that
    callers may freely supply mixed-case strings.
    """
    if not groups:
        return []
    out: List[frozenset[str]] = []
    for grp in groups:
        if isinstance(grp, str):
            continue
        try:
            members = [str(x).strip().lower() for x in grp if str(x).strip()]
        except Exception:
            continue
        if len(members) >= 2:
            out.append(frozenset(members))
    return out


def load_label_alias_groups(cfg_or_policy=None) -> List[frozenset[str]]:
    """Load alias groups from a config mapping / policy object.

    Supported shapes (any one is sufficient):

    1. ``cfg["alias_groups"]`` — list of lists/sets of strings.
    2. ``cfg["label_aliases"]`` — mapping ``canonical -> [synonyms]``.
    3. ``cfg["functional_graph"]["alias_groups"]`` — same as (1) but
       nested under a top-level ``functional_graph`` key (matches the
       project's YAML layout).
    4. Object-style policy with ``alias_groups`` / ``label_aliases``
       attributes.

    Returns an empty list when no usable shape is found.  Never raises.
    """
    if cfg_or_policy is None:
        return list(DEFAULT_ALIAS_GROUPS)
    cfg = cfg_or_policy
    candidates: list = []
    # Mapping access.
    if isinstance(cfg, dict):
        if cfg.get("alias_groups"):
            candidates = list(cfg["alias_groups"])
        elif "label_aliases" in cfg and isinstance(cfg["label_aliases"], dict):
            for canonical, syns in cfg["label_aliases"].items():
                grp = [str(canonical).strip().lower()]
                grp.extend(str(s).strip().lower() for s in (syns or []))
                candidates.append(grp)
        elif "functional_graph" in cfg and isinstance(cfg["functional_graph"], dict):
            return load_label_alias_groups(cfg["functional_graph"])
    else:
        # Object access.
        for attr in ("alias_groups", "label_alias_groups"):
            val = getattr(cfg, attr, None)
            if val:
                candidates = list(val)
                break
        if not candidates:
            la = getattr(cfg, "label_aliases", None)
            if isinstance(la, dict):
                for canonical, syns in la.items():
                    grp = [str(canonical).strip().lower()]
                    grp.extend(str(s).strip().lower() for s in (syns or []))
                    candidates.append(grp)
    return _coerce_groups(candidates)


def label_similarity(
    label_a: str,
    label_b: str,
    alias_groups: Optional[List[frozenset[str]]] = None,
) -> float:
    """Section A2: deterministic similarity in ``{0.0, 0.9, 1.0}``.

    * 1.0 — exact match after ``strip().lower()``.
    * 0.9 — both labels appear in the *same* alias group.
    * 0.0 — otherwise.

    ``alias_groups=None`` falls back to ``DEFAULT_ALIAS_GROUPS`` (empty in
    production).  Callers are encouraged to pass an explicit list so the
    behaviour is reproducible and free of hidden category rules.
    """
    a = (label_a or "").strip().lower()
    b = (label_b or "").strip().lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    groups = alias_groups if alias_groups is not None else DEFAULT_ALIAS_GROUPS
    for grp in groups:
        if a in grp and b in grp:
            return 0.9
    return 0.0
