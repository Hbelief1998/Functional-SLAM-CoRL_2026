from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class PersistentEdge:
    src_node_id: str
    dst_node_id: str
    edge_type: str
    relation_text: str = ""
    committed_kf: int = -1
    support_count: int = 0
    evidence_score: float = 0.0
    status: str = "committed"
    first_seen_frame: int = -1
    last_seen_frame: int = -1
    last_update_source: str = ""
    margin: float = 0.0
    latest_margin: float = 0.0
    recent_support_count: int = 0
    switch_count: int = 0
    retention_policy: str = "ttl"

    def to_dict(self) -> dict:
        return {
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "edge_type": self.edge_type,
            "relation_text": self.relation_text,
            "committed_kf": self.committed_kf,
            "support_count": self.support_count,
            "evidence_score": self.evidence_score,
            "status": self.status,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "last_update_source": self.last_update_source,
            "margin": self.margin,
            "latest_margin": self.latest_margin,
            "recent_support_count": self.recent_support_count,
            "switch_count": self.switch_count,
            "retention_policy": self.retention_policy,
        }


@dataclass
class HierarchySnapshot:
    parent_u: Dict[str, str] = field(default_factory=dict)
    parent_c: Dict[str, str] = field(default_factory=dict)
    chains_uco: list[Tuple[str, str, str]] = field(default_factory=list)
    direct_uo: list[Tuple[str, str]] = field(default_factory=list)
    cabinet_groups: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "parent_u": dict(self.parent_u),
            "parent_c": dict(self.parent_c),
            "chains_uco": [list(chain) for chain in self.chains_uco],
            "direct_uo": [list(pair) for pair in self.direct_uo],
            "cabinet_groups": dict(self.cabinet_groups),
        }


class PersistentFunctionalGraph:
    def __init__(self) -> None:
        self.local_edges: Dict[Tuple[str, str, str], PersistentEdge] = {}
        self.remote_edges: Dict[str, PersistentEdge] = {}
        self.hierarchy = HierarchySnapshot()

    @staticmethod
    def _status_rank(status: str) -> int:
        return 1 if str(status or "").strip().lower() == "committed" else 0

    @classmethod
    def _edge_strength_tuple(cls, edge: PersistentEdge) -> tuple[float, ...]:
        return (
            float(cls._status_rank(edge.status)),
            float(edge.recent_support_count),
            float(edge.support_count),
            float(edge.margin),
            float(edge.latest_margin),
            float(edge.evidence_score),
        )

    @classmethod
    def _should_replace_tentative(cls, existing: PersistentEdge, incoming: PersistentEdge) -> bool:
        if existing.src_node_id == incoming.src_node_id and existing.dst_node_id == incoming.dst_node_id and existing.edge_type == incoming.edge_type:
            return True
        return cls._edge_strength_tuple(incoming) > cls._edge_strength_tuple(existing)

    @staticmethod
    def _merge_edge(existing: PersistentEdge, incoming: PersistentEdge) -> PersistentEdge:
        existing.relation_text = incoming.relation_text or existing.relation_text
        existing.committed_kf = max(int(existing.committed_kf), int(incoming.committed_kf))
        existing.support_count = max(int(existing.support_count), int(incoming.support_count))
        existing.evidence_score = max(float(existing.evidence_score), float(incoming.evidence_score))
        existing.first_seen_frame = (
            min(existing.first_seen_frame, incoming.first_seen_frame)
            if existing.first_seen_frame >= 0 and incoming.first_seen_frame >= 0
            else max(existing.first_seen_frame, incoming.first_seen_frame)
        )
        existing.last_seen_frame = max(int(existing.last_seen_frame), int(incoming.last_seen_frame))
        existing.last_update_source = incoming.last_update_source or existing.last_update_source
        existing.margin = max(float(existing.margin), float(incoming.margin))
        existing.latest_margin = float(incoming.latest_margin)
        existing.recent_support_count = max(int(existing.recent_support_count), int(incoming.recent_support_count))
        existing.switch_count = int(incoming.switch_count)
        # Retention policy upgrade: "until_contradicted" wins over "ttl";
        # committed edges always retain their effective semantics regardless
        # of the incoming policy.
        existing_policy = str(existing.retention_policy or "ttl")
        incoming_policy = str(incoming.retention_policy or "ttl")
        if existing_policy != "until_contradicted" and incoming_policy == "until_contradicted":
            existing.retention_policy = "until_contradicted"
        elif existing_policy == "until_contradicted" and incoming_policy == "ttl":
            # Keep stronger retention.
            pass
        else:
            existing.retention_policy = incoming_policy or existing_policy
        return existing

    def _find_local_edge_key_for_child(self, dst_node_id: str) -> Tuple[str, str, str] | None:
        for key, edge in self.local_edges.items():
            if edge.dst_node_id == dst_node_id:
                return key
        return None

    def edge_status_counts(self) -> dict:
        counts = {
            "local_tentative": 0,
            "local_committed": 0,
            "remote_tentative": 0,
            "remote_committed": 0,
        }
        for edge in self.local_edges.values():
            counts[f"local_{'committed' if self._status_rank(edge.status) > 0 else 'tentative'}"] += 1
        for edge in self.remote_edges.values():
            counts[f"remote_{'committed' if self._status_rank(edge.status) > 0 else 'tentative'}"] += 1
        counts["total_edges"] = sum(counts.values())
        return counts

    def upsert_local_edge(
        self,
        src_node_id: str,
        dst_node_id: str,
        edge_type: str,
        *,
        relation_text: str,
        committed_kf: int,
        support_count: int,
        evidence_score: float,
        status: str = "committed",
        first_seen_frame: int = -1,
        last_seen_frame: int = -1,
        last_update_source: str = "",
        margin: float = 0.0,
        latest_margin: float = 0.0,
        recent_support_count: int = 0,
        switch_count: int = 0,
        retention_policy: str = "ttl",
    ) -> PersistentEdge:
        new_edge = PersistentEdge(
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            edge_type=edge_type,
            relation_text=relation_text,
            committed_kf=committed_kf,
            support_count=support_count,
            evidence_score=evidence_score,
            status=status,
            first_seen_frame=first_seen_frame,
            last_seen_frame=last_seen_frame,
            last_update_source=last_update_source,
            margin=margin,
            latest_margin=latest_margin,
            recent_support_count=recent_support_count,
            switch_count=switch_count,
            retention_policy=str(retention_policy or "ttl"),
        )
        existing_key = self._find_local_edge_key_for_child(dst_node_id)
        if existing_key is None:
            self.local_edges[(src_node_id, dst_node_id, edge_type)] = new_edge
            return new_edge

        existing = self.local_edges[existing_key]
        existing_rank = self._status_rank(existing.status)
        incoming_rank = self._status_rank(new_edge.status)

        if existing_rank > incoming_rank:
            if (
                existing.src_node_id == new_edge.src_node_id
                and existing.dst_node_id == new_edge.dst_node_id
                and existing.edge_type == new_edge.edge_type
            ):
                return self._merge_edge(existing, new_edge)
            return existing

        if incoming_rank > existing_rank or self._should_replace_tentative(existing, new_edge):
            # Same-identity tentative-vs-tentative: merge so retention
            # policy upgrades (e.g. until_contradicted) are preserved.
            if (
                existing_rank == incoming_rank
                and existing.src_node_id == new_edge.src_node_id
                and existing.dst_node_id == new_edge.dst_node_id
                and existing.edge_type == new_edge.edge_type
            ):
                return self._merge_edge(existing, new_edge)
            self.local_edges.pop(existing_key, None)
            self.local_edges[(src_node_id, dst_node_id, edge_type)] = new_edge
            return new_edge

        return existing

    def upsert_remote_edge(
        self,
        relation_key: str,
        src_node_id: str,
        dst_node_id: str,
        relation_text: str,
        *,
        committed_kf: int,
        support_count: int,
        evidence_score: float,
        status: str = "committed",
        first_seen_frame: int = -1,
        last_seen_frame: int = -1,
        last_update_source: str = "",
        margin: float = 0.0,
        latest_margin: float = 0.0,
        recent_support_count: int = 0,
        switch_count: int = 0,
        retention_policy: str = "ttl",
    ) -> PersistentEdge:
        new_edge = PersistentEdge(
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            edge_type="remote",
            relation_text=relation_text,
            committed_kf=committed_kf,
            support_count=support_count,
            evidence_score=evidence_score,
            status=status,
            first_seen_frame=first_seen_frame,
            last_seen_frame=last_seen_frame,
            last_update_source=last_update_source,
            margin=margin,
            latest_margin=latest_margin,
            recent_support_count=recent_support_count,
            switch_count=switch_count,
            retention_policy=str(retention_policy or "ttl"),
        )
        existing = self.remote_edges.get(relation_key)
        if existing is None:
            self.remote_edges[relation_key] = new_edge
            return new_edge

        existing_rank = self._status_rank(existing.status)
        incoming_rank = self._status_rank(new_edge.status)
        if existing_rank > incoming_rank:
            if existing.src_node_id == new_edge.src_node_id and existing.dst_node_id == new_edge.dst_node_id:
                return self._merge_edge(existing, new_edge)
            return existing

        if incoming_rank > existing_rank or self._should_replace_tentative(existing, new_edge):
            if (
                existing_rank == incoming_rank
                and existing.src_node_id == new_edge.src_node_id
                and existing.dst_node_id == new_edge.dst_node_id
            ):
                return self._merge_edge(existing, new_edge)
            self.remote_edges[relation_key] = new_edge
            return new_edge
        return existing

    def prune_stale_tentative_edges(
        self,
        frame_idx: int,
        ttl_frames: int,
        *,
        known_node_ids: "set[str] | None" = None,
    ) -> None:
        cutoff = int(frame_idx) - int(ttl_frames)

        def _keep_local(edge: PersistentEdge) -> bool:
            # Drop edges whose endpoints no longer exist.
            if known_node_ids is not None:
                if edge.src_node_id not in known_node_ids or edge.dst_node_id not in known_node_ids:
                    return False
            if self._status_rank(edge.status) > 0:
                return True
            policy = str(edge.retention_policy or "ttl").lower()
            if policy == "until_contradicted":
                return True
            return int(edge.last_seen_frame) >= cutoff

        def _keep_remote(edge: PersistentEdge) -> bool:
            if known_node_ids is not None:
                if edge.src_node_id not in known_node_ids or edge.dst_node_id not in known_node_ids:
                    return False
            if self._status_rank(edge.status) > 0:
                return True
            policy = str(edge.retention_policy or "ttl").lower()
            if policy == "until_contradicted":
                return True
            return int(edge.last_seen_frame) >= cutoff

        self.local_edges = {key: edge for key, edge in self.local_edges.items() if _keep_local(edge)}
        self.remote_edges = {key: edge for key, edge in self.remote_edges.items() if _keep_remote(edge)}

    def update_hierarchy(
        self,
        *,
        parent_u: Dict[str, str],
        parent_c: Dict[str, str],
        chains_uco: list[Tuple[str, str, str]],
        direct_uo: list[Tuple[str, str]],
        cabinet_groups: Dict[str, dict] | None = None,
    ) -> None:
        self.hierarchy = HierarchySnapshot(
            parent_u=dict(parent_u),
            parent_c=dict(parent_c),
            chains_uco=list(chains_uco),
            direct_uo=list(direct_uo),
            cabinet_groups=dict(cabinet_groups or {}),
        )

    def to_dict(self) -> dict:
        return {
            "local_edges": [edge.to_dict() for edge in self.local_edges.values()],
            "remote_edges": [edge.to_dict() for edge in self.remote_edges.values()],
            "hierarchy": self.hierarchy.to_dict(),
            "edge_status_counts": self.edge_status_counts(),
        }
