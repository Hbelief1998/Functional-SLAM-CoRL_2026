from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LocalParentPosterior:
    child_node_id: str
    child_role: str
    candidate_parent_scores: Dict[str, float] = field(default_factory=dict)
    candidate_support_frames: Dict[str, list[int]] = field(default_factory=dict)
    top1_parent_id: Optional[str] = None
    top2_parent_id: Optional[str] = None
    margin: float = 0.0
    latest_evidence_margin: float = 0.0
    ambiguity_counter: int = 0
    stable_parent_id: Optional[str] = None
    stable_since_kf: Optional[int] = None
    stable_parent_source: str = "preferred"
    owner_mode: str = "prefer_object"
    preferred_parent_ids: set[str] = field(default_factory=set)
    fallback_parent_ids: set[str] = field(default_factory=set)
    has_viable_preferred_parent: bool = False
    recent_top1_history: list[Optional[str]] = field(default_factory=list)
    recent_frame_history: list[int] = field(default_factory=list)
    llava_requested: bool = False

    def drop_candidates(self, parent_ids: set[str]) -> None:
        for parent_id in parent_ids:
            self.candidate_parent_scores.pop(parent_id, None)
            self.candidate_support_frames.pop(parent_id, None)

    def update(
        self,
        frame_idx: int,
        evidence_by_parent: Dict[str, float],
        *,
        decay: float,
        min_support_frames: int,
        stable_margin: float,
        history_size: int = 6,
        max_switches: int = 1,
        recent_consistency_frames: int = 3,
    ) -> None:
        ranked_now = sorted(evidence_by_parent.items(), key=lambda item: item[1], reverse=True)
        now_top1 = ranked_now[0][1] if ranked_now else 0.0
        now_top2 = ranked_now[1][1] if len(ranked_now) > 1 else 0.0
        self.latest_evidence_margin = float(now_top1 - now_top2)

        for parent_id in list(self.candidate_parent_scores.keys()):
            self.candidate_parent_scores[parent_id] *= decay
            if self.candidate_parent_scores[parent_id] < 1e-6:
                self.candidate_parent_scores.pop(parent_id, None)

        for parent_id, score in evidence_by_parent.items():
            self.candidate_parent_scores[parent_id] = self.candidate_parent_scores.get(parent_id, 0.0) + float(score)
            frames = self.candidate_support_frames.setdefault(parent_id, [])
            if not frames or frames[-1] != frame_idx:
                frames.append(frame_idx)

        ranked = sorted(self.candidate_parent_scores.items(), key=lambda item: item[1], reverse=True)
        self.top1_parent_id = ranked[0][0] if ranked else None
        self.top2_parent_id = ranked[1][0] if len(ranked) > 1 else None
        top1_score = ranked[0][1] if ranked else 0.0
        top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
        self.margin = float(top1_score - top2_score)

        self.recent_top1_history.append(self.top1_parent_id)
        if len(self.recent_top1_history) > history_size:
            self.recent_top1_history = self.recent_top1_history[-history_size:]
        self.recent_frame_history.append(frame_idx)
        if len(self.recent_frame_history) > history_size:
            self.recent_frame_history = self.recent_frame_history[-history_size:]

        unstable_now = not self._is_current_top1_stable(
            min_support_frames=min_support_frames,
            stable_margin=stable_margin,
            max_switches=max_switches,
            recent_consistency_frames=recent_consistency_frames,
        )
        if self.top1_parent_id is not None and self.top2_parent_id is not None and unstable_now:
            self.ambiguity_counter += 1
        else:
            self.ambiguity_counter = 0

        if not unstable_now and self.top1_parent_id is not None:
            if self.stable_parent_id != self.top1_parent_id:
                self.stable_parent_id = self.top1_parent_id
                self.stable_since_kf = None
                self.llava_requested = False
            if self.top1_parent_id in self.fallback_parent_ids:
                self.stable_parent_source = "fallback"
            else:
                self.stable_parent_source = "preferred"

    def support_count(self, parent_id: Optional[str]) -> int:
        if parent_id is None:
            return 0
        return len(self.candidate_support_frames.get(parent_id, []))

    def recent_support_count(self, parent_id: Optional[str], *, window: int) -> int:
        if parent_id is None:
            return 0
        frames = self.candidate_support_frames.get(parent_id, [])
        if not frames:
            return 0
        if not self.recent_frame_history:
            return min(len(frames), window)
        cutoff = self.recent_frame_history[-1] - max(1, window) + 1
        return sum(1 for frame_idx in frames if frame_idx >= cutoff)

    def recent_switch_count(self, *, window: Optional[int] = None) -> int:
        history = [node_id for node_id in self.recent_top1_history if node_id is not None]
        if window is not None and window > 0:
            history = history[-window:]
        if len(history) <= 1:
            return 0
        return sum(1 for prev, cur in zip(history[:-1], history[1:]) if prev != cur)

    def recent_top1_consistent(self, *, window: int) -> bool:
        if window <= 1:
            return self.top1_parent_id is not None
        history = self.recent_top1_history[-window:]
        if len(history) < window:
            return False
        if any(node_id is None for node_id in history):
            return False
        return len(set(history)) == 1

    def _is_current_top1_stable(
        self,
        *,
        min_support_frames: int,
        stable_margin: float,
        max_switches: int,
        recent_consistency_frames: int,
    ) -> bool:
        if self.top1_parent_id is None:
            return False
        if self.support_count(self.top1_parent_id) < min_support_frames:
            return False
        if self.recent_support_count(self.top1_parent_id, window=max(min_support_frames, recent_consistency_frames)) < recent_consistency_frames:
            return False
        if self.margin < stable_margin:
            return False
        if self.top2_parent_id is not None:
            per_step_margin_thr = stable_margin / max(1, min_support_frames)
            if self.latest_evidence_margin < per_step_margin_thr:
                return False
        if self.recent_switch_count(window=max(min_support_frames + 1, recent_consistency_frames + 1)) > max_switches:
            return False
        return self.recent_top1_consistent(window=recent_consistency_frames)

    def is_unresolved(self, *, margin_threshold: float, unresolved_frames: int) -> bool:
        effective_margin = self.latest_evidence_margin if self.top2_parent_id is not None else self.margin
        return (
            self.stable_parent_id is None
            and self.top1_parent_id is not None
            and self.top2_parent_id is not None
            and effective_margin <= margin_threshold
            and self.ambiguity_counter >= unresolved_frames
        )

    def is_commit_ready(
        self,
        *,
        min_support_frames: int,
        stable_margin: float,
        max_switches: int,
        recent_consistency_frames: int,
        unresolved_margin: float,
        unresolved_frames: int,
    ) -> bool:
        if self.stable_parent_id is None or self.stable_parent_id != self.top1_parent_id:
            return False
        if self.is_unresolved(margin_threshold=unresolved_margin, unresolved_frames=unresolved_frames):
            return False
        return self._is_current_top1_stable(
            min_support_frames=min_support_frames,
            stable_margin=stable_margin,
            max_switches=max_switches,
            recent_consistency_frames=recent_consistency_frames,
        )

    def top_support_summary(self, *, limit: int = 2) -> list[dict]:
        ranked = sorted(self.candidate_parent_scores.items(), key=lambda item: item[1], reverse=True)
        summary = []
        for parent_id, score in ranked[:limit]:
            summary.append(
                {
                    "parent_id": parent_id,
                    "score": float(score),
                    "support_count": self.support_count(parent_id),
                }
            )
        return summary

    def mark_committed(self, kf_idx: int) -> None:
        if self.stable_parent_id is not None and self.stable_since_kf is None:
            self.stable_since_kf = kf_idx

    def to_dict(self) -> dict:
        return {
            "child_node_id": self.child_node_id,
            "child_role": self.child_role,
            "candidate_parent_scores": dict(self.candidate_parent_scores),
            "candidate_support_frames": {key: list(value) for key, value in self.candidate_support_frames.items()},
            "top1_parent_id": self.top1_parent_id,
            "top2_parent_id": self.top2_parent_id,
            "margin": self.margin,
            "latest_evidence_margin": self.latest_evidence_margin,
            "ambiguity_counter": self.ambiguity_counter,
            "stable_parent_id": self.stable_parent_id,
            "stable_since_kf": self.stable_since_kf,
            "stable_parent_source": self.stable_parent_source,
            "owner_mode": self.owner_mode,
            "preferred_parent_ids": sorted(self.preferred_parent_ids),
            "fallback_parent_ids": sorted(self.fallback_parent_ids),
            "has_viable_preferred_parent": self.has_viable_preferred_parent,
            "recent_top1_history": list(self.recent_top1_history),
            "recent_frame_history": list(self.recent_frame_history),
        }
