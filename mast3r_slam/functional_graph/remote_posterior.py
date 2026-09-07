from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class RemotePairPosterior:
    relation_key: str
    relation_text: str
    src_label: str
    dst_label: str
    candidate_pairs: Dict[Tuple[str, str], float] = field(default_factory=dict)
    support_frames: Dict[Tuple[str, str], list[int]] = field(default_factory=dict)
    top1_pair: Optional[Tuple[str, str]] = None
    top2_pair: Optional[Tuple[str, str]] = None
    margin: float = 0.0
    ambiguity_counter: int = 0
    stable_pair: Optional[Tuple[str, str]] = None
    best_view_frames: list[int] = field(default_factory=list)
    recent_top1_history: list[Optional[Tuple[str, str]]] = field(default_factory=list)
    llava_requested: bool = False

    def support_count(self, pair: Optional[Tuple[str, str]]) -> int:
        if pair is None:
            return 0
        return len(self.support_frames.get(pair, []))

    def update(
        self,
        frame_idx: int,
        pair_scores: Dict[Tuple[str, str], float],
        *,
        decay: float,
        min_support_frames: int,
        stable_margin: float,
        history_size: int = 6,
    ) -> None:
        for pair in list(self.candidate_pairs.keys()):
            self.candidate_pairs[pair] *= decay
            if self.candidate_pairs[pair] < 1e-6:
                self.candidate_pairs.pop(pair, None)

        for pair, score in pair_scores.items():
            self.candidate_pairs[pair] = self.candidate_pairs.get(pair, 0.0) + float(score)
            frames = self.support_frames.setdefault(pair, [])
            if not frames or frames[-1] != frame_idx:
                frames.append(frame_idx)
        if pair_scores and (not self.best_view_frames or self.best_view_frames[-1] != frame_idx):
            self.best_view_frames.append(frame_idx)
            self.best_view_frames = self.best_view_frames[-5:]

        ranked = sorted(self.candidate_pairs.items(), key=lambda item: item[1], reverse=True)
        self.top1_pair = ranked[0][0] if ranked else None
        self.top2_pair = ranked[1][0] if len(ranked) > 1 else None
        top1_score = ranked[0][1] if ranked else 0.0
        top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
        self.margin = float(top1_score - top2_score)

        self.recent_top1_history.append(self.top1_pair)
        if len(self.recent_top1_history) > history_size:
            self.recent_top1_history = self.recent_top1_history[-history_size:]

        if self.top1_pair is not None and self.top2_pair is not None and self.margin < stable_margin:
            self.ambiguity_counter += 1
        else:
            self.ambiguity_counter = 0

        if self._is_stable(min_support_frames=min_support_frames, stable_margin=stable_margin):
            if self.stable_pair != self.top1_pair:
                self.stable_pair = self.top1_pair
                self.llava_requested = False

    def _is_stable(self, *, min_support_frames: int, stable_margin: float) -> bool:
        if self.top1_pair is None:
            return False
        support = self.support_count(self.top1_pair)
        if support < min_support_frames:
            return False
        if self.margin < stable_margin:
            return False
        return self.switch_count() <= 2

    def switch_count(self, *, window: Optional[int] = None) -> int:
        history = [pair for pair in self.recent_top1_history if pair is not None]
        if window is not None and window > 0:
            history = history[-window:]
        if len(history) <= 1:
            return 0
        return sum(1 for prev, cur in zip(history[:-1], history[1:]) if prev != cur)

    def is_unresolved(self, *, margin_threshold: float, unresolved_frames: int) -> bool:
        return (
            self.stable_pair is None
            and self.top1_pair is not None
            and self.top2_pair is not None
            and self.margin <= margin_threshold
            and self.ambiguity_counter >= unresolved_frames
        )

    def to_dict(self) -> dict:
        return {
            "relation_key": self.relation_key,
            "relation_text": self.relation_text,
            "src_label": self.src_label,
            "dst_label": self.dst_label,
            "candidate_pairs": {f"{src}->{dst}": score for (src, dst), score in self.candidate_pairs.items()},
            "support_frames": {f"{src}->{dst}": list(frames) for (src, dst), frames in self.support_frames.items()},
            "top1_pair": list(self.top1_pair) if self.top1_pair is not None else None,
            "top2_pair": list(self.top2_pair) if self.top2_pair is not None else None,
            "margin": self.margin,
            "ambiguity_counter": self.ambiguity_counter,
            "stable_pair": list(self.stable_pair) if self.stable_pair is not None else None,
            "best_view_frames": list(self.best_view_frames),
        }
