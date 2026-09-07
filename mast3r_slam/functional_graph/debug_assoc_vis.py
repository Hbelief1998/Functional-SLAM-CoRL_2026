from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


_STATUS_COLORS = {
    "stage1": (0, 255, 0),
    "stage2": (0, 255, 255),
    "new_birth": (0, 0, 255),
    "extract_fail": (180, 120, 255),
    "extract_skip": (180, 180, 180),
    "unknown": (150, 150, 150),
}


def _load_image(*, base_image_path: Optional[str | Path], frame_uimg) -> np.ndarray:
    if base_image_path is not None and Path(base_image_path).exists():
        image = cv2.imread(str(base_image_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image

    if isinstance(frame_uimg, torch.Tensor):
        arr = (frame_uimg.detach().clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    else:
        arr = np.asarray(frame_uimg)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    raise ValueError("frame_uimg must be an RGB image.")


def _center_text_anchor(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_shape: tuple[int, int, int],
    text: str,
    *,
    font_scale: float,
    thickness: int,
) -> tuple[int, int]:
    h, w = image_shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))
    x = cx - tw // 2
    y = cy + (th - baseline) // 2
    x = max(0, min(w - tw - 1, x))
    y = max(th + 1, min(h - baseline - 1, y))
    return x, y


def _assoc_status(det_idx: int, *, assoc_by_det: dict[int, dict], extract_by_det: dict[int, dict]) -> tuple[str, Optional[str]]:
    assoc = assoc_by_det.get(det_idx)
    if assoc is not None:
        assigned_stage = assoc.get("assigned_stage")
        if assigned_stage == "stage1":
            return "stage1", assoc.get("matched_node_id")
        if assigned_stage == "stage2":
            return "stage2", assoc.get("matched_node_id")
        if assigned_stage == "new_birth":
            return "new_birth", assoc.get("matched_node_id")
    extract = extract_by_det.get(det_idx)
    if extract is not None:
        status = str(extract.get("status") or "")
        if status in {"low_points", "after_depth_boxplot_low_points", "after_dbscan_low_points", "missing_mask_and_box"}:
            return "extract_fail", None
        if status in {"skipped_policy"}:
            return "extract_skip", None
    return "unknown", None


def export_assoc_overlay_image(
    output_path: str | Path,
    *,
    det,
    frame_assoc_debug: Optional[dict],
    frame_extract_debug: Optional[list[dict]],
    frame_uimg,
    base_image_path: Optional[str | Path] = None,
) -> Path:
    image = _load_image(base_image_path=base_image_path, frame_uimg=frame_uimg)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assoc_by_det = {
        int(item["det_idx"]): dict(item)
        for item in ((frame_assoc_debug or {}).get("observation_debug", []) or [])
        if item.get("det_idx") is not None
    }
    extract_by_det = {
        int(item["det_idx"]): dict(item)
        for item in (frame_extract_debug or [])
        if item.get("det_idx") is not None
    }

    labels = list(getattr(det, "labels", []) or [])
    boxes = getattr(det, "boxes", None)
    if boxes is not None:
        for det_idx, _label in enumerate(labels):
            box = boxes[det_idx]
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            status_key, node_id = _assoc_status(det_idx, assoc_by_det=assoc_by_det, extract_by_det=extract_by_det)
            color = _STATUS_COLORS.get(status_key, _STATUS_COLORS["unknown"])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            if node_id:
                font_scale = 0.3
                thickness = 1
                tx, ty = _center_text_anchor(
                    x1,
                    y1,
                    x2,
                    y2,
                    image.shape,
                    str(node_id),
                    font_scale=font_scale,
                    thickness=thickness,
                )
                cv2.putText(
                    image,
                    str(node_id),
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(node_id),
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

    cv2.imwrite(str(output_path), image)
    return output_path
