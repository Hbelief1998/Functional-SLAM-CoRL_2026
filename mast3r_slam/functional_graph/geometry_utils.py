from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def frame_hw(frame) -> Tuple[int, int]:
    if hasattr(frame, "img_shape") and frame.img_shape is not None:
        vals = frame.img_shape.reshape(-1).tolist()
        if len(vals) >= 2:
            return int(vals[0]), int(vals[1])
    if hasattr(frame, "uimg") and frame.uimg is not None:
        return int(frame.uimg.shape[0]), int(frame.uimg.shape[1])
    raise ValueError("unable to infer frame resolution")


def reshape_pointmap(pointmap: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if pointmap.dim() == 3:
        if pointmap.shape[-1] == 3:
            return pointmap
        if pointmap.shape[0] == 3:
            return pointmap.permute(1, 2, 0)
    if pointmap.dim() == 2 and pointmap.shape[-1] == 3:
        return pointmap.reshape(height, width, 3)
    raise ValueError(f"unsupported pointmap shape: {tuple(pointmap.shape)}")


def reshape_conf(conf_map: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if conf_map.dim() == 3 and conf_map.shape[-1] == 1:
        return conf_map[..., 0]
    if conf_map.dim() == 2:
        if conf_map.shape[0] == height and conf_map.shape[1] == width:
            return conf_map
        if conf_map.shape[1] == 1:
            return conf_map.reshape(height, width)
    raise ValueError(f"unsupported confidence shape: {tuple(conf_map.shape)}")


def resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    mask_t = mask.float()
    if mask_t.dim() == 2:
        mask_t = mask_t[None, None, ...]
    elif mask_t.dim() == 3:
        if mask_t.shape[0] == 1:
            mask_t = mask_t[None, ...]
        else:
            mask_t = mask_t[:, None, ...]
    elif mask_t.dim() != 4:
        raise ValueError(f"unsupported mask shape: {tuple(mask.shape)}")
    mask_rs = F.interpolate(mask_t, size=(height, width), mode="nearest")
    return mask_rs[0, 0] > 0.5


def box_to_mask(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    mask = torch.zeros((height, width), dtype=torch.bool, device=box.device)
    if x2 >= x1 and y2 >= y1:
        mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def bbox_from_points(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return points.min(dim=0).values, points.max(dim=0).values


def bbox_size(bbox: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    bmin, bmax = bbox
    return torch.clamp(bmax - bmin, min=0.0)


def bbox_diag(bbox: Tuple[torch.Tensor, torch.Tensor]) -> float:
    return float(torch.linalg.norm(bbox_size(bbox)).item())


def transform_points_world(frame, points_frame: torch.Tensor) -> torch.Tensor:
    pose = getattr(frame, "T_WC", None)
    if pose is None:
        return points_frame
    if hasattr(pose, "act"):
        return pose.act(points_frame)
    if torch.is_tensor(pose) and pose.shape[-2:] == (4, 4):
        ones = torch.ones((points_frame.shape[0], 1), device=points_frame.device, dtype=points_frame.dtype)
        homo = torch.cat([points_frame, ones], dim=-1)
        return (homo @ pose.transpose(-1, -2))[..., :3]
    return points_frame


def transform_points_frame(frame, points_world: torch.Tensor) -> torch.Tensor:
    pose = getattr(frame, "T_WC", None)
    if pose is None:
        return points_world
    if hasattr(pose, "inv"):
        return pose.inv().act(points_world)
    if torch.is_tensor(pose) and pose.shape[-2:] == (4, 4):
        pose_inv = torch.linalg.inv(pose)
        ones = torch.ones((points_world.shape[0], 1), device=points_world.device, dtype=points_world.dtype)
        homo = torch.cat([points_world, ones], dim=-1)
        return (homo @ pose_inv.transpose(-1, -2))[..., :3]
    return points_world


def frame_intrinsics(frame) -> Optional[torch.Tensor]:
    K = getattr(frame, "K", None)
    if K is None:
        return None
    if torch.is_tensor(K):
        return K.reshape(3, 3)
    return torch.tensor(K, dtype=torch.float32).reshape(3, 3)


def project_points_to_image(frame, points_world: torch.Tensor) -> Optional[torch.Tensor]:
    if points_world.numel() == 0:
        return None
    points_frame = transform_points_frame(frame, points_world)
    valid = torch.isfinite(points_frame).all(dim=-1) & (points_frame[:, 2] > 1e-6)
    if int(valid.sum().item()) == 0:
        return None
    points_frame = points_frame[valid]
    z = points_frame[:, 2]

    K = frame_intrinsics(frame)
    if K is None:
        h, w = frame_hw(frame)
        device = points_frame.device
        dtype = points_frame.dtype
        focal = float(max(h, w)) * 0.5
        K = torch.tensor(
            [[focal, 0.0, float(w) * 0.5], [0.0, focal, float(h) * 0.5], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
    else:
        K = K.to(device=points_frame.device, dtype=points_frame.dtype)

    x = points_frame[:, 0]
    y = points_frame[:, 1]
    u = K[0, 0] * (x / z) + K[0, 2]
    v = K[1, 1] * (y / z) + K[1, 2]
    return torch.stack([u, v], dim=-1)


def project_box_xyxy(frame, points_world: torch.Tensor) -> Optional[list[float]]:
    uv = project_points_to_image(frame, points_world)
    if uv is None or uv.numel() == 0:
        return None
    h, w = frame_hw(frame)
    x = uv[:, 0].clamp(0.0, float(w - 1))
    y = uv[:, 1].clamp(0.0, float(h - 1))
    if x.numel() == 0 or y.numel() == 0:
        return None
    return [float(x.min().item()), float(y.min().item()), float(x.max().item()), float(y.max().item())]


def box_iou_xyxy(box_a: Optional[list[float]], box_b: Optional[list[float]]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


def box_overlap_ratio_xyxy(box_a: Optional[list[float]], box_b: Optional[list[float]]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = max(1e-9, min(area_a, area_b))
    return float(inter / smaller)


def box_overlap_over_obs_area(projected_box: Optional[list[float]], obs_box: Optional[list[float]]) -> float:
    if projected_box is None or obs_box is None:
        return 0.0
    px1, py1, px2, py2 = [float(v) for v in projected_box]
    ox1, oy1, ox2, oy2 = [float(v) for v in obs_box]
    inter_x1 = max(px1, ox1)
    inter_y1 = max(py1, oy1)
    inter_x2 = min(px2, ox2)
    inter_y2 = min(py2, oy2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    obs_area = max(1e-9, max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1))
    return float(inter / obs_area)


def box_touches_image_border(box_xyxy: Optional[list[float]], h: int, w: int, margin_px: int = 2) -> bool:
    if box_xyxy is None:
        return False
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    margin = float(max(0, margin_px))
    return bool(x1 <= margin or y1 <= margin or x2 >= float(w - 1) - margin or y2 >= float(h - 1) - margin)


def box_center_distance_xyxy(box_a: Optional[list[float]], box_b: Optional[list[float]]) -> float:
    if box_a is None or box_b is None:
        return 1.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    acx, acy = 0.5 * (ax1 + ax2), 0.5 * (ay1 + ay2)
    bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
    norm = max(1.0, ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1)
    return float((((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / norm)


def voxel_downsample_points(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    if points.numel() == 0:
        return points
    if voxel_size <= 0.0:
        return points

    # Keep voxel dedup on CPU. torch.unique(..., dim=0) on large CUDA tensors has
    # occasionally hit illegal-address faults in long SLAM runs.
    pts_cpu = points.detach().cpu().numpy()
    coords_np = np.floor(pts_cpu / float(voxel_size)).astype(np.int64)
    unique_coords, inverse = np.unique(coords_np, axis=0, return_inverse=True)
    if unique_coords.shape[0] == points.shape[0]:
        return points

    sums = np.zeros((unique_coords.shape[0], 3), dtype=pts_cpu.dtype)
    np.add.at(sums, inverse, pts_cpu)
    counts = np.bincount(inverse, minlength=unique_coords.shape[0]).astype(pts_cpu.dtype)
    downsampled = sums / np.clip(counts[:, None], a_min=1.0, a_max=None)
    return torch.as_tensor(downsampled, dtype=points.dtype, device=points.device)


def depth_boxplot_filter_mask(
    points: torch.Tensor,
    *,
    iqr_scale: float,
    min_keep_ratio: float,
    near_iqr_scale: float | None = None,
) -> torch.Tensor:
    num_points = int(points.shape[0]) if points.dim() >= 2 else 0
    if num_points == 0:
        return torch.zeros((0,), dtype=torch.bool, device=points.device)
    keep_mask = torch.ones((num_points,), dtype=torch.bool, device=points.device)
    if num_points < 4:
        return keep_mask

    depth = points[:, 2]
    valid = torch.isfinite(depth)
    if int(valid.sum().item()) < 4:
        return keep_mask

    depth_valid = depth[valid]
    q1 = torch.quantile(depth_valid, 0.25)
    q3 = torch.quantile(depth_valid, 0.75)
    iqr = q3 - q1
    if not torch.isfinite(q1) or not torch.isfinite(q3) or not torch.isfinite(iqr):
        return keep_mask

    lower_scale = float(iqr_scale) if near_iqr_scale is None else float(near_iqr_scale)
    lower = q1 - lower_scale * iqr
    upper = q3 + float(iqr_scale) * iqr
    eps = torch.finfo(depth_valid.dtype).eps * max(1.0, float(torch.max(depth_valid.abs()).item()))
    keep_valid = (depth_valid >= (lower - eps)) & (depth_valid <= (upper + eps))

    keep_mask = torch.zeros((num_points,), dtype=torch.bool, device=points.device)
    keep_mask[valid] = keep_valid
    keep_count = int(keep_mask.sum().item())
    keep_ratio = float(keep_count) / max(1, int(valid.sum().item()))
    if keep_count < 4 or keep_ratio < float(min_keep_ratio):
        return torch.ones((num_points,), dtype=torch.bool, device=points.device)
    return keep_mask


def dbscan_filter_points(
    points: torch.Tensor,
    *,
    eps: float,
    min_samples: int,
    min_keep_ratio: float,
    pre_voxel_size: float | None = None,
) -> torch.Tensor:
    if points.numel() == 0:
        return points
    if points.shape[0] < max(2, min_samples):
        return points

    sample_points = points
    if pre_voxel_size is not None and pre_voxel_size > 0.0:
        sample_points = voxel_downsample_points(points, pre_voxel_size)
        if sample_points.shape[0] < max(2, min_samples):
            return points

    try:
        from sklearn.cluster import DBSCAN
    except Exception:
        return points

    labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(sample_points.detach().cpu().numpy()).labels_
    labels_t = torch.as_tensor(labels, device=sample_points.device)
    valid = labels_t >= 0
    if int(valid.sum().item()) == 0:
        return points

    valid_labels = labels_t[valid]
    uniq, counts = torch.unique(valid_labels, return_counts=True)
    main_label = int(uniq[torch.argmax(counts)].item())
    main_mask_sample = labels_t == main_label

    keep_ratio_sample = float(main_mask_sample.float().mean().item())
    if keep_ratio_sample < float(min_keep_ratio):
        return points

    if sample_points.shape[0] == points.shape[0]:
        return points[main_mask_sample]

    # Pull original points near the downsampled main cluster back into the inlier set.
    main_cluster = sample_points[main_mask_sample]
    if main_cluster.numel() == 0:
        return points
    # Use cKDTree instead of torch.cdist to avoid O(N*M) GPU memory allocation
    # which can exceed GPU capacity with large point clouds (e.g. 50k×40k = 8 GB).
    from scipy.spatial import cKDTree
    main_tree = cKDTree(main_cluster.detach().cpu().numpy())
    min_dist_np, _ = main_tree.query(points.detach().cpu().numpy(), k=1)
    min_dist = torch.as_tensor(min_dist_np, dtype=points.dtype, device=points.device)
    keep_mask = min_dist <= (float(eps) * 1.5)
    keep_ratio = float(keep_mask.float().mean().item())
    if keep_ratio < float(min_keep_ratio) or int(keep_mask.sum().item()) < max(2, min_samples):
        return points
    return points[keep_mask]


def expanded_box_contains(point: torch.Tensor, bbox: Tuple[torch.Tensor, torch.Tensor], expansion: float = 0.02) -> bool:
    bmin, bmax = bbox
    lower = bmin - expansion
    upper = bmax + expansion
    return bool(torch.all(point >= lower) and torch.all(point <= upper))


def extract_points_from_mask(
    frame,
    mask: torch.Tensor,
    *,
    max_points: int,
    conf_thr: float,
    min_points: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    height, width = frame_hw(frame)
    pointmap = reshape_pointmap(frame.X_canon, height, width)
    conf_map = reshape_conf(frame.C, height, width)
    mask_rs = resize_mask(mask, height, width)
    mask_area = int(mask_rs.sum().item())
    if mask_area <= 0:
        return None

    idx = torch.nonzero(mask_rs.reshape(-1), as_tuple=False).reshape(-1)
    points = pointmap.reshape(-1, 3)[idx]
    conf = conf_map.reshape(-1)[idx]

    valid = torch.isfinite(points).all(dim=-1) & torch.isfinite(conf) & (conf > conf_thr)
    if int(valid.sum().item()) < min_points:
        return None

    points = points[valid]
    conf = conf[valid]
    del max_points  # Deprecated compatibility parameter: no top-k truncation is applied.
    return points, conf, mask_area


def observation_view_score(score: float, mask_area: int, mean_conf: float, frame_area: int) -> float:
    area_ratio = float(mask_area) / max(1, int(frame_area))
    return float(score) * float(mean_conf) * area_ratio


def world_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.norm(a - b).item())


def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()
