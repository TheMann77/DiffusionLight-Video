import numpy as np
import torch
import OpenEXR
import Imath
import open3d as o3d
from scipy.spatial import cKDTree

def load_exr(path):
    exr = OpenEXR.InputFile(path)
    header = exr.header()

    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    r = np.frombuffer(exr.channel('R', pt), dtype=np.float32).reshape(height, width)
    g = np.frombuffer(exr.channel('G', pt), dtype=np.float32).reshape(height, width)
    b = np.frombuffer(exr.channel('B', pt), dtype=np.float32).reshape(height, width)
    img = np.stack([r, g, b], axis=-1)
    return img

def build_voxel_lookup(pointcloud, grid_min, voxel_size, grid_shape):
    voxel_indices = np.floor((pointcloud - grid_min[None, :]) / voxel_size).astype(np.int64)
    valid = np.all((voxel_indices >= 0) & (voxel_indices < grid_shape[None, :]), axis=1)
    voxel_indices = voxel_indices[valid]
    valid_point_indices = np.flatnonzero(valid)
    flat_ids = np.ravel_multi_index(
            voxel_indices.T,
            dims=tuple(grid_shape),
    )
    order = np.argsort(flat_ids)
    point_indices_sorted = valid_point_indices[order]
    return {
        "flat_sorted" : flat_ids[order],
        "voxel_indices_sorted" : voxel_indices[order],
        "point_indices_sorted" : point_indices_sorted,
        "voxel_centres_sorted" : pointcloud[point_indices_sorted]
    }

def ray_pointcloud_intersection_batch(
    P_batch,
    D_batch,
    grid_min,
    grid_max,
    grid_shape,
    voxel_size,
    occupied_flat_sorted,
    occupied_point_indices_sorted,
    occupied_voxel_indices_sorted,
    occupied_voxel_centres_sorted,
    max_mu=None,
    eps=1e-9,
    max_steps=None,
    normalize_directions=True,
):
    """
    Batched ray / voxelized point-cloud intersection.

    Rays:
        X_i(mu) = P_batch[i] + mu * D_batch[i]

    Each occupied voxel corresponds to one point in the original point cloud.

    Parameters
    ----------
    P_batch : np.ndarray, shape (n, 3)
        Ray origins.

    D_batch : np.ndarray, shape (n, 3)
        Ray directions.

    grid_min : np.ndarray, shape (3,)
        Minimum world coordinate of the voxel grid.

    grid_max : np.ndarray, shape (3,)
        Maximum world coordinate of the voxel grid.

    grid_shape : np.ndarray or tuple, shape (3,)
        Number of voxels along x, y, z.

    voxel_size : float
        Side length of each voxel.

    occupied_flat_sorted : np.ndarray, shape (m,)
        Sorted flattened occupied voxel IDs.

    occupied_point_indices_sorted : np.ndarray, shape (m,)
        Original pointcloud index for each occupied voxel.

    occupied_voxel_indices_sorted : np.ndarray, shape (m, 3)
        3D voxel index for each occupied voxel.

    occupied_voxel_centres_sorted : np.ndarray, shape (m, 3)
        Original pointcloud coordinate for each occupied voxel.

    max_mu : float or None
        Optional maximum ray distance/parameter.

    eps : float
        Small numerical tolerance.

    max_steps : int or None
        Maximum number of DDA steps. If None, a conservative default is used.

    normalize_directions : bool
        If True, D_batch is normalized internally, so returned mu is a world-space distance.

    Returns
    -------
    result : dict
        {
            "hit_mask": shape (n,), bool
            "mu": shape (n,), float - the mu value of the first intersection with a voxel
            "hit_point": shape (n, 3), float - the coordinates of the first intersection with a voxel
            "voxel_index": shape (n, 3), int - integer grid coordinate of the intersected voxel
            "voxel_centre": shape (n, 3), float - the pointcloud point of first intersection
            "point_index": shape (n,), int - the index of that point in the pointcloud array
        }

        For rays with no hit:
            hit_mask[i] == False
            mu[i] == np.inf
            point_index[i] == -1
    """

    P_batch = np.asarray(P_batch, dtype=float)
    D_batch = np.asarray(D_batch, dtype=float)

    grid_min = np.asarray(grid_min, dtype=float)
    grid_max = np.asarray(grid_max, dtype=float)
    grid_shape = np.asarray(grid_shape, dtype=np.int64)

    if P_batch.ndim != 2 or P_batch.shape[1] != 3:
        raise ValueError("P_batch must have shape (n, 3)")

    if D_batch.shape != P_batch.shape:
        raise ValueError("D_batch must have the same shape as P_batch")

    n = P_batch.shape[0]

    if normalize_directions:
        D_norm = np.linalg.norm(D_batch, axis=1, keepdims=True)
        valid_dir = D_norm[:, 0] > eps

        D = np.divide(
            D_batch,
            D_norm,
            out=np.zeros_like(D_batch),
            where=D_norm > eps,
        )
    else:
        D = D_batch.copy()
        valid_dir = np.linalg.norm(D, axis=1) > eps

    # ------------------------------------------------------------------
    # Output arrays
    # ------------------------------------------------------------------
    hit_mask = np.zeros(n, dtype=bool)
    hit_mu = np.full(n, np.inf, dtype=float)
    hit_point = np.full((n, 3), np.nan, dtype=float)
    hit_voxel_index = np.full((n, 3), -1, dtype=np.int64)
    hit_voxel_centre = np.full((n, 3), np.nan, dtype=float)
    hit_point_index = np.full(n, -1, dtype=np.int64)

    # ------------------------------------------------------------------
    # 1. Batched ray / overall grid AABB intersection
    # ------------------------------------------------------------------
    parallel = np.abs(D) < eps

    outside_parallel = parallel & (
        (P_batch < grid_min[None, :]) |
        (P_batch > grid_max[None, :])
    )

    invalid = np.any(outside_parallel, axis=1) | (~valid_dir)

    mu1 = np.full((n, 3), -np.inf, dtype=float)
    mu2 = np.full((n, 3), np.inf, dtype=float)

    nonparallel = ~parallel

    mu1[nonparallel] = (
        (grid_min[None, :] - P_batch)[nonparallel]
        / D[nonparallel]
    )

    mu2[nonparallel] = (
        (grid_max[None, :] - P_batch)[nonparallel]
        / D[nonparallel]
    )

    axis_enter = np.minimum(mu1, mu2)
    axis_exit = np.maximum(mu1, mu2)

    mu_enter = np.max(axis_enter, axis=1)
    mu_exit = np.min(axis_exit, axis=1)

    active = (
        (~invalid)
        & (mu_enter <= mu_exit)
        & (mu_exit >= 0.0)
    )

    if max_mu is not None:
        active &= mu_enter <= max_mu
        mu_exit = np.minimum(mu_exit, max_mu)

    if not np.any(active):
        return {
            "hit_mask": hit_mask,
            "mu": hit_mu,
            "hit_point": hit_point,
            "voxel_index": hit_voxel_index,
            "voxel_centre": hit_voxel_centre,
            "point_index": hit_point_index,
        }

    # Start at first valid point inside the grid AABB.
    mu = np.maximum(mu_enter, 0.0)

    X_start = P_batch + (mu[:, None] + eps) * D

    current_idx = np.floor(
        (X_start - grid_min[None, :]) / voxel_size
    ).astype(np.int64)

    current_idx = np.clip(
        current_idx,
        0,
        grid_shape[None, :] - 1,
    )

    # ------------------------------------------------------------------
    # 2. Batched DDA setup
    # ------------------------------------------------------------------
    step = np.zeros((n, 3), dtype=np.int64)
    mu_next = np.full((n, 3), np.inf, dtype=float)
    mu_delta = np.full((n, 3), np.inf, dtype=float)

    positive = D > eps
    negative = D < -eps

    step[positive] = 1
    step[negative] = -1

    for axis in range(3):
        pos = positive[:, axis]
        neg = negative[:, axis]

        if np.any(pos):
            next_boundary_pos = (
                grid_min[axis]
                + (current_idx[:, axis] + 1) * voxel_size
            )

            mu_next[pos, axis] = (
                next_boundary_pos[pos] - P_batch[pos, axis]
            ) / D[pos, axis]

            mu_delta[pos, axis] = voxel_size / np.abs(D[pos, axis])

        if np.any(neg):
            next_boundary_neg = (
                grid_min[axis]
                + current_idx[:, axis] * voxel_size
            )

            mu_next[neg, axis] = (
                next_boundary_neg[neg] - P_batch[neg, axis]
            ) / D[neg, axis]

            mu_delta[neg, axis] = voxel_size / np.abs(D[neg, axis])

    if max_steps is None:
        # Conservative upper bound for crossing the grid.
        max_steps = int(np.sum(grid_shape) + 3)

    still_running = active.copy()

    num_occupied = len(occupied_flat_sorted)

    # ------------------------------------------------------------------
    # 3. Batched DDA traversal
    # ------------------------------------------------------------------
    for _ in range(max_steps):
        ray_ids = np.flatnonzero(still_running)

        if len(ray_ids) == 0:
            break

        idx = current_idx[ray_ids]

        inside = np.all(
            (idx >= 0) & (idx < grid_shape[None, :]),
            axis=1,
        )

        if not np.all(inside):
            still_running[ray_ids[~inside]] = False

            ray_ids = ray_ids[inside]
            idx = idx[inside]

            if len(ray_ids) == 0:
                continue

        # Flatten current voxel indices
        current_flat = np.ravel_multi_index(
            idx.T,
            dims=tuple(grid_shape),
        )

        # Sparse occupied lookup using sorted flat voxel IDs.
        search_pos = np.searchsorted(
            occupied_flat_sorted,
            current_flat,
        )

        valid_search_pos = search_pos < num_occupied

        found = np.zeros(len(ray_ids), dtype=bool)

        safe_pos = search_pos[valid_search_pos]
        found[valid_search_pos] = (
            occupied_flat_sorted[safe_pos]
            == current_flat[valid_search_pos]
        )

        # --------------------------------------------------------------
        # Record hits
        # --------------------------------------------------------------
        if np.any(found):
            found_ray_ids = ray_ids[found]
            found_pos = search_pos[found]

            hit_mask[found_ray_ids] = True

            hit_mu[found_ray_ids] = np.maximum(
                mu[found_ray_ids],
                0.0,
            )

            hit_point[found_ray_ids] = (
                P_batch[found_ray_ids]
                + hit_mu[found_ray_ids, None] * D[found_ray_ids]
            )

            hit_voxel_index[found_ray_ids] = (
                occupied_voxel_indices_sorted[found_pos]
            )

            hit_voxel_centre[found_ray_ids] = (
                occupied_voxel_centres_sorted[found_pos]
            )

            hit_point_index[found_ray_ids] = (
                occupied_point_indices_sorted[found_pos]
            )

            still_running[found_ray_ids] = False

        # --------------------------------------------------------------
        # Advance non-hit rays
        # --------------------------------------------------------------
        remaining_ray_ids = ray_ids[~found]

        if len(remaining_ray_ids) == 0:
            continue

        axes = np.argmin(mu_next[remaining_ray_ids], axis=1)

        new_mu = mu_next[remaining_ray_ids, axes]

        past_exit = new_mu > mu_exit[remaining_ray_ids]

        if np.any(past_exit):
            still_running[remaining_ray_ids[past_exit]] = False

        advance_ray_ids = remaining_ray_ids[~past_exit]
        advance_axes = axes[~past_exit]

        if len(advance_ray_ids) == 0:
            continue

        mu[advance_ray_ids] = mu_next[
            advance_ray_ids,
            advance_axes,
        ]

        current_idx[
            advance_ray_ids,
            advance_axes,
        ] += step[
            advance_ray_ids,
            advance_axes,
        ]

        mu_next[
            advance_ray_ids,
            advance_axes,
        ] += mu_delta[
            advance_ray_ids,
            advance_axes,
        ]

    return {
        "hit_mask": hit_mask,
        "mu": hit_mu,
        "hit_point": hit_point,
        "voxel_index": hit_voxel_index,
        "voxel_centre": hit_voxel_centre,
        "point_index": hit_point_index,
    }

def build_gpu_grid(pointcloud, grid_min, voxel_size, grid_shape, device="cuda"):
    pointcloud = torch.tensor(pointcloud, device=device, dtype=torch.float32)
    grid_min = torch.tensor(grid_min, device=device, dtype=torch.float32)

    voxel_indices = torch.floor((pointcloud - grid_min) / voxel_size).long()

    valid = ((voxel_indices >= 0) & (voxel_indices < torch.tensor(grid_shape, device=device))).all(dim=1)

    voxel_indices = voxel_indices[valid]
    point_indices = torch.nonzero(valid).squeeze(1)

    # Flatten voxel indices
    flat = voxel_indices[:, 0] * (grid_shape[1] * grid_shape[2]) \
         + voxel_indices[:, 1] * grid_shape[2] \
         + voxel_indices[:, 2]

    # Build hash table
    max_flat = grid_shape[0] * grid_shape[1] * grid_shape[2]

    # Sparse: use dict-like structure via two tensors
    flat_sorted, order = torch.sort(flat)
    point_sorted = point_indices[order]
    voxel_sorted = voxel_indices[order]

    return {
        "flat": flat_sorted,
        "point_idx": point_sorted,
        "voxel_idx": voxel_sorted,
        "grid_min": grid_min,
        "grid_shape": torch.tensor(grid_shape, device=device),
        "voxel_size": voxel_size,
    }

def ray_pointcloud_intersection_batch_torch(
    P, D, grid, max_steps=None, eps=1e-6
):
    device = P.device

    grid_min = grid["grid_min"]
    grid_shape = grid["grid_shape"]
    voxel_size = grid["voxel_size"]

    flat_occ = grid["flat"]
    point_idx_occ = grid["point_idx"]

    n = P.shape[0]

    # Normalize directions
    D = D / (torch.norm(D, dim=1, keepdim=True) + eps)

    grid_max = grid_min + voxel_size * grid_shape

    invD = torch.where(torch.abs(D) > eps, 1.0 / D, torch.full_like(D, float("inf")))

    t1 = (grid_min - P) * invD
    t2 = (grid_max - P) * invD

    tmin = torch.minimum(t1, t2)
    tmax = torch.maximum(t1, t2)

    mu_enter = torch.max(tmin, dim=1).values
    mu_exit = torch.min(tmax, dim=1).values

    active = (mu_enter <= mu_exit) & (mu_exit >= 0)

    mu = torch.clamp(mu_enter, min=0.0)

    # Start point
    X = P + (mu.unsqueeze(1) + eps) * D

    idx = torch.floor((X - grid_min) / voxel_size).long()
    idx = torch.maximum(idx, torch.zeros_like(idx))
    idx = torch.minimum(idx, grid_shape - 1)

    # DDA setup
    step = torch.sign(D).long()

    next_boundary = grid_min + (idx + (step > 0).long()) * voxel_size

    mu_next = (next_boundary - P) / D
    mu_delta = voxel_size / torch.abs(D)

    hit_mask = torch.zeros(n, dtype=torch.bool, device=device)
    hit_mu = torch.full((n,), float("inf"), device=device)
    hit_point_idx = torch.full((n,), -1, dtype=torch.long, device=device)

    still = active.clone()

    if max_steps == None:
        max_steps = max_steps = int(grid_shape.sum() + 3)

    for _ in range(max_steps):
        if not still.any():
            break

        idx_active = idx[still]

        flat = idx_active[:, 0] * (grid_shape[1] * grid_shape[2]) \
             + idx_active[:, 1] * grid_shape[2] \
             + idx_active[:, 2]

        # searchsorted (GPU)
        pos = torch.searchsorted(flat_occ, flat)

        valid = pos < flat_occ.shape[0]
        match = torch.zeros_like(valid)

        match[valid] = flat_occ[pos[valid]] == flat[valid]

        # hits
        hit_ids = torch.where(still)[0][match]

        if len(hit_ids) > 0:
            hit_mask[hit_ids] = True
            hit_mu[hit_ids] = mu[hit_ids]
            hit_point_idx[hit_ids] = point_idx_occ[pos[match]]

            still[hit_ids] = False

        # advance
        remain = torch.where(still)[0]

        if len(remain) == 0:
            break

        axes = torch.argmin(mu_next[remain], dim=1)

        mu_new = mu_next[remain, axes]

        done = mu_new > mu_exit[remain]
        still[remain[done]] = False

        adv = remain[~done]
        ax = axes[~done]

        mu[adv] = mu_next[adv, ax]

        idx[adv, ax] += step[adv, ax]
        mu_next[adv, ax] += mu_delta[adv, ax]

    return {
        "hit_mask": hit_mask,
        "mu": hit_mu,
        "point_index": hit_point_idx,
    }

def smooth_pointcloud_colors(
    points,
    rgb_hdr,
    k=10,
    alpha=0.5,
    exposure=1.0,
    gamma=2.2,
    output_path=None,
):
    """
    Smooth pointcloud colors using only already-coloured nearby points.

    Parameters
    ----------
    points : (N, 3) array
        Point positions.
    rgb_hdr : (N, 3) array
        HDR colors. Uncoloured points should be exactly [0, 0, 0].
    k : int
        Number of coloured neighbours to use.
    alpha : float
        Blend factor for already-coloured points only:
            new = (1 - alpha) * original + alpha * smoothed
        Missing points use only the smoothed value.
    exposure : float
        Exposure for LDR conversion.
    gamma : float
        Gamma for LDR conversion.
    output_path : str or None
        If given, writes a coloured PLY to this path.

    Returns
    -------
    rgb_new : (N, 3) array
        Smoothed HDR colors.
    rgb_ldr : (N, 3) array
        Gamma-corrected LDR colors.
    """
    points = np.asarray(points)
    rgb_hdr = np.asarray(rgb_hdr)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if rgb_hdr.shape != points.shape:
        raise ValueError("rgb_hdr must have shape (N, 3)")

    missing = np.all(rgb_hdr == 0, axis=1)
    coloured = ~missing
    n_coloured = int(coloured.sum())

    if n_coloured == 0:
        rgb_new = rgb_hdr.copy()
    else:
        coloured_points = points[coloured]
        coloured_rgb = rgb_hdr[coloured]

        tree = cKDTree(coloured_points)

        # Query one extra neighbour so coloured points can drop self-match.
        k_search = min(k + 1, n_coloured)
        dists, idxs = tree.query(points, k=k_search)

        # Make shapes consistent when k_search == 1
        if k_search == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        # For coloured points, the closest neighbour is usually itself.
        # Drop that first neighbour when present.
        if k_search > 1:
            dists_missing = dists[missing, :k]
            idxs_missing = idxs[missing, :k]

            dists_coloured = dists[coloured, 1 : k + 1]
            idxs_coloured = idxs[coloured, 1 : k + 1]

            # If there are fewer than k+1 coloured points, the slice may be short.
            # Pad by taking whatever is available.
            if dists_coloured.shape[1] < k:
                dists_coloured = dists[coloured, 1:]
                idxs_coloured = idxs[coloured, 1:]
        else:
            dists_missing = dists[missing]
            idxs_missing = idxs[missing]
            dists_coloured = dists[coloured]
            idxs_coloured = idxs[coloured]

        # Compute weighted average for missing points
        rgb_new = rgb_hdr.copy()

        if missing.any():
            w = 1.0 / np.maximum(dists_missing, 1e-8)
            w /= w.sum(axis=1, keepdims=True)
            rgb_new[missing] = np.sum(
                coloured_rgb[idxs_missing] * w[..., None],
                axis=1,
            )

        # Compute weighted average for already-coloured points
        if coloured.any():
            if idxs_coloured.shape[1] == 0:
                smoothed_coloured = coloured_rgb[idxs_coloured[:, :0]]  # empty
            else:
                w = 1.0 / np.maximum(dists_coloured, 1e-8)
                w /= w.sum(axis=1, keepdims=True)
                smoothed_coloured = np.sum(
                    coloured_rgb[idxs_coloured] * w[..., None],
                    axis=1,
                )

            rgb_new[coloured] = (1.0 - alpha) * rgb_hdr[coloured] + alpha * smoothed_coloured

    rgb_ldr = (exposure * rgb_new) / (1.0 + exposure * rgb_new)
    rgb_ldr = np.clip(rgb_ldr, 0, 1) ** (1.0 / gamma)

    if output_path is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgb_ldr)
        o3d.io.write_point_cloud(output_path, pcd)

    return rgb_new, rgb_ldr