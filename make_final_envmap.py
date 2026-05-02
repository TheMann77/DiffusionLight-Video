import open3d as o3d
import numpy as np
from natsort import natsorted
import glob, os, torch
import OpenEXR, ezexr
import Imath
from tqdm import tqdm
import open3d as o3d

ball_type = "naive"
relative_envmap_positions = np.asarray([[.5, .5, .5]])

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
    envmap = np.stack([r, g, b], axis=-1)
    return envmap

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

    invD = 1.0 / torch.clamp(D, min=eps, max=None)

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

print("Loading files")
lightcloud = np.load(f"output/{ball_type}/lightcloud.npy") # (p, 6)
backup_envmap = load_exr(f"output/{ball_type}/backup_envmap.exr") # (h, w)
voxel_size = np.load("intermediate/depth_vggt/voxel_size.npy").item()

pointcloud = lightcloud[:, :3]
point_colors = lightcloud[:, 3:]

p, _ = lightcloud.shape
h, w, _ = backup_envmap.shape

print("Setting up voxel grid")
# Build a voxel grid around pointcloud, assuming each point is centre of a voxel
grid_min = pointcloud.min(axis=0) - 0.5 * voxel_size
grid_max = pointcloud.max(axis=0) + 0.5 * voxel_size
grid_shape = np.ceil((grid_max - grid_min) / voxel_size).astype(np.int64)

envmap_positions = grid_min + (grid_max - grid_min) * relative_envmap_positions

alg_type = "torch" # numpy or torch
if alg_type == "numpy":
    voxel_lookup = build_voxel_lookup(pointcloud, grid_min, voxel_size, grid_shape)
    occupied_flat_sorted=voxel_lookup["flat_sorted"]
    occupied_point_indices_sorted=voxel_lookup["point_indices_sorted"]
    occupied_voxel_indices_sorted=voxel_lookup["voxel_indices_sorted"]
    occupied_voxel_centres_sorted=voxel_lookup["voxel_centres_sorted"]
elif alg_type == "torch":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available, running on CPU")
    grid = build_gpu_grid(pointcloud, grid_min, voxel_size, grid_shape)

# (x, y) are coordinates in the new envmap
# D_P is the 3D direction that the coordinates represent
# P is the origin of the new envmap (in envmap_positions)
# X_out(mu) = P + mu * D_P; is the ray from P
# We generate the envmap in world coordinate space

xs = np.arange(w)
ys = np.arange(h)
theta = 2.0 * np.pi * xs / (w - 1)
phi = np.pi * ys / (h - 1)
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="xy") # (h, w)
sin_phi, cos_phi, sin_theta, cos_theta = np.sin(phi_grid), np.cos(phi_grid), np.sin(theta_grid), np.cos(theta_grid)
D_P = np.stack(
    [
        sin_phi * cos_theta,
        sin_phi * sin_theta,
        cos_phi,
    ],
    axis=-1,
)  # (h, w, 3)
D_P_flat = D_P.reshape(-1, 3)

for i, envmap_position in enumerate(tqdm(envmap_positions)):
    if alg_type == "numpy":
        intersection_result = ray_pointcloud_intersection_batch(
            P_batch=np.broadcast_to(envmap_position, (h*w, 3)),
            D_batch=D_P_flat,
            grid_min=grid_min,
            grid_max=grid_max,
            grid_shape=grid_shape,
            voxel_size=voxel_size,
            occupied_flat_sorted=occupied_flat_sorted,
            occupied_point_indices_sorted=occupied_point_indices_sorted,
            occupied_voxel_indices_sorted=occupied_voxel_indices_sorted,
            occupied_voxel_centres_sorted=occupied_voxel_centres_sorted,
        )
        hit_mask = intersection_result["hit_mask"]
        hit_point_indices = intersection_result["point_index"]
        dist2 = intersection_result["mu"][hit_mask] ** 2

    if alg_type == "torch":
        P_batch = torch.from_numpy(
            np.broadcast_to(envmap_position, (h*w, 3)).copy()
        ).float().cuda()
        D_batch = torch.from_numpy(D_P_flat).float().cuda()
        intersection_result = ray_pointcloud_intersection_batch_torch(
            P=P_batch,
            D=D_batch,
            grid=grid,
        )
        hit_mask = intersection_result["hit_mask"]
        hit_point_indices = intersection_result["point_index"].cpu().numpy()
        dist2 = intersection_result["mu"][hit_mask].cpu().numpy() ** 2
    
    idx = hit_point_indices.reshape(h, w)
    # Set backup envmap
    envmap = np.copy(backup_envmap)
    # Change pixels with no value to red for testing
    mask = np.all(envmap == [0, 0, 0], axis=-1)
    envmap[mask] = [1, 0, 0]
    valid = idx >= 0
    # Weight the lighting intensities by the square of how far away that point is
    envmap[valid] = point_colors[idx[valid]] / dist2[:, None]

    ezexr.imwrite(f"output/{ball_type}/envmap_{i}.exr", envmap.astype(np.float32))