import open3d as o3d
import numpy as np
from natsort import natsorted
import glob, os
import OpenEXR
import Imath
from tqdm import tqdm
import open3d as o3d

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

print("Loading files")
pcd = o3d.io.read_point_cloud("intermediate/depth_vggt/pointcloud.ply")
envmap_files = natsorted(glob.glob(os.path.join("intermediate/ball_frames/naive/hdr", "*.exr")))
balls = np.load("intermediate/depth_vggt/balls.npz")
data = np.load("intermediate/depth_vggt/data.npz")
voxel_size = np.load("intermediate/depth_vggt/voxel_size.npy").item()

# p = number of points in pointcloud
# F = number of input frames into DiffusionLight and VGGT
# w, h = width/height of HDR envmaps, from DepthAnything output
# W, H = width/height of frames in pixels, from VGGT output (not original)
pointcloud = np.asarray(pcd.points) # (p, 3)
envmaps = np.stack([load_exr(f) for f in envmap_files], axis=0) # (F, h, w, 3)
ball_centres = balls["centres"] # (F, 3)
ball_radii = balls["radii"] # (F,)
extrinsics = data["extrinsic"] # (F, 3, 4), world-to-camera
intrinsics = data["intrinsic"] # (F, 3, 3)
depths = data["depth"] # (F, H, W, 1)
depth_confs = data["depth_conf"] # (F, H, W)
all_points = data["points_unproj"] # (F, H, W, 3)
images = data["images"] # (F, H, W, 3)

p, _ = pointcloud.shape
f, h, w, _ = envmaps.shape
F, H, W, _ = depths.shape
assert f == f, "Number of frames inputted to DiffusionLight and VGGT must be equal"

print("Setting up voxel grid")
# Build a voxel grid around pointcloud, assuming each point is centre of a voxel
grid_min = pointcloud.min(axis=0) - 0.5 * voxel_size
grid_max = pointcloud.max(axis=0) + 0.5 * voxel_size

voxel_indices = np.floor((pointcloud - grid_min[None, :]) / voxel_size).astype(np.int64)
grid_shape = np.ceil((grid_max - grid_min) / voxel_size).astype(np.int64)
valid = np.all((voxel_indices >= 0) & (voxel_indices < grid_shape[None, :]), axis=1)
voxel_indices = voxel_indices[valid]
valid_point_indices = np.flatnonzero(valid)
flat_ids = np.ravel_multi_index(
    voxel_indices.T,
    dims=tuple(grid_shape),
)
order = np.argsort(flat_ids)
occupied_flat_sorted = flat_ids[order]
occupied_voxel_indices_sorted = voxel_indices[order]
occupied_point_indices_sorted = valid_point_indices[order]
occupied_voxel_centres_sorted = pointcloud[occupied_point_indices_sorted]


pointcloud_sum_intensities = np.zeros((p, 3)) # Array of total R, G, B intensity values for each pointcloud point
pointcloud_num_hits = np.zeros((p)) # Array of number of hits for each pointcloud point

# (x, y) are coordinates in the envmap
# N is the surface normal of P
# P is the surface point on the actual ball
# O is the camera position
# X_in(lambda) = O + lambda * D_in; is the ray from the camera to P
# X_out(mu) = P + mu * D_out; is the reflected ray from P


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
V = np.array([1.0, 0.0, 0.0])
N_cam = D_P + V
N_cam_norm = np.linalg.norm(N_cam, axis=-1, keepdims=True)
valid_env = N_cam_norm[..., 0] > 1e-12
N_cam = np.divide(
    N_cam,
    N_cam_norm,
    out=np.zeros_like(N_cam),
    where=N_cam_norm > 1e-12,
)
N_cam_flat = N_cam.reshape(-1, 3)
valid_env_flat = valid_env.reshape(-1)
valid_indices = np.flatnonzero(valid_env_flat)

print("Setup complete")
print("Iterating frames:")
for frame in tqdm(range(F)):
    R = extrinsics[frame][:, :3]
    t = extrinsics[frame][:, 3]
    C_ball = ball_centres[frame]
    r_ball = ball_radii[frame]
    O_world = -R.T @ t
    N_world_flat = N_cam_flat @ R
    N_world_flat /= (
        np.linalg.norm(N_world_flat, axis=1, keepdims=True) + 1e-12
    )
    P_world_flat = C_ball[None, :] + r_ball * N_world_flat
    D_in_flat = P_world_flat - O_world[None, :]
    D_in_flat /= (
        np.linalg.norm(D_in_flat, axis=1, keepdims=True) + 1e-12
    )
    dots = np.sum(D_in_flat * N_world_flat, axis=1, keepdims=True)
    D_out_flat = D_in_flat - 2.0 * dots * N_world_flat
    D_out_flat /= (
        np.linalg.norm(D_out_flat, axis=1, keepdims=True) + 1e-12
    )

    env_flat = envmaps[frame].reshape(-1, envmaps.shape[-1])

    intersection_result = ray_pointcloud_intersection_batch(
        P_batch=P_world_flat[valid_indices],
        D_batch=D_out_flat[valid_indices],
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
    hit_point_indices = intersection_result["point_index"][hit_mask]
    hit_intensities = env_flat[valid_indices][hit_mask]
    # Weight the lighting intensities by the square of how far away that point is
    # So the result is the intensity at 1 unit distance from that point
    dist2 = intersection_result["mu"][hit_mask] ** 2
    weighted_intensities = hit_intensities * dist2[:, None]
    np.add.at(
        pointcloud_sum_intensities,
        hit_point_indices,
        weighted_intensities,
    )
    np.add.at(
        pointcloud_num_hits,
        hit_point_indices,
        1,
    )

average_rgb = np.zeros((p, 3), dtype=float)

mask = pointcloud_num_hits > 0
average_rgb[mask] = (
    pointcloud_sum_intensities[mask]
    / pointcloud_num_hits[mask, None]
)

lightcloud = np.concatenate(
    [pointcloud, average_rgb],
    axis=1
)

os.makedirs("output/naive", exist_ok=True)
np.save("output/naive/lightcloud.npy", lightcloud)

# Output for testing:
points = lightcloud[:, :3]   # (p, 3)
rgb_hdr = lightcloud[:, 3:]  # (p, 3)
rgb_ldr = rgb_hdr / (1.0 + rgb_hdr)
gamma = 2.2
rgb_ldr = np.clip(rgb_ldr, 0, 1) ** (1.0 / gamma)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(rgb_ldr)

o3d.io.write_point_cloud("lightcloud.ply", pcd)
