import numpy as np
import torch
import ezexr
from tqdm import tqdm
from ray_functions import *

ball_type = "smooth"
relative_envmap_positions = np.asarray([[.5, .7, .6]])

print("Loading files")
lightcloud = np.load(f"output/LEDiff/lightcloud.npy") # (p, 6)
missing_envmap = load_exr(f"output/{ball_type}/missing_envmap.exr") # (h, w)
hitting_envmap = load_exr(f"output/{ball_type}/hitting_envmap.exr") # (h, w)
voxel_size = np.load("intermediate/depth_vggt/voxel_size.npy").item()

pointcloud = lightcloud[:, :3]
point_colors = lightcloud[:, 3:]

p, _ = lightcloud.shape
h, w, _ = missing_envmap.shape

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
        ).float().to(device)
        D_batch = torch.from_numpy(D_P_flat).float().to(device)
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
    # envmap = np.zeros((h, w, 3))
    envmap = np.copy(missing_envmap)
    # Change pixels with no value to red for testing
    mask = np.all(envmap == [0, 0, 0], axis=-1)
    envmap[mask] = [envmap[:, :, 0].max(), 0, 0]
    valid = idx >= 0
    # Weight the lighting intensities by the square of how far away that point is
    envmap[valid] = point_colors[idx[valid]] / dist2[:, None]

    ezexr.imwrite(f"output/{ball_type}/envmap_{i}.exr", envmap.astype(np.float32))