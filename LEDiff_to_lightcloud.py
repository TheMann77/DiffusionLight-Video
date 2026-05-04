import numpy as np
import open3d as o3d
import os, torch
from tqdm import tqdm
import cv2
from ray_functions import *

alg_type = "torch" # numpy or torch

# w, h = width/height of HDR images, from LEDiff output
# F = number of input frames into LEDiff and VGGT
hdrs = np.load("intermediate/LEDiff/hdr.npy") # (F, h, w, 3)
pcd = o3d.io.read_point_cloud("intermediate/depth_vggt/pointcloud.ply")
data = np.load("intermediate/depth_vggt/data.npz")
voxel_size = np.load("intermediate/depth_vggt/voxel_size.npy").item()

# p = number of points in pointcloud
# W, H = width/height of frames in pixels, from VGGT output (not original)
pointcloud = np.asarray(pcd.points) # (p, 3)
extrinsics = data["extrinsic"] # (F, 3, 4), world-to-camera
intrinsics = data["intrinsic"] # (F, 3, 3)
depths = data["depth"] # (F, H, W, 1)
depth_confs = data["depth_conf"] # (F, H, W)
all_points = data["points_unproj"] # (F, H, W, 3)
images = data["images"] # (F, H, W, 3)
R = extrinsics[:, :, :3] # (F, 3, 3), world-to-camera
t = extrinsics[:, :, 3] # (F, 3)

# Camera origin in world space
O_world = -np.einsum("fji,fj->fi", R, t) # (F, 3)

K_inv = np.linalg.inv(intrinsics)

p, _ = pointcloud.shape
f, h, w, _ = hdrs.shape
F, H, W, _ = depths.shape
assert f == F, "Number of frames inputted to LEDiff and VGGT must be equal"
hdrs = np.stack([
    cv2.resize(hdr, (W, H), interpolation=cv2.INTER_CUBIC)
    for hdr in hdrs
], axis=0) # (F, H, W, 3)
h, w = H, W

print("Setting up voxel grid")
# Build a voxel grid around pointcloud, assuming each point is centre of a voxel
point_min = pointcloud.min(axis=0)
point_max = pointcloud.max(axis=0)
camera_min = O_world.min(axis=0)
camera_max = O_world.max(axis=0)
grid_min = np.minimum(point_min, camera_min) - 0.5 * voxel_size
grid_max = np.maximum(point_max, camera_max) + 0.5 * voxel_size
grid_shape = np.ceil((grid_max - grid_min) / voxel_size).astype(np.int64)

if alg_type == "numpy":
    if torch.cuda.is_available():
        print("Warning: CUDA available but not being used")
    voxel_lookup = build_voxel_lookup(pointcloud, grid_min, voxel_size, grid_shape)
    occupied_flat_sorted=voxel_lookup["flat_sorted"]
    occupied_point_indices_sorted=voxel_lookup["point_indices_sorted"]
    occupied_voxel_indices_sorted=voxel_lookup["voxel_indices_sorted"]
    occupied_voxel_centres_sorted=voxel_lookup["voxel_centres_sorted"]
    pointcloud_sum_intensities = np.zeros((p, 3)) # Array of total R, G, B intensity values for each pointcloud point
    pointcloud_num_hits = np.zeros((p)) # Array of number of hits for each pointcloud point
elif alg_type == "torch":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available, running on CPU")
    grid = build_gpu_grid(pointcloud, grid_min, voxel_size, grid_shape)
    pointcloud_sum_intensities_torch = torch.zeros((p, 3), device=device)
    pointcloud_num_hits_torch = torch.zeros(p, device=device)
pointcloud_hit_intensities = [[] for _ in range(p)]

# (x, y) is the pixel coordinates
# d is the 3D world direction from the camera to the point (x, y)
xs = np.arange(w)
ys = np.arange(h)
uu, vv = np.meshgrid(xs, ys, indexing="xy")
pix = np.stack([uu, vv, np.ones_like(uu)], axis=-1).astype(np.float64)
d_cam = np.einsum("fij,hwj->fhwi", K_inv, pix)    # (F, H, W, 3)
d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-12
# Rotate into world space
R_c2w = np.transpose(R, (0, 2, 1))
d_world = np.einsum("fij,fhwj->fhwi", R_c2w, d_cam)
d_world /= np.linalg.norm(d_world, axis=-1, keepdims=True) + 1e-12

for frame in tqdm(range(F)):
    hdr_flat = hdrs[frame].copy().reshape(-1, hdrs.shape[-1])
    O_batch = np.broadcast_to(O_world[frame], (h*w, 3)).copy()
    d_batch = d_world[frame].copy().reshape(-1, 3)
    if alg_type == "numpy":
        intersection_result = ray_pointcloud_intersection_batch(
            P_batch=O_batch,
            D_batch=d_batch,
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
        # print(np.sum(hit_mask))
        hit_point_indices = intersection_result["point_index"][hit_mask]
        hit_intensities = hdr_flat[hit_mask]
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
        hit_point_indices_np = hit_point_indices
        weighted_intensities_np = weighted_intensities

    if alg_type == "torch":
        O_batch_torch = torch.from_numpy(O_batch).float().to(device)
        d_batch_torch = torch.from_numpy(d_batch).float().to(device)
        intersection_result = ray_pointcloud_intersection_batch_torch(
            P=O_batch_torch,
            D=d_batch_torch,
            grid=grid,
        )
        hit_mask = intersection_result["hit_mask"]
        # print(hit_mask.sum().item())
        hit_point_indices = intersection_result["point_index"][hit_mask]
        env_flat_torch = torch.from_numpy(hdr_flat).float().to(device)
        hit_intensities = env_flat_torch[hit_mask]
        dist2 = intersection_result["mu"][hit_mask] ** 2

        weighted_intensities = hit_intensities * dist2.unsqueeze(1)
        pointcloud_sum_intensities_torch.index_add_(
            0,
            hit_point_indices,
            weighted_intensities
        )

        pointcloud_num_hits_torch.index_add_(
            0,
            hit_point_indices,
            torch.ones_like(hit_point_indices, dtype=torch.float32)
        )
        hit_point_indices_np = hit_point_indices.detach().cpu().numpy()
        weighted_intensities_np = weighted_intensities.detach().cpu().numpy()
    
    for idx_pt, rgb in zip(hit_point_indices_np, weighted_intensities_np):
        pointcloud_hit_intensities[int(idx_pt)].append(rgb)

if alg_type == "torch":
    pointcloud_sum_intensities = pointcloud_sum_intensities_torch.cpu().numpy()
    pointcloud_num_hits = pointcloud_num_hits_torch.cpu().numpy()

mean_rgb = np.zeros((p, 3), dtype=float)
median_rgb = np.zeros((p, 3), dtype=float)

mask = pointcloud_num_hits > 0
mean_rgb[mask] = (
    pointcloud_sum_intensities[mask]
    / pointcloud_num_hits[mask, None]
)
for i, samples in enumerate(pointcloud_hit_intensities):
    if len(samples) > 0:
        median_rgb[i] = np.median(np.stack(samples, axis=0), axis=0)

# Can use mean or median here:
lightcloud = np.concatenate(
    [pointcloud, median_rgb],
    axis=1
)

os.makedirs(f"output/LEDiff", exist_ok=True)

# Output for testing:
gamma = 2.2
exposure = 1.0

points = lightcloud[:, :3]   # (p, 3)
rgb_hdr = lightcloud[:, 3:]  # (p, 3)
rgb_ldr = (exposure * rgb_hdr) / (1.0 + exposure * rgb_hdr)
rgb_ldr = np.clip(rgb_ldr, 0, 1) ** (1.0 / gamma)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(rgb_ldr)

o3d.io.write_point_cloud(f"output/LEDiff/raw_lightcloud.ply", pcd)

print("Total points:", p)
mask = np.any(lightcloud[:, 3:] != 0, axis=1)
coloured_lightcloud = lightcloud[mask]
print("Coloured points:", coloured_lightcloud.shape[0])

rgb_new, rgb_ldr = smooth_pointcloud_colors(
    points=points,
    rgb_hdr=rgb_hdr,
    k=10,
    alpha=0.5,
    exposure=exposure,
    gamma=gamma,
    output_path=f"output/LEDiff/smoothed_lightcloud.ply",
)

lightcloud[:, 3:] = rgb_new
np.save(f"output/LEDiff/lightcloud.npy", lightcloud)

# Output only coloured points
points = coloured_lightcloud[:, :3]   # (p, 3)
rgb_hdr = coloured_lightcloud[:, 3:]  # (p, 3)
rgb_ldr = (exposure * rgb_hdr) / (1.0 + exposure * rgb_hdr)
rgb_ldr = np.clip(rgb_ldr, 0, 1) ** (1.0 / gamma)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(rgb_ldr)
o3d.io.write_point_cloud(f"output/LEDiff/coloured_lightcloud.ply", pcd)

#np.save(f"output/LEDiff/lightcloud.npy", coloured_lightcloud)