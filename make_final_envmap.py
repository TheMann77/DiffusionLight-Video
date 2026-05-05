import numpy as np
import ezexr
from ray_functions import *

ball_type = "smooth"
alg_type = "torch" # numpy or torch
relative_envmap_positions = np.asarray([[.5, .7, .6], [.5, .7, .5], [.5, .7, .7]])

print("Loading files")
lightcloud = np.load(f"output/LEDiff/lightcloud.npy") # (p, 6)
missing_envmap = load_exr(f"output/{ball_type}/missing_envmap.exr") # (h, w)
hitting_envmap = load_exr(f"output/{ball_type}/hitting_envmap.exr") # (h, w)
voxel_size = np.load("intermediate/depth_vggt/voxel_size.npy").item()
lightcloud_downscale = np.load(f"output/{ball_type}/lightcloud_downscale.npy").item()

pointcloud = lightcloud[:, :3]
point_colors = lightcloud[:, 3:]

p, _ = lightcloud.shape
h, w, _ = missing_envmap.shape

# Build a voxel grid around pointcloud, assuming each point is centre of a voxel
grid_min = pointcloud.min(axis=0) - 0.5 * voxel_size
grid_max = pointcloud.max(axis=0) + 0.5 * voxel_size
grid_shape = np.ceil((grid_max - grid_min) / voxel_size).astype(np.int64)

envmap_positions = grid_min + (grid_max - grid_min) * relative_envmap_positions

envmaps = build_envmaps_from_lightcloud(
    envmap_positions=envmap_positions,
    lightcloud=lightcloud,
    voxel_size=voxel_size,
    envmap_shape=(h, w),
    alg_type=alg_type,
) / lightcloud_downscale

# Replace missing values with missing_envmap
mask = np.all(envmaps == 0, axis=-1, keepdims=True)  # (n, h, w, 1)
envmaps = np.where(mask, missing_envmap[None, ...], envmaps)

envmaps = fill_missing_pixels(envmaps)
    
for i, envmap in enumerate(envmaps):
    ezexr.imwrite(f"output/{ball_type}/envmap_{i}.exr", envmap.astype(np.float32))