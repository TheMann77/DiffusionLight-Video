import numpy as np
import glob, os
from natsort import natsorted
from ray_functions import *

ball_type = "naive"
alg_type = "torch" # numpy or torch

print("Loading files")
lightcloud = np.load(f"output/LEDiff/lightcloud.npy") # (p, 6)
voxel_size = np.load("intermediate/depth_vggt/voxel_size.npy").item()
balls = np.load("intermediate/depth_vggt/balls.npz")
ball_centres = balls["centres"] # (F, 3)
data = np.load("intermediate/depth_vggt/data.npz")
extrinsics = data["extrinsic"] # (F, 3, 4)
R = extrinsics[:, :, :3] # (F, 3, 3), world-to-camera
envmap_files = natsorted(glob.glob(os.path.join(f"intermediate/ball_frames/{ball_type}/hdr", "*.exr")))
DL_envmaps = np.stack([rotate_envmap_camera_to_world(load_exr(f), R[i]) for i, f in enumerate(envmap_files)], axis=0) # (F, h, w, 3)

pointcloud = lightcloud[:, :3]
point_colors = lightcloud[:, 3:]

F, h, w, _ = DL_envmaps.shape
f, _ = ball_centres.shape
p, _ = lightcloud.shape
assert f == F, "Number of frames inputted to VGGT and DiffusionLight must be equal"

LC_envmaps = build_envmaps_from_lightcloud(
    envmap_positions=ball_centres,
    lightcloud=lightcloud,
    voxel_size=voxel_size,
    envmap_shape=(h, w),
    alg_type=alg_type,
) # (F, h, w, 3)

eps = 1e-8
mask = np.any(LC_envmaps != 0, axis=-1)
DL_valid = DL_envmaps[mask]
LC_valid = LC_envmaps[mask]
# luminance
DL_luminance = 0.2126 * DL_envmaps[..., 0] + 0.7152 * DL_envmaps[..., 1] + 0.0722 * DL_envmaps[..., 2]
LC_luminance = 0.2126 * LC_envmaps[..., 0] + 0.7152 * LC_envmaps[..., 1] + 0.0722 * LC_envmaps[..., 2]
log_scale = np.median(np.log(LC_luminance[mask] + eps) - np.log(DL_luminance[mask] + eps))
scale = np.exp(log_scale)
DL_ave = np.median(DL_valid, axis=0)
LC_ave = np.median(LC_valid, axis=0)

print(LC_ave / DL_ave)
print(scale)

np.save(f"output/{ball_type}/lightcloud_downscale.npy", np.array(scale))