import open3d as o3d
import numpy as np
from natsort import natsorted
import glob, os, torch
import ezexr
from tqdm import tqdm
import open3d as o3d
from ray_functions import *

ball_type = "naive"
alg_type = "torch" # numpy or torch


print("Loading files")
pcd = o3d.io.read_point_cloud("intermediate/depth_vggt/pointcloud.ply")
envmap_files = natsorted(glob.glob(os.path.join(f"intermediate/ball_frames/{ball_type}/hdr", "*.exr")))
balls = np.load("intermediate/depth_vggt/balls.npz")
data = np.load("intermediate/depth_vggt/data.npz")
voxel_size = np.load("intermediate/depth_vggt/voxel_size.npy").item()

# p = number of points in pointcloud
# F = number of input frames into DiffusionLight and VGGT
# w, h = width/height of HDR envmaps, from DiffusionLight output
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
assert f == F, "Number of frames inputted to DiffusionLight and VGGT must be equal"

print("Setting up voxel grid")
# Build a voxel grid around pointcloud, assuming each point is centre of a voxel
grid_min = pointcloud.min(axis=0) - 0.5 * voxel_size
grid_max = pointcloud.max(axis=0) + 0.5 * voxel_size
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

envmap_missing_sum = np.zeros((256, 512, 3))
envmap_missing_count = np.zeros((256, 512))
envmap_hitting_sum = np.zeros((256, 512, 3))
envmap_hitting_count = np.zeros((256, 512))

print("Iterating frames:")
for frame in tqdm(range(F)):
    R = extrinsics[frame][:, :3]
    t = extrinsics[frame][:, 3]
    C_ball = ball_centres[frame]
    r_ball = ball_radii[frame]
    O_world = -R.T @ t
    N_world_flat = N_cam_flat @ R.T
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

    D_P_flat = D_P.reshape(-1, 3)
    D_P_world_flat = D_P_flat @ R.T

    env_flat = envmaps[frame].reshape(-1, envmaps.shape[-1])

    if alg_type == "numpy":
        intersection_result = ray_pointcloud_intersection_batch(
            P_batch=P_world_flat,
            D_batch=D_out_flat,
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
        hit_intensities = env_flat[hit_mask]
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
        P_batch = torch.from_numpy(P_world_flat).float().to(device)
        D_batch = torch.from_numpy(D_out_flat).float().to(device)
        intersection_result = ray_pointcloud_intersection_batch_torch(
            P=P_batch,
            D=D_batch,
            grid=grid,
        )
        hit_mask = intersection_result["hit_mask"]
        hit_point_indices = intersection_result["point_index"][hit_mask]
        env_flat_torch = torch.from_numpy(env_flat).float().to(device)
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
    
    miss_mask = ~hit_mask
    if alg_type == "torch":
        miss_mask = ~hit_mask.cpu().numpy()
    # Find rays which missed
    D_miss = D_out_flat[miss_mask]
    env_miss = env_flat[miss_mask]
    # Convert to envmap coordinates to build backup envmap
    x_miss, y_miss, z_miss = D_miss[:, 0], D_miss[:, 1], D_miss[:, 2]
    theta_miss = np.arctan2(y_miss, x_miss)
    phi_miss = np.arccos(z_miss)
    theta_miss = np.mod(theta_miss, 2 * np.pi)
    u_miss = theta_miss / (2 * np.pi)
    v_miss = phi_miss / np.pi
    px_miss = (u_miss * 512).astype(int)
    py_miss = (v_miss * 256).astype(int)
    px_miss = np.clip(px_miss, 0, 512 - 1)
    py_miss = np.clip(py_miss, 0, 256 - 1)
    np.add.at(envmap_missing_sum, (py_miss, px_miss), env_miss)
    np.add.at(envmap_missing_count, (py_miss, px_miss), 1)

    # Find rays which hit
    D_hit = D_out_flat[~miss_mask]
    env_hit = env_flat[~miss_mask]
    # Convert to envmap coordinates to build backup envmap
    x_hit, y_hit, z_hit = D_hit[:, 0], D_hit[:, 1], D_hit[:, 2]
    theta_hit = np.arctan2(y_hit, x_hit)
    phi_hit = np.arccos(z_hit)
    theta_hit = np.mod(theta_hit, 2 * np.pi)
    u_hit = theta_hit / (2 * np.pi)
    v_hit = phi_hit / np.pi
    px_hit = (u_hit * 512).astype(int)
    py_hit = (v_hit * 256).astype(int)
    px_hit = np.clip(px_hit, 0, 512 - 1)
    py_hit = np.clip(py_hit, 0, 256 - 1)
    np.add.at(envmap_hitting_sum, (py_hit, px_hit), env_hit)
    np.add.at(envmap_hitting_count, (py_hit, px_hit), 1)

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

envmap_missing_avg = (envmap_missing_sum / np.maximum(envmap_missing_count[..., None], 1)).astype(np.float32)
envmap_hitting_avg = (envmap_hitting_sum / np.maximum(envmap_hitting_count[..., None], 1)).astype(np.float32)

os.makedirs(f"output/{ball_type}", exist_ok=True)

# Output for testing:
gamma = 2.2
exposure = 3.0

points = lightcloud[:, :3]   # (p, 3)
rgb_hdr = lightcloud[:, 3:]  # (p, 3)
rgb_ldr = (exposure * rgb_hdr) / (1.0 + exposure * rgb_hdr)
rgb_ldr = np.clip(rgb_ldr, 0, 1) ** (1.0 / gamma)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(rgb_ldr)

o3d.io.write_point_cloud(f"output/{ball_type}/raw_lightcloud.ply", pcd)

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
    output_path=f"output/{ball_type}/smoothed_lightcloud.ply",
)

lightcloud[:, 3:] = rgb_new
np.save(f"output/{ball_type}/lightcloud.npy", lightcloud)

# Output only coloured points
points = coloured_lightcloud[:, :3]   # (p, 3)
rgb_hdr = coloured_lightcloud[:, 3:]  # (p, 3)
rgb_ldr = (exposure * rgb_hdr) / (1.0 + exposure * rgb_hdr)
rgb_ldr = np.clip(rgb_ldr, 0, 1) ** (1.0 / gamma)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(rgb_ldr)
o3d.io.write_point_cloud(f"output/{ball_type}/coloured_lightcloud.ply", pcd)

ezexr.imwrite(f"output/{ball_type}/missing_envmap.exr", envmap_missing_avg)
ezexr.imwrite(f"output/{ball_type}/hitting_envmap.exr", envmap_hitting_avg)