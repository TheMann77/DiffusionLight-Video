import numpy as np
import open3d as o3d

n = 16
depths = np.load(f"intermediate/depth/raw/depth.npy")[:n]
intrinsics = np.load(f"intermediate/depth/raw/intrinsics.npy")[:n]
extrinsics = np.load(f"intermediate/depth/raw/extrinsics.npy")[:n]
conf = np.load(f"intermediate/depth/raw/conf.npy")[:n]
# Filters top 95% of confidence, but ideally we want to weight point using confidence
# thresholds = np.percentile(conf, 5, axis=(1, 2), keepdims=True)
# conf_mask = conf > thresholds
num_images, image_height, image_width = depths.shape

u, v = np.meshgrid(np.arange(image_width), np.arange(image_height))
u = u[None]
v = v[None]

fx = intrinsics[:, 0, 0][:, None, None]
fy = intrinsics[:, 1, 1][:, None, None]
cx = intrinsics[:, 0, 2][:, None, None]
cy = intrinsics[:, 1, 2][:, None, None]

Z = depths
X = (u - cx) / fx * Z
Y = (v - cy) / fy * Z

points_cam = np.stack([X, Y, Z], axis=-1)

R = extrinsics[:, :3, :3]
t = extrinsics[:, :3, 3]

points_world = (
    R[:, None, None, :, :] @ points_cam[..., None]
).squeeze(-1) + t[:, None, None, :]

points_world = points_world.reshape(-1, 3)

pts = points_world.reshape(-1, 3)
conf_flat = conf.reshape(-1)
# conf_mask_flat = conf_mask.reshape(-1)
valid = np.isfinite(pts).all(axis=1) & (conf_flat > 0) # & conf_mask_flat
pts = pts[valid]
conf_flat = conf_flat[valid]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

# Colour by confidence:
# conf_norm = (conf_flat - conf_flat.min()) / (conf_flat.max() - conf_flat.min() + 1e-8)
# colors = np.stack([conf_norm, 0.5 * conf_norm, 1 - conf_norm], axis=1)
# pcd.colors = o3d.utility.Vector3dVector(colors)

pcd = pcd.voxel_down_sample(voxel_size=0.01)

pcd, _ = pcd.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)

pts_final = np.asarray(pcd.points)

# Colour by depth:
z = pts_final[:, 2]
z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
colors = np.stack([z_norm, 0.5 * z_norm, 1 - z_norm], axis=1)

# Colour by confidence:
# conf_final = np.asarray(pcd.colors)[:, 0]
# colors = np.stack([conf_final, 0.5 * conf_final, 1 - conf_final], axis=1)

pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud("cloud.ply", pcd)