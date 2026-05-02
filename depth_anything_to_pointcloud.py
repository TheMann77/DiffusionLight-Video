import numpy as np
import open3d as o3d

print("Loading files")
depths = np.load(f"intermediate/depth/raw/depth.npy")
intrinsics = np.load(f"intermediate/depth/raw/intrinsics.npy")
extrinsics = np.load(f"intermediate/depth/raw/extrinsics.npy")
conf = np.load(f"intermediate/depth/raw/conf.npy")
# Filters by confidence (5 = top 95% of points)
thresholds = np.percentile(conf, 1, axis=(1, 2), keepdims=True)
conf_mask = conf > thresholds
num_images, image_height, image_width = depths.shape

voxel_size = 0.05

print("Extracting values")
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

print("Calculating world points")
R_inv = np.transpose(R, (0, 2, 1))  # R^T
t_inv = -np.einsum('nij,nj->ni', R_inv, t)  # -R^T t

points_world = (
    R_inv[:, None, None, :, :] @ points_cam[..., None]
).squeeze(-1) + t_inv[:, None, None, :]

points_world = points_world.reshape(-1, 3)

print("Filtering valid points")
pts = points_world.reshape(-1, 3)
conf_flat = conf.reshape(-1)
depth_flat = depths.reshape(-1)
# min_depth = np.percentile(depths, 5)
conf_mask_flat = conf_mask.reshape(-1)
valid = (
    np.isfinite(pts).all(axis=1)
     & (conf_flat > 0)
     # & (depth_flat > min_depth)
     & conf_mask_flat
)
pts = pts[valid]
conf_flat = conf_flat[valid]

print("Generating point cloud")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

# Colour by confidence:
conf_norm = (conf_flat - conf_flat.min()) / (conf_flat.max() - conf_flat.min() + 1e-8)
colors = np.stack([conf_norm, 0.5 * conf_norm, 1 - conf_norm], axis=1)
pcd.colors = o3d.utility.Vector3dVector(colors)

print("Downsampling")
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

print("Removing outliers")
pcd, _ = pcd.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)

pts_final = np.asarray(pcd.points)

print("Colouring")
# Colour by depth:
z = pts_final[:, 2]
z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
colors = np.stack([z_norm, 0.5 * z_norm, 1 - z_norm], axis=1)

# Colour by confidence:
# conf_final = np.asarray(pcd.colors)[:, 0]
# colors = np.stack([conf_final, 0.5 * conf_final, 1 - conf_final], axis=1)

pcd.colors = o3d.utility.Vector3dVector(colors)

print("Writing file")
o3d.io.write_point_cloud("cloud.ply", pcd)