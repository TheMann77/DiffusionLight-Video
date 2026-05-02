import numpy as np
import open3d as o3d

ball_type = "naive"

lightcloud = np.load(f"output/{ball_type}/lightcloud.npy")
mask = np.any(lightcloud[:, 3:] != 0, axis=1)#
lightcloud = lightcloud[mask]

# Output for testing:
points = lightcloud[:, :3]   # (p, 3)
rgb_hdr = lightcloud[:, 3:]  # (p, 3)
rgb_ldr = rgb_hdr / (1.0 + rgb_hdr)
gamma = 2.2
rgb_ldr = np.clip(rgb_ldr, 0, 1) ** (1.0 / gamma)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(rgb_ldr)

o3d.io.write_point_cloud(f"output/{ball_type}/filtered_lightcloud.ply", pcd)