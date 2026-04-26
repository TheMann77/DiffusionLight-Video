import numpy as np
import open3d as o3d

data = np.load("intermediate/depth_vggt/data.npz")

extrinsic = data["extrinsic"]
intrinsic = data["intrinsic"]
depth = data["depth"]
conf = data["depth_conf"]
points = data["points_unproj"]
imgs = data["images"]
# Quantile 0.1 keeps 90% of points
threshold = np.quantile(conf, 0.1)
mask = conf > threshold
points_filtered = points[mask]
print(points_filtered.shape)

colors = imgs[mask]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_filtered.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

# Optional cleanup/downsample
pcd = pcd.voxel_down_sample(voxel_size=0.002)

o3d.io.write_point_cloud("vggt_pointcloud.ply", pcd)