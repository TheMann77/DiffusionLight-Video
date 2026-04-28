import numpy as np
import open3d as o3d

diffusionlight_img_size = 1024
diffusionlight_ball_radius = 256 // 2
voxel_size = 0.005
print("Voxel size:", voxel_size)

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
print("Total points:", points_filtered.shape[0])

# Calculate ball centres in world coordinates:
N, image_height, image_width, _ = imgs.shape
# intrinsic matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
fx = intrinsic[:, 0, 0] # (N,)
fy = intrinsic[:, 1, 1] # (N,)
cx = intrinsic[:, 0, 2] # (N,)
cy = intrinsic[:, 1, 2] # (N,)
R = extrinsic[:, :, :3] # (88, 3, 3)
t = extrinsic[:, :, 3] # (88, 3)
u = image_width / 2.0 - 0.5 # scalar
v = image_height / 2.0 - 0.5 # scalar
d = depth.min(axis=(1, 2, 3)) # (N,)
x_cam = (u - cx) * d / fx # (N,)
y_cam = (v - cy) * d / fy # (N,)
z_cam = d # (N,)
X_cam = np.stack([x_cam, y_cam, z_cam], axis=1) # (N, 3)
X_world = []
for i in range(N):
    X_world.append(R[i].T @ (X_cam[i] - t[i]))
ball_centres = np.stack(X_world) #(N, 3)
# Calculate ball radii in world coordinates:
ball_radius_pixels = diffusionlight_ball_radius / diffusionlight_img_size * max(image_height, image_width)
f = (fx + fy) / 2
ball_radii = (d * ball_radius_pixels) / (np.sqrt(np.square(f) + np.square(ball_radius_pixels)))

np.savez_compressed(
        "intermediate/depth_vggt/balls.npz",
        centres = ball_centres,
        radii = ball_radii
    )

colors = imgs[mask]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_filtered.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

o3d.io.write_point_cloud("intermediate/depth_vggt/pointcloud.ply", pcd)
np.save("intermediate/depth_vggt/voxel_size.npy", np.array(voxel_size))
print("Downsampled points:", len(pcd.points))

# Visualise spheres for sanity check:
"""sphere_points = []

for c, r in zip(ball_centres, ball_radii):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    mesh.translate(c)

    # sample points on surface
    pts = mesh.sample_points_uniformly(number_of_points=1000)
    sphere_points.append(np.asarray(pts.points))

sphere_points = np.vstack(sphere_points)

sphere_pcd = o3d.geometry.PointCloud()
sphere_pcd.points = o3d.utility.Vector3dVector(sphere_points)
sphere_pcd.paint_uniform_color([1, 0, 0])

combined_pcd = pcd + sphere_pcd

o3d.io.write_point_cloud("scene_with_spheres.ply", combined_pcd)"""