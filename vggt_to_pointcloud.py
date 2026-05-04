import numpy as np
import open3d as o3d
import os
import cv2

diffusionlight_img_size = 1024
diffusionlight_ball_radius = 256 // 2
voxel_size = 0.005
print("Voxel size:", voxel_size)

data = np.load("intermediate/depth_vggt/data.npz")
if os.path.isfile("intermediate/LEDiff/hdr.npy"):
    has_hdr = True
    hdr = np.load("intermediate/LEDiff/hdr.npy")
else:
    has_hdr = False
    print("No HDR files found, generating LDR pointcloud")

extrinsic = data["extrinsic"]
intrinsic = data["intrinsic"]
depth = data["depth"]
conf = data["depth_conf"]
points = data["points_unproj"]
imgs = data["images"]
# Quantile 0.1 keeps 90% of points
threshold = np.quantile(conf * depth[..., 0], 0.1)
mask = (conf * depth[..., 0]) > threshold
points_filtered = points[mask]
print("Total points:", points_filtered.shape[0])

# Calculate ball centres in world coordinates:
N, image_height, image_width, _ = imgs.shape

if has_hdr:
    #Resize HDRs:
    resized_hdr = []
    for frame in hdr:
        resized = cv2.resize(
            frame.astype("float32"),
            (image_width, image_height),
            interpolation=cv2.INTER_CUBIC
        )
        resized_hdr.append(resized)
    hdr = np.stack(resized_hdr)

# Find ball geometry:
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

"""# Visualise a sphere at custom coordinates:
print(np.min(points_filtered, axis=0), np.max(points_filtered, axis=0))
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
sphere.translate([0., 0., 1.])
# sample points on sphere
sphere_pcd = sphere.sample_points_uniformly(number_of_points=2000)
# color it red
sphere_colors = np.tile([1, 0, 0], (np.asarray(sphere_pcd.points).shape[0], 1))
sphere_pcd.colors = o3d.utility.Vector3dVector(sphere_colors)
# combine with your point cloud
combined = pcd + sphere_pcd
o3d.io.write_point_cloud("intermediate/depth_vggt/pointcloud_with_sphere.ply", combined)"""