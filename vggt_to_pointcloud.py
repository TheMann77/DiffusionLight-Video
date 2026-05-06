import numpy as np
import open3d as o3d
import argparse

# Required diffusionlight-video environment

def vggt_to_pointcloud(
        data_file,
        out_folder,
        conf_quantile=0.1,
        voxel_size=0.005,
        pointcloud_name="pointcloud",
        diffusionlight_img_size=1024,
        diffusionlight_ball_radius=128,
        logs=True
):
    def log(txt):
        if logs:
            print(txt)
    log("Voxel size:", voxel_size)
    data = np.load(data_file)

    extrinsic = data["extrinsic"]
    intrinsic = data["intrinsic"]
    depth = data["depth"]
    conf = data["depth_conf"]
    points = data["points_unproj"]
    imgs = data["images"]
    # Quantile 0.1 keeps 90% of points
    threshold = np.quantile(conf, conf_quantile)
    mask = conf > threshold
    points_filtered = points[mask]
    log("Total points:", points_filtered.shape[0])

    # Calculate ball centres in world coordinates:
    N, image_height, image_width, _ = imgs.shape

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
            f"{out_folder}/balls.npz",
            centres = ball_centres,
            radii = ball_radii
        )

    colors = imgs[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    o3d.io.write_point_cloud(f"{out_folder}/{pointcloud_name}.ply", pcd)

    np.save(f"{out_folder}/voxel_size.npy", np.array(voxel_size))
    log("Downsampled points:", len(pcd.points))

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="intermediate/depth_anything/data.npz", help='.npz file containing DepthAnything result')
    parser.add_argument("--output_folder", type=str, default="intermediate/depth_anything", help='folder to save pointcloud in')
    parser.add_argument("--pointcloud_name", type=str, default="pointcloud", help='name for pointcloud .ply file')
    parser.add_argument("--voxel_size", type=float, default=0.05, help='voxel size for pointcloud')
    parser.add_argument("--conf_quantile", type=float, default=0.1, help="Filters by points confidence (.05 = keep top 95 percent of points)")
    parser.add_argument('--hide-logs', dest='logs', action='store_false', help="hide logs")
    parser.set_defaults(logs=True)

    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    vggt_to_pointcloud(
        data_file=args.data_file,
        out_folder=args.output_folder,
        conf_quantile=args.conf_quantile,
        voxel_size=args.voxel_size,
        pointcloud_name=args.pointcloud_name,
        logs=args.logs
    )