import numpy as np
import open3d as o3d
import argparse

# Requires depthanything environment

def depth_anything_to_pointcloud(
        data_file,
        out_folder,
        conf_quantile=.05, # Filters by confidence (.05 = keep top 95% of points)
        voxel_size = 0.05,
        pointcloud_name="pointcloud",
        logs=True
        ):
    def log(txt):
        if logs:
            print(txt)

    log("Loading files")
    data = np.load(data_file)
    depths = data["depth"]
    intrinsics = data["intrinsic"]
    extrinsics = data["extrinsic"]
    conf = data["depth_conf"]
    
    thresholds = np.quantile(conf, conf_quantile, axis=(1, 2), keepdims=True)
    conf_mask = conf > thresholds
    num_images, image_height, image_width = depths.shape

    log("Extracting values")
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

    log("Calculating world points")
    R_inv = np.transpose(R, (0, 2, 1))  # R^T
    t_inv = -np.einsum('nij,nj->ni', R_inv, t)  # -R^T t

    points_world = (
        R_inv[:, None, None, :, :] @ points_cam[..., None]
    ).squeeze(-1) + t_inv[:, None, None, :]

    points_world = points_world.reshape(-1, 3)

    log("Filtering valid points")
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

    log("Generating point cloud")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Colour by confidence:
    conf_norm = (conf_flat - conf_flat.min()) / (conf_flat.max() - conf_flat.min() + 1e-8)
    colors = np.stack([conf_norm, 0.5 * conf_norm, 1 - conf_norm], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    log("Downsampling")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    log("Removing outliers")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    pts_final = np.asarray(pcd.points)

    log("Colouring")
    # Colour by depth:
    z = pts_final[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors = np.stack([z_norm, 0.5 * z_norm, 1 - z_norm], axis=1)

    # Colour by confidence:
    # conf_final = np.asarray(pcd.colors)[:, 0]
    # colors = np.stack([conf_final, 0.5 * conf_final, 1 - conf_final], axis=1)

    pcd.colors = o3d.utility.Vector3dVector(colors)

    log("Writing file")
    o3d.io.write_point_cloud(f"{out_folder}/{pointcloud_name}.ply", pcd)
    np.save(f"{out_folder}/voxel_size.npy", np.array(voxel_size))

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="intermediate/depth_anything/data.npz", help='.npz file containing DepthAnything result')
    parser.add_argument("--output_folder", type=str, default="intermediate/depth_anything", help='folder to save pointcloud in')
    parser.add_argument("--pointcloud_name", type=str, default="pointcloud", help='name for pointcloud .ply file')
    parser.add_argument("--voxel_size", type=float, default=0.05, help='voxel size for pointcloud')
    parser.add_argument("--conf_quantile", type=float, default=0.05, help="Filters by points confidence (.05 = keep top 95 percent of points)")
    parser.add_argument('--hide-logs', dest='logs', action='store_false', help="hide logs")
    parser.set_defaults(logs=True)

    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    depth_anything_to_pointcloud(
        data_file=args.data_file,
        out_folder=args.output_folder,
        conf_quantile=args.conf_quantile,
        voxel_size=args.voxel_size,
        pointcloud_name=args.pointcloud_name,
        logs=args.logs
    )