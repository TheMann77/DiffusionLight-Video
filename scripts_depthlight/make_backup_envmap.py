import open3d as o3d
import numpy as np
import argparse
from natsort import natsorted
import glob, os, torch
import ezexr
from tqdm import tqdm
import open3d as o3d
from utility_functions import *

# Requires diffusionlight-video environment

def make_backup_envmap(
        pointcloud_file,
        ball_frames_folder,
        depth_data_folder,
        output_folder,
        save_lightcloud=False,
        lightcloud_name="lightcloud",
        backup_envmap_name="missing_envmap",
        only_hitpoints_name=None, # If included, saves a pointcloud of only the points which were hit directly
        no_torch=False,
        assume_envmap_reflects=False, # If true, we assume the envmap is the reflection as visible from the camera rather than the true envmap (false assumption)
        weight_distance=True,
        colour_average_type="median", # median or mean
        logs=True,
):
    def log(txt):
        if logs:
            print(txt)
    if colour_average_type not in ["median", "mean"]:
        raise ValueError("colour_average_type must be median or mean")
    log("Loading files")
    pcd = o3d.io.read_point_cloud(pointcloud_file)
    envmap_files = natsorted(glob.glob(os.path.join(f"{ball_frames_folder}/hdr", "*.exr")))
    if len(envmap_files) == 0:
        raise ValueError("No environment maps found")
    balls = np.load(f"{depth_data_folder}/balls.npz")
    data = np.load(f"{depth_data_folder}/data.npz")
    voxel_size = np.load(f"{depth_data_folder}/voxel_size.npy").item()

    # p = number of points in pointcloud
    # F = number of input frames into DiffusionLight and VGGT
    # w, h = width/height of HDR envmaps, from DiffusionLight output
    # W, H = width/height of frames in pixels, from VGGT output (not original)
    pointcloud = np.asarray(pcd.points) # (p, 3)
    envmaps = np.stack([load_exr(f) for f in envmap_files], axis=0) # (F, h, w, 3)
    ball_centres = balls["centres"] # (F, 3)
    ball_radii = balls["radii"] # (F,)
    extrinsics = data["extrinsic"] # (F, 3, 4), world-to-camera
    depths = data["depth"] # (F, H, W, 1)

    p, _ = pointcloud.shape
    f, h, w, _ = envmaps.shape
    F, H, W = depths.shape
    assert f == F, "Number of frames inputted to DiffusionLight and depth predictor must be equal"

    log("Setting up voxel grid")
    # Build a voxel grid around pointcloud, assuming each point is centre of a voxel
    grid_min = pointcloud.min(axis=0) - 0.5 * voxel_size
    grid_max = pointcloud.max(axis=0) + 0.5 * voxel_size
    grid_shape = np.ceil((grid_max - grid_min) / voxel_size).astype(np.int64)

    alg_type = "numpy"
    if torch.cuda.is_available():
        if no_torch:
            log("Warning: CUDA available but not being used")
        else:
            alg_type = "torch"
    elif not no_torch:
        log("Warning: CUDA not available, using slower CPU version")
    if alg_type == "numpy":
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
            log("CUDA not available, running on CPU")
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

    D_P, D_P_flat = envmap_to_directions(w, h)

    V = np.array([0.0, 0.0, 1.0])
    N_cam = D_P - V
    N_cam_norm = np.linalg.norm(N_cam, axis=-1, keepdims=True)
    N_cam = np.divide(
        N_cam,
        N_cam_norm,
        out=np.zeros_like(N_cam),
        where=N_cam_norm > 1e-12,
    )
    N_cam_flat = N_cam.reshape(-1, 3)

    envmap_missing_sum = np.zeros((256, 512, 3))
    envmap_missing_count = np.zeros((256, 512))

    log("Iterating frames:")
    for frame in tqdm(range(F)):
        R = extrinsics[frame][:, :3]
        t = extrinsics[frame][:, 3]
        C_ball = ball_centres[frame]
        r_ball = ball_radii[frame]
        O_world = -R.T @ t
        N_world_flat = N_cam_flat @ R
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

        D_P_world_flat = D_P_flat @ R

        env_flat = envmaps[frame].reshape(-1, envmaps.shape[-1])

        if alg_type == "numpy":
            intersection_result = ray_pointcloud_intersection_batch(
                P_batch=P_world_flat,
                D_batch=D_out_flat if assume_envmap_reflects else D_P_world_flat,
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
            if weight_distance:
                weighted_intensities = hit_intensities * dist2[:, None]
            else:
                weighted_intensities = hit_intensities
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
            if assume_envmap_reflects:
                D_batch = torch.from_numpy(D_out_flat).float().to(device)
            else:
                D_batch = torch.from_numpy(D_P_world_flat).float().to(device)
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

            if weight_distance:
                weighted_intensities = hit_intensities * dist2.unsqueeze(1)
            else:
                weighted_intensities = hit_intensities
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
        # D_miss = D_out_flat[miss_mask]
        D_miss = D_P_world_flat[miss_mask]
        env_miss = env_flat[miss_mask]
        # Convert to envmap coordinates to build backup envmap
        u_miss, v_miss = directions_to_envmap(D_miss)
        px_miss = (u_miss * 512).astype(int)
        py_miss = (v_miss * 256).astype(int)
        px_miss = np.clip(px_miss, 0, 512 - 1)
        py_miss = np.clip(py_miss, 0, 256 - 1)
        np.add.at(envmap_missing_sum, (py_miss, px_miss), env_miss)
        np.add.at(envmap_missing_count, (py_miss, px_miss), 1)
    
    envmap_missing_avg = (envmap_missing_sum / np.maximum(envmap_missing_count[..., None], 1)).astype(np.float32)

    os.makedirs(output_folder, exist_ok=True)
    
    if save_lightcloud:
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

        if colour_average_type == "median":
            avg_rgb = median_rgb
        elif colour_average_type == "mean":
            avg_rgb == mean_rgb
        # Can use mean or median here:
        lightcloud = np.concatenate(
            [pointcloud, avg_rgb],
            axis=1
        )

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

        log("Total points:", p)
        mask = np.any(lightcloud[:, 3:] != 0, axis=1)
        coloured_lightcloud = lightcloud[mask]
        log("Coloured points:", coloured_lightcloud.shape[0])

        rgb_new, rgb_ldr = smooth_pointcloud_colors(
            points=points,
            rgb_hdr=rgb_hdr,
            k=10,
            alpha=0.5,
            exposure=exposure,
            gamma=gamma,
            output_path=f"{output_folder}/{lightcloud_name}.ply",
        )

        lightcloud[:, 3:] = rgb_new
        np.save(f"{output_folder}/{lightcloud_name}.npy", lightcloud)

        if only_hitpoints_name:
            # Output only coloured points
            points = coloured_lightcloud[:, :3]   # (p, 3)
            rgb_hdr = coloured_lightcloud[:, 3:]  # (p, 3)
            rgb_ldr = (exposure * rgb_hdr) / (1.0 + exposure * rgb_hdr)
            rgb_ldr = np.clip(rgb_ldr, 0, 1) ** (1.0 / gamma)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(rgb_ldr)
            o3d.io.write_point_cloud(f"{output_folder}/{only_hitpoints_name}.ply", pcd)

    ezexr.imwrite(f"{output_folder}/{backup_envmap_name}.exr", envmap_missing_avg)

def create_argparser():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--pointcloud", type=str, default="intermediate/depth_vggt/pointcloud.ply", help="pointcloud file to read (.ply)")
    parser.add_argument("--ball_frames_folder", type=str, default="intermediate/ball_frames", help="folder containing the DiffusionLight ball frames, including envmap, hdr, raw and square folders")
    parser.add_argument("--depth_data", type=str, default="intermediate/depth_vggt", help="the folder containing the output of VGGT/DepthAnything")
    parser.add_argument("--out_folder", type=str, default="output", help="The folder to place the lightcloud and backup environment map in")
    
    parser.add_argument("--lightcloud_name", type=str, default="lightcloud", help="the name of the output lightcloud .ply file")
    parser.add_argument("--backup_envmap_name", type=str, default="missing_envmap", help="the name of the output backup envmap .exr file")
    parser.add_argument("--only_hitpoints_name", type=str, help="if provided, saves a pointcloud with only the points which were hit directly")
    
    parser.add_argument('--save_lightcloud', dest='save_lightcloud', action='store_true', help="save the generated lightcloud (usually we only use the backup envmap)")
    parser.set_defaults(save_lightcloud=False)
    parser.add_argument('--no_torch', dest='no_torch', action='store_true', help="use numpy rather than pytorch (slower)")
    parser.set_defaults(no_torch=False)
    parser.add_argument('--no_weighted_distance', dest='weight_distance', action='store_false', help="don't weight lighting by distance away from ball")
    parser.set_defaults(weight_distance=True)
    parser.add_argument("--colour_average_type", type=str, default="median", help="where multiple rays hit a point, average by 'median' (default) or 'mean'")
    parser.add_argument('--hide-logs', dest='logs', action='store_false', help="hide logs")
    parser.set_defaults(logs=True)

    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    make_backup_envmap(
        pointcloud_file=args.pointcloud,
        ball_frames_folder=args.ball_frames_folder,
        depth_data_folder=args.depth_data,
        output_folder=args.out_folder,
        save_lightcloud=args.save_lightcloud,
        lightcloud_name=args.lightcloud_name,
        backup_envmap_name=args.backup_envmap_name,
        only_hitpoints_name=args.only_hitpoints_name,
        no_torch=args.no_torch,
        weight_distance=args.weight_distance,
        colour_average_type=args.colour_average_type,
        logs=args.logs,
    )