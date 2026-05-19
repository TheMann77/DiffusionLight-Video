import numpy as np
import ezexr
import os
import argparse
from scripts_depthlight.utility_functions import *

# Requires diffusionlight-video environment

def make_final_envmap(
        lightcloud_npy,
        backup_envmap_file,
        lightcloud_downscale_file,
        relative_envmap_positions,
        output_folder,
        output_filestem="envmap",
        voxel_size_file=None,
        voxel_size=None,
        downscale_type="overall",
        weight_distance=False,
        no_torch=False,
        logs=True,
):
    def log(txt):
        if logs:
            print(txt)

    if downscale_type not in ["overall", "uniform", "scale_shift", "none", "map", "channel_map"]:
        raise ValueError("downscale_type must be overall, uniform, channel, scale_shift, none, map, or channel_map")

    log("Loading files")
    lightcloud = np.load(lightcloud_npy) # (p, 6)
    missing_envmap = load_exr(backup_envmap_file) # (h, w, 3)

    if voxel_size is None:
        voxel_size = np.load(voxel_size_file).item()
    if downscale_type not in ["map", "channel_map"]:
        lightcloud_downscale_data = np.load(lightcloud_downscale_file)
        lightcloud_downshift = 0
        if downscale_type == "uniform":
            lightcloud_downscale = lightcloud_downscale_data["uniform"].item()
        elif downscale_type == "channel":
            lightcloud_downscale = lightcloud_downscale_data["channel"]
        elif downscale_type == "overall":
            lightcloud_downscale = lightcloud_downscale_data["overall"]
        elif downscale_type == "scale_shift":
            LC_mean, DL_mean, LC_std, DL_std = lightcloud_downscale_data["scale_shift"]
            lightcloud_downscale = LC_std / DL_std
            lightcloud_downshift = (DL_std / LC_std) * LC_mean - DL_mean
            # This is the downscale, downshift required to act on lightcloud so that its mean and std match the DiffusionLight
        elif downscale_type == "none":
            lightcloud_downscale = 4
            lightcloud_downshift = .05

    #print(lightcloud_downscale, lightcloud_downshift)

    pointcloud = lightcloud[:, :3]

    h, w, _ = missing_envmap.shape

    # Build a voxel grid around pointcloud, assuming each point is centre of a voxel
    grid_min = pointcloud.min(axis=0) - 0.5 * voxel_size
    grid_max = pointcloud.max(axis=0) + 0.5 * voxel_size

    envmap_positions = relative_envmap_positions * np.maximum(np.abs(grid_min), np.abs(grid_max))
    envmap_positions[:, 2] = (grid_max * relative_envmap_positions)[:, 2]
    n, _ = envmap_positions.shape

    alg_type = "numpy"
    if torch.cuda.is_available():
        if no_torch:
            log("Warning: CUDA available but not being used")
        else:
            alg_type = "torch"
    elif not no_torch:
        log("Warning: CUDA not available, using slower CPU version")

    envmaps = build_envmaps_from_lightcloud(
        envmap_positions=envmap_positions,
        lightcloud=lightcloud,
        voxel_size=voxel_size,
        envmap_shape=(h, w),
        alg_type=alg_type,
        weight_distance=weight_distance,
    )

    # Replace missing values with missing_envmap
    zero_mask = np.all(envmaps == 0, axis=-1, keepdims=True)  # (n, h, w, 1)

    if downscale_type in ["map", "channel_map"]:
        non_zero_mask = ~zero_mask
        params = np.array([[ 0.98339655,  1.05078523,  1.82091059, -0.43246688],
                    [ 0.62897061,  1.28343737,  1.46967096, -0.48621854],
                    [ 0.36608806,  1.40202038,  1.27231366, -0.50369265]])
        if downscale_type == "map":
            params = params.mean(axis=0)
            params = np.stack([params, params, params])
        mask3d = non_zero_mask[..., 0]   # (n, h, w)
        for i in range(3):
            a, b, c, d = params[i]
            x = envmaps[..., i]
            x_non_zero = x[mask3d]
            y_non_zero = np.arctan((x_non_zero - b) / a) * c / np.pi - d
            envmaps[..., i][mask3d] = y_non_zero
    else:
        non_zero_mask = (~zero_mask).repeat(3, axis=-1)
        envmaps[non_zero_mask] = envmaps[non_zero_mask] / lightcloud_downscale - lightcloud_downshift

    envmaps = np.where(zero_mask, missing_envmap[None, ...], envmaps)
    envmaps = np.maximum(envmaps, 0)
    envmaps = fill_missing_pixels(envmaps)
        
    os.makedirs(output_folder, exist_ok=True)
    for i, envmap in enumerate(envmaps):
        ezexr.imwrite(f"{output_folder}/{output_filestem}{i}.exr", envmap.astype(np.float32))
        save_hdr_as_ldr(envmap, f"envmap_{i}.png")

def create_argparser():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lightcloud", type=str, default="output/lightcloud.npy", help="lightcloud file to read (.npy)")
    parser.add_argument("--envmap", type=str, default="output/missing_envmap.exr", help=".exr file containing the backup environment map")
    parser.add_argument("--lightcloud_downscale", type=str, default="output/lightcloud_downscale.npz", help=".npz file containing lightcloud downscale information")
    parser.add_argument("--out_folder", type=str, default="final", help="The folder to place the generated environment maps in")
    parser.add_argument("--out_filestem", type=str, default="envmap", help="save the output enviornment maps as out_filestem[i].exr")
    parser.add_argument("--x", type=str, default="0,-.25,.25", help="the relative x coordinate to place the envmap, -1=left, 1=right. Comma separate multiple values.")
    parser.add_argument("--y", type=str, default="0,0,0", help="the relative y coordinate to place the envmap, -1=top, 1=bottom. Comma separate multiple values.")
    parser.add_argument("--z", type=str, default=".5,.5,.5", help="the relative z coordinate to place the envmap, 0=front, 1=back. Comma separate multiple values.")

    parser.add_argument("--voxel_size_file", type=str, default="intermediate/depth/voxel_size.npy", help=".npy file containing the voxel size, or use --voxel_size")
    parser.add_argument("--voxel_size", type=float, default=None, help="voxel size, or use --voxel_size_file")

    parser.add_argument("--downscale_type", type=str, default="overall", help="Downscale type: overall uses total averages, uniform and channel match corresponding directions either linearly or channel-wise")

    parser.add_argument('--weight_distance', dest='weight_distance', action='store_true', help="weight lighting by distance away from ball")
    parser.set_defaults(weight_distance=False)
    parser.add_argument('--no_torch', dest='no_torch', action='store_true', help="use numpy rather than pytorch (slower)")
    parser.set_defaults(no_torch=False)
    parser.add_argument('--hide-logs', dest='logs', action='store_false', help="hide logs")
    parser.set_defaults(logs=True)

    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    xs = [float(v) for v in args.x.split(',')]
    ys = [float(v) for v in args.y.split(',')]
    zs = [float(v) for v in args.z.split(',')]
    assert (len(xs) == len(ys) and len(xs) == len(zs)), "xs, ys and zs must be the same length"
    pos = np.asarray([[xs[i], ys[i], zs[i]] for i in range(len(xs))])
    make_final_envmap(
        lightcloud_npy=args.lightcloud,
        backup_envmap_file=args.envmap,
        lightcloud_downscale_file=args.lightcloud_downscale,
        relative_envmap_positions=pos,
        output_folder=args.out_folder,
        output_filestem=args.out_filestem,
        voxel_size_file=args.voxel_size_file,
        voxel_size=args.voxel_size,
        downscale_type=args.downscale_type,
        weight_distance=args.weight_distance,
        no_torch=args.no_torch,
        logs=args.logs,
    )