import numpy as np
import ezexr
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
        no_torch=False,
        logs=True,
):
    def log(txt):
        if logs:
            log(txt)

    if downscale_type not in ["overall", "uniform", "channel"]:
        raise ValueError("colour_average_type must be median or mean")

    log("Loading files")
    lightcloud = np.load(lightcloud_npy) # (p, 6)
    missing_envmap = load_exr(backup_envmap_file) # (h, w)
    if voxel_size is None:
        voxel_size = np.load(voxel_size_file).item()
    lightcloud_downscale_data = np.load(lightcloud_downscale_file)
    if downscale_type == "uniform":
        lightcloud_downscale = lightcloud_downscale_data["uniform"].item()
    elif downscale_type == "channel":
        lightcloud_downscale = lightcloud_downscale_data["channel"]
    elif downscale_type == "overall":
        lightcloud_downscale = lightcloud_downscale_data["overall"]

    pointcloud = lightcloud[:, :3]

    h, w, _ = missing_envmap.shape

    # Build a voxel grid around pointcloud, assuming each point is centre of a voxel
    grid_min = pointcloud.min(axis=0) - 0.5 * voxel_size
    grid_max = pointcloud.max(axis=0) + 0.5 * voxel_size

    envmap_positions = grid_min + (grid_max - grid_min) * relative_envmap_positions

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
    ) / lightcloud_downscale

    # Replace missing values with missing_envmap
    mask = np.all(envmaps == 0, axis=-1, keepdims=True)  # (n, h, w, 1)
    envmaps = np.where(mask, missing_envmap[None, ...], envmaps)
    envmaps = fill_missing_pixels(envmaps)
        
    for i, envmap in enumerate(envmaps):
        ezexr.imwrite(f"{output_folder}/{output_filestem}{i}.exr", envmap.astype(np.float32))

def create_argparser():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lightcloud", type=str, default="output/lightcloud.npy", help="lightcloud file to read (.npy)")
    parser.add_argument("--envmap", type=str, default="output/missing_envmap.exr", help=".exr file containing the backup environment map")
    parser.add_argument("--lightcloud_downscale", type=str, default="output/lightcloud_downscale.npz", help=".npz file containing lightcloud downscale information")
    parser.add_argument("--out_folder", type=str, default="final", help="The folder to place the generated environment maps in")
    parser.add_argument("--out_filestem", type=str, default="envmap", help="save the output enviornment maps as out_filestem[i].exr")
    
    parser.add_argument("--voxel_size_file", type=str, default="intermediate/depth/voxel_size.npy", help=".npy file containing the voxel size, or use --voxel_size")
    parser.add_argument("--voxel_size", type=float, default=None, help="voxel size, or use --voxel_size_file")

    parser.add_argument("--downscale_type", type=str, default="overall", help="Downscale type: overall uses total averages, uniform and channel match corresponding directions either linearly or channel-wise")

    parser.add_argument('--no_torch', dest='no_torch', action='store_true', help="use numpy rather than pytorch (slower)")
    parser.set_defaults(no_torch=False)
    parser.add_argument('--hide-logs', dest='logs', action='store_false', help="hide logs")
    parser.set_defaults(logs=True)

    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    make_final_envmap(
        lightcloud_npy=args.lightcloud,
        backup_envmap_file=args.envmap,
        lightcloud_downscale_file=args.lightcloud_downscale,
        relative_envmap_positions=np.asarray([[.5, .5, .4], [.5, .5, .5], [.5, .5, .6]]),
        output_folder=args.out_folder,
        output_filestem=args.out_filestem,
        voxel_size_file=args.voxel_size_file,
        voxel_size=args.voxel_size,
        downscale_type=args.downscale_type,
        no_torch=args.no_torch,
        logs=args.logs,
    )