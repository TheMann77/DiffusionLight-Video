import numpy as np
import glob, os
from natsort import natsorted
import argparse
try:
    from .utility_functions import *
except ImportError:
    from utility_functions import *
import matplotlib.pyplot as plt

# Requires diffusionlight-video environment

def forward_facing(envmap):
    h, w, _ = envmap.shape
    _, D_P_flat = envmap_to_directions(w, h)
    forward_mask_flat = D_P_flat[:, 2] < 0
    forward_mask = forward_mask_flat.reshape(h, w)
    forward_envmap = np.where(forward_mask[..., None], envmap, 0)
    return forward_envmap

def scale_lightcloud(
        lightcloud_npy,
        backup_envmap,
        depth_data_folder,
        ball_frames_folder,
        output_folder,
        weight_distance=False,
        only_forward_facing=False,
        no_torch=False,
        logs=True,
):
    def log(txt):
        if logs:
            print(txt)

    log("Loading files")
    lightcloud = np.load(lightcloud_npy) # (p, 6)
    voxel_size = np.load(f"{depth_data_folder}/voxel_size.npy").item()
    balls = np.load(f"{depth_data_folder}/balls.npz")
    ball_centres = balls["centres"] # (F, 3)
    data = np.load(f"{depth_data_folder}/data.npz")
    extrinsics = data["extrinsic"] # (F, 3, 4)
    backup_envmap = load_exr(backup_envmap)
    R = extrinsics[:, :, :3] # (F, 3, 3), world-to-camera
    envmap_files = natsorted(glob.glob(os.path.join(f"{ball_frames_folder}/hdr", "*.exr")))
    f = len(envmap_files)
    F, _ = ball_centres.shape
    assert f >= F, "Number of frames inputted to DiffusionLight must be at least the number of frames inputted to DepthAnything/VGGT"
    if f > F:
        log("More DiffusionLight frames than depth frames, sampling")
        indices = np.linspace(0, f-1, F, dtype=int)
        envmap_files = [envmap_files[i] for i in indices]

    # Transform DiffusionLight envmaps to world-coordinates
    if only_forward_facing:
        # Only use parts of envmap which face towards camera, because DiffusionLight knows very little about the rest
        DL_envmaps = np.stack([rotate_envmap_camera_to_world(forward_facing(load_exr(f)), R[i]) for i, f in enumerate(envmap_files)], axis=0) # (F, h, w, 3)
    else:
        DL_envmaps = np.stack([rotate_envmap_camera_to_world(load_exr(f), R[i]) for i, f in enumerate(envmap_files)], axis=0) # (F, h, w, 3)
    

    _, h, w, _ = DL_envmaps.shape

    use_backup_envmap = True
    if use_backup_envmap:
        DL_envmaps = np.broadcast_to(backup_envmap[None], (F, h, w, 3))

    alg_type = "numpy"
    if torch.cuda.is_available():
        if no_torch:
            log("Warning: CUDA available but not being used")
        else:
            alg_type = "torch"
    elif not no_torch:
        log("Warning: CUDA not available, using slower CPU version")
    LC_envmaps = build_envmaps_from_lightcloud(
        envmap_positions=ball_centres,
        lightcloud=lightcloud,
        voxel_size=voxel_size,
        envmap_shape=(h, w),
        alg_type=alg_type,
        weight_distance=weight_distance,
    ) # (F, h, w, 3)

    save_hdr_as_ldr(LC_envmaps[0], "LEDiff.png")
    save_hdr_as_ldr(DL_envmaps[0], "DiffusionLight.png")

    # Compare DiffusionLight envmaps with lightcloud, on pixels where the lightcloud hits
    eps = 1e-8
    LC_empty_mask = np.all(LC_envmaps <= eps, axis=-1)   # (F, h, w)
    DL_empty_mask = np.all(DL_envmaps <= eps, axis=-1)   # (F, h, w)
    mask = (
        (~LC_empty_mask) & (~DL_empty_mask)
    )

    DL_valid = DL_envmaps[mask]
    LC_valid = LC_envmaps[mask]

    mean_std = np.array([LC_valid.mean(), DL_valid.mean(), LC_valid.std(), DL_valid.std()])
    """plt.figure()
    for i, c in enumerate(["red", "green", "blue"]):
        plt.scatter(DL_valid[:, i], LC_valid[:, i], c=c, s=.01)
        plt.xlabel("Backup environment map intensity")
        plt.ylabel("Lightcloud environment map intensity")
        plt.savefig(f"colmap.png")"""

    # Find scaling factor of HDRs
    DL_luminance = 0.2126 * DL_envmaps[..., 0] + 0.7152 * DL_envmaps[..., 1] + 0.0722 * DL_envmaps[..., 2]
    LC_luminance = 0.2126 * LC_envmaps[..., 0] + 0.7152 * LC_envmaps[..., 1] + 0.0722 * LC_envmaps[..., 2]
    log_scale = np.median(np.log(LC_luminance[mask] + eps) - np.log(DL_luminance[mask] + eps))
    scale = np.exp(log_scale)
    DL_ave = np.median(DL_valid, axis=0)
    LC_ave = np.median(LC_valid, axis=0)
    channel_scale = LC_ave / DL_ave
    overall_scale = LC_envmaps[~LC_empty_mask].mean() / DL_envmaps[~DL_empty_mask].mean()

    np.savez(
        f"{output_folder}/lightcloud_downscale.npz",
        uniform=np.array(scale),
        channel=channel_scale,
        overall=np.array(overall_scale),
        scale_shift=mean_std,
    )

    log("Per-channel median scaling: " + str(channel_scale))
    log("Log-space luminance scale: " + str(scale))
    log("Overall mean scale: " + str(overall_scale))
    log("Scale and shift: " + str(mean_std))

def create_argparser():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lightcloud", type=str, default="output/lightcloud.npy", help="lightcloud file to read (.npy)")
    parser.add_argument("--backup_envmap", type=str, default="output/missing_envmap.exr", help="the backup HDR envmap")
    parser.add_argument("--depth_data", type=str, default="intermediate/depth", help="folder containing results of depth estimator")
    parser.add_argument("--ball_frames_folder", type=str, default="intermediate/ball_frames", help="folder containing the DiffusionLight ball frames, including envmap, hdr, raw and square folders")
    parser.add_argument("--out_folder", type=str, default="output", help="The folder to place the generated scaling information in")

    parser.add_argument('--only_forward_facing', dest='only_forward_facing', action='store_true', help="Only use parts of envmap which face towards camera, because DiffusionLight knows very little about the rest")
    parser.set_defaults(only_forward_facing=False)
    parser.add_argument('--no_torch', dest='no_torch', action='store_true', help="use numpy rather than pytorch (slower)")
    parser.set_defaults(no_torch=False)
    parser.add_argument('--hide-logs', dest='logs', action='store_false', help="hide logs")
    parser.set_defaults(logs=True)

    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    scale_lightcloud(
        lightcloud_npy=args.lightcloud,
        backup_envmap=args.backup_envmap,
        depth_data_folder=args.depth_data,
        ball_frames_folder=args.ball_frames_folder,
        output_folder=args.out_folder,
        only_forward_facing=args.only_forward_facing,
        no_torch=args.no_torch,
        logs=args.logs,
    )