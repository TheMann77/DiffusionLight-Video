import cv2
import glob, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.cuda.empty_cache()
from depth_anything_3.api import DepthAnything3
import argparse
from natsort import natsorted
import numpy as np
from tqdm import tqdm

# Requires depthanything environment

def unpad(img_files, original_frame, padded=True):
    #If padded is true, assumes file is 1024x1024
    #Otherwise resizes them
    imgs = [cv2.imread(img_file) for img_file in img_files]
    original = cv2.imread(original_frame)

    goal_height, goal_width = 1024, 1024
    if original.shape[0] < original.shape[1]:
        goal_height = 1024 * original.shape[0] // original.shape[1]
    elif original.shape[0] > original.shape[1]:
        goal_width = 1024 * original.shape[1] // original.shape[0]
    new_imgs = []
    for img in imgs:
        if padded:
            new_img = img[512-(goal_height//2):512-(goal_height//2)+goal_height, 512-(goal_width//2):512-(goal_width//2)+goal_width, :]
        else:
            new_img = cv2.resize(
                img,
                (goal_width, goal_height),
                interpolation=cv2.INTER_LINEAR
            )
        new_imgs.append(new_img)
    return new_imgs

def resize_frames(frames, width, height):
    resized_frames = []
    for i in range(len(frames)):
        frame = frames[i]
        frame_resized = cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )
        resized_frames.append(frame_resized)
    frames = np.stack(resized_frames, axis=0)

def make_sliding_windows(num_frames, window_size, stride):
    starts = list(range(0, max(num_frames - window_size + 1, 1), stride))
    last_start = max(0, num_frames - window_size)
    if starts[-1] != last_start:
        starts.append(last_start)

    return [(start, min(start + window_size, num_frames)) for start in starts]

def to_homogeneous(T):
    # Convert N x 3 x 4 (or N x 4 x 4) to homogeneous N x 4 x 4
    T = np.asarray(T)
    if T.shape[-2:] == (4, 4):
        return T.copy()

    if T.shape[-2:] != (3, 4):
        raise ValueError(f"Expected (N,3,4) or (N,4,4), got {T.shape}")

    out = np.zeros(T.shape[:-2] + (4, 4), dtype=T.dtype)
    out[..., :3, :4] = T
    out[..., 3, 3] = 1.0
    return out

def from_homogeneous(T):
    return T[..., :3, :4]

def align_windows_to_global(results, num_frames, use_scale_alignment=True):
    """
    Align DA3 per-window extrinsics into one global coordinate system.

    Assumes DA3 extrinsics are world-to-cam:
        X_cam = R @ X_world + t

    Returns:
        depths_global:      (N, H, W)
        intrinsics_global:  (N, 3, 3)
        extrinsics_global:  (N, 3, 4)
        conf_global:        (N, H, W)
    """

    global_T_by_frame = {}
    global_depth_by_frame = {}
    global_K_by_frame = {}
    global_conf_by_frame = {}

    for window_idx, r in enumerate(results):
        start = r["start"]
        end = r["end"]

        depth_local = r["depth"]
        K_local = r["intrinsics"]
        T_local_w2c = to_homogeneous(r["extrinsics"])
        T_local = invert_extrinsics(T_local_w2c)
        conf_local = r["conf"]

        frame_ids = list(range(start, end))

        # First window defines global coordinate system
        if window_idx == 0:
            scale = 1.0
            T_global = T_local.copy()

        else:
            # Find frames shared with already-aligned previous windows
            overlap_frames = [
                f for f in frame_ids
                if f in global_T_by_frame
            ]

            if len(overlap_frames) == 0:
                raise ValueError(
                    f"Window {window_idx} has no overlap with global trajectory. "
                    "Use smaller stride or larger window size."
                )

            # Use middle overlap frame as anchor
            anchor_frame = overlap_frames[len(overlap_frames) // 2]
            anchor_local_idx = anchor_frame - start

            T_anchor_global = global_T_by_frame[anchor_frame]
            T_anchor_local = T_local[anchor_local_idx]

            Rg = T_anchor_global[:3, :3]
            tg = T_anchor_global[:3, 3]

            Rl = T_anchor_local[:3, :3]
            tl = T_anchor_local[:3, 3]

            # Rotation that maps this window's local world axes into global axes
            R_align = Rg @ Rl.T

            # Optional scale alignment using camera centers in overlapping frames
            if use_scale_alignment and len(overlap_frames) >= 2:
                local_centers = []
                global_centers = []

                for f in overlap_frames:
                    li = f - start
                    local_centers.append(T_local[li, :3, 3])
                    global_centers.append(global_T_by_frame[f][:3, 3])

                local_centers = np.asarray(local_centers)
                global_centers = np.asarray(global_centers)

                d_local = np.linalg.norm(np.diff(local_centers, axis=0), axis=1)
                d_global = np.linalg.norm(np.diff(global_centers, axis=0), axis=1)

                valid = d_local > 1e-8

                if np.any(valid):
                    scale = np.median(d_global[valid] / d_local[valid])
                else:
                    scale = 1.0
            else:
                scale = 1.0

            # Translation that maps the anchor camera center into the global one
            t_align = tg - scale * (R_align @ tl)

            # Apply alignment to all poses in this window
            T_global = np.zeros_like(T_local)
            T_global[:, 3, 3] = 1.0

            T_global[:, :3, :3] = R_align[None, :, :] @ T_local[:, :3, :3]

            T_global[:, :3, 3] = (
                scale * (R_align @ T_local[:, :3, 3].T).T
                + t_align[None, :]
            )

        # If scale alignment was used, depth must be scaled too.
        # Otherwise camera translations and depths live in different scales.
        depth_global = depth_local * scale

        # Store only first occurrence of each frame to avoid duplicates.
        for f in frame_ids:
            local_idx = f - start

            if f not in global_T_by_frame:
                global_T_by_frame[f] = T_global[local_idx]
                global_depth_by_frame[f] = depth_global[local_idx]
                global_K_by_frame[f] = K_local[local_idx]
                global_conf_by_frame[f] = conf_local[local_idx]

    # Stack in original frame order
    depths_global = np.stack([global_depth_by_frame[i] for i in range(num_frames)])
    intrinsics_global = np.stack([global_K_by_frame[i] for i in range(num_frames)])
    T_global_c2w = np.stack([global_T_by_frame[i] for i in range(num_frames)])
    T_global_w2c = invert_extrinsics(T_global_c2w)
    extrinsics_global = from_homogeneous(T_global_w2c)
    conf_global = np.stack([global_conf_by_frame[i] for i in range(num_frames)])

    return depths_global, intrinsics_global, extrinsics_global, conf_global

def camera_centers(T):
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    return -np.einsum('nij,nj->ni', R.transpose(0,2,1), t)

def invert_extrinsics(T):
    """Invert homogeneous transforms (N,4,4)"""
    R = T[:, :3, :3]
    t = T[:, :3, 3]

    T_inv = np.zeros_like(T)
    T_inv[:, 3, 3] = 1.0

    R_inv = np.transpose(R, (0, 2, 1))
    t_inv = -np.einsum('nij,nj->ni', R_inv, t)

    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3] = t_inv

    return T_inv

def run_depth_anything(
        frames_folder,
        output_folder,
        batch_size=32,
        save_pngs=False,
):
    stride = batch_size // 2

    file_filter = "*.png"
    image_files = natsorted(glob.glob(os.path.join(frames_folder, file_filter)))
    images = unpad(image_files, image_files[0], False)
    num_images, (image_height, image_width, _) = len(images), images[0].shape
    os.makedirs(output_folder, exist_ok=True)
    if save_pngs:
        os.makedirs(f"{output_folder}/frames", exist_ok=True)

    results = []

    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)
    model.eval()

    for start, end in tqdm(make_sliding_windows(num_images, window_size=batch_size, stride=stride)):
        batch = images[start:end]

        with torch.inference_mode():
            pred = model.inference(batch, process_res=1024)
        
        depths = pred.depth.copy()
        confs = pred.conf.copy()

        resize_frames(depths, image_width, image_height)
        resize_frames(confs, image_width, image_height)

        results.append({
            "start": start,
            "end": end,
            "depth": depths,
            "intrinsics": np.asarray(pred.intrinsics),
            "extrinsics": np.asarray(pred.extrinsics),
            "conf": confs,
        })

        del pred
        torch.cuda.empty_cache()

    depth, intrinsics, extrinsics, conf = (
        align_windows_to_global(
            results,
            num_frames=num_images,
            use_scale_alignment=True,
        )
    )

    if save_pngs:
        for i in range(depth.shape[0]):
            d = depth[i]
            d_norm = (d - depth.min()) / (depth.max() - depth.min() + 1e-8)
            d_uint8 = (d_norm * 255).astype(np.uint8)
            cv2.imwrite(
                f"{output_folder}/frames/depth_{i}.png",
                d_uint8
            )

    np.savez_compressed(f"{output_folder}/data.npz", 
                        extrinsic=extrinsics,
                        intrinsic=extrinsics,
                        depth=depth,
                        depth_conf=conf,
                        imgs=images,
                        )

def create_argparser():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--frames", type=str, default="input/example", help="folder of input .png frames")
    parser.add_argument("--out_folder", type=str, default="final", help="The folder to place the generated depths in")
    parser.add_argument("--batch_size", type=int, default=32, help="number of frames per batch, reduce if memory runs out")
    parser.add_argument("--save_pngs", dest="save_pngs", action='store_true', help="save the per-frame depth pngs")
    parser.set_defaults(save_pngs=False)

    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    run_depth_anything(
        frames_folder=args.frames,
        output_folder=args.out_folder,
        batch_size=args.batch_size,
        save_pngs=args.save_pngs,
    )