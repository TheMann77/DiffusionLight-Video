import torch
torch.cuda.empty_cache()
from contextlib import nullcontext
from natsort import natsorted
import glob, os
import argparse
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.load_fn import load_and_preprocess_images
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
import numpy as np

# Requires vggt environment

def run_vggt(
        frames_folder,
        out_folder,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        amp_ctx = torch.amp.autocast('cuda', dtype=dtype)
    else:
        amp_ctx = nullcontext()

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device=device, dtype=dtype)
    model.eval()

    file_filter = "*.png"
    image_names = natsorted(glob.glob(os.path.join(frames_folder, file_filter)))
    # Load and preprocess example images
    images = load_and_preprocess_images(image_names).to(device)
    print(images.shape)
    with torch.inference_mode():
        with amp_ctx:
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            # Predict Depth Maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))

        # Move to CPU + numpy
        extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()        # (88, 3, 4)
        intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()        # (88, 3, 3)
        depth_np = depth_map.squeeze(0).detach().cpu().numpy().squeeze(-1)          # (88, H, W, 1)
        depth_conf_np = depth_conf.squeeze(0).detach().cpu().numpy()      # (88, H, W)

        imgs = images.squeeze(0).detach().cpu()        # (88, 3, H, W)
        imgs = imgs.permute(0, 2, 3, 1).numpy()        # (88, H, W, 3)

        imgs = np.clip(imgs, 0, 1).astype(np.float32)

        os.makedirs(out_folder, exist_ok=True)
        # Save
        np.savez_compressed(
            f"{out_folder}/data.npz",
            extrinsic=extrinsic_np,
            intrinsic=intrinsic_np,
            depth=depth_np,
            depth_conf=depth_conf_np,
            points_unproj=point_map_by_unprojection,
            images=imgs
        )

def create_argparser():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--frames", type=str, default="input/example", help="folder of input .png frames")
    parser.add_argument("--out_folder", type=str, default="intermediate/depth", help="The folder to place the generated depths in")

    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    run_vggt(
        frames_folder=args.frames,
        out_folder=args.out_folder,
    )