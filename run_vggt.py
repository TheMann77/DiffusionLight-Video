import torch
torch.cuda.empty_cache()
from natsort import natsorted
import glob, os
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.load_fn import load_and_preprocess_images
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

frames_path = "input/example"
file_filter = "*.png"
image_names = natsorted(glob.glob(os.path.join(frames_path, file_filter)))
# Load and preprocess example images
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    print(extrinsic.shape, intrinsic.shape)

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    print(depth_map.shape, depth_conf.shape)

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
    print(point_map.shape, point_conf.shape)
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))
    print(point_map_by_unprojection.shape)

    # Move to CPU + numpy
    extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()        # (88, 3, 4)
    intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()        # (88, 3, 3)
    depth_np = depth_map.squeeze(0).detach().cpu().numpy()            # (88, H, W, 1)
    depth_conf_np = depth_conf.squeeze(0).detach().cpu().numpy()      # (88, H, W)

    imgs = images.squeeze(0).detach().cpu()        # (88, 3, H, W)
    imgs = imgs.permute(0, 2, 3, 1).numpy()        # (88, H, W, 3)

    # Undo normalization if needed (likely required)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    imgs = imgs * std + mean
    imgs = np.clip(imgs, 0, 1).astype(np.float32)

    os.makedirs("intermediate/depth_vggt", exist_ok=True)
    # Save
    np.savez_compressed(
        "intermediate/depth_vggt/data.npz",
        extrinsic=extrinsic_np,
        intrinsic=intrinsic_np,
        depth=depth_np,
        depth_conf=depth_conf_np,
        points_unproj=point_map_by_unprojection,
        images=imgs
    )
    print("Saved VGGT outputs.")