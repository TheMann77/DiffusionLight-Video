import cv2
import glob, os, torch
from depth_anything_3.api import DepthAnything3
from natsort import natsorted
import numpy as np

def unpad(img_files, original_frame):
    imgs = [cv2.imread(img_file) for img_file in img_files]
    original = cv2.imread(original_frame)
    goal_height, goal_width = 1024, 1024
    if original.shape[0] < original.shape[1]:
        goal_height = 1024 * original.shape[0] // original.shape[1]
    elif original.shape[0] > original.shape[1]:
        goal_width = 1024 * original.shape[1] // original.shape[0]
    new_imgs = []
    for img in imgs:
        new_img = img[512-(goal_height//2):512-(goal_height//2)+goal_height, 512-(goal_width//2):512-(goal_width//2)+goal_width, :]
        new_imgs.append(new_img)
    return new_imgs

def make_ball_mask(image_width, image_height, radius):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    center = (image_width // 2, image_height // 2)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask.astype(bool)

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
model.eval()
frames_path = "intermediate/ball_frames/naive/raw"
original_path = "input/example/example0.png"
images = unpad(natsorted(glob.glob(os.path.join(frames_path, "*_ev-00.png"))), original_path)

batch_size = 1

all_depth = []
all_extrinsics = []
all_intrinsics = []

save_pngs = True

ball_radius = 256 // 2

image_height, image_width, _ = images[0].shape
img_sum = None

if save_pngs:
    os.makedirs("intermediate/depth_frames/naive/raw", exist_ok=True)
    os.makedirs("intermediate/depth_frames/naive/averaged", exist_ok=True)
os.makedirs("intermediate/depth/naive", exist_ok=True)

for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    with torch.inference_mode():
        pred = model.inference(batch, process_res=1024)

    # DepthAnything resizes images, so put them back:
    resized_depths = []
    for i in range(len(batch)):
        depth = pred.depth[i]
        depth_resized = cv2.resize(
            depth,
            (image_width, image_height),
            interpolation=cv2.INTER_LINEAR
        )
        resized_depths.append(depth_resized)
    pred.depth = np.stack(resized_depths, axis=0)
    
    for i in range(pred.depth.shape[0]):
        if img_sum is None:
            img_sum = np.zeros_like(pred.depth[i])
        if pred.depth[i].shape != img_sum.shape:
            raise ValueError(f"Image size mismatch: frame {i}")
        img_sum += pred.depth[i]

    all_depth.append(pred.depth.copy())
    all_extrinsics.append(pred.extrinsics.copy())
    all_intrinsics.append(pred.intrinsics.copy())

    del pred
    torch.cuda.empty_cache()

depth = np.concatenate(all_depth, axis=0)
extrinsics = np.concatenate(all_extrinsics, axis=0)
intrinsics = np.concatenate(all_intrinsics, axis=0)

img_avg = img_sum / depth.shape[0]
avg_depth = depth.copy()
ball_mask = make_ball_mask(image_width, image_height, ball_radius)
avg_depth[:, ball_mask] = img_avg[ball_mask]

if save_pngs:
    for i in range(depth.shape[0]):
        d = depth[i]
        d_norm = (d - depth.min()) / (depth.max() - depth.min() + 1e-8)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        cv2.imwrite(
            f"intermediate/depth_frames/naive/raw/depth_{i}_ev-00.png",
            d_uint8
        )

        d = avg_depth[i]
        d_norm = (d - avg_depth.min()) / (avg_depth.max() - avg_depth.min() + 1e-8)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        cv2.imwrite(
            f"intermediate/depth_frames/naive/averaged/depth_{i}_ev-00.png",
            d_uint8
        )

np.save("intermediate/depth/naive/depth.npy", depth)
np.save("intermediate/depth/naive/averaged_depth.npy", depth)
np.save("intermediate/depth/naive/extrinsics.npy", extrinsics)
np.save("intermediate/depth/naive/intrinsics.npy", intrinsics)