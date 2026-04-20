import cv2
import glob, os, torch
torch.cuda.empty_cache()
from depth_anything_3.api import DepthAnything3
from natsort import natsorted
import numpy as np

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

have_ball = False

ev = 25
ball_type = "naive"
if have_ball:
   frames_path = f"intermediate/ball_frames/{ball_type}/raw"
   output_name = ball_type
   ev_suffix = f"_ev-{ev}"
else:
    frames_path = "input/example"
    output_name = "raw"
    ev_suffix = ""

original_path = "input/example/example0.png"
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
model.eval()
file_filter = f"*_ev-{ev}.png" if have_ball else "*.png"
images = unpad(natsorted(glob.glob(os.path.join(frames_path, file_filter))), original_path, False)

batch_size = 1

all_depth = []
all_extrinsics = []
all_intrinsics = []

save_pngs = True

image_height, image_width, _ = images[0].shape

if save_pngs:
    os.makedirs(f"intermediate/depth_frames/{output_name}/raw", exist_ok=True)
os.makedirs(f"intermediate/depth/{output_name}", exist_ok=True)

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

    all_depth.append(pred.depth.copy())
    all_extrinsics.append(pred.extrinsics.copy())
    all_intrinsics.append(pred.intrinsics.copy())

    del pred
    torch.cuda.empty_cache()

depth = np.concatenate(all_depth, axis=0)
extrinsics = np.concatenate(all_extrinsics, axis=0)
intrinsics = np.concatenate(all_intrinsics, axis=0)

if save_pngs:
    for i in range(depth.shape[0]):
        d = depth[i]
        d_norm = (d - depth.min()) / (depth.max() - depth.min() + 1e-8)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        cv2.imwrite(
            f"intermediate/depth_frames/{output_name}/raw/depth_{i}{ev_suffix}.png",
            d_uint8
        )

np.save(f"intermediate/depth/{output_name}/depth{ev_suffix}.npy", depth)
np.save(f"intermediate/depth/{output_name}/extrinsics{ev_suffix}.npy", extrinsics)
np.save(f"intermediate/depth/{output_name}/intrinsics{ev_suffix}.npy", intrinsics)