import numpy as np
import os
import cv2

def make_ball_mask(image_width, image_height, radius):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    center = (image_width // 2, image_height // 2)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask.astype(bool)

save_pngs = True
ball_radius = 256 // 2
ev = "25"

depth = np.load(f"intermediate/depth/naive/depth_ev-{ev}.npy")
intrinsics = np.load(f"intermediate/depth/naive/intrinsics_ev-{ev}.npy")
extrinsics = np.load(f"intermediate/depth/naive/extrinsics_ev-{ev}.npy")
raw_depth = np.load(f"intermediate/depth/raw/depth.npy")
raw_intrinsics = np.load(f"intermediate/depth/raw/intrinsics.npy")
raw_extrinsics = np.load(f"intermediate/depth/raw/extrinsics.npy")
num_images, image_height, image_width = depth.shape
ball_mask = make_ball_mask(image_width, image_height, ball_radius)
os.makedirs("intermediate/depth/raw", exist_ok=True)
if save_pngs:
    os.makedirs("intermediate/depth_frames/naive/averaged", exist_ok=True)
    os.makedirs("intermediate/depth_frames/raw/ball", exist_ok=True)
    

#Averaging method:

avg_image = depth.mean(axis=0)
modified_depth = raw_depth.copy()

#We assume the distance to the centre of the ball is the average estimated distance
ball_distances = avg_image[ball_mask].mean() # in depth units
fx = raw_intrinsics[:, 0, 0].mean() # in pixels
fy = raw_intrinsics[:, 1, 1].mean() # in pixels
cx = raw_intrinsics[:, 0, 2].mean() # in pixels
cy = raw_intrinsics[:, 1, 2].mean() # in pixels
ball_radius_scene = ball_radius * ball_distances / np.mean([fx, fy]) # in depth units

#Calculate angles from centre of each point in ball:
H, W = ball_mask.shape
u, v = np.meshgrid(np.arange(W), np.arange(H))
du = (u - cx) / fx
dv = (v - cy) / fy
theta = np.zeros((H, W), dtype=np.float32)
theta[ball_mask] = np.arctan(np.sqrt(du[ball_mask]**2 + dv[ball_mask]**2))

#Calculate depths at each point
ball_depth = np.zeros((H, W), dtype=np.float32)
#Geometrically consistent method
ball_depth[ball_mask] = ball_distances * np.cos(theta[ball_mask]) - np.sqrt(np.maximum((ball_distances ** 2) * (np.square(np.cos(theta[ball_mask])) - 1) + ball_radius_scene ** 2, 0))
#Pixel average method
#ball_depth[ball_mask] = avg_image[ball_mask]

modified_depth[:, ball_mask] = ball_depth[ball_mask]

if save_pngs:
    for i in range(modified_depth.shape[0]):
        d = modified_depth[i]
        d_norm = (d - modified_depth.min()) / (modified_depth.max() - modified_depth.min() + 1e-8)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        cv2.imwrite(
            f"intermediate/depth_frames/naive/averaged/depth_{i}_ev-{ev}.png",
            d_uint8
        )

np.save("intermediate/depth/naive/averaged_depth.npy", modified_depth)


# Calculated method - where we use the fact that DiffusionLight tries to insert the ball at the scene's closest depth:

modified_depth = raw_depth.copy()

for i in range(num_images):
    #We assume the distance to the centre of the ball is the minimum depth of the scene
    frame = raw_depth[i]
    ball_distance = frame.min() # in depth units
    fx = raw_intrinsics[i, 0, 0] # in pixels
    fy = raw_intrinsics[i, 1, 1] # in pixels
    cx = raw_intrinsics[i, 0, 2] # in pixels
    cy = raw_intrinsics[i, 1, 2] # in pixels
    ball_radius_scene = ball_radius * ball_distance / np.mean([fx, fy]) # in depth units

    #Calculate angles from centre of each point in ball:
    H, W = ball_mask.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    du = (u - cx) / fx
    dv = (v - cy) / fy
    theta = np.zeros((H, W), dtype=np.float32)
    theta[ball_mask] = np.arctan(np.sqrt(du[ball_mask]**2 + dv[ball_mask]**2))

    #Calculate depths at each point
    ball_depth = np.zeros((H, W), dtype=np.float32)
    #Geometrically consistent method
    ball_depth[ball_mask] = ball_distance * np.cos(theta[ball_mask]) - np.sqrt(np.maximum((ball_distance ** 2) * (np.square(np.cos(theta[ball_mask])) - 1) + ball_radius_scene ** 2, 0))

    modified_depth[i, ball_mask] = ball_depth[ball_mask]

if save_pngs:
    for i in range(modified_depth.shape[0]):
        d = modified_depth[i]
        d_norm = (d - modified_depth.min()) / (modified_depth.max() - modified_depth.min() + 1e-8)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        cv2.imwrite(
            f"intermediate/depth_frames/raw/ball/depth_{i}.png",
            d_uint8
        )

np.save("intermediate/depth/raw/ball_depth.npy", modified_depth)