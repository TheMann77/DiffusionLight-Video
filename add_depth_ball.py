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
input_filename = "example"
framerate_reduction = 5

data = np.load(f"intermediate/depth_video/raw/{input_filename}_depths.npz")
depth = data['depths']
depth = 1 / depth
print(depth.min(), depth.mean(), depth.max())
print(depth[0])
num_images, image_height, image_width = depth.shape
ball_mask = make_ball_mask(image_width, image_height, ball_radius)
os.makedirs("intermediate/depth_video/ball", exist_ok=True)
if save_pngs:
    os.makedirs("intermediate/depth_video/ball/depth", exist_ok=True)

frame_index = 0
modified_frames = []
while frame_index < num_images:
    frame = depth[frame_index]
    modified_depth = frame.copy()

    #We assume the distance to the centre of the ball is the minimum depth in the scene
    ball_distance = frame.min() # in depth units
    f = image_width * 0.8 # PLACEHOLDER ESTIMATE FOCAL LENGTH # in pixels
    fx, fy = f, f
    cx = image_width // 2 # in pixels
    cy = image_height // 2 # in pixels
    ball_radius_scene = ball_radius * ball_distance / f # in depth units

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
    print(ball_depth[ball_mask].min(), ball_depth[ball_mask].mean(), ball_depth[ball_mask].max())
    modified_depth[ball_mask] = ball_depth[ball_mask]
    modified_frames.append(modified_depth)

    for _ in range(framerate_reduction - 1):
        frame_index += 1

modified_depths = np.stack(modified_frames, axis=0)

if save_pngs:
    for i in range(modified_depths.shape[0]):
        d = modified_depths[i]
        d_norm = (d - modified_depths.min()) / (modified_depths.max() - modified_depths.min() + 1e-8)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        cv2.imwrite(
            f"intermediate/depth_video/ball/depth/depth_{i}.png",
            d_uint8
        )

np.save("intermediate/depth_video/ball/depth.npy", modified_depths)