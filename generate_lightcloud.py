import open3d as o3d
import numpy as np
from natsort import natsorted
import glob, os
import OpenEXR
import Imath

def load_exr(path):
    exr = OpenEXR.InputFile(path)
    header = exr.header()

    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    r = np.frombuffer(exr.channel('R', pt), dtype=np.float32).reshape(height, width)
    g = np.frombuffer(exr.channel('G', pt), dtype=np.float32).reshape(height, width)
    b = np.frombuffer(exr.channel('B', pt), dtype=np.float32).reshape(height, width)

    img = np.stack([r, g, b], axis=-1)
    return img

def ray_pointcloud_intersection(P, D, max_mu: float | None = None, eps: float = 1e-9):
    # Assume a ray X_out(mu) = P + mu * D, and find first intersection with pointcloud
    # If hit, returns {
    #   "mu" - the mu value of the first intersection with a voxel
    #   "hit_point" - the coordinates of the first intersection with a voxel               
    #   "voxel_index" - integer grid coordinate of the intersected voxel
    #   "voxel_centre" - the pointcloud point of first intersection
    #   "point_index" - the index of that point in the pointcloud array
    # }
    # Otherwise returns None
    mu_enter = -np.inf
    mu_exit = np.inf
    for axis in range(3):
        if abs(D[axis]) < eps:
            # Ray is parallel to this pair of box planes.
            if P[axis] < grid_min[axis] or P[axis] > grid_max[axis]:
                return None
        else:
            mu1 = (grid_min[axis] - P[axis]) / D[axis]
            mu2 = (grid_max[axis] - P[axis]) / D[axis]

            axis_enter = min(mu1, mu2)
            axis_exit = max(mu1, mu2)

            mu_enter = max(mu_enter, axis_enter)
            mu_exit = min(mu_exit, axis_exit)

    if mu_enter > mu_exit:
        return None

    if mu_exit < 0:
        return None

    if max_mu is not None and mu_enter > max_mu:
        return None

    # Start at the first point where the ray enters the point-cloud AABB.
    mu = max(mu_enter, 0.0)

    if max_mu is not None:
        mu_exit = min(mu_exit, max_mu)

    # Small push forward to avoid ambiguity if exactly on a voxel boundary.
    X_start = P + (mu + eps) * D

    current_idx = np.floor((X_start - grid_min) / voxel_size).astype(int)

    # Clamp in case numerical precision puts us just outside.
    current_idx = np.clip(current_idx, 0, grid_shape - 1)

    # Set up DDA traversal:
    step = np.zeros(3, dtype=int)
    mu_next = np.empty(3, dtype=float)
    mu_delta = np.empty(3, dtype=float)
    for axis in range(3):
        if abs(D[axis]) < eps:
            step[axis] = 0
            mu_next[axis] = np.inf
            mu_delta[axis] = np.inf
        elif D[axis] > 0:
            step[axis] = 1
            next_boundary = grid_min[axis] + (current_idx[axis] + 1) * voxel_size
            mu_next[axis] = (next_boundary - P[axis]) / D[axis]
            mu_delta[axis] = voxel_size / abs(D[axis])
        else:
            step[axis] = -1
            next_boundary = grid_min[axis] + current_idx[axis] * voxel_size
            mu_next[axis] = (next_boundary - P[axis]) / D[axis]
            mu_delta[axis] = voxel_size / abs(D[axis])
    
    # Walk through voxels by increasing mu:
    while True:
        # Stop if outside grid.
        if np.any(current_idx < 0) or np.any(current_idx >= grid_shape):
            return None

        idx_tuple = tuple(current_idx)

        # If this voxel is occupied, this is the first hit.
        if idx_tuple in occupied:
            point_index, voxel_centre = occupied[idx_tuple]

            # The actual entry mu is the current mu.
            # If the ray starts inside this voxel, mu can be 0.
            mu_hit = max(mu, 0.0)
            hit_point = P + mu_hit * D

            return {
                "mu": mu_hit,
                "hit_point": hit_point,
                "voxel_index": current_idx.copy(),
                "voxel_centre": voxel_centre,
                "point_index": point_index,
            }

        # Move to the next voxel boundary.
        axis = int(np.argmin(mu_next))

        mu = mu_next[axis]

        if mu > mu_exit:
            return None

        current_idx[axis] += step[axis]
        mu_next[axis] += mu_delta[axis]

print("Loading files")
pcd = o3d.io.read_point_cloud("intermediate/depth_vggt/pointcloud.ply")
envmap_files = natsorted(glob.glob(os.path.join("intermediate/ball_frames/naive/hdr", "*.exr")))
balls = np.load("intermediate/depth_vggt/balls.npz")
data = np.load("intermediate/depth_vggt/data.npz")
voxel_size = np.load("intermediate/depth_vggt/voxel_size.npy").item()

# p = number of points in pointcloud
# F = number of input frames into DiffusionLight and VGGT
# w, h = width/height of HDR envmaps, from DepthAnything output
# W, H = width/height of frames in pixels, from VGGT output (not original)
pointcloud = np.asarray(pcd.points) # (p, 3)
envmaps = np.stack([load_exr(f) for f in envmap_files], axis=0) # (F, h, w, 3)
ball_centres = balls["centres"] # (F, 3)
ball_radii = balls["radii"] # (F,)
extrinsics = data["extrinsic"] # (F, 3, 4), world-to-camera
intrinsics = data["intrinsic"] # (F, 3, 3)
depths = data["depth"] # (F, H, W, 1)
depth_confs = data["depth_conf"] # (F, H, W)
all_points = data["points_unproj"] # (F, H, W, 3)
images = data["images"] # (F, H, W, 3)

p, _ = pointcloud.shape
f, h, w, _ = envmaps.shape
F, H, W, _ = depths.shape
assert f == f, "Number of frames inputted to DiffusionLight and VGGT must be equal"

print("Setting up voxel grid")
# Build a voxel grid around pointcloud, assuming each point is centre of a voxel
grid_min = pointcloud.min(axis=0) - 0.5 * voxel_size
grid_max = pointcloud.max(axis=0) + 0.5 * voxel_size
# Convert point centres to integer voxel indices
voxel_indices = np.floor((pointcloud - grid_min) / voxel_size).astype(int)
grid_shape = voxel_indices.max(axis=0) + 1

occupied = {}
for i, (idx, center) in enumerate(zip(map(tuple, voxel_indices), pointcloud)):
    occupied[idx] = (i, center)

pointcloud_sum_intensities = np.zeros((p, 3)) # Array of total R, G, B intensity values for each pointcloud point
pointcloud_num_hits = np.zeros((p)) # Array of number of hits for each pointcloud point

for frame in range(F):
    print("Frame", frame)
    for x in range(w):
        print(x)
        for y in range(h):
            # (x, y) are coordinates in the envmap
            # N is the surface normal of P
            # P is the surface point on the actual ball
            # O is the camera position
            # X_in(lambda) = O + lambda * D_in; is the ray from the camera to P
            # X_out(mu) = P + mu * D_out; is the reflected ray from P
            theta = 2 * np.pi * x / (w - 1)   # longitude
            phi = np.pi * y / (h - 1)         # latitude
            sin_phi, cos_phi, sin_theta, cos_theta = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta)
            D_P = np.array([sin_phi * cos_theta, sin_phi * sin_theta, cos_phi]) # 3D direction from ball centre to P
            V = np.array([1, 0, 0]) # Unit vector from surface point towards camera. Using DiffusionLight convention.
            N_cam = (D_P + V) / np.linalg.norm(D_P + V)
            R = extrinsics[frame][:, :3]
            t = extrinsics[frame][:, 3]
            N_world = R.T @ N_cam
            P_world = ball_centres[frame] + ball_radii[frame] * N_world
            O_world = -R.T @ t
            D_in = (P_world - O_world) / np.linalg.norm(P_world - O_world)
            D_out = D_in - 2 * np.dot(D_in, N_world) * N_world

            intersection_result = ray_pointcloud_intersection(P_world, D_out)
            if intersection_result is not None:
                point_index = intersection_result["point_index"]
                pointcloud_sum_intensities[point_index] += envmaps[frame, y, x]
                pointcloud_num_hits[point_index] += 1
                
lightcloud = pointcloud_sum_intensities / pointcloud_num_hits[:, None]