import argparse

from scripts_depthlight.depth_to_pointcloud import depth_to_pointcloud
from scripts_depthlight.make_backup_envmap import make_backup_envmap
from scripts_depthlight.make_lightcloud import make_lightcloud
from scripts_depthlight.scale_lightcloud import scale_lightcloud

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_frames_folder", type=str, required=True, help='the folder containing the DiffusionLight ball frames')
    parser.add_argument("--depth_folder", type=str, required=True, help='the folder containing the depth data')
    parser.add_argument("--lediff_folder", type=str, required=True, help='the folder containing the lediff output')
    parser.add_argument("--out_folder", type=str, required=True, help='the folder to put the output')
    
    parser.add_argument("--conf_quantile", type=float, default=.1, help='the proportion of points to exclude based on confidence (.1 = exclude 10%)')
    parser.add_argument("--voxel_size", type=float, default=0.005, help="downsizing voxel size of pointcloud. Recommend ~0.05 for DepthAnything, ~0.005 for VGGT")
    
    parser.add_argument('--video', dest='is_video', action='store_true', help="are the input images the frames of a video?")
    parser.set_defaults(is_video=False)

    return parser

def main():
    args = create_argparser().parse_args()
    print("Making pointcloud:")
    depth_to_pointcloud(
        data_file=f"{args.depth_folder}/data.npz",
        out_folder=args.depth_folder,
        conf_quantile=args.conf_quantile,
        voxel_size=args.voxel_size,
    )
    print("Making lightcloud:")
    make_lightcloud(
        pointcloud_file=f"{args.depth_folder}/pointcloud.ply",
        hdr_frames_file=f"{args.lediff_folder}/hdr_bgr.npy",
        depth_data_folder=args.depth_folder,
        output_folder=args.out_folder,
    )
    print("Making backup envmap:")
    make_backup_envmap(
        pointcloud_file=f"{args.depth_folder}/pointcloud.ply",
        ball_frames_folder=args.ball_frames_folder,
        depth_data_folder=args.depth_data_folder,
        output_folder=args.out_folder,
    )
    print("Scaling lightcloud:")
    scale_lightcloud(
        lightcloud_npy=f"{args.out_folder}/lightcloud.npy",
        depth_data_folder=args.depth_folder,
        ball_frames_folder=args.ball_frames_folder,
        output_folder=args.out_folder,
    )
    print("Done")

if __name__ == "__main__":
    main()