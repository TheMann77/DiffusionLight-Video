import argparse
import os

from scripts_depthlight.video_to_frames import generate_frames_from_video

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help='input video file or directory of frames')
    parser.add_argument("--framerate_reduction_factor", type=int, default=5, help="if input is a video file, reduce the framerate by this to increase speed")
    parser.add_argument("--max_frames", type=int, default=-1, help="use at most this many frames, -1 for use all")
    parser.add_argument("--ball_style", type=str, default="naive", help="the ball style to use, naive, smooth or one_seed")
    parser.add_argument("--output_dir", type=str, default="intermediate/ball_frames", help="the directory to put the DiffusionLight output in")
    parser.add_argument("--save_out_video", dest="save_out_video", action="store_true", help="save a video with the chrome balls")
    parser.set_defaults(save_out_video=False)
    parser.add_argument("--out_video_fps", type=int, default=6, help="The frames per second of the output video")

    return parser

def main():
    args = create_argparser().parse_args()

    input_path = args.input
    if os.path.isfile(input_path):
        print("Converting video to frames:")
        dir_path = os.path.dirname(input_path)
        file_stem = os.path.splitext(os.path.basename(input_path))[0]
        generate_frames_from_video(
            input_path,
            dir_path,
            file_stem,
            framerate_reduction=args.framerate_reduction_factor,
            max_frames=args.max_frames,
        )
        input_path = os.path.join(dir_path, file_stem)
    if not os.path.isdir(input_path):
        raise ValueError(f"No files found at input path {input_path}", input_path)
    output_dir = args.output_dir
    print("Inpainting chrome balls:")
    os.system(f"conda run -n diffusionlight-video python inpaint.py --dataset {input_path} --output_dir {output_dir} --video {"--one_seed" if args.ball_style == "one_seed" else "--smooth_frames" if args.ball_style == "smooth" else ""}")
    print("Generating envmaps:")
    os.system(f"conda run -n diffusionlight-video python ball2envmap.py --ball_dir {output_dir}/square --envmap_dir {output_dir}/envmap")
    print("Generating HDR envmaps:")
    os.system(f"conda run diffusionlight-video python exposure2hdr.py --input_dir {output_dir}/envmap --output_dir {output_dir}/hdr")
    if args.save_out_video:
        print("Saving video:")
        os.system(f"conda run -n diffusionlight-video python frames_to_video.py --input_dir {output_dir} --output_dir {output_dir} --fps {args.out_video_fps}")
    
if __name__ == "__main__":
    main()