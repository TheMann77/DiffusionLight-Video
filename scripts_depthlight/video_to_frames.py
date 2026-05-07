import argparse
import cv2
import os
from tqdm import tqdm

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", type=str, required=True ,help='the video file to convert')
    parser.add_argument("--output_dir", default="", type=str, help="output directory")
    parser.add_argument("--output_filename", default="", type=str, help="the first frame will be [output_filename]_0.png")
    parser.add_argument("--framerate_reduction_factor", default=1, type=int, help="reduce the framerate by an integer factor")
    parser.add_argument("--max_frames", default=-1, type=int, help="maximum number of frames of the video to export")
    return parser

def generate_frames_from_video(video_path, output_dir, output_filename, framerate_reduction, max_frames):
    if output_filename == "":
        output_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = output_dir + "/" + output_filename
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames == -1:
        max_frames = total_frames
    for frame_index in tqdm(range(min(total_frames // framerate_reduction, max_frames))):
        success, image = video_capture.read()
        if success:
            cv2.imwrite(output_path + "/" + output_filename + str(frame_index) + ".png", image)
        else:
            break
        for _ in range(framerate_reduction - 1):
            video_capture.read()

def main():
    # load arguments
    args = create_argparser().parse_args()
    generate_frames_from_video(
        args.video_file,
        args.output_dir,
        args.output_filename,
        args.framerate_reduction_factor,
        args.max_frames,
    )

if __name__ == "__main__":
    main()