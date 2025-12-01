import sys
import argparse
from moviepy.editor import VideoFileClip

def extract_video_segment(input_path, output_path, start_time, end_time):
    with VideoFileClip(input_path) as video:
        # 剪辑视频
        clipped = video.subclip(start_time, end_time)
        # 写入输出文件
        clipped.write_videofile(output_path, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取视频片段")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("start", type=float, help="开始时间（秒）")
    parser.add_argument("end", type=float, help="结束时间（秒）")
    parser.add_argument("--output", "-o", default="output.mp4", help="输出文件路径")

    args = parser.parse_args()

    extract_video_segment(args.input, args.output, args.start, args.end)