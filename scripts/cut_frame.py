import cv2
import argparse

def show_video_frames(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 显示当前帧（可选）
        cv2.imshow('Frame', frame)
        cv2.imwrite(f"{output_path}/frame_{frame_count:04d}.jpg", frame)
        print(f"正在处理第 {frame_count} 帧")
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"总共处理了 {frame_count} 帧")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="输出视频每一帧")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("--output", "-o", default="./output", help="输出文件路径")
    args = parser.parse_args()
    
    show_video_frames(args.video, args.output)