import cv2
import numpy as np
import os
import argparse
from PIL import Image

def convert_4row_to_2x2(input_path, output_path, format='auto', fps=None, max_frames=None, gif_half_res=False):
    """
    将4行垂直拼接的视频转换为2x2网格的正方形视频或GIF
    
    参数:
        input_path: 输入视频路径
        output_path: 输出文件路径
        format: 输出格式 ('mp4', 'gif', 'auto'-根据扩展名自动判断)
        fps: 输出帧率 (None表示使用原视频帧率)
        max_frames: 最大处理帧数 (None表示处理全部)
        gif_half_res: GIF输出时是否降低分辨率为一半
    """
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return
    
    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 读取第一帧获取尺寸
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        return
    
    h, w = frame.shape[:2]
    print(f"Input video: {w}x{h}, FPS: {video_fps}, Frames: {total_frames}")
    
    # 每行的高度（4行）
    row_height = h // 4
    # 2x2 输出尺寸
    out_w = w * 2
    out_h = row_height * 2
    
    # 确定输出格式
    if format == 'auto':
        is_gif = output_path.lower().endswith('.gif')
    elif format == 'gif':
        is_gif = True
    else:
        is_gif = False
    
    # GIF 降低分辨率
    if is_gif and gif_half_res:
        out_w = out_w // 2
        out_h = out_h // 2
        print(f"GIF half resolution enabled: {w*2}x{row_height*2} -> {out_w}x{out_h}")
    
    # 确定输出FPS
    if fps is None:
        fps = video_fps
    
    # 确定处理帧数
    if max_frames is not None:
        process_frames = min(max_frames, total_frames)
    else:
        process_frames = total_frames
    
    print(f"Output: {out_w}x{out_h}, FPS: {fps}, Frames: {process_frames}, Format: {'GIF' if is_gif else 'MP4'}")
    
    if is_gif:
        # GIF 输出
        frames = []
        for i in range(process_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 分割4行
            row0 = frame[0:row_height, :]
            row1 = frame[row_height:row_height*2, :]
            row2 = frame[row_height*2:row_height*3, :]
            row3 = frame[row_height*3:row_height*4, :]
            
            # 拼接成2x2网格
            top = np.hstack([row0, row1])
            bottom = np.hstack([row2, row3])
            out_frame = np.vstack([top, bottom])
            
            # 如果需要降低分辨率
            if gif_half_res:
                out_frame = cv2.resize(out_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            
            # 转换为 PIL Image
            out_frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(out_frame_rgb)
            frames.append(pil_img)
            
            if i % 50 == 0:
                print(f"Processing: {i}/{process_frames}")
        
        # 保存 GIF
        if frames:
            duration = 1000 / fps  # 每帧持续时间（毫秒）
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                optimize=True
            )
            print(f"Done! GIF saved to {output_path}")
    else:
        # MP4 输出
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        
        # 重置到开头
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for i in range(process_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 分割4行
            row0 = frame[0:row_height, :]
            row1 = frame[row_height:row_height*2, :]
            row2 = frame[row_height*2:row_height*3, :]
            row3 = frame[row_height*3:row_height*4, :]
            
            # 拼接成2x2网格
            top = np.hstack([row0, row1])
            bottom = np.hstack([row2, row3])
            out_frame = np.vstack([top, bottom])
            
            out.write(out_frame)
            
            if i % 50 == 0:
                print(f"Processing: {i}/{process_frames}")
        
        out.release()
        print(f"Done! Video saved to {output_path}")
    
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert 4-row vertical video to 2x2 grid')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input video path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file path')
    parser.add_argument('-f', '--format', type=str, default='auto', choices=['mp4', 'gif', 'auto'],
                        help='Output format (default: auto based on extension)')
    parser.add_argument('--fps', type=float, default=None, help='Output FPS (default: same as input)')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum frames to process')
    parser.add_argument('--gif-half-res', action='store_true', default=True,
                        help='Reduce GIF resolution to half size (default: True)')
    
    args = parser.parse_args()
    
    convert_4row_to_2x2(
        input_path=args.input,
        output_path=args.output,
        format=args.format,
        fps=args.fps,
        max_frames=args.max_frames,
        gif_half_res=args.gif_half_res
    )
