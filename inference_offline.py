"""
PersonaLive Offline Inference - 视频人物替换

使用示例：
  1. 基础用法（不分块，适合短视频 L<=300）：
     python inference_offline.py -L 300 --reference_image ref.jpg --driving_video drive.mp4 --name test

  2. 长视频分块模式（节省内存，适合 L>300）：
     python inference_offline.py -L 1000 --reference_image ref.jpg --driving_video drive.mp4 --name test --chunk

  3. 自定义块大小：
     python inference_offline.py -L 1000 --reference_image ref.jpg --driving_video drive.mp4 --name test --chunk --chunk_size 200

参数说明：
  -L                  : 生成帧数（默认100）
  --reference_image   : 参考人物图片（要替换成的人物）
  --driving_video     : 驱动视频（提供动作和表情）
  --name              : 输出文件夹名称
  --chunk             : 开启分块处理，降低内存占用（默认关闭）
  --chunk_size        : 每块帧数，仅 --chunk 开启时生效（默认100）
  --num_inference_steps : 推理步数（默认4）
  --temporal_adaptive_step : 时序自适应步数，需整除 num_inference_steps（默认4）
  -W, -H              : 输出视频宽高（默认512x512）
  --seed              : 随机种子（默认42）
  --device            : 运行设备（默认cuda）
  --stream_gen        : 使用流式生成策略降低显存（默认True）

输出位置：
  results/{日期}--{name}/concat_vid/{视频名}.mp4   (4行对比视频)
  results/{日期}--{name}/split_vid/{视频名}.mp4    (仅生成结果)
"""

import argparse
import os
import sys
from datetime import datetime
import mediapipe as mp
import numpy as np
import cv2
import torch
from skimage.transform import resize
from diffusers import AutoencoderKLTemporalDecoder, AutoencoderKL, AutoencoderTiny
from src.scheduler.scheduler_ddim import DDIMScheduler
import random
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline, Pose2VideoPipeline_Stream
from src.utils.util import save_videos_grid, crop_face, VideoStreamWriter
from decord import VideoReader
from diffusers.utils.import_utils import is_xformers_available

from src.models.motion_encoder.encoder import MotEncoder
from src.liveportrait.motion_extractor import MotionExtractor
from src.models.pose_guider import PoseGuider
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/prompts/personalive_offline.yaml')
    parser.add_argument("--name", type=str, default='personalive_offline')
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_xformers", type=bool, default=True)
    parser.add_argument("--stream_gen", type=bool, default=True, help='use streaming generation strategy to reduce VRAM usage.')
    parser.add_argument("--reference_image", type=str, default='', help='Path to reference image. If provided, overrides test_cases from config.')
    parser.add_argument("--driving_video", type=str, default='', help='Path to driving video. If provided, overrides test_cases from config.')
    parser.add_argument("--num_inference_steps", type=int, default=4, help='Number of denoising steps. Must be divisible by temporal_adaptive_step.')
    parser.add_argument("--temporal_adaptive_step", type=int, default=4, help='Temporal adaptive step. Must be a divisor of num_inference_steps.')
    parser.add_argument("--chunk", action='store_true', default=False, help='Enable chunk-based processing to reduce memory usage. Default: disabled (original behavior).')
    parser.add_argument("--chunk_size", type=int, default=100, help='Number of frames per chunk when --chunk is enabled. Default: 100.')
    args = parser.parse_args()

    return args

def main(args):
    device = args.device
    print('device', device)
    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(config.vae_path).to(device, dtype=weight_dtype)
    # if use tiny VAE
    # vae_tiny = AutoencoderTiny.from_pretrained(config.vae_tiny_path).to(device, dtype=weight_dtype)

    infer_config = OmegaConf.load(config.inference_config)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(device=device, dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    motion_encoder = MotEncoder().to(dtype=weight_dtype, device=device).eval()
    pose_guider = PoseGuider().to(device=device, dtype=weight_dtype)
    pose_encoder = MotionExtractor(num_kp=21).to(device=device, dtype=weight_dtype).eval()
    
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(
        OmegaConf.load(config.inference_config).noise_scheduler_kwargs
    )
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)
    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False
    )
    reference_unet.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'reference_unet'),
            map_location="cpu",
        ),
        strict=True,
    )
    motion_encoder.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'motion_encoder'),
            map_location="cpu",
        ),
        strict=True,
    )
    pose_guider.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'pose_guider'),
            map_location="cpu",
        ),
        strict=True,
    )
    denoising_unet.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'temporal_module'),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_encoder.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'motion_extractor'),
            map_location="cpu",
        ),
        strict=False,
    )
    
    if args.use_xformers:
        if is_xformers_available(): 
            try:
                reference_unet.enable_xformers_memory_efficient_attention()
                denoising_unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print("Failed to enable xformers:", e)
        else:
            print("xformers is not available. Make sure it is installed correctly.")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    if args.stream_gen:
        pipeline = Pose2VideoPipeline_Stream
    else:
        pipeline = Pose2VideoPipeline
    
    pipe = pipeline(
        vae=vae,
        # vae_tiny=vae_tiny,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        motion_encoder=motion_encoder,
        pose_encoder=pose_encoder,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)

    date_str = datetime.now().strftime("%Y%m%d")
    if args.name is None:
        time_str = datetime.now().strftime("%H%M")
        save_dir_name = f"{date_str}--{time_str}"
    else:
        save_dir_name = f"{date_str}--{args.name}"
    save_vid_dir = os.path.join('results', save_dir_name, 'concat_vid')
    os.makedirs(save_vid_dir, exist_ok=True)
    save_split_vid_dir = os.path.join('results', save_dir_name, 'split_vid')
    os.makedirs(save_split_vid_dir, exist_ok=True)

    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )

    if args.reference_image and args.driving_video:
        args.test_cases = {args.reference_image: [args.driving_video]}
    else:
        args.test_cases = OmegaConf.load(args.config)["test_cases"]

    for ref_image_path in list(args.test_cases.keys()):
        for pose_video_path in args.test_cases[ref_image_path]:
            video_name = os.path.basename(pose_video_path).split(".")[0]
            source_name = os.path.basename(ref_image_path).split(".")[0]

            vid_name = f"{source_name}_{video_name}.mp4"
            save_vid_path = os.path.join(save_vid_dir, vid_name)
            print(save_vid_path)
            if os.path.exists(save_vid_path):
                continue

            if ref_image_path.endswith('.mp4'):
                src_vid = VideoReader(ref_image_path)
                ref_img = src_vid[0].asnumpy()
                ref_img = Image.fromarray(ref_img).convert("RGB")
            else:
                ref_img = Image.open(ref_image_path).convert("RGB")

            control = VideoReader(pose_video_path)
            total_frames = min(len(control), args.L)
            total_frames = (total_frames // 4) * 4  # 确保是4的倍数

            ref_image_pil = ref_img.copy()
            ref_patch = crop_face(ref_image_pil, face_mesh)
            ref_face_pil = Image.fromarray(ref_patch).convert("RGB")

            size = args.H
            generator = torch.Generator(device=device)
            generator.manual_seed(42)

            save_vid_path = os.path.join(save_vid_dir, vid_name)
            save_split_vid_path = save_vid_path.replace(save_vid_dir, save_split_vid_dir)

            if args.chunk:
                # 分块模式：逐块处理，节省内存
                CHUNK_SIZE = args.chunk_size
                num_chunks = (total_frames + CHUNK_SIZE - 1) // CHUNK_SIZE
                print(f"Total frames: {total_frames}, processing in {num_chunks} chunks of size {CHUNK_SIZE}")

                video_writer = VideoStreamWriter(save_split_vid_path, fps=25, width=512, height=512, crf=18)
                concat_writer = VideoStreamWriter(save_vid_path, fps=25, width=512, height=512*4, crf=18)

                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * CHUNK_SIZE
                    chunk_end = min(chunk_start + CHUNK_SIZE, total_frames)
                    chunk_indices = list(range(chunk_start, chunk_end))
                    chunk_frames = control.get_batch(chunk_indices).asnumpy()
                    chunk_length = len(chunk_frames)
                    print(f"Processing chunk {chunk_idx+1}/{num_chunks}: frames {chunk_start}-{chunk_end-1}")

                    dri_faces = []
                    ori_pose_images = []
                    for idx_control, pose_image_pil in tqdm(enumerate(chunk_frames), total=chunk_length, desc='cropping faces'):
                        pose_image_pil = Image.fromarray(pose_image_pil).convert("RGB")
                        ori_pose_images.append(pose_image_pil)
                        dri_face = crop_face(pose_image_pil, face_mesh)
                        dri_face_pil = Image.fromarray(dri_face).convert("RGB")
                        dri_faces.append(dri_face_pil)

                    face_tensor_list = []
                    ori_pose_tensor_list = []
                    ref_tensor_list = []

                    for idx, pose_image_pil in enumerate(ori_pose_images):
                        face_tensor_list.append(pose_transform(dri_faces[idx]))
                        ori_pose_tensor_list.append(pose_transform(pose_image_pil))
                        ref_tensor_list.append(pose_transform(ref_image_pil))

                    ref_tensor = torch.stack(ref_tensor_list, dim=0)
                    ref_tensor = ref_tensor.transpose(0, 1).unsqueeze(0)

                    face_tensor = torch.stack(face_tensor_list, dim=0)
                    face_tensor = face_tensor.transpose(0, 1).unsqueeze(0)

                    ori_pose_tensor = torch.stack(ori_pose_tensor_list, dim=0)
                    ori_pose_tensor = ori_pose_tensor.transpose(0, 1).unsqueeze(0)

                    chunk_length_aligned = (chunk_length // 4) * 4
                    if chunk_length_aligned == 0:
                        continue

                    gen_video = pipe(
                        ori_pose_images[:chunk_length_aligned],
                        ref_image_pil,
                        dri_faces[:chunk_length_aligned],
                        ref_face_pil,
                        width,
                        height,
                        chunk_length_aligned,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=1.0,
                        generator=generator,
                        temporal_window_size = 4,
                        temporal_adaptive_step = args.temporal_adaptive_step,
                    ).videos

                    for frame_idx in range(gen_video.shape[2]):
                        gen_frame_tensor = gen_video[0, :, frame_idx, :, :]
                        ref_frame = ref_tensor[0, :, frame_idx, :, :] if frame_idx < ref_tensor.shape[2] else ref_tensor[0, :, -1, :, :]
                        face_frame = face_tensor[0, :, frame_idx, :, :] if frame_idx < face_tensor.shape[2] else face_tensor[0, :, -1, :, :]
                        pose_frame = ori_pose_tensor[0, :, frame_idx, :, :] if frame_idx < ori_pose_tensor.shape[2] else ori_pose_tensor[0, :, -1, :, :]
                        concat_frame = torch.cat([ref_frame, face_frame, pose_frame, gen_frame_tensor], dim=1)
                        concat_frame = concat_frame.transpose(0, 1).transpose(1, 2)
                        concat_frame = (concat_frame * 255).clamp(0, 255).byte().cpu().numpy()
                        concat_writer.write_frame(Image.fromarray(concat_frame))
                        gen_frame = gen_frame_tensor.transpose(0, 1).transpose(1, 2)
                        gen_frame = (gen_frame * 255).clamp(0, 255).byte().cpu().numpy()
                        video_writer.write_frame(Image.fromarray(gen_frame))

                    del gen_video, ref_tensor, face_tensor, ori_pose_tensor
                    del dri_faces, ori_pose_images, face_tensor_list, ori_pose_tensor_list, ref_tensor_list
                    torch.cuda.empty_cache()

                video_writer.close(audio_source=pose_video_path)
                concat_writer.close()

            else:
                # 原始模式：一次性处理所有帧
                print(f"Total frames: {total_frames}, processing all at once (original mode)")

                dri_faces = []
                ori_pose_images = []
                all_frames = control.get_batch(list(range(total_frames))).asnumpy()
                for idx_control, pose_image_pil in tqdm(enumerate(all_frames), total=total_frames, desc='cropping faces'):
                    pose_image_pil = Image.fromarray(pose_image_pil).convert("RGB")
                    ori_pose_images.append(pose_image_pil)
                    dri_face = crop_face(pose_image_pil, face_mesh)
                    dri_face_pil = Image.fromarray(dri_face).convert("RGB")
                    dri_faces.append(dri_face_pil)

                face_tensor_list = []
                ori_pose_tensor_list = []
                ref_tensor_list = []

                for idx, pose_image_pil in enumerate(ori_pose_images):
                    face_tensor_list.append(pose_transform(dri_faces[idx]))
                    ori_pose_tensor_list.append(pose_transform(pose_image_pil))
                    ref_tensor_list.append(pose_transform(ref_image_pil))

                ref_tensor = torch.stack(ref_tensor_list, dim=0)
                ref_tensor = ref_tensor.transpose(0, 1).unsqueeze(0)

                face_tensor = torch.stack(face_tensor_list, dim=0)
                face_tensor = face_tensor.transpose(0, 1).unsqueeze(0)

                ori_pose_tensor = torch.stack(ori_pose_tensor_list, dim=0)
                ori_pose_tensor = ori_pose_tensor.transpose(0, 1).unsqueeze(0)

                total_frames_aligned = (total_frames // 4) * 4

                gen_video = pipe(
                    ori_pose_images[:total_frames_aligned],
                    ref_image_pil,
                    dri_faces[:total_frames_aligned],
                    ref_face_pil,
                    width,
                    height,
                    total_frames_aligned,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=1.0,
                    generator=generator,
                    temporal_window_size = 4,
                    temporal_adaptive_step = args.temporal_adaptive_step,
                ).videos

                # 使用 save_videos_grid 保存（原始方式）
                from src.utils.util import save_videos_grid
                save_videos_grid(gen_video, save_split_vid_path, fps=25)

                # 保存 concat 视频
                concat_frames = []
                for frame_idx in range(gen_video.shape[2]):
                    gen_frame_tensor = gen_video[0, :, frame_idx, :, :]
                    ref_frame = ref_tensor[0, :, frame_idx, :, :] if frame_idx < ref_tensor.shape[2] else ref_tensor[0, :, -1, :, :]
                    face_frame = face_tensor[0, :, frame_idx, :, :] if frame_idx < face_tensor.shape[2] else face_tensor[0, :, -1, :, :]
                    pose_frame = ori_pose_tensor[0, :, frame_idx, :, :] if frame_idx < ori_pose_tensor.shape[2] else ori_pose_tensor[0, :, -1, :, :]
                    concat_frame = torch.cat([ref_frame, face_frame, pose_frame, gen_frame_tensor], dim=1)
                    concat_frames.append(concat_frame)

                concat_video = torch.stack(concat_frames, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)
                save_videos_grid(concat_video, save_vid_path, fps=25)

            print(f"Video saved to {save_split_vid_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
