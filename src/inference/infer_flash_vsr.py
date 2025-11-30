#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from typing import List, Tuple, Union

import cv2
import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from diffsynth import FlashVSRTinyLongPipeline, FlashVSRTinyPipeline, ModelManager

from ..utils.TCDecoder import build_tcdecoder
from ..utils.utils import Causal_LQ4x_Proj
from ..utils.video_saver import save_video

MODEL_FOLDER = os.path.join("examples", "WanVSR", "FlashVSR-v1.1")


def tensor2video(frames: torch.Tensor) -> List[Image.Image]:
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def natural_key(name: str) -> List[Union[int, str]]:
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"([0-9]+)", os.path.basename(name))
    ]


def list_images_natural(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def largest_8n1_leq(n: int) -> int:  # 8n+1
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def next_8n5(n: int) -> int:  # next 8n+5
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5


def is_video(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(
        (".mp4", ".mov", ".avi", ".mkv")
    )


def compute_scaled_and_target_dims(
    w0: int, h0: int, scale: float = 4.0, multiple: int = 128
) -> Tuple[int, int, int, int]:
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def prepare_input_cpu(
    frames: List[Image.Image],
    sW: int,
    sH: int,
    tW: int,
    tH: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Returns frames in a tensor of shape (1, C, F, H, W) ([-1, 1] normalized)"""
    frames_np = [np.array(frame) for frame in frames]  # (H, W, C)

    F = len(frames_np)
    C = frames_np[0].shape[-1]

    # Preallocate output array (F, sH, sW, C)
    resized_frames = np.empty((F, sH, sW, C), dtype=np.float32)

    # Vectorized for-loop (fastest option with OpenCV)
    for i, arr in enumerate(frames_np):
        resized_frames[i] = cv2.resize(arr, (sW, sH), interpolation=cv2.INTER_CUBIC)

    frames_np = None

    # Apply center crop (sH → tH, sW → tW)
    if sH != tH or sW != tW:
        top = (sH - tH) // 2
        left = (sW - tW) // 2
        resized_frames = resized_frames[:, top : top + tH, left : left + tW]

    # Convert to torch (F, C, H, W)
    frames_tensor = (
        torch.from_numpy(resized_frames)
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device="cpu", dtype=torch.float32, non_blocking=True)
    )

    resized_frames = None  # free memory

    # Normalize to [-1, 1]
    frames_tensor = frames_tensor / 255.0
    frames_tensor = (frames_tensor * 2.0 - 1.0).to(dtype=dtype)
    return frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)


def prepare_input_tensor(
    path: str,
    scale: float = 4,
    dtype=torch.bfloat16,
) -> Tuple[torch.Tensor, int, int, int, float]:
    if not os.path.isdir(path) and not is_video(path):
        raise ValueError(f"Unsupported input: {path}")

    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")

        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(
            f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {N0}"
        )

        sW, sH, tW, tH = compute_scaled_and_target_dims(
            w0, h0, scale=scale, multiple=128
        )
        print(
            f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}"
        )

        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(
                f"Not enough frames after padding in {path}. Got {len(paths)}."
            )
        paths = paths[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F - 4}")

        fps = 30

    elif is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert("RGB")
        w0, h0 = first.size

        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get("fps", 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get("nframes", None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        r.get_data(n)
                        n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        print(
            f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}"
        )

        sW, sH, tW, tH = compute_scaled_and_target_dims(
            w0, h0, scale=scale, multiple=128
        )
        print(
            f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}"
        )

        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(
                f"Not enough frames after padding in {path}. Got {len(idx)}."
            )
        idx = idx[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F - 4}")

    frames_cpu = prepare_input_cpu(
        [Image.open(p).convert("RGB") for p in paths],
        sW=sW,
        sH=sH,
        tW=tW,
        tH=tH,
        dtype=dtype,
    )

    return frames_cpu, tH, tW, F, fps


def init_pipeline(long_vid_pipeline: bool = False):
    print(
        torch.cuda.current_device(),
        torch.cuda.get_device_name(torch.cuda.current_device()),
    )
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models(
        [
            os.path.join(
                MODEL_FOLDER,
                "diffusion_pytorch_model_streaming_dmd.safetensors",
            ),
        ]
    )
    if long_vid_pipeline:
        pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device="cuda")
    else:
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
        in_dim=3, out_dim=1536, layer_num=1
    ).to("cuda", dtype=torch.bfloat16)
    LQ_proj_in_path = os.path.join(MODEL_FOLDER, "LQ_proj_in.ckpt")
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(
            torch.load(LQ_proj_in_path, map_location="cpu"), strict=True
        )
    pipe.denoising_model().LQ_proj_in.to("cuda")

    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(
        new_channels=multi_scale_channels, new_latent_channels=16 + 768
    )
    mis = pipe.TCDecoder.load_state_dict(
        torch.load(os.path.join(MODEL_FOLDER, "TCDecoder.ckpt")),
        strict=False,
    )
    print(mis)

    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def inference_pipeline(
    frames: List[Image.Image],
    long_vid_pipeline: bool = False,
    scale: float = 4.0,
    seed: int = 0,
    sparse_ratio: float = 2.0,
    local_range: int = 11,
    pipe=None,
) -> List[Image.Image]:
    if pipe is None:
        pipe = init_pipeline(long_vid_pipeline=long_vid_pipeline)

    total_frames = len(frames)

    w0, h0 = frames[0].size
    print(f"Original Resolution: {w0}x{h0} | Original Frames: {total_frames}")

    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
    print(f"Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

    # (1, C, F, H, W)
    frames_cpu = prepare_input_cpu(frames=frames, sW=sW, sH=sH, tW=tW, tH=tH)
    del frames

    # We will it so that chunks have the same nb of frames by padding the last one
    n_pad_frames = next_8n5(total_frames) - total_frames
    if n_pad_frames != 0:
        print(f"In order not to drop any frames, padding with {n_pad_frames}.")

    padding_frames = frames_cpu[:, :, -1:, :, :].repeat(1, 1, n_pad_frames, 1, 1)
    frames_cpu = torch.cat([frames_cpu, padding_frames], dim=2)

    F = largest_8n1_leq(total_frames + n_pad_frames + 4)

    # Add 4 padding frames
    LQ = torch.cat([frames_cpu, frames_cpu.repeat(1, 1, 4, 1, 1)], dim=2).to(
        device="cuda"
    )

    video = pipe(
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=seed,
        LQ_video=LQ,
        num_frames=F,
        height=tH,
        width=tW,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=sparse_ratio * 768 * 1280 / (tH * tW),
        kv_ratio=3.0,
        local_range=local_range,
        color_fix=True,
    )

    # Convert tensor to frames
    frames_out = tensor2video(video)
    print(f"{len(frames_out)=}")

    # Remove potential padded frames
    if n_pad_frames > 0:
        frames_out = frames_out[:-n_pad_frames]
        print(f"after removal, {len(frames_out)=}")

    del video, LQ
    torch.cuda.empty_cache()

    return frames_out


def main():
    RESULT_ROOT = "./results"
    os.makedirs(RESULT_ROOT, exist_ok=True)
    inputs = [
        "./inputs/example0.mp4",
        "./inputs/example1.mp4",
        "./inputs/example2.mp4",
        "./inputs/example3.mp4",
    ]
    seed, scale, dtype, device = 0, 4.0, torch.bfloat16, "cuda"
    sparse_ratio = 2.0  # Recommended: 1.5 or 2.0. 1.5 → faster; 2.0 → more stable.
    pipe = init_pipeline()

    for p in inputs:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        name = os.path.basename(p.rstrip("/"))
        if name.startswith("."):
            continue
        try:
            LQ, th, tw, F, fps = prepare_input_tensor(
                p, scale=scale, dtype=dtype, device=device
            )
        except Exception as e:
            print(f"[Error] {name}: {e}")
            continue

        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=seed,
            LQ_video=LQ,
            num_frames=F,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=sparse_ratio * 768 * 1280 / (th * tw),
            kv_ratio=3.0,
            local_range=11,  # Recommended: 9 or 11. local_range=9 → sharper details; 11 → more stable results.
            color_fix=True,
        )
        video = tensor2video(video)
        save_video(
            video,
            os.path.join(
                RESULT_ROOT, f"FlashVSR_v1.1_Tiny_{name.split('.')[0]}_seed{seed}.mp4"
            ),
            fps=fps,
            quality=6,
        )

    print("Done.")


if __name__ == "__main__":
    main()
