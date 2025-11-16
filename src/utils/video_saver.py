import os
import shutil
import subprocess
from typing import Dict, Optional, Tuple

import imageio
import numpy as np
import torch
from tqdm import tqdm

_FFMPEG_ENCODER_CACHE: Dict[Tuple[str, str], bool] = {}
FFMPEG_ENCODER_RESOLVED: Optional[str] = None


def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer_kwargs = dict(fps=fps, quality=quality)
    try:
        w = imageio.get_writer(save_path, macro_block_size=None, **writer_kwargs)
    except TypeError:
        w = imageio.get_writer(save_path, **writer_kwargs)
    iterable = frames if isinstance(frames, (list, tuple)) else frames
    if isinstance(frames, np.ndarray):
        for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
            w.append_data(f)
    else:
        for f in tqdm(iterable, desc=f"Saving {os.path.basename(save_path)}"):
            w.append_data(np.array(f))
    w.close()


def _resolve_ffmpeg_path() -> str:
    for name in ("ffmpeg", "ffmpeg.exe"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError("未在 PATH 中找到 ffmpeg，可安装后再启用 --fast-video-save")


def _ffmpeg_supports_encoder(ffmpeg_bin: str, encoder: str) -> bool:
    key = (ffmpeg_bin, encoder)
    if key in _FFMPEG_ENCODER_CACHE:
        return _FFMPEG_ENCODER_CACHE[key]
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=2x2:d=0.1",
        "-frames:v",
        "1",
        "-c:v",
        encoder,
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        support = False
    else:
        support = result.returncode == 0
    _FFMPEG_ENCODER_CACHE[key] = support
    return support


def _select_ffmpeg_encoder(ffmpeg_bin: str, encoder_hint: str) -> str:
    global FFMPEG_ENCODER_RESOLVED
    if encoder_hint != "auto":
        if not _ffmpeg_supports_encoder(ffmpeg_bin, encoder_hint):
            raise RuntimeError(f"ffmpeg 不支持编码器 {encoder_hint}")
        FFMPEG_ENCODER_RESOLVED = encoder_hint
        return encoder_hint

    if FFMPEG_ENCODER_RESOLVED:
        return FFMPEG_ENCODER_RESOLVED

    candidates = []
    if torch.cuda.is_available():
        candidates.extend(["h264_nvenc", "hevc_nvenc"])
    candidates.extend(["libx264", "libx265"])
    for cand in candidates:
        if _ffmpeg_supports_encoder(ffmpeg_bin, cand):
            FFMPEG_ENCODER_RESOLVED = cand
            return cand
    raise RuntimeError(
        "无法找到 ffmpeg 支持的编码器，请在配置中手动指定 FFMPEG_ENCODER"
    )


def init_ffmpeg_encoder_global(encoder: str = "auto"):
    try:
        ffmpeg_bin = _resolve_ffmpeg_path()
        encoder_used = _select_ffmpeg_encoder(ffmpeg_bin, encoder)
    except Exception as exc:
        print(f"[Warn] FFmpeg 编码器预热失败: {exc}")
        return


def _quality_to_crf(q: int) -> int:
    q = int(round(q))
    q = max(1, min(10, q))
    crf = int(round(38 - 2.5 * q))
    return max(0, min(51, crf))


def _quality_to_nvenc_cq(q: int) -> int:
    q = int(round(q))
    q = max(1, min(10, q))
    cq = int(round(35 - 2.5 * q))
    return max(0, min(51, cq))


def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer_kwargs = dict(fps=fps, quality=quality)
    try:
        w = imageio.get_writer(save_path, macro_block_size=None, **writer_kwargs)
    except TypeError:
        w = imageio.get_writer(save_path, **writer_kwargs)
    iterable = frames if isinstance(frames, (list, tuple)) else frames
    if isinstance(frames, np.ndarray):
        for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
            w.append_data(f)
    else:
        for f in tqdm(iterable, desc=f"Saving {os.path.basename(save_path)}"):
            w.append_data(np.array(f))
    w.close()


def save_video_ffmpeg(
    frames,
    save_path: str,
    fps: int = 30,
    quality: int = 6,
    *,
    encoder: str = "auto",
    preset: Optional[str] = None,
    pix_fmt: Optional[str] = "yuv420p",
    threads: Optional[int] = None,
):
    if isinstance(frames, np.ndarray):
        if frames.size == 0:
            raise ValueError("save_video_ffmpeg 收到空帧数组")
    else:
        if not frames:
            raise ValueError("save_video_ffmpeg 收到空帧列表")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ffmpeg_bin = _resolve_ffmpeg_path()
    array_input = isinstance(frames, np.ndarray)
    if array_input:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("save_video_ffmpeg 仅支持 (T,H,W,3) 的 numpy 数组")
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        _, height, width, _ = frames.shape
    else:
        width, height = frames[0].size

    if not array_input:
        if any(f.size != frames[0].size for f in frames):
            raise ValueError("所有帧必须保持一致分辨率后再写入 ffmpeg")

    encoder_used = _select_ffmpeg_encoder(ffmpeg_bin, encoder)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if threads is not None and threads > 0:
        cmd += ["-threads", str(threads)]
    cmd += [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        encoder_used,
    ]

    if encoder_used.endswith("_nvenc"):
        cq = _quality_to_nvenc_cq(quality)
        cmd += ["-preset", preset or "p5", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
    elif encoder_used.startswith("libx26"):
        crf = _quality_to_crf(quality)
        cmd += ["-preset", preset or "medium", "-crf", str(crf)]
    elif preset:
        cmd += ["-preset", preset]

    if pix_fmt:
        cmd += ["-pix_fmt", pix_fmt]

    cmd.append(save_path)

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    error_message = ""
    try:
        if array_input:
            try:
                proc.stdin.write(frames.tobytes())
            except BrokenPipeError:
                pass
        else:
            for f in tqdm(frames, desc=f"FFmpeg 保存 {os.path.basename(save_path)}"):
                if f.mode != "RGB":
                    frame_rgb = f.convert("RGB")
                else:
                    frame_rgb = f
                frame_array = np.asarray(frame_rgb, dtype=np.uint8)
                try:
                    proc.stdin.write(frame_array.tobytes())
                except BrokenPipeError:
                    break
        proc.stdin.close()
        error_message = proc.stderr.read().decode("utf-8", errors="ignore")
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg 编码失败，返回码 {return_code}，信息：{error_message}"
            )
    finally:
        if proc.poll() is None:
            proc.kill()
        if proc.stderr:
            proc.stderr.close()
        if proc.stdin:
            proc.stdin.close()

    return encoder_used
