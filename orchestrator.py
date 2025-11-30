import asyncio
import io
import math
import os
from typing import Any, Dict, List

import aiohttp
import av
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from src.inference.infer_flash_vsr import next_8n5
from src.utils.tile_and_chunk_utils import (
    calculate_tile_coords,
    crop_frames_to_tile,
    stitch_tiles_spatially,
    stitch_tiles_temporally,
)
from src.utils.video_bytes_conv import load_video_from_bytes, save_frames_to_video_bytes
from src.utils.video_saver import (
    init_ffmpeg_encoder_global,
    save_video,
    save_video_ffmpeg,
)

RESULTS_DIR = "results"

# Changed from https://github.com/OpenImagingLab/FlashVSR/pull/43/commits/d2d0f2aed3f26dc710d36c687d039ee2c9781c52
USE_FFMPEG_SAVE = True
FFMPEG_ENCODER = "auto"  # auto / h264_nvenc / hevc_nvenc / libx264 ...
FFMPEG_PRESET = None  # None 表示使用对应编码器的默认 preset
FFMPEG_PIX_FMT = "yuv420p"  # 常用输出像素格式，兼容性较好
FFMPEG_THREADS = None  # None 表示由 ffmpeg 自行调度线程

app = FastAPI()

# Configuration: List of worker URLs
WORKER_URLS = [
    "http://localhost:8001",
    # Add more workers as needed:
    # "http://192.168.1.10:8001",
    # "http://192.168.1.11:8001",
]


async def send_chunk_to_worker(
    worker_url: str,
    chunk_frames: List[Image.Image],
    fps: int,
    scale: float,
    seed: int,
    sparse_ratio: float,
    local_range: int,
    chunk_id: str,
) -> bytes:
    """Send a chunk to a worker and get the processed result."""
    print(f"[Orchestrator] Sending {chunk_id} to {worker_url}")

    # Convert frames to video bytes
    video_buffer = save_frames_to_video_bytes(chunk_frames, fps)

    # Send to worker
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field(
            "file",
            video_buffer,
            filename=f"{chunk_id}.mp4",
            content_type="video/mp4",
        )

        params = {
            "scale": scale,
            "seed": seed,
            "sparse_ratio": sparse_ratio,
            "local_range": local_range,
        }

        async with session.post(
            f"{worker_url}/process-video", data=data, params=params
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Worker {worker_url} returned status {response.status}"
                )
            result_bytes = await response.read()

    print(f"[Orchestrator] Received processed {chunk_id} from {worker_url}")
    return result_bytes


@app.post("/process-video-distributed/")
async def process_video_distributed(
    file: UploadFile = File(...),
    use_tiling: bool = False,
    tile_size: int = 256,
    tile_overlap: int = 32,
    temp_chunk_size: int = 133,
    temp_overlap: int = 8,
    scale: float = 4.0,
    seed: int = 0,
    sparse_ratio: float = 2.0,
    local_range: int = 11,
):
    """
    Orchestrator endpoint that distributes video processing across multiple workers.

    Args:
        file: Input video file
        use_tiling: Whether to split spatially into tiles
        tile_size: Size of each spatial tile (original resolution)
        tile_overlap: Overlap between spatial tiles (original resolution)
        temp_chunk_size: Number of frames per temporal chunk
        temp_overlap: Overlap between temporal chunks
        scale: Upscaling factor
        seed: Random seed
        sparse_ratio: Sparse attention ratio
        local_range: Local attention range
    """
    print("[Orchestrator] Starting distributed processing")
    if USE_FFMPEG_SAVE:
        init_ffmpeg_encoder_global(encoder=FFMPEG_ENCODER)

    # Read the uploaded video
    file_bytes = await file.read()
    container = av.open(io.BytesIO(file_bytes))
    video_stream = container.streams.video[0]

    # Get FPS
    fps = round(
        float(video_stream.average_rate)
        if video_stream.average_rate
        else float(video_stream.rate)
    )

    # Extract all frames
    frames: List[Image.Image] = [
        frame.to_image() for frame in container.decode(video=0)
    ]
    container.close()

    total_frames = len(frames)
    original_width, original_height = frames[0].size

    print(
        f"[Orchestrator] Video info: {total_frames} frames, {original_width}x{original_height}, {fps} fps"
    )

    # Optimize temp_chunk_size to 8n+5
    opt_temp_chunk_size = next_8n5(temp_chunk_size)
    if opt_temp_chunk_size != temp_chunk_size:
        print(
            f"[Orchestrator] Adjusted temp_chunk_size from {temp_chunk_size} to {opt_temp_chunk_size}"
        )
        temp_chunk_size = opt_temp_chunk_size

    # Step 1: Determine spatial tiles
    if use_tiling:
        tile_coords = calculate_tile_coords(
            original_height, original_width, tile_size, tile_overlap
        )
        print(f"[Orchestrator] Created {len(tile_coords)} spatial tiles")
    else:
        # No tiling: process entire frame
        tile_coords = [(0, 0, original_width, original_height)]
        print("[Orchestrator] No spatial tiling")

    # Step 2: Determine temporal chunks
    num_temp_chunks = (
        math.ceil((total_frames - temp_chunk_size) / (temp_chunk_size - temp_overlap))
        + 1
    )
    print(f"[Orchestrator] Created {num_temp_chunks} temporal chunks")

    # Step 3: Create all jobs (tile x temporal_chunk combinations)
    jobs: List[Dict[str, Any]] = []
    for tile_idx, (x1, y1, x2, y2) in enumerate(tile_coords):
        # Crop all frames to this tile
        tile_frames = crop_frames_to_tile(frames, x1, y1, x2, y2)

        for chunk_idx in range(num_temp_chunks):
            start_frame = chunk_idx * (temp_chunk_size - temp_overlap)
            end_frame = start_frame + temp_chunk_size

            chunk_frames = tile_frames[start_frame:end_frame]

            job_id = f"tile{tile_idx}_chunk{chunk_idx}"
            jobs.append(
                {
                    "tile_idx": tile_idx,
                    "chunk_idx": chunk_idx,
                    "chunk_frames": chunk_frames,
                    "job_id": job_id,
                }
            )

    print(f"[Orchestrator] Total jobs: {len(jobs)}")

    # Step 4: Distribute jobs to workers in parallel
    results: Dict[int, Dict[int, Any]] = {}
    worker_idx = 0
    max_concurrent = 5 * len(WORKER_URLS)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_job(job):
        nonlocal worker_idx
        async with semaphore:
            # Round-robin worker selection
            worker_url = WORKER_URLS[worker_idx % len(WORKER_URLS)]
            worker_idx += 1

            result_bytes = await send_chunk_to_worker(
                worker_url=worker_url,
                chunk_frames=job["chunk_frames"],
                fps=fps,
                scale=scale,
                seed=seed,
                sparse_ratio=sparse_ratio,
                local_range=local_range,
                chunk_id=job["job_id"],
            )

            # Load the processed frames
            processed_frames = load_video_from_bytes(result_bytes)

            return (job["tile_idx"], job["chunk_idx"], processed_frames)

    # Run all jobs in parallel (limited by semaphore)
    print(
        f"[Orchestrator] Distributing jobs to workers (max {max_concurrent} concurrent)..."
    )
    job_results = await asyncio.gather(*[process_job(job) for job in jobs])

    # Organize results
    for tile_idx, chunk_idx, processed_frames in job_results:
        if tile_idx not in results:
            results[tile_idx] = {}
        results[tile_idx][chunk_idx] = processed_frames

    # Step 5: Stitch temporal chunks for each tile
    print("[Orchestrator] Stitching temporal chunks...")
    tile_results = []

    for tile_idx in sorted(results.keys()):
        tile_chunks = [
            results[tile_idx][chunk_idx]
            for chunk_idx in sorted(results[tile_idx].keys())
        ]

        # Stitch temporal chunks
        stitched_tile = stitch_tiles_temporally(tile_chunks, temp_overlap)

        tile_results.append(stitched_tile)

    # Step 6: Stitch spatial tiles
    if use_tiling:
        print("[Orchestrator] Stitching spatial tiles...")
        final_frames = stitch_tiles_spatially(
            tile_results,
            tile_coords,
            int(original_width * scale),
            int(original_height * scale),
            scale,
            tile_overlap,
        )
    else:
        final_frames = tile_results[0]

    print(f"[Orchestrator] Final output: {len(final_frames)} frames")

    # Step 7: Save final video
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"processed_{file.filename}")

    video_array = np.stack([np.array(img) for img in final_frames], axis=0)

    if USE_FFMPEG_SAVE:
        save_video_ffmpeg(
            video_array,
            output_path,
            fps=fps,
            quality=6,
            encoder=FFMPEG_ENCODER,
            preset=FFMPEG_PRESET,
            pix_fmt=FFMPEG_PIX_FMT,
            threads=FFMPEG_THREADS,
        )
    else:
        save_video(video_array, output_path, fps=fps, quality=6)

    print(f"[Orchestrator] Saved final video to {output_path}")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"processed_{file.filename}",
    )


@app.get("/")
async def root():
    return {
        "message": "FlashVSR Orchestrator",
        "workers": WORKER_URLS,
        "num_workers": len(WORKER_URLS),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
