import io
import os
from typing import List

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from src.inference.infer_flash_vsr import inference_pipeline
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

# # Global variable to store the pipeline
# pipeline_cache = None


# def get_pipeline():
#     global pipeline_cache
#     if pipeline_cache is None:
#         pipeline_cache = init_pipeline()
#     return pipeline_cache


# @app.post("/process-video/")
# async def process_video(
#     file: UploadFile = File(...),
#     scale: float = 4.0,
#     seed: int = 0,
#     sparse_ratio: float = 2.0,
#     local_range: int = 11,
# ):
#     # Create a temporary directory to store the uploaded file
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Save the uploaded file to the temporary directory
#         file_path = os.path.join(temp_dir, file.filename)
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         # Prepare the input tensor
#         LQ, th, tw, F, fps = prepare_input_tensor(
#             file_path, scale=scale, dtype=torch.bfloat16, device="cuda"
#         )

#     # Get the pipeline
#     pipe = get_pipeline()

#     # Process the video
#     video = pipe(
#         prompt="",
#         negative_prompt="",
#         cfg_scale=1.0,
#         num_inference_steps=1,
#         seed=seed,
#         LQ_video=LQ,
#         num_frames=F,
#         height=th,
#         width=tw,
#         is_full_block=False,
#         if_buffer=True,
#         topk_ratio=sparse_ratio * 768 * 1280 / (th * tw),
#         kv_ratio=3.0,
#         local_range=local_range,
#         color_fix=True,
#     )

#     # Convert the tensor to video frames
#     video_frames = tensor2video(video)

#     # Create a temporary file for the output video
#     output_path = os.path.join(RESULTS_DIR, f"processed_{file.filename}")

#     # Save the video
#     save_video(video_frames, output_path, fps=fps, quality=6)

#     # Return the processed video
#     return FileResponse(
#         output_path,
#         media_type="video/mp4",
#         filename=f"processed_{file.filename}",
#     )


@app.post("/process-video-chunked/")
async def process_video_chunked(
    file: UploadFile = File(...),
    long_vid_pipeline: bool = False,
    temp_chunk_size: int = 133,
    temp_overlap: int = 8,
    scale: float = 4.0,
    seed: int = 0,
    sparse_ratio: float = 2.0,
    local_range: int = 11,
):
    if USE_FFMPEG_SAVE:
        init_ffmpeg_encoder_global(encoder=FFMPEG_ENCODER)

    # Read entire uploaded file into memory buffer
    file_bytes = await file.read()

    # Open video using PyAV
    container = av.open(io.BytesIO(file_bytes))

    # Access the first video stream
    video_stream = container.streams.video[0]

    # Get FPS
    fps = round(
        float(video_stream.average_rate)
        if video_stream.average_rate
        else float(video_stream.rate)
    )

    # Convert AVFrame -> RGB PIL Image
    frames: List[Image.Image] = [
        frame.to_image() for frame in container.decode(video=0)
    ]

    output_frames = inference_pipeline(
        frames=frames,
        long_vid_pipeline=long_vid_pipeline,
        temp_chunk_size=temp_chunk_size,
        temp_overlap=temp_overlap,
        scale=scale,
        seed=seed,
        sparse_ratio=sparse_ratio,
        local_range=local_range,
    )

    # --- Save final output video ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"processed_{file.filename}")

    video_array = np.stack([np.array(img) for img in output_frames], axis=0)

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

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"processed_{file.filename}",
    )


@app.get("/")
async def root():
    return {"message": "Video Processing API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
