import io
from threading import Lock
from typing import List

import av
import uvicorn
from fastapi import FastAPI, File, Response, UploadFile
from PIL import Image

from src.inference.infer_flash_vsr import inference_pipeline, init_pipeline
from src.utils.video_bytes_conv import save_frames_to_video_bytes

app = FastAPI()


# Cache the pipeline globally
_cached_pipe = None
_pipeline_lock = Lock()


def get_pipeline(long_vid_pipeline: bool = False):
    global _cached_pipe
    if _cached_pipe is None:
        with _pipeline_lock:
            if _cached_pipe is None:
                print("[Worker] Initializing pipeline (first request)...")
                _cached_pipe = init_pipeline(long_vid_pipeline=long_vid_pipeline)
                print("[Worker] Pipeline initialized and cached")
    return _cached_pipe


# Do not pass sync as LQ cache will be shared between threads...
@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    long_vid_pipeline: bool = False,
    scale: float = 4.0,
    seed: int = 0,
    sparse_ratio: float = 2.0,
    local_range: int = 11,
):
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
        scale=scale,
        seed=seed,
        sparse_ratio=sparse_ratio,
        local_range=local_range,
        pipe=get_pipeline(long_vid_pipeline=long_vid_pipeline),
    )

    # Return the processed video
    return Response(
        content=save_frames_to_video_bytes(frames=output_frames, fps=fps).getvalue(),
        media_type="video/mp4",
    )


@app.get("/")
async def root():
    return {"message": "Video Processing API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
