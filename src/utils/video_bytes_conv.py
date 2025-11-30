import io
from typing import List

import av
import imageio.v3
import numpy as np
from PIL import Image


def save_frames_to_video_bytes(frames: List[Image.Image], fps: int) -> io.BytesIO:
    """Save frames to an in-memory video buffer."""

    buffer = io.BytesIO()
    video_array = np.stack([np.array(frame) for frame in frames], axis=0)
    imageio.v3.imwrite(buffer, video_array, extension=".mp4", codec="libx264", fps=fps)
    buffer.seek(0)
    return buffer


def load_video_from_bytes(video_bytes: bytes) -> List[Image.Image]:
    """Load video frames from bytes."""
    container = av.open(io.BytesIO(video_bytes))
    frames = [frame.to_image() for frame in container.decode(video=0)]
    container.close()
    return frames
