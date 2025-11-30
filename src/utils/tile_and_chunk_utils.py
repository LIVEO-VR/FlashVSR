import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def calculate_tile_coords(
    height: int, width: int, tile_size: int, overlap: int
) -> List[Tuple[int, int, int, int]]:
    """Calculate tile coordinates with overlap."""
    coords: List[Tuple[int, int, int, int]] = []

    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)

    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride

            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)

            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)

            coords.append((x1, y1, x2, y2))

    return coords


def crop_frames_to_tile(
    frames: List[Image.Image], x1: int, y1: int, x2: int, y2: int
) -> List[Image.Image]:
    """Crop all frames to a specific tile region."""
    return [frame.crop((x1, y1, x2, y2)) for frame in frames]


def create_feather_mask_numpy(
    size: Tuple[int, int], overlap: int
) -> NDArray[np.float32]:
    """Create a feathering mask for blending tiles."""
    H, W = size
    mask = np.ones((H, W, 1), dtype=np.float32)

    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)

        mask[:, :overlap, :] *= ramp[np.newaxis, :, np.newaxis]
        mask[:, -overlap:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]

        mask[:overlap, :, :] *= ramp[:, np.newaxis, np.newaxis]
        mask[-overlap:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]

    return mask


def stitch_tiles_temporally(
    tile_chunks: List[List[Image.Image]], temp_overlap: int
) -> List[Image.Image]:
    """Stitch temporal chunks together using blending."""
    if not tile_chunks:
        return []

    if len(tile_chunks) == 1:
        return tile_chunks[0]

    output_frames: List[Image.Image] = []

    for chunk_idx, chunk_frames in enumerate(tile_chunks):
        if chunk_idx == 0:
            # First chunk: add all frames
            output_frames.extend(chunk_frames)
        else:
            # Blend overlapping frames
            for f_idx in range(temp_overlap):
                if f_idx < len(output_frames) and f_idx < len(chunk_frames):
                    blend_factor = (f_idx + 1) / (temp_overlap + 1)
                    output_frames[-(temp_overlap - f_idx)] = Image.blend(
                        output_frames[-(temp_overlap - f_idx)],
                        chunk_frames[f_idx],
                        alpha=blend_factor,
                    )

            # Add non-overlapping frames
            output_frames.extend(chunk_frames[temp_overlap:])

    return output_frames


def stitch_tiles_spatially(
    tile_results: List[List[Image.Image]],
    tile_coords: List[Tuple[int, int, int, int]],
    final_width: int,
    final_height: int,
    scale: float,
    tile_overlap: int,
    num_workers: int | None = None,
) -> List[Image.Image]:
    """Stitch spatial tiles together using feathering (parallelized)."""
    if not tile_results:
        return []

    num_frames = len(tile_results[0])
    num_tiles = len(tile_results)

    # Calculate scaled overlap
    scaled_overlap = int(tile_overlap * scale)

    # Pre-convert all tile frames to numpy arrays (avoids repeated conversions)
    # Shape: tile_arrays[tile_idx][frame_idx] -> np.ndarray
    print(f"[Stitcher] Pre-converting {num_tiles} tiles x {num_frames} frames to numpy...")
    tile_arrays: List[List[NDArray[np.float32]]] = []
    tile_dims: List[Tuple[int, int]] = []  # (tile_h, tile_w) for each tile
    for tile_idx in range(num_tiles):
        tile_frames_np = [
            np.array(frame, dtype=np.float32) / 255.0
            for frame in tile_results[tile_idx]
        ]
        tile_arrays.append(tile_frames_np)
        tile_dims.append((tile_frames_np[0].shape[0], tile_frames_np[0].shape[1]))

    # Pre-compute feather masks for each unique tile size
    mask_cache: dict[Tuple[int, int], NDArray[np.float32]] = {}
    for tile_h, tile_w in tile_dims:
        if (tile_h, tile_w) not in mask_cache:
            mask_cache[(tile_h, tile_w)] = create_feather_mask_numpy(
                (tile_h, tile_w), scaled_overlap
            )

    def process_frame(frame_idx: int) -> Image.Image:
        canvas = np.zeros((final_height, final_width, 3), dtype=np.float32)
        weight_canvas = np.zeros((final_height, final_width, 3), dtype=np.float32)

        # Iterate over tile_coords directly (like original) to ensure correspondence
        for tile_idx, (x1, y1, _, _) in enumerate(tile_coords):
            tile_array = tile_arrays[tile_idx][frame_idx]
            tile_h, tile_w = tile_dims[tile_idx]
            mask = mask_cache[(tile_h, tile_w)]

            out_x1, out_y1 = int(x1 * scale), int(y1 * scale)
            out_x2, out_y2 = out_x1 + tile_w, out_y1 + tile_h

            canvas[out_y1:out_y2, out_x1:out_x2, :] += tile_array * mask
            weight_canvas[out_y1:out_y2, out_x1:out_x2, :] += mask

        # Normalize
        weight_canvas[weight_canvas == 0] = 1.0
        stitched = canvas / weight_canvas

        # Convert back to image
        stitched_uint8 = (np.clip(stitched, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(stitched_uint8)

    # Determine number of workers
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, num_frames)

    print(f"[Stitcher] Processing {num_frames} frames with {num_workers} workers...")

    # Process frames in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        output_frames = list(executor.map(process_frame, range(num_frames)))

    return output_frames
