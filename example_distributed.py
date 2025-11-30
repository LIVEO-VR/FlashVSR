#!/usr/bin/env python3
"""
Example script demonstrating distributed video processing with FlashVSR.

Usage:
    python example_distributed.py input_video.mp4 --workers 4
"""

import argparse
import requests
import sys


def process_video_distributed(
    input_path: str,
    output_path: str = None,
    orchestrator_url: str = "http://localhost:8000",
    use_tiling: bool = False,
    tile_size: int = 512,
    tile_overlap: int = 64,
    temp_chunk_size: int = 133,
    temp_overlap: int = 8,
    scale: float = 4.0,
    seed: int = 0,
    sparse_ratio: float = 2.0,
    local_range: int = 11,
):
    """
    Process a video using the distributed FlashVSR system.

    Args:
        input_path: Path to input video
        output_path: Path for output video (default: processed_{input_name})
        orchestrator_url: URL of the orchestrator service
        use_tiling: Whether to use spatial tiling
        tile_size: Size of spatial tiles
        tile_overlap: Overlap between spatial tiles
        temp_chunk_size: Frames per temporal chunk
        temp_overlap: Overlap between temporal chunks
        scale: Upscaling factor
        seed: Random seed
        sparse_ratio: Sparse attention ratio
        local_range: Local attention range
    """
    if output_path is None:
        import os
        basename = os.path.basename(input_path)
        name, ext = os.path.splitext(basename)
        output_path = f"processed_{name}{ext}"

    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    print(f"Orchestrator: {orchestrator_url}")
    print(f"Spatial tiling: {use_tiling}")
    if use_tiling:
        print(f"  Tile size: {tile_size}x{tile_size}")
        print(f"  Tile overlap: {tile_overlap}")
    print(f"Temporal chunking:")
    print(f"  Chunk size: {temp_chunk_size}")
    print(f"  Overlap: {temp_overlap}")
    print(f"Scale: {scale}x")
    print(f"Sparse ratio: {sparse_ratio}")
    print(f"Local range: {local_range}")
    print(f"Seed: {seed}")
    print("-" * 50)

    # Prepare request
    url = f"{orchestrator_url}/process-video-distributed/"
    files = {"file": open(input_path, "rb")}
    params = {
        "use_tiling": use_tiling,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "temp_chunk_size": temp_chunk_size,
        "temp_overlap": temp_overlap,
        "scale": scale,
        "seed": seed,
        "sparse_ratio": sparse_ratio,
        "local_range": local_range,
    }

    # Send request
    print("Sending request to orchestrator...")
    try:
        response = requests.post(url, files=files, params=params, timeout=3600)
        response.raise_for_status()

        # Save result
        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"✓ Processing complete!")
        print(f"✓ Output saved to: {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Process video using distributed FlashVSR"
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument(
        "--orchestrator",
        default="http://localhost:8000",
        help="Orchestrator URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--tiling",
        action="store_true",
        help="Enable spatial tiling (for large videos)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size in pixels (default: 512)",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=64,
        help="Tile overlap in pixels (default: 64)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=133,
        help="Temporal chunk size in frames (default: 133)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=8,
        help="Temporal chunk overlap in frames (default: 8)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Upscaling factor (default: 4.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--sparse-ratio",
        type=float,
        default=2.0,
        help="Sparse attention ratio (default: 2.0)",
    )
    parser.add_argument(
        "--local-range",
        type=int,
        default=11,
        help="Local attention range (default: 11)",
    )

    args = parser.parse_args()

    process_video_distributed(
        input_path=args.input,
        output_path=args.output,
        orchestrator_url=args.orchestrator,
        use_tiling=args.tiling,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        temp_chunk_size=args.chunk_size,
        temp_overlap=args.chunk_overlap,
        scale=args.scale,
        seed=args.seed,
        sparse_ratio=args.sparse_ratio,
        local_range=args.local_range,
    )


if __name__ == "__main__":
    main()
