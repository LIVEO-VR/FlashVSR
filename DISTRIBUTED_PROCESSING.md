# FlashVSR Distributed Processing Guide

This guide explains how to use the distributed video processing system for FlashVSR.

## Architecture

The system consists of two components:

1. **Worker (`worker.py`)**: Processes individual video chunks. Multiple workers can run on different machines.
2. **Orchestrator (`orchestrator.py`)**: Splits videos into chunks, distributes them to workers, and stitches results together.

```
┌─────────────┐
│ Orchestrator│
│  (Port 8000)│
└──────┬──────┘
       │
       ├──────────┬──────────┬──────────┐
       │          │          │          │
   ┌───▼───┐  ┌──▼────┐  ┌──▼────┐  ┌──▼────┐
   │Worker1│  │Worker2│  │Worker3│  │Worker4│
   │(8001) │  │(8001) │  │(8001) │  │(8001) │
   │GPU 1  │  │GPU 2  │  │GPU 3  │  │GPU 4  │
   └───────┘  └───────┘  └───────┘  └───────┘
   Machine1   Machine2   Machine3   Machine4
```

## Setup

### 1. Install Dependencies

Ensure you have all required packages installed:

```bash
pip install aiohttp fastapi uvicorn av imageio numpy pillow torch
```

### 2. Start Worker Instances

On each machine with a GPU, start a worker:

```bash
# On Machine 1
python worker.py
# This will start on port 8001 by default

# On Machine 2
python worker.py

# On Machine 3
python worker.py
```

**Note**: If running multiple workers on the same machine (e.g., for testing), you can specify different ports:

```bash
python worker.py --port 8001
python worker.py --port 8002
python worker.py --port 8003
```

### 3. Configure Orchestrator

Edit `orchestrator.py` to specify your worker URLs:

```python
WORKER_URLS = [
    "http://192.168.1.10:8001",  # Machine 1
    "http://192.168.1.11:8001",  # Machine 2
    "http://192.168.1.12:8001",  # Machine 3
    "http://192.168.1.13:8001",  # Machine 4
]
```

For local testing with multiple workers on different ports:

```python
WORKER_URLS = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]
```

### 4. Start Orchestrator

```bash
python orchestrator.py
# This will start on port 8000
```

## Usage

### API Endpoint

```
POST http://localhost:8000/process-video-distributed/
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | Input video file |
| `use_tiling` | bool | false | Enable spatial tiling for large videos |
| `tile_size` | int | 512 | Size of each spatial tile (original resolution) |
| `tile_overlap` | int | 64 | Overlap between spatial tiles (original resolution) |
| `temp_chunk_size` | int | 133 | Number of frames per temporal chunk |
| `temp_overlap` | int | 8 | Overlap between temporal chunks |
| `scale` | float | 4.0 | Upscaling factor |
| `seed` | int | 0 | Random seed for reproducibility |
| `sparse_ratio` | float | 2.0 | Sparse attention ratio (1.5-2.0 recommended) |
| `local_range` | int | 11 | Local attention range (9 or 11 recommended) |

### Example: Using cURL

**Basic usage (no spatial tiling):**

```bash
curl -X POST "http://localhost:8000/process-video-distributed/" \
  -F "file=@input_video.mp4" \
  -F "use_tiling=false" \
  -F "temp_chunk_size=133" \
  -F "temp_overlap=8" \
  -F "scale=4.0" \
  -o output_video.mp4
```

**With spatial tiling (for very large videos):**

```bash
curl -X POST "http://localhost:8000/process-video-distributed/" \
  -F "file=@large_video.mp4" \
  -F "use_tiling=true" \
  -F "tile_size=512" \
  -F "tile_overlap=64" \
  -F "temp_chunk_size=133" \
  -F "temp_overlap=8" \
  -F "scale=4.0" \
  -o output_video.mp4
```

### Example: Using Python

```python
import requests

url = "http://localhost:8000/process-video-distributed/"

files = {"file": open("input_video.mp4", "rb")}
params = {
    "use_tiling": False,
    "temp_chunk_size": 133,
    "temp_overlap": 8,
    "scale": 4.0,
    "seed": 0,
    "sparse_ratio": 2.0,
    "local_range": 11,
}

response = requests.post(url, files=files, params=params)

with open("output_video.mp4", "wb") as f:
    f.write(response.content)
```

## How It Works

### Processing Flow

1. **Video Upload**: Client sends video to orchestrator
2. **Spatial Splitting** (if enabled): Video frames are split into overlapping tiles
3. **Temporal Splitting**: Each tile is split into overlapping temporal chunks
4. **Distribution**: All chunks are distributed to workers in parallel using round-robin
5. **Processing**: Each worker processes its assigned chunk
6. **Temporal Stitching**: For each tile, temporal chunks are blended together
7. **Spatial Stitching** (if enabled): Tiles are blended using feathering masks
8. **Output**: Final video is saved and returned

### Temporal Stitching

Uses linear blending on overlapping frames:
- Overlap region is blended with gradually increasing weight
- Ensures smooth transitions between chunks

### Spatial Stitching

Uses feathering masks on tile edges:
- Creates smooth ramps on all edges within overlap region
- Prevents visible seams between tiles
- Corner regions have 2D feathering applied

## Performance Considerations

### When to Use Spatial Tiling

Enable `use_tiling=true` when:
- Video resolution is very large (e.g., 4K+)
- GPU memory is limited
- You want to process different regions in parallel on different GPUs

Recommended settings for 4K video:
- `tile_size=512` (original resolution)
- `tile_overlap=64` (original resolution)

### Temporal Chunking

- Larger `temp_chunk_size`: Better quality, more memory, fewer chunks
- Smaller `temp_chunk_size`: Less memory, more chunks, more overhead
- `temp_overlap=8`: Good balance for most videos

The system automatically adjusts `temp_chunk_size` to the next `8n+5` value for optimal FlashVSR processing.

### Worker Distribution

Jobs are distributed using round-robin:
- With 4 workers and 16 jobs: each worker gets 4 jobs
- Workers process jobs sequentially, but all workers run in parallel
- More workers = faster overall processing

## Comparison with Original System

### Original (`app.py`)
- Single machine processing
- Sequential temporal chunking
- No spatial tiling
- Simpler setup

### Distributed (`orchestrator.py`)
- Multi-machine processing
- Parallel temporal chunking across workers
- Optional spatial tiling
- More complex setup, much faster for large videos

## Troubleshooting

### Workers not responding

Check worker URLs in `orchestrator.py`:
```bash
# Test worker health
curl http://localhost:8001/health
```

### Out of memory on workers

- Reduce `temp_chunk_size`
- Enable spatial tiling with smaller `tile_size`
- Use fewer parallel jobs per worker

### Seams visible in output

- Increase `temp_overlap` (temporal)
- Increase `tile_overlap` (spatial)
- Check that feathering masks are applied correctly

## Advanced Configuration

### Running Workers on Specific GPUs

```python
# worker.py - modify as needed
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
```

### Custom Worker Port

Modify the last line of `worker.py`:

```python
if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    uvicorn.run(app, host="0.0.0.0", port=port)
```

Then run:
```bash
python worker.py 8002
```

## Monitoring

The orchestrator prints progress messages:
- Job distribution
- Worker responses
- Stitching progress

Example output:
```
[Orchestrator] Starting distributed processing
[Orchestrator] Video info: 300 frames, 1920x1080, 30 fps
[Orchestrator] Created 4 spatial tiles
[Orchestrator] Created 3 temporal chunks
[Orchestrator] Total jobs: 12
[Orchestrator] Distributing jobs to workers...
[Orchestrator] Sending tile0_chunk0 to http://localhost:8001
[Orchestrator] Sending tile1_chunk0 to http://localhost:8002
...
[Orchestrator] Stitching temporal chunks...
[Orchestrator] Stitching spatial tiles...
[Orchestrator] Final output: 300 frames
[Orchestrator] Saved final video to results/processed_video.mp4
```
