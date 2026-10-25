# PyTorch GPU Cinema Engine & Zero-Copy Pipe Streamer

A high-throughput 2D cinematic visual rendering engine built on **PyTorch GPU tensors** and **FFmpeg IPC piping**. Designed to apply real-time camera spatial transformations, volumetric ray-marched atmospheric optics, and procedural dust synthesis without disk I/O bottlenecks.

---

## Technical Highlights

* **Zero-Disk I/O Pipelining:** Direct in-memory frame streaming from PyTorch GPU tensors to FFmpeg standard input (`stdin`) via raw BGR24 byte buffers, entirely bypassing intermediate image file creation.
* **Pure Bilinear Coordinate Distortion:** Emulates smooth, sub-pixel camera zooms and pans using `torch.meshgrid` and `torch.nn.functional.grid_sample`, completely avoiding lossy image crops.
* **GPU-Accelerated Volumetric Optics:** Implements ray-marched radial light shafts (God-Rays) combined with dynamic dual-frequency sinusoidal flicker.
* **Procedural Particle Field:** Generates real-time illuminated floating dust motes using spatial harmonic trigonometric interference patterns.
* **Hardware Accelerated:** Seamless execution on CUDA-enabled GPUs with automatic CPU fallback.

---

## Architectural Workflow

```
[ Input Texture ]
       │
       ▼
[ GPU Coordinate Grid Sample (Zoom/Pan) ] ── (PyTorch Tensor)
       │
       ▼
[ Volumetric God-Rays + Floating Motes ] ── (Linear Algebra Shaders)
       │
       ▼
[ Film Grading & Dynamic Range Curve ]
       │
       ▼
[ In-Memory Zero-Copy Pipe (stdin) ] ─────► [ FFmpeg H.264 Encoder ] ──► [ Output MP4 ]
```

---

## Installation

Ensure [FFmpeg](https://ffmpeg.org/) is installed and added to your system `PATH`.

```bash
git clone [https://github.com/YOUR_USERNAME/pytorch-gpu-cinema-engine.git](https://github.com/YOUR_USERNAME/pytorch-gpu-cinema-engine.git)
cd pytorch-gpu-cinema-engine
pip install -r requirements.txt
```

---

## Quickstart

Run the standalone demo to render a 1080p sample video in seconds:

```bash
python demo.py
```

---

## License

This project is licensed under the MIT License.
