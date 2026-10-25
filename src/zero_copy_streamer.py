import subprocess
import numpy as np

class ZeroCopyFFmpegStreamer:
    """
    Manages an active FFmpeg child process via standard input pipe (stdin).
    Streams raw BGR24 memory buffers directly to hardware/software encoders.
    """
    def __init__(self, output_path, width=1920, height=1080, fps=24, bitrate="14000k"):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p",
            output_path
        ]

        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write_frame(self, frame_bgr_np):
        """
        Writes a single contiguous uint8 BGR numpy array directly to the pipe.
        """
        if not frame_bgr_np.flags['C_CONTIGUOUS']:
            frame_bgr_np = np.ascontiguousarray(frame_bgr_np)
        self.proc.stdin.write(frame_bgr_np.tobytes())

    def close(self):
        if self.proc and self.proc.stdin:
            self.proc.stdin.close()
            self.proc.wait()