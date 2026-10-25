import os
import cv2
import torch
import numpy as np
from src.vision_engine import CinematicVisionEngine
from src.zero_copy_streamer import ZeroCopyFFmpegStreamer

def main():
    width, height, fps, duration_sec = 1920, 1080, 24, 5
    total_frames = fps * duration_sec
    output_video = "cinematic_output.mp4"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Engine on: [{device.upper()}]")

    # إنشاء صورة تجريبية بدقة 1080p إذا لم تتوفر صورة
    dummy_img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (width // 2, height // 2), 260, (180, 160, 140), -1)
    cv2.putText(dummy_img, "PYTORCH GPU ENGINE", (width // 2 - 280, height // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)

    img_tensor = torch.from_numpy(dummy_img).permute(2, 0, 1).to(device, dtype=torch.float32) / 255.0

    engine = CinematicVisionEngine(width, height, device=device)
    streamer = ZeroCopyFFmpegStreamer(output_video, width, height, fps=fps)

    print(f"[*] Rendering {total_frames} frames via Zero-Copy Pipe...")
    for f in range(total_frames):
        t = f / float(fps)
        p = f / float(total_frames)

        # Smooth camera zoom & pan
        ease = p * p * (3.0 - 2.0 * p)
        scale = 1.0 + 0.05 * ease
        pan_x = 0.01 * ease

        cam_frame = engine.sample_rigid_camera(img_tensor, scale, pan_x, 0.0)
        rays = engine.render_volumetric_god_rays(cam_frame, -0.2, -0.6, global_time=t)
        lit_frame = torch.clamp(cam_frame + rays, 0.0, 1.0)
        dust_frame = engine.render_floating_dust(lit_frame, global_time=t)
        graded_frame = engine.apply_film_grade(dust_frame)

        # التحويل إلى مصفوفة بايتات وضخها إلى FFmpeg مباشرة
        frame_np = (graded_frame.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        streamer.write_frame(frame_np)

    streamer.close()
    print(f"[OK] Render complete! Video saved to: {output_video}")

if __name__ == "__main__":
    main()