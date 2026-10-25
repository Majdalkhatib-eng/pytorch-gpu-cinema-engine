import torch
import torch.nn.functional as F
import numpy as np

class CinematicVisionEngine:
    """
    GPU-Accelerated 2D Optical & Camera Transformation Engine using PyTorch Tensors.
    Implements bilinear coordinate sampling, ray-marched volumetric god rays,
    procedural floating motes, and film grain grading.
    """
    def __init__(self, width=1920, height=1080, device=None):
        self.w = width
        self.h = height
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        # 1. إنشاء شبكة الإحداثيات المعيارية على الـ GPU
        y_c = torch.linspace(-1, 1, height, device=self.device, dtype=self.dtype)
        x_c = torch.linspace(-1, 1, width, device=self.device, dtype=self.dtype)
        self.y_mesh, self.x_mesh = torch.meshgrid(y_c, x_c, indexing="ij")

        # 2. أقنعة التعتيم المحيطي والـ Bokeh
        self.rad_dist = torch.sqrt(self.x_mesh**2 + (self.y_mesh * 1.12)**2)
        self.vignette = torch.clamp(1.0 - (self.rad_dist - 0.42) * 0.48, 0.52, 1.0).unsqueeze(0)
        
        focus_dist = torch.sqrt((self.x_mesh * 0.85)**2 + ((self.y_mesh + 0.10) * 1.10)**2)
        self.edge_bokeh_mask = torch.clamp((focus_dist - 0.48) / 0.55, 0.0, 1.0).unsqueeze(0)

        # 3. بنك الضجيج الحبيبي لمحاكاة السينما التناظرية
        torch.manual_seed(42)
        self.grain_bank = [torch.randn((1, height, width), device=self.device, dtype=self.dtype) * 0.012 for _ in range(8)]
        self.frame_counter = 0

    def fast_gpu_blur(self, img_tensor, radius_scale=0.16):
        bh = max(16, int(self.h * radius_scale))
        bw = max(16, int(self.w * radius_scale))
        small = F.avg_pool2d(
            F.interpolate(img_tensor.unsqueeze(0), size=(bh, bw), mode='bilinear', align_corners=False),
            kernel_size=3, stride=1, padding=1
        )
        return F.interpolate(small, size=(self.h, self.w), mode='bilinear', align_corners=False).squeeze(0)

    def sample_rigid_camera(self, img_tensor, base_scale, pan_x, pan_y):
        """
        Bilinear resample using coordinate grid distortion instead of slow image crops.
        """
        grid = torch.stack(
            ((self.x_mesh - pan_x) / base_scale, (self.y_mesh - pan_y) / base_scale),
            dim=-1
        ).unsqueeze(0)
        return F.grid_sample(img_tensor.unsqueeze(0), grid, mode='bilinear', padding_mode='reflection', align_corners=True).squeeze(0)

    def render_volumetric_god_rays(self, base_tensor, light_x, light_y, global_time=0.0):
        """
        Approximates atmospheric light shafts via multi-scale radial sampling.
        """
        luma = 0.114 * base_tensor[0] + 0.587 * base_tensor[1] + 0.299 * base_tensor[2]
        mean_luma = float(torch.mean(luma).item())
        adaptive_strength = max(0.08, min(0.22, (0.36 - mean_luma) * 0.80))

        bright_soft = self.fast_gpu_blur(
            torch.clamp((luma - 0.52) / 0.48, 0.0, 1.0).unsqueeze(0) * base_tensor, 
            radius_scale=0.20
        )
        rays = torch.zeros_like(base_tensor)
        for s in range(1, 7):
            sc = 1.0 + s * 0.038
            grid = torch.stack(
                ((self.x_mesh - light_x) / sc + light_x, (self.y_mesh - light_y) / sc + light_y),
                dim=-1
            ).unsqueeze(0)
            rays += F.grid_sample(bright_soft.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(0) * ((1.0 - s / 7.0) * adaptive_strength)

        torch_flicker = 1.0 + 0.035 * np.sin(global_time * 6.2) * np.cos(global_time * 10.8)
        return torch.stack([rays[0] * 0.60, rays[1] * 0.90, rays[2] * 1.18], dim=0) * torch_flicker

    def render_floating_dust(self, frame_tensor, global_time):
        """
        Procedural floating dust motes based on harmonic spatial sinusoidal frequencies.
        """
        t = global_time * 0.36
        w1 = torch.sin(self.x_mesh * 26.0 + t * 1.3 + torch.cos(self.y_mesh * 19.0 - t * 0.9))
        w2 = torch.cos(self.y_mesh * 31.0 - t * 1.1 + torch.sin(self.x_mesh * 23.0 + t * 0.8))
        motes = (torch.clamp((w1 * w2) - 0.875, 0.0, 1.0) * 8.0) ** 2

        luma = 0.114 * frame_tensor[0] + 0.587 * frame_tensor[1] + 0.299 * frame_tensor[2]
        light_zone = torch.clamp((luma + 0.15) / 0.55, 0.15, 1.0)
        dust_intensity = motes * light_zone * 0.14
        dust_bgr = torch.stack([dust_intensity * 0.42, dust_intensity * 0.78, dust_intensity * 1.05], dim=0)
        return torch.clamp(frame_tensor + dust_bgr, 0.0, 1.0)

    def apply_film_grade(self, frame_tensor):
        graded = frame_tensor * self.vignette
        # S-Curve Contrast Enhancement
        graded = 0.5 * (torch.sin(torch.pi * (graded - 0.5)) + 1.0) * 0.14 + graded * 0.86
        grain = self.grain_bank[self.frame_counter % len(self.grain_bank)]
        self.frame_counter += 1
        return torch.clamp(graded + grain, 0.0, 1.0)