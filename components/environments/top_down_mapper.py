import numpy as np


class OrthoTopDownMapper:
    """
    Maps world (x,z) to image (u,v) pixels for an orthographic top-down camera at yaw=0.
    The visible world bounds are implied by (center_x, center_z, ortho_size, aspect).
    """

    def __init__(self, center_x, center_z, ortho_size, img_h, img_w):
        self.cx = center_x
        self.cz = center_z
        self.os = ortho_size
        self.h = img_h
        self.w = img_w
        self.aspect = img_w / img_h

        # Visible world bounds (derived from camera settings)
        self.x_min = self.cx - self.os * self.aspect
        self.x_max = self.cx + self.os * self.aspect
        self.z_min = self.cz - self.os
        self.z_max = self.cz + self.os

        self.dx = max(1e-8, self.x_max - self.x_min)
        self.dz = max(1e-8, self.z_max - self.z_min)

    def world_to_pixel(self, x, z):
        # Normalize world to [0,1] within the visible bounds
        u01 = (x - self.x_min) / self.dx
        v01 = (z - self.z_min) / self.dz
        # Convert to pixel coords; note v-axis inversion (image origin at top-left)
        u = np.clip(u01 * (self.w - 1), 0, self.w - 1)
        v = np.clip((1.0 - v01) * (self.h - 1), 0, self.h - 1)
        return int(round(u)), int(round(v))
