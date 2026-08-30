
from __future__ import annotations

import numpy as np


class Figure8:

    def __init__(self, radius: float = 1.0, lap_time: float = 6.0, altitude: float = 1.0):
        self.radius = float(radius)
        self.lap_time = float(lap_time)
        self.altitude = float(altitude)
        self.w = 2.0 * np.pi / self.lap_time

    def pos(self, t):
        t = np.asarray(t, dtype=float)
        wt = self.w * t
        x = self.radius * np.sin(wt)
        y = 0.5 * self.radius * np.sin(2.0 * wt)
        z = np.full_like(x, self.altitude)
        return np.stack([x, y, z], axis=-1)

    def vel(self, t):
        t = np.asarray(t, dtype=float)
        wt = self.w * t
        vx = self.radius * self.w * np.cos(wt)
        vy = self.radius * self.w * np.cos(2.0 * wt)
        vz = np.zeros_like(vx)
        return np.stack([vx, vy, vz], axis=-1)

    def peak_speed(self) -> float:
        t = np.linspace(0.0, self.lap_time, 2000)
        return float(np.linalg.norm(self.vel(t), axis=-1).max())

    def start_pos(self):
        return self.pos(0.0)
