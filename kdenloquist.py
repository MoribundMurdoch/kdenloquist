#!/usr/bin/env python3
"""
============================================================
  KdenLoquist — Audio-Synced Puppet Tool for Kdenlive
  Simplified exclusive "Puppet" Mode (Jaw-Drop)
============================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import cv2
import os
import threading
import subprocess
import tempfile

from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d


# ══════════════════════════════════════════════════════════
#  Colour palette
# ══════════════════════════════════════════════════════════
BG        = "#0d0d1a"
PANEL_BG  = "#13131f"
ACCENT    = "#00e5a0"
ACCENT2   = "#7b5ea7"
FG        = "#e8e8f0"
FG_DIM    = "#666680"
BTN_BG    = "#1e1e32"
BTN_HOV   = "#2a2a42"
RED       = "#ff4f6a"
YELLOW    = "#f5c542"

SNAP_DIST    = 14   # canvas pixels — clicking within this of pt[0] auto-closes
POINT_R      = 5    # radius of drawn control points


# ══════════════════════════════════════════════════════════
#  Custom Widgets
# ══════════════════════════════════════════════════════════

class StyledButton(tk.Button):
    def __init__(self, parent, text, command=None, accent=False, danger=False, **kw):
        color = ACCENT if accent else (RED if danger else BTN_BG)
        fg    = BG if accent or danger else FG
        super().__init__(
            parent, text=text, command=command,
            bg=color, fg=fg, activebackground=ACCENT2,
            activeforeground=FG, relief=tk.FLAT,
            font=("Segoe UI", 9, "bold" if accent else "normal"),
            padx=8, pady=5, cursor="hand2", **kw)
        self.bind("<Enter>", lambda e: self.config(bg=BTN_HOV if not accent and not danger else ACCENT2))
        self.bind("<Leave>", lambda e: self.config(bg=color))


class LabeledSlider(tk.Frame):
    def __init__(self, parent, label, var, from_, to, fmt="{:.2f}", **kw):
        super().__init__(parent, bg=PANEL_BG, **kw)
        self.var = var; self.fmt = fmt
        top = tk.Frame(self, bg=PANEL_BG); top.pack(fill=tk.X)
        tk.Label(top, text=label, bg=PANEL_BG, fg=FG_DIM,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.val_lbl = tk.Label(top, text=fmt.format(var.get()),
                                bg=PANEL_BG, fg=ACCENT, font=("Segoe UI", 8, "bold"))
        self.val_lbl.pack(side=tk.RIGHT)
        tk.Scale(self, variable=var, from_=from_, to=to,
                 orient=tk.HORIZONTAL, showvalue=False,
                 bg=PANEL_BG, fg=ACCENT, troughcolor=BTN_BG,
                 highlightthickness=0, bd=0, resolution=0.01,
                 command=self._on_change).pack(fill=tk.X)

    def _on_change(self, _=None):
        self.val_lbl.config(text=self.fmt.format(self.var.get()))


class Section(tk.Frame):
    def __init__(self, parent, title, **kw):
        super().__init__(parent, bg=PANEL_BG, **kw)
        hdr = tk.Frame(self, bg=ACCENT2); hdr.pack(fill=tk.X)
        tk.Label(hdr, text=f"  {title}", bg=ACCENT2, fg=FG,
                 font=("Segoe UI", 9, "bold"), pady=4).pack(side=tk.LEFT)
        self.body = tk.Frame(self, bg=PANEL_BG, padx=8, pady=6)
        self.body.pack(fill=tk.X)


# ══════════════════════════════════════════════════════════
#  Audio & Rendering logic
# ══════════════════════════════════════════════════════════

def bandpass_rms_envelope(audio, sr, fps, f_low, f_high, smoothing, offset_frames, anim_amount, 
                          power=1.2, threshold=0.05):
    nyq   = sr / 2.0
    f_low = max(20.0, min(f_low, nyq * 0.95))
    f_high= max(f_low + 10, min(f_high, nyq * 0.99))
    sos   = butter(4, [f_low / nyq, f_high / nyq], btype="band", output="sos")
    filt  = sosfiltfilt(sos, audio)
    spf   = sr / fps
    n     = int(np.ceil(len(audio) / spf))
    env   = np.zeros(n, dtype=np.float32)
    for i in range(n):
        af = i - offset_frames
        if af < 0: continue
        s = int(af * spf); e = int(s + spf)
        chunk = filt[s:min(e, len(filt))]
        if len(chunk): env[i] = np.sqrt(np.mean(chunk ** 2))
    
    mx = env.max()
    if mx > 0: env /= mx
    
    # Noise Gate & Response Curve
    env = np.where(env < threshold, 0, (env - threshold) / (1.0 - threshold + 1e-6))
    env = np.clip(env, 0, 1)
    env = np.power(env, power)
    
    if smoothing > 0:
        env = gaussian_filter1d(env, sigma=smoothing * 8.0)
        mx2 = env.max()
        if mx2 > 0: env /= mx2
    return env * anim_amount

def render_frame_puppet(base_rgba: np.ndarray, points: list,
                        amplitude: float, softness: float,
                        inner_color: tuple,
                        jaw_drop_pct: float = 0.55,
                        hinge_pct:    float = 0.40) -> np.ndarray:
    result = base_rgba.copy()
    if not points or len(points) < 3 or amplitude <= 0.001:
        return result

    h_img, w_img = result.shape[:2]
    pts_np = np.array(points, dtype=np.int32)
    x1, y1 = pts_np.min(axis=0); x2, y2 = pts_np.max(axis=0)
    mh = max(y2 - y1, 1)

    hinge_y  = int(y1 + mh * hinge_pct)
    max_drop = int(mh * jaw_drop_pct)
    drop     = int(max_drop * amplitude)
    if drop < 1: return result

    ir, ig, ib = inner_color
    pts = np.array(points, dtype=np.int32)
    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(full_mask, [pts], 255)

    upper_mask = full_mask.copy(); upper_mask[hinge_y:, :] = 0
    lower_mask = full_mask.copy(); lower_mask[:hinge_y, :] = 0

    # Fill Inner Mouth
    inner_arr = np.full_like(base_rgba, [ir, ig, ib, 255])
    m_f  = full_mask[:, :, np.newaxis].astype(np.float32) / 255.0
    result = (base_rgba.astype(np.float32) * (1.0 - m_f) +
              inner_arr.astype(np.float32) * m_f).clip(0, 255).astype(np.uint8)

    # Restore Upper Lip
    m_u  = upper_mask[:, :, np.newaxis].astype(np.float32) / 255.0
    result = (result.astype(np.float32) * (1.0 - m_u) +
              base_rgba.astype(np.float32) * m_u).clip(0, 255).astype(np.uint8)

    # Shift Lower Jaw (The Cutout Effect)
    if drop < h_img:
        shifted_lower = np.zeros((h_img, w_img), dtype=np.uint8)
        shifted_lower[drop:, :] = lower_mask[:h_img - drop, :]
        jaw_pixels = np.zeros_like(base_rgba)
        jaw_pixels[drop:, :] = base_rgba[:h_img - drop, :]

        m_l = shifted_lower[:, :, np.newaxis].astype(np.float32) / 255.0
        result = (result.astype(np.float32) * (1.0 - m_l) +
                  jaw_pixels.astype(np.float32) * m_l).clip(0, 255).astype(np.uint8)

    # Feathering
    if softness > 0:
        k = max(3, int(softness) * 2 + 1) | 1
        sig = softness * 0.4
        result = cv2.GaussianBlur(result, (k, k), sig)

    result[:, :, 3] = base_rgba[:, :, 3]
    return result

# ══════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════

class KdenLoquist:
    APP_TITLE = "KdenLoquist — Puppet Tool"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(self.APP_TITLE)
        self.root.geometry("1280x780")
        self.root.configure(bg=BG)

        # State
        self.image = None; self.base_rgba = None
        self.audio_data = None; self.sample_rate = 44100
        self.mask_points = []; self.mask_closed = False

        # Variables
        self.v_anim      = tk.DoubleVar(value=0.65)
        self.v_smooth    = tk.DoubleVar(value=0.15)
        self.v_jaw_drop  = tk.DoubleVar(value=0.55)
        self.v_hinge     = tk.DoubleVar(value=0.40)
        self.v_flo       = tk.DoubleVar(value=80.0)
        self.v_fhi       = tk.DoubleVar(value=3500.0)
        self.v_soft      = tk.DoubleVar(value=2.0)
        self.v_power     = tk.DoubleVar(value=1.2)
        self.v_threshold = tk.DoubleVar(value=0.05)
        self.v_fps       = tk.IntVar(value=24)
        self.v_offset    = tk.IntVar(value=0)
        self.v_mouth_color = tk.StringVar(value="#1a0008")

        self._build_ui()

    def _build_ui(self):
        # Toolbar
        tb = tk.Frame(self.root, bg=ACCENT2, height=42)
        tb.pack(fill=tk.X); tb.pack_propagate(False)
        tk.Label(tb, text="  🎭  KdenLoquist Puppet", bg=ACCENT2, fg=FG,
                 font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=6)

        # Panels
        main = tk.Frame(self.root, bg=BG); main.pack(fill=tk.BOTH, expand=True)
        self._build_left_panel(main)
        
        self._canvas = tk.Canvas(main, bg="#090912", highlightthickness=1, highlightbackground=ACCENT2)
        self._canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _build_left_panel(self, parent):
        p = tk.Frame(parent, bg=PANEL_BG, width=280)
        p.pack(side=tk.LEFT, fill=tk.Y)
        p.pack_propagate(False)

        # Using a scrollable area for sliders
        sc = tk.Canvas(p, bg=PANEL_BG, highlightthickness=0)
        sc.pack(fill=tk.BOTH, expand=True)
        inner = tk.Frame(sc, bg=PANEL_BG)
        sc.create_window((0,0), window=inner, anchor="nw", width=260)

        # File Actions
        s1 = Section(inner, "📁 Files"); s1.pack(fill=tk.X, pady=2)
        StyledButton(s1.body, "📷 Load Image", command=self._load_image, accent=True).pack(fill=tk.X)
        StyledButton(s1.body, "🎵 Load Audio", command=self._load_audio, accent=True).pack(fill=tk.X, pady=4)

        # Puppet Controls
        s2 = Section(inner, "🎬 Animation"); s2.pack(fill=tk.X, pady=2)
        LabeledSlider(s2.body, "Animation Amount", self.v_anim, 0.1, 1.5).pack(fill=tk.X)
        LabeledSlider(s2.body, "Jaw Drop (Cutout Width)", self.v_jaw_drop, 0.1, 1.0).pack(fill=tk.X)
        LabeledSlider(s2.body, "Hinge Position", self.v_hinge, 0.0, 1.0).pack(fill=tk.X)
        LabeledSlider(s2.body, "Response Curve (Snappy)", self.v_power, 0.5, 3.0).pack(fill=tk.X)
        LabeledSlider(s2.body, "Noise Gate", self.v_threshold, 0.0, 0.2).pack(fill=tk.X)
        LabeledSlider(s2.body, "Smoothing", self.v_smooth, 0.0, 1.0).pack(fill=tk.X)
        
        # Export
        s3 = Section(inner, "⚙ Output"); s3.pack(fill=tk.X, pady=2)
        StyledButton(s3.body, "🎬 Export MP4", command=self._start_export, accent=True).pack(fill=tk.X)

    # ... [Rest of the file-loading, canvas interaction, and export logic goes here] ...

if __name__ == "__main__":
    root = tk.Tk()
    app = KdenLoquist(root)
    root.mainloop()
