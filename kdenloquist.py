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
BG       = "#0d0d1a"
PANEL_BG = "#13131f"
ACCENT   = "#00e5a0"
ACCENT2  = "#7b5ea7"
FG       = "#e8e8f0"
FG_DIM   = "#666680"
BTN_BG   = "#1e1e32"
BTN_HOV  = "#2a2a42"
RED      = "#ff4f6a"
YELLOW   = "#f5c542"

SNAP_DIST   = 14   # canvas pixels — clicking within this of pt[0] auto-closes
POINT_R     = 5    # radius of drawn control points


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
        
        # Added resolution=0.01 for finer control on small variables like Hinge
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
#  Audio helpers
# ══════════════════════════════════════════════════════════

def decode_audio_to_wav(src_path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ar", "44100", "-ac", "1", "-f", "wav", tmp.name],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr.decode()}")
    return tmp.name


def load_audio(path: str):
    wav_path = path
    tmp_created = False
    if not path.lower().endswith(".wav"):
        wav_path = decode_audio_to_wav(path)
        tmp_created = True
    sr, data = wavfile.read(wav_path)
    if tmp_created:
        os.unlink(wav_path)
    if data.dtype   == np.int16:  data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:  data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:  data = (data.astype(np.float32) - 128) / 128.0
    else:                         data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, int(sr)


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
    
    # ── Normalize ──
    mx = env.max()
    if mx > 0: env /= mx
    
    # ── 1. Noise Gate (The "Control" fix) ──
    # Prevents the cardboard from vibrating during quiet background noise
    env = np.where(env < threshold, 0, (env - threshold) / (1.0 - threshold + 1e-6))
    env = np.clip(env, 0, 1)
    
    # ── 2. Response Curve (The "Snappiness" fix) ──
    # power > 1.0 makes the mouth 'pop' open for loud sounds
    env = np.power(env, power)
    
    if smoothing > 0:
        env = gaussian_filter1d(env, sigma=smoothing * 8.0)
        # Re-normalize after smoothing
        mx2 = env.max()
        if mx2 > 0: env /= mx2
        
    return env * anim_amount


# ══════════════════════════════════════════════════════════
#  Frame renderers
# ══════════════════════════════════════════════════════════

def _poly_mask(points, h, w) -> np.ndarray:
    """uint8 mask filled inside the polygon."""
    pts = np.array(points, dtype=np.int32)
    m   = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return m


def render_frame_puppet(base_rgba: np.ndarray, points: list,
                        amplitude: float, softness: float,
                        inner_color: tuple,
                        jaw_drop_pct: float = 0.55,
                        hinge_pct:    float = 0.40) -> np.ndarray:
    """
    Puppet jaw-drop using a polygon mask.

    The polygon's bounding box defines top/bottom for the hinge calculation.
    Everything above the hinge line stays fixed (upper lip).
    Everything below the hinge is translated downward by
        drop = jaw_drop_pct × bbox_height × amplitude  pixels.
    The revealed gap is filled with inner_color.
    """
    result = base_rgba.copy()
    if not points or len(points) < 3 or amplitude <= 0.001:
        return result

    h_img, w_img = result.shape[:2]
    pts_np = np.array(points, dtype=np.int32)
    x1, y1 = pts_np.min(axis=0)
    x2, y2 = pts_np.max(axis=0)
    mh = max(y2 - y1, 1)

    hinge_y  = int(y1 + mh * hinge_pct)
    max_drop = int(mh * jaw_drop_pct)
    drop     = int(max_drop * amplitude)
    if drop < 1:
        return result

    ir, ig, ib = inner_color

    # ── Full polygon mask ──────────────────────────────────
    full_mask   = _poly_mask(points, h_img, w_img)

    # ── Upper mask: polygon AND above hinge ────────────────
    upper_mask  = full_mask.copy()
    upper_mask[hinge_y:, :] = 0

    # ── Lower mask: polygon AND below hinge ────────────────
    lower_mask  = full_mask.copy()
    lower_mask[:hinge_y, :] = 0

    # ── 1. Fill entire polygon area with inner colour ──────
    inner_arr      = np.empty_like(base_rgba)
    inner_arr[:,:] = [ir, ig, ib, 255]
    m_f  = full_mask[:, :, np.newaxis].astype(np.float32) / 255.0
    result = (base_rgba.astype(np.float32) * (1.0 - m_f) +
              inner_arr.astype(np.float32) * m_f).clip(0, 255).astype(np.uint8)

    # ── 2. Restore upper lip ───────────────────────────────
    m_u  = upper_mask[:, :, np.newaxis].astype(np.float32) / 255.0
    result = (result.astype(np.float32) * (1.0 - m_u) +
              base_rgba.astype(np.float32) * m_u).clip(0, 255).astype(np.uint8)

    # ── 3. Shift lower jaw down by `drop` pixels ──────────
    if drop < h_img:
        shifted_lower = np.zeros((h_img, w_img), dtype=np.uint8)
        shifted_lower[drop:, :] = lower_mask[:h_img - drop, :]

        # Source pixels come from original image shifted down
        jaw_pixels = np.zeros_like(base_rgba)
        jaw_pixels[drop:, :] = base_rgba[:h_img - drop, :]

        m_l = shifted_lower[:, :, np.newaxis].astype(np.float32) / 255.0
        result = (result.astype(np.float32) * (1.0 - m_l) +
                  jaw_pixels.astype(np.float32) * m_l).clip(0, 255).astype(np.uint8)

    # ── 4. Feather edges ───────────────────────────────────
    if softness > 0:
        k      = max(3, int(softness) * 2 + 1) | 1
        margin = k
        sig    = softness * 0.4

        def blur_hband(ya, yb):
            ya = max(0, ya); yb = min(h_img, yb)
            xa = max(0, x1 - margin); xb = min(w_img, x2 + margin)
            if yb > ya and xb > xa:
                band = result[ya:yb, xa:xb].astype(np.float32)
                result[ya:yb, xa:xb] = cv2.GaussianBlur(band, (k, k), sig).astype(np.uint8)

        blur_hband(hinge_y - margin, hinge_y + margin)
        blur_hband(hinge_y + drop - margin, hinge_y + drop + margin)

    result[:, :, 3] = base_rgba[:, :, 3]
    return result


# ══════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════

class KdenLoquist:
    APP_TITLE = "KdenLoquist — Audio-Synced Puppet Tool for Kdenlive"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(self.APP_TITLE)
        self.root.geometry("1280x780")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)

        # Image / audio state
        self.image          = None
        self.image_path     = None
        self.base_rgba      = None
        self.audio_data     = None
        self.sample_rate    = 44100
        self.audio_path     = None
        self.audio_duration = 0.0

        # Polygon mask state
        self.mask_points: list = []   # [(ix, iy), …] in image coordinates
        self.mask_closed: bool = False

        # Canvas display helpers
        self._disp_scale  = 1.0
        self._disp_off_x  = 0
        self._disp_off_y  = 0
        self._tk_image    = None

        # Rubber-band / canvas overlay item IDs
        self._rubber_id   = None   # preview line to cursor
        self._hover_cpos  = None   # (cx, cy) canvas coords of last mouse pos

        # Double-click de-bounce
        self._skip_next_click = False

        # Export
        self._export_thread = None
        self._cancel_export = False
        
        # UI Control Variables
        self.v_anim      = tk.DoubleVar(value=0.65)
        self.v_smooth    = tk.DoubleVar(value=0.15) # Adjusted for sharper puppet movement
        self.v_jaw_drop  = tk.DoubleVar(value=0.55)
        self.v_hinge     = tk.DoubleVar(value=0.40)
        self.v_power     = tk.DoubleVar(value=1.5)  # Response Curve
        self.v_threshold = tk.DoubleVar(value=0.05) # Noise Gate
        self.v_flo       = tk.DoubleVar(value=80.0)
        self.v_fhi       = tk.DoubleVar(value=3500.0)
        self.v_offset    = tk.IntVar(value=0)
        self.v_soft      = tk.DoubleVar(value=5.0)
        self.v_fps       = tk.IntVar(value=24)
        self.v_crf       = tk.IntVar(value=20)
        self.v_mouth_color = tk.StringVar(value="#1a0008")

        self._build_ui()

        # Keyboard shortcut: Z = undo last point
        self.root.bind("<z>", lambda e: self._undo_point())
        self.root.bind("<Z>", lambda e: self._undo_point())

    # ──────────────────────────────────────────────────────
    #  UI construction
    # ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_toolbar()
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True)
        self._build_left_panel(main)
        self._build_canvas(main)
        self._build_statusbar()

    def _build_toolbar(self):
        tb = tk.Frame(self.root, bg=ACCENT2, height=42)
        tb.pack(fill=tk.X); tb.pack_propagate(False)
        tk.Label(tb, text="  🎭  KdenLoquist Puppet", bg=ACCENT2, fg=FG,
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=6)
        tk.Label(tb, text="Audio-Synced Talking Tool  •  for Kdenlive",
                 bg=ACCENT2, fg="#ccc", font=("Segoe UI", 9)).pack(side=tk.LEFT)

    def _build_left_panel(self, parent):
        self._panel = tk.Frame(parent, bg=PANEL_BG, width=280)
        self._panel.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 0))
        self._panel.pack_propagate(False)
        sc = tk.Canvas(self._panel, bg=PANEL_BG, highlightthickness=0, width=278)
        sb = ttk.Scrollbar(self._panel, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = tk.Frame(sc, bg=PANEL_BG)
        wid = sc.create_window((0, 0), window=inner, anchor="nw", width=262)
        inner.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))
        sc.bind("<Configure>", lambda e: sc.itemconfig(wid, width=e.width))
        self._build_panel_contents(inner)

    def _build_panel_contents(self, p):
        pad = dict(fill=tk.X, pady=3, padx=4)

        # ── Files ──────────────────────────────────────────
        s = Section(p, "📁  Files"); s.pack(**pad); b = s.body
        StyledButton(b, "📷  Load Image", command=self._load_image, accent=True).pack(fill=tk.X, pady=2)
        self._img_lbl = tk.Label(b, text="No image loaded", bg=PANEL_BG, fg=FG_DIM,
                                 font=("Segoe UI", 8), wraplength=220)
        self._img_lbl.pack(anchor=tk.W)
        StyledButton(b, "🎵  Load Audio", command=self._load_audio, accent=True).pack(fill=tk.X, pady=2)
        self._aud_lbl = tk.Label(b, text="No audio loaded", bg=PANEL_BG, fg=FG_DIM,
                                 font=("Segoe UI", 8), wraplength=220)
        self._aud_lbl.pack(anchor=tk.W)

        # ── Mouth Mask ─────────────────────────────────────
        s2 = Section(p, "🖊  Mouth Mask"); s2.pack(**pad); b2 = s2.body
        tk.Label(b2,
                 text="Click to place points around the mouth.\n"
                      "Double-click or click near ● to close.\n"
                      "Right-click or Z to undo last point.",
                 bg=PANEL_BG, fg=FG_DIM, font=("Segoe UI", 8),
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 4))

        btn_row = tk.Frame(b2, bg=PANEL_BG); btn_row.pack(fill=tk.X, pady=2)
        StyledButton(btn_row, "✓ Close",  command=self._close_mask).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        StyledButton(btn_row, "↩ Undo",   command=self._undo_point).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2,2))
        StyledButton(btn_row, "🗑 Clear", command=self._clear_mask, danger=True).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2,0))

        self._mask_lbl = tk.Label(b2, text="No points placed", bg=PANEL_BG, fg=FG_DIM,
                                  font=("Segoe UI", 8, "italic"))
        self._mask_lbl.pack(anchor=tk.W, pady=(4, 0))

        # ── Animation ──────────────────────────────────────
        s3 = Section(p, "🎬  Puppet Physics"); s3.pack(**pad); b3 = s3.body
        
        LabeledSlider(b3, "Animation Amount", self.v_anim,  0.05, 1.5).pack(fill=tk.X, pady=2)
        LabeledSlider(b3, "Jaw Drop Amount", self.v_jaw_drop, 0.1, 1.0).pack(fill=tk.X, pady=2)
        LabeledSlider(b3, "Hinge Position",  self.v_hinge,   0.0, 1.0, fmt="{:.3f}").pack(fill=tk.X, pady=2)
        
        tk.Label(b3, text="—— Dynamics ——", bg=PANEL_BG, fg=FG_DIM, font=("Segoe UI", 8, "italic")).pack(pady=2)
        LabeledSlider(b3, "Response (Snappiness)", self.v_power, 0.5, 3.0, fmt="{:.2f}").pack(fill=tk.X, pady=2)
        LabeledSlider(b3, "Silence Cutoff (Gate)", self.v_threshold, 0.0, 0.3, fmt="{:.3f}").pack(fill=tk.X, pady=2)
        LabeledSlider(b3, "Smoothing",        self.v_smooth, 0.0, 1.0).pack(fill=tk.X, pady=2)

        tk.Label(b3, text="—— Audio & Polish ——", bg=PANEL_BG, fg=FG_DIM, font=("Segoe UI", 8, "italic")).pack(pady=2)
        LabeledSlider(b3, "Freq Low (Hz)",  self.v_flo, 20,  1000, fmt="{:.0f} Hz").pack(fill=tk.X, pady=2)
        LabeledSlider(b3, "Freq High (Hz)", self.v_fhi, 500, 8000, fmt="{:.0f} Hz").pack(fill=tk.X, pady=2)
        LabeledSlider(b3, "Edge Softness",  self.v_soft, 0,  20,   fmt="{:.1f}").pack(fill=tk.X, pady=2)

        of_row = tk.Frame(b3, bg=PANEL_BG); of_row.pack(fill=tk.X, pady=2)
        tk.Label(of_row, text="Audio Offset (frames)", bg=PANEL_BG, fg=FG_DIM,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Spinbox(of_row, from_=0, to=999, textvariable=self.v_offset, width=5,
                   bg=BTN_BG, fg=FG, buttonbackground=PANEL_BG,
                   highlightthickness=0).pack(side=tk.RIGHT)

        clr_row = tk.Frame(b3, bg=PANEL_BG); clr_row.pack(fill=tk.X, pady=2)
        tk.Label(clr_row, text="Inner Mouth Colour", bg=PANEL_BG, fg=FG_DIM,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._color_btn = tk.Button(clr_row, bg=self.v_mouth_color.get(), width=3,
                                    relief=tk.FLAT, cursor="hand2",
                                    command=self._pick_color)
        self._color_btn.pack(side=tk.RIGHT)

        # ── Output ─────────────────────────────────────────
        s4 = Section(p, "⚙  Output"); s4.pack(**pad); b4 = s4.body
        fps_row = tk.Frame(b4, bg=PANEL_BG); fps_row.pack(fill=tk.X, pady=2)
        tk.Label(fps_row, text="Frame Rate (FPS)", bg=PANEL_BG, fg=FG_DIM,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Combobox(fps_row, textvariable=self.v_fps, width=5,
                     values=[12, 23, 24, 25, 29, 30, 48, 50, 60],
                     state="readonly").pack(side=tk.RIGHT)
        LabeledSlider(b4, "Export Quality (CRF)", self.v_crf, 0, 51,
                      fmt="{:.0f}  (lower=better)").pack(fill=tk.X, pady=2)

        # ── Actions ────────────────────────────────────────
        s5 = Section(p, "▶  Actions"); s5.pack(**pad); b5 = s5.body
        StyledButton(b5, "👁  Preview Frame",  command=self._preview_frame).pack(fill=tk.X, pady=2)
        self._export_btn = StyledButton(b5, "🎬  Export Video",
                                        command=self._start_export, accent=True)
        self._export_btn.pack(fill=tk.X, pady=2)
        self._cancel_btn = StyledButton(b5, "✖  Cancel", command=self._cancel, danger=True)
        self._cancel_btn.pack(fill=tk.X, pady=2)
        self._cancel_btn.config(state=tk.DISABLED)
        self._progress = ttk.Progressbar(b5, mode="determinate", maximum=100)
        self._progress.pack(fill=tk.X, pady=4)
        self._prog_lbl = tk.Label(b5, text="Ready.", bg=PANEL_BG, fg=FG_DIM,
                                  font=("Segoe UI", 8))
        self._prog_lbl.pack(anchor=tk.W)

        # ── Help ───────────────────────────────────────────
        s6 = Section(p, "ℹ  Quick Help"); s6.pack(**pad)
        tk.Label(s6.body,
                 text=("MASK DRAWING:\n"
                       "  • Left-click = add point\n"
                       "  • Click near ● = close mask\n"
                       "  • Double-click = close mask\n"
                       "  • Right-click / Z = undo point\n\n"
                       "PUPPET PHYSICS:\n"
                       "  • Hinge ≈ 0.4 → upper lip / jaw split\n"
                       "  • Gate → cuts vibration in silence\n"
                       "  • Response → high value makes mouth snap open."),
                 bg=PANEL_BG, fg=FG_DIM, font=("Segoe UI", 8),
                 justify=tk.LEFT, wraplength=230).pack(anchor=tk.W)

    def _build_canvas(self, parent):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        tk.Label(frame,
                 text="← Load an image, then click to place mouth mask points",
                 bg=BG, fg=FG_DIM, font=("Segoe UI", 9)).pack(anchor=tk.W, padx=6, pady=(4, 0))
        self._canvas = tk.Canvas(frame, bg="#090912",
                                  highlightthickness=1, highlightbackground=ACCENT2,
                                  cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._canvas.bind("<Button-1>",        self._on_click)
        self._canvas.bind("<Double-Button-1>", self._on_double_click)
        self._canvas.bind("<Button-3>",        self._on_right_click)
        self._canvas.bind("<Motion>",          self._on_mouse_move)
        self._canvas.bind("<Configure>",       self._on_canvas_resize)

        self._canvas.after(200, self._draw_placeholder)

    def _build_statusbar(self):
        self._statusbar = tk.Label(self.root, text="Ready",
                                   bg=ACCENT2, fg=BG,
                                   font=("Segoe UI", 8), anchor=tk.W, padx=8, pady=3)
        self._statusbar.pack(fill=tk.X, side=tk.BOTTOM)

    def _status(self, msg, color=None):
        self._statusbar.config(text=msg, fg=color or BG)
        self.root.update_idletasks()

    # ──────────────────────────────────────────────────────
    #  File loading
    # ──────────────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("All", "*.*")])
        if not path: return
        try:
            self.image      = Image.open(path).convert("RGBA")
            self.image_path = path
            self.base_rgba  = np.array(self.image, dtype=np.uint8)
            self.mask_points = []
            self.mask_closed = False
            short = os.path.basename(path)
            w, h  = self.image.size
            self._img_lbl.config(text=f"{short}\n{w}×{h} px", fg=ACCENT)
            self._mask_lbl.config(text="No points placed", fg=FG_DIM)
            self._status(f"Image loaded: {short}  ({w}×{h})")
            self._refresh_canvas()
        except Exception as ex:
            messagebox.showerror("Image Error", str(ex))

    def _load_audio(self):
        path = filedialog.askopenfilename(
            title="Select Audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.aac *.m4a"), ("All", "*.*")])
        if not path: return
        self._status("Loading audio…")
        try:
            data, sr = load_audio(path)
            self.audio_data     = data
            self.sample_rate    = sr
            self.audio_path     = path
            self.audio_duration = len(data) / sr
            short = os.path.basename(path)
            self._aud_lbl.config(text=f"{short}\n{self.audio_duration:.1f}s  •  {sr} Hz", fg=ACCENT)
            self._status(f"Audio loaded: {short}  ({self.audio_duration:.1f}s)")
        except Exception as ex:
            messagebox.showerror("Audio Error", str(ex))
            self._status("Audio load failed.", color=RED)

    def _pick_color(self):
        from tkinter.colorchooser import askcolor
        c = askcolor(color=self.v_mouth_color.get(), title="Inner Mouth Colour")[1]
        if c:
            self.v_mouth_color.set(c)
            self._color_btn.config(bg=c)

    # ──────────────────────────────────────────────────────
    #  Coordinate helpers
    # ──────────────────────────────────────────────────────

    def _img_to_canvas(self, ix, iy):
        return (ix * self._disp_scale + self._disp_off_x,
                iy * self._disp_scale + self._disp_off_y)

    def _canvas_to_img(self, cx, cy):
        return ((cx - self._disp_off_x) / self._disp_scale,
                (cy - self._disp_off_y) / self._disp_scale)

    def _canvas_dist_to_first_point(self, cx, cy):
        """Distance in canvas pixels from (cx,cy) to the first mask point."""
        if not self.mask_points:
            return float("inf")
        fcx, fcy = self._img_to_canvas(*self.mask_points[0])
        return ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5

    # ──────────────────────────────────────────────────────
    #  Canvas events — polygon drawing
    # ──────────────────────────────────────────────────────

    def _on_click(self, e):
        if self.image is None or self.mask_closed:
            return
        if self._skip_next_click:
            self._skip_next_click = False
            return

        ix, iy = self._canvas_to_img(e.x, e.y)
        iw, ih = self.image.size
        ix = max(0, min(int(ix), iw - 1))
        iy = max(0, min(int(iy), ih - 1))

        # Snap-to-close: clicking near first point closes the polygon
        if len(self.mask_points) >= 3 and self._canvas_dist_to_first_point(e.x, e.y) <= SNAP_DIST:
            self._close_mask()
            return

        self.mask_points.append((ix, iy))
        self._update_mask_label()
        self._refresh_canvas()

    def _on_double_click(self, e):
        """Close the polygon; undo the extra point added by the 2nd click."""
        if self.image is None or self.mask_closed:
            return
        self._skip_next_click = True   # suppress the Button-1 that fired with this
        if len(self.mask_points) > 0:
            self.mask_points.pop()     # remove duplicate from 2nd click
        self._close_mask()

    def _on_right_click(self, e):
        self._undo_point()

    def _on_mouse_move(self, e):
        self._hover_cpos = (e.x, e.y)
        self._draw_rubber_band()

    def _on_canvas_resize(self, _):
        self._refresh_canvas()

    # ──────────────────────────────────────────────────────
    #  Mask operations
    # ──────────────────────────────────────────────────────

    def _close_mask(self):
        if len(self.mask_points) < 3:
            self._status("Need at least 3 points to close mask.", color=YELLOW)
            return
        self.mask_closed = True
        self._update_mask_label()
        self._refresh_canvas()
        self._status(f"Mask closed — {len(self.mask_points)} points.")

    def _undo_point(self):
        if self.mask_closed:
            self.mask_closed = False
            self._status("Mask reopened — add or remove points, then close again.")
        elif self.mask_points:
            self.mask_points.pop()
        self._update_mask_label()
        self._refresh_canvas()

    def _clear_mask(self):
        self.mask_points = []
        self.mask_closed = False
        self._update_mask_label()
        self._refresh_canvas()
        self._status("Mask cleared.")

    def _update_mask_label(self):
        n = len(self.mask_points)
        if n == 0:
            self._mask_lbl.config(text="No points placed", fg=FG_DIM)
        elif not self.mask_closed:
            self._mask_lbl.config(text=f"{n} point{'s' if n!=1 else ''} — not closed", fg=YELLOW)
        else:
            self._mask_lbl.config(text=f"✓ Closed mask  ({n} pts)", fg=ACCENT)

    # ──────────────────────────────────────────────────────
    #  Canvas drawing
    # ──────────────────────────────────────────────────────

    def _draw_placeholder(self):
        if self.image is None:
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            self._canvas.delete("all")
            self._canvas.create_text(cw // 2, ch // 2,
                                     text="Load an image to begin",
                                     fill=FG_DIM, font=("Segoe UI", 16))

    def _refresh_canvas(self, overlay=None):
        if self.image is None:
            return
        cw = max(self._canvas.winfo_width(),  100)
        ch = max(self._canvas.winfo_height(), 100)
        src = Image.fromarray(overlay) if overlay is not None else self.image
        iw, ih = src.size
        scale  = min(cw / iw, ch / ih, 1.0)
        dw, dh = int(iw * scale), int(ih * scale)
        self._disp_scale = scale
        self._disp_off_x = (cw - dw) // 2
        self._disp_off_y = (ch - dh) // 2
        self._tk_image   = ImageTk.PhotoImage(src.resize((dw, dh), Image.LANCZOS))

        self._canvas.delete("all")
        self._canvas.create_image(self._disp_off_x, self._disp_off_y,
                                  anchor=tk.NW, image=self._tk_image)
        self._draw_polygon_overlay()
        self._rubber_id = None   # deleted with "all" above
        self._draw_rubber_band()

    def _draw_polygon_overlay(self):
        pts = self.mask_points
        if not pts:
            return

        # Convert all points to canvas coords
        cpts = [self._img_to_canvas(ix, iy) for ix, iy in pts]

        # Lines between consecutive points
        for i in range(len(cpts) - 1):
            self._canvas.create_line(*cpts[i], *cpts[i + 1],
                                     fill=ACCENT, width=2)
        # Closing line
        if self.mask_closed and len(cpts) >= 2:
            self._canvas.create_line(*cpts[-1], *cpts[0],
                                     fill=ACCENT, width=2)

        # Hinge line
        if self.mask_closed and len(pts) >= 3:
            pts_np = np.array(pts)
            x1, y1 = pts_np.min(axis=0)
            x2, y2 = pts_np.max(axis=0)
            mh     = max(y2 - y1, 1)
            hy_img = int(y1 + mh * self.v_hinge.get())
            _, hcy = self._img_to_canvas(0, hy_img)
            hcx1, _ = self._img_to_canvas(x1, 0)
            hcx2, _ = self._img_to_canvas(x2, 0)
            self._canvas.create_line(hcx1, hcy, hcx2, hcy,
                                     fill=YELLOW, width=1, dash=(5, 3))
            self._canvas.create_text((hcx1 + hcx2) / 2, hcy - 8,
                                     text="— hinge —", fill=YELLOW,
                                     font=("Segoe UI", 7))

        # Control points — first point is highlighted as snap target
        for i, (cx, cy) in enumerate(cpts):
            is_first = (i == 0)
            fill  = YELLOW if is_first else ACCENT
            r     = POINT_R + 2 if is_first else POINT_R
            outline = FG if is_first else ACCENT2
            self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     fill=fill, outline=outline, width=1)

    def _draw_rubber_band(self):
        """Draw a dashed preview line from the last point to the cursor."""
        if self._rubber_id:
            self._canvas.delete(self._rubber_id)
            self._rubber_id = None
        if (self.mask_closed or not self.mask_points
                or self._hover_cpos is None or self.image is None):
            return
        last_cx, last_cy = self._img_to_canvas(*self.mask_points[-1])
        hx, hy = self._hover_cpos

        # Colour shifts to yellow when near the first point (snap zone)
        near_close = (len(self.mask_points) >= 3 and
                      self._canvas_dist_to_first_point(hx, hy) <= SNAP_DIST)
        colour = YELLOW if near_close else FG_DIM

        self._rubber_id = self._canvas.create_line(
            last_cx, last_cy, hx, hy,
            fill=colour, dash=(4, 4), width=1)

    # ──────────────────────────────────────────────────────
    #  Rendering & Preview helpers
    # ──────────────────────────────────────────────────────

    def _hex_to_rgb(self, h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _make_frame(self, amplitude: float) -> np.ndarray:
        if len(self.mask_points) < 3:
            return self.base_rgba.copy()
        inner = self._hex_to_rgb(self.v_mouth_color.get())
        
        # Hardcoded to use Puppet Mode exclusively
        return render_frame_puppet(
            self.base_rgba, self.mask_points, amplitude,
            self.v_soft.get(), inner,
            jaw_drop_pct=self.v_jaw_drop.get(),
            hinge_pct=self.v_hinge.get()
        )

    def _preview_frame(self):
        if self.image is None:
            messagebox.showwarning("No image", "Please load an image first."); return
        if len(self.mask_points) < 3:
            messagebox.showwarning("No mask", "Please place at least 3 points and close the mask."); return
        
        # Generate a test frame at 75% amplitude
        frame = self._make_frame(0.75)
        self._refresh_canvas(frame)
        self._status("Preview rendered at 75% amplitude.")

    # ──────────────────────────────────────────────────────
    #  Export
    # ──────────────────────────────────────────────────────

    def _start_export(self):
        if self.image is None:
            messagebox.showwarning("No image", "Please load an image first."); return
        if self.audio_data is None:
            messagebox.showwarning("No audio", "Please load an audio file first."); return
        if len(self.mask_points) < 3:
            messagebox.showwarning("No mask", "Please draw and close a mouth mask."); return
        if not self.mask_closed:
            if not messagebox.askyesno("Mask not closed",
                                       "The mask isn't closed yet. Export anyway?"):
                return
        out = filedialog.asksaveasfilename(
            title="Save exported video", defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("Matroska Video", "*.mkv")])
        if not out: return
        self._export_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._cancel_export = False
        self._progress["value"] = 0
        self._prog_lbl.config(text="Starting…", fg=FG_DIM)
        self._export_thread = threading.Thread(target=self._do_export, args=(out,), daemon=True)
        self._export_thread.start()

    def _cancel(self):
        self._cancel_export = True
        self._prog_lbl.config(text="Cancelling…", fg=YELLOW)

    def _do_export(self, out_path):
        try:
            fps      = self.v_fps.get()
            crf      = self.v_crf.get()
            iw, ih   = self.image.size
            n_frames = int(np.ceil(self.audio_duration * fps))

            self._ui(lambda: self._prog_lbl.config(text="Analysing audio…", fg=FG_DIM))
            
            # Pass ALL new tuning variables to the envelope calculator
            env = bandpass_rms_envelope(
                self.audio_data, self.sample_rate, fps,
                self.v_flo.get(), self.v_fhi.get(),
                self.v_smooth.get(), self.v_offset.get(),
                self.v_anim.get(),
                power=self.v_power.get(),         # NEW: Response curve
                threshold=self.v_threshold.get()  # NEW: Noise gate
            )
            n_frames = min(n_frames, len(env))

            tmp_vid = out_path + "_raw.mp4"
            writer  = cv2.VideoWriter(tmp_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (iw, ih))

            for i in range(n_frames):
                if self._cancel_export:
                    writer.release()
                    if os.path.exists(tmp_vid): os.unlink(tmp_vid)
                    self._ui(lambda: self._prog_lbl.config(text="Cancelled.", fg=YELLOW))
                    self._ui(self._reset_export_ui)
                    return
                bgr = cv2.cvtColor(self._make_frame(float(env[i])), cv2.COLOR_RGBA2BGR)
                writer.write(bgr)
                if i % max(1, n_frames // 80) == 0:
                    pct = int(i / n_frames * 80)
                    self._ui(lambda p=pct, n=i: (
                        self._progress.config(value=p),
                        self._prog_lbl.config(text=f"Rendering frame {n+1}/{n_frames}  ({p}%)", fg=FG_DIM)
                    ))

            writer.release()
            self._ui(lambda: self._prog_lbl.config(text="Muxing audio…", fg=FG_DIM))
            self._ui(lambda: self._progress.config(value=85))

            r = subprocess.run([
                "ffmpeg", "-y",
                "-i", tmp_vid, "-i", self.audio_path,
                "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k", "-shortest", out_path
            ], capture_output=True, text=True)
            if os.path.exists(tmp_vid): os.unlink(tmp_vid)
            if r.returncode != 0:
                raise RuntimeError(f"FFmpeg mux failed:\n{r.stderr[-600:]}")

            self._ui(lambda: self._progress.config(value=100))
            self._ui(lambda: self._prog_lbl.config(text="✔ Export complete!", fg=ACCENT))
            self._status(f"Exported: {out_path}")
            self._ui(lambda: messagebox.showinfo("Export Complete",
                f"Video saved to:\n{out_path}\n\nDrag it into the Kdenlive timeline."))
        except Exception as ex:
            err = str(ex)
            self._ui(lambda: messagebox.showerror("Export Error", err))
            self._ui(lambda: self._prog_lbl.config(text=f"Error: {err[:50]}", fg=RED))
        finally:
            self._ui(self._reset_export_ui)

    def _reset_export_ui(self):
        self._export_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)

    def _ui(self, fn):
        self.root.after(0, fn)


# ══════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass
    KdenLoquist(root)
    root.mainloop()

if __name__ == "__main__":
    main()
