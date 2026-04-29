#!/usr/bin/env python3
"""
============================================================
  KdenLoquist — Audio-Synced Talking Tool for Kdenlive
  Inspired by ProLoquist Volume 2 (FCPX)
  
  Workflow:
    1. Load an image (photo, cartoon, etc.)
    2. Draw a mouth mask by clicking & dragging on the canvas
    3. Load an audio file (WAV / MP3 / FLAC / OGG)
    4. Adjust animation controls
    5. Preview a test frame
    6. Export an MP4 ready to import into Kdenlive
============================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import cv2
import os
import threading
import subprocess
import wave
import struct
import tempfile

# ─── scipy imports ──────────────────────────────────────
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, sosfiltfilt
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


# ══════════════════════════════════════════════════════════
#  Custom Widgets
# ══════════════════════════════════════════════════════════

class StyledButton(tk.Button):
    def __init__(self, parent, text, command=None, accent=False, danger=False, **kw):
        color = ACCENT if accent else (RED if danger else BTN_BG)
        fg    = BG   if accent or danger else FG
        super().__init__(
            parent, text=text, command=command,
            bg=color, fg=fg, activebackground=ACCENT2,
            activeforeground=FG, relief=tk.FLAT,
            font=("Segoe UI", 9, "bold" if accent else "normal"),
            padx=8, pady=5, cursor="hand2", **kw
        )
        self.bind("<Enter>", lambda e: self.config(bg=BTN_HOV if not accent and not danger else ACCENT2))
        self.bind("<Leave>", lambda e: self.config(bg=color))


class LabeledSlider(tk.Frame):
    def __init__(self, parent, label, var, from_, to, fmt="{:.2f}", **kw):
        super().__init__(parent, bg=PANEL_BG, **kw)
        self.var = var
        self.fmt = fmt
        top = tk.Frame(self, bg=PANEL_BG)
        top.pack(fill=tk.X)
        tk.Label(top, text=label, bg=PANEL_BG, fg=FG_DIM,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.val_lbl = tk.Label(top, text=fmt.format(var.get()),
                                bg=PANEL_BG, fg=ACCENT, font=("Segoe UI", 8, "bold"))
        self.val_lbl.pack(side=tk.RIGHT)
        tk.Scale(self, variable=var, from_=from_, to=to,
                 orient=tk.HORIZONTAL, showvalue=False,
                 bg=PANEL_BG, fg=ACCENT, troughcolor=BTN_BG,
                 highlightthickness=0, bd=0,
                 command=self._on_change).pack(fill=tk.X)

    def _on_change(self, _=None):
        self.val_lbl.config(text=self.fmt.format(self.var.get()))


class Section(tk.Frame):
    def __init__(self, parent, title, **kw):
        super().__init__(parent, bg=PANEL_BG, **kw)
        hdr = tk.Frame(self, bg=ACCENT2)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=f"  {title}", bg=ACCENT2, fg=FG,
                 font=("Segoe UI", 9, "bold"), pady=4).pack(side=tk.LEFT)
        self.body = tk.Frame(self, bg=PANEL_BG, padx=8, pady=6)
        self.body.pack(fill=tk.X)


# ══════════════════════════════════════════════════════════
#  Audio helpers
# ══════════════════════════════════════════════════════════

def decode_audio_to_wav(src_path: str) -> str:
    """Convert any audio format to a temporary WAV via ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-y", "-i", src_path, "-ar", "44100",
           "-ac", "1", "-f", "wav", tmp.name]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr.decode()}")
    return tmp.name


def load_audio(path: str):
    """Return (samples_float32, sample_rate). Handles any format via ffmpeg."""
    wav_path = path
    tmp_created = False
    if not path.lower().endswith(".wav"):
        wav_path = decode_audio_to_wav(path)
        tmp_created = True
    sr, data = wavfile.read(wav_path)
    if tmp_created:
        os.unlink(wav_path)
    # Normalise to float [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, int(sr)


def bandpass_rms_envelope(audio: np.ndarray, sr: int, fps: int,
                           f_low: float, f_high: float,
                           smoothing: float, offset_frames: int,
                           anim_amount: float) -> np.ndarray:
    """Compute per-frame RMS amplitude in a vocal frequency band."""
    # Clamp frequencies
    nyq = sr / 2.0
    f_low  = max(20.0, min(f_low,  nyq * 0.95))
    f_high = max(f_low + 10, min(f_high, nyq * 0.99))

    sos = butter(4, [f_low / nyq, f_high / nyq], btype="band", output="sos")
    filtered = sosfiltfilt(sos, audio)

    samples_per_frame = sr / fps
    duration_frames   = int(np.ceil(len(audio) / samples_per_frame))
    envelope = np.zeros(duration_frames, dtype=np.float32)

    for i in range(duration_frames):
        af = i - offset_frames
        if af < 0:
            continue
        s = int(af * samples_per_frame)
        e = int(s + samples_per_frame)
        chunk = filtered[s:min(e, len(filtered))]
        if len(chunk) == 0:
            continue
        envelope[i] = np.sqrt(np.mean(chunk ** 2))

    # Normalise
    mx = envelope.max()
    if mx > 0:
        envelope /= mx

    # Smooth
    if smoothing > 0:
        sigma = smoothing * 8.0
        envelope = gaussian_filter1d(envelope, sigma=sigma)
        mx2 = envelope.max()
        if mx2 > 0:
            envelope /= mx2

    # Scale by animation amount
    envelope *= anim_amount
    return envelope


# ══════════════════════════════════════════════════════════
#  Frame renderer
# ══════════════════════════════════════════════════════════

def render_frame(base_rgba: np.ndarray, mouth_rect,
                 amplitude: float, softness: float,
                 inner_color: tuple) -> np.ndarray:
    """
    Animate the mouth region.
    mouth_rect: (x1, y1, x2, y2) in image pixels
    amplitude:  0..1 how wide open the mouth is
    inner_color: (R, G, B) — colour of the inner mouth (default near-black)
    """
    result = base_rgba.copy()
    if mouth_rect is None or amplitude <= 0.001:
        return result

    x1, y1, x2, y2 = mouth_rect
    mw = x2 - x1
    mh = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    h_img, w_img = result.shape[:2]

    # Ellipse half-axes
    half_w = max(1, int(mw * 0.42))
    half_h = max(1, int(mh * 0.50 * amplitude))

    # Draw mask on float array
    mask = np.zeros((h_img, w_img), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (half_w, half_h), 0, 0, 360, 1.0, -1)

    # Soft edges
    if softness > 0:
        k = int(softness) * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), softness * 0.5)

    # Blend: original → inner_color
    ir, ig, ib = inner_color
    inner = np.full_like(result, 0)
    inner[:, :, 0] = ir
    inner[:, :, 1] = ig
    inner[:, :, 2] = ib
    inner[:, :, 3] = 255

    m4 = mask[:, :, np.newaxis]
    result = (result.astype(np.float32) * (1.0 - m4) +
              inner.astype(np.float32) * m4).clip(0, 255).astype(np.uint8)

    # Preserve original alpha
    result[:, :, 3] = base_rgba[:, :, 3]
    return result


# ══════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════

class KdenLoquist:
    APP_TITLE = "KdenLoquist — Audio-Synced Talking Tool for Kdenlive"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(self.APP_TITLE)
        self.root.geometry("1280x780")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)

        # State
        self.image: Image.Image | None = None
        self.image_path: str | None = None
        self.base_rgba: np.ndarray | None = None  # H×W×4 uint8
        self.audio_data: np.ndarray | None = None
        self.sample_rate: int = 44100
        self.audio_path: str | None = None
        self.audio_duration: float = 0.0

        self.mouth_rect: tuple | None = None   # (x1, y1, x2, y2) image coords
        self._draw_start: tuple | None = None
        self._temp_rect_id = None

        # Display helpers
        self._disp_scale   = 1.0
        self._disp_off_x   = 0
        self._disp_off_y   = 0
        self._tk_image     = None
        self._preview_open = False

        self._export_thread: threading.Thread | None = None
        self._cancel_export = False

        self._build_ui()

    # ──────────────────────────────────────────────────────
    #  UI Construction
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
        tb.pack(fill=tk.X)
        tb.pack_propagate(False)
        tk.Label(tb, text="  🎭  KdenLoquist", bg=ACCENT2, fg=FG,
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=6)
        tk.Label(tb, text="Audio-Synced Talking Tool  •  for Kdenlive",
                 bg=ACCENT2, fg="#ccc", font=("Segoe UI", 9)).pack(side=tk.LEFT)

    def _build_left_panel(self, parent):
        self._panel = tk.Frame(parent, bg=PANEL_BG, width=270)
        self._panel.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 0))
        self._panel.pack_propagate(False)

        scroll_canvas = tk.Canvas(self._panel, bg=PANEL_BG,
                                  highlightthickness=0, width=268)
        scrollbar = ttk.Scrollbar(self._panel, orient="vertical",
                                  command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(scroll_canvas, bg=PANEL_BG)
        win_id = scroll_canvas.create_window((0, 0), window=inner,
                                              anchor="nw", width=252)
        inner.bind("<Configure>", lambda e: scroll_canvas.configure(
            scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.bind("<Configure>",
            lambda e: scroll_canvas.itemconfig(win_id, width=e.width))

        self._build_panel_contents(inner)

    def _build_panel_contents(self, p):
        pad = dict(fill=tk.X, pady=3, padx=4)

        # ── FILES ──────────────────────────────────────────
        s = Section(p, "📁  Files")
        s.pack(**pad)
        b = s.body
        StyledButton(b, "📷  Load Image", command=self._load_image,
                     accent=True).pack(fill=tk.X, pady=2)
        self._img_lbl = tk.Label(b, text="No image loaded",
                                 bg=PANEL_BG, fg=FG_DIM,
                                 font=("Segoe UI", 8), wraplength=220)
        self._img_lbl.pack(anchor=tk.W)

        StyledButton(b, "🎵  Load Audio", command=self._load_audio,
                     accent=True).pack(fill=tk.X, pady=2)
        self._aud_lbl = tk.Label(b, text="No audio loaded",
                                 bg=PANEL_BG, fg=FG_DIM,
                                 font=("Segoe UI", 8), wraplength=220)
        self._aud_lbl.pack(anchor=tk.W)

        # ── MOUTH MASK ─────────────────────────────────────
        s2 = Section(p, "🖊  Mouth Mask")
        s2.pack(**pad)
        b2 = s2.body
        tk.Label(b2, text="Click & drag on the image to draw\nthe mouth region.",
                 bg=PANEL_BG, fg=FG_DIM, font=("Segoe UI", 8),
                 justify=tk.LEFT).pack(anchor=tk.W)
        StyledButton(b2, "🗑  Clear Mask", command=self._clear_mask,
                     danger=True).pack(fill=tk.X, pady=(4, 2))
        self._mask_lbl = tk.Label(b2, text="No mask drawn",
                                  bg=PANEL_BG, fg=FG_DIM,
                                  font=("Segoe UI", 8, "italic"))
        self._mask_lbl.pack(anchor=tk.W)

        # ── ANIMATION ──────────────────────────────────────
        s3 = Section(p, "🎬  Animation")
        s3.pack(**pad)
        b3 = s3.body

        self.v_anim   = tk.DoubleVar(value=0.65)
        self.v_smooth = tk.DoubleVar(value=0.35)
        self.v_flo    = tk.DoubleVar(value=80.0)
        self.v_fhi    = tk.DoubleVar(value=3500.0)
        self.v_offset = tk.IntVar(value=0)
        self.v_soft   = tk.DoubleVar(value=7.0)

        LabeledSlider(b3, "Animation Amount", self.v_anim,
                      0.05, 1.0).pack(fill=tk.X, pady=2)
        LabeledSlider(b3, "Smoothing",        self.v_smooth,
                      0.0,  1.0).pack(fill=tk.X, pady=2)
        LabeledSlider(b3, "Freq Low (Hz)",    self.v_flo,
                      20,   1000, fmt="{:.0f} Hz").pack(fill=tk.X, pady=2)
        LabeledSlider(b3, "Freq High (Hz)",   self.v_fhi,
                      500,  8000, fmt="{:.0f} Hz").pack(fill=tk.X, pady=2)
        LabeledSlider(b3, "Edge Softness",    self.v_soft,
                      0,    30,   fmt="{:.1f}").pack(fill=tk.X, pady=2)

        of_row = tk.Frame(b3, bg=PANEL_BG)
        of_row.pack(fill=tk.X, pady=2)
        tk.Label(of_row, text="Audio Offset (frames)",
                 bg=PANEL_BG, fg=FG_DIM,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Spinbox(of_row, from_=0, to=999, textvariable=self.v_offset,
                   width=5, bg=BTN_BG, fg=FG,
                   buttonbackground=PANEL_BG,
                   highlightthickness=0).pack(side=tk.RIGHT)

        # Inner mouth colour
        clr_row = tk.Frame(b3, bg=PANEL_BG)
        clr_row.pack(fill=tk.X, pady=2)
        tk.Label(clr_row, text="Inner Mouth Colour",
                 bg=PANEL_BG, fg=FG_DIM,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.v_mouth_color = tk.StringVar(value="#1a0008")
        self._color_btn = tk.Button(
            clr_row, bg=self.v_mouth_color.get(), width=3,
            relief=tk.FLAT, cursor="hand2",
            command=self._pick_color)
        self._color_btn.pack(side=tk.RIGHT)

        # ── OUTPUT ─────────────────────────────────────────
        s4 = Section(p, "⚙  Output")
        s4.pack(**pad)
        b4 = s4.body

        fps_row = tk.Frame(b4, bg=PANEL_BG)
        fps_row.pack(fill=tk.X, pady=2)
        tk.Label(fps_row, text="Frame Rate (FPS)",
                 bg=PANEL_BG, fg=FG_DIM,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.v_fps = tk.IntVar(value=24)
        ttk.Combobox(fps_row, textvariable=self.v_fps, width=5,
                     values=[12, 23, 24, 25, 29, 30, 48, 50, 60],
                     state="readonly").pack(side=tk.RIGHT)

        self.v_crf = tk.IntVar(value=20)
        LabeledSlider(b4, "Export Quality (CRF)", self.v_crf,
                      0, 51, fmt="{:.0f}  (lower=better)").pack(fill=tk.X, pady=2)

        # ── ACTIONS ────────────────────────────────────────
        s5 = Section(p, "▶  Actions")
        s5.pack(**pad)
        b5 = s5.body

        StyledButton(b5, "👁  Preview Frame",
                     command=self._preview_frame).pack(fill=tk.X, pady=2)
        self._export_btn = StyledButton(b5, "🎬  Export Video",
                                        command=self._start_export, accent=True)
        self._export_btn.pack(fill=tk.X, pady=2)
        self._cancel_btn = StyledButton(b5, "✖  Cancel",
                                        command=self._cancel,
                                        danger=True)
        self._cancel_btn.pack(fill=tk.X, pady=2)
        self._cancel_btn.config(state=tk.DISABLED)

        self._progress = ttk.Progressbar(b5, mode="determinate",
                                         maximum=100)
        self._progress.pack(fill=tk.X, pady=4)
        self._prog_lbl = tk.Label(b5, text="Ready.",
                                  bg=PANEL_BG, fg=FG_DIM,
                                  font=("Segoe UI", 8))
        self._prog_lbl.pack(anchor=tk.W)

        # ── HELP ───────────────────────────────────────────
        s6 = Section(p, "ℹ  Quick Help")
        s6.pack(**pad)
        tk.Label(s6.body,
                 text=(
                     "1. Load Image → draw mouth box\n"
                     "2. Load Audio\n"
                     "3. Tune sliders → Preview\n"
                     "4. Export MP4\n"
                     "5. Drag MP4 into Kdenlive timeline\n\n"
                     "Tip: Use 'Freq Low/High' to isolate\n"
                     "speech frequencies (80–3500 Hz).\n"
                     "Lower smoothing = snappier mouth."
                 ),
                 bg=PANEL_BG, fg=FG_DIM,
                 font=("Segoe UI", 8), justify=tk.LEFT,
                 wraplength=230).pack(anchor=tk.W)

    def _build_canvas(self, parent):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        lbl = tk.Label(frame,
                       text="← Load an image, then click & drag to mark the mouth region",
                       bg=BG, fg=FG_DIM, font=("Segoe UI", 9))
        lbl.pack(anchor=tk.W, padx=6, pady=(4, 0))

        self._canvas = tk.Canvas(frame, bg="#090912",
                                  highlightthickness=1,
                                  highlightbackground=ACCENT2,
                                  cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",        self._on_drag)
        self._canvas.bind("<ButtonRelease-1>",  self._on_release)
        self._canvas.bind("<Configure>",        self._on_canvas_resize)

        # Placeholder
        self._canvas.after(200, self._draw_placeholder)

    def _build_statusbar(self):
        self._statusbar = tk.Label(self.root, text="Ready",
                                   bg=ACCENT2, fg=BG,
                                   font=("Segoe UI", 8), anchor=tk.W,
                                   padx=8, pady=3)
        self._statusbar.pack(fill=tk.X, side=tk.BOTTOM)

    def _status(self, msg: str, color=None):
        self._statusbar.config(text=msg,
                               fg=color or BG)
        self.root.update_idletasks()

    # ──────────────────────────────────────────────────────
    #  File Loading
    # ──────────────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("All", "*.*")])
        if not path:
            return
        try:
            self.image      = Image.open(path).convert("RGBA")
            self.image_path = path
            self.base_rgba  = np.array(self.image, dtype=np.uint8)
            self.mouth_rect = None
            self._mask_lbl.config(text="No mask drawn", fg=FG_DIM)
            short = os.path.basename(path)
            w, h  = self.image.size
            self._img_lbl.config(
                text=f"{short}\n{w}×{h} px", fg=ACCENT)
            self._status(f"Image loaded: {short}  ({w}×{h})")
            self._refresh_canvas()
        except Exception as ex:
            messagebox.showerror("Image Error", str(ex))

    def _load_audio(self):
        path = filedialog.askopenfilename(
            title="Select Audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.aac *.m4a"),
                       ("All", "*.*")])
        if not path:
            return
        self._status("Loading audio…")
        try:
            data, sr = load_audio(path)
            self.audio_data   = data
            self.sample_rate  = sr
            self.audio_path   = path
            self.audio_duration = len(data) / sr
            short = os.path.basename(path)
            self._aud_lbl.config(
                text=f"{short}\n{self.audio_duration:.1f}s  •  {sr} Hz",
                fg=ACCENT)
            self._status(f"Audio loaded: {short}  ({self.audio_duration:.1f}s)")
        except Exception as ex:
            messagebox.showerror("Audio Error", str(ex))
            self._status("Audio load failed.", color=RED)

    # ──────────────────────────────────────────────────────
    #  Colour picker
    # ──────────────────────────────────────────────────────

    def _pick_color(self):
        from tkinter.colorchooser import askcolor
        color = askcolor(color=self.v_mouth_color.get(),
                         title="Inner Mouth Colour")[1]
        if color:
            self.v_mouth_color.set(color)
            self._color_btn.config(bg=color)

    # ──────────────────────────────────────────────────────
    #  Canvas / Mask drawing
    # ──────────────────────────────────────────────────────

    def _draw_placeholder(self):
        if self.image is None:
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            self._canvas.delete("all")
            self._canvas.create_text(
                cw // 2, ch // 2,
                text="Load an image to begin",
                fill=FG_DIM, font=("Segoe UI", 16))

    def _img_to_canvas(self, ix, iy):
        return (ix * self._disp_scale + self._disp_off_x,
                iy * self._disp_scale + self._disp_off_y)

    def _canvas_to_img(self, cx, cy):
        return ((cx - self._disp_off_x) / self._disp_scale,
                (cy - self._disp_off_y) / self._disp_scale)

    def _refresh_canvas(self, overlay: np.ndarray = None):
        if self.image is None:
            return
        cw = max(self._canvas.winfo_width(),  100)
        ch = max(self._canvas.winfo_height(), 100)

        src = Image.fromarray(overlay) if overlay is not None else self.image
        iw, ih = src.size

        scale = min(cw / iw, ch / ih, 1.0)
        dw, dh = int(iw * scale), int(ih * scale)
        self._disp_scale  = scale
        self._disp_off_x  = (cw - dw) // 2
        self._disp_off_y  = (ch - dh) // 2

        display_img = src.resize((dw, dh), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(display_img)

        self._canvas.delete("all")
        self._canvas.create_image(
            self._disp_off_x, self._disp_off_y,
            anchor=tk.NW, image=self._tk_image)

        # Redraw mask overlay
        if self.mouth_rect:
            x1, y1, x2, y2 = self.mouth_rect
            cx1, cy1 = self._img_to_canvas(x1, y1)
            cx2, cy2 = self._img_to_canvas(x2, y2)
            self._canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=ACCENT, width=2, dash=(6, 3))
            self._canvas.create_oval(
                cx1, cy1, cx2, cy2,
                outline=YELLOW, width=1, dash=(4, 4))
            mid_x = (cx1 + cx2) / 2
            self._canvas.create_text(
                mid_x, cy1 - 10,
                text="MOUTH MASK", fill=ACCENT,
                font=("Segoe UI", 8, "bold"))

    def _on_canvas_resize(self, _):
        self._refresh_canvas()

    def _on_press(self, e):
        if self.image is None:
            return
        self._draw_start = (e.x, e.y)
        if self._temp_rect_id:
            self._canvas.delete(self._temp_rect_id)

    def _on_drag(self, e):
        if not self._draw_start:
            return
        if self._temp_rect_id:
            self._canvas.delete(self._temp_rect_id)
        sx, sy = self._draw_start
        self._temp_rect_id = self._canvas.create_rectangle(
            sx, sy, e.x, e.y,
            outline=ACCENT, width=2, dash=(6, 3))

    def _on_release(self, e):
        if not self._draw_start or self.image is None:
            return
        sx, sy = self._draw_start
        self._draw_start = None
        if self._temp_rect_id:
            self._canvas.delete(self._temp_rect_id)
            self._temp_rect_id = None

        # Convert to image coords
        ix1, iy1 = self._canvas_to_img(min(sx, e.x), min(sy, e.y))
        ix2, iy2 = self._canvas_to_img(max(sx, e.x), max(sy, e.y))

        iw, ih = self.image.size
        ix1 = int(max(0, min(ix1, iw)))
        iy1 = int(max(0, min(iy1, ih)))
        ix2 = int(max(0, min(ix2, iw)))
        iy2 = int(max(0, min(iy2, ih)))

        if (ix2 - ix1) < 5 or (iy2 - iy1) < 5:
            return

        self.mouth_rect = (ix1, iy1, ix2, iy2)
        mw, mh = ix2 - ix1, iy2 - iy1
        self._mask_lbl.config(
            text=f"Mask set: {mw}×{mh} px  @ ({ix1},{iy1})",
            fg=ACCENT)
        self._refresh_canvas()

    def _clear_mask(self):
        self.mouth_rect = None
        self._mask_lbl.config(text="No mask drawn", fg=FG_DIM)
        self._refresh_canvas()

    # ──────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _get_envelope(self, n_frames: int) -> np.ndarray:
        return bandpass_rms_envelope(
            self.audio_data, self.sample_rate,
            self.v_fps.get(),
            self.v_flo.get(), self.v_fhi.get(),
            self.v_smooth.get(), self.v_offset.get(),
            self.v_anim.get())

    def _make_frame(self, amplitude: float) -> np.ndarray:
        inner_rgb = self._hex_to_rgb(self.v_mouth_color.get())
        return render_frame(
            self.base_rgba, self.mouth_rect,
            amplitude, self.v_soft.get(), inner_rgb)

    # ──────────────────────────────────────────────────────
    #  Preview
    # ──────────────────────────────────────────────────────

    def _preview_frame(self):
        if self.image is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        if self.mouth_rect is None:
            messagebox.showwarning("No mask", "Please draw a mouth mask first.")
            return
        frame = self._make_frame(0.75)
        self._refresh_canvas(Image.fromarray(frame))
        self._status("Preview rendered at 75% amplitude.")

    # ──────────────────────────────────────────────────────
    #  Export
    # ──────────────────────────────────────────────────────

    def _start_export(self):
        if self.image is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        if self.audio_data is None:
            messagebox.showwarning("No audio", "Please load an audio file first.")
            return
        if self.mouth_rect is None:
            messagebox.showwarning("No mask", "Please draw a mouth mask on the image.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save exported video",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"),
                       ("Matroska Video", "*.mkv")])
        if not out_path:
            return

        self._export_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._cancel_export = False
        self._progress["value"] = 0
        self._prog_lbl.config(text="Starting…", fg=FG_DIM)

        self._export_thread = threading.Thread(
            target=self._do_export, args=(out_path,), daemon=True)
        self._export_thread.start()

    def _cancel(self):
        self._cancel_export = True
        self._prog_lbl.config(text="Cancelling…", fg=YELLOW)
        self._status("Cancelling export…")

    def _do_export(self, out_path: str):
        try:
            fps      = self.v_fps.get()
            crf      = self.v_crf.get()
            iw, ih   = self.image.size
            n_frames = int(np.ceil(self.audio_duration * fps))

            self._ui(lambda: self._prog_lbl.config(
                text="Analysing audio…", fg=FG_DIM))
            self._status("Analysing audio amplitude…")

            envelope = self._get_envelope(n_frames)
            n_frames = min(n_frames, len(envelope))

            # Write raw video to temp file
            tmp_vid = out_path + "_tmp_raw.mp4"
            fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
            writer  = cv2.VideoWriter(tmp_vid, fourcc, fps, (iw, ih))

            for i in range(n_frames):
                if self._cancel_export:
                    writer.release()
                    if os.path.exists(tmp_vid):
                        os.unlink(tmp_vid)
                    self._ui(lambda: self._prog_lbl.config(
                        text="Cancelled.", fg=YELLOW))
                    self._status("Export cancelled.")
                    self._ui(self._reset_export_ui)
                    return

                amp   = float(envelope[i])
                frame = self._make_frame(amp)
                # RGBA → BGR
                bgr   = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                writer.write(bgr)

                if i % max(1, n_frames // 80) == 0:
                    pct = int(i / n_frames * 80)
                    msg = f"Rendering frame {i+1}/{n_frames}  ({pct}%)"
                    self._ui(lambda p=pct, m=msg: (
                        self._progress.config(value=p),
                        self._prog_lbl.config(text=m, fg=FG_DIM)
                    ))

            writer.release()

            # Mux audio
            self._ui(lambda: self._prog_lbl.config(
                text="Muxing audio with ffmpeg…", fg=FG_DIM))
            self._status("Muxing audio…")
            self._ui(lambda: self._progress.config(value=85))

            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_vid,
                "-i", self.audio_path,
                "-c:v", "libx264",
                "-crf", str(crf),
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                out_path
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if os.path.exists(tmp_vid):
                os.unlink(tmp_vid)

            if r.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg mux failed:\n{r.stderr[-600:]}")

            self._ui(lambda: self._progress.config(value=100))
            self._ui(lambda: self._prog_lbl.config(
                text="✔ Export complete!", fg=ACCENT))
            self._status(f"Exported: {out_path}")
            self._ui(lambda: messagebox.showinfo(
                "Export Complete",
                f"Video saved to:\n{out_path}\n\n"
                "You can now drag it into the Kdenlive timeline."))
        except Exception as ex:
            err = str(ex)
            self._ui(lambda: messagebox.showerror("Export Error", err))
            self._ui(lambda: self._prog_lbl.config(
                text=f"Error: {err[:50]}", fg=RED))
            self._status(f"Export error: {err[:60]}", color=RED)
        finally:
            self._ui(self._reset_export_ui)

    def _reset_export_ui(self):
        self._export_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)

    def _ui(self, fn):
        """Schedule fn on the main thread."""
        self.root.after(0, fn)


# ══════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass
    app = KdenLoquist(root)
    root.mainloop()


if __name__ == "__main__":
    main()
