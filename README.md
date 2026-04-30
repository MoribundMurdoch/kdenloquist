# 🎭 KdenLoquist — Audio-Synced Talking Tool for Kdenlive

> A ProLoquist Volume 2-style ventriloquist/talking-photo effect tool,
> built specifically for Kdenlive users.

---

## What it does

KdenLoquist animates a **mouth region** on any image so that it opens and
closes in sync with a spoken/sung audio track — no keyframes needed.
The output is a standard **MP4** file you can drop straight into the
Kdenlive timeline.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python      | 3.9+    |
| numpy       | ≥ 1.24  |
| scipy       | ≥ 1.10  |
| Pillow      | ≥ 10.0  |
| opencv-python | ≥ 4.8 |
| **ffmpeg**  | any recent (must be in PATH) |

Install Python packages:
```bash
pip install -r requirements.txt
```

Install ffmpeg (if not already):
```bash
# Ubuntu/Debian
sudo apt install ffmpeg
# Arch Linux
sudo pacman -S ffmpeg
# macOS (Homebrew)
brew install ffmpeg
# Windows — download from https://ffmpeg.org/download.html
```
```

---

## Quick Start

```bash
python kdenloquist.py
```

### Workflow

1. **Load Image** — any photo, cartoon, or illustration (PNG / JPG / etc.)
2. **Draw Mouth Mask** — click and drag a rectangle over the mouth area
3. **Load Audio** — WAV, MP3, FLAC, OGG, AAC (MP3+ require ffmpeg)
4. **Adjust sliders** (see Controls below)
5. **Preview Frame** — see a single frame at 75% amplitude
6. **Export Video** — renders full animation → MP4 with audio
7. **Import into Kdenlive** — drag the MP4 onto your Kdenlive timeline

---

## Controls

| Control | Description |
|---------|-------------|
| **Animation Amount** | How wide the mouth opens at peak volume (0–1) |
| **Smoothing** | Temporal smoothing — higher = more relaxed movement |
| **Freq Low / High** | Band-pass filter to isolate vocals (default 80–3500 Hz) |
| **Edge Softness** | Feathering of the mouth mask edges |
| **Audio Offset** | Shift audio by N frames (compensates for sync delay) |
| **Inner Mouth Colour** | Colour shown inside the mouth (default near-black) |
| **FPS** | Output frame rate (match your Kdenlive project) |
| **Export Quality (CRF)** | 0=lossless, 18=great, 28=small file |

---

## Tips

- **Cartoon / artwork** — works best with a clear oval or rectangular mouth area.
- **Real photos** — draw the mask tightly around the lips.
- **Speech vs. singing** — for speech keep Freq High ≤ 4000 Hz; for singing
  try 150–8000 Hz.
- **Smoothing** — set around 0.3 for speech, 0.5 for singing.
- **Kdenlive workflow** — place the exported MP4 on a track above your
  original image/clip; use Kdenlive's built-in masking or blend modes
  if you need to composite it.

---

## Limitations / Roadmap

- [ ] Bezier mouth mask (currently rectangle → ellipse)
- [ ] Video input (not just static images)
- [ ] Real-time audio waveform preview
- [ ] Kdenlive `.mlt` project export with keyframes
- [ ] Teeth / tongue layer

---

## Licence

MIT — free to use, modify, and distribute.
