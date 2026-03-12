# ComfyUI Depth Warp 6-DOF

A ComfyUI custom node for depth-based geometric view synthesis.  
Uses a depth map (e.g. from Depth Anything V2) to warp a background image to a new camera position, producing realistic parallax for multi-camera virtual studio setups.

---

## How It Works

Standard PTZ rotation only pans/tilts the image — no parallax. This node supports full **6-DOF camera transforms** (rotation + translation), so nearby objects shift more than distant ones, just like a real camera position change.

```
Translation ON  → parallax (near objects move more)
Rotation only   → no parallax (flat pan/tilt)
```

Occluded regions revealed by the warp are output as a **hole mask**, ready to be inpainted with FLUX Fill or any other inpainting model.

---

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/moondive-cinema/comfyui-depth-warp.git
# Restart ComfyUI
```

Node location: `MagicHour/Background` → **Depth Warp 6-DOF (MagicHour)**

---

## Parameters

### Camera Intrinsics
| Parameter | Default | Description |
|-----------|---------|-------------|
| `focal_length` | 1000 | Focal length in pixels. Higher = narrower FOV. |

### Translation (produces parallax)
| Parameter | Range | Description |
|-----------|-------|-------------|
| `t_x` | ±3.0 | Lateral (left/right) shift. The primary parameter for multi-camera setups. |
| `t_y` | ±3.0 | Vertical shift. |
| `t_z` | ±3.0 | Forward/backward shift. Similar to a zoom effect. |

> **Unit:** Relative to the mean scene depth (= 1.0). `t_x = 0.3` means the camera moved 30% of the average subject distance. Tune visually — absolute metric values are not available from monocular depth.

### Rotation
| Parameter | Range | Description |
|-----------|-------|-------------|
| `yaw`   | ±45° | Pan left/right. Typically used together with `t_x`. |
| `pitch` | ±45° | Tilt up/down. |
| `roll`  | ±45° | Roll. Usually 0. |

### Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth_invert` | True | Set True for Depth Anything V2 output (disparity — higher = closer). Set False if your depth map uses the opposite convention. |
| `mask_dilate_px` | 20 | Hole mask dilation in pixels. Expand the inpainting region slightly for cleaner FLUX Fill results. |

---

## Workflow

```
LoadImage (master background)
    └→ Depth Anything V2
            └→ Depth Warp 6-DOF  ──→ warped_image ──┐
                                  └→ hole_mask    ──┤→ FluxFillConditioning → KSampler → VAE Decode
                                                     └ CLIPTextEncode (background prompt)
```

---

## Multi-Camera Studio Example

Generating left and right camera backgrounds from a master center shot:

```
Master (center):  t_x= 0.0,  yaw=  0.0
Left camera:      t_x=-0.15, yaw= +8.0   (camera is left, pointing right)
Right camera:     t_x=+0.15, yaw= -8.0   (camera is right, pointing left)
```

Suggested starting values for interview / talk show setups:

```
t_x            = 0.05 ~ 0.15
yaw            = 5 ~ 15°
focal_length   = 1200 ~ 2000
mask_dilate_px = 20 ~ 30
```

---

## Standalone Test (no ComfyUI required)

```bash
python test_warp.py --image background.jpg --t_x 0.1 --yaw 8 --out result.jpg
```

Output: side-by-side image `[original | warped | hole mask]`  
If no depth image is provided, a synthetic depth map is used automatically.

---

## Known Limitations

- **Monocular depth scale:** Depth Anything V2 outputs relative depth only. Translation values have no absolute metric meaning — tune visually per scene.
- **Occluded regions:** Areas hidden behind objects in the source image are filled by inpainting. Results are plausible but not geometrically accurate. For small camera offsets (typical in fixed studio setups) this is generally acceptable.
- **Performance:** ~2–4s per 1080p image on CPU. GPU acceleration not implemented — background generation is offline so this is not a bottleneck.
