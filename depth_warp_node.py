"""
Depth Warp Node for ComfyUI
Magic Hour Inc. — Virtual Studio Background Pipeline

6-DOF camera transform (Rotation + Translation) with depth-based geometric warp.
Translation produces parallax — closer objects shift more than farther ones.

Input:
  - image  : source background image  (B, H, W, 3) float32 0~1
  - depth  : depth map from Depth Anything V2  (B, H, W, C)
Output:
  - warped_image : geometrically warped image (B, H, W, 3)
  - hole_mask    : occlusion / out-of-bounds mask (B, H, W)  — feed to FLUX Fill
"""

import numpy as np
import torch
import cv2


# ──────────────────────────────────────────────────────────────
# Rotation helpers
# ──────────────────────────────────────────────────────────────

def _Rx(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0],
                     [0,  np.cos(a), -np.sin(a)],
                     [0,  np.sin(a),  np.cos(a)]], dtype=np.float32)

def _Ry(deg):
    a = np.radians(deg)
    return np.array([[ np.cos(a), 0, np.sin(a)],
                     [0,          1, 0         ],
                     [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)

def _Rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0,          0,          1]], dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# Core warp (single frame)
# ──────────────────────────────────────────────────────────────

def _warp_frame(img_np, dep_np, focal_length,
                t_x, t_y, t_z,
                yaw, pitch, roll,
                depth_invert, mask_dilate_px):
    """
    img_np  : (H, W, 3) float32
    dep_np  : (H, W)    float32   0~1
    Returns : warped (H, W, 3), hole_mask (H, W) float32
    """
    H, W = dep_np.shape
    cx, cy = W / 2.0, H / 2.0
    fx = fy = focal_length

    # ── Depth 해석 ──────────────────────────────────────────────
    # Depth Anything V2 출력: 높을수록 가깝다 (disparity, inverse-depth)
    # depth_invert=True  → 그대로 사용 (disparity → metric: 1/d)
    # depth_invert=False → 이미 depth (클수록 멀다)
    if depth_invert:
        # disparity → relative metric depth
        depth_metric = 1.0 / (dep_np + 1e-6)
    else:
        depth_metric = dep_np + 1e-6

    # mean=1로 정규화 → t_x, t_y 단위가 "평균 거리" 기준으로 직관적
    depth_metric = depth_metric / (depth_metric.mean() + 1e-6)

    # ── Pixel → 3D ─────────────────────────────────────────────
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
    X = (u_grid - cx) * depth_metric / fx   # (H, W)
    Y = (v_grid - cy) * depth_metric / fy
    Z = depth_metric                          # (H, W)

    # ── Rotation ────────────────────────────────────────────────
    R = _Ry(yaw) @ _Rx(pitch) @ _Rz(roll)   # (3, 3)

    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)  # (3, N)
    pts_rot = R @ pts                                            # (3, N)

    # ── Translation ─────────────────────────────────────────────
    # 카메라가 (t_x, t_y, t_z)만큼 이동 → 씬이 반대 방향으로 이동
    pts_rot[0] -= t_x
    pts_rot[1] -= t_y
    pts_rot[2] -= t_z

    # ── Perspective projection ──────────────────────────────────
    Zp = pts_rot[2]
    valid = Zp > 1e-4

    u_dst = np.full(H * W, -1.0, dtype=np.float32)
    v_dst = np.full(H * W, -1.0, dtype=np.float32)
    u_dst[valid] = fx * pts_rot[0, valid] / Zp[valid] + cx
    v_dst[valid] = fy * pts_rot[1, valid] / Zp[valid] + cy

    u_dst = u_dst.reshape(H, W)
    v_dst = v_dst.reshape(H, W)

    # ── Forward warp with Z-buffer (painter's algorithm) ────────
    # 먼 픽셀 먼저 → 가까운 픽셀 나중에 덮어쓰기
    depth_flat = depth_metric.ravel()
    sort_idx = np.argsort(depth_flat)          # ascending: 먼 것부터 (metric이 클수록 멀다)

    warped  = np.zeros_like(img_np)
    zbuffer = np.full((H, W), np.inf, dtype=np.float32)
    weight  = np.zeros((H, W), dtype=np.float32)

    # 소스 픽셀 좌표
    v_src_all = (sort_idx // W).astype(np.int32)
    u_src_all = (sort_idx  % W).astype(np.int32)

    u_d = u_dst.ravel()[sort_idx]
    v_d = v_dst.ravel()[sort_idx]
    z_d = Zp[sort_idx]
    valid_s = valid[sort_idx]
    colors = img_np[v_src_all, u_src_all]     # (N, 3)

    # in-bounds 필터
    in_b = (valid_s &
            (u_d >= 0) & (u_d < W - 1) &
            (v_d >= 0) & (v_d < H - 1))

    u_d  = u_d[in_b];  v_d = v_d[in_b]
    z_d  = z_d[in_b];  colors = colors[in_b]

    # Bilinear splat
    u0 = np.floor(u_d).astype(np.int32)
    v0 = np.floor(v_d).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1
    wu = u_d - u0   # [0,1)
    wv = v_d - v0

    corners = [
        (u0, v0, (1 - wu) * (1 - wv)),
        (u1, v0,       wu * (1 - wv)),
        (u0, v1, (1 - wu) *       wv),
        (u1, v1,       wu *       wv),
    ]

    for (ua, va, w) in corners:
        mask_c = (ua >= 0) & (ua < W) & (va >= 0) & (va < H)
        np.add.at(warped,  (va[mask_c], ua[mask_c]), colors[mask_c] * w[mask_c, None])
        np.add.at(weight,  (va[mask_c], ua[mask_c]), w[mask_c])

    # Normalize
    filled = weight > 0
    warped[filled] /= weight[filled, None]

    # ── Hole mask ───────────────────────────────────────────────
    hole = (~filled).astype(np.uint8)

    if mask_dilate_px > 0:
        k = mask_dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        hole = cv2.dilate(hole, kernel)

    # hole 영역은 warped를 0으로 (FLUX Fill이 완전히 채우도록)
    warped[hole > 0] = 0.0

    return warped, hole.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# ComfyUI Node
# ──────────────────────────────────────────────────────────────

class DepthWarpNode:
    """
    Depth-based 6-DOF geometric warp.
    Translation → parallax (가까운 물체가 더 많이 이동).
    Rotation    → PTZ-like pan/tilt/roll.

    스튜디오 다중 카메라 사용 예:
      마스터(중앙) → t_x=0,    yaw=0
      좌측 카메라 → t_x=-0.3, yaw=+8
      우측 카메라 → t_x=+0.3, yaw=-8
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":         ("IMAGE",),
                "depth":         ("IMAGE",),

                # ── Intrinsics ──────────────────────────────────
                "focal_length":  ("FLOAT",  {"default": 1000.0,
                                             "min": 100.0, "max": 5000.0,
                                             "step": 10.0,
                                             "tooltip": "픽셀 단위 초점거리. 화각이 좁을수록 크게."}),

                # ── Translation (패럴랙스) ──────────────────────
                "t_x":           ("FLOAT",  {"default": 0.0,
                                             "min": -3.0, "max": 3.0, "step": 0.01,
                                             "tooltip": "좌우 이동. 양수=오른쪽. 패럴랙스 발생."}),
                "t_y":           ("FLOAT",  {"default": 0.0,
                                             "min": -3.0, "max": 3.0, "step": 0.01,
                                             "tooltip": "상하 이동. 양수=아래쪽."}),
                "t_z":           ("FLOAT",  {"default": 0.0,
                                             "min": -3.0, "max": 3.0, "step": 0.01,
                                             "tooltip": "전후 이동. 양수=앞으로. 줌 효과와 유사."}),

                # ── Rotation ────────────────────────────────────
                "yaw":           ("FLOAT",  {"default": 0.0,
                                             "min": -45.0, "max": 45.0, "step": 0.1,
                                             "tooltip": "좌우 pan (도). 양수=오른쪽."}),
                "pitch":         ("FLOAT",  {"default": 0.0,
                                             "min": -45.0, "max": 45.0, "step": 0.1,
                                             "tooltip": "상하 tilt (도). 양수=아래쪽."}),
                "roll":          ("FLOAT",  {"default": 0.0,
                                             "min": -45.0, "max": 45.0, "step": 0.1,
                                             "tooltip": "회전 (도). 보통 0."}),

                # ── Options ─────────────────────────────────────
                "depth_invert":  ("BOOLEAN", {"default": True,
                                              "tooltip": "True: Depth Anything V2 (disparity). False: 이미 depth map."}),
                "mask_dilate_px": ("INT",    {"default": 20,
                                              "min": 0, "max": 100, "step": 1,
                                              "tooltip": "홀 마스크 팽창(px). FLUX Fill 인페인팅 경계 처리용."}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("warped_image", "hole_mask")
    FUNCTION      = "warp"
    CATEGORY      = "MagicHour/Background"
    DESCRIPTION   = "Depth-based 6-DOF warp. Translation produces parallax for multi-camera virtual studio backgrounds."

    def warp(self, image, depth,
             focal_length,
             t_x, t_y, t_z,
             yaw, pitch, roll,
             depth_invert, mask_dilate_px):

        B = image.shape[0]
        imgs_out  = []
        masks_out = []

        for b in range(B):
            img_np = image[b].cpu().numpy().astype(np.float32)   # (H, W, 3)
            dep_raw = depth[b].cpu().numpy().astype(np.float32)  # (H, W, C)

            # depth가 multi-channel이면 첫 채널 사용
            if dep_raw.ndim == 3:
                dep_np = dep_raw[..., 0]
            else:
                dep_np = dep_raw

            # 이미지와 depth 해상도 맞추기
            H, W = img_np.shape[:2]
            if dep_np.shape != (H, W):
                dep_np = cv2.resize(dep_np, (W, H), interpolation=cv2.INTER_LINEAR)

            warped, hole = _warp_frame(
                img_np, dep_np, focal_length,
                t_x, t_y, t_z,
                yaw, pitch, roll,
                depth_invert, mask_dilate_px
            )

            imgs_out.append(torch.from_numpy(warped))
            masks_out.append(torch.from_numpy(hole))

        warped_batch = torch.stack(imgs_out,  dim=0)   # (B, H, W, 3)
        mask_batch   = torch.stack(masks_out, dim=0)   # (B, H, W)

        return (warped_batch, mask_batch)


# ──────────────────────────────────────────────────────────────
# Node registration
# ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "MH_DepthWarp": DepthWarpNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MH_DepthWarp": "Depth Warp 6-DOF (MagicHour)",
}
