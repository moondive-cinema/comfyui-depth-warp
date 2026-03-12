"""
독립 실행 테스트 스크립트 — ComfyUI 없이 warp 결과 확인용

사용법:
  python test_warp.py --image background.jpg --depth depth.png

depth 없으면 합성 depth map으로 테스트
"""

import argparse
import sys
import numpy as np
import cv2
sys.path.insert(0, ".")
from depth_warp_node import _warp_frame


def make_synthetic_depth(H, W):
    """배경 테스트용 합성 depth: 중앙 가깝고 주변 멀다."""
    u, v = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    d = 1.0 - 0.5 * np.sqrt(u**2 + v**2)
    return d.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--depth", default=None)
    ap.add_argument("--t_x",  type=float, default=0.3,
                    help="좌우 translation (패럴랙스 테스트 기본값 0.3)")
    ap.add_argument("--yaw",  type=float, default=0.0)
    ap.add_argument("--focal", type=float, default=1000.0)
    ap.add_argument("--out",   default="warp_result.jpg")
    args = ap.parse_args()

    # Load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"ERROR: 이미지 로드 실패: {args.image}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W = img_rgb.shape[:2]
    print(f"Image: {W}x{H}")

    # Load or generate depth
    if args.depth:
        dep = cv2.imread(args.depth, cv2.IMREAD_GRAYSCALE)
        if dep is None:
            print(f"WARN: depth 로드 실패, 합성 depth 사용")
            dep_f = make_synthetic_depth(H, W)
        else:
            dep = cv2.resize(dep, (W, H))
            dep_f = dep.astype(np.float32) / 255.0
    else:
        print("depth 없음 → 합성 depth map 사용")
        dep_f = make_synthetic_depth(H, W)

    # Save depth visualization
    dep_vis = (dep_f * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite("depth_vis.jpg", dep_vis)
    print("depth_vis.jpg 저장됨")

    # Warp
    print(f"Warp 실행: t_x={args.t_x}, yaw={args.yaw}, focal={args.focal}")
    warped, hole = _warp_frame(
        img_rgb, dep_f,
        focal_length=args.focal,
        t_x=args.t_x, t_y=0.0, t_z=0.0,
        yaw=args.yaw, pitch=0.0, roll=0.0,
        depth_invert=True,
        mask_dilate_px=20
    )

    # Save result
    warped_bgr = cv2.cvtColor((warped * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    hole_vis   = (hole * 255).astype(np.uint8)

    # Side by side: original | warped | hole_mask
    original_bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    hole_3ch = cv2.cvtColor(hole_vis, cv2.COLOR_GRAY2BGR)
    
    compare = np.hstack([original_bgr, warped_bgr, hole_3ch])
    cv2.imwrite(args.out, compare)
    print(f"결과 저장: {args.out}")
    print(f"  [좌] 원본  [중] warped  [우] hole mask (흰색=인페인팅 필요)")

    hole_pct = hole.mean() * 100
    print(f"홀 비율: {hole_pct:.1f}%")


if __name__ == "__main__":
    main()
