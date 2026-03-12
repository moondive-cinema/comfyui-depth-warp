# Depth Warp 6-DOF — ComfyUI Custom Node
**Magic Hour Inc. — Virtual Studio Background Pipeline**

Depth Anything V2 깊이 맵을 이용한 6-DOF 시점 변환 노드.  
Translation 파라미터로 패럴랙스를 발생시켜 다중 카메라 화각 배경을 생성한다.

---

## 설치

```bash
# ComfyUI/custom_nodes/ 에 복사
cp -r comfyui-depth-warp/ ~/ComfyUI/custom_nodes/
# ComfyUI 재시작
```

---

## 노드 위치

`MagicHour/Background` → **Depth Warp 6-DOF (MagicHour)**

---

## 파라미터

### 카메라 내부 파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `focal_length` | 1000 | 초점거리(px). 화각이 좁을수록 크게. 35mm 환산 50mm ≈ 1000~1500 |

### Translation (패럴랙스 발생)
| 파라미터 | 단위 | 설명 |
|---------|------|------|
| `t_x` | 상대값 | 좌우 이동. 양수=오른쪽. **스튜디오 다중 화각의 핵심 파라미터** |
| `t_y` | 상대값 | 상하 이동. 양수=아래쪽 |
| `t_z` | 상대값 | 전후 이동. 줌과 유사한 효과 |

> **단위 기준:** depth map의 mean depth = 1.0 기준.  
> `t_x = 0.3` → 평균 피사체 거리의 30% 만큼 이동.  
> 절대 수치가 아니므로 시각적으로 튜닝해야 한다.

### Rotation
| 파라미터 | 단위 | 설명 |
|---------|------|------|
| `yaw`   | 도(°) | 좌우 pan. Translation과 함께 사용. |
| `pitch` | 도(°) | 상하 tilt |
| `roll`  | 도(°) | 회전. 보통 0 |

### 옵션
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `depth_invert` | True | Depth Anything V2 출력(disparity, 높을수록 가깝다) 사용 시 True |
| `mask_dilate_px` | 20 | 홀 마스크 팽창량. FLUX Fill 경계 처리용. |

---

## 스튜디오 3-카메라 설정 예시

마스터 샷 기준으로 좌/우 카메라 배경 생성:

```
마스터 (중앙):   t_x=0.0,  yaw=0.0
좌측 카메라:    t_x=-0.3, yaw=+8.0   (카메라가 왼쪽에 있고 오른쪽을 향함)
우측 카메라:    t_x=+0.3, yaw=-8.0   (카메라가 오른쪽에 있고 왼쪽을 향함)
```

---

## 워크플로우 구성

```
[LoadImage: 마스터 배경]
        ↓
[Depth Anything V2]          ← controlnet-aux 안에 포함
        ↓
[Depth Warp 6-DOF]  ← t_x, yaw 설정
   ↓              ↓
[warped_image]  [hole_mask]
        ↓              ↓
   [VAE Encode] + [mask] → [FluxFillConditioning] → [KSampler] → [VAE Decode]
                                    ↑
                        [CLIPTextEncode: 배경 프롬프트]
```

---

## 독립 실행 테스트

```bash
cd comfyui-depth-warp/
python test_warp.py --image background.jpg --t_x 0.3 --out result.jpg
# depth 없으면 합성 depth map으로 패럴랙스 효과 확인 가능
```

출력: `[원본 | warped | hole_mask]` 3분할 이미지

---

## 알려진 한계

1. **Monocular depth scale:** Depth Anything V2는 상대 깊이만 추정. t_x의 절대 수치는 의미 없고 시각적 튜닝 필요.
2. **Occlusion 영역:** 가려져 있던 영역은 FLUX Fill이 채우지만 기하학적으로 정확하지 않음. 인터뷰/토크쇼처럼 카메라 이동이 작은 경우 충분한 품질.
3. **속도:** CPU 기준 1080p 이미지 약 2~4초. GPU 가속 미구현 (배경 생성은 오프라인이므로 문제 없음).
