"""
SEM 图像拼接工具 (Image Stitching for SEM)
使用 SIFT + FLANN + 纯平移变换 (Translation Only) 拼接两张扫描电子显微镜图像。
SEM 拍摄时 stage 仅做 XY 移动，图像之间只存在平移关系，不做旋转和缩放。
"""

import cv2
import numpy as np
import os


# ============================================================
# 1. 配置参数
# ============================================================
# 图片路径（图片放在子文件夹中）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "F82H-BA07-054_2026_06_22")
IMG1_PATH = os.path.join(IMG_DIR, "F82H-BA07-054_002.tif")
IMG2_PATH = os.path.join(IMG_DIR, "F82H-BA07-054_003.tif")
OUTPUT_PATH = os.path.join(BASE_DIR, "stitched_2_3.tif")

# SIFT 参数
SIFT_N_FEATURES = 0        # 0 = 保留所有特征点
SIFT_CONTRAST_THRESHOLD = 0.04  # 降低以保留更多低对比度特征点（SEM 图像通常偏暗）
SIFT_EDGE_THRESHOLD = 10   # 略微放宽边缘阈值

# FLANN 参数
FLANN_TREES = 5
FLANN_CHECKS = 100

# Lowe's ratio test 阈值（越小越严格）
LOWE_RATIO = 0.75

# RANSAC 参数
RANSAC_REPROJ_THRESH = 5.0
RANSAC_MAX_ITERS = 2000

# CLAHE 参数
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)


# ============================================================
# 2. 读取图像
# ============================================================
print(f"[1/7] 读取图像...")
img1_orig = cv2.imread(IMG1_PATH, cv2.IMREAD_GRAYSCALE)
img2_orig = cv2.imread(IMG2_PATH, cv2.IMREAD_GRAYSCALE)

if img1_orig is None:
    raise FileNotFoundError(f"无法读取图像: {IMG1_PATH}")
if img2_orig is None:
    raise FileNotFoundError(f"无法读取图像: {IMG2_PATH}")

print(f"  图像 002: {img1_orig.shape[1]}x{img1_orig.shape[0]}, "
      f"灰度范围 [{img1_orig.min()}, {img1_orig.max()}]")
print(f"  图像 003: {img2_orig.shape[1]}x{img2_orig.shape[0]}, "
      f"灰度范围 [{img2_orig.min()}, {img2_orig.max()}]")


# ============================================================
# 3. 预处理：CLAHE 增强对比度（仅用于特征提取）
# ============================================================
print(f"[2/7] 应用 CLAHE 预处理（用于特征提取）...")

clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                        tileGridSize=CLAHE_TILE_SIZE)

img1_enhanced = clahe.apply(img1_orig)
img2_enhanced = clahe.apply(img2_orig)

print(f"  CLAHE 处理后 - 图像 002 范围: [{img1_enhanced.min()}, {img1_enhanced.max()}]")
print(f"  CLAHE 处理后 - 图像 003 范围: [{img2_enhanced.min()}, {img2_enhanced.max()}]")


# ============================================================
# 4. SIFT 特征点检测
# ============================================================
print(f"[3/7] 使用 SIFT 提取特征点...")

sift = cv2.SIFT_create(
    nfeatures=SIFT_N_FEATURES,
    contrastThreshold=SIFT_CONTRAST_THRESHOLD,
    edgeThreshold=SIFT_EDGE_THRESHOLD,
)

kp1, des1 = sift.detectAndCompute(img1_enhanced, None)
kp2, des2 = sift.detectAndCompute(img2_enhanced, None)

print(f"  图像 002 特征点: {len(kp1)}")
print(f"  图像 003 特征点: {len(kp2)}")


# ============================================================
# 5. FLANN 特征匹配
# ============================================================
print(f"[4/7] FLANN 特征匹配...")

# FLANN 参数（适用于 SIFT 浮点描述子）
flann_index_kdtree = 1
index_params = dict(algorithm=flann_index_kdtree, trees=FLANN_TREES)
search_params = dict(checks=FLANN_CHECKS)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# KNN 匹配：每个描述子找最近的两个邻居
matches = flann.knnMatch(des1, des2, k=2)

print(f"  初始 KNN 匹配数: {len(matches)}")

# Lowe's ratio test: 只保留 "最近距离 < ratio * 次近距离" 的匹配
good_matches = []
for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < LOWE_RATIO * n.distance:
            good_matches.append(m)

print(f"  经过 Lowe's ratio test (ratio={LOWE_RATIO}) 后的 Good Matches: {len(good_matches)}")

# 按匹配距离排序（可选，用于可视化时只显示最好的匹配）
good_matches = sorted(good_matches, key=lambda x: x.distance)

if len(good_matches) < 10:
    raise RuntimeError(
        f"Good Matches 数量不足 ({len(good_matches)} < 10)，"
        f"请尝试调整 CLAHE 参数或 Lowe's ratio 阈值。"
    )


# ============================================================
# 6. 计算纯平移向量 (Translation Only, 2 DOF)
#    SEM stage 仅做 XY 移动，图像之间只有 dx, dy 平移
#    使用 RANSAC 找出共识平移量，自动排除错误匹配
# ============================================================
print(f"[5/7] 计算纯平移向量 (Translation Only, 2 DOF)...")

# 从 good matches 中提取对应的点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# 所有匹配对的位移向量
displacements = dst_pts - src_pts  # shape (N, 2)

# RANSAC 寻找共识平移量
n_matches = len(good_matches)
rng = np.random.default_rng(42)
best_inliers = None
best_dx, best_dy = 0.0, 0.0
best_count = 0

for _ in range(RANSAC_MAX_ITERS):
    # 随机选一个匹配对作为候选平移
    idx = rng.integers(0, n_matches)
    dx = displacements[idx, 0]
    dy = displacements[idx, 1]
    # 计算所有点与该平移量的距离
    dists = np.sqrt((displacements[:, 0] - dx) ** 2 +
                    (displacements[:, 1] - dy) ** 2)
    inliers = dists < RANSAC_REPROJ_THRESH
    count = inliers.sum()
    if count > best_count:
        best_count = count
        best_inliers = inliers
        best_dx, best_dy = dx, dy

# 用所有内点精化平移量（取均值）
dx = displacements[best_inliers, 0].mean()
dy = displacements[best_inliers, 1].mean()

# 构建纯平移变换矩阵（2x3 仿射格式）
# [[1, 0, dx],
#  [0, 1, dy]]
M = np.array([
    [1.0, 0.0, dx],
    [0.0, 1.0, dy]
], dtype=np.float64)

# 统计位移分布（用于评估匹配置信度）
all_dx = displacements[:, 0]
all_dy = displacements[:, 1]

print(f"  原始位移范围: dx=[{all_dx.min():.1f}, {all_dx.max():.1f}], "
      f"dy=[{all_dy.min():.1f}, {all_dy.max():.1f}]")
print(f"  纯平移向量: Δx={dx:.1f} px, Δy={dy:.1f} px")
print(f"  RANSAC 内点 (inliers): {best_count}/{n_matches}")
print(f"  变换矩阵 M (2x3):\n{M}")

if best_count < 10:
    raise RuntimeError(
        f"平移估计内点数量不足 ({best_count} < 10)，"
        f"可能两张图像重叠区域太小或特征匹配错误。"
    )


# ============================================================
# 7. 图像拼接（纯平移，零变形）
# ============================================================
print(f"[6/7] 图像拼接（使用原始图像以确保最佳质量）...")

h1, w1 = img1_orig.shape
h2, w2 = img2_orig.shape

# 在纯平移下，角点变换简化为直接加减
corners1 = np.float32([
    [0 + dx, 0 + dy],
    [0 + dx, h1 + dy],
    [w1 + dx, h1 + dy],
    [w1 + dx, 0 + dy]
])
corners2 = np.float32([
    [0, 0],
    [0, h2],
    [w2, h2],
    [w2, 0]
])

# 计算拼接画布范围
all_x = np.concatenate([corners1[:, 0], corners2[:, 0]])
all_y = np.concatenate([corners1[:, 1], corners2[:, 1]])
x_min = int(np.floor(all_x.min()))
x_max = int(np.ceil(all_x.max()))
y_min = int(np.floor(all_y.min()))
y_max = int(np.ceil(all_y.max()))

canvas_w = x_max - x_min
canvas_h = y_max - y_min

print(f"  拼接后画布尺寸: {canvas_w} x {canvas_h}")

# 创建空白画布，直接将两张图贴到对应位置（不做任何变换）
stitched = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

# 图像 1 放到平移后的位置
x1_start = int(dx) - x_min
y1_start = int(dy) - y_min
stitched[y1_start:y1_start + h1, x1_start:x1_start + w1] = img1_orig

# 图像 2 放到原点位置（作为参考）
x2_start = -x_min
y2_start = -y_min
stitched[y2_start:y2_start + h2, x2_start:x2_start + w2] = img2_orig

print(f"  拼接完成！（纯平移，零变形）")


# ============================================================
# 8. 保存与显示
# ============================================================
print(f"[7/7] 保存与显示结果...")

# 保存为 TIFF（无损）
cv2.imwrite(OUTPUT_PATH, stitched)
print(f"  已保存至: {OUTPUT_PATH}")

# 尝试显示（无 GUI 环境会自动跳过）
try:
    display_img = stitched.copy()
    max_display_size = 1400
    h_disp, w_disp = display_img.shape
    scale = min(max_display_size / max(h_disp, w_disp), 1.0)
    if scale < 1.0:
        display_img = cv2.resize(display_img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
        print(f"  显示缩放: {scale:.2f}x")

    cv2.imshow("Stitched Result (002 + 003)", display_img)
    # 短暂等待后检查窗口是否可见（无 GUI 环境下窗口不可见）
    cv2.waitKey(100)
    try:
        visible = cv2.getWindowProperty("Stitched Result (002 + 003)", cv2.WND_PROP_VISIBLE)
    except Exception:
        visible = -1
    if visible < 1:
        print(f"  (无 GUI 环境，跳过窗口显示)")
        cv2.destroyAllWindows()
    else:
        print(f"  按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
except Exception:
    print(f"  (无 GUI 环境，跳过窗口显示)")
    cv2.destroyAllWindows()

print(f"\n===== 全部完成! =====")
print(f"输出文件: {OUTPUT_PATH}")
print(f"Good Matches: {len(good_matches)}, Inliers: {best_count}")
