"""
SEM 图像拼接工具 (Image Stitching for SEM)
===========================================
使用 SIFT + FLANN + RANSAC 纯平移向量拼接多张扫描电子显微镜图像。
- 纯平移变换（2 DOF）：SEM stage 仅做 XY 移动，不做旋转和缩放
- 透明度羽化融合（Opacity Blending）：重叠区域线性渐变过渡，消除拼接缝
- 顺序拼接：逐张增量拼接，支持任意数量图像
"""

import argparse
from collections import deque
import cv2
import numpy as np
import os


# ============================================================
# SEMStitcher 类：封装特征提取、匹配、平移计算、融合拼接
# ============================================================
class SEMStitcher:
    """SEM 图像拼接器，基于纯平移 + 透明度羽化融合"""

    def __init__(
        self,
        # SIFT 参数
        sift_n_features=0,
        sift_contrast_threshold=0.01,  # 降低以适配边缘图
        sift_edge_threshold=200,
        # FLANN 参数
        flann_trees=5,
        flann_checks=100,
        # Lowe's ratio test
        lowe_ratio=0.75,
        # RANSAC 参数
        ransac_reproj_thresh=5.0,
        ransac_max_iters=2000,
        # CLAHE 参数
        clahe_clip_limit=2.0,
        clahe_tile_size=(8, 8),
    ):
        # ---- 保存参数 ----
        self.lowe_ratio = lowe_ratio
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.ransac_max_iters = ransac_max_iters

        # ---- SIFT ----
        self.sift = cv2.SIFT_create(
            nfeatures=sift_n_features,
            contrastThreshold=sift_contrast_threshold,
            edgeThreshold=sift_edge_threshold,
            sigma=2.0             # 【关键修改2】增加初始高斯模糊(默认1.6)，过滤掉微小划痕
        )

        # ---- FLANN ----
        flann_index_kdtree = 1
        self.flann_index_params = dict(algorithm=flann_index_kdtree, trees=flann_trees)
        self.flann_search_params = dict(checks=flann_checks)

        # ---- CLAHE ----
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size,
        )

        # ---- GUI 回调 ----
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """设置进度回调函数 callback(msg: str)，供 GUI 实时显示日志"""
        self._progress_callback = callback

    def _log(self, msg: str):
        """统一日志输出：有回调则发给 GUI，否则 print"""
        if self._progress_callback:
            self._progress_callback(msg)
        else:
            print(msg)

    # ================================================================
    # 预处理：CLAHE 增强对比度（仅用于特征提取，不修改原图）
    # ================================================================
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """对灰度图应用 CLAHE 增强局部对比度"""
        return self.clahe.apply(img)

    # ================================================================
    # 边缘图提取：降噪 → Sobel → 归一化，强制匹配宏观结构
    # ================================================================
    def _extract_edge_map(self, img: np.ndarray) -> np.ndarray:
        """
        提取边缘梯度幅值图，用于相位相关精细对齐。

        1) CLAHE 增强局部对比度（提升暗区的梯度响应）
        2) GaussianBlur 模糊掉高频划痕/噪点（σ≈3, kernel=11）
        3) Sobel 提取梯度幅值（宏观边缘，如圆孔边界）
        4) 归一化到 0-255
        """
        # 先增强对比度，否则暗区 Sobel 梯度太弱
        enhanced = self.clahe.apply(img)
        # 强降噪：模糊掉划痕和细微纹理，保留宏观结构
        blurred = cv2.GaussianBlur(enhanced, (11, 11), 3.0)
        # Sobel 梯度幅值
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX)
        return edge_map.astype(np.uint8)

    # ================================================================
    # 特征提取
    # ================================================================
    def _extract_features(self, img: np.ndarray) -> tuple:
        """使用 SIFT 检测关键点和描述子"""
        return self.sift.detectAndCompute(img, None)

    # ================================================================
    # 特征匹配 → 计算纯平移向量 (dx, dy)
    # ================================================================
    def _compute_translation(self, kp1, des1, kp2, des2) -> tuple:
        """
        通过 FLANN + Lowe's ratio test + RANSAC 计算纯平移向量。

        参数:
            kp1, des1: 图像1（待平移图）的关键点和描述子
            kp2, des2: 图像2（参考图）的关键点和描述子

        返回:
            (dx, dy, inlier_count, n_good_matches)
            dx, dy → 将图像1平移到图像2坐标系的平移量（浮点数）
            若匹配失败则返回 (None, None, 0, 0)
        """
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return None, None, 0, 0

        # ---- FLANN KNN 匹配 ----
        flann = cv2.FlannBasedMatcher(self.flann_index_params, self.flann_search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # ---- Lowe's ratio test ----
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.lowe_ratio * n.distance:
                    good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        n_good = len(good_matches)

        if n_good < 8:
            return None, None, 0, n_good

        # ---- 提取所有匹配对的位移 ----
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        displacements = dst_pts - src_pts  # shape (N, 2)

        # ---- RANSAC 寻找共识平移量 ----
        rng = np.random.default_rng(42)
        best_inliers = None
        best_dx, best_dy = 0.0, 0.0
        best_count = 0

        for _ in range(self.ransac_max_iters):
            idx = rng.integers(0, n_good)
            dx = displacements[idx, 0]
            dy = displacements[idx, 1]
            dists = np.sqrt(
                (displacements[:, 0] - dx) ** 2 +
                (displacements[:, 1] - dy) ** 2
            )
            inliers = dists < self.ransac_reproj_thresh
            count = inliers.sum()
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_dx, best_dy = dx, dy

        if best_count < 8:
            return None, None, best_count, n_good

        # ---- 用内点精化平移量（取均值，得到亚像素精度） ----
        dx = displacements[best_inliers, 0].mean()
        dy = displacements[best_inliers, 1].mean()

        return dx, dy, best_count, n_good

    # ================================================================
    # 相位相关精细对齐（亚像素精度，频域方法，对噪点鲁棒）
    # ================================================================
    def _fine_align_phase(self, img1: np.ndarray, img2: np.ndarray,
                          dx_coarse: float, dy_coarse: float) -> tuple:
        """
        在 SIFT+RANSAC 粗对齐基础上，用相位相关 (Phase Correlation)
        做纯平移精细对齐。

        相位相关在频域计算两张图的互功率谱，通过 FFT 直接定位
        平移量，天然适合纯平移估计，对 SEM 噪点和重复纹理鲁棒。

        img1, img2: 原始灰度图（不经 CLAHE，保留真实边缘位置）
        返回: (dx_fine, dy_fine) 叠加到粗平移上的微调量
        """
        h1, w1 = img1.shape
        h2, w2 = img2.shape

        dx_int = int(round(dx_coarse))
        dy_int = int(round(dy_coarse))

        # 计算粗对齐后的重叠区域（img2 坐标）
        ox1 = max(0, dx_int)
        oy1 = max(0, dy_int)
        ox2 = min(w2, w1 + dx_int)
        oy2 = min(h2, h1 + dy_int)
        ow = ox2 - ox1
        oh = oy2 - oy1

        # 重叠区需要足够大才能做可靠的相位相关
        if ow < 60 or oh < 60:
            return 0.0, 0.0

        # 提取两张图的重叠区（img1 坐标系中对应位置）
        i1_x1 = max(0, -dx_int)
        i1_y1 = max(0, -dy_int)

        # 转换为 float32（相位相关要求）
        patch1 = img1[i1_y1:i1_y1 + oh, i1_x1:i1_x1 + ow].astype(np.float32)
        patch2 = img2[oy1:oy1 + oh, ox1:ox1 + ow].astype(np.float32)

        # Hann 窗：抑制 FFT 边缘效应
        window = np.outer(np.hanning(oh), np.hanning(ow)).astype(np.float32)

        # 相位相关
        try:
            shift, response = cv2.phaseCorrelate(
                patch1 * window, patch2 * window
            )
            dx_fine = float(shift[0])
            dy_fine = float(shift[1])
            if response < 0.05:
                dx_fine, dy_fine = 0.0, 0.0
            # 修正幅度 >10px → 相位相关跳到错误圆孔，丢弃
            if abs(dx_fine) > 10.0 or abs(dy_fine) > 10.0:
                dx_fine, dy_fine = 0.0, 0.0
        except cv2.error:
            dx_fine, dy_fine = 0.0, 0.0

        return dx_fine, dy_fine

    # ================================================================
    # 线性羽化融合 (Linear Blending)
    def _blend_images(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        dx: float,
        dy: float,
    ) -> np.ndarray:
        """
        将 img1 平移 (dx, dy) 后与参考图 img2 融合。

        img2 是参考图，放在画布上固定不动。
        img1 由 (dx, dy) 指定其在画布上相对于 img2 的位移。
        在两张图的重叠区域使用线性羽化融合，消除拼接缝。

        返回:
            融合后的 uint8 灰度图
        """
        h1, w1 = img1.shape
        h2, w2 = img2.shape

        # ---- 将浮点位移取整（确保像素对齐） ----
        dx_int = int(round(dx))
        dy_int = int(round(dy))

        # ---- 计算两张图在画布上的整数位置 ----
        x2, y2 = 0, 0         # img2（参考）在原点
        x1, y1 = dx_int, dy_int  # img1 平移后

        # ---- 把负坐标平移到正坐标 ----
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x1 -= x_min
        x2 -= x_min
        y1 -= y_min
        y2 -= y_min

        # ---- 画布尺寸 ----
        canvas_w = max(x1 + w1, x2 + w2)
        canvas_h = max(y1 + h1, y2 + h2)

        # ---- 浮点画布（用于加权累加） ----
        sum_img = np.zeros((canvas_h, canvas_w), dtype=np.float64)
        sum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)

        # ---- 初始权重全 1 ----
        w1_mask = np.ones((h1, w1), dtype=np.float64)
        w2_mask = np.ones((h2, w2), dtype=np.float64)

        # ---- 计算重叠区域（画布坐标，整型） ----
        ox1 = max(x1, x2)
        oy1 = max(y1, y2)
        ox2 = min(x1 + w1, x2 + w2)
        oy2 = min(y1 + h1, y2 + h2)
        ow = ox2 - ox1  # 重叠宽度 (int)
        oh = oy2 - oy1  # 重叠高度 (int)

        if ow > 0 and oh > 0:
            # 重叠区域在各自图像中的局部坐标
            o1_x1 = ox1 - x1
            o1_y1 = oy1 - y1
            o1_x2 = o1_x1 + ow
            o1_y2 = o1_y1 + oh

            o2_x1 = ox1 - x2
            o2_y1 = oy1 - y2
            o2_x2 = o2_x1 + ow
            o2_y2 = o2_y1 + oh

            # 沿重叠区域较窄的方向做线性渐变
            if ow <= oh:
                # 水平融合: ramp 从左到右 [0 → 1], shape (1, ow)
                ramp = np.linspace(0, 1, ow, dtype=np.float64)[np.newaxis, :]
                if x1 < x2:
                    # img1 在左 (权重从左到右 1→0), img2 在右 (0→1)
                    w1_mask[o1_y1:o1_y2, o1_x1:o1_x2] = 1.0 - ramp
                    w2_mask[o2_y1:o2_y2, o2_x1:o2_x2] = ramp
                else:
                    # img2 在左, img1 在右
                    w2_mask[o2_y1:o2_y2, o2_x1:o2_x2] = 1.0 - ramp
                    w1_mask[o1_y1:o1_y2, o1_x1:o1_x2] = ramp
            else:
                # 垂直融合: ramp 从上到下 [0 → 1], shape (oh, 1)
                ramp = np.linspace(0, 1, oh, dtype=np.float64)[:, np.newaxis]
                if y1 < y2:
                    # img1 在上 (权重从上到下 1→0), img2 在下 (0→1)
                    w1_mask[o1_y1:o1_y2, o1_x1:o1_x2] = 1.0 - ramp
                    w2_mask[o2_y1:o2_y2, o2_x1:o2_x2] = ramp
                else:
                    # img2 在上, img1 在下
                    w2_mask[o2_y1:o2_y2, o2_x1:o2_x2] = 1.0 - ramp
                    w1_mask[o1_y1:o1_y2, o1_x1:o1_x2] = ramp

        # ---- 将两张图累加到画布（加权） ----
        sum_img[y1:y1 + h1, x1:x1 + w1] += img1.astype(np.float64) * w1_mask
        sum_weight[y1:y1 + h1, x1:x1 + w1] += w1_mask

        sum_img[y2:y2 + h2, x2:x2 + w2] += img2.astype(np.float64) * w2_mask
        sum_weight[y2:y2 + h2, x2:x2 + w2] += w2_mask

        # ---- 归一化（避免除以零） ----
        valid = sum_weight > 1e-6
        sum_img[valid] /= sum_weight[valid]

        return np.clip(np.round(sum_img), 0, 255).astype(np.uint8)

    # ================================================================
    # 拼接两 images（核心方法）
    # ================================================================
    def stitch_two(self, img1: np.ndarray, img2: np.ndarray,
                   label1: str = "img1", label2: str = "img2") -> np.ndarray:
        """
        拼接两 images。

        img2 作为参考图，img1 平移到对齐位置后融合。
        返回拼接后的图像。

        参数:
            img1: 待拼接图像（将被平移以对齐 img2）
            img2: 参考图像（保持原位）
            label1: 图像1的名称标签（用于日志）
            label2: 图像2的名称标签（用于日志）
        返回:
            拼接后的 uint8 灰度图
        """
        # 1. 预处理（CLAHE）
        img1_enhanced = self._preprocess(img1)
        img2_enhanced = self._preprocess(img2)

        # 2. 特征提取
        kp1, des1 = self._extract_features(img1_enhanced)
        kp2, des2 = self._extract_features(img2_enhanced)
        self._log(f"  Keypoints: {label1}={len(kp1)}, {label2}={len(kp2)}")

        # 3. 计算平移量
        dx, dy, inliers, n_good = self._compute_translation(kp1, des1, kp2, des2)

        if dx is None:
            raise RuntimeError(
                f"无法计算平移量！内点={inliers}, Good Matches={n_good}。"
                f"请检查图像重叠是否足够。"
            )

        self._log(f"  Good Matches: {n_good}, RANSAC inliers: {inliers}")
        self._log(f"  Translation: Δx={dx:.1f} px, Δy={dy:.1f} px")

        # 4. 融合拼接（保持原图亮度，仅重叠区羽化过渡）
        result = self._blend_images(img1, img2, dx, dy)
        self._log(f"  Blend done -> canvas: {result.shape[1]}x{result.shape[0]}")
        # 报告重叠量
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        x1, y1 = dx, dy
        x2, y2 = 0.0, 0.0
        ox_w = min(x1 + w1, x2 + w2) - max(x1, x2)
        ox_h = min(y1 + h1, y2 + h2) - max(y1, y2)
        if ox_w > 0 and ox_h > 0:
            self._log(f"  Blend overlap: {int(ox_w)}x{int(ox_h)} px "
                  f"({'水平' if ox_w <= ox_h else '垂直'}blend)")

        return result

    # ================================================================
    # 结构对齐匹配（边缘图 SIFT + 相位相关）
    # ================================================================
    def _try_match_pair(self, img_a: np.ndarray, img_b: np.ndarray,
                        label_a: str = "A", label_b: str = "B") -> tuple:
        """
        Try匹配两 images，使用边缘图强制算法关注宏观结构。

        1) 提取边缘图 (GaussianBlur + Sobel)
        2) 在边缘图上 SIFT → FLANN → RANSAC 粗对齐
        3) 在边缘图上相位相关精细对齐

        返回: (dx, dy, quality)
          dx, dy  = img_a → img_b 的总平移量（None 表示匹配失败）
          quality = RANSAC 内点数量（用于 MST 边权重）
        """
        # ---- 提取边缘图（SIFT 用，切断高频噪点） ----
        edge_a = self._extract_edge_map(img_a)
        edge_b = self._extract_edge_map(img_b)

        # ---- SIFT 粗对齐（在边缘图上） ----
        kp_a, des_a = self._extract_features(edge_a)
        kp_b, des_b = self._extract_features(edge_b)
        dx, dy, inliers, n_good = self._compute_translation(kp_a, des_a, kp_b, des_b)

        self._log(f"  Keypoints (edge map): {label_a}={len(kp_a)}, {label_b}={len(kp_b)}")
        if dx is not None:
            self._log(f"  SIFT coarse: Matches={n_good}, Inliers={inliers}, "
                  f"Δx={dx:.1f}, Δy={dy:.1f} px")
        else:
            self._log(f"  SIFT failed: Matches={n_good}, Inliers={inliers}")

        if dx is None:
            return None, None, 0

        # ---- 相位相关精细对齐（在 CLAHE 图上，保留完整圆孔结构） ----
        enhanced_a = self._preprocess(img_a)
        enhanced_b = self._preprocess(img_b)
        dx_fine, dy_fine = self._fine_align_phase(enhanced_a, enhanced_b, dx, dy)

        if abs(dx_fine) > 0.01 or abs(dy_fine) > 0.01:
            self._log(f"  Phase corr fine: Δx={dx_fine:.2f}, Δy={dy_fine:.2f} px")

        return dx, dy, inliers  # quality = inlier count

    # ================================================================
    # 将单 imagesAdd到预分配的浮点画布（含曝光补偿 + 羽化融合）
    # ================================================================
    def _blend_onto_canvas(self,
                           sum_img: np.ndarray,
                           sum_weight: np.ndarray,
                           new_img: np.ndarray,
                           x_pos: int, y_pos: int) -> None:
        """
        将 new_img 加权叠加到浮点画布上（原地修改 sum_img, sum_weight）。

        sum_img:   累积加权像素 (float64)
        sum_weight: 累积权重 (float64)
        new_img:   新图像 (uint8)
        x_pos, y_pos: new_img 在画布上的整数左上角坐标（必须 ≥ 0）
        """
        # 强制转为纯 Python int
        h_new, w_new = new_img.shape
        h_new = int(h_new)
        w_new = int(w_new)
        x_pos = int(x_pos)
        y_pos = int(y_pos)
        ch, cw = sum_img.shape

        # ---- 计算重叠区域 ----
        ox1 = max(x_pos, 0)
        oy1 = max(y_pos, 0)
        ox2 = min(x_pos + w_new, cw)
        oy2 = min(y_pos + h_new, ch)
        ow = ox2 - ox1
        oh = oy2 - oy1

        # ---- 创建新图的透明度掩码（线性羽化） ----
        w_new_mask = np.ones((h_new, w_new), dtype=np.float64)

        if ow > 0 and oh > 0:
            o_n_x1 = ox1 - x_pos
            o_n_y1 = oy1 - y_pos
            o_c_x1 = ox1
            o_c_y1 = oy1

            # 确定画布内容的重心（用于判断画布在新图的哪一侧）
            w_overlap = sum_weight[o_c_y1:o_c_y1 + oh, o_c_x1:o_c_x1 + ow]
            col_weights = w_overlap.sum(axis=0)  # 每列的权重和
            row_weights = w_overlap.sum(axis=1)  # 每行的权重和

            if ow <= oh:
                # 水平融合
                ramp = np.linspace(0, 1, ow, dtype=np.float64)[np.newaxis, :]
                # 画布重心 X vs 新图重心 X
                cols = np.arange(ox1, ox2)
                total_w = col_weights.sum()
                canvas_center_x = float(np.average(cols, weights=col_weights)) if total_w > 0 else ox1
                new_center_x = x_pos + w_new / 2.0
                # 画布在左 → 画布权重从左到右 1→0，新图权重 0→1
                if canvas_center_x <= new_center_x:
                    # 画布在左：画布权重从左到右 1→0，新图权重 0→1
                    # ★ 同步缩放 sum_img 和 sum_weight，避免曝光异常
                    sum_img[o_c_y1:o_c_y1 + oh, o_c_x1:o_c_x1 + ow] *= (1.0 - ramp)
                    sum_weight[o_c_y1:o_c_y1 + oh, o_c_x1:o_c_x1 + ow] *= (1.0 - ramp)
                    w_new_mask[o_n_y1:o_n_y1 + oh, o_n_x1:o_n_x1 + ow] = ramp
                else:
                    # 画布在右：画布权重从左到右 0→1，新图权重 1→0
                    sum_img[o_c_y1:o_c_y1 + oh, o_c_x1:o_c_x1 + ow] *= ramp
                    sum_weight[o_c_y1:o_c_y1 + oh, o_c_x1:o_c_x1 + ow] *= ramp
                    w_new_mask[o_n_y1:o_n_y1 + oh, o_n_x1:o_n_x1 + ow] = 1.0 - ramp
            else:
                # 垂直融合
                ramp = np.linspace(0, 1, oh, dtype=np.float64)[:, np.newaxis]
                rows = np.arange(oy1, oy2)
                total_w = row_weights.sum()
                canvas_center_y = float(np.average(rows, weights=row_weights)) if total_w > 0 else oy1
                new_center_y = y_pos + h_new / 2.0
                if canvas_center_y <= new_center_y:
                    # 画布在上：画布权重从上到下 1→0，新图权重 0→1
                    sum_img[o_c_y1:o_c_y1 + oh, o_c_x1:o_c_x1 + ow] *= (1.0 - ramp)
                    sum_weight[o_c_y1:o_c_y1 + oh, o_c_x1:o_c_x1 + ow] *= (1.0 - ramp)
                    w_new_mask[o_n_y1:o_n_y1 + oh, o_n_x1:o_n_x1 + ow] = ramp
                else:
                    # 画布在下：画布权重从上到下 0→1，新图权重 1→0
                    sum_img[o_c_y1:o_c_y1 + oh, o_c_x1:o_c_x1 + ow] *= ramp
                    sum_weight[o_c_y1:o_c_y1 + oh, o_c_x1:o_c_x1 + ow] *= ramp
                    w_new_mask[o_n_y1:o_n_y1 + oh, o_n_x1:o_n_x1 + ow] = 1.0 - ramp

        # ---- 加权累加 ----
        y1, y2 = int(y_pos), int(y_pos + h_new)
        x1, x2 = int(x_pos), int(x_pos + w_new)
        sum_img[y1:y2, x1:x2] += new_img.astype(np.float64) * w_new_mask
        sum_weight[y1:y2, x1:x2] += w_new_mask

        # ---- 汇报 ----
        if ow > 0 and oh > 0:
            # 只统计画布已有内容的实际重叠
            w_actual = sum_weight[oy1:oy1 + oh, ox1:ox1 + ow]
            actual_overlap_px = int((w_actual > 0).sum())
            total_px = ow * oh
            direction = "水平" if ow <= oh else "垂直"
            self._log(f"  Blend overlap: {ow}x{oh} px ({direction}blend, "
                  f"content={actual_overlap_px}/{total_px} px)")

    # ================================================================
    # 配准图约束、全局定位与合成
    # ================================================================
    @staticmethod
    def _grid_neighbor_pairs(n: int, grid_shape: tuple, serpentine: bool) -> set:
        """Return index pairs for a row-major (optionally serpentine) grid."""
        try:
            rows, cols = grid_shape
            rows, cols = int(rows), int(cols)
        except (TypeError, ValueError) as exc:
            raise ValueError("grid_shape must be a (rows, cols) pair") from exc
        if rows <= 0 or cols <= 0 or rows * cols != n:
            raise ValueError(
                f"grid_shape {grid_shape} does not match {n} input images"
            )

        grid = []
        for row in range(rows):
            indices = list(range(row * cols, (row + 1) * cols))
            if serpentine and row % 2 == 1:
                indices.reverse()
            grid.append(indices)

        pairs = set()
        for row in range(rows):
            for col in range(cols):
                if col + 1 < cols:
                    pairs.add(tuple(sorted((grid[row][col], grid[row][col + 1]))))
                if row + 1 < rows:
                    pairs.add(tuple(sorted((grid[row][col], grid[row + 1][col]))))
        return pairs

    @staticmethod
    def _overlap_geometry(
        img_a: np.ndarray, img_b: np.ndarray, dx: float, dy: float,
    ) -> tuple:
        """Return overlap width, height, and the longest-axis overlap ratio."""
        h_a, w_a = img_a.shape
        h_b, w_b = img_b.shape
        overlap_w = max(0.0, min(dx + w_a, w_b) - max(dx, 0.0))
        overlap_h = max(0.0, min(dy + h_a, h_b) - max(dy, 0.0))
        axis_ratio = max(
            overlap_w / min(w_a, w_b),
            overlap_h / min(h_a, h_b),
        )
        return overlap_w, overlap_h, axis_ratio

    @staticmethod
    def _connected_order(n: int, edges: list, root: int) -> list:
        """Return a BFS order for root's component."""
        adjacency = [[] for _ in range(n)]
        for _, i, j, _, _ in edges:
            adjacency[i].append(j)
            adjacency[j].append(i)

        order = []
        visited = [False] * n
        queue = deque([root])
        visited[root] = True
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return order

    @staticmethod
    def _select_anchor(n: int, edges: list) -> tuple:
        """Choose the most connected image as the coordinate-system anchor."""
        degree = [0] * n
        for _, i, j, _, _ in edges:
            degree[i] += 1
            degree[j] += 1
        return max(range(n), key=lambda index: degree[index]), degree

    @staticmethod
    def _solve_global_positions(n: int, edges: list, root: int) -> tuple:
        """Solve weighted translation constraints with one anchored image."""
        A_x, B_x, A_y, B_y = [], [], [], []
        for quality, i, j, dx, dy in edges:
            # Least squares squares the row coefficients. sqrt(quality) therefore
            # gives the intended linear quality weight instead of quality**2.
            weight = np.sqrt(float(quality))
            row = np.zeros(n)
            row[i] = -weight
            row[j] = weight
            A_x.append(row)
            B_x.append(-weight * float(dx))
            row = np.zeros(n)
            row[i] = -weight
            row[j] = weight
            A_y.append(row)
            B_y.append(-weight * float(dy))

        constraint_weight = max(np.sqrt(float(q)) for q, _, _, _, _ in edges) * 1000.0
        row = np.zeros(n)
        row[root] = constraint_weight
        A_x.append(row)
        B_x.append(0.0)
        A_y.append(row)
        B_y.append(0.0)

        A_x = np.array(A_x)
        B_x = np.array(B_x)
        A_y = np.array(A_y)
        B_y = np.array(B_y)
        sol_x, residual_x, rank_x, _ = np.linalg.lstsq(A_x, B_x, rcond=None)
        sol_y, residual_y, rank_y, _ = np.linalg.lstsq(A_y, B_y, rcond=None)
        return (
            sol_x.tolist(), sol_y.tolist(), residual_x, residual_y,
            rank_x, rank_y, A_x.shape,
        )

    @staticmethod
    def _edge_residuals(edges: list, global_x: list, global_y: list) -> list:
        """Measure each translation constraint's error in the global solution."""
        residuals = []
        for edge in edges:
            _, i, j, dx, dy = edge
            error_x = (global_x[i] - global_x[j]) - dx
            error_y = (global_y[i] - global_y[j]) - dy
            residuals.append((float(np.hypot(error_x, error_y)), edge))
        return residuals

    def stitch_sequence(
        self,
        image_paths: list,
        grid_shape: tuple = None,
        serpentine: bool = False,
        min_axis_overlap_ratio: float = 0.70,
        residual_threshold: float = 10.0,
    ) -> np.ndarray:
        """
        Stitch multiple images with pairwise translation constraints.

        Small corner-only overlaps are rejected by default because they are prone
        to false SIFT matches. Supply ``grid_shape`` and ``serpentine=True`` for
        a known snake-scanned grid to match only direct neighbours.
        """
        n = len(image_paths)
        if n < 2:
            raise ValueError("At least two images are required for stitching")
        if not 0.0 <= min_axis_overlap_ratio <= 1.0:
            raise ValueError("min_axis_overlap_ratio must be between 0 and 1")
        if residual_threshold <= 0:
            raise ValueError("residual_threshold must be positive")

        allowed_pairs = None
        if grid_shape is not None:
            allowed_pairs = self._grid_neighbor_pairs(n, grid_shape, serpentine)

        self._log(f"\n{'=' * 60}")
        self._log(f"Starting multi-stitch, {n} images")
        self._log("Strategy: screened pairwise match -> robust global positioning -> blend")
        if allowed_pairs is not None:
            self._log(
                f"Grid constraint: {grid_shape[0]}x{grid_shape[1]}, "
                f"serpentine={serpentine} ({len(allowed_pairs)} direct-neighbour pairs)"
            )
        self._log(f"{'=' * 60}")

        # ---- Phase 1: load images ----
        images = []
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {path}")
            images.append(img)
            self._log(f"  Loaded: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

        candidate_count = len(allowed_pairs) if allowed_pairs is not None else n * (n - 1) // 2
        self._log(f"\n--- Phase 1: Pairwise matching ({candidate_count} pairs) ---")
        edges = []  # (quality, i, j, dx, dy), where img_i is shifted by (dx, dy) into img_j.

        for i in range(n):
            for j in range(i + 1, n):
                if allowed_pairs is not None and (i, j) not in allowed_pairs:
                    continue

                fname_i = os.path.basename(image_paths[i])
                fname_j = os.path.basename(image_paths[j])
                self._log(f"\n  Try [{i}↔{j}]: {fname_i} ↔ {fname_j}")
                dx, dy, quality = self._try_match_pair(
                    images[i], images[j], label_a=fname_i, label_b=fname_j
                )

                if dx is None or quality < 8:
                    self._log("  No match / no overlap")
                    continue

                overlap_w, overlap_h, axis_ratio = self._overlap_geometry(
                    images[i], images[j], dx, dy
                )
                if axis_ratio < min_axis_overlap_ratio:
                    self._log(
                        f"  Reject corner-only overlap: {overlap_w:.0f}x{overlap_h:.0f} px, "
                        f"axis ratio={axis_ratio:.2f} < {min_axis_overlap_ratio:.2f}"
                    )
                    continue

                edges.append((quality, i, j, dx, dy))
                self._log(
                    f"  OK quality={quality}, overlap={overlap_w:.0f}x{overlap_h:.0f} px, "
                    f"axis ratio={axis_ratio:.2f}"
                )

        if len(edges) < n - 1:
            raise RuntimeError(
                f"Only {len(edges)} reliable edges for {n} images; cannot form a connected graph."
            )

        root, degree = self._select_anchor(n, edges)
        order = self._connected_order(n, edges, root)
        if len(order) != n:
            missing = [os.path.basename(image_paths[index]) for index in range(n) if index not in order]
            raise RuntimeError(
                "Reliable matches do not connect every image. Missing: " + ", ".join(missing)
            )

        # ---- Phase 2: weighted least squares and one robust outlier-pruning pass ----
        self._log(f"\n--- Phase 2: Weighted least squares ({len(edges)} edges) ---")
        for pass_index in range(2):
            root, degree = self._select_anchor(n, edges)
            (
                global_x, global_y, residual_x, residual_y,
                rank_x, rank_y, matrix_shape,
            ) = self._solve_global_positions(n, edges, root)

            if rank_x != n or rank_y != n:
                raise RuntimeError(
                    f"Global positioning is rank-deficient (rank X={rank_x}, Y={rank_y})."
                )

            if pass_index == 0:
                self._log(
                    f"  Anchor: [{root}] {os.path.basename(image_paths[root])} "
                    f"(degree={degree[root]})"
                )
                self._log(f"  Matrix: {matrix_shape[0]} eqns x {matrix_shape[1]} unknowns")

            edge_residuals = self._edge_residuals(edges, global_x, global_y)
            worst_residual = max(residual for residual, _ in edge_residuals)
            self._log(
                f"  Pass {pass_index + 1}: worst edge residual={worst_residual:.2f} px "
                f"(threshold={residual_threshold:.2f} px)"
            )
            outliers = [edge for residual, edge in edge_residuals if residual > residual_threshold]
            if not outliers or pass_index == 1:
                break

            retained_edges = [edge for edge in edges if edge not in outliers]
            retained_order = self._connected_order(n, retained_edges, root)
            if len(retained_order) != n:
                self._log("  Keep residual outliers: removing them would disconnect the image graph")
                break

            self._log(f"  Pruning {len(outliers)} inconsistent edge(s) and solving again:")
            for residual, edge in edge_residuals:
                if edge in outliers:
                    _, i, j, _, _ = edge
                    self._log(
                        f"    drop [{i}↔{j}] {os.path.basename(image_paths[i])} ↔ "
                        f"{os.path.basename(image_paths[j])}: residual={residual:.2f} px"
                    )
            edges = retained_edges

        residual_x_text = f"X residual={residual_x[0]:.1f}" if len(residual_x) > 0 else "X determined"
        residual_y_text = f"Y residual={residual_y[0]:.1f}" if len(residual_y) > 0 else "Y determined"
        self._log(f"  {residual_x_text}, {residual_y_text} (rank X={rank_x}, Y={rank_y})")
        for index in range(n):
            self._log(
                f"  [{index}] {os.path.basename(image_paths[index])}: "
                f"({global_x[index]:.1f}, {global_y[index]:.1f})"
            )

        # ---- Phase 3: pre-allocate canvas and blend in graph order ----
        self._log("\n--- Phase 3: Blend composition ---")
        all_left, all_top, all_right, all_bottom = [], [], [], []
        for index, (gx, gy) in enumerate(zip(global_x, global_y)):
            height, width = images[index].shape
            all_left.append(gx)
            all_top.append(gy)
            all_right.append(gx + width)
            all_bottom.append(gy + height)

        canvas_x_min = int(np.floor(min(all_left)))
        canvas_y_min = int(np.floor(min(all_top)))
        canvas_w = int(np.ceil(max(all_right) - canvas_x_min))
        canvas_h = int(np.ceil(max(all_bottom) - canvas_y_min))
        self._log(f"  Canvas: {canvas_w} x {canvas_h}")

        sum_img = np.zeros((canvas_h, canvas_w), dtype=np.float64)
        sum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)
        order = self._connected_order(n, edges, root)
        for index in order:
            fname = os.path.basename(image_paths[index])
            x_pos = int(round(global_x[index])) - canvas_x_min
            y_pos = int(round(global_y[index])) - canvas_y_min
            self._log(f"\n  Add [{index}] {fname} at ({x_pos}, {y_pos})")
            self._blend_onto_canvas(sum_img, sum_weight, images[index], x_pos, y_pos)

        valid = sum_weight > 1e-6
        sum_img[valid] /= sum_weight[valid]
        result = np.clip(np.round(sum_img), 0, 255).astype(np.uint8)

        self._log(f"\n{'=' * 60}")
        self._log(f"All {n} images stitched!")
        self._log(f"Final size: {result.shape[1]} x {result.shape[0]}")
        self._log(f"{'=' * 60}")
        return result


# ============================================================
# 命令行入口
# ============================================================
def _parse_grid_shape(value: str) -> tuple:
    """Parse a CLI grid specification such as ``3x3``."""
    try:
        rows_text, cols_text = value.lower().split("x")
        rows, cols = int(rows_text), int(cols_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid must use ROWSxCOLS, e.g. 3x3") from exc
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("grid dimensions must be positive")
    return rows, cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stitch SEM images with pure-translation alignment."
    )
    parser.add_argument(
        "--grid",
        type=_parse_grid_shape,
        metavar="ROWSxCOLS",
        help="match only direct neighbours in a known grid",
    )
    parser.add_argument(
        "--serpentine",
        action="store_true",
        help="interpret alternate grid rows in reverse acquisition order",
    )
    parser.add_argument(
        "--min-axis-overlap",
        type=float,
        default=0.70,
        help="reject matches unless one overlap axis reaches this fraction (default: 0.70)",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=10.0,
        help="prune connected graph edges above this global residual in pixels (default: 10)",
    )
    parser.add_argument("images", nargs="+", help="input TIFF image paths in acquisition order")
    args = parser.parse_args()

    if len(args.images) < 2:
        parser.error("at least two images are required")
    if args.serpentine and args.grid is None:
        parser.error("--serpentine requires --grid")
    if args.grid is not None and args.grid[0] * args.grid[1] != len(args.images):
        parser.error(
            f"grid {args.grid[0]}x{args.grid[1]} requires {args.grid[0] * args.grid[1]} images, "
            f"but received {len(args.images)}"
        )
    if not 0.0 <= args.min_axis_overlap <= 1.0:
        parser.error("--min-axis-overlap must be between 0 and 1")
    if args.residual_threshold <= 0:
        parser.error("--residual-threshold must be positive")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "stitched_panorama.tif")
    image_paths = [os.path.abspath(path) for path in args.images]
    for path in image_paths:
        if not os.path.isfile(path):
            parser.error(f"file not found: {path}")

    print(f"Found {len(image_paths)} images to stitch:")
    for path in image_paths:
        print(f"  - {os.path.basename(path)}")

    stitcher = SEMStitcher(
        sift_n_features=0,
        sift_contrast_threshold=0.01,
        sift_edge_threshold=200,
        flann_trees=5,
        flann_checks=100,
        lowe_ratio=0.75,
        ransac_reproj_thresh=5.0,
        ransac_max_iters=2000,
        clahe_clip_limit=2.0,
        clahe_tile_size=(8, 8),
    )

    result = stitcher.stitch_sequence(
        image_paths,
        grid_shape=args.grid,
        serpentine=args.serpentine,
        min_axis_overlap_ratio=args.min_axis_overlap,
        residual_threshold=args.residual_threshold,
    )

    if not cv2.imwrite(output_path, result):
        raise RuntimeError(f"Failed to save result to: {output_path}")
    print(f"\nResult saved to: {output_path}")

    # 预览
    try:
        display_img = result.copy()
        max_display_size = 1400
        h_disp, w_disp = display_img.shape
        scale = min(max_display_size / max(h_disp, w_disp), 1.0)
        if scale < 1.0:
            display_img = cv2.resize(display_img, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
            print(f"  Preview scale: {scale:.2f}x")
        cv2.imshow("Stitched Panorama", display_img)
        cv2.waitKey(100)
        try:
            visible = cv2.getWindowProperty("Stitched Panorama", cv2.WND_PROP_VISIBLE)
        except Exception:
            visible = -1
        if visible < 1:
            print("  (headless, skip display)")
            cv2.destroyAllWindows()
        else:
            print("  Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception:
        print("  (headless, skip display)")
        cv2.destroyAllWindows()

    print("\n===== All done! =====")
