"""
SEM 图像拼接工具 (Image Stitching for SEM)
===========================================
使用 SIFT + FLANN + RANSAC 纯平移向量拼接多张扫描电子显微镜图像。
- 纯平移变换（2 DOF）：SEM stage 仅做 XY 移动，不做旋转和缩放
- 透明度羽化融合（Opacity Blending）：重叠区域线性渐变过渡，消除拼接缝
- 顺序拼接：逐张增量拼接，支持任意数量图像
"""

import cv2
import numpy as np
import os
import glob
import re


# ============================================================
# SEMStitcher 类：封装特征提取、匹配、平移计算、融合拼接
# ============================================================
class SEMStitcher:
    """SEM 图像拼接器，基于纯平移 + 透明度羽化融合"""

    def __init__(
        self,
        # SIFT 参数
        sift_n_features=0,
        sift_contrast_threshold=0.04,
        sift_edge_threshold=10,
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
            # 相位相关的峰值响应，值越高匹配越可信
            if response < 0.05:
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
    # 拼接两张图像（核心方法）
    # ================================================================
    def stitch_two(self, img1: np.ndarray, img2: np.ndarray,
                   label1: str = "img1", label2: str = "img2") -> np.ndarray:
        """
        拼接两张图像。

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
        print(f"  特征点: {label1}={len(kp1)}, {label2}={len(kp2)}")

        # 3. 计算平移量
        dx, dy, inliers, n_good = self._compute_translation(kp1, des1, kp2, des2)

        if dx is None:
            raise RuntimeError(
                f"无法计算平移量！内点={inliers}, Good Matches={n_good}。"
                f"请检查图像重叠是否足够。"
            )

        print(f"  Good Matches: {n_good}, RANSAC 内点: {inliers}")
        print(f"  平移向量: Δx={dx:.1f} px, Δy={dy:.1f} px")

        # 4. 融合拼接（保持原图亮度，仅重叠区羽化过渡）
        result = self._blend_images(img1, img2, dx, dy)
        print(f"  融合完成 → 画布尺寸: {result.shape[1]}x{result.shape[0]}")
        # 报告重叠量
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        x1, y1 = dx, dy
        x2, y2 = 0.0, 0.0
        ox_w = min(x1 + w1, x2 + w2) - max(x1, x2)
        ox_h = min(y1 + h1, y2 + h2) - max(y1, y2)
        if ox_w > 0 and ox_h > 0:
            print(f"  羽化重叠区域: {int(ox_w)}x{int(ox_h)} px "
                  f"({'水平' if ox_w <= ox_h else '垂直'}方向渐变)")

        return result

    # ================================================================
    # 结构对齐匹配（边缘图 SIFT + 相位相关）
    # ================================================================
    def _try_match_pair(self, img_a: np.ndarray, img_b: np.ndarray,
                        label_a: str = "A", label_b: str = "B") -> tuple:
        """
        尝试匹配两张图像，使用边缘图强制算法关注宏观结构。

        1) 提取边缘图 (GaussianBlur + Sobel)
        2) 在边缘图上 SIFT → FLANN → RANSAC 粗对齐
        3) 在边缘图上相位相关精细对齐

        返回: (dx, dy, quality)
          dx, dy  = img_a → img_b 的总平移量（None 表示匹配失败）
          quality = RANSAC 内点数量（用于 MST 边权重）
        """
        # ---- SIFT 粗对齐（在 CLAHE 增强图上，特征丰富） ----
        enhanced_a = self._preprocess(img_a)
        enhanced_b = self._preprocess(img_b)
        kp_a, des_a = self._extract_features(enhanced_a)
        kp_b, des_b = self._extract_features(enhanced_b)
        dx, dy, inliers, n_good = self._compute_translation(kp_a, des_a, kp_b, des_b)

        print(f"  特征点(CLAHE): {label_a}={len(kp_a)}, {label_b}={len(kp_b)}")
        if dx is not None:
            print(f"  SIFT粗对齐: Matches={n_good}, Inliers={inliers}, "
                  f"Δx={dx:.1f}, Δy={dy:.1f} px")
        else:
            print(f"  SIFT匹配失败: Matches={n_good}, Inliers={inliers}")

        if dx is None:
            return None, None, 0

        # ---- 提取边缘图 ----
        edge_a = self._extract_edge_map(img_a)
        edge_b = self._extract_edge_map(img_b)

        # ---- 相位相关精细对齐（在边缘图上，聚焦宏观结构） ----
        dx_fine, dy_fine = self._fine_align_phase(edge_a, edge_b, dx, dy)
        dx += dx_fine
        dy += dy_fine

        if abs(dx_fine) > 0.01 or abs(dy_fine) > 0.01:
            print(f"  相位相关精调: Δx={dx_fine:.2f}, Δy={dy_fine:.2f} px")

        return dx, dy, inliers  # quality = inlier count

    # ================================================================
    # 将单张图像添加到预分配的浮点画布（含曝光补偿 + 羽化融合）
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
            print(f"  羽化重叠: {ow}x{oh} px ({direction}渐变, "
                  f"实际内容={actual_overlap_px}/{total_px} px)")

    # ================================================================
    # 批量顺序拼接（连续匹配 + 全局定位）
    # ================================================================
    # ================================================================
    # 图优化全局定位 + 合成
    # ================================================================
    def stitch_sequence(self, image_paths: list) -> np.ndarray:
        """
        拼接多张图像（支持 2D 网格）。

        策略（解决一维链式累加误差）：
        1. 两两全对匹配所有可能重叠的图像对
        2. 用匹配质量（RANSAC 内点数）构建最大生成树 (MST)
        3. 从 MST 根节点 BFS 遍历，分配全局坐标
        4. 按坐标逐一将图像添加到画布，保留羽化融合

        这样拼接路径永远选择"最可靠的边缘"，切断误差累积。
        """
        n = len(image_paths)
        if n < 2:
            raise ValueError("至少需要 2 张图像才能拼接")

        print(f"\n{'='*60}")
        print(f"开始多图拼接，共 {n} 张图像")
        print(f"策略: 两两匹配 → MST 图优化 → 全局定位 → 羽化融合")
        print(f"{'='*60}")

        # ---- 阶段 1：加载所有图像 ----
        images = []
        for p in image_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"无法读取图像: {p}")
            images.append(img)
            print(f"  已加载: {os.path.basename(p)} ({img.shape[1]}x{img.shape[0]})")

        # ---- 阶段 2：两两全对匹配（N×(N-1)/2 对） ----
        print(f"\n--- 阶段 1: 全对匹配（共 {n*(n-1)//2} 对） ---")
        edges = []  # (quality, i, j, dx, dy)  quality = RANSAC 内点数

        for i in range(n):
            for j in range(i + 1, n):
                fname_i = os.path.basename(image_paths[i])
                fname_j = os.path.basename(image_paths[j])
                print(f"\n  尝试 [{i}↔{j}]: {fname_i} ↔ {fname_j}")

                dx, dy, quality = self._try_match_pair(
                    images[i], images[j], label_a=fname_i, label_b=fname_j
                )

                if dx is not None and quality >= 8:
                    edges.append((quality, i, j, dx, dy))
                    print(f"  ✓ 匹配成功，质量={quality}")
                else:
                    print(f"  ✗ 匹配失败或无重叠")

        if len(edges) < n - 1:
            raise RuntimeError(
                f"匹配边数量不足 ({len(edges)} < {n-1})，无法构成连通图！"
                f"请检查图像重叠是否足够。"
            )

        # ---- 阶段 3：Kruskal 最大生成树 (MST) ----
        print(f"\n--- 阶段 2: 构建最大生成树 (MST) ---")
        edges.sort(key=lambda e: e[0], reverse=True)  # 按质量降序

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        mst_edges = []
        for weight, i, j, dx, dy in edges:
            if union(i, j):
                mst_edges.append((i, j, dx, dy, weight))
                fname_i = os.path.basename(image_paths[i])
                fname_j = os.path.basename(image_paths[j])
                print(f"  MST 边: [{i}]{fname_i} ↔ [{j}]{fname_j} "
                      f"(质量={weight}, Δx={dx:.1f}, Δy={dy:.1f})")
            if len(mst_edges) == n - 1:
                break

        # ---- 阶段 4: BFS 分配全局坐标 ----
        print(f"\n--- 阶段 3: 分配全局坐标 ---")
        # 构建邻接表
        adj = [[] for _ in range(n)]
        for i, j, dx, dy, w in mst_edges:
            # 边 (i→j): pos[j] = pos[i] - dx
            adj[i].append((j, dx, dy))
            # 反向 (j→i): pos[i] = pos[j] - (-dx) = pos[j] + dx
            adj[j].append((i, -dx, -dy))

        # 根节点：选连接度最高的（在 MST 中边最多）
        root = max(range(n), key=lambda x: len(adj[x]))
        print(f"  MST 根节点: [{root}] {os.path.basename(image_paths[root])} "
              f"(连接度={len(adj[root])})")

        global_x = [None] * n
        global_y = [None] * n
        global_x[root] = 0.0
        global_y[root] = 0.0

        stack = [root]
        while stack:
            u = stack.pop()
            for v, dx, dy in adj[u]:
                if global_x[v] is None:
                    global_x[v] = global_x[u] - dx
                    global_y[v] = global_y[u] - dy
                    stack.append(v)

        for i in range(n):
            fname = os.path.basename(image_paths[i])
            print(f"  [{i}] {fname}: ({global_x[i]:.0f}, {global_y[i]:.0f})")

        # ---- 阶段 5：预分配画布 + 逐一羽化融合 ----
        print(f"\n--- 阶段 4: 融合合成 ---")

        all_left = []
        all_top = []
        all_right = []
        all_bottom = []
        for i in range(n):
            gx = global_x[i]
            gy = global_y[i]
            h, w = images[i].shape
            all_left.append(gx)
            all_top.append(gy)
            all_right.append(gx + w)
            all_bottom.append(gy + h)

        canvas_x_min = int(np.floor(min(all_left)))
        canvas_y_min = int(np.floor(min(all_top)))
        canvas_w = int(np.ceil(max(all_right) - canvas_x_min))
        canvas_h = int(np.ceil(max(all_bottom) - canvas_y_min))

        print(f"  画布尺寸: {canvas_w} x {canvas_h}")

        sum_img = np.zeros((canvas_h, canvas_w), dtype=np.float64)
        sum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)

        # 按 MST 遍历顺序添加（从根开始广度优先）
        order = []
        visited = [False] * n
        from collections import deque
        q = deque([root])
        visited[root] = True
        while q:
            u = q.popleft()
            order.append(u)
            for v, _, _ in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)

        for idx in order:
            fname = os.path.basename(image_paths[idx])
            x_pos = int(round(global_x[idx])) - canvas_x_min
            y_pos = int(round(global_y[idx])) - canvas_y_min
            x_pos = int(x_pos)
            y_pos = int(y_pos)
            print(f"\n  添加 [{idx}] {fname} at ({x_pos}, {y_pos})")
            self._blend_onto_canvas(sum_img, sum_weight, images[idx], x_pos, y_pos)

        valid = sum_weight > 1e-6
        sum_img[valid] /= sum_weight[valid]
        result = np.clip(np.round(sum_img), 0, 255).astype(np.uint8)

        print(f"\n{'='*60}")
        print(f"全部 {n} 张图像拼接完成！")
        print(f"最终尺寸: {result.shape[1]} x {result.shape[0]}")
        print(f"{'='*60}")

        return result


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    # ---- 配置 ----
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR = os.path.join(BASE_DIR, "F82H-BA07-054_2026_06_22")
    OUTPUT_PATH = os.path.join(BASE_DIR, "stitched_panorama.tif")

    # ---- 自动读取目标文件夹下的所有匹配图像 ----
    pattern = os.path.join(IMG_DIR, "F82H-BA07-054_*.tif")
    image_paths = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"_(\d+)\.tif$", p).group(1))
    )

    if len(image_paths) < 2:
        raise RuntimeError(
            f"在 {IMG_DIR} 下找到的图像少于 2 张！\n"
            f"匹配模式: {pattern}\n"
            f"找到: {image_paths}"
        )

    print(f"找到 {len(image_paths)} 张待拼接图像:")
    for p in image_paths:
        print(f"  - {os.path.basename(p)}")

    # ---- 创建拼接器 ----
    stitcher = SEMStitcher(
        sift_n_features=0,
        sift_contrast_threshold=0.04,
        sift_edge_threshold=10,
        flann_trees=5,
        flann_checks=100,
        lowe_ratio=0.75,
        ransac_reproj_thresh=5.0,
        ransac_max_iters=2000,
        clahe_clip_limit=2.0,
        clahe_tile_size=(8, 8),
    )

    # ---- 执行拼接 ----
    result = stitcher.stitch_sequence(image_paths)

    # ---- 保存 ----
    cv2.imwrite(OUTPUT_PATH, result)
    print(f"\n结果已保存至: {OUTPUT_PATH}")

    # ---- 预览（无 GUI 环境自动跳过） ----
    try:
        display_img = result.copy()
        max_display_size = 1400
        h_disp, w_disp = display_img.shape
        scale = min(max_display_size / max(h_disp, w_disp), 1.0)
        if scale < 1.0:
            display_img = cv2.resize(display_img, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
            print(f"  预览缩放: {scale:.2f}x")

        cv2.imshow("Stitched Panorama", display_img)
        cv2.waitKey(100)
        try:
            visible = cv2.getWindowProperty("Stitched Panorama", cv2.WND_PROP_VISIBLE)
        except Exception:
            visible = -1
        if visible < 1:
            print("  (无 GUI 环境，跳过窗口显示)")
            cv2.destroyAllWindows()
        else:
            print("  按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception:
        print("  (无 GUI 环境，跳过窗口显示)")
        cv2.destroyAllWindows()

    print("\n===== 全部完成! =====")
