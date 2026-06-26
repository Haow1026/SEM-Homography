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
    # 匹配相邻图像对（仅两张原图之间，不涉及 base）
    # ================================================================
    def _match_pair(self, img_a: np.ndarray, img_b: np.ndarray,
                    label_a: str = "A", label_b: str = "B") -> tuple:
        """
        匹配相邻两张图像，返回相对平移量。

        img_a 平移到 img_b 坐标系的 (dx, dy)。

        返回: (dx, dy, inliers, n_good)
        """
        kp_a, des_a = self._extract_features(self._preprocess(img_a))
        kp_b, des_b = self._extract_features(self._preprocess(img_b))
        dx, dy, inliers, n_good = self._compute_translation(kp_a, des_a, kp_b, des_b)

        print(f"  特征点: {label_a}={len(kp_a)}, {label_b}={len(kp_b)}")
        print(f"  Good Matches: {n_good}, RANSAC 内点: {inliers}")
        if dx is not None:
            print(f"  相对平移: Δx={dx:.1f} px, Δy={dy:.1f} px")

        return dx, dy, inliers, n_good

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
                    sum_weight[o_c_y1:o_c_y1 + oh, o_c_x1:o_c_x1 + ow] *= (1.0 - ramp)
                    w_new_mask[o_n_y1:o_n_y1 + oh, o_n_x1:o_n_x1 + ow] = ramp
                else:
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
                    sum_weight[o_c_y1:o_c_y1 + oh, o_c_x1:o_c_x1 + ow] *= (1.0 - ramp)
                    w_new_mask[o_n_y1:o_n_y1 + oh, o_n_x1:o_n_x1 + ow] = ramp
                else:
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
    def stitch_sequence(self, image_paths: list) -> np.ndarray:
        """
        拼接多张图像。

        策略：
        1. 相邻图像对匹配 → 获取相对位移
        2. 累加得到每张图像在全局画布上的坐标
        3. 按坐标顺序逐一将图像添加到画布，重叠区做曝光补偿 + 羽化

        参数:
            image_paths: 已排序的图像文件路径列表
        返回:
            最终拼接结果 (uint8 灰度图)
        """
        n = len(image_paths)
        if n < 2:
            raise ValueError("至少需要 2 张图像才能拼接")

        print(f"\n{'='*60}")
        print(f"开始多图顺序拼接，共 {n} 张图像")
        print(f"策略: 相邻匹配 → 全局定位 → 逐一融合")
        print(f"{'='*60}")

        # ---- 阶段 1：加载所有图像 ----
        images = []
        for p in image_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"无法读取图像: {p}")
            images.append(img)
            print(f"  已加载: {os.path.basename(p)} ({img.shape[1]}x{img.shape[0]})")

        # ---- 阶段 2：匹配相邻图像对，获取相对位移 ----
        print(f"\n--- 阶段 1: 相邻匹配 ---")
        rel_dx = [0.0]  # img[0] 无相对位移
        rel_dy = [0.0]
        match_inliers = [0]

        for i in range(1, n):
            fname_a = os.path.basename(image_paths[i - 1])
            fname_b = os.path.basename(image_paths[i])
            print(f"\n  匹配 [{i-1}→{i}]: {fname_a} → {fname_b}")

            dx, dy, inliers, n_good = self._match_pair(
                images[i - 1], images[i], label_a=fname_a, label_b=fname_b
            )

            if dx is None:
                raise RuntimeError(
                    f"相邻匹配失败 [{i-1}→{i}]！"
                    f"内点={inliers}, Good Matches={n_good}"
                )

            rel_dx.append(dx)
            rel_dy.append(dy)
            match_inliers.append(inliers)

        # ---- 阶段 3：累加全局坐标 ----
        # pos[i] = pos[i-1] - (rel_dx[i], rel_dy[i])
        # 因为 rel_dx 是 img[i-1] → img[i] 的平移，所以 img[i] 在 img[i-1] 的 (-dx, -dy) 方向
        print(f"\n--- 阶段 2: 全局坐标 ---")
        global_x = [0.0]   # 纯 Python float
        global_y = [0.0]
        for i in range(1, n):
            global_x.append(float(global_x[-1] - float(rel_dx[i])))
            global_y.append(float(global_y[-1] - float(rel_dy[i])))
            fname = os.path.basename(image_paths[i])
            print(f"  [{i}] {fname}: 全局位置 ({global_x[-1]:.0f}, {global_y[-1]:.0f}) "
                  f"| 相对位移 ({-rel_dx[i]:.0f}, {-rel_dy[i]:.0f})")

        # ---- 阶段 4：预分配画布 + 逐一融合 ----
        print(f"\n--- 阶段 3: 融合合成 ---")

        # 计算全局包围盒
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

        print(f"  全局包围盒: ({canvas_x_min}, {canvas_y_min}) → "
              f"({canvas_x_min + canvas_w}, {canvas_y_min + canvas_h})")
        print(f"  画布尺寸: {canvas_w} x {canvas_h}")

        # 预分配浮点画布
        sum_img = np.zeros((canvas_h, canvas_w), dtype=np.float64)
        sum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)

        # 逐一添加图像（按原始顺序，连续图像共享最多重叠）
        for i in range(n):
            fname = os.path.basename(image_paths[i])
            x_pos = int(round(global_x[i])) - canvas_x_min
            y_pos = int(round(global_y[i])) - canvas_y_min
            # 确保是 Python int
            x_pos = int(x_pos)
            y_pos = int(y_pos)
            print(f"\n  添加 [{i}] {fname} at ({x_pos}, {y_pos})")
            self._blend_onto_canvas(sum_img, sum_weight, images[i], x_pos, y_pos)

        # 最终归一化
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
