"""
SEM 图像拼接工具 (Image Stitching for SEM)
===========================================
使用 SIFT + FLANN + RANSAC 纯平移向量拼接多张扫描电子显微镜图像。
- 纯平移变换（2 DOF）：SEM stage 仅做 XY 移动，不做旋转和缩放
- 透明度羽化融合（Opacity Blending）：重叠区域线性渐变过渡，消除拼接缝
- 全局平差定位 + 亚像素羽化融合，支持多图网格
"""

import argparse
from collections import deque
import cv2
import numpy as np
import os


class StitchingCancelled(Exception):
    """Raised at a safe checkpoint when the caller requests cancellation."""


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
        self._cancel_callback = None

    def set_cancel_callback(self, callback):
        """Set a thread-safe predicate, e.g. threading.Event.is_set."""
        self._cancel_callback = callback

    def _check_cancelled(self):
        if self._cancel_callback is not None and self._cancel_callback():
            raise StitchingCancelled("Stitching cancelled")

    def set_progress_callback(self, callback):
        """设置进度回调函数 callback(msg: str)，供 GUI 实时显示日志"""
        self._progress_callback = callback

    def _log(self, msg: str):
        """统一日志输出：有回调则发给 GUI，否则 print"""
        self._check_cancelled()
        if self._progress_callback:
            self._progress_callback(msg)
        else:
            print(msg)

    @staticmethod
    def _matching_image(img):
        """Create an 8-bit feature image without modifying source intensities."""
        SEMStitcher._validate_images([img])
        if img.dtype == np.uint8:
            return img
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ================================================================
    # 预处理：CLAHE 增强对比度（仅用于特征提取，不修改原图）
    # ================================================================
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """对灰度图应用 CLAHE 增强局部对比度"""
        return self.clahe.apply(self._matching_image(img))

    # ================================================================
    # 边缘图提取：降噪 → Sobel → 归一化，强制匹配宏观结构
    # ================================================================
    def _extract_edge_map(self, img: np.ndarray) -> np.ndarray:
        """
        提取边缘梯度幅值图，用于 SIFT 粗对齐。

        1) CLAHE 增强局部对比度（提升暗区的梯度响应）
        2) GaussianBlur 模糊掉高频划痕/噪点（σ≈3, kernel=11）
        3) Sobel 提取梯度幅值（宏观边缘，如圆孔边界）
        4) 归一化到 0-255
        """
        # 先增强对比度，否则暗区 Sobel 梯度太弱
        enhanced = self._preprocess(img)
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
        self._check_cancelled()
        features = self.sift.detectAndCompute(img, None)
        self._check_cancelled()
        return features

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
        self._check_cancelled()

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

        for iteration in range(self.ransac_max_iters):
            if iteration % 64 == 0:
                self._check_cancelled()
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

        self._check_cancelled()

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

        # Remove the DC component; featureless overlaps cannot constrain a shift.
        patch1 -= patch1.mean()
        patch2 -= patch2.mean()
        if patch1.std() < 1e-6 or patch2.std() < 1e-6:
            return 0.0, 0.0

        # Hann 窗：抑制 FFT 边缘效应
        window = np.outer(np.hanning(oh), np.hanning(ow)).astype(np.float32)

        # phaseCorrelate centres its peak at half the padded FFT dimensions.
        # Odd optimal sizes (e.g. 225) introduce a half-pixel offset. Explicitly
        # pad both windowed patches to even optimal sizes before correlation.
        fft_h, fft_w = cv2.getOptimalDFTSize(oh), cv2.getOptimalDFTSize(ow)
        if fft_h % 2:
            fft_h = cv2.getOptimalDFTSize(fft_h + 1)
        if fft_w % 2:
            fft_w = cv2.getOptimalDFTSize(fft_w + 1)
        padding = ((0, fft_h - oh), (0, fft_w - ow))

        # 相位相关
        try:
            shift, response = cv2.phaseCorrelate(
                np.pad(patch1 * window, padding), np.pad(patch2 * window, padding)
            )
            if not np.isfinite([*shift, response]).all() or response < 0.05:
                return 0.0, 0.0
            # The patches were cropped at the *rounded* coarse shift. Convert
            # their residual back to a correction of the original float shift.
            dx_fine = dx_int + float(shift[0]) - dx_coarse
            dy_fine = dy_int + float(shift[1]) - dy_coarse
            # 修正幅度 >10px → 相位相关跳到错误圆孔，丢弃
            if abs(dx_fine) > 10.0 or abs(dy_fine) > 10.0:
                dx_fine, dy_fine = 0.0, 0.0
        except cv2.error:
            dx_fine, dy_fine = 0.0, 0.0

        return dx_fine, dy_fine

    # ================================================================
    # 线性羽化融合 (Linear Blending)
    def _blend_images(self, img1, img2, dx, dy) -> np.ndarray:
        """Place img1 at (dx, dy) relative to img2, retaining subpixel shifts."""
        return self._compose_images([img1, img2], [(dx, dy), (0.0, 0.0)])

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
            拼接后保持输入位深的灰度图
        """
        # Use the same coarse/fine registration path as multi-image stitching.
        dx, dy, quality = self._try_match_pair(img1, img2, label1, label2)
        if dx is None:
            raise RuntimeError("无法计算平移量，请检查图像是否有足够的重叠和纹理。")
        self._log(f"  Inliers: {quality}, translation: ({dx:.3f}, {dy:.3f}) px")

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
        3) 在原始灰度重叠区上相位相关精细对齐

        返回: (dx, dy, quality)
          dx, dy  = img_a → img_b 的总平移量（None 表示匹配失败）
          quality = RANSAC 内点数量（用于全局平差权重）
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

        # Refine on original intensities; independent CLAHE tile boundaries can
        # introduce artificial structure in otherwise identical overlap patches.
        dx_fine, dy_fine = self._fine_align_phase(img_a, img_b, dx, dy)

        if abs(dx_fine) > 0.01 or abs(dy_fine) > 0.01:
            self._log(f"  Phase corr fine: Δx={dx_fine:.2f}, Δy={dy_fine:.2f} px")

        return dx + dx_fine, dy + dy_fine, inliers

    @staticmethod
    def _validate_images(images):
        """Preserve scientific grayscale data; never silently mix bit depths."""
        if not images:
            raise ValueError("No images to compose")
        dtype = images[0].dtype
        for image in images:
            if image.ndim != 2 or image.size == 0:
                raise ValueError("Expected non-empty grayscale images")
            if image.dtype not in (np.uint8, np.uint16):
                raise ValueError("Only uint8 and uint16 grayscale images are supported")
            if image.dtype != dtype:
                raise ValueError("All images must have the same bit depth")
        return dtype

    def _blend_onto_canvas(self, sum_img, sum_weight, new_img, x_pos, y_pos):
        """Accumulate a translated tile with positive distance-to-edge weights.

        Warp premultiplied intensities and weights together. This avoids dark
        interpolation borders and keeps the result independent of tile order,
        including at three-/four-tile junctions. Only a tile-sized ROI is warped.
        """
        self._check_cancelled()
        h, w = new_img.shape
        x0, y0 = int(np.floor(x_pos)), int(np.floor(y_pos))
        x1, y1 = int(np.ceil(x_pos + w)), int(np.ceil(y_pos + h))
        if x0 < 0 or y0 < 0 or x1 > sum_img.shape[1] or y1 > sum_img.shape[0]:
            raise ValueError("Translated image falls outside the canvas")

        x_dist = np.minimum(np.arange(w) + 1, w - np.arange(w))
        y_dist = np.minimum(np.arange(h) + 1, h - np.arange(h))
        weights = np.minimum(y_dist[:, None], x_dist[None, :]).astype(np.float64)
        weighted = new_img.astype(np.float64) * weights
        fractional_x, fractional_y = x_pos - x0, y_pos - y0
        if fractional_x != 0.0 or fractional_y != 0.0:
            transform = np.array([[1, 0, fractional_x], [0, 1, fractional_y]], dtype=np.float64)
            size = (x1 - x0, y1 - y0)
            weighted = cv2.warpAffine(weighted, transform, size, flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            self._check_cancelled()
            weights = cv2.warpAffine(weights, transform, size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sum_img[y0:y1, x0:x1] += weighted
        sum_weight[y0:y1, x0:x1] += weights

    def _compose_images(self, images, positions):
        """Blend grayscale tiles once at their global floating-point positions."""
        dtype = self._validate_images(images)
        self._check_cancelled()
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (len(images), 2) or not np.isfinite(positions).all():
            raise ValueError("Each image requires a finite (x, y) position")
        # Remove only solver round-off, not real subpixel displacement.
        rounded = np.rint(positions)
        positions = np.where(np.abs(positions - rounded) < 1e-7, rounded, positions)
        origin = np.floor(positions.min(axis=0))
        positions = positions - origin
        sizes = np.array([(image.shape[1], image.shape[0]) for image in images])
        canvas_w, canvas_h = np.ceil((positions + sizes).max(axis=0)).astype(int)
        self._log(f"  Canvas: {canvas_w} x {canvas_h}")
        sum_img = np.zeros((canvas_h, canvas_w), dtype=np.float64)
        self._check_cancelled()
        sum_weight = np.zeros_like(sum_img)
        for image, (x_pos, y_pos) in zip(images, positions):
            self._blend_onto_canvas(sum_img, sum_weight, image, x_pos, y_pos)
        self._check_cancelled()
        np.divide(sum_img, sum_weight, out=sum_img, where=sum_weight > 0)
        return np.clip(np.rint(sum_img), 0, np.iinfo(dtype).max).astype(dtype)

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
        if not np.isfinite(residual_threshold) or residual_threshold <= 0:
            raise ValueError("residual_threshold must be finite and positive")

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
            self._check_cancelled()
            img = cv2.imread(os.fspath(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {path}")
            images.append(img)
            self._log(f"  Loaded: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

        self._validate_images(images)

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

                if dx is None or dy is None or quality < 8 or not np.isfinite([dx, dy]).all():
                    self._log("  No match / no overlap")
                    continue

                overlap_w, overlap_h, axis_ratio = self._overlap_geometry(
                    images[i], images[j], dx, dy
                )
                if overlap_w <= 0 or overlap_h <= 0 or axis_ratio < min_axis_overlap_ratio:
                    self._log(
                        f"  Reject insufficient overlap: {overlap_w:.0f}x{overlap_h:.0f} px, "
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
            self._check_cancelled()
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

        if worst_residual > residual_threshold:
            raise RuntimeError(
                f"Inconsistent image matches: worst residual {worst_residual:.2f} px "
                f"exceeds {residual_threshold:.2f} px. Check overlap, scan order, "
                "and grid settings before exporting a panorama."
            )

        residual_x_text = f"X residual={residual_x[0]:.1f}" if len(residual_x) > 0 else "X determined"
        residual_y_text = f"Y residual={residual_y[0]:.1f}" if len(residual_y) > 0 else "Y determined"
        self._log(f"  {residual_x_text}, {residual_y_text} (rank X={rank_x}, Y={rank_y})")
        for index in range(n):
            self._log(
                f"  [{index}] {os.path.basename(image_paths[index])}: "
                f"({global_x[index]:.3f}, {global_y[index]:.3f})"
            )

        # Each image is resampled once, directly into global coordinates.
        self._log("\n--- Phase 3: Subpixel feather composition ---")
        result = self._compose_images(images, list(zip(global_x, global_y)))

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
    if not np.isfinite(args.residual_threshold) or args.residual_threshold <= 0:
        parser.error("--residual-threshold must be finite and positive")

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
