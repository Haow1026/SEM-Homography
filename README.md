# SEM 图像拼接工具 (SEM Image Stitching Tool)

Mac 桌面应用，用于拼接扫描电子显微镜 (SEM) 图像。纯平移对齐（2 DOF），支持 3×3 网格、蛇形扫描与自动匹配。输入为位深一致的 8 位或 16 位灰度图像。

A Mac desktop app for stitching Scanning Electron Microscope (SEM) images. Pure translation alignment (2 DOF), supports 3×3 grids, serpentine scans, and arbitrary topologies.

---

## 功能特性  Features

- **纯平移对齐**：SEM stage 仅做 XY 移动，不做旋转和缩放；保留浮点位移到最终合成
- **混合对齐策略**：边缘图 SIFT（抗划痕）+ 相位相关（亚像素精细对齐）
- **稳健全局平差**：过滤小角落重叠，并在求解后剔除不一致匹配，避免错误边拉偏全景
- **网格/蛇形扫描模式**：已知网格时仅匹配上下左右直接邻居，避免对角线误匹配
- **亚像素羽化融合**：按到图像边缘的距离加权，统一归一化，减少黑边、重影及多图交汇接缝；结果不依赖叠加顺序
- **16 位 TIFF**：特征提取使用 8 位工作副本，原始位深保留到融合和导出；预览归一化不影响导出数据
- **多语言界面**：简体中文 / English / 日本語

---

## 安装  Installation

```bash
# 依赖
pip3 install opencv-contrib-python numpy customtkinter pillow

# 克隆
git clone https://github.com/Haow1026/SEM-Homography.git
cd SEM-Homography
```

---

## 使用  Usage

### GUI 模式（推荐）

```bash
python3 gui.py
```

1. 点击 **选择图片文件**，多选要拼接的 TIFF 图像
2. 点击 **开始拼接**，实时日志显示进度
3. 右侧预览拼接结果，点击 **导出全景图** 保存

### 命令行模式

```bash
python3 main.py image1.tif image2.tif image3.tif ...
```

传入要拼接的 TIFF 文件路径（至少 2 张，均为同一位深的灰度图），输出项目目录下的 `stitched_panorama.tif`。16 位输入输出仍为 16 位；GUI 导出 16 位结果时请选择 TIFF 或 PNG。

对于已知的 3×3 蛇形扫描，请按采集顺序传入图像并启用网格模式。例如采集顺序为
第一行从左到右、第二行从右到左、第三行从左到右时：

```bash
python3 main.py --grid 3x3 --serpentine \
  F82H-BA07-057_003.tif F82H-BA07-057_004.tif F82H-BA07-057_005.tif \
  F82H-BA07-057_006.tif F82H-BA07-057_007.tif F82H-BA07-057_008.tif \
  F82H-BA07-057_009.tif F82H-BA07-057_010.tif F82H-BA07-057_011.tif
```

网格模式只匹配上下左右邻居；默认自动模式也会过滤“仅角落相交”的小重叠匹配。
网格参数目前通过命令行或 Python API 设置，GUI 使用自动匹配模式。

- `--min-axis-overlap 0.70`：默认要求至少一个方向的重叠比例达到 70%。斜向扫描或不规则布局可按实际情况调低；两个方向仍必须有正重叠。
- `--residual-threshold 10`：全局匹配边残差阈值（像素）。剔除异常边后仍无法满足阈值时停止并报错，避免输出已知不一致的全景。

### 精度与数据说明

相位相关在原始灰度重叠区上计算，校正整数裁剪产生的坐标差后，将精细位移真正传入全局平差。
每张图只在最终合成时做一次双线性重采样；整数位置无需重采样。亚像素对齐会改变采样值，
羽化融合也会平均重叠区，输出不应被视为原始测量像素的逐点拷贝。软件不做自动曝光补偿。

弱纹理、周期重复结构、漂移、旋转或曝光差异仍可能造成误配或可见接缝；已知扫描布局时优先指定网格。
匹配图必须连通。算法只处理平移，不能修正扫描畸变。当前回归验证使用合成数据，实际 SEM 数据仍需检查对齐效果。

### 运行测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖已知整数/小数位移、低可信度相位相关、真实 SIFT 网格拼接、四图交汇、
叠加顺序不变性、16 位 TIFF 往返和无效匹配。GitHub Actions 在 Linux/Windows、Python 3.11/3.13 上运行。

---

## 算法流程  Algorithm

```
图像加载 → SIFT + 相位精对齐 → 几何筛选 → 全局平差 → 亚像素羽化融合
                │                    │
           ┌────┴────┐          ┌────┴────┐
      边缘图 SIFT    重叠几何筛选  残差剔除   网格邻接
      (抗划痕)    (拒绝角落匹配)  (异常边)  (可选)
```

| 阶段 | 算法 | 图像 | 目的 |
|------|------|------|------|
| 粗对齐 | SIFT + FLANN + RANSAC | 边缘图 (GaussianBlur + Sobel) | 抑制细小划痕，匹配宏观结构 |
| 精细对齐 | 相位相关 (Phase Correlation) | 原始灰度重叠区 | 校正粗位移，保留亚像素偏移 |
| 全局定位 | 加权最小二乘 + 残差剔除 | 可靠匹配边 + 中心锚定 | 抑制错误边，保持闭环约束 |
| 融合 | 距边权重 + 亚像素重采样 | 原位深图像 | 减少接缝，避免零权重黑边 |

---

## 文件结构  Files

```
SEM Homography/
├── main.py              # 拼接算法核心 (SEMStitcher)
├── gui.py               # macOS GUI (CustomTkinter 深色模式)
├── tests/               # 合成数据回归测试
├── .github/workflows/   # 测试及 Windows 打包
├── .gitignore           
└── README.md
```

---

## 语言切换  Language

界面右上角下拉菜单支持：

- **简体中文**（自动检测 `zh` 系统语言）
- **English**（默认）
- **日本語**（自动检测 `ja` 系统语言）

---

## 许可  License

MIT
