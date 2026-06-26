# SEM 图像拼接工具 (SEM Image Stitching Tool)

Mac 桌面应用，用于拼接扫描电子显微镜 (SEM) 图像。纯平移对齐（2 DOF），支持 3×3 网格、蛇形扫描等任意拓扑。

A Mac desktop app for stitching Scanning Electron Microscope (SEM) images. Pure translation alignment (2 DOF), supports 3×3 grids, serpentine scans, and arbitrary topologies.

---

## 功能特性  Features

- **纯平移对齐**：SEM stage 仅做 XY 移动，不做旋转和缩放，保留原始像素
- **混合对齐策略**：边缘图 SIFT（抗划痕）+ 相位相关（亚像素精细对齐）
- **全局平差**：加权最小二乘全局坐标求解，消除链式累积误差
- **羽化融合**：重叠区线性透明度渐变，无拼接缝，无曝光失真
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

传入要拼接的 TIFF 文件路径（至少 2 张），输出 `stitched_panorama.tif`。

---

## 算法流程  Algorithm

```
图像加载 → 两两全对匹配 → 最小二乘全局平差 → 羽化融合
                │                    │
           ┌────┴────┐          ┌────┴────┐
      边缘图 SIFT    相位相关    中心锚定   行级微调
      (抗划痕)    (CLAHE 图)   (1000x权重)
```

| 阶段 | 算法 | 图像 | 目的 |
|------|------|------|------|
| 粗对齐 | SIFT + FLANN + RANSAC | 边缘图 (GaussianBlur + Sobel) | 免疫划痕，匹配宏观结构 |
| 精细对齐 | 相位相关 (Phase Correlation) | CLAHE 增强图 | 亚像素精度，保留圆孔结构 |
| 全局定位 | 加权最小二乘 | 所有匹配边 + 中心锚定 | 消除链式误差，闭环约束 |
| 融合 | 线性羽化 (Linear Blending) | 原始图像 | 无曝光失真 |

---

## 文件结构  Files

```
SEM Homography/
├── main.py              # 拼接算法核心 (SEMStitcher)
├── gui.py               # macOS GUI (CustomTkinter 深色模式)
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
