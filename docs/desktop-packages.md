# SEM 图像拼接工具：桌面版

下载与你的系统对应的 ZIP 后完整解压。Python、Tk 与图像处理依赖已包含在包内。

| 文件 | 适用架构 | 启动方式 |
| --- | --- | --- |
| `SEM-Stitcher-macos-arm64.zip` | Apple Silicon Mac（M 系列） | 双击 `SEM_Stitcher.app` |
| `SEM-Stitcher-macos-intel.zip` | Intel Mac | 双击 `SEM_Stitcher.app` |
| `SEM-Stitcher-windows-x64.zip` | Windows x64 | 在 `SEM_Stitcher` 文件夹中双击 `SEM_Stitcher.exe`，保留旁边的 `_internal` |

Mac 版使用临时签名，未经过 Apple 公证；Windows 版未进行发布者签名。系统可能在首次运行时提示。确认来源后可参考 [Apple 的打开方式](https://support.apple.com/en-us/102445)。

`build.json` 记录来源提交、架构与依赖版本。每个包均在对应平台解压并运行打包后的应用，验证工作台启动、真实 SIFT 拼接、接缝并排对照和 16 位 TIFF 往返。外部 `.sha256` 文件用于校验 ZIP。

使用方法见 [项目说明](https://github.com/Haow1026/SEM-Homography#使用--usage)。
