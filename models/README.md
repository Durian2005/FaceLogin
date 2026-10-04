# models/ — 人脸模型文件

这两个 ONNX 模型来自 [OpenCV Zoo](https://github.com/opencv/opencv_zoo)，
**版权归原作者所有，遵循各自的上游许可证**，不在本仓库 MIT 授权的覆盖范围内。

| 文件 | 大小 (字节) | 用途 | 许可证 |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | 232,589 | 人脸检测（YuNet） | MIT |
| `face_recognition_sface_2021dec.onnx` | 38,696,353 | 人脸识别 / 特征提取（SFace） | Apache-2.0 |

许可证原文已随模型一并放在本目录：

- `LICENSE_yunet.txt` — MIT，`Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>`
- `LICENSE_sface.txt` — Apache License 2.0（上游文件中的版权人为未填写的占位符，照原样保留）

**再分发时请一并保留这两个许可证文件。**

## 校验值（SHA-256）

```
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  face_detection_yunet_2023mar.onnx
0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79  face_recognition_sface_2021dec.onnx
```

## 需要重新下载时

模型在 OpenCV Zoo 仓库里由 **Git LFS** 托管，直接访问 `raw.githubusercontent.com`
只会拿到 LFS 指针文件（131 字节）而不是模型本体。请用仓库自带的脚本：

```bash
python tools/download_models.py
```

脚本会走 LFS 直链下载、校验 SHA-256，并把两份许可证一起取回。

## 关于路径（重要）

程序默认就在本目录（`<项目根>/models/`）查找模型，**不需要额外配置**。

但 OpenCV 无法从**非 ASCII 路径**读取 ONNX。如果你的项目目录或用户名含中文，
`app/model_paths.py` 会自动把模型镜像到一个纯 ASCII 的缓存目录再加载。
想手动指定位置时，设置环境变量 `FACELOGIN_MODELS` 即可：

```bat
set FACELOGIN_MODELS=D:\some\ascii\path\models
```

> 顺带一提，这也是本项目**默认值不使用绝对路径**的原因 —— 每个人把仓库
> clone 到哪儿都不一样，写死路径会让别人一步都跑不起来。
