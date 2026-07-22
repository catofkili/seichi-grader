#!/usr/bin/env python3
"""对比 fp16 原版与 weight-only int8 版 ISNet 的输出差异。

复刻 ai-segment.js 的预处理（最长边缩到 size、左上对齐补零、/255-ImageNet均值）
与后处理（有效区 min-max 归一化）。指标：
- raw MAE / max|diff|（模型原始输出）
- 归一化 mask 的 MAE 与阈值(>110/255) IoU —— 对应 app 实际用法
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)


def preprocess(img: Image.Image, size: int):
    w, h = img.size
    scale = size / max(w, h)
    vw, vh = round(w * scale), round(h * scale)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    resized = np.asarray(img.convert("RGB").resize((vw, vh), Image.BILINEAR), dtype=np.float32)
    canvas[:vh, :vw] = resized / 255.0 - MEAN
    chw = canvas.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(chw), vw, vh


def normalize(raw: np.ndarray, vw: int, vh: int):
    valid = raw[:vh, :vw]
    mn, mx = valid.min(), valid.max()
    rng = (mx - mn) or 1.0
    return np.clip((raw - mn) / rng, 0, 1), mx


def main():
    fp16_path, w8_path, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
    imgs = sys.argv[4:]
    so = ort.SessionOptions()
    s_fp16 = ort.InferenceSession(fp16_path, so, providers=["CPUExecutionProvider"])
    s_w8 = ort.InferenceSession(w8_path, so, providers=["CPUExecutionProvider"])
    iname = s_fp16.get_inputs()[0].name

    for p in imgs:
        x, vw, vh = preprocess(Image.open(p), size)
        r16 = s_fp16.run(None, {iname: x})[0][0, 0]
        r8 = s_w8.run(None, {s_w8.get_inputs()[0].name: x})[0][0, 0]
        raw_mae = float(np.abs(r16 - r8).mean())
        raw_max = float(np.abs(r16 - r8).max())
        n16, mx16 = normalize(r16, vw, vh)
        n8, mx8 = normalize(r8, vw, vh)
        mask_mae = float(np.abs(n16 - n8)[:vh, :vw].mean())
        t16 = n16[:vh, :vw] > (110 / 255)
        t8 = n8[:vh, :vw] > (110 / 255)
        union = (t16 | t8).sum()
        iou = float((t16 & t8).sum() / union) if union else 1.0
        print(f"{Path(p).name:24s} rawMAE={raw_mae:.5f} rawMax={raw_max:.4f} "
              f"maskMAE={mask_mae:.5f} IoU={iou:.4f} rawMax16={mx16:.3f} rawMax8={mx8:.3f}")


if __name__ == "__main__":
    main()
