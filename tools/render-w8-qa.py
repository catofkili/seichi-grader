#!/usr/bin/env python3
"""生成 fp16 vs w8 抠图目检对比表：棋盘格背景，[原图 | fp16 | w8] × [1024 | 512]。"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SRC = ROOT / "test-anime.png"
OUT = ROOT / "work" / "isnet-w8-qa.png"


def preprocess(img, size):
    w, h = img.size
    scale = size / max(w, h)
    vw, vh = round(w * scale), round(h * scale)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[:vh, :vw] = np.asarray(img.convert("RGB").resize((vw, vh), Image.BILINEAR),
                                  dtype=np.float32) / 255.0 - MEAN
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), vw, vh


def cutout(img, model_path, size):
    x, vw, vh = preprocess(img, size)
    s = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    raw = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
    valid = raw[:vh, :vw]
    mn, mx = valid.min(), valid.max()
    mask = np.clip((valid - mn) / ((mx - mn) or 1), 0, 1)
    alpha = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    checker = make_checker(img.size)
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    checker.paste(rgba, (0, 0), rgba)
    return checker


def make_checker(size, cell=24):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    board = ((xx // cell + yy // cell) % 2 * 40 + 180).astype(np.uint8)
    return Image.merge("RGB", [Image.fromarray(board)] * 3)


def main():
    img = Image.open(SRC)
    thumb_w = 640
    rows = []
    for size, m16, m8 in [
        (1024, ROOT / "models/isnet-anime-fp16.onnx", ROOT / "models/isnet-anime-w8.onnx"),
        (512, ROOT / "models/isnet-anime-512-fp16.onnx", ROOT / "models/isnet-anime-512-w8.onnx"),
    ]:
        cells = [img.convert("RGB"), cutout(img, m16, size), cutout(img, m8, size)]
        scale = thumb_w / img.width
        cells = [c.resize((thumb_w, round(img.height * scale))) for c in cells]
        row = Image.new("RGB", (thumb_w * 3 + 20, cells[0].height), "white")
        for i, c in enumerate(cells):
            row.paste(c, (i * (thumb_w + 10), 0))
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + 10), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 10
    OUT.parent.mkdir(exist_ok=True)
    sheet.save(OUT)
    print("saved:", OUT, sheet.size)


if __name__ == "__main__":
    main()
