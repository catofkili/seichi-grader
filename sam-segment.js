// sam-segment.js — SlimSAM 点提示分割兜底（Xenova/slimsam-77-uniform ONNX）。
// ISNet 是显著性模型，对远景小人/骑车/夜景角色经常完全无响应（rawMax≈0）；
// SAM 靠检测框中心的提示点强制它分割指定位置，正好补这个短板。
// 全部在访客浏览器里跑，encoder 23MB + decoder 17MB，仅在需要兜底时才懒加载。
import { getSession, MODEL_BASE } from './ort-env.js';
import { createCanvas } from './canvas-util.js';

const ENC_URL = `${MODEL_BASE}/models/sam-encoder.onnx`;
const DEC_URL = `${MODEL_BASE}/models/sam-decoder.onnx`;
const SIZE = 1024;
// SamImageProcessor 的 ImageNet 归一化（÷255 后减均值除方差）
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// letterbox 到 1024（左上对齐，补 0），返回 CHW 张量和有效区尺寸
function preprocess(imageData) {
  const W = imageData.width, H = imageData.height;
  const s = SIZE / Math.max(W, H);
  const vw = Math.round(W * s), vh = Math.round(H * s);
  const c = createCanvas(SIZE, SIZE);
  const ctx = c.getContext('2d');
  const tmp = createCanvas(W, H);
  tmp.getContext('2d').putImageData(imageData, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, vw, vh);
  const d = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const plane = SIZE * SIZE;
  const chw = new Float32Array(3 * plane);
  for (let y = 0; y < vh; y++) {
    for (let x = 0; x < vw; x++) {
      const p = y * SIZE + x, i = p * 4;
      chw[p] = (d[i] / 255 - MEAN[0]) / STD[0];
      chw[plane + p] = (d[i + 1] / 255 - MEAN[1]) / STD[1];
      chw[2 * plane + p] = (d[i + 2] / 255 - MEAN[2]) / STD[2];
    }
  }
  return { chw, scale: s, vw, vh };
}

// SAM 失败的典型形态是吐出一个"矩形色块"（把裁剪区/车窗当对象）。
// 特征：mask 几乎填满自身外接框。真实角色剪影的填充率远低于此。
function looksRectangular(alpha, w, h) {
  let minX = w, minY = h, maxX = -1, maxY = -1, cnt = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (alpha[y * w + x] > 127) {
      cnt++;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  if (!cnt) return { empty: true };
  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const fill = cnt / (bw * bh);
  const cov = cnt / (w * h);
  // 填满外接框 92% 以上 = 矩形伪影；盖满几乎整个裁剪区 = 分割失败
  return { empty: false, reject: fill > 0.92 || cov > 0.9, fill, cov };
}

// 对裁剪区跑 SAM：innerBox 是检测框在裁剪区坐标系里的位置，
// 取框中心线上 3 个点做正向提示。返回 alpha(Uint8, 裁剪区尺寸)；拒收返回 null。
async function samMaskForBox(imageData, innerBox, opts = {}) {
  // SlimSAM 的 LayerNorm/Resize 组合在 WebGPU EP 尚不稳定，统一 WASM
  const { ort, session: enc } = await getSession(ENC_URL, { onProgress: opts.onProgress, eps: [['wasm']] });
  const { session: dec } = await getSession(DEC_URL, { onProgress: opts.onProgress, eps: [['wasm']] });

  const W = imageData.width, H = imageData.height;
  const { chw, scale, vw, vh } = preprocess(imageData);
  const encInput = new ort.Tensor('float32', chw, [1, 3, SIZE, SIZE]);
  let encOut, decOut, pointsTensor, labelsTensor;
  try {
  encOut = await enc.run({ pixel_values: encInput });
  const embeds = encOut[enc.outputNames[0]];
  const posEmbeds = encOut[enc.outputNames[1]];

  const cx = innerBox.x + innerBox.w / 2;
  // 默认提示点取框中心线三点；圈选路径可传 opts.points（裁剪区坐标，
  // 如用户笔迹质心）——姿势偏斜时质心比框中心更可能落在角色本体上。
  let pts = [
    [cx, innerBox.y + innerBox.h * 0.5],
    [cx, innerBox.y + innerBox.h * 0.25],
    [cx, innerBox.y + innerBox.h * 0.75],
  ];
  if (opts.points && opts.points.length) {
    pts = [...opts.points, [cx, innerBox.y + innerBox.h * 0.5]];
  }
  const coords = new Float32Array(pts.length * 2);
  pts.forEach((p, i) => { coords[i * 2] = p[0] * scale; coords[i * 2 + 1] = p[1] * scale; });
  const labels = new BigInt64Array(pts.length).fill(1n);

  pointsTensor = new ort.Tensor('float32', coords, [1, 1, pts.length, 2]);
  labelsTensor = new ort.Tensor('int64', labels, [1, 1, pts.length]);
  decOut = await dec.run({
    input_points: pointsTensor,
    input_labels: labelsTensor,
    image_embeddings: embeds,
    image_positional_embeddings: posEmbeds,
  });
  const ious = decOut.iou_scores.data;      // [1,1,3]
  const masks = decOut.pred_masks.data;     // [1,1,3,256,256] logits
  let best = 0;
  for (let i = 1; i < 3; i++) if (ious[i] > ious[best]) best = i;

  // 256×256 logits（对应补边后的 1024 空间）→ 画到 canvas → 裁有效区放大回裁剪尺寸
  const MS = 256;
  const mc = createCanvas(MS, MS);
  const mctx = mc.getContext('2d');
  const mimg = mctx.createImageData(MS, MS);
  const off = best * MS * MS;
  for (let i = 0; i < MS * MS; i++) {
    const v = masks[off + i] > 0 ? 255 : 0;
    mimg.data[i * 4] = v; mimg.data[i * 4 + 1] = v; mimg.data[i * 4 + 2] = v; mimg.data[i * 4 + 3] = 255;
  }
  mctx.putImageData(mimg, 0, 0);
  const oc = createCanvas(W, H);
  const octx = oc.getContext('2d');
  octx.imageSmoothingEnabled = true;
  octx.drawImage(mc, 0, 0, vw / 4, vh / 4, 0, 0, W, H);
  const od = octx.getImageData(0, 0, W, H).data;
  const alpha = new Uint8ClampedArray(W * H);
  for (let i = 0, p = 0; i < od.length; i += 4, p++) alpha[p] = od[i];

  const shape = looksRectangular(alpha, W, H);
  if (shape.empty || shape.reject) return null;
  return alpha;
  } finally {
    encInput.dispose?.();
    pointsTensor?.dispose?.();
    labelsTensor?.dispose?.();
    if (decOut) for (const value of Object.values(decOut)) value?.dispose?.();
    if (encOut) for (const value of Object.values(encOut)) value?.dispose?.();
  }
}

export { samMaskForBox };
