// detect.js — 浏览器内动画人物检测（YOLOv8s ONNX，deepghs/anime_person_detection）。
// 用途：先找到"哪里有角色"，再把每个框裁下来交给 ISNet 抠图，
// 解决显著性模型抠不到小角色、误抠食物/建筑的问题。
import { getSession, evictSession } from './ort-env.js';
import { createCanvas } from './canvas-util.js';

const MODEL_URL = './models/person-detect.onnx';

// 检测人物框。返回 [{x, y, w, h, score}]（原图像素坐标，按置信度降序）。
// opts.size: 推理分辨率，默认 1024；远景小人用 1536 + 低 conf。
async function detectPersons(imageData, opts = {}) {
  const size = opts.size || 1024;
  const conf = opts.conf ?? 0.25;
  const iouThr = opts.iou ?? 0.5;
  const maxDet = opts.maxDet ?? 8;
  // YOLOv8 的 DFL Softmax（axis=1）在 onnxruntime-web 1.19 WebGPU EP 运行期报错，
  // 故默认 WASM；下面 run 仍包一层回退以防未来换 EP
  const modelUrl = opts.modelUrl || MODEL_URL;
  const eps = opts.eps || [['wasm']];
  let { ort, session, ep } = await getSession(modelUrl, { onProgress: opts.onProgress, eps });
  opts.onStage && opts.onStage(`检测角色…(${ep})`);

  // letterbox：保宽高比缩放，左上对齐，余下补 YOLO 惯例的灰 114
  const W = imageData.width, H = imageData.height;
  const s = size / Math.max(W, H);
  const vw = Math.round(W * s), vh = Math.round(H * s);
  const c = createCanvas(size, size);
  const ctx = c.getContext('2d');
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, size, size);
  const tmp = createCanvas(W, H);
  tmp.getContext('2d').putImageData(imageData, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, vw, vh);
  const d = ctx.getImageData(0, 0, size, size).data;
  const plane = size * size;
  const chw = new Float32Array(3 * plane);
  for (let p = 0; p < plane; p++) {
    chw[p] = d[p * 4] / 255;
    chw[plane + p] = d[p * 4 + 1] / 255;
    chw[2 * plane + p] = d[p * 4 + 2] / 255;
  }
  const tensor = new ort.Tensor('float32', chw, [1, 3, size, size]);
  let results;
  try {
  try {
    results = await session.run({ [session.inputNames[0]]: tensor });
  } catch (e) {
    if (ep === 'wasm') throw e;
    // WebGPU 运行期失败：换 WASM 重建会话重跑
    await evictSession(modelUrl);
    ({ ort, session, ep } = await getSession(modelUrl, { eps: [['wasm']] }));
    opts.onStage && opts.onStage(`检测角色…(${ep})`);
    results = await session.run({ [session.inputNames[0]]: tensor });
  }
  const out = results[session.outputNames[0]];

  // 输出 [1, 4+类数, N]（个别导出是 [1, N, 4+类数]，按维度大小判别）
  const dims = out.dims, data = out.data;
  let C = dims[1], N = dims[2], transposed = false;
  if (dims[1] > dims[2]) { C = dims[2]; N = dims[1]; transposed = true; }
  const get = transposed ? (ci, ni) => data[ni * C + ci] : (ci, ni) => data[ci * N + ni];

  const cand = [];
  for (let i = 0; i < N; i++) {
    let sc = 0;
    for (let ci = 4; ci < C; ci++) { const v = get(ci, i); if (v > sc) sc = v; }
    if (sc < conf) continue;
    const cx = get(0, i), cy = get(1, i), bw = get(2, i), bh = get(3, i);
    cand.push({
      x1: (cx - bw / 2) / s, y1: (cy - bh / 2) / s,
      x2: (cx + bw / 2) / s, y2: (cy + bh / 2) / s, score: sc,
    });
  }
  cand.sort((a, b) => b.score - a.score);

  // NMS
  const kept = [];
  for (const b of cand) {
    let ok = true;
    for (const k of kept) {
      const ix = Math.max(0, Math.min(b.x2, k.x2) - Math.max(b.x1, k.x1));
      const iy = Math.max(0, Math.min(b.y2, k.y2) - Math.max(b.y1, k.y1));
      const inter = ix * iy;
      const uni = (b.x2 - b.x1) * (b.y2 - b.y1) + (k.x2 - k.x1) * (k.y2 - k.y1) - inter;
      if (uni > 0 && inter / uni > iouThr) { ok = false; break; }
    }
    if (ok) { kept.push(b); if (kept.length >= maxDet) break; }
  }

  return kept.map((b) => {
    const x = Math.max(0, Math.round(b.x1)), y = Math.max(0, Math.round(b.y1));
    return {
      x, y,
      w: Math.min(W, Math.round(b.x2)) - x,
      h: Math.min(H, Math.round(b.y2)) - y,
      score: b.score,
    };
  }).filter((b) => b.w >= 6 && b.h >= 6);
  } finally {
    tensor.dispose?.();
    if (results) for (const value of Object.values(results)) value?.dispose?.();
  }
}

export { detectPersons };
