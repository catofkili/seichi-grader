// ai-segment.js — 浏览器内 AI 抠图（onnxruntime-web + 量化 ISNet）。
// 全部在访客浏览器里跑，服务器零算力成本。按需懒加载。
// 两种模式：
//   extractForegroundAI  — 整图直抠（显著性），适合角色占画面主体的特写
//   extractCharactersAI  — 检测→裁剪→抠→合并，多角色/角色偏小时用这个
import { cleanupAlpha } from './segment.js';
import { getSession, releaseAllSessions, MODEL_BASE } from './ort-env.js';
import { detectPersons } from './detect.js';
import { samMaskForBox } from './sam-segment.js';
import { createCanvas } from './canvas-util.js';

const DEFAULT_SIZE = 1024;
// isnet-anime 预处理：/255 后减均值（RGB），std=1，保宽高比 letterbox 补边到 1024
const MEAN = [0.485, 0.456, 0.406];
// 推理输出（sigmoid 0..1）最大值低于此值视为"没找到前景"——
// 直接 min-max 归一化会把空场景的噪声放大成假前景
const MIN_RAW_MAX = 0.05;

// letterbox 预处理：保宽高比缩放到 longest=1024，左上对齐，其余补 0。
// 返回 { chw, validW, validH }（valid 为有效区像素尺寸，补边区张量值=0）
function preprocess(imageData, size = DEFAULT_SIZE) {
  const W = imageData.width, H = imageData.height;
  const scale = size / Math.max(W, H);
  const validW = Math.round(W * scale), validH = Math.round(H * scale);
  const c = createCanvas(size, size);
  const ctx = c.getContext('2d');
  const tmp = createCanvas(W, H);
  tmp.getContext('2d').putImageData(imageData, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, validW, validH); // 左上对齐
  const d = ctx.getImageData(0, 0, size, size).data;
  const chw = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const p = y * size + x;
      if (x < validW && y < validH) {
        const i = p * 4;
        chw[p] = d[i] / 255 - MEAN[0];
        chw[plane + p] = d[i + 1] / 255 - MEAN[1];
        chw[2 * plane + p] = d[i + 2] / 255 - MEAN[2];
      } // 否则保持 0（补边）
    }
  }
  return { chw, validW, validH };
}

// 取 mask 有效区（validW×validH），缩放回原尺寸，得到 alpha(Uint8 w*h)
function postprocess(mask01, validW, validH, w, h, size = DEFAULT_SIZE) {
  const mc = createCanvas(size, size);
  const mctx = mc.getContext('2d');
  const img = mctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const v = Math.max(0, Math.min(255, Math.round(mask01[i] * 255)));
    img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
  }
  mctx.putImageData(img, 0, 0);
  // 裁出有效区再缩放回原尺寸
  const oc = createCanvas(w, h);
  const octx = oc.getContext('2d');
  octx.imageSmoothingEnabled = true;
  octx.drawImage(mc, 0, 0, validW, validH, 0, 0, w, h);
  const od = octx.getImageData(0, 0, w, h).data;
  const alpha = new Uint8ClampedArray(w * h);
  for (let i = 0, p = 0; i < od.length; i += 4, p++) alpha[p] = od[i];
  return alpha;
}

// 从 ImageData 裁一块子区域出来
function cropImageData(imageData, rect) {
  const c = createCanvas(imageData.width, imageData.height);
  const ctx = c.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
  return ctx.getImageData(rect.x, rect.y, rect.w, rect.h);
}

// 整图直抠：返回 { alpha, width, height, bbox, coverage, empty }
async function extractForegroundAI(imageData, opts = {}) {
  const modelUrl = opts.modelUrl || `${MODEL_BASE}/models/isnet-anime-w8.onnx`;
  const size = opts.inputSize || DEFAULT_SIZE;
  // ISNet 含 ceil_mode 的 MaxPool，WebGPU EP 暂不支持，统一用 WASM
  const { ort, session, ep } = await getSession(modelUrl, {
    onProgress: opts.onProgress, eps: [['wasm']],
  });
  opts.onStage && opts.onStage(`推理中…(${ep})`);

  const { chw, validW, validH } = preprocess(imageData, size);
  const tensor = new ort.Tensor('float32', chw, [1, 3, size, size]);
  const feeds = { [session.inputNames[0]]: tensor };
  let results;
  try {
  results = await session.run(feeds);
  const out = results[session.outputNames[0]];
  const data = out.data; // Float32Array, [1,1,SIZE,SIZE]

  // 仅在有效区统计（避免补边 0 干扰）
  let mn = Infinity, mx = -Infinity;
  for (let y = 0; y < validH; y++) for (let x = 0; x < validW; x++) {
    const v = data[y * size + x];
    if (v < mn) mn = v; if (v > mx) mx = v;
  }

  const w = imageData.width, h = imageData.height;
  const debug = { rawMin: mn, rawMax: mx, validW, validH, ep };

  // 空场景守卫：模型输出整体极低说明没找到前景，直接返回空
  if (!(mx >= (opts.minRawMax ?? MIN_RAW_MAX))) {
    return { alpha: new Uint8ClampedArray(w * h), width: w, height: h, bbox: null, coverage: 0, empty: true, debug };
  }

  const range = (mx - mn) || 1;
  const mask01 = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) mask01[i] = Math.max(0, Math.min(1, (data[i] - mn) / range));

  // 返回原始 alpha（不清理）；清理/阈值/收边交给上层
  let alpha = postprocess(mask01, validW, validH, w, h, size);
  if (opts.cleanup === true) alpha = cleanupAlpha(alpha, w, h, { thr: 110, featherR: 2 });

  // bbox + coverage
  let minX = w, minY = h, maxX = 0, maxY = 0, cnt = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (alpha[y * w + x] > 24) {
      cnt++;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  const bbox = cnt ? { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 } : null;
  return { alpha, width: w, height: h, bbox, coverage: cnt / (w * h), empty: false, debug };
  } finally {
    tensor.dispose?.();
    if (results) for (const value of Object.values(results)) value?.dispose?.();
  }
}

// 两级流水线：检测人物框 → 每框外扩裁剪 → ISNet 抠裁剪区。
// 返回 { chars: [{ box, rect, score, alpha, empty }], width, height }
//   box  = 检测框（原图坐标）
//   rect = 外扩后的裁剪区（alpha 与 rect 同尺寸）
//   empty= ISNet 对该框没有响应（角色太小/太糊），mask 为空
// opts.hires: 远景小人模式（1536 检测 + 低置信度阈值）
async function extractCharactersAI(imageData, opts = {}) {
  const hires = !!opts.hires;
  const boxes = await detectPersons(imageData, {
    size: hires ? 1536 : 1024,
    conf: hires ? 0.12 : 0.25,
    onProgress: opts.onProgress,
    onStage: opts.onStage,
  });

  const W = imageData.width, H = imageData.height;
  const chars = [];
  // 第一阶段：所有框只跑 ISNet。不要在循环中交错加载 SAM，否则检测器、
  // ISNet、SAM encoder/decoder 会同时常驻 WASM 堆，iOS 峰值可达数 GB。
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    opts.onStage && opts.onStage(`抠角色 ${i + 1}/${boxes.length}…`);
    // 外扩：横向多留些（手臂/围巾），纵向少些
    const padX = Math.round(b.w * 0.2), padY = Math.round(b.h * 0.12);
    const x = Math.max(0, b.x - padX), y = Math.max(0, b.y - padY);
    const rect = {
      x, y,
      w: Math.min(W, b.x + b.w + padX) - x,
      h: Math.min(H, b.y + b.h + padY) - y,
    };
    const crop = cropImageData(imageData, rect);
    const res = await extractForegroundAI(crop, {
      onProgress: opts.onProgress, modelUrl: opts.isnetModelUrl, inputSize: opts.isnetSize,
    });
    chars.push({ box: b, rect, score: b.score, alpha: res.alpha, empty: res.empty, via: 'isnet' });
  }

  // 第二阶段：释放检测器/ISNet，再集中处理所有需要 SAM 的框。
  const fallback = chars.map((char, i) => char.empty ? i : -1).filter((i) => i >= 0);
  if (fallback.length && opts.samFallback !== false) {
    opts.onStage && opts.onStage('释放检测模型，准备 SAM 小角色兜底…');
    await releaseAllSessions();
    for (let n = 0; n < fallback.length; n++) {
      const i = fallback[n], char = chars[i], b = char.box;
      opts.onStage && opts.onStage(`SAM 兜底 ${n + 1}/${fallback.length}…`);
      try {
        const sPadX = Math.round(b.w * 0.35), sPadY = Math.round(b.h * 0.25);
        const sx = Math.max(0, b.x - sPadX), sy = Math.max(0, b.y - sPadY);
        const sRect = {
          x: sx, y: sy,
          w: Math.min(W, b.x + b.w + sPadX) - sx,
          h: Math.min(H, b.y + b.h + sPadY) - sy,
        };
        const sCrop = cropImageData(imageData, sRect);
        const inner = { x: b.x - sx, y: b.y - sy, w: b.w, h: b.h };
        const samAlpha = await samMaskForBox(sCrop, inner, { onProgress: opts.onProgress });
        if (samAlpha) Object.assign(char, { alpha: samAlpha, rect: sRect, empty: false, via: 'sam' });
      } catch (e) {
        console.warn('SAM 兜底失败，保持"太小未抠出"', e);
      }
    }
  }
  return { chars, width: W, height: H };
}

// 手动圈选（LR 式）：在用户指定的范围内智能检测并抠取角色。
// 流程：圈选区裁剪放大 → 区域内人物检测（小区域等效超高分辨率，专治全图漏检）
//   → 每个检出者各走"ISNet → SAM 兜底"single-box 流水线
//   → 区域内一个都没检出时，退回 SAM 按笔迹质心直接分割（抠错对象的最后手段）。
// box = {x,y,w,h}（原图坐标）；opts.samPoints = 笔迹质心（原图坐标）。
// 返回 chars 数组（元素与 extractCharactersAI 的条目同构，manual: true）。
async function extractCharactersInRegion(imageData, box, opts = {}) {
  const W = imageData.width, H = imageData.height;
  // 区域外扩 15% 后裁剪，检测在裁剪图上跑：200px 的圈在 1024 检测分辨率下
  // 等效 5 倍放大，比全图远景模式(1536)的有效分辨率还高 3 倍以上
  const rPadX = Math.round(box.w * 0.15), rPadY = Math.round(box.h * 0.15);
  const rx = Math.max(0, box.x - rPadX), ry = Math.max(0, box.y - rPadY);
  const region = {
    x: rx, y: ry,
    w: Math.min(W, box.x + box.w + rPadX) - rx,
    h: Math.min(H, box.y + box.h + rPadY) - ry,
  };
  opts.onStage && opts.onStage('在圈选范围内检测角色…');
  const regionCrop = cropImageData(imageData, region);
  // 检测分辨率自适应：把放大倍数控制在 ~2.5×。过度放大（小圈直接怼到 1024）
  // 会把模糊小人拉出检测器的训练分布，反而漏检。
  const target = Math.max(region.w, region.h) * 2.5;
  const detSize = target <= 512 ? 512 : target <= 768 ? 768 : 1024;
  let boxes = [];
  try {
    boxes = await detectPersons(regionCrop, {
      size: detSize, conf: 0.12, maxDet: 8,
      onProgress: opts.onProgress,
    });
  } catch (e) {
    console.warn('区域检测失败，退回单框流程', e);
  }
  if (boxes.length) {
    const chars = [];
    for (let i = 0; i < boxes.length; i++) {
      opts.onStage && opts.onStage(`圈选区角色 ${i + 1}/${boxes.length}…`);
      // 裁剪区坐标 → 原图坐标
      const b = { x: boxes[i].x + region.x, y: boxes[i].y + region.y, w: boxes[i].w, h: boxes[i].h, score: boxes[i].score };
      const char = await extractCharacterInBox(imageData, b, { ...opts, samPoints: [] });
      char.score = b.score;
      chars.push(char);
    }
    return chars;
  }
  // 区域内没检出：整个圈当一个对象，SAM 按质心分割
  return [await extractCharacterInBox(imageData, box, opts)];
}

// 单框流水线："ISNet 先试 → SAM 兜底"，与自动路径同一套参数。
// 返回与 extractCharactersAI 的 chars[] 同构的单个条目（manual: true）。
async function extractCharacterInBox(imageData, box, opts = {}) {
  const W = imageData.width, H = imageData.height;
  const padX = Math.round(box.w * 0.2), padY = Math.round(box.h * 0.12);
  const x = Math.max(0, box.x - padX), y = Math.max(0, box.y - padY);
  const rect = {
    x, y,
    w: Math.min(W, box.x + box.w + padX) - x,
    h: Math.min(H, box.y + box.h + padY) - y,
  };
  opts.onStage && opts.onStage('圈选区域抠图中…');
  const crop = cropImageData(imageData, rect);
  const res = await extractForegroundAI(crop, {
    onProgress: opts.onProgress, modelUrl: opts.isnetModelUrl, inputSize: opts.isnetSize,
  });
  const char = { box, rect, score: 1, alpha: res.alpha, empty: res.empty, via: 'isnet', manual: true };

  // 质量闸：ISNet 名义上有输出但覆盖率过低（<4%）时视同失败——
  // 空场景守卫刚好被擦过时，min-max 归一化会把噪声放大成稀碎的假 mask
  if (!char.empty) {
    let cnt = 0;
    for (let i = 0; i < char.alpha.length; i++) if (char.alpha[i] > 127) cnt++;
    if (cnt / char.alpha.length < 0.04) char.empty = true;
  }

  if (char.empty && opts.samFallback !== false) {
    opts.onStage && opts.onStage('ISNet 无响应，SAM 按圈选位置分割…');
    try {
      const sPadX = Math.round(box.w * 0.35), sPadY = Math.round(box.h * 0.25);
      const sx = Math.max(0, box.x - sPadX), sy = Math.max(0, box.y - sPadY);
      const sRect = {
        x: sx, y: sy,
        w: Math.min(W, box.x + box.w + sPadX) - sx,
        h: Math.min(H, box.y + box.h + sPadY) - sy,
      };
      const sCrop = cropImageData(imageData, sRect);
      const inner = { x: box.x - sx, y: box.y - sy, w: box.w, h: box.h };
      const points = (opts.samPoints || []).map((p) => [p[0] - sx, p[1] - sy]);
      const samAlpha = await samMaskForBox(sCrop, inner, { onProgress: opts.onProgress, points });
      if (samAlpha) Object.assign(char, { alpha: samAlpha, rect: sRect, empty: false, via: 'sam' });
    } catch (e) {
      console.warn('圈选 SAM 分割失败', e);
    }
  }
  return char;
}

// 把各角色的裁剪区 alpha 合并回整图尺寸（取最大值）。
// included: Set<index>；缺省合并全部
function mergeCharacterAlphas(chars, width, height, included) {
  const alpha = new Uint8ClampedArray(width * height);
  chars.forEach((c, i) => {
    if (included && !included.has(i)) return;
    const { x, y, w, h } = c.rect;
    for (let yy = 0; yy < h; yy++) {
      const src = yy * w;
      const dst = (y + yy) * width + x;
      for (let xx = 0; xx < w; xx++) {
        const v = c.alpha[src + xx];
        if (v > alpha[dst + xx]) alpha[dst + xx] = v;
      }
    }
  });
  return alpha;
}

export { extractForegroundAI, extractCharactersAI, extractCharacterInBox, extractCharactersInRegion, mergeCharacterAlphas };
