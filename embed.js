// embed.js — 场景嵌入（DINOv3 ViT-S/16 int8，21MB）：动画截图 ↔ 实景照片相似度匹配。
// 每张图编码成 384 维单位向量，余弦相似度=点积；换一张截图重排序只需重算截图向量。
// 全部在访客浏览器里跑，服务器零算力成本。按需懒加载。
import { getSession, MODEL_BASE } from './ort-env.js';
import { createCanvas } from './canvas-util.js';

const MODEL_URL = `${MODEL_BASE}/models/scene-embed-int8.onnx`;
const SIZE = 224;
// timm vit_small_patch16_dinov3 预处理：/255 后按 ImageNet 均值方差归一化
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// 整图压扁到 224×224。不裁剪、不留边：保留完整构图信息，
// Python 端实测（tools/build-scene-embed.py 产物）压扁比中心裁剪的同场景相似度更高。
function preprocess(source) {
  const c = createCanvas(SIZE, SIZE);
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  if (source instanceof ImageData) {
    const tmp = createCanvas(source.width, source.height);
    tmp.getContext('2d').putImageData(source, 0, 0);
    ctx.drawImage(tmp, 0, 0, SIZE, SIZE);
  } else {
    ctx.drawImage(source, 0, 0, SIZE, SIZE);
  }
  const d = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const plane = SIZE * SIZE;
  const chw = new Float32Array(3 * plane);
  for (let p = 0; p < plane; p++) {
    const i = p * 4;
    chw[p] = (d[i] / 255 - MEAN[0]) / STD[0];
    chw[plane + p] = (d[i + 1] / 255 - MEAN[1]) / STD[1];
    chw[2 * plane + p] = (d[i + 2] / 255 - MEAN[2]) / STD[2];
  }
  return chw;
}

// 编码一张图（ImageData / ImageBitmap / canvas / img），返回 384 维单位向量。
// int8 量化算子（DynamicQuantizeLinear/MatMulInteger）WebGPU EP 不支持，统一 WASM。
async function embedImage(source, opts = {}) {
  const { ort, session } = await getSession(MODEL_URL, {
    onProgress: opts.onProgress, eps: [['wasm']],
  });
  const tensor = new ort.Tensor('float32', preprocess(source), [1, 3, SIZE, SIZE]);
  let results;
  try {
    results = await session.run({ [session.inputNames[0]]: tensor });
    const out = results[session.outputNames[0]].data;
    const emb = Float32Array.from(out);
    let norm = 0;
    for (let i = 0; i < emb.length; i++) norm += emb[i] * emb[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < emb.length; i++) emb[i] /= norm;
    return emb;
  } finally {
    tensor.dispose?.();
    if (results) for (const value of Object.values(results)) value?.dispose?.();
  }
}

// 单位向量的余弦相似度
function cosineSimilarity(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export { embedImage, cosineSimilarity, MODEL_URL as SCENE_EMBED_MODEL_URL };
