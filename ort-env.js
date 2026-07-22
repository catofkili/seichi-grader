// ort-env.js — onnxruntime-web 加载与模型会话缓存，供 ai-segment.js / detect.js 共用。
// 全部在访客浏览器里跑，服务器零算力成本。
import { profile as DEVICE } from './platform.js';

const ORT_VER = '1.19.2';
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/ort.webgpu.mjs`;
const WASM_PATHS = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;

// 模型从 GitHub Pages 加载，页面本体留在 Cloudflare。实测大陆直连（真实用户无梯子）
// GitHub 明显快于 Cloudflare（CF 常卡死/龟速），而页面留 CF 才有 COOP/COEP →
// crossOriginIsolated → ONNX 多线程。GitHub 对模型文件发 ACAO:*，故 COEP 页面可跨域取。
// 换托管只改这一行；设为 '.' 即回落到同源（页面所在站点）。
const MODEL_BASE = 'https://anitabi.github.io/seichi-grader';

let ortPromise = null;
const sessionCache = new Map(); // modelUrl -> Promise<{ort, session, ep}>

async function loadOrt() {
  if (!ortPromise) {
    ortPromise = import(ORT_URL).then((m) => {
      const ort = m.default && m.default.InferenceSession ? m.default : m;
      ort.env.wasm.wasmPaths = WASM_PATHS;
      // cross-origin isolation（COOP/COEP 头，见启动脚本）开启时才允许多线程，
      // 否则 SharedArrayBuffer 不可用，回退单线程
      ort.env.wasm.numThreads = self.crossOriginIsolated ? DEVICE.wasmThreads : 1;
      return ort;
    });
  }
  return ortPromise;
}

// 分块模型清单：GitHub Pages 不解析 Git LFS，本机代理又拦 >约 40MB 的推送，
// 且 Cloudflare Pages 单文件上限 25MiB，故两个 84MB 的 ISNet 拆成 22MiB 分块
// 入库（models/xxx.onnx.part00..03），浏览器按序取回再拼成完整 ArrayBuffer。值 = 分块数。
const CHUNKED_MODELS = {
  'isnet-anime-fp16.onnx': 4,
  'isnet-anime-512-fp16.onnx': 4,
};

// 带进度的单文件下载，返回 { chunks:Uint8Array[], received }
async function fetchStream(url, baseReceived, grandTotal, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error('模型下载失败 ' + resp.status);
  if (!resp.body) { const b = new Uint8Array(await resp.arrayBuffer()); onProgress && onProgress(baseReceived + b.length, grandTotal); return { chunks: [b], received: b.length }; }
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress && onProgress(baseReceived + received, grandTotal);
  }
  return { chunks, received };
}

// 带进度的下载，返回 ArrayBuffer。分块模型自动按 .partNN 取回并拼接。
async function fetchWithProgress(url, onProgress) {
  const base = url.split('?')[0];
  const name = base.slice(base.lastIndexOf('/') + 1);
  const parts = CHUNKED_MODELS[name];

  if (parts) {
    // 分块：先各自 HEAD 拿总大小做进度分母（失败则用估算），再顺序流式取回拼接
    const urls = Array.from({ length: parts }, (_, i) => `${base}.part${String(i).padStart(2, '0')}`);
    let grandTotal = 0;
    await Promise.all(urls.map(async (u) => {
      try { const h = await fetch(u, { method: 'HEAD' }); grandTotal += +h.headers.get('content-length') || 0; } catch { /* 忽略 */ }
    }));
    if (!grandTotal) grandTotal = parts * 22 * 1024 * 1024; // 估算兜底
    const all = [];
    let received = 0;
    for (const u of urls) {
      const r = await fetchStream(u, received, grandTotal, onProgress);
      all.push(...r.chunks); received += r.received;
    }
    const buf = new Uint8Array(received);
    let pos = 0;
    for (const c of all) { buf.set(c, pos); pos += c.length; }
    return buf.buffer;
  }

  const resp = await fetch(url);
  if (!resp.ok) throw new Error('模型下载失败 ' + resp.status);
  const total = +resp.headers.get('content-length') || 0;
  if (!resp.body || !total) return await resp.arrayBuffer();
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress && onProgress(received, total);
  }
  const buf = new Uint8Array(received);
  let pos = 0;
  for (const c of chunks) { buf.set(c, pos); pos += c.length; }
  return buf.buffer;
}

// 取（或创建）模型会话。eps 是执行提供器候选列表，依次尝试：
// 如 [['webgpu'], ['wasm']]。失败会清缓存，下次点击可重试。
async function getSession(modelUrl, opts = {}) {
  if (!sessionCache.has(modelUrl)) {
    const p = (async () => {
      const ort = await loadOrt();
      const buf = await fetchWithProgress(modelUrl, opts.onProgress);
      const tries = opts.eps || [['wasm']];
      let lastErr = null;
      for (const eps of tries) {
        try {
          const session = await ort.InferenceSession.create(buf, {
            executionProviders: eps,
            graphOptimizationLevel: 'all',
          });
          return { ort, session, ep: eps[0] };
        } catch (e) { lastErr = e; }
      }
      throw lastErr;
    })();
    p.catch(() => sessionCache.delete(modelUrl));
    sessionCache.set(modelUrl, p);
  }
  return sessionCache.get(modelUrl);
}

// 驱逐某模型的会话缓存（如 WebGPU 运行期失败后，用 WASM 重建）
async function evictSession(modelUrl) {
  const pending = sessionCache.get(modelUrl);
  sessionCache.delete(modelUrl);
  if (!pending) return;
  try {
    const { session } = await pending;
    await session.release?.();
  } catch { /* 创建失败的会话无需释放 */ }
}

async function releaseAllSessions() {
  const entries = [...sessionCache.values()];
  sessionCache.clear();
  await Promise.allSettled(entries.map(async (pending) => {
    const { session } = await pending;
    await session.release?.();
  }));
}

export { loadOrt, fetchWithProgress, getSession, evictSession, releaseAllSessions, MODEL_BASE };
