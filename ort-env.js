// ort-env.js — onnxruntime-web 加载与模型会话缓存，供 ai-segment.js / detect.js 共用。
// 全部在访客浏览器里跑，服务器零算力成本。
import { profile as DEVICE } from './platform.js';

const ORT_VER = '1.19.2';
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/ort.webgpu.mjs`;
const WASM_PATHS = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;

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

// 带进度的下载，返回 ArrayBuffer
async function fetchWithProgress(url, onProgress) {
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

export { loadOrt, fetchWithProgress, getSession, evictSession, releaseAllSessions };
