// sw.js — 模型持久缓存。
// ONNX 模型与 onnxruntime-web 运行时采用 cache-first；它们仅在用户按需下载离线包
// 或实际运行 AI 时进入 Cache Storage。其余请求不拦截（开发时改代码即时生效）。
const CACHE = 'seichi-models-v2';
const APP_CACHE = 'seichi-app-v8'; // v8: GIF LZW 位宽切换修复
const APP_SHELL = [
  './', './index.html', './style.css', './app.js', './color.js', './segment.js',
  './ai-segment.js', './detect.js', './sam-segment.js', './ort-env.js', './platform.js', './canvas-util.js', './ai-worker.js', './embed.js',
  './camera/camera-session.js', './camera/overlay-renderer.js', './camera/capture-adapter.js', './camera/viewfinder.js',
  './manifest.webmanifest', './icon.svg', './icon-180.png',
];
const SHOULD_CACHE = (url) => /\/models\/.+\.onnx(\.part\d+)?($|\?)|cdn\.jsdelivr\.net\/npm\/onnxruntime-web/.test(url);

self.addEventListener('install', (e) => e.waitUntil((async () => {
  const cache = await caches.open(APP_CACHE);
  await Promise.allSettled(APP_SHELL.map((url) => cache.add(url)));
  await self.skipWaiting();
})()));
self.addEventListener('activate', (e) => e.waitUntil((async () => {
  const keep = new Set([CACHE, APP_CACHE]);
  await Promise.all((await caches.keys()).filter((name) => name.startsWith('seichi-') && !keep.has(name)).map((name) => caches.delete(name)));
  await self.clients.claim();
})()));

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  if (!SHOULD_CACHE(e.request.url)) {
    const url = new URL(e.request.url);
    if (url.origin !== self.location.origin) return;
    // 程序文件用 network-first：开发时立即看到改动，断网时仍能启动完整页面。
    // 注意：cache.put 必须放进 waitUntil 后台执行，不能 await 在 return resp 之前——
    // 否则图片类请求会在 clone 流的背压上死锁（fetch 能拿到头，<img> 永远等不到体）。
    // 缓存键去掉查询串（?v= ?t= 等），避免同一文件的无限变体撑爆 Cache Storage。
    e.respondWith((async () => {
      const cache = await caches.open(APP_CACHE);
      const key = new Request(url.origin + url.pathname);
      try {
        const resp = await fetch(e.request);
        if (resp.ok && resp.status === 200) e.waitUntil(cache.put(key, resp.clone()).catch(() => {}));
        return resp;
      } catch {
        return (await cache.match(key)) || (await cache.match('./index.html'));
      }
    })());
    return;
  }
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(e.request, { ignoreVary: true });
    if (hit) return hit;
    const resp = await fetch(e.request);
    // 只缓存完整 200 响应（Range/206 不能存）；同样后台写入，失败时透传
    if (resp.ok && resp.status === 200) e.waitUntil(cache.put(e.request, resp.clone()).catch(() => {}));
    return resp;
  })());
});
