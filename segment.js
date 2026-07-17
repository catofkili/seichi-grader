// segment.js — 经典启发式前景(角色)提取，纯前端无依赖。
// 思路：动画角色 = 锐利(背景浅景深虚化) + 硬描边 + 平涂低方差 + 高饱和 + 居中。
// 把这些线索加权成前景概率图 -> Otsu 阈值 -> 最大连通域 -> 填洞 -> 形态学 -> 羽化。

// 灰度
function toGray(d, w, h) {
  const g = new Float32Array(w * h);
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    g[p] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  }
  return g;
}

// HSV 饱和度 0..1
function satMap(d, w, h) {
  const s = new Float32Array(w * h);
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    const r = d[i], g = d[i + 1], b = d[i + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    s[p] = mx === 0 ? 0 : (mx - mn) / mx;
  }
  return s;
}

// 积分图，便于 O(1) 求窗口和
function integral(src, w, h) {
  const I = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    for (let x = 0; x < w; x++) {
      rowSum += src[y * w + x];
      I[(y + 1) * (w + 1) + (x + 1)] = I[y * (w + 1) + (x + 1)] + rowSum;
    }
  }
  return I;
}
function boxSum(I, w, x0, y0, x1, y1) {
  // 闭区间 [x0,x1] x [y0,y1]
  const W = w + 1;
  return I[(y1 + 1) * W + (x1 + 1)] - I[y0 * W + (x1 + 1)] - I[(y1 + 1) * W + x0] + I[y0 * W + x0];
}

// 拉普拉斯绝对值 -> 局部锐度密度(景深虚化背景=低)
function sharpnessMap(gray, w, h, radius = 3) {
  const lap = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const p = y * w + x;
      lap[p] = Math.abs(4 * gray[p] - gray[p - 1] - gray[p + 1] - gray[p - w] - gray[p + w]);
    }
  }
  const I = integral(lap, w, h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - radius), y1 = Math.min(h - 1, y + radius);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - radius), x1 = Math.min(w - 1, x + radius);
      const area = (x1 - x0 + 1) * (y1 - y0 + 1);
      out[y * w + x] = boxSum(I, w, x0, y0, x1, y1) / area;
    }
  }
  return out;
}

// 局部颜色方差(平涂角色=低；背景渐变/纹理=高) -> 取反作为前景线索
function flatnessMap(gray, w, h, radius = 3) {
  const sq = new Float32Array(w * h);
  for (let i = 0; i < gray.length; i++) sq[i] = gray[i] * gray[i];
  const I1 = integral(gray, w, h), I2 = integral(sq, w, h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - radius), y1 = Math.min(h - 1, y + radius);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - radius), x1 = Math.min(w - 1, x + radius);
      const area = (x1 - x0 + 1) * (y1 - y0 + 1);
      const m = boxSum(I1, w, x0, y0, x1, y1) / area;
      const v = boxSum(I2, w, x0, y0, x1, y1) / area - m * m;
      out[y * w + x] = Math.sqrt(Math.max(0, v)); // 标准差，后面取反
    }
  }
  return out;
}

// 用直方图取稳健最大值(99 分位)归一化到 0..1
function normalizeRobust(arr, invert = false) {
  const n = arr.length;
  let mx = 0, mn = Infinity;
  for (let i = 0; i < n; i++) { if (arr[i] > mx) mx = arr[i]; if (arr[i] < mn) mn = arr[i]; }
  const range = mx - mn || 1;
  const bins = new Float64Array(256);
  for (let i = 0; i < n; i++) bins[Math.min(255, ((arr[i] - mn) / range * 255) | 0)]++;
  let acc = 0, hi = 255;
  for (let b = 0; b < 256; b++) { acc += bins[b]; if (acc >= n * 0.99) { hi = b; break; } }
  const cap = mn + (hi / 255) * range || 1;
  const denom = (cap - mn) || 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = (arr[i] - mn) / denom;
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    out[i] = invert ? 1 - v : v;
  }
  return out;
}

// Otsu 阈值(输入 0..1)
function otsu(score) {
  const n = score.length;
  const hist = new Float64Array(256);
  for (let i = 0; i < n; i++) hist[Math.min(255, (score[i] * 255) | 0)]++;
  let sum = 0; for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, max = 0, thr = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (wB === 0) continue;
    const wF = n - wB; if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > max) { max = between; thr = t; }
  }
  return thr / 255;
}

// 连通域筛选(4 邻接)。mode:
//   'largest' — 只留最大域(经典算法用，前景假设唯一)
//   'multi'   — 保留所有足够大的域(≥最大域 15% 且 ≥画面 0.03%)，多角色同框不再只剩一人
function filterComponents(bin, w, h, mode) {
  const label = new Int32Array(w * h).fill(-1);
  const stack = new Int32Array(w * h);
  const sizes = [];
  for (let s = 0; s < w * h; s++) {
    if (bin[s] === 0 || label[s] !== -1) continue;
    const cur = sizes.length;
    let sp = 0; stack[sp++] = s; label[s] = cur; let size = 0;
    while (sp > 0) {
      const p = stack[--sp]; size++;
      const x = p % w, y = (p / w) | 0;
      if (x > 0 && bin[p - 1] && label[p - 1] === -1) { label[p - 1] = cur; stack[sp++] = p - 1; }
      if (x < w - 1 && bin[p + 1] && label[p + 1] === -1) { label[p + 1] = cur; stack[sp++] = p + 1; }
      if (y > 0 && bin[p - w] && label[p - w] === -1) { label[p - w] = cur; stack[sp++] = p - w; }
      if (y < h - 1 && bin[p + w] && label[p + w] === -1) { label[p + w] = cur; stack[sp++] = p + w; }
    }
    sizes.push(size);
  }
  const out = new Uint8Array(w * h);
  if (!sizes.length) return out;
  let maxSize = 0;
  for (const sz of sizes) if (sz > maxSize) maxSize = sz;
  const minKeep = mode === 'largest' ? maxSize : Math.max(maxSize * 0.15, w * h * 0.0003);
  for (let i = 0; i < w * h; i++) out[i] = label[i] >= 0 && sizes[label[i]] >= minKeep ? 1 : 0;
  return out;
}

// 最大连通域(4 邻接)
function largestComponent(bin, w, h) {
  return filterComponents(bin, w, h, 'largest');
}

// 从边界 flood 背景，未到达的非前景=内部洞 -> 填为前景
function fillHoles(mask, w, h) {
  const outside = new Uint8Array(w * h);
  const stack = [];
  const push = (p) => { if (mask[p] === 0 && outside[p] === 0) { outside[p] = 1; stack.push(p); } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (stack.length) {
    const p = stack.pop(); const x = p % w, y = (p / w) | 0;
    if (x > 0) push(p - 1); if (x < w - 1) push(p + 1);
    if (y > 0) push(p - w); if (y < h - 1) push(p + w);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = (mask[i] === 1 || outside[i] === 0) ? 1 : 0;
  return out;
}

// 3x3 二值膨胀/腐蚀，重复 r 次实现半径 r
function morph(mask, w, h, r, dilate) {
  let cur = mask;
  for (let it = 0; it < r; it++) {
    const next = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const p = y * w + x; let v = cur[p];
        const neigh = [
          x > 0 ? cur[p - 1] : v, x < w - 1 ? cur[p + 1] : v,
          y > 0 ? cur[p - w] : v, y < h - 1 ? cur[p + w] : v,
        ];
        if (dilate) next[p] = (v || neigh.some(n => n)) ? 1 : 0;
        else next[p] = (v && neigh.every(n => n)) ? 1 : 0;
      }
    }
    cur = next;
  }
  return cur;
}

// box blur 羽化得到 0..255 alpha
function feather(mask, w, h, radius = 2) {
  const f = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) f[i] = mask[i] ? 255 : 0;
  const I = integral(f, w, h);
  const out = new Uint8ClampedArray(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - radius), y1 = Math.min(h - 1, y + radius);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - radius), x1 = Math.min(w - 1, x + radius);
      const area = (x1 - x0 + 1) * (y1 - y0 + 1);
      out[y * w + x] = boxSum(I, w, x0, y0, x1, y1) / area;
    }
  }
  return out;
}

// 主入口：返回 { alpha(Uint8 w*h), width, height, bbox, coverage }
// weights 可调；centerBias 0..1 控制居中先验强度
function extractForeground(imageData, opts = {}) {
  const { width: w, height: h, data: d } = imageData;
  const W = {
    sharp: opts.sharp ?? 0.4,
    flat: opts.flat ?? 0.2,
    sat: opts.sat ?? 0.25,
    edge: opts.edge ?? 0.15,
  };
  const centerBias = opts.centerBias ?? 0.5;

  const gray = toGray(d, w, h);
  const sharp = normalizeRobust(sharpnessMap(gray, w, h, 3));
  const flat = normalizeRobust(flatnessMap(gray, w, h, 3), true); // 取反：平=高分
  const sat = satMap(d, w, h); // 已 0..1
  // 边缘密度复用锐度近似（拉普拉斯已含边缘），这里用 sharp 的高频部分即可，
  // 单独再算 sat 的边缘意义不大，edge 权重并到 sharp。
  const score = new Float32Array(w * h);
  const cx = 0.5, cy = 0.58, sigmaX = 0.42, sigmaY = 0.5;
  for (let y = 0; y < h; y++) {
    const ny = y / h;
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      let s = W.sharp * sharp[p] + W.flat * flat[p] + W.sat * sat[p] + W.edge * sharp[p];
      // 居中高斯先验
      const nx = x / w;
      const dx = (nx - cx) / sigmaX, dy = (ny - cy) / sigmaY;
      const prior = Math.exp(-0.5 * (dx * dx + dy * dy));
      s *= (1 - centerBias) + centerBias * prior;
      score[p] = s;
    }
  }
  const normScore = normalizeRobust(score);
  const thr = otsu(normScore);
  const bin = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) bin[i] = normScore[i] >= thr ? 1 : 0;

  let mask = largestComponent(bin, w, h);
  mask = morph(mask, w, h, 2, true);   // 闭运算：先膨胀
  mask = morph(mask, w, h, 2, false);  // 再腐蚀
  mask = fillHoles(mask, w, h);
  mask = largestComponent(mask, w, h); // 再取一次最大域去碎块

  const alpha = feather(mask, w, h, 2);

  // bbox + 覆盖率
  let minX = w, minY = h, maxX = 0, maxY = 0, cnt = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (alpha[y * w + x] > 16) {
      cnt++;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  const bbox = cnt ? { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 } : null;
  return { alpha, width: w, height: h, bbox, coverage: cnt / (w * h) };
}

// 用 alpha 生成裁剪后的角色 canvas（RGBA，仅 bbox 区域）
function cutoutCanvas(imageData, result) {
  const { width: w } = imageData;
  const { alpha, bbox } = result;
  if (!bbox) return null;
  const c = document.createElement('canvas');
  c.width = bbox.w; c.height = bbox.h;
  const ctx = c.getContext('2d');
  const out = ctx.createImageData(bbox.w, bbox.h);
  const src = imageData.data;
  for (let y = 0; y < bbox.h; y++) {
    for (let x = 0; x < bbox.w; x++) {
      const sp = ((bbox.y + y) * w + (bbox.x + x));
      const dp = (y * bbox.w + x) * 4;
      out.data[dp] = src[sp * 4];
      out.data[dp + 1] = src[sp * 4 + 1];
      out.data[dp + 2] = src[sp * 4 + 2];
      out.data[dp + 3] = alpha[sp];
    }
  }
  ctx.putImageData(out, 0, 0);
  return c;
}

// 清理任意 alpha（如 AI 输出）：阈值 -> 连通域筛选 -> 闭运算 -> 填洞 -> 羽化
// 能去掉与主体断开的噪点杂边。
// keepLargest: true=只留最大域(旧行为) / false=全保留 / 缺省=保留多个足够大的域(多角色安全)
function cleanupAlpha(alpha, w, h, opts = {}) {
  const thr = opts.thr ?? 110;
  const featherR = opts.featherR ?? 2;
  const erode = opts.erode ?? 0; // 收边：向内收缩像素数，去背景晕边
  const bin = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) bin[i] = alpha[i] >= thr ? 1 : 0;
  let mask = opts.keepLargest === false ? bin
    : filterComponents(bin, w, h, opts.keepLargest === true ? 'largest' : 'multi');
  mask = morph(mask, w, h, 1, true);
  mask = morph(mask, w, h, 1, false);
  mask = fillHoles(mask, w, h);
  if (erode > 0) mask = morph(mask, w, h, erode, false);
  return feather(mask, w, h, featherR);
}

// 从 alpha 计算 bbox（alpha>thr 的外接框）
function alphaBBox(alpha, w, h, thr = 16) {
  let minX = w, minY = h, maxX = 0, maxY = 0, cnt = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (alpha[y * w + x] > thr) {
      cnt++;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  return cnt ? { bbox: { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 }, coverage: cnt / (w * h) } : { bbox: null, coverage: 0 };
}

// 近闭合轮廓补洞。普通 fillHoles 只能填「完全被围死」的洞；轮廓上有小缺口时，
// 脸/手/服装内部会经缺口连到外部背景而漏填。策略：
//   1) 强闭运算（半径 seal）临时封住缺口；
//   2) 边界 flood 找出「封缝后仍被围住」的候选洞；
//   3) 逐洞颜色仲裁——动画是平涂：真镂空透出的是外部背景色，
//      假洞（脸、手、衣服内部）颜色接近洞周前景，只填后者；
//   4) 面积上限只防病态情况（洞占裁剪区一半以上不填）——「只抠到外轮廓、
//      内部整个是洞」正是要治的主症，所以不能按前景占比设限，把关交给颜色仲裁。
// 直接原地把接受的洞在 alpha 上置 255；返回补入的像素数。
function fillNearlyClosedHoles(alpha, w, h, imageData, opts = {}) {
  const thr = opts.thr ?? 128;
  const seal = opts.seal ?? 3;
  const maxFrac = opts.maxHoleFrac ?? 0.5;
  const bin = new Uint8Array(w * h);
  let fgArea = 0;
  for (let i = 0; i < w * h; i++) { if (alpha[i] >= thr) { bin[i] = 1; fgArea++; } }
  if (!fgArea) return 0;
  let closed = morph(bin, w, h, seal, true);
  closed = morph(closed, w, h, seal, false);
  // 封缝后仍连着边界的背景 = 确定的外部
  const outside = new Uint8Array(w * h);
  const stack = [];
  const pushOut = (p) => { if (!closed[p] && !outside[p]) { outside[p] = 1; stack.push(p); } };
  for (let x = 0; x < w; x++) { pushOut(x); pushOut((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { pushOut(y * w); pushOut(y * w + w - 1); }
  while (stack.length) {
    const p = stack.pop(); const x = p % w, y = (p / w) | 0;
    if (x > 0) pushOut(p - 1); if (x < w - 1) pushOut(p + 1);
    if (y > 0) pushOut(p - w); if (y < h - 1) pushOut(p + w);
  }
  const d = imageData.data;
  // 外部背景平均色（原本就不是前景、且确定连外的像素）
  let br = 0, bgc = 0, bb = 0, bn = 0;
  for (let i = 0; i < w * h; i++) {
    if (outside[i] && !bin[i]) { const p = i * 4; br += d[p]; bgc += d[p + 1]; bb += d[p + 2]; bn++; }
  }
  if (!bn) return 0;
  br /= bn; bgc /= bn; bb /= bn;
  const seen = new Uint8Array(w * h);
  let filled = 0;
  for (let s = 0; s < w * h; s++) {
    if (bin[s] || closed[s] || outside[s] || seen[s]) continue;
    // 收集一个洞（封缝后被围住的背景连通域）
    const hole = [];
    const st = [s]; seen[s] = 1;
    while (st.length) {
      const p = st.pop(); hole.push(p);
      const x = p % w, y = (p / w) | 0;
      for (const n of [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, y > 0 ? p - w : -1, y < h - 1 ? p + w : -1]) {
        if (n >= 0 && !bin[n] && !closed[n] && !outside[n] && !seen[n]) { seen[n] = 1; st.push(n); }
      }
    }
    if (hole.length > w * h * maxFrac) continue;
    // 洞平均色
    let hr = 0, hg = 0, hb = 0;
    for (const p of hole) { const q = p * 4; hr += d[q]; hg += d[q + 1]; hb += d[q + 2]; }
    hr /= hole.length; hg /= hole.length; hb /= hole.length;
    // 从洞向外做深度受限 BFS：撞到真前景就采样「洞周前景色」，
    // 途经的封缝带像素记为 extra（接受时和洞一起填，顺便堵上缺口通道）
    let rr = 0, rg = 0, rb = 0, rn = 0;
    const extra = [];
    let ring = hole.slice();
    const walked = new Uint8Array(w * h);
    for (const p of hole) walked[p] = 1;
    for (let depth = 0; depth < seal + 2 && ring.length; depth++) {
      const next = [];
      for (const p of ring) {
        const x = p % w, y = (p / w) | 0;
        for (const n of [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, y > 0 ? p - w : -1, y < h - 1 ? p + w : -1]) {
          if (n < 0 || walked[n]) continue;
          walked[n] = 1;
          if (bin[n]) { const q = n * 4; rr += d[q]; rg += d[q + 1]; rb += d[q + 2]; rn++; }
          else { extra.push(n); next.push(n); }
        }
      }
      ring = next;
    }
    if (!rn) continue;
    rr /= rn; rg /= rn; rb /= rn;
    const dRing = (hr - rr) ** 2 + (hg - rg) ** 2 + (hb - rb) ** 2;
    const dBg = (hr - br) ** 2 + (hg - bgc) ** 2 + (hb - bb) ** 2;
    if (dRing < dBg) {
      for (const p of hole) { if (alpha[p] < 255) { alpha[p] = 255; filled++; } }
      for (const p of extra) { if (alpha[p] < 255) { alpha[p] = 255; filled++; } }
    }
  }
  return filled;
}

export { extractForeground, cutoutCanvas, cleanupAlpha, alphaBBox, fillNearlyClosedHoles };
