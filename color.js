// color.js — 调色核心：LAB 迁移、CDF 影调、天空分区、MKL 色彩映射、Bloom 与 LUT。

// ---------- sRGB <-> Linear / CIELAB ----------
// 调色会处理约 110 万像素；把最昂贵的幂函数预烘成 LUT，连续值用线性插值。
const SRGB_TO_LINEAR = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  SRGB_TO_LINEAR[i] = c <= .04045 ? c / 12.92 : Math.pow((c + .055) / 1.055, 2.4);
}
const LINEAR_LUT_SIZE = 8192, LINEAR_TO_SRGB = new Float32Array(LINEAR_LUT_SIZE + 1);
for (let i = 0; i <= LINEAR_LUT_SIZE; i++) {
  const c = i / LINEAR_LUT_SIZE;
  LINEAR_TO_SRGB[i] = (c <= .0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - .055) * 255;
}
function srgbToLinear(c) {
  if (c >= 0 && c <= 255 && Number.isInteger(c)) return SRGB_TO_LINEAR[c];
  c /= 255;
  return c <= .04045 ? c / 12.92 : Math.pow((c + .055) / 1.055, 2.4);
}
function linearToSrgb(c) {
  if (c <= 0) return 0;
  if (c >= 1) return 255;
  const x = c * LINEAR_LUT_SIZE, lo = Math.floor(x), f = x - lo;
  return Math.round(LINEAR_TO_SRGB[lo] + (LINEAR_TO_SRGB[lo + 1] - LINEAR_TO_SRGB[lo]) * f);
}
const Xn = 0.95047, Yn = 1, Zn = 1.08883;
const EPS = 216 / 24389, KAPPA = 24389 / 27;
const LAB_F_MAX = 1.25, LAB_F_SIZE = 8192, LAB_F = new Float32Array(LAB_F_SIZE + 1);
for (let i = 0; i <= LAB_F_SIZE; i++) {
  const t = i / LAB_F_SIZE * LAB_F_MAX;
  LAB_F[i] = t > EPS ? Math.cbrt(t) : (KAPPA * t + 16) / 116;
}
const labF = (t) => {
  if (t <= 0) return (KAPPA * t + 16) / 116;
  if (t >= LAB_F_MAX) return Math.cbrt(t);
  const x = t / LAB_F_MAX * LAB_F_SIZE, lo = Math.floor(x), f = x - lo;
  return LAB_F[lo] + (LAB_F[lo + 1] - LAB_F[lo]) * f;
};
const labFinv = (t) => (t * t * t) > EPS ? t * t * t : (116 * t - 16) / KAPPA;

function rgb2lab(r, g, b) {
  const lr = srgbToLinear(r), lg = srgbToLinear(g), lb = srgbToLinear(b);
  const X = lr * .4124564 + lg * .3575761 + lb * .1804375;
  const Y = lr * .2126729 + lg * .7151522 + lb * .072175;
  const Z = lr * .0193339 + lg * .119192 + lb * .9503041;
  const fx = labF(X / Xn), fy = labF(Y / Yn), fz = labF(Z / Zn);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}
function labToLinearRgb(L, a, b) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const X = Xn * labFinv(fx), Y = Yn * labFinv(fy), Z = Zn * labFinv(fz);
  return [X * 3.2404542 + Y * -1.5371385 + Z * -.4985314, X * -.969266 + Y * 1.8760108 + Z * .041556, X * .0556434 + Y * -.2040259 + Z * 1.0572252];
}
function linearRgbToLab(lr, lg, lb) {
  const X = lr * .4124564 + lg * .3575761 + lb * .1804375;
  const Y = lr * .2126729 + lg * .7151522 + lb * .072175;
  const Z = lr * .0193339 + lg * .119192 + lb * .9503041;
  const fx = labF(X / Xn), fy = labF(Y / Yn), fz = labF(Z / Zn);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}
function lab2rgb(L, a, b) {
  const rgb = labToLinearRgb(L, a, b);
  return [linearToSrgb(rgb[0]), linearToSrgb(rgb[1]), linearToSrgb(rgb[2])];
}

// 色域收敛。此前的"全程软滚降"会把线性 0.16 以下的整个暗部抬灰
// （黑位 0.001→0.059），毁掉动画感的干净黑位，也破坏恒等性。
// 正确姿势：绝大部分界内值严格恒等——
//   x ≤ 0        → 0（负光直接截断，暗部保持纯净）
//   0 < x < 0.96 → x（恒等）
//   x ≥ 0.96     → 1 - 0.04·e^((0.96-x)/0.04)（高光软肩，C1 连续，渐近 1）
// 只有线性 >0.96（sRGB ≈250 以上）的极高光被柔化，肉眼不可察。
function softClip01(x) {
  const s = .96, knee = .04;
  if (x <= 0) return 0;
  if (x < s) return x;
  return 1 - knee * Math.exp((s - x) / knee);
}

// 颜色迁移可能把 Lab 推出 sRGB。把变换后的线性 RGB 做连续软滚降，再转回 Lab；
// 这样页面直接结果和 3D LUT 在色域边界都没有硬折线。
function compressLabToSrgb(L, a, b) {
  L = Math.max(0, Math.min(100, L));
  const raw = labToLinearRgb(L, a, b);
  return linearRgbToLab(softClip01(raw[0]), softClip01(raw[1]), softClip01(raw[2]));
}

// 把整张图片预转为交错 LAB（L,a,b,L,a,b...），供统计与滑杆重算复用。
function imageDataToLab(imageData) {
  const src = imageData.data, out = new Float32Array(imageData.width * imageData.height * 3);
  for (let p = 0, i = 0; i < src.length; p++, i += 4) {
    const lab = rgb2lab(src[i], src[i + 1], src[i + 2]), j = p * 3;
    out[j] = lab[0]; out[j + 1] = lab[1]; out[j + 2] = lab[2];
  }
  return out;
}

function eachSample(imageData, step, opts, fn) {
  const { width: w, height: h, data } = imageData;
  const weightMask = opts.weightMask;
  const ignoreBottom = opts.ignoreBottomRatio || 0;
  const maxY = Math.max(0, Math.floor(h * (1 - ignoreBottom)));
  for (let y = 0; y < maxY; y += step) for (let x = 0; x < w; x += step) {
    const p = y * w + x, i = p * 4;
    if (data[i + 3] < 8) continue;
    const weight = weightMask ? weightMask[p] : 1;
    if (weight > 1e-5) fn(i, x, y, weight, p);
  }
}

// opts: { weightMask, ignoreBottomRatio, cov }。旧调用 labStats(img, step) 仍有效。
function labStats(imageData, sampleStep = 1, opts = {}) {
  let n = 0, sumL = 0, sumA = 0, sumB = 0, sumL2 = 0, sumA2 = 0, sumB2 = 0, sumAB = 0;
  eachSample(imageData, sampleStep, opts, (i, x, y, weight, p) => {
    const d = imageData.data, j = p * 3;
    let L, a, b;
    if (opts.labData) { L = opts.labData[j]; a = opts.labData[j + 1]; b = opts.labData[j + 2]; }
    else [L, a, b] = rgb2lab(d[i], d[i + 1], d[i + 2]);
    n += weight; sumL += L * weight; sumA += a * weight; sumB += b * weight;
    sumL2 += L * L * weight; sumA2 += a * a * weight; sumB2 += b * b * weight; sumAB += a * b * weight;
  });
  n ||= 1;
  const meanL = sumL / n, meanA = sumA / n, meanB = sumB / n;
  const out = {
    meanL, meanA, meanB,
    stdL: Math.sqrt(Math.max(0, sumL2 / n - meanL * meanL)),
    stdA: Math.sqrt(Math.max(0, sumA2 / n - meanA * meanA)),
    stdB: Math.sqrt(Math.max(0, sumB2 / n - meanB * meanB)),
  };
  if (opts.cov) {
    out.covAA = Math.max(0, sumA2 / n - meanA * meanA);
    out.covAB = sumAB / n - meanA * meanB;
    out.covBB = Math.max(0, sumB2 / n - meanB * meanB);
  }
  return out;
}

// ---------- CDF 亮度匹配 ----------
function makeLumaCdfMap(srcImageData, tgtImageData, opts = {}) {
  const bins = 256, hs = new Float64Array(bins), ht = new Float64Array(bins);
  const add = (img, hist, sampleOpts, labData) => eachSample(img, opts.sampleStep || 2, sampleOpts, (i, x, y, weight, p) => {
    const d = img.data, L = labData ? labData[p * 3] : rgb2lab(d[i], d[i + 1], d[i + 2])[0];
    hist[Math.max(0, Math.min(255, Math.round(L / 100 * 255)))] += weight;
  });
  add(srcImageData, hs, { weightMask: opts.srcWeightMask }, opts.srcLabData);
  add(tgtImageData, ht, { weightMask: opts.tgtWeightMask, ignoreBottomRatio: opts.ignoreBottomRatio || 0 }, opts.tgtLabData);
  const cdf = (h) => {
    const out = new Float64Array(bins); let total = 0;
    for (const v of h) total += v;
    if (!total) return out;
    let s = 0; for (let i = 0; i < bins; i++) { s += h[i]; out[i] = s / total; }
    return out;
  };
  const cs = cdf(hs), ct = cdf(ht), raw = new Float64Array(bins);
  let sameHist = true;
  for (let i = 0; i < bins; i++) if (Math.abs(hs[i] - ht[i]) > 1e-8) { sameHist = false; break; }
  if (sameHist) for (let i = 0; i < bins; i++) raw[i] = i / 255 * 100;
  let j = 0;
  if (!sameHist) for (let i = 0; i < bins; i++) { while (j < 255 && ct[j] < cs[i]) j++; raw[i] = j / 255 * 100; }
  // 同图自迁移必须严格保持恒等；其余情况按规格做两遍半径 4 的盒式平滑。
  let map = sameHist ? Float64Array.from({ length: bins }, (_, i) => i / 255 * 100) : raw;
  if (!sameHist) for (let pass = 0; pass < 2; pass++) {
    const next = new Float64Array(bins), prefix = new Float64Array(bins + 1);
    for (let i = 0; i < bins; i++) prefix[i + 1] = prefix[i] + map[i];
    for (let i = 0; i < bins; i++) {
      const lo = Math.max(0, i - 4), hi = Math.min(255, i + 4);
      next[i] = (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1);
    }
    map = next;
  }
  const binWidth = 100 / 255;
  const minStep = .25 * binWidth, maxStep = 4 * binWidth;
  map[0] = Math.max(0, Math.min(100 - minStep * 255, map[0]));
  for (let i = 1; i < bins; i++) {
    // 给后续 bin 预留最小斜率空间，避免末端被截成一段平台。
    const upper = 100 - minStep * (255 - i);
    map[i] = Math.min(upper, Math.max(map[i - 1] + minStep, Math.min(map[i], map[i - 1] + maxStep)));
  }
  const table = new Float32Array(1024);
  const build = (strength = 1) => {
    const out = new Float32Array(table.length);
    for (let i = 0; i < out.length; i++) {
      const L = i / (out.length - 1) * 100, f = L / 100 * 255, lo = Math.floor(f), hi = Math.min(255, lo + 1);
      const mapped = map[lo] + (map[hi] - map[lo]) * (f - lo);
      out[i] = L + (mapped - L) * strength;
    }
    table.set(out);
    return (L) => {
      const x = Math.max(0, Math.min(out.length - 1, L / 100 * (out.length - 1)));
      const lo = Math.floor(x), hi = Math.min(out.length - 1, lo + 1);
      return out[lo] + (out[hi] - out[lo]) * (x - lo);
    };
  };
  return { table, build, mapL: build(opts.strength ?? 1) };
}

// ---------- 2×2 MKL 最优传输 ----------
function eigSym(a, b, d) {
  const tr = a + d, disc = Math.sqrt(Math.max(0, (a - d) * (a - d) + 4 * b * b));
  const l1 = (tr + disc) / 2, l2 = (tr - disc) / 2;
  let vx = b, vy = l1 - a;
  if (Math.abs(vx) + Math.abs(vy) < 1e-9) { vx = 1; vy = 0; }
  const n = Math.hypot(vx, vy); vx /= n; vy /= n;
  return { l1, l2, vx, vy };
}
function symPower(m, power) {
  const e = eigSym(m[0], m[1], m[3]);
  const q1 = Math.pow(Math.max(1e-4, e.l1), power), q2 = Math.pow(Math.max(1e-4, e.l2), power);
  const x = e.vx, y = e.vy;
  return [q1 * x * x + q2 * y * y, (q1 - q2) * x * y, (q1 - q2) * x * y, q1 * y * y + q2 * x * x];
}
const mul2 = (a, b) => [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3]];
function mklMatrix(src, tgt) {
  const S = [src.covAA + 1e-4, src.covAB || 0, src.covAB || 0, src.covBB + 1e-4];
  const T = [tgt.covAA + 1e-4, tgt.covAB || 0, tgt.covAB || 0, tgt.covBB + 1e-4];
  const sh = symPower(S, .5), sih = symPower(S, -.5), mid = mul2(mul2(sh, T), sh);
  let M = mul2(mul2(sih, symPower(mid, .5)), sih);
  // MKL 结果为对称正定矩阵，特征值就是奇异值；在此限幅避免色斑。
  M = symPower(M, 1); const e = eigSym(M[0], M[1], M[3]);
  const c1 = Math.max(.25, Math.min(4, e.l1)), c2 = Math.max(.25, Math.min(4, e.l2));
  const x = e.vx, y = e.vy;
  return [c1 * x * x + c2 * y * y, (c1 - c2) * x * y, (c1 - c2) * x * y, c1 * y * y + c2 * x * x];
}

function makeGradeTransform(src, tgt, opts = {}) {
  const mode = opts.mode || 'tone', strength = opts.strength ?? 1, mapL = opts.mapL;
  const M = mklMatrix(src, tgt), satBoost = opts.satBoost || 0;
  return (L, a, b) => {
    // tone=CDF 非线性影调；full=旧版线性明度迁移；chroma=保留照片明度。
    let nL = L;
    if (mode === 'tone' && mapL) nL = mapL(L);
    else if (mode === 'full') nL = (L - src.meanL) * (tgt.stdL / Math.max(1e-4, src.stdL)) + tgt.meanL;
    let na = M[0] * (a - src.meanA) + M[1] * (b - src.meanB) + tgt.meanA;
    let nb = M[2] * (a - src.meanA) + M[3] * (b - src.meanB) + tgt.meanB;
    // CDF 的 mapL 已包含强度；full 的线性迁移在这里混合。
    if (!(mode === 'tone' && mapL)) nL = L + (nL - L) * strength;
    na = a + (na - a) * strength; nb = b + (nb - b) * strength;
    if (satBoost) {
      const c = Math.hypot(na, nb);
      if (c > 1e-6) {
        let nc = c * (1 + satBoost * Math.exp(-Math.pow((c - 35) / 25, 2)));
        if (nc > 90) nc = 90 + (nc - 90) * .3;
        na *= nc / c; nb *= nc / c;
      }
    }
    if (Math.abs(nL - L) + Math.abs(na - a) + Math.abs(nb - b) < 1e-4) return [nL, na, nb];
    return compressLabToSrgb(nL, na, nb);
  };
}

// 兼容旧调用：抠图光照融合仍只需要传统的全局统计迁移。
function makeLabTransform(src, tgt, mode, strength) {
  const sL = Math.max(1e-4, src.stdL), sA = Math.max(1e-4, src.stdA), sB = Math.max(1e-4, src.stdB);
  return (L, a, b) => {
    let nL = mode === 'full' ? (L - src.meanL) * tgt.stdL / sL + tgt.meanL : L;
    let nA = (a - src.meanA) * tgt.stdA / sA + tgt.meanA, nB = (b - src.meanB) * tgt.stdB / sB + tgt.meanB;
    return [L + (nL - L) * strength, a + (nA - a) * strength, b + (nB - b) * strength];
  };
}

function applyTransfer(srcImageData, transform, labData = null) {
  const out = new ImageData(new Uint8ClampedArray(srcImageData.data), srcImageData.width, srcImageData.height), d = out.data;
  for (let p = 0, i = 0; i < d.length; i += 4, p++) {
    if (d[i + 3] < 8) continue;
    const j = p * 3, lab = labData ? null : rgb2lab(d[i], d[i + 1], d[i + 2]);
    const L = labData ? labData[j] : lab[0], a = labData ? labData[j + 1] : lab[1], b = labData ? labData[j + 2] : lab[2];
    const [nL, na, nb] = transform(L, a, b), rgb = lab2rgb(nL, na, nb);
    d[i] = rgb[0]; d[i + 1] = rgb[1]; d[i + 2] = rgb[2];
  }
  return out;
}
function applyTransferRegioned(srcImageData, tSky, tLand, weight, labData = null) {
  const out = new ImageData(new Uint8ClampedArray(srcImageData.data), srcImageData.width, srcImageData.height), d = out.data;
  for (let p = 0, i = 0; i < d.length; i += 4, p++) {
    if (d[i + 3] < 8) continue;
    const j = p * 3, lab = labData ? null : rgb2lab(d[i], d[i + 1], d[i + 2]);
    const L = labData ? labData[j] : lab[0], a = labData ? labData[j + 1] : lab[1], b = labData ? labData[j + 2] : lab[2];
    const s = tSky(L, a, b), l = tLand(L, a, b), w = weight[p];
    const rgb = lab2rgb(l[0] + (s[0] - l[0]) * w, l[1] + (s[1] - l[1]) * w, l[2] + (s[2] - l[2]) * w);
    d[i] = rgb[0]; d[i + 1] = rgb[1]; d[i + 2] = rgb[2];
  }
  return out;
}

// ---------- 天空掩膜：顶边连通、低纹理、偏蓝/中性 ----------
function skyMask(imageData, opts = {}) {
  const W = imageData.width, H = imageData.height, sw = Math.min(256, W), sh = Math.max(1, Math.round(H * sw / W));
  const c = document.createElement('canvas'); c.width = sw; c.height = sh;
  const src = document.createElement('canvas'); src.width = W; src.height = H; src.getContext('2d').putImageData(imageData, 0, 0);
  const ctx = c.getContext('2d'); ctx.drawImage(src, 0, 0, sw, sh); const td = ctx.getImageData(0, 0, sw, sh);
  const ls = new Float32Array(sw * sh), as = new Float32Array(sw * sh), bs = new Float32Array(sw * sh), vals = [];
  const usableH = Math.max(1, Math.floor(sh * (1 - (opts.ignoreBottomRatio || 0))));
  for (let p = 0, i = 0; p < ls.length; p++, i += 4) {
    const lab = rgb2lab(td.data[i], td.data[i + 1], td.data[i + 2]);
    ls[p] = lab[0]; as[p] = lab[1]; bs[p] = lab[2];
    if (((p / sw) | 0) < usableH) vals.push(lab[0]);
  }
  vals.sort((a, b) => a - b);
  const q35 = vals[Math.floor(vals.length * .35)] || 40, q60 = vals[Math.floor(vals.length * .6)] || 60;
  // L 与 L² 积分图，把每像素 5×5 标准差从 25 次访问降为 O(1)。
  const iw = sw + 1, integral = new Float64Array(iw * (sh + 1)), integral2 = new Float64Array(iw * (sh + 1));
  for (let y = 0; y < sh; y++) {
    let row = 0, row2 = 0;
    for (let x = 0; x < sw; x++) {
      const v = ls[y * sw + x]; row += v; row2 += v * v;
      integral[(y + 1) * iw + x + 1] = integral[y * iw + x + 1] + row;
      integral2[(y + 1) * iw + x + 1] = integral2[y * iw + x + 1] + row2;
    }
  }
  const rectSum = (sat, x0, y0, x1, y1) => sat[(y1 + 1) * iw + x1 + 1] - sat[y0 * iw + x1 + 1] - sat[(y1 + 1) * iw + x0] + sat[y0 * iw + x0];
  const cand = new Uint8Array(sw * sh);
  for (let y = 0; y < usableH; y++) for (let x = 0; x < sw; x++) {
    const p = y * sw + x, x0 = Math.max(0, x - 2), x1 = Math.min(sw - 1, x + 2), y0 = Math.max(0, y - 2), y1 = Math.min(sh - 1, y + 2);
    const n = (x1 - x0 + 1) * (y1 - y0 + 1), sum = rectSum(integral, x0, y0, x1, y1), sum2 = rectSum(integral2, x0, y0, x1, y1);
    const sd = Math.sqrt(Math.max(0, sum2 / n - (sum / n) ** 2));
    // 深蓝天空的顶边可能低于全图 60 分位；明显偏蓝时放宽到 35 分位。
    const coolSky = bs[p] < 15 && (ls[p] > q60 || (bs[p] < -5 && ls[p] > q35));
    // 黄昏橙粉天空：偏暖但明亮、色度受限（b<40、a 在粉橙范围），避免把暖色墙面全放进来
    const warmSky = bs[p] >= 15 && bs[p] < 40 && as[p] > -6 && as[p] < 25 && ls[p] > q60;
    if (sd < 6 && (coolSky || warmSky)) cand[p] = 1;
  }
  const keep = new Uint8Array(sw * sh), stack = [];
  // 种子取顶部 30% 行内全部候选像素——顶边被树枝/暗角遮挡时天空仍能被找到
  const seedRows = Math.max(1, Math.round(sh * 0.3));
  for (let y = 0; y < seedRows; y++) for (let x = 0; x < sw; x++) {
    const p = y * sw + x;
    if (cand[p] && !keep[p]) { keep[p] = 1; stack.push(p); }
  }
  while (stack.length) {
    const p = stack.pop(), x = p % sw, y = (p / sw) | 0;
    if (x > 0) { const n = p - 1; if (cand[n] && !keep[n]) { keep[n] = 1; stack.push(n); } }
    if (x + 1 < sw) { const n = p + 1; if (cand[n] && !keep[n]) { keep[n] = 1; stack.push(n); } }
    if (y > 0) { const n = p - sw; if (cand[n] && !keep[n]) { keep[n] = 1; stack.push(n); } }
    if (y + 1 < usableH) { const n = p + sw; if (cand[n] && !keep[n]) { keep[n] = 1; stack.push(n); } }
  }
  // 顶带 = 顶部 8% 行：真实天空必然在顶带有存在感（允许顶边被少量遮挡）
  const topBand = Math.max(1, Math.ceil(sh * .08));
  let area = 0, topHits = 0, sumY = 0, sumL = 0, blueCount = 0, restN = 0, restL = 0;
  for (let p = 0; p < keep.length; p++) {
    const y = (p / sw) | 0;
    if (keep[p]) {
      area++; sumY += y; sumL += ls[p]; if (bs[p] < -5) blueCount++; if (y < topBand) topHits++;
    } else if (y < usableH) { restN++; restL += ls[p]; }
  }
  const coverage = area / (sw * sh), meanY = area ? sumY / area / sh : 1, meanL = area ? sumL / area : 0, blueRatio = area ? blueCount / area : 0;
  const lumaSep = restN ? meanL - restL / restN : 0;
  const weight = new Float32Array(W * H);
  if (coverage < .05) return { weight, valid: false, coverage, reason: '天空候选面积不足 5%' };
  if (topHits < Math.max(3, sw * topBand * .1) || meanY > .62) return { weight, valid: false, coverage, reason: '候选区域不像从画面顶部延伸的天空' };
  if (meanL > 94 && blueRatio < .05) return { weight, valid: false, coverage, reason: '顶部区域接近纯白背景，不按天空处理' };
  // 掩膜区与画面其余部分明暗几乎无差、又不是明确的蓝色 → 多半是水面/墙面这类整幅平滑区
  if (lumaSep < 6 && blueRatio < .35) return { weight, valid: false, coverage, reason: '顶部平滑区与其余画面明暗对比不足，更像水面或墙面', debug: { lumaSep, blueRatio, meanL, meanY, topHits } };
  // 真实天空的质心贴着画面顶部；非明确蓝色的"天空"若质心落到中部，多半是水面/雾面反光
  if (meanY > .38 && blueRatio < .35) return { weight, valid: false, coverage, reason: '平滑亮区质心过低且不是明确的蓝色天空，更像水面反光', debug: { lumaSep, blueRatio, meanL, meanY, topHits } };
  // 最近邻放大 + 三次线性复杂度盒模糊，约 20px 柔和羽化。
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) weight[y * W + x] = keep[Math.min(sh - 1, Math.floor(y * sh / H)) * sw + Math.min(sw - 1, Math.floor(x * sw / W))];
  const radius = Math.max(3, Math.min(20, Math.round(Math.min(W, H) / 35)));
  for (let pass = 0; pass < 3; pass++) boxBlurMask(weight, W, H, radius);
  return { weight, valid: true, coverage, reason: '检测到顶部连通天空', debug: { lumaSep, blueRatio, meanL, meanY, topHits } };
}
function boxBlurMask(a, w, h, r) {
  const temp = new Float32Array(a.length);
  for (let y = 0; y < h; y++) {
    const base = y * w; let sum = 0, lo = 0, hi = Math.min(w - 1, r);
    for (let x = lo; x <= hi; x++) sum += a[base + x];
    for (let x = 0; x < w; x++) {
      temp[base + x] = sum / (hi - lo + 1);
      const nextLo = Math.max(0, x + 1 - r), nextHi = Math.min(w - 1, x + 1 + r);
      if (nextLo > lo) sum -= a[base + lo];
      if (nextHi > hi) sum += a[base + nextHi];
      lo = nextLo; hi = nextHi;
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0, lo = 0, hi = Math.min(h - 1, r);
    for (let y = lo; y <= hi; y++) sum += temp[y * w + x];
    for (let y = 0; y < h; y++) {
      a[y * w + x] = sum / (hi - lo + 1);
      const nextLo = Math.max(0, y + 1 - r), nextHi = Math.min(h - 1, y + 1 + r);
      if (nextLo > lo) sum -= temp[lo * w + x];
      if (nextHi > hi) sum += temp[nextHi * w + x];
      lo = nextLo; hi = nextHi;
    }
  }
}

// ---------- Bloom（空间效果，不进入 LUT） ----------
// 输入已是 1/4 分辨率时，可单独取出模糊亮部层；全分辨率导出用它避免
// 对 4800 万像素整图再分配多份 ImageData。
function makeBloomLayer(imageData) {
  const w = imageData.width, h = imageData.height;
  const out = new ImageData(new Uint8ClampedArray(imageData.data), w, h), md = out.data;
  const bright = new Float32Array(w * h * 3);
  for (let p = 0, i = 0; i < md.length; i += 4, p++) {
    const Y = .2126 * srgbToLinear(md[i]) + .7152 * srgbToLinear(md[i + 1]) + .0722 * srgbToLinear(md[i + 2]);
    const t = Math.max(0, Math.min(1, (Y - .72) / .20)); const m = t * t * (3 - 2 * t);
    bright[p * 3] = md[i] / 255 * m; bright[p * 3 + 1] = md[i + 1] / 255 * m; bright[p * 3 + 2] = md[i + 2] / 255 * m;
  }
  const r = Math.max(1, Math.round(Math.min(w, h) / 50));
  for (let pass = 0; pass < 3; pass++) boxBlurRgb(bright, w, h, r);
  for (let p = 0, i = 0; p < w * h; p++, i += 4) {
    md[i] = bright[p * 3] * 255; md[i + 1] = bright[p * 3 + 1] * 255; md[i + 2] = bright[p * 3 + 2] * 255; md[i + 3] = 255;
  }
  return out;
}

function applyBloom(imageData, gain) {
  if (!(gain > 0)) return imageData;
  const w = imageData.width, h = imageData.height, scale = .25, bw = Math.max(1, Math.round(w * scale)), bh = Math.max(1, Math.round(h * scale));
  const src = document.createElement('canvas'); src.width = w; src.height = h; src.getContext('2d').putImageData(imageData, 0, 0);
  const small = document.createElement('canvas'); small.width = bw; small.height = bh; const sctx = small.getContext('2d'); sctx.drawImage(src, 0, 0, bw, bh);
  const layer = makeBloomLayer(sctx.getImageData(0, 0, bw, bh));
  sctx.putImageData(layer, 0, 0);
  const bloom = document.createElement('canvas'); bloom.width = w; bloom.height = h;
  const bloomCtx = bloom.getContext('2d'); bloomCtx.imageSmoothingEnabled = true; bloomCtx.drawImage(small, 0, 0, w, h);
  const bd = bloom.getContext('2d').getImageData(0, 0, w, h).data, out = new ImageData(new Uint8ClampedArray(imageData.data), w, h), od = out.data;
  for (let i = 0; i < od.length; i += 4) for (let cidx = 0; cidx < 3; cidx++) od[i + cidx] = 255 * (1 - (1 - od[i + cidx] / 255) * (1 - bd[i + cidx] / 255 * gain));
  return out;
}

function boxBlurRgb(a, w, h, r) {
  const temp = new Float32Array(a.length);
  for (let y = 0; y < h; y++) for (let ch = 0; ch < 3; ch++) {
    let sum = 0, lo = 0, hi = Math.min(w - 1, r);
    for (let x = lo; x <= hi; x++) sum += a[(y * w + x) * 3 + ch];
    for (let x = 0; x < w; x++) {
      temp[(y * w + x) * 3 + ch] = sum / (hi - lo + 1);
      const nl = Math.max(0, x + 1 - r), nh = Math.min(w - 1, x + 1 + r);
      if (nl > lo) sum -= a[(y * w + lo) * 3 + ch];
      if (nh > hi) sum += a[(y * w + nh) * 3 + ch];
      lo = nl; hi = nh;
    }
  }
  for (let x = 0; x < w; x++) for (let ch = 0; ch < 3; ch++) {
    let sum = 0, lo = 0, hi = Math.min(h - 1, r);
    for (let y = lo; y <= hi; y++) sum += temp[(y * w + x) * 3 + ch];
    for (let y = 0; y < h; y++) {
      a[(y * w + x) * 3 + ch] = sum / (hi - lo + 1);
      const nl = Math.max(0, y + 1 - r), nh = Math.min(h - 1, y + 1 + r);
      if (nl > lo) sum -= temp[(lo * w + x) * 3 + ch];
      if (nh > hi) sum += temp[(nh * w + x) * 3 + ch];
      lo = nl; hi = nh;
    }
  }
}

// ---------- 主色与 LUT ----------
function extractPalette(imageData, k = 6, sampleStep = 4, opts = {}) {
  const pts = [];
  eachSample(imageData, sampleStep, opts, (i) => pts.push([imageData.data[i], imageData.data[i + 1], imageData.data[i + 2]]));
  if (!pts.length) return [];
  let centroids = Array.from({ length: k }, (_, j) => pts[Math.floor((j + .5) / k * pts.length)].slice()), assign = new Array(pts.length);
  for (let iter = 0; iter < 12; iter++) {
    const sum = Array.from({ length: k }, () => [0, 0, 0, 0]);
    pts.forEach((p, pi) => { let best = 0, bd = Infinity; centroids.forEach((c, ci) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2; if (d < bd) { bd = d; best = ci; } }); assign[pi] = best; sum[best][0] += p[0]; sum[best][1] += p[1]; sum[best][2] += p[2]; sum[best][3]++; });
    sum.forEach((s, i) => { if (s[3]) centroids[i] = [s[0] / s[3], s[1] / s[3], s[2] / s[3]]; });
  }
  const count = new Array(k).fill(0); assign.forEach((a) => count[a]++);
  return centroids.map((rgb, i) => ({ rgb: rgb.map(Math.round), ratio: count[i] / pts.length })).filter((v) => v.ratio > .001).sort((a, b) => b.ratio - a.ratio);
}
function generateCubeLUT(transform, size = 33, title = 'Seichi Grade') {
  const lines = [`TITLE "${title}"`, `LUT_3D_SIZE ${size}`, 'DOMAIN_MIN 0.0 0.0 0.0', 'DOMAIN_MAX 1.0 1.0 1.0'], max = size - 1;
  for (let bi = 0; bi < size; bi++) for (let gi = 0; gi < size; gi++) for (let ri = 0; ri < size; ri++) {
    const [L, a, b] = rgb2lab(ri / max * 255, gi / max * 255, bi / max * 255), [nL, na, nb] = transform(L, a, b), [r, g, bb] = lab2rgb(nL, na, nb);
    lines.push(`${(r / 255).toFixed(6)} ${(g / 255).toFixed(6)} ${(bb / 255).toFixed(6)}`);
  }
  return lines.join('\n');
}

export { rgb2lab, lab2rgb, imageDataToLab, labStats, makeLabTransform, makeLumaCdfMap, makeGradeTransform, applyTransfer, applyTransferRegioned, skyMask, makeBloomLayer, applyBloom, extractPalette, generateCubeLUT };
