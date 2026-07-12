import {
  rgb2lab, lab2rgb, imageDataToLab, labStats, makeLabTransform, makeLumaCdfMap, makeGradeTransform,
  applyTransfer, applyTransferRegioned, skyMask, applyBloom, generateCubeLUT,
} from '../color.js?qa=20260712l';

const $ = (id) => document.getElementById(id);
const tests = [];
function record(name, ok, measured, limit, warn = false) { tests.push({ name, ok, measured, limit, warn }); }

async function loadImageData(url, maxDim = 1400) {
  const img = new Image(); img.src = url; await img.decode();
  const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
  const c = document.createElement('canvas'); c.width = Math.round(img.width * scale); c.height = Math.round(img.height * scale);
  const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0, c.width, c.height);
  return ctx.getImageData(0, 0, c.width, c.height);
}
function draw(id, imageData) { const c = $(id); c.width = imageData.width; c.height = imageData.height; c.getContext('2d').putImageData(imageData, 0, 0); }
function drawMask(id, mask, w, h) {
  const out = new ImageData(w, h);
  for (let p = 0, i = 0; p < mask.length; p++, i += 4) out.data[i] = out.data[i + 1] = out.data[i + 2] = Math.round(mask[p] * 255), out.data[i + 3] = 255;
  draw(id, out);
}
function inverse(mask) { const a = new Float32Array(mask.length); for (let i = 0; i < a.length; i++) a[i] = 1 - mask[i]; return a; }
function maxPixelDiff(a, b) { let max = 0; for (let i = 0; i < a.data.length; i++) max = Math.max(max, Math.abs(a.data[i] - b.data[i])); return max; }
function lumaCdf(img, ignoreBottomRatio = 0) {
  const h = new Float64Array(256), endY = Math.floor(img.height * (1 - ignoreBottomRatio)); let n = 0;
  for (let y = 0; y < endY; y += 2) for (let x = 0; x < img.width; x += 2) {
    const i = (y * img.width + x) * 4, L = rgb2lab(img.data[i], img.data[i + 1], img.data[i + 2])[0]; h[Math.max(0, Math.min(255, Math.round(L * 2.55)))]++; n++;
  }
  let sum = 0; for (let i = 0; i < 256; i++) { sum += h[i]; h[i] = sum / Math.max(1, n); } return h;
}
function ks(a, b) { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; }
function abHistIntersection(a, b, ignoreBottomB = 0) {
  const bins = 24, build = (img, ignoreBottom) => {
    const hist = new Float64Array(bins * bins), maxY = Math.floor(img.height * (1 - ignoreBottom)); let n = 0;
    for (let y = 0; y < maxY; y += 4) for (let x = 0; x < img.width; x += 4) {
      const i = (y * img.width + x) * 4, lab = rgb2lab(img.data[i], img.data[i + 1], img.data[i + 2]);
      const ai = Math.max(0, Math.min(bins - 1, Math.floor((lab[1] + 128) / 256 * bins))), bi = Math.max(0, Math.min(bins - 1, Math.floor((lab[2] + 128) / 256 * bins)));
      hist[bi * bins + ai]++; n++;
    }
    for (let i = 0; i < hist.length; i++) hist[i] /= Math.max(1, n); return hist;
  };
  const ha = build(a, 0), hb = build(b, ignoreBottomB); let sum = 0;
  for (let i = 0; i < ha.length; i++) sum += Math.min(ha[i], hb[i]); return sum;
}
function clippingRatio(img) {
  let clipped = 0, n = img.width * img.height * 3;
  for (let i = 0; i < img.data.length; i += 4) for (let c = 0; c < 3; c++) if (img.data[i + c] === 0 || img.data[i + c] === 255) clipped++;
  return clipped / n;
}
function bloomContrast(base, bloomed, labData) {
  let hi = 0, hn = 0, lo = 0, ln = 0;
  for (let p = 0, i = 0; i < base.data.length; p++, i += 4) {
    const diff = (Math.abs(base.data[i] - bloomed.data[i]) + Math.abs(base.data[i + 1] - bloomed.data[i + 1]) + Math.abs(base.data[i + 2] - bloomed.data[i + 2])) / 3;
    const L = labData[p * 3]; if (L > 72) { hi += diff; hn++; } else if (L < 40) { lo += diff; ln++; }
  }
  return { high: hi / Math.max(1, hn), dark: lo / Math.max(1, ln) };
}

function parseCube(text) {
  const lines = text.split(/\r?\n/).filter((l) => l && !/^(TITLE|LUT_|DOMAIN|#)/.test(l));
  return new Float32Array(lines.flatMap((l) => l.trim().split(/\s+/).map(Number)));
}
function cubeSample(data, size, r, g, b) {
  const max = size - 1, xr = r * max, xg = g * max, xb = b * max;
  const r0 = Math.floor(xr), r1 = Math.min(max, r0 + 1), g0 = Math.floor(xg), g1 = Math.min(max, g0 + 1), b0 = Math.floor(xb), b1 = Math.min(max, b0 + 1);
  const fr = xr - r0, fg = xg - g0, fb = xb - b0, at = (rr, gg, bb, ch) => data[((bb * size + gg) * size + rr) * 3 + ch];
  const out = [0, 0, 0];
  for (let ch = 0; ch < 3; ch++) {
    const c00 = at(r0, g0, b0, ch) * (1 - fr) + at(r1, g0, b0, ch) * fr;
    const c10 = at(r0, g1, b0, ch) * (1 - fr) + at(r1, g1, b0, ch) * fr;
    const c01 = at(r0, g0, b1, ch) * (1 - fr) + at(r1, g0, b1, ch) * fr;
    const c11 = at(r0, g1, b1, ch) * (1 - fr) + at(r1, g1, b1, ch) * fr;
    out[ch] = (c00 * (1 - fg) + c10 * fg) * (1 - fb) + (c01 * (1 - fg) + c11 * fg) * fb;
  }
  return out;
}

async function run() {
  // 黄昏验收对：白天风景作为源图，紫橙黄昏动画作为目标。
  const target = await loadImageData('../test-izu-far.jpg'), source = await loadImageData('../test-izu-scenery.jpg');
  const noSky = await loadImageData('../test-anime.png');
  draw('target', target); draw('source', source);

  const sourceLab = imageDataToLab(source), targetLab = imageDataToLab(target);

  const srcOld = labStats(source, 2, { labData: sourceLab }), tgtOld = labStats(target, 2, { ignoreBottomRatio: .12, labData: targetLab });
  const baseline = applyTransfer(source, makeLabTransform(srcOld, tgtOld, 'full', .85), sourceLab); draw('baseline', baseline);

  const cacheStart = performance.now();
  const sourceSky = skyMask(source), targetSky = skyMask(target, { ignoreBottomRatio: .12 }), whiteSky = skyMask(noSky);
  const src = labStats(source, 2, { cov: true, labData: sourceLab }), tgt = labStats(target, 2, { cov: true, ignoreBottomRatio: .12, labData: targetLab });
  const globalMap = makeLumaCdfMap(source, target, { ignoreBottomRatio: .12, srcLabData: sourceLab, tgtLabData: targetLab });
  const cacheMs = performance.now() - cacheStart;
  drawMask('sourceMask', sourceSky.weight, source.width, source.height); drawMask('targetMask', targetSky.weight, target.width, target.height);

  const strength = .85, satBoost = .15;
  const globalTf = makeGradeTransform(src, tgt, { mode: 'tone', strength, mapL: globalMap.build(strength), satBoost });
  const applyStart = performance.now();
  let graded;
  const sourceLand = inverse(sourceSky.weight), targetLand = inverse(targetSky.weight);
  if (sourceSky.valid && targetSky.valid) {
    const sl = sourceLand, tl = targetLand;
    const skyMap = makeLumaCdfMap(source, target, { srcWeightMask: sourceSky.weight, tgtWeightMask: targetSky.weight, ignoreBottomRatio: .12, srcLabData: sourceLab, tgtLabData: targetLab });
    const landMap = makeLumaCdfMap(source, target, { srcWeightMask: sl, tgtWeightMask: tl, ignoreBottomRatio: .12, srcLabData: sourceLab, tgtLabData: targetLab });
    const tSky = makeGradeTransform(labStats(source, 2, { cov: true, weightMask: sourceSky.weight, labData: sourceLab }), labStats(target, 2, { cov: true, weightMask: targetSky.weight, ignoreBottomRatio: .12, labData: targetLab }), { mode: 'tone', strength, mapL: skyMap.build(strength), satBoost });
    const tLand = makeGradeTransform(labStats(source, 2, { cov: true, weightMask: sl, labData: sourceLab }), labStats(target, 2, { cov: true, weightMask: tl, ignoreBottomRatio: .12, labData: targetLab }), { mode: 'tone', strength, mapL: landMap.build(strength), satBoost });
    graded = applyTransferRegioned(source, tSky, tLand, sourceSky.weight, sourceLab);
  } else graded = applyTransfer(source, globalTf, sourceLab);
  const gradedNoBloom = graded;
  graded = applyBloom(gradedNoBloom, .25); const applyMs = performance.now() - applyStart; draw('graded', graded);
  // WP1 按规格指定的 test-izu-scenery 目标对单独验收；主视觉对继续用黄昏目标验收 WP3。
  const wp1SrcStats = labStats(target, 2, { cov: true, labData: targetLab });
  const wp1TgtStats = labStats(source, 2, { cov: true, ignoreBottomRatio: .12, labData: sourceLab });
  const wp1Baseline = applyTransfer(target, makeLabTransform(wp1SrcStats, wp1TgtStats, 'full', strength), targetLab);
  const wp1Map = makeLumaCdfMap(target, source, { ignoreBottomRatio: .12, srcLabData: targetLab, tgtLabData: sourceLab });
  const wp1Cdf = applyTransfer(target, makeGradeTransform(wp1SrcStats, wp1TgtStats, { mode: 'tone', strength, mapL: wp1Map.build(strength), satBoost: 0 }), targetLab);

  const zeroMap = makeLumaCdfMap(source, target, { ignoreBottomRatio: .12, srcLabData: sourceLab, tgtLabData: targetLab }).build(0);
  const zeroOut = applyTransfer(source, makeGradeTransform(src, tgt, { mode: 'tone', strength: 0, mapL: zeroMap, satBoost: 0 }), sourceLab);
  const selfMap = makeLumaCdfMap(source, source, { srcLabData: sourceLab, tgtLabData: sourceLab }).build(1), selfOut = applyTransfer(source, makeGradeTransform(src, src, { mode: 'tone', strength: 1, mapL: selfMap, satBoost: 0 }), sourceLab);
  const bloomZero = applyBloom(source, 0);
  record('WP1 strength=0 恒等', maxPixelDiff(source, zeroOut) <= 1, `${maxPixelDiff(source, zeroOut)} / 255`, '≤ 1 / 255');
  record('WP1 CDF 比 Reinhard 更接近目标亮度', ks(lumaCdf(wp1Cdf), lumaCdf(source, .12)) < ks(lumaCdf(wp1Baseline), lumaCdf(source, .12)), `CDF ${ks(lumaCdf(wp1Cdf), lumaCdf(source, .12)).toFixed(4)}；基线 ${ks(lumaCdf(wp1Baseline), lumaCdf(source, .12)).toFixed(4)}`, 'CDF KS < 基线 KS');
  globalMap.build(1); let minSlope = Infinity, maxSlope = -Infinity;
  for (let i = 1; i < globalMap.table.length; i++) { const slope = (globalMap.table[i] - globalMap.table[i - 1]) / (100 / (globalMap.table.length - 1)); minSlope = Math.min(minSlope, slope); maxSlope = Math.max(maxSlope, slope); }
  record('WP1 CDF 单调与斜率限幅', minSlope >= .24 && maxSlope <= 4.01, `最小 ${minSlope.toFixed(3)}；最大 ${maxSlope.toFixed(3)}`, '0.25～4（插值容差）');
  record('WP2 源图/目标天空检测', sourceSky.valid && targetSky.valid, `覆盖率 ${(sourceSky.coverage * 100).toFixed(1)}% / ${(targetSky.coverage * 100).toFixed(1)}%`, '两侧 valid=true');
  record('WP2 白底角色图不误判天空', !whiteSky.valid, `${whiteSky.valid ? '误判' : '正确拒绝'}；${whiteSky.reason}`, 'valid=false');
  const gradedLab = imageDataToLab(gradedNoBloom), globalOut = applyTransfer(source, globalTf, sourceLab), globalLab = imageDataToLab(globalOut);
  const skyOutStats = labStats(gradedNoBloom, 2, { weightMask: sourceSky.weight, labData: gradedLab }), skyTgtStats = labStats(target, 2, { weightMask: targetSky.weight, ignoreBottomRatio: .12, labData: targetLab });
  const skyDeltaAB = Math.hypot(skyOutStats.meanA - skyTgtStats.meanA, skyOutStats.meanB - skyTgtStats.meanB);
  record('WP2 天空平均色相接近目标', skyDeltaAB < 10, `Δab ${skyDeltaAB.toFixed(2)}`, '< 10');
  const landRegion = labStats(gradedNoBloom, 2, { weightMask: sourceLand, labData: gradedLab }), landGlobal = labStats(globalOut, 2, { weightMask: sourceLand, labData: globalLab }), landTgt = labStats(target, 2, { weightMask: targetLand, ignoreBottomRatio: .12, labData: targetLab });
  record('WP2 地景不被天空颜色平摊', Math.abs(landRegion.meanB - landTgt.meanB) <= Math.abs(landGlobal.meanB - landTgt.meanB), `分区 Δb ${Math.abs(landRegion.meanB - landTgt.meanB).toFixed(2)}；全局 Δb ${Math.abs(landGlobal.meanB - landTgt.meanB).toFixed(2)}`, '分区 ≤ 全局');
  record('WP3 同图自迁移', maxPixelDiff(source, selfOut) <= 2, `${maxPixelDiff(source, selfOut)} / 255`, '≤ 2 / 255');
  const mklOut = applyTransfer(source, makeGradeTransform(src, tgt, { mode: 'chroma', strength: 1, satBoost: 0 }), sourceLab), reinhardOut = applyTransfer(source, makeLabTransform(srcOld, tgtOld, 'chroma', 1), sourceLab);
  const mklOverlap = abHistIntersection(mklOut, target, .12), reinhardOverlap = abHistIntersection(reinhardOut, target, .12);
  record('WP3 二维 ab 分布优于独立通道', mklOverlap >= reinhardOverlap, `MKL ${mklOverlap.toFixed(3)}；Reinhard ${reinhardOverlap.toFixed(3)}`, 'MKL ≥ Reinhard');
  record('WP3 溢色截断未增加', clippingRatio(mklOut) <= clippingRatio(reinhardOut) + .01, `MKL ${(clippingRatio(mklOut) * 100).toFixed(2)}%；基线 ${(clippingRatio(reinhardOut) * 100).toFixed(2)}%`, '不高于基线 1%');
  record('WP4 Bloom=0 恒等', maxPixelDiff(source, bloomZero) === 0, `${maxPixelDiff(source, bloomZero)} / 255`, '= 0');
  const bloomEffect = bloomContrast(gradedNoBloom, graded, gradedLab);
  record('WP4 Bloom 主要作用于亮部', bloomEffect.high > bloomEffect.dark * 2, `亮部 ${bloomEffect.high.toFixed(2)}；暗部 ${bloomEffect.dark.toFixed(2)} RGB级`, '亮部 > 暗部×2');

  const lutTf = globalTf, cubeSize = 65, lutStart = performance.now(), cubeText = generateCubeLUT(lutTf, cubeSize), lutGenerateMs = performance.now() - lutStart;
  const cube = parseCube(cubeText), lcg = (() => { let s = 0x12345678; return () => ((s = (1664525 * s + 1013904223) >>> 0) / 4294967296); })();
  let lutMax = 0, lutWorst = null;
  for (let i = 0; i < 4096; i++) {
    const r = lcg(), g = lcg(), b = lcg(), lab = rgb2lab(r * 255, g * 255, b * 255), out = lutTf(...lab), rgb = lab2rgb(...out).map((v) => v / 255), lut = cubeSample(cube, cubeSize, r, g, b);
    const err = Math.max(...rgb.map((v, ch) => Math.abs(v - lut[ch])));
    if (err > lutMax) { lutMax = err; lutWorst = { input: [r, g, b], direct: rgb, lut }; }
  }
  record(`LUT ${cubeSize}³ · 4096 点三线性回验`, lutMax <= 2 / 255, `${(lutMax * 255).toFixed(2)} / 255；输入 ${lutWorst.input.map((v) => v.toFixed(3)).join(',')}；直接 ${lutWorst.direct.map((v) => v.toFixed(3)).join(',')}；LUT ${lutWorst.lut.map((v) => v.toFixed(3)).join(',')}`, '≤ 2 / 255');
  record('65³ LUT 生成耗时', lutGenerateMs < 3000, `${lutGenerateMs.toFixed(1)} ms；${(cubeText.length / 1048576).toFixed(1)} MB`, '< 3000 ms');
  record('首次派生数据性能', cacheMs < 300, `${cacheMs.toFixed(1)} ms`, '< 300 ms', cacheMs >= 300);
  record('滑杆重算链路性能（含 Bloom）', applyMs < 300, `${applyMs.toFixed(1)} ms`, '< 300 ms', applyMs >= 300);

  const body = $('results');
  for (const t of tests) {
    const tr = document.createElement('tr'), cls = t.ok ? 'pass' : t.warn ? 'warn' : 'fail';
    tr.innerHTML = `<td>${t.name}</td><td class="${cls}">${t.ok ? '通过' : t.warn ? '性能警告' : '失败'}</td><td>${t.measured}</td><td>${t.limit}</td>`; body.appendChild(tr);
  }
  const failed = tests.filter((t) => !t.ok && !t.warn).length, warned = tests.filter((t) => !t.ok && t.warn).length;
  $('summary').innerHTML = failed ? `<span class="fail">未通过：${failed} 项</span>，性能警告 ${warned} 项。` : `<span class="pass">全部数值验收通过</span>${warned ? `，性能警告 ${warned} 项` : ''}。`;
  document.title = failed ? `❌ ${failed} 项失败 · 调色验收` : '✅ 调色验收通过';
}

window.addEventListener('load', () => setTimeout(() => run().catch((error) => {
  $('summary').innerHTML = `<span class="fail">测试异常：${error.message}</span>`;
  console.error(error);
}), 50), { once: true });
