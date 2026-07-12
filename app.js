// app.js — UI 编排：上传、迁移、对比预览、导出
import {
  imageDataToLab, labStats, makeLabTransform, makeLumaCdfMap, makeGradeTransform,
  applyTransfer, applyTransferRegioned, skyMask, makeBloomLayer, applyBloom, extractPalette, generateCubeLUT,
} from './color.js?v=20260712d';
import { extractForeground, cutoutCanvas, cleanupAlpha, alphaBBox } from './segment.js';
import { extractForegroundAI, extractCharactersAI, extractCharactersInRegion, mergeCharacterAlphas } from './ai-segment.js';
import { releaseAllSessions } from './ort-env.js';
import { profile as DEVICE } from './platform.js';

const IS_MOBILE = DEVICE.isMobile;
const MAX_DIM = DEVICE.previewMax;
const EXPORT_TILE = DEVICE.exportTile;
const EXPORT_MAX_PIXELS = DEVICE.exportMaxPixels;
const EXPORT_MAX_SIDE = 16_384;

const state = {
  anime: null,   // { imgData, stats, width, height }
  photo: null,   // { imgData, width, height, srcUrl, align }
                 // srcUrl = 原始文件 objectURL，导出时全分辨率重放用
                 // align  = 构图对齐裁剪框 {x,y,w,h}（原图归一化坐标），null=未裁剪
  gradedData: null,
  transform: null,
  cutout: null,        // 清理后的角色 canvas（bbox 裁剪）
  rawAlpha: null,      // 抠图原始 alpha（Uint8，与 anime 同尺寸），供阈值/收边重算
  charSeg: null,       // AI 检测流水线结果 { chars, included:Set }，供勾选角色重合并
  rawW: 0, rawH: 0,
  charPos: { cx: 0.5, cy: 0.62 }, // 角色中心在场景中的归一化位置
  charDraw: null,      // 角色在 canvas 坐标的绘制矩形 {dx,dy,dw,dh}，用于拖拽命中
  harmonizedCache: null,
  gradeCache: null,   // 图片不变时复用统计、CDF 与天空掩膜；滑杆只重套用
  lastExport: null,   // iOS 二次用户手势分享用 { blob, name, width, height }
  aiBusy: false,
};

const $ = (id) => document.getElementById(id);
const setStatus = (t) => { $('status').textContent = t; };

// 把 File 读成按 MAX_DIM 缩放后的 ImageData
function fileToImageData(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      let { width, height } = img;
      const scale = Math.min(1, MAX_DIM / Math.max(width, height));
      width = Math.round(width * scale);
      height = Math.round(height * scale);
      const c = document.createElement('canvas');
      c.width = width; c.height = height;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0, width, height);
      resolve({ imgData: ctx.getImageData(0, 0, width, height), width, height, url: img.src, originalWidth: img.naturalWidth, originalHeight: img.naturalHeight, fileName: file.name });
    };
    img.onerror = () => {
      URL.revokeObjectURL(img.src);
      reject(new Error(isHeicFile(file)
        ? '此版本浏览器无法解码 HEIC/HEIF，请更新 iOS/Safari 或转换为 JPEG 后重试'
        : '浏览器无法解码这张图片，请转换为 JPEG 或 PNG 后重试'));
    };
    img.src = URL.createObjectURL(file);
  });
}

function isHeicFile(file) {
  return /\.(heic|heif)$/i.test(file.name || '') || /image\/(heic|heif)/i.test(file.type || '');
}

// 开发验收/演示素材走同一条读取路径，避免另写一套“看起来能跑”的测试逻辑。
function urlToImageData(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      let { width, height } = img;
      const scale = Math.min(1, MAX_DIM / Math.max(width, height));
      width = Math.round(width * scale); height = Math.round(height * scale);
      const c = document.createElement('canvas'); c.width = width; c.height = height;
      const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0, width, height);
      resolve({
        imgData: ctx.getImageData(0, 0, width, height), width, height, url,
        originalWidth: img.naturalWidth, originalHeight: img.naturalHeight,
      });
    };
    img.onerror = reject; img.src = url;
  });
}

function drawToCanvas(canvas, imgData) {
  canvas.width = imgData.width;
  canvas.height = imgData.height;
  canvas.getContext('2d').putImageData(imgData, 0, 0);
}

function renderPalette(palette) {
  const box = $('palette');
  box.innerHTML = '';
  for (const c of palette.slice(0, 6)) {
    const [r, g, b] = c.rgb;
    const hex = '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
    const sw = document.createElement('div');
    sw.className = 'swatch';
    sw.style.background = hex;
    sw.dataset.hex = hex;
    sw.title = `${hex} · ${(c.ratio * 100).toFixed(0)}%`;
    box.appendChild(sw);
  }
}

function inverseMask(mask) {
  const out = new Float32Array(mask.length);
  for (let i = 0; i < mask.length; i++) out[i] = 1 - mask[i];
  return out;
}

// 只在换图或字幕设置改变时生成统计与 CDF；滑杆不重新扫直方图/天空。
function getGradeCache() {
  const ignoreSub = $('ignoreSub').checked ? 0.12 : 0;
  const old = state.gradeCache;
  if (old && old.anime === state.anime && old.photo === state.photo && old.ignoreSub === ignoreSub) return old;
  const photoOpts = { cov: true };
  const animeOpts = { cov: true, ignoreBottomRatio: ignoreSub };
  const photoLab = imageDataToLab(state.photo.imgData), animeLab = imageDataToLab(state.anime.imgData);
  const cache = {
    anime: state.anime, photo: state.photo, ignoreSub,
    photoLab, animeLab,
    global: {
      srcStats: labStats(state.photo.imgData, 2, { ...photoOpts, labData: photoLab }),
      tgtStats: labStats(state.anime.imgData, 2, { ...animeOpts, labData: animeLab }),
      map: makeLumaCdfMap(state.photo.imgData, state.anime.imgData, { ignoreBottomRatio: ignoreSub, srcLabData: photoLab, tgtLabData: animeLab }),
    },
    photoSky: skyMask(state.photo.imgData),
    animeSky: skyMask(state.anime.imgData, { ignoreBottomRatio: ignoreSub }),
  };
  if (cache.photoSky.valid && cache.animeSky.valid) {
    const ps = cache.photoSky.weight, as = cache.animeSky.weight;
    const pl = inverseMask(ps), al = inverseMask(as);
    cache.region = {
      sky: {
        srcStats: labStats(state.photo.imgData, 2, { weightMask: ps, cov: true, labData: photoLab }),
        tgtStats: labStats(state.anime.imgData, 2, { weightMask: as, ignoreBottomRatio: ignoreSub, cov: true, labData: animeLab }),
        map: makeLumaCdfMap(state.photo.imgData, state.anime.imgData, { srcWeightMask: ps, tgtWeightMask: as, ignoreBottomRatio: ignoreSub, srcLabData: photoLab, tgtLabData: animeLab }),
      },
      land: {
        srcStats: labStats(state.photo.imgData, 2, { weightMask: pl, cov: true, labData: photoLab }),
        tgtStats: labStats(state.anime.imgData, 2, { weightMask: al, ignoreBottomRatio: ignoreSub, cov: true, labData: animeLab }),
        map: makeLumaCdfMap(state.photo.imgData, state.anime.imgData, { srcWeightMask: pl, tgtWeightMask: al, ignoreBottomRatio: ignoreSub, srcLabData: photoLab, tgtLabData: animeLab }),
      },
    };
  }
  state.gradeCache = cache;
  return cache;
}

// 用当前滑杆值构建全套变换闭包。预览（recompute）与全分辨率导出共用，
// 保证两边永远是同一套映射。
function buildTransforms(cache) {
  const mode = $('mode').value;
  const strength = parseInt($('strength').value, 10) / 100;
  const satBoost = parseInt($('satBoost').value, 10) / 100;
  const g = cache.global;
  const out = {
    global: makeGradeTransform(g.srcStats, g.tgtStats, { mode, strength, mapL: g.map.build(strength), satBoost }),
    tSky: null, tLand: null,
  };
  if (cache.region) {
    const s = cache.region.sky, l = cache.region.land;
    out.tSky = makeGradeTransform(s.srcStats, s.tgtStats, { mode, strength, mapL: s.map.build(strength), satBoost });
    out.tLand = makeGradeTransform(l.srcStats, l.tgtStats, { mode, strength, mapL: l.map.build(strength), satBoost });
  }
  return out;
}

// 核心：重新计算调色（重活），再重绘合成（轻活）。
function recompute() {
  if (!state.anime || !state.photo) return;
  const mode = $('mode').value;
  const cache = getGradeCache();
  const skyControl = $('skyRegion');
  skyControl.disabled = !cache.region;
  const skyReason = cache.region ? '照片和动画均检测到天空' : `照片：${cache.photoSky.reason}；动画：${cache.animeSky.reason}`;
  skyControl.title = cache.region ? '' : `已自动使用全局迁移。${skyReason}`;
  const t = buildTransforms(cache);
  state.transform = t.global;
  const useRegion = skyControl.checked && cache.region;
  if (useRegion) {
    state.gradedData = applyTransferRegioned(state.photo.imgData, t.tSky, t.tLand, cache.photoSky.weight, cache.photoLab);
  } else {
    state.gradedData = applyTransfer(state.photo.imgData, t.global, cache.photoLab);
  }
  state.gradedData = applyBloom(state.gradedData, parseInt($('bloom').value, 10) / 100);
  drawToCanvas($('canvasOrig'), state.photo.imgData);
  $('emptyHint').hidden = true;
  redrawComposite();          // 画 gradedData + 角色
  syncCanvasSize();
  ['btnExportImg', 'btnExportCompare', 'btnExportLut'].forEach(id => $(id).disabled = false);
  const modeText = mode === 'tone' ? '影调+色彩' : mode === 'full' ? '完整' : '仅色彩';
  setStatus(`已调色 · ${modeText} · 强度 ${$('strength').value}% · 天空分区：${useRegion ? '已启用' : `未启用（${skyReason}）`}`);
}

let gradeFrame = 0;
function scheduleRecompute() {
  cancelAnimationFrame(gradeFrame);
  gradeFrame = requestAnimationFrame(() => { gradeFrame = 0; recompute(); });
}

// 轻活：把缓存的 gradedData 重画到调色后 canvas，再叠角色（拖拽/缩放时只跑这个）
function redrawComposite() {
  if (!state.gradedData) return;
  drawToCanvas($('canvasGraded'), state.gradedData);
  if (state.cutout && $('composite').checked) compositeCharacter($('canvasGraded'));
}

// 让两个叠放的 canvas 在容器内等比同尺寸显示。
// 尺寸以 canvasGraded 当前像素尺寸为准：普通模式=照片尺寸，对齐模式=动画宽高比画布。
function syncCanvasSize() {
  const compare = $('compare');
  const cw = compare.clientWidth, ch = compare.clientHeight;
  const iw = $('canvasGraded').width || state.photo.width, ih = $('canvasGraded').height || state.photo.height;
  const scale = Math.min(cw / iw, ch / ih);
  const w = iw * scale, h = ih * scale;
  for (const c of [$('canvasGraded'), $('canvasOrig')]) {
    c.style.width = w + 'px';
    c.style.height = h + 'px';
  }
  const ghost = $('alignGhost');
  if (ghost && !ghost.hidden) { ghost.style.width = w + 'px'; ghost.style.height = h + 'px'; }
}

// 光照融合：把实景场景的色调轻轻染到角色上，让它融入场景（带缓存）。
function invalidateHarmonize() { state.harmonizedCache = null; }

// 去色渗：半透明边缘像素的颜色被动画背景污染。用 5×5 邻域内实心像素
// （alpha>235）的均值色替换，越透明越信任内部色。只处理边缘带，开销可忽略。
function decontaminateEdges(img) {
  const { width: w, height: h, data: d } = img;
  const orig = new Uint8ClampedArray(d);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4, a = orig[i + 3];
    if (a < 8 || a > 235) continue;
    let sr = 0, sg = 0, sb = 0, n = 0;
    for (let yy = Math.max(0, y - 2); yy <= Math.min(h - 1, y + 2); yy++) {
      for (let xx = Math.max(0, x - 2); xx <= Math.min(w - 1, x + 2); xx++) {
        const j = (yy * w + xx) * 4;
        if (orig[j + 3] > 235) { sr += orig[j]; sg += orig[j + 1]; sb += orig[j + 2]; n++; }
      }
    }
    if (!n) continue;
    const t = 0.7 * (1 - a / 255);
    d[i] += (sr / n - orig[i]) * t;
    d[i + 1] += (sg / n - orig[i + 1]) * t;
    d[i + 2] += (sb / n - orig[i + 2]) * t;
  }
}

// 亮度颗粒：动画平涂 vs 照片噪点的质感差是"贴纸感"来源之一。
// 确定性 xorshift，参数不变时输出不变（缓存友好、不闪烁）。
function addLumaGrain(img, amp) {
  const d = img.data;
  let s = 0x9e3779b9;
  for (let i = 0; i < d.length; i += 4) {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    if (d[i + 3] < 8) continue;
    const n = ((s / 4294967296) - 0.5) * 2 * amp;
    d[i] += n; d[i + 1] += n; d[i + 2] += n;
  }
}

function harmonizedCutout() {
  if (state.harmonizedCache) return state.harmonizedCache;
  const src = state.cutout;
  const amt = parseInt($('harmonize').value, 10) / 100;
  const grain = parseInt($('grain').value, 10) / 100;
  if ((amt <= 0 && grain <= 0) || !state.photo) { state.harmonizedCache = src; return src; }
  let cdata = src.getContext('2d').getImageData(0, 0, src.width, src.height);
  if (amt > 0) {
    const charStats = labStats(cdata, 1);
    const sceneStats = labStats(state.photo.imgData, 2);
    const tf = makeLabTransform(charStats, sceneStats, 'chroma', amt);
    cdata = applyTransfer(cdata, tf); // 仅改 RGB，保留 alpha
  }
  if (grain > 0) addLumaGrain(cdata, grain * 18);
  const oc = document.createElement('canvas');
  oc.width = src.width; oc.height = src.height;
  oc.getContext('2d').putImageData(cdata, 0, 0);
  state.harmonizedCache = oc;
  return oc;
}

// 按 charPos（中心归一化坐标）+ 角色大小滑块绘制角色，并记录绘制矩形供拖拽命中。
// 全部用相对坐标，因此预览画布和全分辨率导出画布走同一函数。
// record=false 时不更新拖拽命中矩形（导出时用，避免污染预览坐标）。
function compositeCharacter(canvas, record = true) {
  const ctx = canvas.getContext('2d');
  const ch = harmonizedCutout();
  const scalePct = parseInt($('charScale').value, 10) / 100;
  const dh = canvas.height * scalePct;
  const scale = dh / ch.height;
  const dw = ch.width * scale;
  const dx = state.charPos.cx * canvas.width - dw / 2;
  const dy = state.charPos.cy * canvas.height - dh / 2;
  // 落地软阴影：贴着角色脚底的横椭圆，径向渐变淡出
  const shadowAmt = parseInt($('shadow').value, 10) / 100;
  if (shadowAmt > 0) {
    const rx = dw * 0.44, ry = Math.max(4, dw * 0.10);
    const offset = parseInt($('shadowOffset').value, 10) / 100;
    const cx = dx + dw / 2 + dw * offset * 0.55, cy = dy + dh - ry * 0.5;
    ctx.save();
    ctx.translate(cx, cy); ctx.scale(1, ry / rx);
    const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
    grad.addColorStop(0, `rgba(0,0,0,${(shadowAmt * 0.8).toFixed(3)})`);
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(0, 0, rx, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }
  ctx.drawImage(ch, dx, dy, dw, dh);
  if (record) state.charDraw = { dx, dy, dw, dh };
}

// 抠图后/调参后：从 rawAlpha 重建清理过的角色 cutout
function applyRefine(resetPos) {
  if (!state.rawAlpha || !state.anime) return;
  const w = state.rawW, h = state.rawH;
  const thr = parseInt($('maskThr').value, 10);
  const erode = parseInt($('maskErode').value, 10);
  const clean = cleanupAlpha(state.rawAlpha, w, h, { thr, erode, featherR: 2 });
  const { bbox, coverage } = alphaBBox(clean, w, h);
  state.cutout = bbox ? cutoutCanvas(state.anime.imgData, { alpha: clean, bbox }) : null;
  if (state.cutout) {
    // 去色渗：羽化边缘的 RGB 混有动画背景色，向内部实心像素取色修正
    const ctx = state.cutout.getContext('2d');
    const cimg = ctx.getImageData(0, 0, state.cutout.width, state.cutout.height);
    decontaminateEdges(cimg);
    ctx.putImageData(cimg, 0, 0);
  }
  invalidateHarmonize();
  if (resetPos) state.charPos = { cx: 0.5, cy: 0.62 };
  redrawComposite();
  return coverage;
}

// ---------- 文件上传绑定 ----------
function bindDrop(dropId, inputId, thumbId, onLoad) {
  const drop = $(dropId), input = $(inputId), thumb = $(thumbId);
  const handle = async (file) => {
    if (!file) return;
    const imageExt = /\.(jpe?g|png|webp|heic|heif)$/i.test(file.name || '');
    if (!file.type.startsWith('image/') && !imageExt) { setStatus('请选择 JPEG、PNG、WebP 或 HEIC 图片'); return; }
    try {
      setStatus(isHeicFile(file) ? '使用系统解码 HEIC 并校正方向…' : '读取图片并校正 EXIF 方向…');
      const data = await fileToImageData(file);
      thumb.src = data.url; thumb.hidden = false;
      await onLoad(data);
    } catch (e) { setStatus('读取失败：' + (e.message || e)); }
  };
  drop.addEventListener('click', () => input.click());
  input.addEventListener('change', (e) => handle(e.target.files[0]));
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('dragover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault(); drop.classList.remove('dragover');
    handle(e.dataTransfer.files[0]);
  });
}

async function handleAnimeData(data) {
  if (state.anime?.srcUrl?.startsWith('blob:')) URL.revokeObjectURL(state.anime.srcUrl);
  state.anime = {
    imgData: data.imgData, width: data.width, height: data.height, srcUrl: data.url,
  };
  state.gradeCache = null;
  renderPalette(extractPalette(data.imgData, 6, 4, { ignoreBottomRatio: $('ignoreSub').checked ? 0.12 : 0 }));
  state.cutout = null; state.rawAlpha = null; setCharSeg(null); invalidateHarmonize();
  $('btnExtract').disabled = false;
  $('btnExtractAI').disabled = false;
  $('btnLasso').disabled = false;
  $('extractStatus').textContent = '点击「扒取人物」试试';
  $('btnAlign').disabled = !state.photo;
  recompute();
}

async function handlePhotoData(data) {
  if (state.photo?.srcUrl?.startsWith('blob:')) URL.revokeObjectURL(state.photo.srcUrl);
  state.photo = {
    imgData: data.imgData, width: data.width, height: data.height, srcUrl: data.url, align: null,
    originalWidth: data.originalWidth || data.width, originalHeight: data.originalHeight || data.height,
    fileName: data.fileName || '',
  };
  state.gradeCache = null;
  exitAlignMode(false);
  $('btnAlign').disabled = !state.anime;
  setStatus(`照片已读取 · 原图 ${state.photo.originalWidth}×${state.photo.originalHeight} · 预览 ${data.width}×${data.height}`);
  recompute();
}

bindDrop('dropAnime', 'fileAnime', 'thumbAnime', handleAnimeData);
bindDrop('dropPhoto', 'filePhoto', 'thumbPhoto', handlePhotoData);

// ---------- 控件 ----------
$('strength').addEventListener('input', (e) => {
  $('strengthVal').textContent = e.target.value + '%';
  scheduleRecompute();
});
$('mode').addEventListener('change', recompute);
$('skyRegion').addEventListener('change', recompute);
$('ignoreSub').addEventListener('change', () => {
  state.gradeCache = null;
  if (state.anime) renderPalette(extractPalette(state.anime.imgData, 6, 4, { ignoreBottomRatio: $('ignoreSub').checked ? 0.12 : 0 }));
  recompute();
});
$('satBoost').addEventListener('input', (e) => { $('satBoostVal').textContent = e.target.value + '%'; scheduleRecompute(); });
$('bloom').addEventListener('input', (e) => { $('bloomVal').textContent = e.target.value + '%'; scheduleRecompute(); });

$('composite').addEventListener('change', redrawComposite);

$('harmonize').addEventListener('input', (e) => {
  $('harmonizeVal').textContent = e.target.value + '%';
  invalidateHarmonize();
  redrawComposite();
});
$('shadow').addEventListener('input', (e) => {
  $('shadowVal').textContent = e.target.value + '%';
  redrawComposite();
});
$('shadowOffset').addEventListener('input', (e) => {
  $('shadowOffsetVal').textContent = e.target.value + '%';
  redrawComposite();
});
$('grain').addEventListener('input', (e) => {
  $('grainVal').textContent = e.target.value + '%';
  invalidateHarmonize();
  redrawComposite();
});

$('charScale').addEventListener('input', (e) => {
  $('charScaleVal').textContent = e.target.value + '%';
  redrawComposite();
});

// 抠图阈值 / 收边：从 rawAlpha 重算，无需重新推理
$('maskThr').addEventListener('input', (e) => {
  $('maskThrVal').textContent = e.target.value;
  const cov = applyRefine(false);
  if (cov != null) $('extractStatus').textContent = `调整中 · 占画面 ${(cov * 100).toFixed(0)}%`;
});
$('maskErode').addEventListener('input', (e) => {
  $('maskErodeVal').textContent = e.target.value + 'px';
  const cov = applyRefine(false);
  if (cov != null) $('extractStatus').textContent = `调整中 · 占画面 ${(cov * 100).toFixed(0)}%`;
});

// ---------- AI 检测流水线：角色勾选 ----------
function setCharSeg(seg) {
  state.charSeg = seg;
  const box = $('charChips');
  box.innerHTML = '';
  box.hidden = !seg;
  if (!seg) return;
  seg.chars.forEach((c, i) => {
    const chip = document.createElement('label');
    chip.className = 'char-chip' + (c.empty ? ' empty' : '');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = seg.included.has(i);
    cb.disabled = c.empty;
    cb.addEventListener('change', () => {
      if (cb.checked) seg.included.add(i); else seg.included.delete(i);
      applyCharSelection(false);
    });
    const text = document.createElement('span');
    const name = c.manual ? `圈选${i + 1}` : `角色${i + 1}`;
    text.textContent = c.empty
      ? `${name} · 未抠出`
      : `${name}${c.manual ? '' : ` · ${(c.score * 100).toFixed(0)}%`}${c.via === 'sam' ? ' · SAM兜底' : ''}`;
    chip.appendChild(cb); chip.appendChild(text);
    box.appendChild(chip);
  });
}

// 按勾选合并各角色 mask -> rawAlpha -> 走统一的清理/重绘
function applyCharSelection(resetPos) {
  const seg = state.charSeg;
  if (!seg || !state.anime) return;
  state.rawAlpha = mergeCharacterAlphas(seg.chars, state.anime.width, state.anime.height, seg.included);
  state.rawW = state.anime.width; state.rawH = state.anime.height;
  const cov = applyRefine(resetPos);
  return cov;
}

function extractCharactersInWorker(imageData, opts = {}) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('./ai-worker.js', { type: 'module', name: 'seichi-ai-once' });
    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'progress') opts.onProgress?.(msg.received, msg.total);
      else if (msg.type === 'stage') opts.onStage?.(msg.text);
      else if (msg.type === 'done') { worker.terminate(); resolve({ seg: msg.seg, whole: msg.whole }); }
      else if (msg.type === 'error') {
        worker.terminate();
        const error = new Error(msg.error?.message || 'AI Worker 失败'); error.name = msg.error?.name || 'Error'; error.stack = msg.error?.stack || error.stack;
        reject(error);
      }
    };
    worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'AI Worker 加载失败')); };
    worker.postMessage({ imageData, hires: !!opts.hires, samFallback: opts.samFallback !== false, mobileModel: !!opts.mobileModel });
  });
}

$('btnExtract').addEventListener('click', () => {
  if (!state.anime) return;
  $('extractStatus').textContent = '分析中…';
  // 让状态文字先渲染
  setTimeout(() => {
    const t0 = performance.now();
    setCharSeg(null);
    const res = extractForeground(state.anime.imgData, { centerBias: 0.5 });
    state.rawAlpha = res.alpha; state.rawW = res.width; state.rawH = res.height;
    const cov = applyRefine(true);
    const ms = Math.round(performance.now() - t0);
    $('extractStatus').textContent = state.cutout
      ? `已提取 · 占画面 ${(cov * 100).toFixed(0)}% · ${ms}ms · 可调阈值/收边/拖拽`
      : '未找到明显前景，换张图试试';
  }, 30);
});

$('btnExtractAI').addEventListener('click', async () => {
  if (!state.anime) return;
  const btn = $('btnExtractAI');
  state.aiBusy = true;
  btn.disabled = true; $('btnExtract').disabled = true;
  $('btnExportImg').disabled = true;
  const onProgress = (recv, total) => {
    $('extractStatus').textContent = `下载模型 ${(recv / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)}MB`;
  };
  const onStage = (s) => { $('extractStatus').textContent = s; };
  try {
    const t0 = performance.now();
    // 两级流水线：先检测角色框，再逐框抠图
    const runAI = () => DEVICE.isAppleMobile
      ? extractCharactersInWorker(state.anime.imgData, { hires: $('hiresDet').checked, samFallback: $('mobileSam').checked, mobileModel: true, onProgress, onStage })
      : extractCharactersAI(state.anime.imgData, { hires: $('hiresDet').checked, onProgress, onStage }).then((seg) => ({ seg, whole: null }));
    let aiResult;
    try {
      aiResult = await runAI();
    } catch (firstError) {
      // Safari/WASM 在内存压力后偶尔留下不可用会话：只自动重建一次，避免死循环。
      onStage('释放模型内存并重试一次…');
      await releaseAllSessions();
      try { aiResult = await runAI(); }
      catch (retryError) { retryError.cause = firstError; throw retryError; }
    }
    const { seg, whole } = aiResult;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);

    if (seg.chars.length) {
      const included = new Set(seg.chars.map((c, i) => (c.empty ? -1 : i)).filter((i) => i >= 0));
      setCharSeg({ chars: seg.chars, included });
      applyCharSelection(true);
      const nEmpty = seg.chars.filter((c) => c.empty).length;
      const nSam = seg.chars.filter((c) => c.via === 'sam').length;
      $('extractStatus').textContent =
        `检测到 ${seg.chars.length} 个角色` +
        (nSam ? `（${nSam} 个由 SAM 兜底）` : '') +
        (nEmpty ? `（${nEmpty} 个太小未抠出${DEVICE.isAppleMobile && !$('mobileSam').checked ? '，可开启高质量 SAM' : $('hiresDet').checked ? '' : '，可试远景小人模式'}）` : '') +
        ` · ${secs}s · 可勾选角色/调阈值/拖拽`;
    } else {
      // 没检测到角色：回退整图直抠（老行为），风景图会正确报"未找到"
      setCharSeg(null);
      onStage('未检测到角色，改用整图抠取…');
      const res = whole || await extractForegroundAI(state.anime.imgData, { onProgress, onStage });
      state.rawAlpha = res.alpha; state.rawW = res.width; state.rawH = res.height;
      const cov = applyRefine(true);
      $('extractStatus').textContent = state.cutout
        ? `未检测到角色 · 整图抠取 占画面 ${(cov * 100).toFixed(0)}%`
        : '未检测到角色，也没有明显前景（纯风景图？）';
    }
  } catch (e) {
    console.error(e);
    $('extractStatus').textContent = 'AI 抠图失败：' + (e.message || e);
  } finally {
    // iOS/iPadOS 更容易被 Jetsam；结果已复制成普通数组后释放 Session，
    // 下次从 Cache Storage 重建，换取连续处理多张时的稳定性。
    if (DEVICE.isAppleMobile) await releaseAllSessions();
    state.aiBusy = false;
    btn.disabled = false; $('btnExtract').disabled = false;
    if (state.gradedData) $('btnExportImg').disabled = false;
    updateModelCacheStatus();
  }
});

// ---------- 构图对齐（洋葱皮）----------
// 动画截图半透明叠在照片上，拖拽平移 + 滑杆缩放，把照片裁到和动画同构图同宽高比。
// 裁剪框以原图归一化坐标存进 state.photo.align，预览和全分辨率导出共用。
const alignState = {
  active: false,
  bitmap: null,          // 原始照片全画幅的处理分辨率副本（canvas）
  bw: 0, bh: 0,          // bitmap 尺寸
  crop: { cx: 0.5, cy: 0.5, scale: 1 }, // scale = 裁剪宽 / 照片宽
  maxScale: 1,
};

function alignAspect() { return state.anime.width / state.anime.height; }

// 当前裁剪框（bitmap 像素坐标）
function alignCropRect() {
  const A = alignAspect();
  const wc = alignState.crop.scale * alignState.bw;
  const hc = wc / A;
  let x = alignState.crop.cx * alignState.bw - wc / 2;
  let y = alignState.crop.cy * alignState.bh - hc / 2;
  x = Math.max(0, Math.min(alignState.bw - wc, x));
  y = Math.max(0, Math.min(alignState.bh - hc, y));
  // 回写钳位后的中心，避免边缘"卡住"后中心漂移
  alignState.crop.cx = (x + wc / 2) / alignState.bw;
  alignState.crop.cy = (y + hc / 2) / alignState.bh;
  return { x, y, w: wc, h: hc };
}

function drawAlignPreview() {
  const g = $('canvasGraded');
  g.width = state.anime.width; g.height = state.anime.height;
  const ctx = g.getContext('2d');
  const r = alignCropRect();
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, g.width, g.height);
  ctx.drawImage(alignState.bitmap, r.x, r.y, r.w, r.h, 0, 0, g.width, g.height);
  syncCanvasSize();
}

async function enterAlignMode() {
  if (!state.anime || !state.photo || !state.photo.srcUrl) return;
  setStatus('对齐模式：拖拽照片、滑杆缩放，虚影为动画构图');
  // 始终基于原始全画幅照片对齐（可反复调整，不叠加裁剪）
  const img = new Image();
  img.src = state.photo.srcUrl;
  await img.decode();
  const sc = Math.min(1, MAX_DIM / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement('canvas');
  c.width = Math.round(img.naturalWidth * sc); c.height = Math.round(img.naturalHeight * sc);
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  alignState.bitmap = c; alignState.bw = c.width; alignState.bh = c.height;
  const A = alignAspect();
  alignState.maxScale = Math.min(1, A * c.height / c.width);
  const al = state.photo.align;
  if (al) { // 已有裁剪：从上次的框恢复
    alignState.crop = { cx: al.x + al.w / 2, cy: al.y + al.h / 2, scale: al.w };
  } else {
    alignState.crop = { cx: 0.5, cy: 0.5, scale: alignState.maxScale };
  }
  $('alignZoom').value = Math.round(alignState.maxScale / alignState.crop.scale * 100);
  $('alignZoomVal').textContent = $('alignZoom').value + '%';
  alignState.active = true;
  $('compare').classList.add('align-on');
  $('alignBar').hidden = false;
  $('btnAlign').textContent = '退出对齐';
  const ghost = $('alignGhost');
  ghost.src = $('thumbAnime').src;
  ghost.hidden = false;
  $('btnAlignReset').hidden = !state.photo.align;
  drawAlignPreview();
}

function exitAlignMode(redraw = true) {
  if (!alignState.active) { if ($('alignBar')) $('alignBar').hidden = true; return; }
  alignState.active = false;
  alignState.bitmap = null;
  $('compare').classList.remove('align-on');
  $('alignBar').hidden = true;
  $('alignGhost').hidden = true;
  $('btnAlign').textContent = '构图对齐';
  if (redraw && state.gradedData) { redrawComposite(); drawToCanvas($('canvasOrig'), state.photo.imgData); syncCanvasSize(); }
}

// 应用裁剪：从原始文件按裁剪框重取处理分辨率图，替换 state.photo
async function applyAlignCrop() {
  const r = alignCropRect();
  const norm = { x: r.x / alignState.bw, y: r.y / alignState.bh, w: r.w / alignState.bw, h: r.h / alignState.bh };
  setStatus('应用构图裁剪…');
  const img = new Image();
  img.src = state.photo.srcUrl;
  await img.decode();
  const nw = img.naturalWidth, nh = img.naturalHeight;
  const sx = norm.x * nw, sy = norm.y * nh, sw = norm.w * nw, sh = norm.h * nh;
  const sc = Math.min(1, MAX_DIM / Math.max(sw, sh));
  const c = document.createElement('canvas');
  c.width = Math.round(sw * sc); c.height = Math.round(sh * sc);
  c.getContext('2d').drawImage(img, sx, sy, sw, sh, 0, 0, c.width, c.height);
  state.photo.imgData = c.getContext('2d').getImageData(0, 0, c.width, c.height);
  state.photo.width = c.width; state.photo.height = c.height;
  state.photo.align = norm;
  state.gradeCache = null;
  invalidateHarmonize();
  exitAlignMode(false);
  recompute();
  setStatus(`构图已对齐 · 裁剪 ${Math.round(norm.w * 100)}% 画幅 · 导出仍为全分辨率`);
}

// 重置：恢复整幅照片
async function resetAlignCrop() {
  const img = new Image();
  img.src = state.photo.srcUrl;
  await img.decode();
  const sc = Math.min(1, MAX_DIM / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement('canvas');
  c.width = Math.round(img.naturalWidth * sc); c.height = Math.round(img.naturalHeight * sc);
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  state.photo.imgData = c.getContext('2d').getImageData(0, 0, c.width, c.height);
  state.photo.width = c.width; state.photo.height = c.height;
  state.photo.align = null;
  state.gradeCache = null;
  invalidateHarmonize();
  exitAlignMode(false);
  recompute();
  setStatus('已恢复整幅照片');
}

$('btnAlign').addEventListener('click', () => { alignState.active ? exitAlignMode() : enterAlignMode(); });
$('btnAlignApply').addEventListener('click', applyAlignCrop);
$('btnAlignCancel').addEventListener('click', () => exitAlignMode());
$('btnAlignReset').addEventListener('click', resetAlignCrop);
$('alignZoom').addEventListener('input', (e) => {
  $('alignZoomVal').textContent = e.target.value + '%';
  if (!alignState.active) return;
  alignState.crop.scale = alignState.maxScale / (parseInt(e.target.value, 10) / 100);
  drawAlignPreview();
});
$('alignOpacity').addEventListener('input', (e) => {
  $('alignOpacityVal').textContent = e.target.value + '%';
  $('alignGhost').style.opacity = String(parseInt(e.target.value, 10) / 100);
});

// ---------- 对比滑块 + 角色拖拽 ----------
(function setupInteract() {
  const compare = $('compare'), clip = $('clip'), handle = $('handle');
  const gradedCanvas = $('canvasGraded');
  let sliderDrag = false, charDrag = false, grabDX = 0, grabDY = 0;
  let alignDrag = false, alignStart = null;

  const moveSlider = (clientX) => {
    const rect = compare.getBoundingClientRect();
    let pct = ((clientX - rect.left) / rect.width) * 100;
    pct = Math.max(0, Math.min(100, pct));
    clip.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    handle.style.left = pct + '%';
  };

  // 屏幕坐标 -> 调色 canvas 内部坐标
  const toCanvas = (clientX, clientY) => {
    const r = gradedCanvas.getBoundingClientRect();
    return {
      x: (clientX - r.left) / r.width * gradedCanvas.width,
      y: (clientY - r.top) / r.height * gradedCanvas.height,
      inDisplay: clientX >= r.left && clientX <= r.right && clientY >= r.top && clientY <= r.bottom,
    };
  };
  const hitChar = (cx, cy) => {
    const d = state.charDraw;
    return d && cx >= d.dx && cx <= d.dx + d.dw && cy >= d.dy && cy <= d.dy + d.dh;
  };

  handle.addEventListener('pointerdown', (e) => { sliderDrag = true; e.stopPropagation(); });

  compare.addEventListener('pointerdown', (e) => {
    const p = toCanvas(e.clientX, e.clientY);
    if (alignState.active) {
      alignDrag = true;
      alignStart = { x: p.x, y: p.y, cx: alignState.crop.cx, cy: alignState.crop.cy };
      compare.classList.add('grabbing');
      return;
    }
    if (state.cutout && $('composite').checked && p.inDisplay && hitChar(p.x, p.y)) {
      charDrag = true;
      grabDX = p.x - (state.charPos.cx * gradedCanvas.width);
      grabDY = p.y - (state.charPos.cy * gradedCanvas.height);
      compare.classList.add('grabbing');
    } else {
      moveSlider(e.clientX); sliderDrag = true;
    }
  });

  window.addEventListener('pointermove', (e) => {
    if (alignDrag) {
      const p = toCanvas(e.clientX, e.clientY);
      // 画布 1px = 裁剪框宽/画布宽 的照片像素；拖照片方向与拖裁剪框相反
      const r = alignState.crop.scale * alignState.bw / gradedCanvas.width;
      alignState.crop.cx = alignStart.cx - (p.x - alignStart.x) * r / alignState.bw;
      alignState.crop.cy = alignStart.cy - (p.y - alignStart.y) * r / alignState.bh;
      drawAlignPreview();
      return;
    }
    if (charDrag) {
      const p = toCanvas(e.clientX, e.clientY);
      state.charPos.cx = Math.max(0, Math.min(1, (p.x - grabDX) / gradedCanvas.width));
      state.charPos.cy = Math.max(0, Math.min(1, (p.y - grabDY) / gradedCanvas.height));
      redrawComposite();
    } else if (sliderDrag) {
      moveSlider(e.clientX);
    }
  });
  window.addEventListener('pointerup', () => {
    sliderDrag = false; charDrag = false; alignDrag = false; compare.classList.remove('grabbing');
  });
})();

window.addEventListener('resize', () => { if (state.photo) syncCanvasSize(); });

// ---------- 导出 ----------
function download(blobOrUrl, name) {
  const a = document.createElement('a');
  a.href = blobOrUrl; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}

// 全分辨率导出：最终画布保持原始像素，逐块重放调色，避免同时持有整张照片的
// 多份 ImageData/Float32Array。48MP 桌面照片可保持 8000×6000；低内存手机会
// 明确降级而不是直接 OOM。Bloom 在 1/4 全画幅上统一生成，分块之间没有接缝。

function maskCanvasFromWeight(weight, w, h) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const d = new ImageData(w, h);
  for (let p = 0, i = 0; p < weight.length; p++, i += 4) {
    const v = Math.round(weight[p] * 255);
    d.data[i] = d.data[i + 1] = d.data[i + 2] = v; d.data[i + 3] = 255;
  }
  c.getContext('2d').putImageData(d, 0, 0); return c;
}

function weightForTile(maskCanvas, x, y, w, h, fullW, fullH) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d'); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  const sx = x / fullW * maskCanvas.width, sy = y / fullH * maskCanvas.height;
  const sw = w / fullW * maskCanvas.width, sh = h / fullH * maskCanvas.height;
  ctx.drawImage(maskCanvas, sx, sy, sw, sh, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data, out = new Float32Array(w * h);
  for (let p = 0, i = 0; p < out.length; p++, i += 4) out[p] = rgba[i] / 255;
  return out;
}

function applyBloomToCanvas(canvas, gain) {
  if (!(gain > 0)) return;
  const bw = Math.max(1, Math.round(canvas.width / 4)), bh = Math.max(1, Math.round(canvas.height / 4));
  const small = document.createElement('canvas'); small.width = bw; small.height = bh;
  const sctx = small.getContext('2d'); sctx.imageSmoothingEnabled = true; sctx.imageSmoothingQuality = 'high';
  sctx.drawImage(canvas, 0, 0, bw, bh);
  const layer = makeBloomLayer(sctx.getImageData(0, 0, bw, bh));
  sctx.putImageData(layer, 0, 0);
  const ctx = canvas.getContext('2d'); ctx.save();
  ctx.globalCompositeOperation = 'screen'; ctx.globalAlpha = gain;
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
  ctx.restore();
}

async function renderFullRes(onStage, maxPixels = EXPORT_MAX_PIXELS) {
  const img = new Image();
  img.src = state.photo.srcUrl;
  await img.decode();
  let sx = 0, sy = 0, sw = img.naturalWidth, sh = img.naturalHeight;
  const al = state.photo.align;
  if (al) { sx = al.x * sw; sy = al.y * sh; sw *= al.w; sh *= al.h; }
  const scale = Math.min(1, EXPORT_MAX_SIDE / Math.max(sw, sh), Math.sqrt(maxPixels / (sw * sh)));
  const W = Math.round(sw * scale), H = Math.round(sh * scale);
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  if (!ctx) throw new Error('浏览器无法创建该尺寸画布，请改用桌面 Chrome 或减小照片');
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, W, H);
  const cache = getGradeCache();
  const t = buildTransforms(cache);
  const useRegion = $('skyRegion').checked && cache.region;
  const maskCanvas = useRegion ? maskCanvasFromWeight(cache.photoSky.weight, state.photo.width, state.photo.height) : null;
  const tilesX = Math.ceil(W / EXPORT_TILE), tilesY = Math.ceil(H / EXPORT_TILE), total = tilesX * tilesY;
  let done = 0;
  if (scale < .999) onStage && onStage(`设备内存保护：将 ${Math.round(sw)}×${Math.round(sh)} 降为 ${W}×${H}`);
  else onStage && onStage(`全分辨率分块调色 ${W}×${H}…`);
  await new Promise((r) => setTimeout(r, 30));
  for (let y = 0; y < H; y += EXPORT_TILE) {
    const th = Math.min(EXPORT_TILE, H - y);
    for (let x = 0; x < W; x += EXPORT_TILE) {
      const tw = Math.min(EXPORT_TILE, W - x), tile = ctx.getImageData(x, y, tw, th);
      const out = useRegion
        ? applyTransferRegioned(tile, t.tSky, t.tLand, weightForTile(maskCanvas, x, y, tw, th, W, H))
        : applyTransfer(tile, t.global);
      ctx.putImageData(out, x, y); done++;
    }
    onStage && onStage(`全分辨率调色 ${done}/${total} · ${W}×${H}`);
    await new Promise((r) => requestAnimationFrame(r));
  }
  onStage && onStage('生成全画幅辉光…');
  await new Promise((r) => requestAnimationFrame(r));
  applyBloomToCanvas(c, parseInt($('bloom').value, 10) / 100);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  if (state.cutout && $('composite').checked) compositeCharacter(c, false);
  c.dataset.originalWidth = Math.round(sw); c.dataset.originalHeight = Math.round(sh);
  c.dataset.wasDownscaled = scale < .999 ? '1' : '0';
  return c;
}

$('btnExportImg').addEventListener('click', async () => {
  const btn = $('btnExportImg');
  btn.disabled = true;
  try {
    if (state.photo && state.photo.srcUrl) {
      if (DEVICE.isAppleMobile) {
        setStatus('释放 AI 模型内存，为导出腾出空间…');
        await releaseAllSessions();
      }
      let maxPixels = EXPORT_MAX_PIXELS, c = null, blob = null, lastError = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          setStatus(attempt ? `降低导出尺寸后重试（${attempt}/2）…` : '读取原始分辨率…');
          c = await renderFullRes(setStatus, maxPixels);
          blob = await new Promise((r) => c.toBlob(r, 'image/jpeg', 0.92));
          if (!blob) throw new Error('浏览器无法编码此尺寸的 JPEG');
          break;
        } catch (e) {
          lastError = e;
          c = null; blob = null;
          if (!IS_MOBILE || attempt === 2) throw e;
          maxPixels = Math.max(2_000_000, Math.floor(maxPixels * 0.62));
          await new Promise((r) => setTimeout(r, 80));
        }
      }
      if (!blob || !c) throw lastError || new Error('导出失败');
      state.lastExport = { blob, name: 'seichi-graded.jpg', width: c.width, height: c.height };
      const shareFile = new File([blob], state.lastExport.name, { type: 'image/jpeg' });
      const canShareFile = typeof navigator.share === 'function' && typeof navigator.canShare === 'function' && navigator.canShare({ files: [shareFile] });
      $('btnShareImg').hidden = !canShareFile;
      const sizeText = c.dataset.wasDownscaled === '1'
        ? `原裁剪 ${c.dataset.originalWidth}×${c.dataset.originalHeight}，生成 ${c.width}×${c.height}`
        : `生成 ${c.width}×${c.height} 原始分辨率`;
      if (canShareFile && DEVICE.isAppleMobile) {
        setStatus(`${sizeText} JPEG · 请点“保存到照片 / 分享”`);
      } else {
        const url = URL.createObjectURL(blob);
        download(url, state.lastExport.name);
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        setStatus(`${sizeText} JPEG`);
      }
    } else {
      // 没有原始文件引用（不应发生）：退回导出预览画布
      const c = document.createElement('canvas');
      c.width = $('canvasGraded').width; c.height = $('canvasGraded').height;
      c.getContext('2d').drawImage($('canvasGraded'), 0, 0);
      download(c.toDataURL('image/png'), 'seichi-graded.png');
    }
  } catch (e) {
    console.error(e);
    setStatus('导出失败：' + (e.message || e));
  } finally {
    btn.disabled = false;
  }
});

$('btnShareImg').addEventListener('click', async () => {
  const exp = state.lastExport;
  if (!exp) { setStatus('请先生成调色图'); return; }
  const file = new File([exp.blob], exp.name, { type: 'image/jpeg' });
  try {
    if (navigator.canShare?.({ files: [file] })) {
      await navigator.share({ files: [file], title: '圣地巡礼调色图' });
      setStatus(`已打开系统分享 · ${exp.width}×${exp.height}`);
    } else {
      const url = URL.createObjectURL(exp.blob); download(url, exp.name);
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      setStatus('浏览器不支持文件分享，已改为下载 JPEG');
    }
  } catch (e) {
    if (e?.name !== 'AbortError') setStatus('分享失败：' + (e.message || e));
  }
});

$('btnExportLut').addEventListener('click', () => {
  const btn = $('btnExportLut'); btn.disabled = true;
  setStatus('正在生成 65³ 高精度 LUT…');
  // 先让状态文字绘制出来，再执行数十万采样点的同步烘焙。
  setTimeout(() => {
    try {
      const cube = generateCubeLUT(state.transform, 65, 'Seichi Grade');
      const blob = new Blob([cube], { type: 'text/plain' }), url = URL.createObjectURL(blob);
      download(url, 'seichi-grade.cube');
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setStatus('LUT 已生成 · 65³ · 不含天空分区与辉光');
    } finally { btn.disabled = false; }
  }, 30);
});

$('btnExportCompare').addEventListener('click', () => {
  const layout = $('layout').value;
  const anime = state.anime, gw = state.gradedData.width, gh = state.gradedData.height;
  // 临时画布把各图绘制出来
  const tmpGraded = document.createElement('canvas');
  tmpGraded.width = gw; tmpGraded.height = gh;
  tmpGraded.getContext('2d').drawImage($('canvasGraded'), 0, 0);
  const tmpOrig = document.createElement('canvas');
  tmpOrig.width = state.photo.width; tmpOrig.height = state.photo.height;
  tmpOrig.getContext('2d').putImageData(state.photo.imgData, 0, 0);
  const tmpAnime = document.createElement('canvas');
  tmpAnime.width = anime.width; tmpAnime.height = anime.height;
  tmpAnime.getContext('2d').putImageData(anime.imgData, 0, 0);

  const gap = 8;
  const out = document.createElement('canvas');
  const ctx = out.getContext('2d');
  // 以调色图宽度为基准统一缩放
  const W = gw;
  const fit = (cv) => ({ cv, h: Math.round(cv.height * (W / cv.width)) });

  let panels;
  if (layout === 'triple') panels = [fit(tmpAnime), fit(tmpOrig), fit(tmpGraded)];
  else panels = [fit(tmpAnime), fit(tmpGraded)];

  if (layout === 'leftright') {
    const H = Math.max(...panels.map(p => p.h));
    out.width = W * panels.length + gap * (panels.length - 1);
    out.height = H;
    ctx.fillStyle = '#0e1014'; ctx.fillRect(0, 0, out.width, out.height);
    panels.forEach((p, i) => ctx.drawImage(p.cv, i * (W + gap), 0, W, p.h));
  } else { // updown / triple 纵向
    out.width = W;
    out.height = panels.reduce((s, p) => s + p.h, 0) + gap * (panels.length - 1);
    ctx.fillStyle = '#0e1014'; ctx.fillRect(0, 0, out.width, out.height);
    let y = 0;
    panels.forEach((p) => { ctx.drawImage(p.cv, 0, y, W, p.h); y += p.h + gap; });
  }
  download(out.toDataURL('image/png'), 'seichi-compare.png');
});

// ---------- 参数自动恢复与风格预设 ----------
const SETTINGS_KEY = 'seichi-current-settings-v1';
const PRESETS_KEY = 'seichi-style-presets-v1';
const SETTING_IDS = [
  'mode', 'skyRegion', 'ignoreSub', 'strength', 'satBoost', 'bloom', 'composite',
  'hiresDet', 'mobileSam', 'maskThr', 'maskErode', 'charScale', 'harmonize', 'shadow',
  'shadowOffset', 'grain', 'layout',
];

function captureSettings() {
  const values = {};
  for (const id of SETTING_IDS) {
    const el = $(id);
    values[id] = el.type === 'checkbox' ? el.checked : el.value;
  }
  return { version: 1, values, charPos: { ...state.charPos } };
}

function syncControlLabels() {
  const pairs = {
    strength: ['strengthVal', '%'], satBoost: ['satBoostVal', '%'], bloom: ['bloomVal', '%'],
    maskThr: ['maskThrVal', ''], maskErode: ['maskErodeVal', 'px'], charScale: ['charScaleVal', '%'],
    harmonize: ['harmonizeVal', '%'], shadow: ['shadowVal', '%'],
    shadowOffset: ['shadowOffsetVal', '%'], grain: ['grainVal', '%'],
  };
  for (const [id, [label, suffix]] of Object.entries(pairs)) $(label).textContent = $(id).value + suffix;
}

function applySettings(saved, rerender = true) {
  if (!saved || !saved.values) return false;
  for (const [id, value] of Object.entries(saved.values)) {
    const el = $(id);
    if (!el || !SETTING_IDS.includes(id)) continue;
    if (el.type === 'checkbox') el.checked = Boolean(value);
    else el.value = String(value);
  }
  if (saved.charPos && Number.isFinite(saved.charPos.cx) && Number.isFinite(saved.charPos.cy)) {
    state.charPos = { cx: saved.charPos.cx, cy: saved.charPos.cy };
  }
  syncControlLabels();
  invalidateHarmonize(); state.gradeCache = null;
  if (rerender) recompute();
  return true;
}

function readPresets() {
  try { return JSON.parse(localStorage.getItem(PRESETS_KEY) || '{}'); } catch { return {}; }
}

function refreshPresetList(selected = '') {
  const select = $('presetSelect'), presets = readPresets();
  select.innerHTML = '<option value="">选择预设…</option>';
  for (const name of Object.keys(presets).sort((a, b) => a.localeCompare(b, 'zh-CN'))) {
    const option = document.createElement('option'); option.value = name; option.textContent = name;
    select.appendChild(option);
  }
  select.value = selected && presets[selected] ? selected : '';
  $('btnPresetDelete').disabled = !select.value;
}

let saveSettingsTimer = 0;
function queueSettingsSave() {
  clearTimeout(saveSettingsTimer);
  saveSettingsTimer = setTimeout(() => {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(captureSettings())); } catch { /* 隐私模式可禁用 */ }
  }, 180);
}

for (const id of SETTING_IDS) {
  $(id).addEventListener('input', queueSettingsSave);
  $(id).addEventListener('change', queueSettingsSave);
}
$('compare').addEventListener('pointerup', queueSettingsSave);
window.addEventListener('pagehide', () => {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(captureSettings())); } catch { /* 忽略 */ }
  releaseAllSessions();
});
$('presetSelect').addEventListener('change', (e) => { $('btnPresetDelete').disabled = !e.target.value; });
$('btnPresetApply').addEventListener('click', () => {
  const name = $('presetSelect').value, preset = readPresets()[name];
  if (!preset) { setStatus('请先选择一个预设'); return; }
  applySettings(preset); queueSettingsSave(); setStatus(`已应用风格预设「${name}」`);
});
$('btnPresetSave').addEventListener('click', () => {
  const name = $('presetName').value.trim();
  if (!name) { setStatus('请先给预设起一个名字'); return; }
  const presets = readPresets(); presets[name] = captureSettings();
  try { localStorage.setItem(PRESETS_KEY, JSON.stringify(presets)); }
  catch { setStatus('浏览器禁止本地存储，预设未保存'); return; }
  $('presetName').value = ''; refreshPresetList(name); setStatus(`已保存风格预设「${name}」`);
});
$('btnPresetDelete').addEventListener('click', () => {
  const name = $('presetSelect').value; if (!name) return;
  const presets = readPresets(); delete presets[name];
  localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
  refreshPresetList(); setStatus(`已删除预设「${name}」`);
});

try { applySettings(JSON.parse(localStorage.getItem(SETTINGS_KEY) || 'null'), false); } catch { /* 忽略损坏的旧状态 */ }
refreshPresetList();

// ---------- 模型持久缓存状态 ----------
const MODEL_CACHE = 'seichi-models-v2';
const MODEL_URLS = [
  './models/person-detect.onnx', DEVICE.isAppleMobile ? './models/isnet-anime-512-fp16.onnx' : './models/isnet-anime-fp16.onnx',
  './models/sam-encoder.onnx', './models/sam-decoder.onnx',
];
const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
const RUNTIME_URLS = [
  'ort.webgpu.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
].map((name) => ORT_BASE + name);
const AI_CACHE_URLS = [...MODEL_URLS, ...RUNTIME_URLS];

async function updateModelCacheStatus() {
  const label = $('modelCacheStatus');
  if (!('caches' in window) || !('serviceWorker' in navigator)) {
    label.textContent = '此浏览器不支持模型持久缓存'; $('btnCacheModels').disabled = true; return;
  }
  try {
    const cache = await caches.open(MODEL_CACHE);
    let modelCount = 0, runtimeCount = 0;
    for (const url of MODEL_URLS) if (await cache.match(new URL(url, location.href).href, { ignoreVary: true })) modelCount++;
    for (const url of RUNTIME_URLS) if (await cache.match(url, { ignoreVary: true })) runtimeCount++;
    const ready = modelCount === MODEL_URLS.length && runtimeCount === RUNTIME_URLS.length;
    label.textContent = ready
      ? 'AI 模型与运行时已缓存 ✓ · 系统清理后可重新下载'
      : `模型 ${modelCount}/${MODEL_URLS.length} · 运行时 ${runtimeCount}/${RUNTIME_URLS.length}`;
    $('btnCacheModels').textContent = ready ? '重新校验离线缓存' : '缓存全部 AI 模型（约 180MB）';
  } catch (e) { label.textContent = '无法读取模型缓存：' + (e.message || e); }
}

$('btnCacheModels').addEventListener('click', async () => {
  const btn = $('btnCacheModels'), label = $('modelCacheStatus'); btn.disabled = true;
  try {
    await navigator.serviceWorker.ready;
    if (navigator.storage?.persist) await navigator.storage.persist().catch(() => false);
    for (let i = 0; i < AI_CACHE_URLS.length; i++) {
      label.textContent = i < MODEL_URLS.length
        ? `正在缓存 AI 模型 ${i + 1}/${MODEL_URLS.length}…`
        : `正在缓存推理运行时 ${i - MODEL_URLS.length + 1}/${RUNTIME_URLS.length}…`;
      const response = await fetch(AI_CACHE_URLS[i]);
      if (!response.ok) throw new Error(`${AI_CACHE_URLS[i]} 下载失败（${response.status}）`);
      await response.arrayBuffer(); // 等响应完整落入 Service Worker Cache 后再取下一个
    }
    await updateModelCacheStatus();
  } catch (e) { label.textContent = '模型缓存失败：' + (e.message || e); }
  finally { btn.disabled = false; }
});

$('deviceStatus').textContent = IS_MOBILE
  ? `${DEVICE.isIPhone ? 'iPhone' : DEVICE.isIPad ? 'iPad' : '移动端'}保护：预览最长边 ${MAX_DIM}px，导出上限约 ${Math.round(EXPORT_MAX_PIXELS / 1e6)}MP`
  : `桌面预览最长边 ${MAX_DIM}px；导出按原始分辨率分块处理`;
$('mobileSamRow').hidden = !DEVICE.isAppleMobile;

setStatus('请上传动画截图与实景照片');

// Service Worker：把 190MB 的 ONNX 模型钉进 Cache Storage，二次访问/离线可用。
// file:// 或不支持时静默跳过，不影响功能。
if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
  navigator.serviceWorker.register('./sw.js').then(() => navigator.serviceWorker.ready).then(updateModelCacheStatus).catch(updateModelCacheStatus);
} else {
  updateModelCacheStatus();
}

// 隐藏的开发验收钩子：仅供脚本化验收调用内部函数，不出现在 UI
// ---------- 圈选抠图（LR 式手动指定范围）----------
// 自动检测漏检/抠错/骑车构图无响应时，用户在截图上随手圈出角色，
// 走与自动流水线相同的"框裁剪 → ISNet → SAM 兜底"，笔迹质心作为 SAM 提示点。
const lassoState = { pts: [], drawing: false, busy: false };

function lassoRedraw() {
  const c = $('lassoCanvas'), ctx = c.getContext('2d');
  ctx.putImageData(state.anime.imgData, 0, 0);
  if (lassoState.pts.length > 1) {
    ctx.save();
    ctx.lineWidth = Math.max(2, c.width / 350);
    ctx.strokeStyle = 'rgba(255,80,140,.95)';
    ctx.fillStyle = 'rgba(255,80,140,.14)';
    ctx.beginPath();
    ctx.moveTo(lassoState.pts[0][0], lassoState.pts[0][1]);
    for (let i = 1; i < lassoState.pts.length; i++) ctx.lineTo(lassoState.pts[i][0], lassoState.pts[i][1]);
    ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();
  }
}

function lassoBBox() {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of lassoState.pts) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const W = state.anime.width, H = state.anime.height;
  minX = Math.max(0, minX); minY = Math.max(0, minY);
  maxX = Math.min(W, maxX); maxY = Math.min(H, maxY);
  return { x: Math.round(minX), y: Math.round(minY), w: Math.round(maxX - minX), h: Math.round(maxY - minY) };
}

function openLasso() {
  if (!state.anime) return;
  const c = $('lassoCanvas');
  c.width = state.anime.width; c.height = state.anime.height;
  lassoState.pts = []; lassoState.drawing = false;
  $('btnLassoRun').disabled = true;
  lassoRedraw();
  $('lassoModal').hidden = false;
}

function closeLasso() { $('lassoModal').hidden = true; }

// 圈选核心：bbox + 笔迹质心 → 单框流水线 → 追加为可勾选角色
async function runLassoBox(box, centroid) {
  if (lassoState.busy) return null;
  lassoState.busy = true;
  $('btnLassoRun').disabled = true;
  const onStage = (s) => { $('extractStatus').textContent = s; };
  const onProgress = (recv, total) => {
    $('extractStatus').textContent = `下载模型 ${(recv / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)}MB`;
  };
  try {
    const mobileOpts = DEVICE.isAppleMobile
      ? { isnetModelUrl: './models/isnet-anime-512-fp16.onnx', isnetSize: 512 }
      : {};
    const found = await extractCharactersInRegion(state.anime.imgData, box, {
      ...mobileOpts, samPoints: centroid ? [centroid] : [], onStage, onProgress,
    });
    if (!state.charSeg) state.charSeg = { chars: [], included: new Set() };
    const seg = state.charSeg;
    for (const char of found) {
      seg.chars.push(char);
      if (!char.empty) seg.included.add(seg.chars.length - 1);
    }
    setCharSeg(seg);
    const hadCutout = !!state.cutout;
    applyCharSelection(!hadCutout);
    const ok = found.filter((c) => !c.empty).length;
    $('extractStatus').textContent = ok === 0
      ? '圈选区域没抠出内容——试着圈大一点、或让圈更贴近角色'
      : `圈选区抠出 ${ok} 个角色 · 可勾选/调阈值/拖拽`;
    return found;
  } catch (e) {
    console.error(e);
    $('extractStatus').textContent = '圈选抠图失败：' + (e.message || e);
    return null;
  } finally {
    if (DEVICE.isAppleMobile) await releaseAllSessions();
    lassoState.busy = false;
    $('btnLassoRun').disabled = lassoState.pts.length < 8;
  }
}

$('btnLasso').addEventListener('click', openLasso);
$('btnLassoClose').addEventListener('click', closeLasso);
$('btnLassoClear').addEventListener('click', () => { lassoState.pts = []; $('btnLassoRun').disabled = true; lassoRedraw(); });
$('btnLassoRun').addEventListener('click', async () => {
  const box = lassoBBox();
  if (box.w < 12 || box.h < 12) { $('extractStatus').textContent = '圈得太小了，重新圈一下'; return; }
  let sx = 0, sy = 0;
  for (const [x, y] of lassoState.pts) { sx += x; sy += y; }
  const centroid = [sx / lassoState.pts.length, sy / lassoState.pts.length];
  closeLasso();
  await runLassoBox(box, centroid);
});

(function setupLassoDraw() {
  const c = $('lassoCanvas');
  const toImg = (e) => {
    const r = c.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width * c.width, (e.clientY - r.top) / r.height * c.height];
  };
  c.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    try { c.setPointerCapture(e.pointerId); } catch { /* 某些环境/合成事件会抛，不影响绘制 */ }
    lassoState.drawing = true;
    lassoState.pts = [toImg(e)];
    lassoRedraw();
  });
  c.addEventListener('pointermove', (e) => {
    if (!lassoState.drawing) return;
    const p = toImg(e), last = lassoState.pts[lassoState.pts.length - 1];
    if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 3) { lassoState.pts.push(p); lassoRedraw(); }
  });
  const end = () => {
    if (!lassoState.drawing) return;
    lassoState.drawing = false;
    $('btnLassoRun').disabled = lassoState.pts.length < 8 || lassoState.busy;
  };
  c.addEventListener('pointerup', end);
  c.addEventListener('pointercancel', end);
})();

window.__qa = { state, renderFullRes, enterAlignMode, applyAlignCrop, alignState, recompute, runLassoBox };

// 隐藏的开发验收入口：http://localhost:8126/?qa-demo=1
if (new URLSearchParams(location.search).has('qa-demo')) {
  setStatus('正在载入演示素材…');
  Promise.all([urlToImageData('./test-izu-far.jpg'), urlToImageData('./test-izu-scenery.jpg')])
    .then(([anime, photo]) => {
      $('thumbAnime').src = anime.url; $('thumbAnime').hidden = false;
      $('thumbPhoto').src = photo.url; $('thumbPhoto').hidden = false;
      return handleAnimeData(anime).then(() => handlePhotoData(photo));
    })
    .catch((e) => setStatus(`演示素材加载失败：${e.message || e}`));
}
