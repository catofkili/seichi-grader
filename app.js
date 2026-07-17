// app.js — UI 编排：上传、迁移、对比预览、导出
import {
  imageDataToLab, labStats, makeLabTransform, makeLumaCdfMap, makeGradeTransform,
  applyTransfer, applyTransferRegioned, skyMask, makeBloomLayer, applyBloom, extractPalette, generateCubeLUT,
} from './color.js?v=20260712d';
import { extractForeground, cutoutCanvas, cleanupAlpha, alphaBBox } from './segment.js';
import { mergeCharacterAlphas } from './ai-segment.js';
import { releaseAllSessions } from './ort-env.js';
import { embedImage, cosineSimilarity, SCENE_EMBED_MODEL_URL } from './embed.js';
import { profile as DEVICE } from './platform.js';
import { launchViewfinder } from './camera/viewfinder.js';

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
  forcedAlpha: null,   // 用户蓝色画笔强制保留区；始终盖过算法结果与阈值/收边
  charSeg: null,       // AI 检测流水线结果 { chars, included:Set }，供勾选角色重合并
  rawW: 0, rawH: 0,
  charPos: { cx: 0.5, cy: 0.62 }, // 角色中心在场景中的归一化位置
  charBase: null,      // 角色在动画帧里的基准 { relH: bbox高/帧高, cx, cy }；
                       // 构图对齐后照片≈动画取景，按原占比原位落地，100%=与动画同比例
  charLock: false,     // 固定角色：拖拽/捏合/滚轮/滑杆全部忽略，防误触（还原按钮仍有效）
  charDraw: null,      // 角色在 canvas 坐标的绘制矩形 {dx,dy,dw,dh}，用于拖拽命中
  harmonizedCache: null,
  gradeCache: null,   // 图片不变时复用统计、CDF 与天空掩膜；滑杆只重套用
  lastExport: null,   // iOS 二次用户手势分享用 { blob, name, width, height }
  aiBusy: false,
};

const $ = (id) => document.getElementById(id);

// Android 对包含 RAW 扩展名的 accept 往往直接打开文件管理器。
// 改用通用图片 MIME 类型，会优先给出系统图库/照片选择器；桌面仍可选 RAW 文件。
if (DEVICE.isAndroid) {
  ['fileAnime', 'filePhoto', 'matchFiles', 'batchFiles'].forEach((id) => {
    $(id).accept = 'image/*';
  });
}

const setStatus = (t) => { $('status').textContent = t; };
const recentErrors = [];
function rememberError(where, error) {
  recentErrors.push({ at: new Date().toISOString(), where, name: error?.name || 'Error', message: String(error?.message || error) });
  if (recentErrors.length > 12) recentErrors.shift();
}
window.addEventListener('error', (event) => rememberError('window.error', event.error || event.message));
window.addEventListener('unhandledrejection', (event) => rememberError('unhandledrejection', event.reason));

// 轻量工作流：不阻塞操作，只告诉第一次来的用户当前下一步是什么。
function updateWorkflow() {
  const steps = [...$('workflowSteps').querySelectorAll('li')];
  const complete = {
    anime: !!state.anime,
    photo: !!state.photo,
    grade: !!state.gradedData,
    compose: !!state.cutout,
  };
  let current = !complete.anime ? 'anime' : !complete.photo ? 'photo' : !complete.grade ? 'grade' : 'compose';
  for (const item of steps) {
    const key = item.dataset.step;
    item.classList.toggle('done', !!complete[key]);
    item.classList.toggle('current', key === current && !complete[key]);
  }
}

// 把 File 读成按 MAX_DIM 缩放后的 ImageData。RAW 先抽内嵌 JPEG 预览再解码。
async function fileToImageData(file) {
  if (isRawFile(file)) return decodeRawViaPreview(file);
  return decodeBlobToImageData(file, file);
}

function decodeBlobToImageData(blob, file) {
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
      // iOS 上所有浏览器都是 WebKit 内核，解不了 HEIC 只可能是系统版本老（iOS 17 起才支持网页解码）
      reject(new Error(isHeicFile(file)
        ? (DEVICE.isAppleMobile
          ? '系统版本较旧无法解码 HEIC：请升级 iOS，或在相册用「导出为 JPEG」后再传（拍摄端可在 设置→相机→格式 选「兼容性最佳」）'
          : '此浏览器无法解码 HEIC：请先转成 JPEG 再传（iPhone 相册可直接「导出为 JPEG」；Windows 双击照片用「照片」应用另存为 JPEG），Mac 用户可改用 Safari 打开本站')
        : '浏览器无法解码这张图片，请转换为 JPEG 或 PNG 后重试'));
    };
    img.src = URL.createObjectURL(blob);
  });
}

function isHeicFile(file) {
  return /\.(heic|heif)$/i.test(file.name || '') || /image\/(heic|heif)/i.test(file.type || '');
}

function isRawFile(file) {
  return /\.(cr2|cr3|nef|arw|dng|orf|rw2|raf|pef|srw)$/i.test(file.name || '');
}

// RAW 兜底：浏览器解不了 RAW 原始数据，但主流相机 RAW 都内嵌完整 JPEG 预览
// （多为全尺寸）。在字节流里定位所有 JPEG 段、从大到小试解码，零依赖零许可负担。
async function decodeRawViaPreview(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const starts = [], ends = [];
  for (let i = 0; i < bytes.length - 2; i++) {
    if (bytes[i] !== 0xff) continue;
    if (bytes[i + 1] === 0xd8 && bytes[i + 2] === 0xff) starts.push(i);
    else if (bytes[i + 1] === 0xd9) ends.push(i);
  }
  const candidates = [];
  let e = 0;
  for (const s of starts) {
    while (e < ends.length && ends[e] < s) e++;
    if (e < ends.length) candidates.push({ start: s, len: ends[e] + 2 - s });
  }
  candidates.sort((a, b) => b.len - a.len);
  for (const c of candidates.slice(0, 3)) {
    try {
      const blob = new Blob([bytes.subarray(c.start, c.start + c.len)], { type: 'image/jpeg' });
      const data = await decodeBlobToImageData(blob, file);
      data.fromRawPreview = true;
      return data;
    } catch { /* 段不完整（如嵌套缩略图截断），试下一个 */ }
  }
  throw new Error('无法从这张 RAW 提取内嵌预览，请先用相机厂商软件或系统相册导出为 JPEG 再上传');
}

// 只读取 JPEG EXIF 的 GPS；没有坐标或解析失败不会影响上传与调色。
async function readExifGPS(file) {
  if (!/image\/jpe?g/i.test(file.type || '') && !/\.jpe?g$/i.test(file.name || '')) return null;
  try {
    const view = new DataView(await file.arrayBuffer());
    if (view.getUint16(0) !== 0xffd8) return null;
    let pos = 2;
    while (pos + 4 < view.byteLength) {
      if (view.getUint8(pos) !== 0xff) break;
      const marker = view.getUint8(pos + 1), len = view.getUint16(pos + 2);
      if (marker === 0xe1 && len >= 10 && view.getUint32(pos + 4) === 0x45786966) {
        const base = pos + 10, little = view.getUint16(base) === 0x4949;
        const u16 = (at) => view.getUint16(at, little), u32 = (at) => view.getUint32(at, little);
        const ifd = base + u32(base + 4), count = u16(ifd);
        let gpsOffset = 0;
        for (let i = 0; i < count; i++) {
          const at = ifd + 2 + i * 12;
          if (u16(at) === 0x8825) { gpsOffset = u32(at + 8); break; }
        }
        if (!gpsOffset) return null;
        const gps = base + gpsOffset, gpsCount = u16(gps), values = new Map();
        for (let i = 0; i < gpsCount; i++) {
          const at = gps + 2 + i * 12, tag = u16(at);
          values.set(tag, { type: u16(at + 2), n: u32(at + 4), value: u32(at + 8), at });
        }
        const ascii = (entry) => entry ? String.fromCharCode(view.getUint8(entry.at + 8)) : '';
        const rational3 = (entry) => {
          if (!entry || entry.type !== 5 || entry.n < 3) return null;
          const at = base + entry.value;
          const r = (i) => { const d = u32(at + i * 8 + 4); return d ? u32(at + i * 8) / d : 0; };
          return r(0) + r(1) / 60 + r(2) / 3600;
        };
        let lat = rational3(values.get(2)), lon = rational3(values.get(4));
        if (lat == null || lon == null) return null;
        if (ascii(values.get(1)) === 'S') lat = -lat;
        if (ascii(values.get(3)) === 'W') lon = -lon;
        return { lat, lon };
      }
      if (len < 2) break;
      pos += len + 2;
    }
  } catch { /* 无 EXIF、损坏文件或浏览器拒绝读取时忽略 */ }
  return null;
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
  redrawComparisonReference();
  syncCanvasSize();
  $('compareModes').hidden = false;
  ['btnOpenExportHub', 'btnExportImg', 'btnExportCompare', 'btnExportCompareLayout', 'btnExportWipe', 'btnExportMorph', 'btnExportLut', 'btnBatchExport'].forEach(id => $(id).disabled = false);
  updateWorkflow();
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
  const showChar = state.cutout && $('composite').checked;
  if (showChar) compositeCharacter($('canvasGraded'));
  // 预览下方的大号缩放条：有角色时才出现，并与控制面板滑杆保持同值
  $('charBar').hidden = !showChar;
  if ($('charScaleQuick').value !== $('charScale').value) {
    $('charScaleQuick').value = $('charScale').value;
  }
  $('charScaleQuickVal').textContent = $('charScale').value + '%';
}

// 角色缩放的唯一入口：钳到滑杆范围，双滑杆/标签同步后重绘
function setCharScale(value) {
  const v = Math.max(20, Math.min(300, Math.round(Number(value) || 100)));
  $('charScale').value = v;
  $('charScaleVal').textContent = v + '%';
  redrawComposite();
}

// 还原：回到抠图时自动给出的大小（100%=与动画同比例）和动画原位
function resetCharPlacement() {
  if (state.charBase) state.charPos = { cx: state.charBase.cx, cy: state.charBase.cy };
  setCharScale(100);
}

function setCharLock(locked) {
  state.charLock = locked;
  $('charScale').disabled = locked;
  $('charScaleQuick').disabled = locked;
  $('btnCharLock').textContent = locked ? '已固定 🔒' : '固定';
  $('btnCharLock').classList.toggle('lock-on', locked);
}

// 让两个叠放的 canvas 在容器内等比同尺寸显示。
// 尺寸以 canvasGraded 当前像素尺寸为准：普通模式=照片尺寸，对齐模式=动画宽高比画布。
function syncCanvasSize() {
  const compare = $('compare');
  const cw = compare.clientWidth, ch = compare.clientHeight;
  const iw = $('canvasGraded').width || state.photo.width, ih = $('canvasGraded').height || state.photo.height;
  const scale = Math.min(cw / iw, ch / ih);
  const w = iw * scale, h = ih * scale;
  for (const c of [$('canvasGraded'), $('canvasOrig'), $('canvasAnimeOverlay')]) {
    c.style.width = w + 'px';
    c.style.height = h + 'px';
  }
  const ghost = $('alignGhost');
  if (ghost && !ghost.hidden) { ghost.style.width = w + 'px'; ghost.style.height = h + 'px'; }
}

// 动画截图用于两种“此处 / 彼处”对比：上下模式保持完整比例；叠加模式则居中 cover，
// 在已构图对齐时会与实景画框完全重合。
function redrawComparisonReference() {
  if (!state.anime || !$('canvasGraded').width) return;
  const stack = $('canvasAnimeStack');
  drawToCanvas(stack, state.anime.imgData);
  const overlay = $('canvasAnimeOverlay');
  const target = $('canvasGraded');
  overlay.width = target.width; overlay.height = target.height;
  const ctx = overlay.getContext('2d');
  const scale = Math.max(overlay.width / stack.width, overlay.height / stack.height);
  const w = stack.width * scale, h = stack.height * scale;
  ctx.drawImage(stack, (overlay.width - w) / 2, (overlay.height - h) / 2, w, h);
  updateOverlayOpacity();
}

let comparisonMode = 'slider';
function updateOverlayOpacity() {
  const value = $('overlayOpacity').value;
  $('overlayOpacityVal').textContent = value + '%';
  $('canvasAnimeOverlay').style.opacity = String(Number(value) / 100);
}

function setComparisonMode(mode) {
  comparisonMode = mode;
  const compare = $('compare');
  const isSlider = mode === 'slider', isOverlay = mode === 'overlay', isStack = mode === 'stack';
  compare.classList.toggle('compare-stack-mode', isStack);
  $('clip').hidden = !isSlider;
  $('handle').hidden = !isSlider;
  $('canvasAnimeOverlay').hidden = !isOverlay;
  $('stackAnimePanel').hidden = !isStack;
  $('badgeLeft').hidden = isStack;
  $('badgeRight').hidden = isStack;
  $('badgeLeft').textContent = isOverlay ? '动画截图' : '原图';
  $('badgeRight').textContent = '调色后';
  $('overlayControls').hidden = !isOverlay;
  document.querySelectorAll('[data-compare-mode]').forEach((button) => button.classList.toggle('on', button.dataset.compareMode === mode));
  if (state.photo) syncCanvasSize();
}

document.querySelectorAll('[data-compare-mode]').forEach((button) => {
  button.addEventListener('click', () => setComparisonMode(button.dataset.compareMode));
});
$('overlayOpacity').addEventListener('input', () => { updateOverlayOpacity(); queueSettingsSave(); });

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
  const dh = canvas.height * (state.charBase?.relH || 0.45) * scalePct;
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
  if (state.forcedAlpha?.length === clean.length) {
    for (let i = 0; i < clean.length; i++) if (state.forcedAlpha[i]) clean[i] = 255;
  }
  const { bbox, coverage } = alphaBBox(clean, w, h);
  state.cutout = bbox ? cutoutCanvas(state.anime.imgData, { alpha: clean, bbox }) : null;
  if (bbox) {
    state.charBase = {
      relH: bbox.h / h,
      cx: (bbox.x + bbox.w / 2) / w,
      cy: (bbox.y + bbox.h / 2) / h,
    };
  }
  if (state.cutout) {
    // 去色渗：羽化边缘的 RGB 混有动画背景色，向内部实心像素取色修正
    const ctx = state.cutout.getContext('2d');
    const cimg = ctx.getImageData(0, 0, state.cutout.width, state.cutout.height);
    decontaminateEdges(cimg);
    ctx.putImageData(cimg, 0, 0);
  }
  invalidateHarmonize();
  $('btnExportCharacter').disabled = !state.cutout;
  $('btnEraseMask').disabled = !state.cutout;
  // 新抠图默认按动画里的原位、原比例落地（照片已与动画同构图），拖拽/滑杆仍可自由调整
  if (resetPos) {
    state.charPos = state.charBase ? { cx: state.charBase.cx, cy: state.charBase.cy } : { cx: 0.5, cy: 0.62 };
    setCharScale(100);
  }
  redrawComposite();
  updateWorkflow();
  return coverage;
}

// ---------- 文件上传绑定 ----------
function bindDrop(dropId, inputId, thumbId, onLoad) {
  const drop = $(dropId), input = $(inputId), thumb = $(thumbId);
  const handle = async (file) => {
    if (!file) return;
    const imageExt = /\.(jpe?g|png|webp|heic|heif)$/i.test(file.name || '') || isRawFile(file);
    if (!file.type.startsWith('image/') && !imageExt) { setStatus('请选择 JPEG、PNG、WebP 或 HEIC 图片'); return; }
    try {
      setStatus(isHeicFile(file) ? '使用系统解码 HEIC 并校正方向…' : '读取图片并校正 EXIF 方向…');
      const [data, gps] = await Promise.all([fileToImageData(file), readExifGPS(file)]);
      data.gps = gps;
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
  state.cutout = null; state.rawAlpha = null; state.forcedAlpha = null; setCharSeg(null); invalidateHarmonize();
  $('btnExtract').disabled = false;
  $('btnExtractAI').disabled = false;
  $('btnLasso').disabled = false;
  $('btnMatchScene').disabled = false;
  $('matchResults').hidden = true; // 旧结果按旧截图排序，换截图后作废
  $('btnEraseMask').disabled = true;
  $('btnExportCharacter').disabled = true;
  $('extractStatus').textContent = '点击「扒取人物」试试';
  $('btnAlign').disabled = !state.photo;
  updateWorkflow();
  recompute();
}

async function handlePhotoData(data) {
  if (state.photo?.srcUrl?.startsWith('blob:')) URL.revokeObjectURL(state.photo.srcUrl);
  state.photo = {
    imgData: data.imgData, width: data.width, height: data.height, srcUrl: data.url, align: null,
    originalWidth: data.originalWidth || data.width, originalHeight: data.originalHeight || data.height,
    fileName: data.fileName || '', gps: data.gps || null,
  };
  state.gradeCache = null;
  exitAlignMode(false);
  $('btnAlign').disabled = !state.anime;
  updateWorkflow();
  setStatus(`照片已读取 · ${data.fromRawPreview ? 'RAW 内嵌预览' : '原图'} ${state.photo.originalWidth}×${state.photo.originalHeight} · 预览 ${data.width}×${data.height}`);
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

$('charScale').addEventListener('input', (e) => { if (!state.charLock) setCharScale(e.target.value); });
$('charScaleQuick').addEventListener('input', (e) => { if (!state.charLock) setCharScale(e.target.value); });
$('btnCharReset').addEventListener('click', resetCharPlacement);
$('btnCharLock').addEventListener('click', () => setCharLock(!state.charLock));

// ⓘ 说明气泡：手机没有 hover，点按切换；点别处或再点一次收起
document.addEventListener('click', (e) => {
  const tip = e.target.closest('.info-tip');
  document.querySelectorAll('.info-tip.open').forEach((el) => { if (el !== tip) el.classList.remove('open'); });
  if (tip) tip.classList.toggle('open');
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

// 模型推理和 1024px 遮罩的预/后处理都不能占用页面主线程。
// 即便是桌面浏览器也统一交给一次性 Worker，避免用户在等待时点击任何控件就让标签页假死。
function runAIInWorker(imageData, opts = {}) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('./ai-worker.js', { type: 'module', name: 'seichi-ai-once' });
    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'progress') opts.onProgress?.(msg.received, msg.total);
      else if (msg.type === 'stage') opts.onStage?.(msg.text);
      else if (msg.type === 'done') { worker.terminate(); resolve(msg.result); }
      else if (msg.type === 'error') {
        worker.terminate();
        const error = new Error(msg.error?.message || 'AI Worker 失败'); error.name = msg.error?.name || 'Error'; error.stack = msg.error?.stack || error.stack;
        reject(error);
      }
    };
    worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'AI Worker 加载失败')); };
    // 不转移 imageData.data.buffer：主页面仍需拿它显示预览/继续调色。结构化复制只发生一次，
    // 后续几十秒的推理与数组处理都会留在 Worker 内。
    worker.postMessage({ imageData, job: opts.job || 'auto', box: opts.box, samPoints: opts.samPoints || [], hires: !!opts.hires, samFallback: opts.samFallback !== false, mobileModel: !!opts.mobileModel });
  });
}

$('btnExtract').addEventListener('click', () => {
  if (!state.anime) return;
  // 轻量算法不擅长从整幅复杂动画里猜主体；先由用户圈出范围再处理。
  openLasso('algorithm');
});

$('btnExtractAI').addEventListener('click', async () => {
  if (!state.anime) return;
  const btn = $('btnExtractAI');
  state.aiBusy = true;
  btn.disabled = true; $('btnExtract').disabled = true; $('btnLasso').disabled = true;
  $('btnExportImg').disabled = true;
  const onProgress = (recv, total) => {
    $('extractStatus').textContent = `下载模型 ${(recv / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)}MB`;
  };
  const onStage = (s) => { $('extractStatus').textContent = s; };
  try {
    const t0 = performance.now();
    // 两级流水线：先检测角色框，再逐框抠图。所有平台都在 Worker 内运行。
    const runAI = () => runAIInWorker(state.anime.imgData, {
      hires: $('hiresDet').checked,
      samFallback: DEVICE.isAppleMobile ? $('mobileSam').checked : true,
      mobileModel: DEVICE.isAppleMobile,
      onProgress, onStage,
    });
    let aiResult;
    try {
      aiResult = await runAI();
    } catch (firstError) {
      // 每次 Worker 都是一次性实例；失败后重建一个干净 Worker，只重试一次避免死循环。
      onStage('重新建立 AI 任务并重试一次…');
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
      const res = whole;
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
    state.aiBusy = false;
    btn.disabled = false; $('btnExtract').disabled = false; $('btnLasso').disabled = false;
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
  $('compareModes').hidden = true;
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
  if (state.gradedData) $('compareModes').hidden = false;
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

  // 双指捏合缩放角色（手机）：第二根手指落下即接管，抬起一根即结束
  const pointers = new Map();
  const pinch = { active: false, startDist: 0, startScale: 100 };
  const pinchDist = () => {
    const [a, b] = [...pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };

  compare.addEventListener('pointerdown', (e) => {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2 && state.cutout && $('composite').checked && !alignState.active && !state.charLock) {
      pinch.active = true;
      charDrag = sliderDrag = false;
      pinch.startDist = pinchDist() || 1;
      pinch.startScale = parseInt($('charScale').value, 10);
      compare.classList.remove('grabbing');
      return;
    }
    const p = toCanvas(e.clientX, e.clientY);
    if (alignState.active) {
      alignDrag = true;
      alignStart = { x: p.x, y: p.y, cx: alignState.crop.cx, cy: alignState.crop.cy };
      compare.classList.add('grabbing');
      return;
    }
    if (state.cutout && $('composite').checked && !state.charLock && p.inDisplay && hitChar(p.x, p.y)) {
      charDrag = true;
      grabDX = p.x - (state.charPos.cx * gradedCanvas.width);
      grabDY = p.y - (state.charPos.cy * gradedCanvas.height);
      compare.classList.add('grabbing');
    } else if (comparisonMode === 'slider') {
      moveSlider(e.clientX); sliderDrag = true;
    }
  });

  window.addEventListener('pointermove', (e) => {
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pinch.active) {
      if (pointers.size >= 2) setCharScale(pinch.startScale * (pinchDist() / pinch.startDist));
      return;
    }
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
  const releasePointer = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch.active = false;
    sliderDrag = false; charDrag = false; alignDrag = false; compare.classList.remove('grabbing');
  };
  window.addEventListener('pointerup', releasePointer);
  window.addEventListener('pointercancel', releasePointer);

  // 桌面：滚轮悬停在角色上直接缩放
  compare.addEventListener('wheel', (e) => {
    if (!state.cutout || !$('composite').checked || alignState.active || state.charLock) return;
    const p = toCanvas(e.clientX, e.clientY);
    if (!p.inDisplay || !hitChar(p.x, p.y)) return;
    e.preventDefault();
    setCharScale(parseInt($('charScale').value, 10) * (e.deltaY < 0 ? 1.06 : 1 / 1.06));
  }, { passive: false });
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

function makeCompareCanvas(maxWidth = 0) {
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
  const W = maxWidth ? Math.min(gw, maxWidth) : gw;
  const fit = (cv) => ({ cv, h: Math.round(cv.height * (W / cv.width)) });

  let panels;
  if (layout === 'triple') panels = [fit(tmpAnime), fit(tmpOrig), fit(tmpGraded)];
  else panels = [fit(tmpAnime), fit(tmpGraded)];

  if (layout === 'postcard') {
    const title = $('compareTitle').value.trim() || '聖地巡礼 · 此处与彼处';
    const manualPlace = $('comparePlace').value.trim();
    const gps = state.photo.gps;
    const gpsText = gps ? `${gps.lat.toFixed(5)}, ${gps.lon.toFixed(5)}` : '';
    const place = [manualPlace, gpsText].filter(Boolean).join(' · ');
    const header = Math.round(W * .20), footer = place ? Math.round(W * .10) : Math.round(W * .045);
    out.width = W;
    out.height = header + panels.reduce((s, p) => s + p.h, 0) + gap + footer;
    ctx.fillStyle = '#0e1014'; ctx.fillRect(0, 0, out.width, out.height);
    ctx.fillStyle = '#ff5e8a'; ctx.fillRect(0, 0, W, Math.max(4, Math.round(W / 100)));
    ctx.fillStyle = '#f2f4f8'; ctx.font = `600 ${Math.round(W / 19)}px sans-serif`; ctx.textBaseline = 'top';
    ctx.fillText(title, Math.round(W * .06), Math.round(W * .055), Math.round(W * .88));
    let y = header;
    panels.forEach((p, i) => {
      ctx.drawImage(p.cv, 0, y, W, p.h);
      ctx.fillStyle = 'rgba(0,0,0,.58)'; ctx.fillRect(12, y + 12, Math.round(W * .19), Math.round(W * .07));
      ctx.fillStyle = '#fff'; ctx.font = `600 ${Math.round(W / 28)}px sans-serif`;
      ctx.fillText(i ? '此处 · 调色后' : '彼处 · 动画', 20, y + 20);
      y += p.h + gap;
    });
    if (place) {
      ctx.fillStyle = '#aeb7c8'; ctx.font = `${Math.round(W / 34)}px sans-serif`;
      ctx.fillText(place, Math.round(W * .06), y + Math.round(W * .025), Math.round(W * .88));
    }
  } else if (layout === 'leftright') {
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
  return out;
}

function exportCompareLayout() {
  const out = makeCompareCanvas();
  download(out.toDataURL('image/png'), 'seichi-compare.png');
}

$('btnExportCompare').addEventListener('click', exportCompareLayout);
$('btnExportCompareLayout').addEventListener('click', exportCompareLayout);

function drawExportHubPreview(canvasId, source) {
  const canvas = $(canvasId), maxW = 340, maxH = 190;
  const sw = source.width, sh = source.height;
  const scale = Math.min(maxW / sw, maxH / sh);
  canvas.width = Math.max(2, Math.round(sw * scale));
  canvas.height = Math.max(2, Math.round(sh * scale));
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
}

function renderExportHubPreviews() {
  if (!state.gradedData || !state.anime) return;
  drawExportHubPreview('hubPreviewImage', $('canvasGraded'));
  drawExportHubPreview('hubPreviewCompare', makeCompareCanvas(340));
  const wipe = makeWipeFrameCanvas(340);
  drawWipeFrame(wipe.frame.getContext('2d'), wipe.original, wipe.graded, wipe.w, wipe.h, .5);
  drawExportHubPreview('hubPreviewWipe', wipe.frame);
  const morph = document.createElement('canvas');
  morph.width = $('canvasGraded').width; morph.height = $('canvasGraded').height;
  const morphCtx = morph.getContext('2d');
  morphCtx.drawImage($('canvasGraded'), 0, 0);
  morphCtx.globalAlpha = .5; morphCtx.drawImage($('canvasAnimeOverlay'), 0, 0); morphCtx.globalAlpha = 1;
  drawExportHubPreview('hubPreviewMorph', morph);
  const characterCard = $('hubCharacterCard');
  characterCard.hidden = !state.cutout;
  if (state.cutout) drawExportHubPreview('hubPreviewCharacter', state.cutout);
  $('hubCompareTitle').textContent = `对比图 · ${$('layout').selectedOptions[0].textContent}`;
}

function openExportHub() {
  renderExportHubPreviews();
  $('exportHubModal').hidden = false;
}

function closeExportHub() { $('exportHubModal').hidden = true; }

$('btnOpenExportHub').addEventListener('click', openExportHub);
$('btnCloseExportHub').addEventListener('click', closeExportHub);
$('exportHubModal').addEventListener('click', (e) => { if (e.target === $('exportHubModal')) closeExportHub(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('exportHubModal').hidden) closeExportHub(); });
document.querySelectorAll('[data-export-action]').forEach((card) => {
  card.addEventListener('click', () => {
    const buttons = { image: 'btnExportImg', compare: 'btnExportCompare', wipe: 'btnExportWipe', morph: 'btnExportMorph', character: 'btnExportCharacter', batch: 'btnBatchExport', lut: 'btnExportLut' };
    const target = $(buttons[card.dataset.exportAction]);
    if (!target || target.disabled) return;
    closeExportHub();
    setTimeout(() => target.click(), 0);
  });
});

$('btnExportCharacter').addEventListener('click', async () => {
  if (!state.cutout) { setStatus('请先抠出角色'); return; }
  const btn = $('btnExportCharacter'); btn.disabled = true;
  try {
    const blob = await new Promise((resolve) => state.cutout.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('浏览器无法编码透明 PNG');
    const url = URL.createObjectURL(blob);
    download(url, 'seichi-character.png');
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    setStatus(`已导出角色透明 PNG · ${state.cutout.width}×${state.cutout.height}`);
  } catch (e) { setStatus('角色 PNG 导出失败：' + (e.message || e)); }
  finally { btn.disabled = false; }
});

// 滑动对比动图使用当前预览分辨率（最长边最多 1080px），避免为了 4 秒视频再次
// 跑一遍全分辨率调色并在手机上占用过多内存。导出内容与页面中央滑杆完全一致。
function makeWipeFrameCanvas(maxSide = 1080) {
  const src = $('canvasGraded');
  const scale = Math.min(1, maxSide / Math.max(src.width, src.height));
  const w = Math.max(2, Math.round(src.width * scale)), h = Math.max(2, Math.round(src.height * scale));
  const original = document.createElement('canvas'); original.width = w; original.height = h;
  original.getContext('2d').drawImage($('canvasOrig'), 0, 0, w, h);
  const graded = document.createElement('canvas'); graded.width = w; graded.height = h;
  graded.getContext('2d').drawImage(src, 0, 0, w, h);
  const frame = document.createElement('canvas'); frame.width = w; frame.height = h;
  return { frame, original, graded, w, h };
}

function drawWipeFrame(ctx, original, graded, w, h, progress) {
  const split = Math.round(w * progress);
  ctx.drawImage(graded, 0, 0);
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, split, h); ctx.clip();
  ctx.drawImage(original, 0, 0); ctx.restore();
  ctx.fillStyle = 'rgba(255,255,255,.96)'; ctx.fillRect(Math.max(0, split - 1), 0, 2, h);
  ctx.font = `600 ${Math.max(14, Math.round(w / 48))}px sans-serif`;
  ctx.textBaseline = 'top';
  const label = (text, x) => {
    const pad = Math.max(7, Math.round(w / 150)), y = pad;
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = 'rgba(0,0,0,.58)'; ctx.fillRect(x, y, tw + pad * 2, Math.max(26, Math.round(w / 34)));
    ctx.fillStyle = '#fff'; ctx.fillText(text, x + pad, y + pad * .7);
  };
  label('原图', Math.max(8, Math.min(split - 70, w - 80)));
  label('调色后', Math.min(w - 92, Math.max(split + 10, 8)));
}

function supportedVideoType() {
  if (!('MediaRecorder' in window) || !HTMLCanvasElement.prototype.captureStream) return '';
  const options = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  return options.find((type) => MediaRecorder.isTypeSupported(type)) || '';
}

async function recordWipeVideo() {
  const mimeType = supportedVideoType();
  if (!mimeType) throw new Error('此浏览器不支持动图视频编码；请用最新版 Chrome、Edge 或 Safari 重试');
  const { frame, original, graded, w, h } = makeWipeFrameCanvas();
  const ctx = frame.getContext('2d'), stream = frame.captureStream(24), chunks = [];
  const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 5_000_000 });
  const duration = 3600;
  const blob = await new Promise((resolve, reject) => {
    recorder.addEventListener('dataavailable', (e) => { if (e.data.size) chunks.push(e.data); });
    recorder.addEventListener('error', () => reject(new Error('浏览器编码动图失败')));
    recorder.addEventListener('stop', () => resolve(new Blob(chunks, { type: mimeType })));
    const started = performance.now();
    const render = (now) => {
      const t = Math.min(1, (now - started) / duration);
      // 从左到右、再回到左侧，形成可循环播放的擦除效果。
      const progress = .06 + .88 * (0.5 - 0.5 * Math.cos(Math.PI * 2 * t));
      drawWipeFrame(ctx, original, graded, w, h, progress);
      if (t < 1) requestAnimationFrame(render); else recorder.stop();
    };
    recorder.start(250);
    requestAnimationFrame(render);
  });
  stream.getTracks().forEach((track) => track.stop());
  if (!blob.size) throw new Error('浏览器没有生成动图数据');
  return { blob, mimeType, width: w, height: h };
}

$('btnExportWipe').addEventListener('click', async () => {
  const btn = $('btnExportWipe'); btn.disabled = true;
  try {
    setStatus('正在录制 4 秒滑动对比动图…请保持页面在前台');
    const video = await recordWipeVideo();
    const ext = video.mimeType.includes('mp4') ? 'mp4' : 'webm';
    const url = URL.createObjectURL(video.blob);
    download(url, `seichi-wipe-compare.${ext}`);
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    setStatus(`已导出 ${ext.toUpperCase()} 动图 · ${video.width}×${video.height} · 4 秒循环`);
  } catch (e) {
    console.error(e);
    setStatus('动图导出失败：' + (e.message || e));
  } finally { btn.disabled = false; }
});

// GIF 最多只能有 256 色。这里用 6×6×6 色立方（216 色）+ 40 阶灰度，
// 比原来的 3-3-2 色表多出蓝色层次，动画天空、线稿与阴影会更细腻。
function gifColorTable() {
  const table = new Uint8Array(256 * 3);
  for (let r = 0; r < 6; r++) for (let g = 0; g < 6; g++) for (let b = 0; b < 6; b++) {
    const i = r * 36 + g * 6 + b;
    table[i * 3] = r * 51; table[i * 3 + 1] = g * 51; table[i * 3 + 2] = b * 51;
  }
  for (let i = 0; i < 40; i++) {
    const v = Math.round(i * 255 / 39), p = (216 + i) * 3;
    table[p] = table[p + 1] = table[p + 2] = v;
  }
  return table;
}

function gifLzw(indices) {
  const clear = 256, end = 257;
  let codeSize = 9, nextCode = 258, bitBuffer = 0, bitCount = 0;
  const out = [], dict = new Map();
  const write = (code) => {
    bitBuffer |= code << bitCount; bitCount += codeSize;
    while (bitCount >= 8) { out.push(bitBuffer & 255); bitBuffer >>>= 8; bitCount -= 8; }
  };
  const reset = () => { dict.clear(); codeSize = 9; nextCode = 258; };
  reset(); write(clear);
  let prefix = indices[0];
  for (let i = 1; i < indices.length; i++) {
    const value = indices[i], key = prefix + ',' + value;
    const known = dict.get(key);
    if (known != null) { prefix = known; continue; }
    write(prefix);
    if (nextCode < 4096) {
      dict.set(key, nextCode++);
      // GIF 解码器在读到“下一条”码时才把同一条词典项加入表；
      // 编码端已提前一步加入，所以位宽也必须晚一项升级（> 而非 ===），
      // 否则从 9 位切到 10 位时比特流会错位，生成损坏的 GIF。
      if (nextCode > (1 << codeSize) && codeSize < 12) codeSize++;
    } else { write(clear); reset(); }
    prefix = value;
  }
  write(prefix); write(end);
  if (bitCount) out.push(bitBuffer & 255);
  return new Uint8Array(out);
}

function gifSubBlocks(bytes) {
  const blocks = [];
  for (let pos = 0; pos < bytes.length; pos += 255) {
    const part = bytes.subarray(pos, pos + 255);
    blocks.push(Uint8Array.of(part.length), part);
  }
  blocks.push(Uint8Array.of(0));
  return blocks;
}

const gifWord = (n) => Uint8Array.of(n & 255, (n >> 8) & 255);
const nextPaint = () => new Promise((resolve) => requestAnimationFrame(resolve));

async function makeAnimeToSceneGif() {
  if (!state.anime || !state.gradedData) throw new Error('请先上传动画截图与实景照片');
  const source = $('canvasGraded'), scale = Math.min(1, 1080 / Math.max(source.width, source.height));
  const w = Math.max(2, Math.round(source.width * scale)), h = Math.max(2, Math.round(source.height * scale));
  const anime = document.createElement('canvas'); anime.width = w; anime.height = h;
  // canvasAnimeOverlay 已按实景画框 cover 裁好，正好和叠加模式相同。
  anime.getContext('2d').drawImage($('canvasAnimeOverlay'), 0, 0, w, h);
  const scene = document.createElement('canvas'); scene.width = w; scene.height = h;
  scene.getContext('2d').drawImage(source, 0, 0, w, h);
  const frame = document.createElement('canvas'); frame.width = w; frame.height = h;
  const ctx = frame.getContext('2d');
  const pieces = [new TextEncoder().encode('GIF89a'), gifWord(w), gifWord(h), Uint8Array.of(0xf7, 0, 0), gifColorTable()];
  const frames = 12, delay = Math.round(200 / frames); // 单位 1/100 秒，合计约 2 秒
  for (let index = 0; index < frames; index++) {
    const realOpacity = index / (frames - 1);
    ctx.globalAlpha = 1; ctx.drawImage(anime, 0, 0);
    ctx.globalAlpha = realOpacity; ctx.drawImage(scene, 0, 0); ctx.globalAlpha = 1;
    const pixels = ctx.getImageData(0, 0, w, h).data;
    const indexed = new Uint8Array(w * h);
    for (let p = 0, i = 0; p < indexed.length; p++, i += 4) {
      const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
      const hi = Math.max(r, g, b), lo = Math.min(r, g, b);
      indexed[p] = hi - lo < 18
        ? 216 + Math.round(((r + g + b) / 3) * 39 / 255)
        : Math.round(r / 51) * 36 + Math.round(g / 51) * 6 + Math.round(b / 51);
    }
    const lzw = gifLzw(indexed);
    // disposal=1：下一帧直接盖上上一帧；不写循环扩展，GIF 结束在实景画面。
    pieces.push(Uint8Array.of(0x21, 0xf9, 4, 0x04, delay & 255, (delay >> 8) & 255, 0, 0));
    pieces.push(Uint8Array.of(0x2c), gifWord(0), gifWord(0), gifWord(w), gifWord(h), Uint8Array.of(0, 8), ...gifSubBlocks(lzw));
    setStatus(`正在生成动画→实景 GIF · ${index + 1}/${frames}`);
    await nextPaint();
  }
  pieces.push(Uint8Array.of(0x3b));
  return { blob: new Blob(pieces, { type: 'image/gif' }), width: w, height: h };
}

$('btnExportMorph').addEventListener('click', async () => {
  const btn = $('btnExportMorph'); btn.disabled = true;
  try {
    const gif = await makeAnimeToSceneGif();
    const url = URL.createObjectURL(gif.blob);
    download(url, 'seichi-anime-to-scene.gif');
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    setStatus(`已导出 GIF · ${gif.width}×${gif.height} · 约 2 秒 · 动画渐变为实景`);
  } catch (e) {
    console.error(e);
    setStatus('GIF 导出失败：' + (e.message || e));
  } finally { btn.disabled = false; }
});

// 批量导出复用当前动画截图、滑杆和预设；逐张渲染/释放，避免同时把多张原图塞进内存。
$('btnBatchExport').addEventListener('click', () => {
  if (!state.anime) { setStatus('请先上传动画截图'); return; }
  $('batchFiles').value = '';
  $('batchFiles').click();
});

$('batchFiles').addEventListener('change', async (event) => {
  const files = [...event.target.files].slice(0, 20);
  if (!files.length) return;
  const saved = { photo: state.photo, gradedData: state.gradedData, transform: state.transform, gradeCache: state.gradeCache, cutout: state.cutout };
  const btn = $('btnBatchExport'); btn.disabled = true;
  let completed = 0, failed = 0, failReason = '';
  try {
    state.cutout = null; // 批量套色不意外带入当前手工摆放的角色。
    for (let i = 0; i < files.length; i++) {
      try {
        setStatus(`批量读取 ${i + 1}/${files.length}…`);
        const [data, gps] = await Promise.all([fileToImageData(files[i]), readExifGPS(files[i])]);
        state.photo = {
          imgData: data.imgData, width: data.width, height: data.height, srcUrl: data.url, align: null,
          originalWidth: data.originalWidth || data.width, originalHeight: data.originalHeight || data.height,
          fileName: data.fileName || files[i].name, gps,
        };
        state.gradeCache = null;
        const canvas = await renderFullRes((text) => setStatus(`批量 ${i + 1}/${files.length} · ${text}`));
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', .92));
        if (!blob) throw new Error('JPEG 编码失败');
        const base = (files[i].name || `photo-${i + 1}`).replace(/\.[^.]+$/, '');
        const url = URL.createObjectURL(blob);
        download(url, `${base}-seichi.jpg`);
        setTimeout(() => URL.revokeObjectURL(url), 8000);
        URL.revokeObjectURL(data.url);
        completed++;
      } catch (error) {
        console.error(error); rememberError('batch-export', error); failed++;
        if (!failReason) failReason = error.message || String(error);
      }
    }
  } finally {
    state.photo = saved.photo; state.gradedData = saved.gradedData; state.transform = saved.transform;
    state.gradeCache = saved.gradeCache; state.cutout = saved.cutout;
    if (state.photo) recompute();
    btn.disabled = !state.anime;
    setStatus(`批量导出完成：${completed} 张成功${failed ? `，${failed} 张失败（${failReason}）` : ''}`);
  }
});

// ---------- 找最像的实景（场景嵌入匹配，见 embed.js） ----------
// 逐张「解码→编码→释放」，全程只保留 Top3 的缩略图与 File 引用，几十张也不会撑爆内存。
const MATCH_MAX_FILES = 60;
const MATCH_TOP_K = 3;

function makeMatchThumb(imgData, maxSide = 220) {
  const scale = Math.min(1, maxSide / Math.max(imgData.width, imgData.height));
  const w = Math.max(1, Math.round(imgData.width * scale));
  const h = Math.max(1, Math.round(imgData.height * scale));
  const src = document.createElement('canvas');
  src.width = imgData.width; src.height = imgData.height;
  src.getContext('2d').putImageData(imgData, 0, 0);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(src, 0, 0, w, h);
  return c;
}

// 点选某张推荐结果：走与手动上传完全相同的读取路径
async function useMatchedPhoto(entry, item) {
  try {
    setStatus('读取选中的实景照片…');
    const [data, gps] = await Promise.all([fileToImageData(entry.file), readExifGPS(entry.file)]);
    data.gps = gps;
    $('thumbPhoto').src = data.url; $('thumbPhoto').hidden = false;
    await handlePhotoData(data);
    [...$('matchGrid').children].forEach((el) => el.classList.toggle('selected', el === item));
  } catch (e) { setStatus('读取失败：' + (e.message || e)); }
}

function renderMatchResults(top) {
  const grid = $('matchGrid');
  grid.textContent = '';
  for (const entry of top) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'match-item';
    item.title = entry.file.name || '';
    item.appendChild(entry.thumb);
    const label = document.createElement('span');
    label.textContent = `相似度 ${Math.max(0, Math.round(entry.sim * 100))}%`;
    item.appendChild(label);
    item.addEventListener('click', () => useMatchedPhoto(entry, item));
    grid.appendChild(item);
  }
  $('matchResults').hidden = !top.length;
}

$('btnMatchScene').addEventListener('click', () => {
  if (!state.anime) { setStatus('请先上传动画截图'); return; }
  $('matchFiles').value = '';
  $('matchFiles').click();
});

$('matchFiles').addEventListener('change', async (event) => {
  const files = [...event.target.files].slice(0, MATCH_MAX_FILES);
  if (!files.length || !state.anime) return;
  const btn = $('btnMatchScene'); btn.disabled = true;
  $('matchResults').hidden = true;
  try {
    setStatus('准备找图匹配模型…');
    const fmtMB = (n) => (n / 1048576).toFixed(1);
    const query = await embedImage(state.anime.imgData, {
      onProgress: (r, t) => setStatus(`下载找图匹配模型 ${fmtMB(r)}/${fmtMB(t)} MB…`),
    });
    const top = [];
    let compared = 0, failed = 0, failReason = '';
    for (let i = 0; i < files.length; i++) {
      try {
        setStatus(`比对 ${i + 1}/${files.length}：${files[i].name || ''}`);
        const data = await fileToImageData(files[i]);
        const sim = cosineSimilarity(query, await embedImage(data.imgData));
        const thumb = makeMatchThumb(data.imgData);
        URL.revokeObjectURL(data.url);
        top.push({ file: files[i], sim, thumb });
        top.sort((a, b) => b.sim - a.sim);
        if (top.length > MATCH_TOP_K) top.length = MATCH_TOP_K; // 落选缩略图交给 GC
        compared++;
      } catch (e) {
        console.warn('找图比对失败', files[i]?.name, e);
        failed++;
        if (!failReason) failReason = e.message || String(e); // 把第一条失败原因带给用户（如 HEIC/RAW 指引）
      }
    }
    renderMatchResults(top);
    setStatus(compared
      ? `找图完成：共比对 ${compared} 张${failed ? `，${failed} 张读取失败（${failReason}）` : ''}${event.target.files.length > MATCH_MAX_FILES ? `（超过 ${MATCH_MAX_FILES} 张的部分未参加）` : ''}，点选最像的一张`
      : `找图失败：所选照片都无法读取（${failReason}）`);
  } catch (e) {
    console.error(e); rememberError('scene-match', e);
    setStatus('找图匹配失败：' + (e.message || e));
  } finally { btn.disabled = !state.anime; }
});

// ---------- 参数自动恢复与风格预设 ----------
const SETTINGS_KEY = 'seichi-current-settings-v1';
const PRESETS_KEY = 'seichi-style-presets-v1';
const SETTING_IDS = [
  'mode', 'skyRegion', 'ignoreSub', 'strength', 'satBoost', 'bloom', 'composite',
  'hiresDet', 'mobileSam', 'maskThr', 'maskErode', 'charScale', 'harmonize', 'shadow',
  'shadowOffset', 'grain', 'layout', 'compareTitle', 'comparePlace', 'overlayOpacity',
];

function captureSettings() {
  const values = {};
  for (const id of SETTING_IDS) {
    const el = $(id);
    values[id] = el.type === 'checkbox' ? el.checked : el.value;
  }
  // version 2：charScale 语义从「占画面高%」改为「相对动画同比例%」
  return { version: 2, values, charPos: { ...state.charPos } };
}

function syncControlLabels() {
  const pairs = {
    strength: ['strengthVal', '%'], satBoost: ['satBoostVal', '%'], bloom: ['bloomVal', '%'],
    maskThr: ['maskThrVal', ''], maskErode: ['maskErodeVal', 'px'], charScale: ['charScaleVal', '%'],
    harmonize: ['harmonizeVal', '%'], shadow: ['shadowVal', '%'],
    shadowOffset: ['shadowOffsetVal', '%'], grain: ['grainVal', '%'], overlayOpacity: ['overlayOpacityVal', '%'],
  };
  for (const [id, [label, suffix]] of Object.entries(pairs)) $(label).textContent = $(id).value + suffix;
}

function applySettings(saved, rerender = true) {
  if (!saved || !saved.values) return false;
  const legacyScale = (saved.version || 1) < 2; // 旧档的 charScale 是「占画面高%」，不可沿用
  for (const [id, value] of Object.entries(saved.values)) {
    const el = $(id);
    if (!el || !SETTING_IDS.includes(id)) continue;
    if (legacyScale && id === 'charScale') continue;
    if (el.type === 'checkbox') el.checked = Boolean(value);
    else el.value = String(value);
  }
  if (saved.charPos && Number.isFinite(saved.charPos.cx) && Number.isFinite(saved.charPos.cy)) {
    state.charPos = { cx: saved.charPos.cx, cy: saved.charPos.cy };
  }
  syncControlLabels();
  updateOverlayOpacity();
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

// ---------- 户外强光模式 ----------
// 拉高界面对比/亮度、增强取景叠加可读性；刻意不改调色预览画布以保色准。
const SUNLIGHT_KEY = 'seichi-sunlight';
function setSunlight(on) {
  document.documentElement.classList.toggle('sunlight', on);
  const btn = $('btnSunlight');
  btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  btn.textContent = on ? '☀️ 强光·开' : '☀️ 强光';
  try { localStorage.setItem(SUNLIGHT_KEY, on ? '1' : '0'); } catch { /* 隐私模式 */ }
  if (state.photo) syncCanvasSize();
}
$('btnSunlight').addEventListener('click', () => {
  setSunlight(!document.documentElement.classList.contains('sunlight'));
});
try { if (localStorage.getItem(SUNLIGHT_KEY) === '1') setSunlight(true); } catch { /* 忽略 */ }

// ---------- 模型持久缓存状态 ----------
const MODEL_CACHE = 'seichi-models-v2';
// 运行时实际读取 ISNet 的三个分块（见 ort-env.js），不是同名的整文件。
// 因此离线包也只缓存分块，避免把同一模型下载两遍。
const ISNET_URL = DEVICE.isAppleMobile ? './models/isnet-anime-512-fp16.onnx' : './models/isnet-anime-fp16.onnx';
const ISNET_PARTS = Array.from({ length: 3 }, (_, i) => `${ISNET_URL}.part${String(i).padStart(2, '0')}`);
const AUTO_MODEL_URLS = ['./models/person-detect.onnx', ...ISNET_PARTS];
const SAM_MODEL_URLS = ['./models/sam-encoder.onnx', './models/sam-decoder.onnx'];
const MATCH_MODEL_URLS = [SCENE_EMBED_MODEL_URL];
const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
const RUNTIME_URLS = [
  'ort.webgpu.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
].map((name) => ORT_BASE + name);
const AUTO_CACHE_URLS = [...AUTO_MODEL_URLS, ...RUNTIME_URLS];
const MATCH_CACHE_URLS = [...MATCH_MODEL_URLS, ...RUNTIME_URLS]; // 找图也要能离线，故连运行时一起缓存
let lastDownloadProbe = null;

const cacheKey = (url) => new URL(url, location.href).href;
async function cacheHas(cache, url) {
  return !!(await cache.match(cacheKey(url), { ignoreVary: true }));
}
async function countCached(cache, urls) {
  return (await Promise.all(urls.map((url) => cacheHas(cache, url)))).filter(Boolean).length;
}

async function updateModelCacheStatus() {
  const label = $('modelCacheStatus');
  if (!('caches' in window) || !('serviceWorker' in navigator)) {
    label.textContent = '此浏览器不支持模型持久缓存';
    ['btnCacheModels', 'btnCacheSam', 'btnCacheMatch', 'btnClearModelCache'].forEach((id) => { $(id).disabled = true; });
    return;
  }
  try {
    const cache = await caches.open(MODEL_CACHE);
    const autoCount = await countCached(cache, AUTO_CACHE_URLS);
    const samCount = await countCached(cache, SAM_MODEL_URLS);
    const matchCount = await countCached(cache, MATCH_CACHE_URLS);
    const autoReady = autoCount === AUTO_CACHE_URLS.length;
    const samReady = samCount === SAM_MODEL_URLS.length;
    const matchReady = matchCount === MATCH_CACHE_URLS.length;
    label.textContent = `自动抠图：${autoReady ? '已可离线使用 ✓' : `${autoCount}/${AUTO_CACHE_URLS.length} 个文件`}`
      + ` · SAM 兜底：${samReady ? '已可离线使用 ✓' : `${samCount}/${SAM_MODEL_URLS.length} 个文件`}`
      + ` · 找图匹配：${matchReady ? '已可离线使用 ✓' : `${matchCount}/${MATCH_CACHE_URLS.length} 个文件`}`;
    $('btnCacheModels').textContent = autoReady
      ? '自动抠图离线包已就绪 ✓'
      : autoCount ? `继续下载自动抠图离线包（已完成 ${autoCount}/${AUTO_CACHE_URLS.length}）` : '下载自动抠图离线包（约 127MB）';
    $('btnCacheSam').textContent = samReady
      ? 'SAM 高质量兜底包已就绪 ✓'
      : samCount ? `继续下载 SAM 兜底包（已完成 ${samCount}/${SAM_MODEL_URLS.length}）` : '下载 SAM 高质量兜底包（约 38MB）';
    $('btnCacheMatch').textContent = matchReady
      ? '找图匹配离线包已就绪 ✓'
      : matchCount ? `继续下载找图匹配包（已完成 ${matchCount}/${MATCH_CACHE_URLS.length}）` : '下载找图匹配离线包（约 21MB）';
  } catch (e) { label.textContent = '无法读取模型缓存：' + (e.message || e); }
}

async function downloadOfflinePackage(kind, urls, button) {
  const label = $('modelCacheStatus'); button.disabled = true;
  try {
    await navigator.serviceWorker.ready;
    if (navigator.storage?.persist) await navigator.storage.persist().catch(() => false);
    const cache = await caches.open(MODEL_CACHE);
    let done = await countCached(cache, urls);
    for (const url of urls) {
      if (await cacheHas(cache, url)) continue;
      label.textContent = `正在下载${kind} ${done + 1}/${urls.length}…请保持此页面打开`;
      // cache.add 会等待完整响应写入 Cache Storage。网络中断后，已完成的文件保留；
      // 再点一次按钮只补未完成的文件，而不是从头下载整个模型。
      await cache.add(cacheKey(url));
      done++;
    }
    await updateModelCacheStatus();
  } catch (e) {
    try {
      const cache = await caches.open(MODEL_CACHE);
      const done = await countCached(cache, urls);
      label.textContent = `${kind}下载暂停（已完成 ${done}/${urls.length}）· 网络恢复后点“继续下载”即可`;
    } catch { label.textContent = `${kind}下载失败：${e.message || e}`; }
  } finally { button.disabled = false; }
}

$('btnCacheModels').addEventListener('click', () => downloadOfflinePackage('自动抠图离线包', AUTO_CACHE_URLS, $('btnCacheModels')));
$('btnCacheSam').addEventListener('click', () => downloadOfflinePackage('SAM 高质量兜底包', SAM_MODEL_URLS, $('btnCacheSam')));
$('btnCacheMatch').addEventListener('click', () => downloadOfflinePackage('找图匹配离线包', MATCH_CACHE_URLS, $('btnCacheMatch')));

// 只发 HEAD 请求，不下载数十 MB 的模型本体。用于定位“模型下载失败”是卡在
// GitHub Pages、分块模型、还是 jsDelivr 的 ONNX 运行环境。
const MODEL_DOWNLOAD_PROBES = [
  ['本站 · 人物检测模型', './models/person-detect.onnx'],
  ['本站 · 抠图模型分块', './models/isnet-anime-fp16.onnx.part00'],
  ['jsDelivr · ONNX 运行环境', `${ORT_BASE}ort-wasm-simd-threaded.wasm`],
];

async function probeModelDownloadRoute(name, url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12_000);
  const target = new URL(url, location.href);
  target.searchParams.set('__probe', String(Date.now()));
  const started = performance.now();
  try {
    const response = await fetch(target, { method: 'HEAD', cache: 'no-store', signal: controller.signal });
    const elapsedMs = Math.round(performance.now() - started);
    const size = Number(response.headers.get('content-length'));
    return {
      name, url: target.origin + target.pathname, ok: response.ok, status: response.status, elapsedMs,
      size: Number.isFinite(size) && size > 0 ? size : null,
      error: response.ok ? null : `HTTP ${response.status}`,
    };
  } catch (error) {
    return {
      name, url: target.origin + target.pathname, ok: false, status: null,
      elapsedMs: Math.round(performance.now() - started), size: null,
      error: error?.name === 'AbortError' ? '12 秒内无响应' : (error?.message || String(error)),
    };
  } finally {
    clearTimeout(timeout);
  }
}

function formatProbeSize(bytes) {
  return bytes ? `${Math.round(bytes / 1024 / 1024)}MB` : '大小未提供';
}

$('btnTestModelDownload').addEventListener('click', async () => {
  const button = $('btnTestModelDownload');
  const output = $('modelDownloadProbe');
  button.disabled = true;
  output.hidden = false; output.className = 'download-probe';
  output.textContent = '正在检查下载线路…（只测试连接，不下载完整模型）';
  const routes = [];
  for (const [name, url] of MODEL_DOWNLOAD_PROBES) {
    output.textContent = `正在检查：${name}…`;
    routes.push(await probeModelDownloadRoute(name, url));
  }
  let storage = null;
  try { storage = await navigator.storage?.estimate?.() || null; } catch { /* 浏览器不提供时省略 */ }
  lastDownloadProbe = { generatedAt: new Date().toISOString(), online: navigator.onLine, routes, storage };
  const allOk = routes.every((route) => route.ok);
  output.classList.add(allOk ? 'ok' : 'problem');
  output.textContent = [
    allOk ? '下载线路正常：' : '发现可能影响下载的问题：',
    ...routes.map((route) => route.ok
      ? `✓ ${route.name} · HTTP ${route.status} · ${route.elapsedMs}ms · ${formatProbeSize(route.size)}`
      : `✕ ${route.name} · ${route.error || '连接失败'} · ${route.elapsedMs}ms`),
    '可点“导出诊断信息”并把 JSON 与截图一起反馈。',
  ].join('\n');
  button.disabled = false;
});

$('btnClearModelCache').addEventListener('click', async () => {
  const label = $('modelCacheStatus');
  $('btnClearModelCache').disabled = true;
  try {
    await releaseAllSessions();
    await caches.delete(MODEL_CACHE);
    label.textContent = 'AI 离线包已删除；基础调色仍可离线使用';
    await updateModelCacheStatus();
  } catch (e) { label.textContent = '删除离线包失败：' + (e.message || e); }
  finally { $('btnClearModelCache').disabled = false; }
});

$('btnExportDiagnostics').addEventListener('click', async () => {
  const info = {
    generatedAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    online: navigator.onLine,
    crossOriginIsolated: self.crossOriginIsolated,
    device: DEVICE,
    image: {
      anime: state.anime ? `${state.anime.width}×${state.anime.height}` : null,
      photo: state.photo ? `${state.photo.originalWidth}×${state.photo.originalHeight}` : null,
      preview: state.photo ? `${state.photo.width}×${state.photo.height}` : null,
      hasCharacter: !!state.cutout,
    },
    storage: null,
    modelDownloadProbe: lastDownloadProbe,
    recentErrors,
  };
  try { info.storage = await navigator.storage?.estimate?.() || null; } catch { /* 不支持时省略 */ }
  const blob = new Blob([JSON.stringify(info, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  download(url, 'seichi-diagnostics.json');
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  setStatus('已导出诊断信息；将该 JSON 与问题截图一起发来即可');
});

$('deviceStatus').textContent = IS_MOBILE
  ? `${DEVICE.isIPhone ? 'iPhone' : DEVICE.isIPad ? 'iPad' : '移动端'}保护：预览最长边 ${MAX_DIM}px，导出上限约 ${Math.round(EXPORT_MAX_PIXELS / 1e6)}MP`
  : `桌面预览最长边 ${MAX_DIM}px；导出按原始分辨率分块处理`;
$('mobileSamRow').hidden = !DEVICE.isAppleMobile;

setStatus('请上传动画截图与实景照片');
updateWorkflow();

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
// shape：rect=拖矩形（默认）| ellipse=拖圆 | free=自由画笔。
// 三种形状统一落成 pts 多边形（圆用 48 边形逼近），包围盒/质心/擦除逻辑全复用。
const lassoState = {
  pts: [], drawing: false, busy: false, mode: 'extract', shape: 'rect',
  keepMode: false, keepDrawing: false, keepStrokes: [], stage: 'select', maskOverlay: null,
};

function lassoReady() {
  if (lassoState.pts.length < 4) return false;
  const box = lassoBBox();
  return box.w >= 12 && box.h >= 12;
}

function updateLassoTip() {
  if (lassoState.stage === 'refine') {
    $('lassoTip').textContent = lassoState.keepMode
      ? '蓝色画笔：刷出的部分会立刻并入识别目标'
      : '识别结果以外的区域已蒙版；如有漏掉的脸、手或发丝，请开启蓝色强制保留画笔';
    return;
  }
  if (lassoState.keepMode) {
    $('lassoTip').textContent = '蓝色画笔：涂过的部分会强制保留，不受算法抠像结果影响';
    return;
  }
  const verb = lassoState.mode === 'erase'
    ? '要从抠图中擦掉的区域'
    : lassoState.mode === 'algorithm' ? '要由算法抠取的角色' : '要由模型抠取的角色';
  $('lassoTip').textContent = lassoState.shape === 'free'
    ? `随手圈出${verb}——不必精确贴边`
    : `按住拖出一个框住${verb}的${lassoState.shape === 'rect' ? '矩形' : '圆'}——不必精确贴边`;
}

function updateLassoGuide() {
  const guide = $('lassoGuide');
  guide.children[0].classList.toggle('active', lassoState.stage === 'select');
  guide.children[1].classList.toggle('active', lassoState.stage === 'refine');
}

function rebuildLassoMaskOverlay() {
  if (!state.rawAlpha || !state.rawW || !state.rawH) { lassoState.maskOverlay = null; return; }
  const overlay = document.createElement('canvas'); overlay.width = state.rawW; overlay.height = state.rawH;
  const image = new ImageData(overlay.width, overlay.height);
  for (let i = 0, p = 0; i < state.rawAlpha.length; i++, p += 4) {
    // 识别目标保持完整；其它部分以深色半透明蒙版呈现，便于明确看出遗漏。
    image.data[p + 3] = Math.round((255 - state.rawAlpha[i]) * .72);
  }
  overlay.getContext('2d').putImageData(image, 0, 0);
  lassoState.maskOverlay = overlay;
}

function lassoRedraw() {
  const c = $('lassoCanvas'), ctx = c.getContext('2d');
  ctx.putImageData(state.anime.imgData, 0, 0);
  if (lassoState.stage === 'refine' && lassoState.maskOverlay) ctx.drawImage(lassoState.maskOverlay, 0, 0);
  if (lassoState.stage === 'select' && lassoState.pts.length > 1) {
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
  if (lassoState.keepStrokes.length) {
    ctx.save();
    ctx.strokeStyle = 'rgba(88,166,255,.95)';
    ctx.fillStyle = 'rgba(88,166,255,.18)';
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (const stroke of lassoState.keepStrokes) {
      if (!stroke.pts.length) continue;
      ctx.lineWidth = stroke.width;
      ctx.beginPath(); ctx.moveTo(stroke.pts[0][0], stroke.pts[0][1]);
      for (let i = 1; i < stroke.pts.length; i++) ctx.lineTo(stroke.pts[i][0], stroke.pts[i][1]);
      ctx.stroke();
      if (stroke.pts.length === 1) { ctx.beginPath(); ctx.arc(stroke.pts[0][0], stroke.pts[0][1], stroke.width / 2, 0, Math.PI * 2); ctx.fill(); }
    }
    ctx.restore();
  }
}

function updateKeepBrushUI() {
  const available = lassoState.mode !== 'erase' && lassoState.stage === 'refine';
  $('btnLassoKeep').hidden = !available;
  $('lassoKeepSize').hidden = !available || !lassoState.keepMode;
  $('btnLassoKeep').classList.toggle('keep-on', lassoState.keepMode);
  $('btnLassoKeep').setAttribute('aria-pressed', String(lassoState.keepMode));
  $('lassoShapes').hidden = lassoState.stage === 'refine';
  $('btnLassoClear').hidden = lassoState.stage === 'refine';
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

function openLasso(mode = 'extract') {
  if (!state.anime) return;
  const c = $('lassoCanvas');
  c.width = state.anime.width; c.height = state.anime.height;
  lassoState.pts = []; lassoState.drawing = false; lassoState.keepMode = false; lassoState.keepDrawing = false; lassoState.keepStrokes = [];
  lassoState.stage = 'select'; lassoState.maskOverlay = null;
  lassoState.mode = mode;
  updateLassoTip();
  updateLassoGuide();
  updateKeepBrushUI();
  $('btnLassoRun').textContent = mode === 'erase' ? '擦除圈选区域'
    : mode === 'algorithm' ? '算法抠取圈选区域' : '模型抠取圈选区域';
  $('btnLassoRun').disabled = true;
  lassoRedraw();
  $('lassoModal').hidden = false;
}

function closeLasso() { $('lassoModal').hidden = true; }

function extractAlgorithmInRegion(box) {
  const crop = document.createElement('canvas');
  crop.width = box.w; crop.height = box.h;
  // 用负坐标直接把原图指定区域拷进小画布，算法只面对用户圈出的局部。
  crop.getContext('2d').putImageData(state.anime.imgData, -box.x, -box.y);
  const res = extractForeground(crop.getContext('2d').getImageData(0, 0, box.w, box.h), { centerBias: 0.5 });
  return {
    box: { ...box, score: 1 }, rect: { ...box }, score: 1,
    alpha: res.alpha, empty: !res.bbox, via: 'algorithm', manual: true,
  };
}

function applyForcedKeepStrokes(strokes) {
  if (!strokes.length || !state.rawAlpha || !state.rawW || !state.rawH) return 0;
  const mask = document.createElement('canvas'); mask.width = state.rawW; mask.height = state.rawH;
  const ctx = mask.getContext('2d');
  ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff'; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  for (const stroke of strokes) {
    if (!stroke.pts.length) continue;
    ctx.lineWidth = stroke.width;
    ctx.beginPath(); ctx.moveTo(stroke.pts[0][0], stroke.pts[0][1]);
    for (let i = 1; i < stroke.pts.length; i++) ctx.lineTo(stroke.pts[i][0], stroke.pts[i][1]);
    ctx.stroke();
    if (stroke.pts.length === 1) { ctx.beginPath(); ctx.arc(stroke.pts[0][0], stroke.pts[0][1], stroke.width / 2, 0, Math.PI * 2); ctx.fill(); }
  }
  const pixels = ctx.getImageData(0, 0, mask.width, mask.height).data;
  let added = 0;
  if (!state.forcedAlpha || state.forcedAlpha.length !== state.rawAlpha.length) state.forcedAlpha = new Uint8ClampedArray(state.rawAlpha.length);
  for (let i = 0, p = 3; p < pixels.length; i++, p += 4) {
    if (pixels[p] && !state.forcedAlpha[i]) { state.forcedAlpha[i] = 255; added++; }
    if (pixels[p]) state.rawAlpha[i] = 255;
  }
  return added;
}

// 圈选核心：模型模式走 Worker；算法模式只处理圈选区域，避免把整幅画面当作主体猜。
async function runLassoBox(box, centroid) {
  if (lassoState.busy) return null;
  lassoState.busy = true;
  $('btnLassoRun').disabled = true;
  const onStage = (s) => { $('extractStatus').textContent = s; };
  const onProgress = (recv, total) => {
    $('extractStatus').textContent = `下载模型 ${(recv / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)}MB`;
  };
  try {
    let found;
    if (lassoState.mode === 'algorithm') {
      onStage('正在对圈选区域进行算法抠像…');
      await nextPaint(); // 先显示状态文字，再开始轻量同步处理。
      found = [extractAlgorithmInRegion(box)];
    } else {
      const result = await runAIInWorker(state.anime.imgData, {
        job: 'region', box, samPoints: centroid ? [centroid] : [], onStage, onProgress,
        samFallback: true, mobileModel: DEVICE.isAppleMobile,
      });
      found = result.chars;
    }
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
    const method = lassoState.mode === 'algorithm' ? '算法' : '模型';
    $('extractStatus').textContent = ok === 0
      ? `圈选区域没有找到明显前景——试着圈大一点、或让圈更贴近角色`
      : `${method}已识别 ${ok} 个角色 · 现在可用蓝色画笔补回遗漏部分`;
    lassoState.stage = 'refine';
    lassoState.keepMode = false;
    rebuildLassoMaskOverlay();
    updateLassoGuide(); updateLassoTip(); updateKeepBrushUI();
    $('btnLassoRun').textContent = '完成抠像';
    $('btnLassoRun').disabled = false;
    lassoRedraw();
    return found;
  } catch (e) {
    console.error(e);
    $('extractStatus').textContent = '圈选抠图失败：' + (e.message || e);
    return null;
  } finally {
    lassoState.busy = false;
    $('btnLassoRun').disabled = !lassoReady();
  }
}

$('btnLasso').addEventListener('click', openLasso);
$('btnEraseMask').addEventListener('click', () => openLasso('erase'));
$('btnLassoClose').addEventListener('click', closeLasso);
$('btnLassoClear').addEventListener('click', () => { lassoState.pts = []; lassoState.keepStrokes = []; $('btnLassoRun').disabled = true; lassoRedraw(); });
$('btnLassoKeep').addEventListener('click', () => {
  if (lassoState.mode === 'erase' || lassoState.stage !== 'refine') return;
  lassoState.keepMode = !lassoState.keepMode;
  updateLassoTip(); updateKeepBrushUI(); lassoRedraw();
});
$('lassoKeepBrush').addEventListener('input', () => lassoRedraw());
$('btnLassoRun').addEventListener('click', async () => {
  if (lassoState.stage === 'refine') {
    closeLasso();
    $('extractStatus').textContent = '已完成圈选抠像 · 可继续调阈值、收边或拖拽角色';
    return;
  }
  const box = lassoBBox();
  if (box.w < 12 || box.h < 12) { $('extractStatus').textContent = '圈得太小了，重新圈一下'; return; }
  if (lassoState.mode === 'erase') {
    const pts = lassoState.pts;
    const inside = (x, y) => {
      let hit = false;
      for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const [xi, yi] = pts[i], [xj, yj] = pts[j];
        if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) hit = !hit;
      }
      return hit;
    };
    for (let y = box.y; y < box.y + box.h; y++) for (let x = box.x; x < box.x + box.w; x++) {
      if (inside(x + .5, y + .5)) {
        const i = y * state.rawW + x;
        state.rawAlpha[i] = 0;
        if (state.forcedAlpha) state.forcedAlpha[i] = 0;
      }
    }
    closeLasso();
    const coverage = applyRefine(false);
    $('extractStatus').textContent = `已擦除圈选区域 · 当前角色占画面 ${(coverage * 100).toFixed(0)}%`;
    return;
  }
  let sx = 0, sy = 0;
  for (const [x, y] of lassoState.pts) { sx += x; sy += y; }
  const centroid = [sx / lassoState.pts.length, sy / lassoState.pts.length];
  await runLassoBox(box, centroid);
});

(function setupLassoDraw() {
  const c = $('lassoCanvas');
  let anchor = null; // rect/ellipse 的起始角
  const toImg = (e) => {
    const r = c.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width * c.width, (e.clientY - r.top) / r.height * c.height];
  };
  const rectPoly = (a, b) => [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]];
  const ellipsePoly = (a, b, n = 48) => {
    const cx = (a[0] + b[0]) / 2, cy = (a[1] + b[1]) / 2;
    const rx = Math.abs(b[0] - a[0]) / 2, ry = Math.abs(b[1] - a[1]) / 2;
    return Array.from({ length: n }, (_, i) => {
      const t = i / n * Math.PI * 2;
      return [cx + rx * Math.cos(t), cy + ry * Math.sin(t)];
    });
  };
  c.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    try { c.setPointerCapture(e.pointerId); } catch { /* 某些环境/合成事件会抛，不影响绘制 */ }
    lassoState.drawing = true;
    anchor = toImg(e);
    if (lassoState.keepMode) {
      lassoState.keepDrawing = true;
      lassoState.keepStrokes.push({ pts: [anchor], width: Number($('lassoKeepBrush').value) });
    } else {
      lassoState.keepDrawing = false;
      lassoState.pts = lassoState.shape === 'free' ? [anchor] : [];
    }
    lassoRedraw();
  });
  c.addEventListener('pointermove', (e) => {
    if (!lassoState.drawing) return;
    const p = toImg(e);
    if (lassoState.keepDrawing) {
      const stroke = lassoState.keepStrokes[lassoState.keepStrokes.length - 1];
      const last = stroke.pts[stroke.pts.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 1) { stroke.pts.push(p); lassoRedraw(); }
    } else if (lassoState.shape === 'free') {
      const last = lassoState.pts[lassoState.pts.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 3) { lassoState.pts.push(p); lassoRedraw(); }
    } else {
      lassoState.pts = lassoState.shape === 'rect' ? rectPoly(anchor, p) : ellipsePoly(anchor, p);
      lassoRedraw();
    }
  });
  const end = () => {
    if (!lassoState.drawing) return;
    const lastKeepStroke = lassoState.keepDrawing ? lassoState.keepStrokes[lassoState.keepStrokes.length - 1] : null;
    lassoState.drawing = false; lassoState.keepDrawing = false;
    if (lastKeepStroke && lassoState.stage === 'refine') {
      const forced = applyForcedKeepStrokes([lastKeepStroke]);
      if (forced) {
        applyRefine(false);
        rebuildLassoMaskOverlay();
        $('extractStatus').textContent = `已强制保留 ${forced.toLocaleString()} 个像素 · 可继续补画或点“完成抠像”`;
      }
      lassoRedraw();
    }
    $('btnLassoRun').disabled = !lassoReady() || lassoState.busy;
  };
  c.addEventListener('pointerup', end);
  c.addEventListener('pointercancel', end);

  $('lassoShapes').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-shape]');
    if (!btn) return;
    lassoState.shape = btn.dataset.shape; lassoState.keepMode = false; lassoState.keepDrawing = false;
    [...$('lassoShapes').children].forEach((el) => el.classList.toggle('shape-on', el === btn));
    lassoState.pts = []; lassoState.drawing = false;
    updateKeepBrushUI(); updateLassoTip();
    $('btnLassoRun').disabled = true;
    updateLassoTip();
    lassoRedraw();
  });
})();

// ---------- 现场取景拍摄 ----------
// 摄像头拍出的 canvas 是全分辨率（grabFrame ~8.7MP / takePhoto 12MP）。
// 转成与 fileToImageData 同构的数据：全图 blob 作 srcUrl（全分辨率导出重放用），
// 另降到 MAX_DIM 作预览 imgData。
async function canvasToPhotoData(canvas) {
  const ow = canvas.width, oh = canvas.height;
  const blob = await new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.95));
  const url = URL.createObjectURL(blob);
  const scale = Math.min(1, MAX_DIM / Math.max(ow, oh));
  const w = Math.round(ow * scale), h = Math.round(oh * scale);
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  c.getContext('2d').drawImage(canvas, 0, 0, w, h);
  return {
    imgData: c.getContext('2d').getImageData(0, 0, w, h), width: w, height: h, url,
    originalWidth: ow, originalHeight: oh, fileName: 'seichi-shot.jpg',
  };
}

$('btnShoot').addEventListener('click', async () => {
  if (!state.anime) { setStatus('请先上传动画截图，取景时叠加它做参考'); return; }
  const btn = $('btnShoot'); btn.disabled = true;
  try {
    const canvas = await launchViewfinder(state.anime.imgData);
    if (canvas) {
      const data = await canvasToPhotoData(canvas);
      $('thumbPhoto').src = data.url; $('thumbPhoto').hidden = false;
      await handlePhotoData(data);
    }
  } catch (e) {
    console.error(e); setStatus('取景失败：' + (e.message || e));
  } finally {
    btn.disabled = false;
  }
});

window.__qa = { state, renderFullRes, enterAlignMode, applyAlignCrop, alignState, recompute, runLassoBox, launchViewfinder };

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
