// app.js — UI 编排：上传、迁移、对比预览、导出
import {
  imageDataToLab, labStats, makeLabTransform, makeLumaCdfMap, makeGradeTransform,
  applyTransfer, applyTransferRegioned, skyMask, makeBloomLayer, applyBloom, extractPalette, generateCubeLUT,
} from './color.js?v=20260712d';
import { extractForeground, cutoutCanvas, cleanupAlpha, alphaBBox, fillNearlyClosedHoles } from './segment.js?v=20260718-stable-rollback';
import { mergeCharacterAlphas } from './ai-segment.js';
import { releaseAllSessions, MODEL_BASE } from './ort-env.js';
import { embedImage, cosineSimilarity, SCENE_EMBED_MODEL_URL } from './embed.js';
import { profile as DEVICE } from './platform.js';
import { launchViewfinder } from './camera/viewfinder.js?v=20260718-reference-switch';

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
  rawAlpha: null,      // 抠图原始 alpha（Uint8，与 anime 同尺寸），只属算法结果，供阈值/收边重算
  maskOps: [],         // 手工修补操作，按时间序重放：keep(补画)/keepRegion(点选补块)/erase(橡皮擦)
  opsOverlay: null,    // maskOps 重放结果缓存 Uint8Array：0 无操作 / 1 强制保留 / 2 强制擦除
  finalAlpha: null,    // applyRefine 的最终输出 alpha（蒙版预览与点选连通域都以它为准）
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
// 耗时操作的总状态仍保留在页面底部，但下载/推理进度也紧贴触发按钮，
// 让用户不用在面板里寻找“刚才点的按钮到底有没有反应”。
function setButtonLoad(id, text = '') {
  const el = $(id);
  if (!el) return;
  el.textContent = text;
  el.hidden = !text;
}
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
  // 界面阶段：两张图都到位后进入 edit——素材区收窄、调色控件上浮强调（见 style.css [data-phase]）
  document.documentElement.dataset.phase = complete.anime && complete.photo ? 'edit' : 'setup';
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
    img.crossOrigin = 'anonymous'; // 跨域图（如 anitabi CDN）需带 CORS 才能读像素；同源无副作用
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
  ['btnOpenExportHub', 'btnExportImg', 'btnExportCompare', 'btnExportCompareLayout', 'btnExportWipe', 'btnExportMorph', 'btnExportApng', 'btnExportLut', 'btnBatchExport'].forEach(id => $(id).disabled = false);
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
  syncStackCanvasSize();
}

// “上下”预览把两张完整图片作为一个整体缩放：共用同一显示宽度，
// 两个画布的边缘直接相接，不再各自在半块面板中居中而留下中缝。
function syncStackCanvasSize() {
  const compare = $('compare');
  if (!compare.classList.contains('compare-stack-mode')) return;
  const animeCanvas = $('canvasAnimeStack'), gradedCanvas = $('canvasGraded');
  if (!animeCanvas.width || !animeCanvas.height || !gradedCanvas.width || !gradedCanvas.height) return;
  const totalHeightPerWidth = animeCanvas.height / animeCanvas.width + gradedCanvas.height / gradedCanvas.width;
  const width = Math.min(compare.clientWidth, compare.clientHeight / totalHeightPerWidth);
  const animeHeight = width * animeCanvas.height / animeCanvas.width;
  const gradedHeight = width * gradedCanvas.height / gradedCanvas.width;
  for (const [panel, canvas, height] of [
    [$('stackAnimePanel'), animeCanvas, animeHeight],
    [$('stackGradedPanel'), gradedCanvas, gradedHeight],
  ]) {
    panel.style.width = width + 'px'; panel.style.height = height + 'px';
    canvas.style.width = width + 'px'; canvas.style.height = height + 'px';
  }
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

// ---------- 手工修补操作（maskOps）：补画 / 点选补块 / 橡皮擦 ----------
// rawAlpha 永远只属算法结果；一切手工修正记录成操作列表，按时间序重放成 opsOverlay，
// 在 applyRefine 里盖到算法输出上。撤销 = 弹出最后一项全量重放。
// 这样勾选角色重建 rawAlpha、调阈值/收边都不会丢手工修正，且每一步可逆。
function rebuildOpsOverlay() {
  const w = state.rawW, h = state.rawH;
  if (!w || !h || !state.maskOps.length) { state.opsOverlay = null; return; }
  const overlay = new Uint8Array(w * h);
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  for (const op of state.maskOps) {
    if (op.type === 'keepRegion') {
      for (const i of op.idx) overlay[i] = 1;
      continue;
    }
    // 橡皮擦在落笔时就记录“当时已是目标”的像素索引；因此无论框到哪里，
    // 都不会把原本未选中的暗区写进操作层。
    if (op.type === 'erase' && op.idx) {
      for (const i of op.idx) overlay[i] = 2;
      continue;
    }
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff';
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    if (op.type === 'keep') {
      const { pts, width } = op.stroke;
      ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.stroke();
      if (pts.length === 1) { ctx.beginPath(); ctx.arc(pts[0][0], pts[0][1], width / 2, 0, Math.PI * 2); ctx.fill(); }
    } else { // erase：圈选多边形填充
      const pts = op.pts;
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath(); ctx.fill();
    }
    const d = ctx.getImageData(0, 0, w, h).data;
    const v = op.type === 'keep' ? 1 : 2;
    for (let i = 0, p = 3; p < d.length; i++, p += 4) if (d[p]) overlay[i] = v;
  }
  state.opsOverlay = overlay;
}

function refreshMaskUndoButtons() {
  const none = !state.maskOps.length;
  $('btnMaskUndo').disabled = none;
  const inModal = $('btnLassoUndo');
  if (inModal) inModal.disabled = none;
}

// 提交一次修补：入列 → 重放 → 重算遮罩。返回 applyRefine 的覆盖率。
function pushMaskOp(op) {
  state.maskOps.push(op);
  rebuildOpsOverlay();
  const cov = applyRefine(false);
  refreshMaskUndoButtons();
  return cov;
}

function undoMaskOp() {
  if (!state.maskOps.length) return;
  state.maskOps.pop();
  rebuildOpsOverlay();
  applyRefine(false);
  refreshMaskUndoButtons();
  if (!$('lassoModal').hidden && (lassoState.stage === 'refine' || lassoState.mode === 'erase')) {
    rebuildLassoMaskOverlay();
    lassoRedraw();
    updateKeepBrushUI();
  }
  $('extractStatus').textContent = state.maskOps.length
    ? `已撤销 · 还剩 ${state.maskOps.length} 步修补可撤销`
    : '已撤销全部手工修补';
}

// ---------- AI 互斥：模型/圈选/找图共用一个开关，避免双 ORT 堆并存 ----------
function setAIBusy(on) {
  state.aiBusy = on;
  refreshAIEntryButtons();
}

// 各 AI 入口按钮的使能条件集中在这里；busy 时全部禁用
function refreshAIEntryButtons() {
  const busy = state.aiBusy;
  $('btnExtract').disabled = busy || !state.anime;
  $('btnExtractAI').disabled = busy || !state.anime;
  $('btnLasso').disabled = busy || !state.anime;
  $('btnMatchScene').disabled = busy || (!state.anime && !state.photo);
  $('btnEraseMask').disabled = busy || !state.cutout;
  refreshCharacterResetButton();
}

function refreshCharacterResetButton() {
  // “两张图刚放上来”的角色状态 = 没有任何抠图、遮罩编辑或角色候选项。
  const hasCharacterWork = !!(state.cutout || state.rawAlpha || state.charSeg || state.maskOps.length);
  $('btnResetCharacter').disabled = state.aiBusy || !hasCharacterWork;
}

// 三种抠图入口（整图模型 / 圈选模型 / 圈选算法）从同一个干净基线起跑。
// 这样任一路径留下的候选框、补画、点选和橡皮擦都不会污染下一条路径的结果。
function prepareIndependentCutout() {
  closeLasso();
  state.cutout = null;
  state.rawAlpha = null; state.rawW = 0; state.rawH = 0; state.finalAlpha = null;
  state.maskOps = []; state.opsOverlay = null;
  state.charBase = null; state.charDraw = null; state.charPos = { cx: 0.5, cy: 0.62 };
  state.harmonizedCache = null;
  setCharSeg(null);
  setCharLock(false);
  setCharScale(100);
  $('btnExportCharacter').disabled = true;
  refreshMaskUndoButtons();
  refreshAIEntryButtons();
  redrawComposite();
  updateWorkflow();
}

function resetCharacterComposite() {
  if (state.aiBusy) return;
  closeLasso();
  state.cutout = null;
  state.rawAlpha = null; state.rawW = 0; state.rawH = 0; state.finalAlpha = null;
  state.maskOps = []; state.opsOverlay = null;
  state.charBase = null; state.charDraw = null; state.charPos = { cx: 0.5, cy: 0.62 };
  state.harmonizedCache = null;
  setCharSeg(null);
  $('maskThr').value = 110; $('maskThrVal').textContent = '110';
  $('maskErode').value = 0; $('maskErodeVal').textContent = '0px';
  $('maskFilter').checked = true; $('maskClose').checked = true;
  $('harmonize').value = 35; $('harmonizeVal').textContent = '35%';
  $('shadow').value = 25; $('shadowVal').textContent = '25%';
  $('shadowOffset').value = 0; $('shadowOffsetVal').textContent = '0%';
  $('grain').value = 12; $('grainVal').textContent = '12%';
  $('composite').checked = true;
  setCharLock(false);
  setCharScale(100);
  $('btnExportCharacter').disabled = true;
  refreshMaskUndoButtons();
  refreshAIEntryButtons();
  redrawComposite();
  updateWorkflow();
  $('extractStatus').textContent = state.anime ? '角色合成已重置 · 可重新选择模型、圈选或算法抠像' : '先上传动画截图';
}

// 抠图后/调参后：从 rawAlpha 重建清理过的角色 cutout
function applyRefine(resetPos) {
  if (!state.rawAlpha || !state.anime) return;
  const w = state.rawW, h = state.rawH;
  const thr = parseInt($('maskThr').value, 10);
  const erode = parseInt($('maskErode').value, 10);
  const clean = cleanupAlpha(state.rawAlpha, w, h, {
    thr, erode, featherR: 2,
    filter: $('maskFilter').checked,
    close: $('maskClose').checked,
  });
  // 手工修补层（补画/点选/擦除的时间序重放结果）盖在算法输出之上
  if (state.opsOverlay?.length === clean.length) {
    const ov = state.opsOverlay;
    for (let i = 0; i < clean.length; i++) {
      if (ov[i] === 1) clean[i] = 255;
      else if (ov[i] === 2) clean[i] = 0;
    }
  }
  state.finalAlpha = clean;
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
  refreshAIEntryButtons();
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
  state.cutout = null; state.rawAlpha = null; setCharSeg(null); invalidateHarmonize();
  state.maskOps = []; state.opsOverlay = null; state.finalAlpha = null;
  refreshMaskUndoButtons();
  refreshAIEntryButtons();
  $('matchResults').hidden = true; // 旧结果按旧截图排序，换截图后作废
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
  refreshAIEntryButtons();
  updateWorkflow();
  setStatus(`照片已读取 · ${data.fromRawPreview ? 'RAW 内嵌预览' : '原图'} ${state.photo.originalWidth}×${state.photo.originalHeight} · 预览 ${data.width}×${data.height}`);
  recompute();
}

bindDrop('dropAnime', 'fileAnime', 'thumbAnime', handleAnimeData);
bindDrop('dropPhoto', 'filePhoto', 'thumbPhoto', handlePhotoData);

// 从 anitabi 地图跳转载入：?url=<巡礼点动画截图>，可选 name/bid/pid/g 作展示与预设标识。
// 仅接受 https 且 anitabi.cn 域名的图，避免被构造链接载入任意外部图片。
async function loadFromQuery() {
  const params = new URLSearchParams(location.search);
  const url = params.get('url');
  if (!url) return;
  let u;
  try { u = new URL(url); } catch { return; }
  if (u.protocol !== 'https:' || !/(^|\.)anitabi\.cn$/i.test(u.hostname)) {
    console.warn('忽略不受信任的跳转图片来源：', url); // 静默回到空状态，不打断正常上传引导
    return;
  }
  const name = params.get('name') || '';
  state.fromMap = {
    name, bid: params.get('bid') || '', pid: params.get('pid') || '', g: params.get('g') || '',
  };
  try {
    setStatus(name ? `正在载入巡礼点「${name}」的动画截图…` : '正在载入动画截图…');
    const data = await urlToImageData(url);
    const thumb = $('thumbAnime'); // 与 bindDrop 一致：更新动画区缩略图
    if (thumb) { thumb.src = data.url; thumb.hidden = false; }
    await handleAnimeData(data);
    setStatus(name
      ? `已载入「${name}」的动画截图 · 现在上传你在当地拍的照片即可开始调色`
      : '动画截图已载入 · 现在上传你拍的实景照片开始调色');
  } catch (e) {
    setStatus('动画截图载入失败（' + (e.message || e) + '）· 你仍可手动上传');
  }
}
loadFromQuery();

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
$('btnResetCharacter').addEventListener('click', (e) => {
  // summary 内的按钮不应顺带折叠/展开角色面板。
  e.preventDefault(); e.stopPropagation();
  resetCharacterComposite();
});

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
['maskFilter', 'maskClose'].forEach((id) => $(id).addEventListener('change', () => applyRefine(false)));

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
    const worker = new Worker('./ai-worker.js?v=20260718-stable-rollback', { type: 'module', name: 'seichi-ai-once' });
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
  prepareIndependentCutout();
  openLasso('algorithm');
});

$('btnExtractAI').addEventListener('click', async () => {
  if (!state.anime) return;
  if (state.aiBusy) { $('extractStatus').textContent = 'AI 任务进行中，请稍候…'; return; }
  prepareIndependentCutout();
  setAIBusy(true);
  $('btnExportImg').disabled = true;
  const report = (text) => { $('extractStatus').textContent = text; setButtonLoad('loadExtractAI', text); };
  report('准备模型…');
  const onProgress = (recv, total) => {
    report(`下载 ${(recv / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)}MB`);
  };
  const onStage = (s) => report(s);
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
    setAIBusy(false);
    setButtonLoad('loadExtractAI');
    if (state.gradedData) $('btnExportImg').disabled = false;
    updateWorkflow();
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

function makeCompareCanvas(maxWidth = 0, layoutOverride = '') {
  const layout = layoutOverride || $('layout').value;
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

  // 普通“上下对比”不留装饰缝，两张图片逐像素紧贴；卡片、三联和左右布局仍保留分隔。
  const gap = layout === 'updown' ? 0 : 8;
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
    ctx.fillStyle = '#10130c'; ctx.fillRect(0, 0, out.width, out.height);
    ctx.fillStyle = '#007ea7'; ctx.fillRect(0, 0, W, Math.max(4, Math.round(W / 100)));
    ctx.fillStyle = '#f2f4ea'; ctx.font = `600 ${Math.round(W / 19)}px sans-serif`; ctx.textBaseline = 'top';
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
      ctx.fillStyle = '#b3bca4'; ctx.font = `${Math.round(W / 34)}px sans-serif`;
      ctx.fillText(place, Math.round(W * .06), y + Math.round(W * .025), Math.round(W * .88));
    }
  } else if (layout === 'leftright') {
    const H = Math.max(...panels.map(p => p.h));
    out.width = W * panels.length + gap * (panels.length - 1);
    out.height = H;
    ctx.fillStyle = '#10130c'; ctx.fillRect(0, 0, out.width, out.height);
    panels.forEach((p, i) => ctx.drawImage(p.cv, i * (W + gap), 0, W, p.h));
  } else { // updown / triple 纵向
    out.width = W;
    out.height = panels.reduce((s, p) => s + p.h, 0) + gap * (panels.length - 1);
    ctx.fillStyle = '#10130c'; ctx.fillRect(0, 0, out.width, out.height);
    let y = 0;
    panels.forEach((p) => { ctx.drawImage(p.cv, 0, y, W, p.h); y += p.h + gap; });
  }
  return out;
}

function exportCompareLayout(layoutOverride = '') {
  const out = makeCompareCanvas(0, layoutOverride);
  const suffix = layoutOverride ? `-${layoutOverride}` : '';
  download(out.toDataURL('image/png'), `seichi-compare${suffix}.png`);
}

$('btnExportCompare').addEventListener('click', exportCompareLayout);
$('btnExportCompareLayout').addEventListener('click', exportCompareLayout);

// 与页面“叠加”模式一致：动画参考图按 cover 裁齐到实景画幅，透明度使用当前滑杆值。
function makeOverlayCompareCanvas(maxWidth = 0) {
  const src = $('canvasGraded');
  const scale = maxWidth ? Math.min(1, maxWidth / src.width) : 1;
  const out = document.createElement('canvas');
  out.width = Math.max(2, Math.round(src.width * scale));
  out.height = Math.max(2, Math.round(src.height * scale));
  const ctx = out.getContext('2d');
  ctx.drawImage(src, 0, 0, out.width, out.height);
  ctx.globalAlpha = Number($('overlayOpacity').value) / 100;
  ctx.drawImage($('canvasAnimeOverlay'), 0, 0, out.width, out.height);
  ctx.globalAlpha = 1;
  return out;
}

function exportOverlayCompare() {
  const out = makeOverlayCompareCanvas();
  download(out.toDataURL('image/png'), 'seichi-overlay-compare.png');
  setStatus(`已导出叠加对照图 · 动画透明度 ${$('overlayOpacity').value}%`);
}

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
  drawExportHubPreview('hubPreviewOverlay', makeOverlayCompareCanvas(340));
  drawExportHubPreview('hubPreviewUpdown', makeCompareCanvas(340, 'updown'));
  drawExportHubPreview('hubPreviewLeftright', makeCompareCanvas(340, 'leftright'));
  drawExportHubPreview('hubPreviewTriple', makeCompareCanvas(340, 'triple'));
  drawExportHubPreview('hubPreviewPostcard', makeCompareCanvas(340, 'postcard'));
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
    const action = card.dataset.exportAction;
    const compareLayouts = {
      'compare-updown': 'updown', 'compare-leftright': 'leftright',
      'compare-triple': 'triple', 'compare-postcard': 'postcard',
    };
    if (action === 'overlay') {
      closeExportHub();
      setTimeout(exportOverlayCompare, 0);
      return;
    }
    if (compareLayouts[action]) {
      closeExportHub();
      setTimeout(() => exportCompareLayout(compareLayouts[action]), 0);
      return;
    }
    const buttons = { image: 'btnExportImg', compare: 'btnExportCompare', wipe: 'btnExportWipe', morph: 'btnExportMorph', apng: 'btnExportApng', character: 'btnExportCharacter', batch: 'btnBatchExport', lut: 'btnExportLut' };
    const target = $(buttons[action]);
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
    // 数字键（prefix≤4095, value≤255）：比字符串拼接键少一次分配和哈希，1080p 帧提速明显
    const value = indices[i], key = prefix * 256 + value;
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
    // 4×4 Bayer 有序抖动：256 色的天空/阴影渐变不再一圈圈色带，视觉上接近连续色
    const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
    for (let p = 0, i = 0; p < indexed.length; p++, i += 4) {
      const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
      const hi = Math.max(r, g, b), lo = Math.min(r, g, b);
      const dith = BAYER[((p / w) & 3) * 4 + (p % w & 3)] / 16 - 0.5;
      if (hi - lo < 18) {
        const gray = Math.max(0, Math.min(39, Math.round(((r + g + b) / 3) * 39 / 255 + dith)));
        indexed[p] = 216 + gray;
      } else {
        const qr = Math.max(0, Math.min(5, Math.round(r / 51 + dith)));
        const qg = Math.max(0, Math.min(5, Math.round(g / 51 + dith)));
        const qb = Math.max(0, Math.min(5, Math.round(b / 51 + dith)));
        indexed[p] = qr * 36 + qg * 6 + qb;
      }
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

// ---------- APNG 高清动图：逐帧 PNG 原样拼装（acTL/fcTL/fdAT），全彩无损 ----------
// GIF 天花板是 256 色；APNG 由浏览器原生播放、行为与 GIF 相同（自动播放、播完停在实景），
// 编码零依赖：每帧用 canvas 自带的 PNG 编码，这里只负责按 APNG 规范拼 chunk。
const PNG_CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function pngCrc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = PNG_CRC_TABLE[(c ^ bytes[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, pngCrc32(out.subarray(4, 8 + data.length)));
  return out;
}

function pngChunksOf(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunks = [];
  let pos = 8; // 跳过 PNG 签名
  while (pos + 12 <= bytes.length) {
    const len = ((bytes[pos] << 24) | (bytes[pos + 1] << 16) | (bytes[pos + 2] << 8) | bytes[pos + 3]) >>> 0;
    const type = String.fromCharCode(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
    chunks.push({ type, data: bytes.subarray(pos + 8, pos + 8 + len) });
    pos += 12 + len;
  }
  return chunks;
}

async function makeAnimeToSceneApng() {
  if (!state.anime || !state.gradedData) throw new Error('请先上传动画截图与实景照片');
  const source = $('canvasGraded'), scale = Math.min(1, 1080 / Math.max(source.width, source.height));
  const w = Math.max(2, Math.round(source.width * scale)), h = Math.max(2, Math.round(source.height * scale));
  const anime = document.createElement('canvas'); anime.width = w; anime.height = h;
  anime.getContext('2d').drawImage($('canvasAnimeOverlay'), 0, 0, w, h);
  const scene = document.createElement('canvas'); scene.width = w; scene.height = h;
  scene.getContext('2d').drawImage(source, 0, 0, w, h);
  const frame = document.createElement('canvas'); frame.width = w; frame.height = h;
  const ctx = frame.getContext('2d');
  const frames = 12, delayMs = Math.round(2000 / frames);
  const perFrame = [];
  for (let index = 0; index < frames; index++) {
    const realOpacity = index / (frames - 1);
    ctx.globalAlpha = 1; ctx.drawImage(anime, 0, 0);
    ctx.globalAlpha = realOpacity; ctx.drawImage(scene, 0, 0); ctx.globalAlpha = 1;
    const blob = await new Promise((r) => frame.toBlob(r, 'image/png'));
    if (!blob) throw new Error('PNG 编码失败');
    perFrame.push(pngChunksOf(await blob.arrayBuffer()));
    setStatus(`正在生成高清 APNG · ${index + 1}/${frames}`);
    await nextPaint();
  }
  const ihdr = perFrame[0].find((c) => c.type === 'IHDR');
  const pieces = [Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10), pngChunk('IHDR', ihdr.data)];
  const acTL = new Uint8Array(8);
  new DataView(acTL.buffer).setUint32(0, frames);
  new DataView(acTL.buffer).setUint32(4, 1); // 播放 1 次，结束停在实景
  pieces.push(pngChunk('acTL', acTL));
  let seq = 0;
  perFrame.forEach((chunks, fi) => {
    const fcTL = new Uint8Array(26);
    const dv = new DataView(fcTL.buffer);
    dv.setUint32(0, seq++); dv.setUint32(4, w); dv.setUint32(8, h);
    dv.setUint16(20, delayMs); dv.setUint16(22, 1000); // delay = delayMs/1000 秒
    pieces.push(pngChunk('fcTL', fcTL));
    for (const c of chunks) {
      if (c.type !== 'IDAT') continue;
      if (fi === 0) { pieces.push(pngChunk('IDAT', c.data)); continue; }
      const fdat = new Uint8Array(4 + c.data.length);
      new DataView(fdat.buffer).setUint32(0, seq++);
      fdat.set(c.data, 4);
      pieces.push(pngChunk('fdAT', fdat));
    }
  });
  pieces.push(pngChunk('IEND', new Uint8Array(0)));
  return { blob: new Blob(pieces, { type: 'image/png' }), width: w, height: h };
}

$('btnExportApng').addEventListener('click', async () => {
  const btn = $('btnExportApng'); btn.disabled = true;
  try {
    const apng = await makeAnimeToSceneApng();
    const url = URL.createObjectURL(apng.blob);
    download(url, 'seichi-anime-to-scene.apng.png');
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    setStatus(`已导出高清 APNG · ${apng.width}×${apng.height} · ${(apng.blob.size / 1048576).toFixed(1)}MB · 全彩，分享兼容性略低于 GIF`);
  } catch (e) {
    console.error(e);
    setStatus('APNG 导出失败：' + (e.message || e));
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

// ---------- 双向对应图匹配（动画→实景 / 实景→动画，场景嵌入见 embed.js） ----------
// 逐张「解码→编码→释放」，保留所有轻量缩略图与 File 引用供用户横向比较；
// 原图像素在每轮比对后立即释放，仍受 60 张上限保护手机内存。
const MATCH_MAX_FILES = 60;
let matchReverse = false;

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

async function useMatchedAnime(entry, item) {
  try {
    setStatus('读取选中的动画截图…');
    const data = await fileToImageData(entry.file);
    $('thumbAnime').src = data.url; $('thumbAnime').hidden = false;
    await handleAnimeData(data);
    [...$('matchGrid').children].forEach((el) => el.classList.toggle('selected', el === item));
  } catch (e) { setStatus('读取失败：' + (e.message || e)); }
}

function renderMatchResults(ranked, reverse = false) {
  const grid = $('matchGrid');
  grid.textContent = '';
  $('matchResultsLabel').textContent = reverse
    ? '全部动画截图已按相似度排序；左右滑动并点选作为动画截图'
    : '全部实景照片已按相似度排序；左右滑动并点选作为实景照片';
  ranked.forEach((entry, index) => {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'match-item';
    item.title = entry.file.name || '';
    item.appendChild(entry.thumb);
    const label = document.createElement('span');
    label.textContent = `#${index + 1} · ${Math.max(0, Math.round(entry.sim * 100))}%`;
    item.appendChild(label);
    item.addEventListener('click', () => reverse ? useMatchedAnime(entry, item) : useMatchedPhoto(entry, item));
    grid.appendChild(item);
  });
  $('matchResults').hidden = !ranked.length;
}

$('btnMatchScene').addEventListener('click', () => {
  if (!state.anime && !state.photo) { setStatus('请先放入一张动画截图或实景照片作为匹配参考'); return; }
  if (state.aiBusy) { setStatus('AI 任务进行中，请稍候…'); return; }
  // 常规：动画 → 多张实景；只放了实景时自动反向：实景 → 多张动画。
  matchReverse = !state.anime && !!state.photo;
  $('matchFiles').value = '';
  $('matchFiles').click();
});

$('matchFiles').addEventListener('change', async (event) => {
  const files = [...event.target.files].slice(0, MATCH_MAX_FILES);
  const queryImage = matchReverse ? state.photo?.imgData : state.anime?.imgData;
  if (!files.length || !queryImage) return;
  // 找图的嵌入模型跑在主线程 ORT；与抠像 Worker 的 ORT 堆互斥，防止双堆并存挤爆内存
  if (state.aiBusy) { setStatus('AI 任务进行中，请稍候…'); return; }
  setAIBusy(true);
  $('matchResults').hidden = true;
  const report = (text) => { setStatus(text); setButtonLoad('loadMatchScene', text); };
  try {
    report('准备模型…');
    const fmtMB = (n) => (n / 1048576).toFixed(1);
    const query = await embedImage(queryImage, {
      onProgress: (r, t) => report(`下载 ${fmtMB(r)}/${fmtMB(t)}MB`),
    });
    const ranked = [];
    let compared = 0, failed = 0, failReason = '';
    for (let i = 0; i < files.length; i++) {
      try {
        report(`比对 ${i + 1}/${files.length}`);
        const data = await fileToImageData(files[i]);
        const sim = cosineSimilarity(query, await embedImage(data.imgData));
        const thumb = makeMatchThumb(data.imgData);
        URL.revokeObjectURL(data.url);
        ranked.push({ file: files[i], sim, thumb });
        compared++;
      } catch (e) {
        console.warn('找图比对失败', files[i]?.name, e);
        failed++;
        if (!failReason) failReason = e.message || String(e); // 把第一条失败原因带给用户（如 HEIC/RAW 指引）
      }
    }
    ranked.sort((a, b) => b.sim - a.sim);
    renderMatchResults(ranked, matchReverse);
    setStatus(compared
      ? `${matchReverse ? '反向匹配动画' : '匹配实景'}完成：共比对 ${compared} 张${failed ? `，${failed} 张读取失败（${failReason}）` : ''}${event.target.files.length > MATCH_MAX_FILES ? `（超过 ${MATCH_MAX_FILES} 张的部分未参加）` : ''}，点选最像的一张`
      : `匹配失败：所选图片都无法读取（${failReason}）`);
  } catch (e) {
    console.error(e); rememberError('scene-match', e);
    setStatus('找图匹配失败：' + (e.message || e));
  } finally { setAIBusy(false); setButtonLoad('loadMatchScene'); }
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
const MODEL_CACHE = 'seichi-models-v8'; // v8: 模型改同源加载（整站迁 GitHub Pages / compose.anitabi.cn），旧 github.io 缓存键作废
// 运行时实际读取 ISNet 的两个分块（见 ort-env.js），不是同名的整文件。
// 因此离线包也只缓存分块，避免把同一模型下载两遍。
const ISNET_URL = DEVICE.isAppleMobile ? `${MODEL_BASE}/models/isnet-anime-512-w8.onnx` : `${MODEL_BASE}/models/isnet-anime-w8.onnx`;
const ISNET_PARTS = Array.from({ length: 2 }, (_, i) => `${ISNET_URL}.part${String(i).padStart(2, '0')}`);
const AUTO_MODEL_URLS = [`${MODEL_BASE}/models/person-detect.onnx`, ...ISNET_PARTS];
const SAM_MODEL_URLS = [`${MODEL_BASE}/models/sam-encoder.onnx`, `${MODEL_BASE}/models/sam-decoder.onnx`];
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
      : autoCount ? `继续下载自动抠图离线包（已完成 ${autoCount}/${AUTO_CACHE_URLS.length}）` : '下载自动抠图离线包（约 85MB）';
    $('btnCacheSam').textContent = samReady
      ? 'SAM 高质量兜底包已就绪 ✓'
      : samCount ? `继续下载 SAM 兜底包（已完成 ${samCount}/${SAM_MODEL_URLS.length}）` : '下载 SAM 高质量兜底包（约 38MB）';
    $('btnCacheMatch').textContent = matchReady
      ? '找图匹配离线包已就绪 ✓'
      : matchCount ? `继续下载找图匹配包（已完成 ${matchCount}/${MATCH_CACHE_URLS.length}）` : '下载找图匹配离线包（约 21MB）';
  } catch (e) { label.textContent = '无法读取模型缓存：' + (e.message || e); }
}

const fmtMB = (bytes) => (bytes / 1048576).toFixed(1);
const fmtSpeed = (bps) => (bps >= 1048576 ? `${(bps / 1048576).toFixed(1)} MB/s` : `${Math.max(1, Math.round(bps / 1024))} KB/s`);

async function downloadOfflinePackage(kind, urls, button) {
  const label = $('modelCacheStatus'); button.disabled = true;
  try {
    await navigator.serviceWorker.ready;
    if (navigator.storage?.persist) await navigator.storage.persist().catch(() => false);
    const cache = await caches.open(MODEL_CACHE);
    let done = await countCached(cache, urls);
    for (const url of urls) {
      const key = cacheKey(url);
      if (await cacheHas(cache, url)) continue;
      const shortName = key.slice(key.lastIndexOf('/') + 1).split('?')[0];
      // 手动流式下载以显示实时进度与速度（cache.add 无字节反馈，慢和卡死看起来一样）。
      // 读完整流后才 cache.put，故中断时不写入半个文件——续传语义与原 cache.add 一致。
      const resp = await fetch(key);
      if (!resp.ok) throw new Error(`${shortName} ${resp.status}`);
      const forCache = resp.clone();
      const total = +resp.headers.get('content-length') || 0;
      const start = performance.now();
      let received = 0, lastPaint = 0;
      if (resp.body) {
        const reader = resp.body.getReader();
        for (;;) {
          const { done: rd, value } = await reader.read();
          if (rd) break;
          received += value.length;
          const now = performance.now();
          if (now - lastPaint > 250) {
            lastPaint = now;
            const speed = received / ((now - start) / 1000 || 1);
            const size = total ? `${fmtMB(received)}/${fmtMB(total)}MB` : `${fmtMB(received)}MB`;
            label.textContent = `下载${kind} ${done + 1}/${urls.length} · ${shortName} ${size} · ${fmtSpeed(speed)}`;
          }
        }
      } else {
        await resp.arrayBuffer();
      }
      await cache.put(key, forCache);
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

// ---------- PWA 安装引导 ----------
// 安装成 PWA 后浏览器不再随意清站点数据（iOS 尤其：普通网页 7 天不访问即清空，
// 已安装的主屏幕应用豁免），是保住上百 MB 离线模型最有效的手段。
(() => {
  const NOTICE_KEY = 'seichi-entry-notice-v1';
  const tip = $('pwaInstallTip');
  const offlineBtn = $('btnPwaInstall');   // 离线包区域的安装按钮
  const notice = $('entryNotice');
  const noticeInstall = $('btnNoticeInstall');
  const isInstalled = () => matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;

  // deferredPrompt 在两处（离线包按钮 + 进站公告）共享：谁有触发条件谁就能装
  let deferredPrompt = null;
  const revealInstall = (visible) => {
    // 仅 Android/桌面 Chromium 会拿到 beforeinstallprompt；iOS 走文字引导
    if (!isInstalled()) { offlineBtn.hidden = !visible; if (visible) noticeInstall.hidden = false; }
  };
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault(); // 拦下浏览器默认横幅，改为按需展示
    deferredPrompt = e;
    revealInstall(true);
  });
  const doInstall = async () => {
    if (!deferredPrompt) return false;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    deferredPrompt = null;
    if (outcome === 'accepted') { offlineBtn.hidden = true; noticeInstall.hidden = true; tip.hidden = true; }
    return outcome === 'accepted';
  };
  offlineBtn.addEventListener('click', doInstall);
  window.addEventListener('appinstalled', () => {
    offlineBtn.hidden = true; noticeInstall.hidden = true; tip.hidden = true;
    closeNotice();
  });

  // 离线包区域的 iOS 文字引导（常驻提示）
  if (DEVICE.isAppleMobile && !isInstalled()) {
    tip.textContent = '建议安装：Safari 底部「分享」→「添加到主屏幕」。否则 iOS 会在 7 天不访问后清空已下载的离线模型。';
    tip.hidden = false;
  }

  // ---- 进站公告：每台设备只弹一次（记 localStorage），已安装则不弹 ----
  function closeNotice() { notice.hidden = true; }
  let seen = false;
  try { seen = localStorage.getItem(NOTICE_KEY) === '1'; } catch { /* 隐私模式 */ }
  if (!seen && !isInstalled()) {
    if (DEVICE.isAppleMobile) $('entryNoticeIosSteps').hidden = false; // iOS 显示添加到主屏幕步骤
    if (deferredPrompt) noticeInstall.hidden = false;                  // Android 若已就绪则显示安装按钮
    notice.hidden = false;
  }
  noticeInstall.addEventListener('click', async () => {
    const ok = await doInstall();
    if (ok) closeNotice();
  });
  $('btnNoticeClose').addEventListener('click', () => {
    try { localStorage.setItem(NOTICE_KEY, '1'); } catch { /* 忽略 */ }
    closeNotice();
  });
})();

// 只发 HEAD 请求，不下载数十 MB 的模型本体。用于定位“模型下载失败”是卡在
// GitHub Pages、分块模型、还是 jsDelivr 的 ONNX 运行环境。
const MODEL_DOWNLOAD_PROBES = [
  ['本站 · 人物检测模型', `${MODEL_BASE}/models/person-detect.onnx`],
  ['本站 · 抠图模型分块', `${MODEL_BASE}/models/isnet-anime-w8.onnx.part00`],
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
  navigator.serviceWorker.register('./sw.js')
    .then(() => navigator.serviceWorker.ready)
    .then(() => {
      // GitHub Pages 无法设 COOP/COEP，改由 SW 合成（见 sw.js withCOI）。首次访问时当前
      // 文档在 SW 接管前已加载、未带隔离头，需重载一次让文档重新经 SW 取回，才能
      // crossOriginIsolated → ONNX 多线程。sessionStorage 保证最多重载一次，隔离失败也不死循环。
      if (self.crossOriginIsolated || sessionStorage.getItem('coiReloaded')) {
        updateModelCacheStatus();
        return;
      }
      const reloadOnce = () => { sessionStorage.setItem('coiReloaded', '1'); location.reload(); };
      // controller 已就绪则立即重载；否则等 SW 接管（controllerchange）后再重载
      if (navigator.serviceWorker.controller) reloadOnce();
      else navigator.serviceWorker.addEventListener('controllerchange', reloadOnce, { once: true });
    })
    .catch(updateModelCacheStatus);
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
  keepMode: false, keepDrawing: false, keepStrokes: [], eraseDrawing: false, eraseStroke: null, stage: 'select', maskOverlay: null,
  pickMode: false,   // 第二步「点选补块」：点一下暗蒙版，把整块连通色域并入目标
  refineBox: null,   // 第一步圈选的 bbox，点选补块只在框内生效
  fillMode: false, fillDrawing: false, fillPts: [], // 「磁性描边」：随手描一段，吸附到线稿
  snapContour: null,                                // 已吸附待确认的闭合轮廓（确认后才锁定内部）
  edgeCost: null, edgeG: null, edgeCostFor: null,   // 边缘代价图缓存（按 state.anime 复用）
};

function lassoReady() {
  if (lassoState.pts.length < 4) return false;
  const box = lassoBBox();
  return box.w >= 12 && box.h >= 12;
}

function updateLassoTip() {
  if (lassoState.mode === 'erase') {
    $('lassoTip').textContent = lassoState.fillMode
      ? '描边擦除：把它当画笔用——像涂鸦一样，手指沿要删掉部件的轮廓涂一圈（不必准、不必封口）；松手自动吸附到线稿，确认后圈内目标整片擦掉'
      : lassoState.pickMode
        ? '点擦整块：点一下亮着的目标（脸、手、衣角），整块颜色相近的区域会一起被擦掉'
        : '亮着的是已识别目标；调好橡皮大小后，按住拖过亮起部分即可擦除；或用「点擦整块 / 描边擦除」';
    return;
  }
  if (lassoState.stage === 'refine') {
    $('lassoTip').textContent = lassoState.keepMode
      ? '补画：刷过的部分会立刻取消暗蒙版，并入识别目标'
      : lassoState.pickMode
        ? '点选补块：点一下暗块里的脸颊/手臂等大片色块，整块（含包住的眼睛嘴巴）会一起补入目标'
        : lassoState.fillMode
          ? '磁性描边：把它当画笔用——像涂鸦一样，手指沿要补回部件的轮廓涂一圈（不必准、不必封口）；松手自动吸附到线稿，确认后圈内不论颜色一律并入目标'
          : '识别结果以外的区域已蒙版；漏掉的脸、手可用「点选补块 / 磁性描边」补回，细碎处用画笔刷开蒙版';
    return;
  }
  if (lassoState.keepMode) {
    $('lassoTip').textContent = '补画：涂过的部分会取消暗蒙版，并强制保留，不受算法抠像结果影响';
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
  guide.hidden = lassoState.mode === 'erase';
  guide.children[0].classList.toggle('active', lassoState.stage === 'select');
  guide.children[1].classList.toggle('active', lassoState.stage === 'refine');
}

function rebuildLassoMaskOverlay() {
  // 以最终 alpha（算法+手工修补+阈值收边之后）为准；补画/点选的效果立即反映在蒙版上
  const src = state.finalAlpha?.length === state.rawW * state.rawH ? state.finalAlpha : state.rawAlpha;
  if (!src || !state.rawW || !state.rawH) { lassoState.maskOverlay = null; return; }
  const overlay = document.createElement('canvas'); overlay.width = state.rawW; overlay.height = state.rawH;
  const image = new ImageData(overlay.width, overlay.height);
  for (let i = 0, p = 0; i < src.length; i++, p += 4) {
    // 识别目标保持完整；其它部分以深色半透明蒙版呈现，便于明确看出遗漏。
    image.data[p + 3] = Math.round((255 - src[i]) * .72);
  }
  overlay.getContext('2d').putImageData(image, 0, 0);
  lassoState.maskOverlay = overlay;
}

function lassoRedraw() {
  const c = $('lassoCanvas'), ctx = c.getContext('2d');
  ctx.putImageData(state.anime.imgData, 0, 0);
  // 第二步补画和橡皮擦都使用同一张遮罩：已选目标亮起，其余区域变暗。
  if ((lassoState.stage === 'refine' || lassoState.mode === 'erase') && lassoState.maskOverlay) ctx.drawImage(lassoState.maskOverlay, 0, 0);
  if (lassoState.stage === 'select' && lassoState.pts.length > 1) {
    ctx.save();
    ctx.lineWidth = Math.max(2, c.width / 350);
    ctx.strokeStyle = 'rgba(0, 126, 167,.95)';
    ctx.fillStyle = 'rgba(0, 126, 167,.14)';
    ctx.beginPath();
    ctx.moveTo(lassoState.pts[0][0], lassoState.pts[0][1]);
    for (let i = 1; i < lassoState.pts.length; i++) ctx.lineTo(lassoState.pts[i][0], lassoState.pts[i][1]);
    ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();
  }
  if (lassoState.keepStrokes.length) {
    ctx.save();
    // 仅在正在落笔时给一条浅色笔迹作即时反馈；提交后不再上色，直接露出原图。
    ctx.strokeStyle = 'rgba(255,255,255,.92)';
    ctx.fillStyle = 'rgba(255,255,255,.10)';
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
  if (lassoState.fillPts.length > 1) {
    ctx.save();
    // 磁性描边的实时笔迹：显示手指原始描线，松手后才吸附到线稿并闭合。
    const erasing = lassoState.mode === 'erase';
    ctx.lineWidth = Math.max(2, c.width / 400);
    ctx.strokeStyle = erasing ? 'rgba(0,0,0,.8)' : 'rgba(0, 167, 225,.95)';
    ctx.fillStyle = erasing ? 'rgba(0,0,0,.22)' : 'rgba(0, 167, 225,.20)';
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    const fp = lassoState.fillPts;
    ctx.beginPath(); ctx.moveTo(fp[0][0], fp[0][1]);
    for (let i = 1; i < fp.length; i++) ctx.lineTo(fp[i][0], fp[i][1]);
    ctx.closePath(); ctx.fill();
    ctx.stroke();
    // 起点→末点的闭合段用虚线，提示“不必手动封口”。
    ctx.setLineDash([6, 5]);
    ctx.beginPath(); ctx.moveTo(fp[fp.length - 1][0], fp[fp.length - 1][1]); ctx.lineTo(fp[0][0], fp[0][1]); ctx.stroke();
    ctx.restore();
  }
  if (lassoState.snapContour && lassoState.snapContour.length > 2) {
    ctx.save();
    // 已吸附到线稿的闭合轮廓：青色实线 + 半透明填充，等用户确认锁定。
    const sc = lassoState.snapContour;
    ctx.lineWidth = Math.max(2.5, c.width / 340);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgba(0, 200, 190, .98)';
    ctx.fillStyle = lassoState.mode === 'erase' ? 'rgba(0,0,0,.24)' : 'rgba(0, 200, 190, .24)';
    ctx.beginPath(); ctx.moveTo(sc[0][0], sc[0][1]);
    for (let i = 1; i < sc.length; i++) ctx.lineTo(sc[i][0], sc[i][1]);
    ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();
  }
  if (lassoState.eraseStroke) {
    const stroke = lassoState.eraseStroke;
    ctx.save();
    // 拖动时预览将要加回的暗蒙版；松手后才作为一整笔可撤销操作提交。
    ctx.strokeStyle = 'rgba(0,0,0,.68)'; ctx.fillStyle = 'rgba(0,0,0,.68)';
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = stroke.width;
    ctx.beginPath(); ctx.moveTo(stroke.pts[0][0], stroke.pts[0][1]);
    for (let i = 1; i < stroke.pts.length; i++) ctx.lineTo(stroke.pts[i][0], stroke.pts[i][1]);
    ctx.stroke();
    if (stroke.pts.length === 1) { ctx.beginPath(); ctx.arc(stroke.pts[0][0], stroke.pts[0][1], stroke.width / 2, 0, Math.PI * 2); ctx.fill(); }
    ctx.restore();
  }
}

function updateKeepBrushUI() {
  const available = lassoState.mode !== 'erase' && lassoState.stage === 'refine';
  const erasing = lassoState.mode === 'erase';
  const canUndoHere = available || erasing;
  $('btnLassoKeep').hidden = !available;
  // 点选、磁性描边两个按钮在「补块 refine」和「擦除」两种场景都出现，只是语义相反。
  $('btnLassoPick').hidden = !available && !erasing;
  $('btnLassoPick').textContent = erasing ? '👆 点擦整块' : '👆 点选补块';
  $('btnLassoFill').hidden = !available && !erasing;
  $('btnLassoFill').textContent = erasing ? '🧲 描边擦除' : '🧲 磁性描边';
  // 描边吸附后出现「锁定此轮廓」确认；未吸附或非描边模式则隐藏。
  $('btnLassoFillConfirm').hidden = !(lassoState.fillMode && lassoState.snapContour);
  $('btnLassoFillConfirm').textContent = erasing ? '✓ 锁定擦除' : '✓ 锁定此轮廓';
  // 橡皮擦同样是可逆编辑：撤销必须留在当前弹窗内，而不是让用户回控制面板找。
  $('btnLassoUndo').hidden = !canUndoHere;
  $('btnLassoUndo').disabled = !state.maskOps.length;
  // 擦除时：笔刷模式才需要橡皮大小，点擦/圈内模式用不到就收起。
  $('lassoKeepSize').hidden = erasing ? (lassoState.pickMode || lassoState.fillMode) : !available || !lassoState.keepMode;
  $('lassoBrushSizeLabel').textContent = erasing ? '橡皮大小' : '画笔粗细';
  $('btnLassoKeep').classList.toggle('keep-on', lassoState.keepMode);
  $('btnLassoKeep').setAttribute('aria-pressed', String(lassoState.keepMode));
  $('btnLassoPick').classList.toggle('keep-on', lassoState.pickMode);
  $('btnLassoPick').setAttribute('aria-pressed', String(lassoState.pickMode));
  $('btnLassoFill').classList.toggle('keep-on', lassoState.fillMode);
  $('btnLassoFill').setAttribute('aria-pressed', String(lassoState.fillMode));
  $('lassoShapes').hidden = lassoState.stage === 'refine' || erasing;
  $('btnLassoRun').hidden = erasing;
  $('btnLassoClear').hidden = lassoState.stage === 'refine' || erasing;
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
  lassoState.eraseDrawing = false; lassoState.eraseStroke = null;
  lassoState.stage = 'select'; lassoState.maskOverlay = null;
  lassoState.pickMode = false; lassoState.refineBox = null;
  lassoState.fillMode = false; lassoState.fillDrawing = false; lassoState.fillPts = []; lassoState.snapContour = null;
  lassoState.mode = mode;
  // 橡皮擦不是重新“选目标”：打开即展示当前目标，未选部分保持暗蒙版。
  if (mode === 'erase') rebuildLassoMaskOverlay();
  updateLassoTip();
  updateLassoGuide();
  updateKeepBrushUI();
  $('btnLassoRun').textContent = mode === 'algorithm' ? '算法抠取圈选区域' : '模型抠取圈选区域';
  $('btnLassoClear').textContent = '重画';
  $('btnLassoRun').disabled = true;
  lassoRedraw();
  $('lassoModal').hidden = false;
}

// 将一笔橡皮刷出的轨迹裁成“当前已经属于目标”的像素集合。这样橡皮擦只会让
// 亮起的目标部分回到暗蒙版，划过本来就暗的背景不会产生任何效果。
function selectedIndicesInStroke(stroke) {
  const w = state.rawW, h = state.rawH;
  const alpha = state.finalAlpha;
  if (!w || !h || !alpha || alpha.length !== w * h || !stroke?.pts?.length) return [];
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff'; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = stroke.width;
  ctx.beginPath(); ctx.moveTo(stroke.pts[0][0], stroke.pts[0][1]);
  for (let i = 1; i < stroke.pts.length; i++) ctx.lineTo(stroke.pts[i][0], stroke.pts[i][1]);
  ctx.stroke();
  if (stroke.pts.length === 1) { ctx.beginPath(); ctx.arc(stroke.pts[0][0], stroke.pts[0][1], stroke.width / 2, 0, Math.PI * 2); ctx.fill(); }
  const painted = ctx.getImageData(0, 0, w, h).data;
  const idx = [];
  for (let i = 0, p = 3; i < alpha.length; i++, p += 4) {
    if (painted[p] && alpha[i] > 8) idx.push(i);
  }
  return idx;
}

// 取「已吸附闭合轮廓」内部的所有像素。轮廓来自 magneticSnap（已贴到真实线稿并闭合），
// 这里只负责按 nonzero 填充规则光栅化、把内部像素收集成 idx。
//   erase=false：轮廓内一律并入目标（锁定线稿围出的整块部件，不再看颜色）；
//   erase=true ：轮廓内只取当前已是目标(alpha>8)的像素，语义与橡皮笔刷一致，不误擦背景。
function contourInteriorIndices(pts, { erase = false } = {}) {
  const w = state.rawW, h = state.rawH, alpha = state.finalAlpha;
  if (!w || !h || !pts || pts.length < 3) return [];
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff';
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath(); ctx.fill();
  const painted = ctx.getImageData(0, 0, w, h).data;
  const idx = [];
  for (let i = 0, p = 3; i < w * h; i++, p += 4) {
    if (!painted[p]) continue;
    if (erase && (!alpha || alpha[i] <= 8)) continue;
    idx.push(i);
  }
  return idx;
}

// ---------- 磁性描边（逐点就近吸附）----------
// 图片里的线稿本身就是现成轮廓。用户像用笔刷一样沿某条线稿粗略描一段，我们把描线的
// 每个点就近吸到最近的真实线稿上，保留用户的走线拓扑，只是「吸紧」到线上。
// （早期用过全局 livewire/Dijkstra，但在动画线稿上会为贴强边而乱窜成锯齿、还窜到旁边
//  的别的线——实测很糟，已弃用，改这套稳定的就近吸附。）

// 边缘幅值图：Sobel 梯度幅值。edgeG 供本吸附 + growRegionAt 的线稿卵墙共用，edgeMax 存最强值。按 anime 缓存。
function ensureEdgeCost() {
  if (lassoState.edgeCostFor === state.anime && lassoState.edgeCost) return lassoState.edgeCost;
  const w = state.anime.width, h = state.anime.height, d = state.anime.imgData.data;
  const lum = new Float32Array(w * h);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) lum[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
  const g = new Float32Array(w * h);
  let maxg = 1e-6;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const gx = -lum[i - 1 - w] - 2 * lum[i - 1] - lum[i - 1 + w] + lum[i + 1 - w] + 2 * lum[i + 1] + lum[i + 1 + w];
    const gy = -lum[i - w - 1] - 2 * lum[i - w] - lum[i - w + 1] + lum[i + w - 1] + 2 * lum[i + w] + lum[i + w + 1];
    const m = Math.sqrt(gx * gx + gy * gy);
    g[i] = m; if (m > maxg) maxg = m;
  }
  const cost = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) { const e = g[i] / maxg; cost[i] = 0.02 + (1 - e) * (1 - e); }
  lassoState.edgeCost = cost; lassoState.edgeG = g; lassoState.edgeMax = maxg; lassoState.edgeCostFor = state.anime;
  return cost;
}

// 逐点就近吸附：在半径 R 内选【离原点最近】且够浓(g>T)的线稿像素。关键是「就近」而非
// 「最强」——描线附近若有更浓的别的线（头发/冰棍），最强会跳错线，就近则始终咬住用户
// 描的那一条。半径内够不到线稿就保留原点（像笔刷一样直穿空白）。
function snapNearestEdge(x, y, g, w, h, R, T) {
  const cx = Math.round(x), cy = Math.round(y);
  let bx = cx, by = cy, bestD = Infinity;
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    const nx = cx + dx, ny = cy + dy;
    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
    if (g[ny * w + nx] <= T) continue;
    const dd = dx * dx + dy * dy;
    if (dd < bestD) { bestD = dd; bx = nx; by = ny; }
  }
  return [bx, by];
}

// 用户原始笔迹 → 重采样 → 逐点就近吸附到线稿 → 轻微平滑。保留用户走线，只吸紧到线上。
function magneticSnap(rawPts) {
  if (!state.anime || !rawPts || rawPts.length < 3) return null;
  const w = state.anime.width, h = state.anime.height;
  ensureEdgeCost();
  const g = lassoState.edgeG, T = 0.35 * lassoState.edgeMax, R = 10;
  // 1) 重采样每 ~4px，让吸附点分布均匀
  const dense = [rawPts[0]];
  for (let i = 1; i < rawPts.length; i++) {
    const a = rawPts[i - 1], b = rawPts[i], d = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const n = Math.max(1, Math.round(d / 4));
    for (let k = 1; k <= n; k++) dense.push([a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]);
  }
  if (dense.length < 3) return null;
  // 2) 逐点就近吸附
  const snapped = dense.map((p) => snapNearestEdge(p[0], p[1], g, w, h, R, T));
  // 3) 移动平均平滑（窗口 ±2），去掉像素级抖动
  const sm = [];
  for (let i = 0; i < snapped.length; i++) {
    let sx = 0, sy = 0, c = 0;
    for (let k = -2; k <= 2; k++) { const j = i + k; if (j < 0 || j >= snapped.length) continue; sx += snapped[j][0]; sy += snapped[j][1]; c++; }
    sm.push([sx / c, sy / c]);
  }
  return sm.length >= 3 ? sm : null;
}

function closeLasso() { $('lassoModal').hidden = true; }

function extractAlgorithmInRegion(box) {
  const crop = document.createElement('canvas');
  crop.width = box.w; crop.height = box.h;
  // 用负坐标直接把原图指定区域拷进小画布，算法只面对用户圈出的局部。
  crop.getContext('2d').putImageData(state.anime.imgData, -box.x, -box.y);
  const cropData = crop.getContext('2d').getImageData(0, 0, box.w, box.h);
  const res = extractForeground(cropData, { centerBias: 0.5 });
  // 近闭合轮廓补洞：脸/手/服装内部被误判成背景时，只要洞的颜色更像
  // 周边前景而非外部背景，就整块补回；真镂空（透出背景色）保持不填。
  if (res.bbox) fillNearlyClosedHoles(res.alpha, box.w, box.h, cropData);
  return {
    box: { ...box, score: 1 }, rect: { ...box }, score: 1,
    alpha: res.alpha, empty: !res.bbox, via: 'algorithm', manual: true,
  };
}

// 点一下就整块选中。核心是两把「色差尺子」+ 一道「线稿卵墙」，专治动画的赛璐璐上色：
//   ① 相邻像素色差（STEP2）——只要挨着的两个像素颜色接近就继续长。赛璐璐的
//      「亮部→阴影」是一道柔和的小台阶（≈55），跨得过去；于是点亮部脸颊能一路
//      长到阴影里，拿到「整张脸的皮肤」而不是只有受光那半边（老算法只跟种子比，
//      42 的容差跨不过阴影带，这正是「点一下只有超级小一块」的根因）。
//   ② 与种子的总色差（GLOBAL2）——防止顺着线稿的抗锯齿渐变一路漏进黑线/背景。
//      线稿是硬边（跳变≈180），总色差一超上限就被挡住，生长干净地停在轮廓上。
//   ③ 线稿卵墙（edge>wall）——碰到够浓的部件外轮廓就停。颜色挡得住时它不多事；
//      但「颜色接着走可该停」处（同色相邻部件、下颌/发际线，尤其暗肤色顺阴影漏成
//      一大片：实测点裙子暗处 7万→1万）由它兜住。阈值卡在阴影线之上、外轮廓之下。
// 生长完再把区域内被完全包住的「孔洞」（眼睛/嘴/纽扣/发饰）一并吞掉——所以点
// 脸颊能得到整张脸，而不是一张挖了眼睛嘴巴的皮。
//   erase=false：在未识别区(alpha<128)里长，用于「点选补块」把漏掉的整块补回；
//   erase=true ：在已识别目标(alpha>=128)里长，用于「点一下擦整块」。
function growRegionAt(x, y, { erase = false } = {}) {
  const w = state.rawW, h = state.rawH;
  const alpha = state.finalAlpha;
  if (!alpha || alpha.length !== w * h || !state.anime) return null;
  // 补块只在第一步粉框内生效；擦除没有粉框，放开到整幅（靠 alpha 目标域自然收边）。
  const box = erase ? { x: 0, y: 0, w, h } : lassoState.refineBox;
  if (!box) return null;
  const px = Math.round(x), py = Math.round(y);
  if (px < box.x || py < box.y || px >= box.x + box.w || py >= box.y + box.h) return { reason: 'outside' };
  const seed = py * w + px;
  // 生长域：擦除只在目标(alpha≥128)里、补块只在非目标(alpha<128)里；抗锯齿边(8~127)天然成界。
  const inDomain = erase ? (i) => alpha[i] >= 128 : (i) => alpha[i] < 128;
  if (erase && alpha[seed] < 128) return { reason: 'notarget' };
  if (!erase && alpha[seed] >= 128) return { reason: 'already' };
  const d = state.anime.imgData.data;
  const sr = d[seed * 4], sg = d[seed * 4 + 1], sb = d[seed * 4 + 2];
  const STEP2 = 70 * 70;     // 相邻色差上限：跨得过赛璐璐阴影，挡得住线稿硬边
  const GLOBAL2 = 118 * 118; // 与种子总色差上限：防止顺着抗锯齿渐变漏出轮廓
  // 线稿卵墙：碰到「浓到够当部件外轮廓」的边缘就停。颜色能挡住的地方它不多事，
  // 但「颜色接着走、可该停」的地方（同色相邻部件、下颌/发际线，尤其暗肤色顺着阴影
  // 漏成一大片）由它兜住。阈值卡在赛璐璐阴影线之上、部件外轮廓之下（实测 0.20*最强边缘）。
  ensureEdgeCost();
  const edge = lassoState.edgeG, wall = 0.20 * lassoState.edgeMax;
  const maxArea = Math.round(box.w * box.h * (erase ? 0.92 : 0.7));
  const visited = new Uint8Array(w * h);
  const inRegion = new Uint8Array(w * h);
  const stack = [seed]; visited[seed] = 1; inRegion[seed] = 1;
  const idx = [seed];
  let minX = px, maxX = px, minY = py, maxY = py;
  while (stack.length) {
    const i = stack.pop();
    const ix = i % w, iy = (i / w) | 0;
    const cr = d[i * 4], cg = d[i * 4 + 1], cb = d[i * 4 + 2];
    for (const n of [ix > 0 ? i - 1 : -1, ix < w - 1 ? i + 1 : -1, iy > 0 ? i - w : -1, iy < h - 1 ? i + w : -1]) {
      if (n < 0 || visited[n]) continue;
      const nx = n % w, ny = (n / w) | 0;
      if (nx < box.x || ny < box.y || nx >= box.x + box.w || ny >= box.y + box.h) continue;
      visited[n] = 1;
      if (!inDomain(n)) continue; // 越过目标/非目标边界即停
      if (edge[n] > wall) continue; // 撞上浓线稿（部件外轮廓）即停
      const p = n * 4, nr = d[p], ng = d[p + 1], nb = d[p + 2];
      const l0 = nr - cr, l1 = ng - cg, l2 = nb - cb;
      if (l0 * l0 + l1 * l1 + l2 * l2 > STEP2) continue;       // 相邻色差（跨阴影）
      const g0 = nr - sr, g1 = ng - sg, g2 = nb - sb;
      if (g0 * g0 + g1 * g1 + g2 * g2 > GLOBAL2) continue;     // 总色差（挡线稿）
      inRegion[n] = 1; idx.push(n); stack.push(n);
      if (nx < minX) minX = nx; if (nx > maxX) maxX = nx;
      if (ny < minY) minY = ny; if (ny > maxY) maxY = ny;
      if (idx.length > maxArea) return { reason: 'toobig' };
    }
  }
  fillEnclosedHoles(inRegion, idx, w, box, minX, minY, maxX, maxY);
  return { idx: Uint32Array.from(idx) };
}

// 从生长区域的包围盒边界向内灌水：灌得到的是外部空隙，灌不到又不在区域里的像素
// 就是被区域完全包住的孔洞（眼睛/嘴/纽扣），并入区域。这样「点皮肤→整张脸」。
function fillEnclosedHoles(inRegion, idx, w, box, minX, minY, maxX, maxY) {
  const rx0 = Math.max(box.x, minX), ry0 = Math.max(box.y, minY);
  const rx1 = Math.min(box.x + box.w - 1, maxX), ry1 = Math.min(box.y + box.h - 1, maxY);
  const rw = rx1 - rx0 + 1, rh = ry1 - ry0 + 1;
  if (rw <= 2 || rh <= 2) return;
  const outside = new Uint8Array(rw * rh);
  const st = [];
  const seedBorder = (gx, gy) => {
    const li = (gy - ry0) * rw + (gx - rx0);
    if (!outside[li] && !inRegion[gy * w + gx]) { outside[li] = 1; st.push(li); }
  };
  for (let gx = rx0; gx <= rx1; gx++) { seedBorder(gx, ry0); seedBorder(gx, ry1); }
  for (let gy = ry0; gy <= ry1; gy++) { seedBorder(rx0, gy); seedBorder(rx1, gy); }
  while (st.length) {
    const li = st.pop(), lx = li % rw, ly = (li / rw) | 0;
    for (const nl of [lx > 0 ? li - 1 : -1, lx < rw - 1 ? li + 1 : -1, ly > 0 ? li - rw : -1, ly < rh - 1 ? li + rw : -1]) {
      if (nl < 0 || outside[nl]) continue;
      const gx = rx0 + (nl % rw), gy = ry0 + ((nl / rw) | 0);
      if (inRegion[gy * w + gx]) continue;
      outside[nl] = 1; st.push(nl);
    }
  }
  for (let ly = 0; ly < rh; ly++) for (let lx = 0; lx < rw; lx++) {
    if (outside[ly * rw + lx]) continue;
    const gi = (ry0 + ly) * w + (rx0 + lx);
    if (inRegion[gi]) continue;
    inRegion[gi] = 1; idx.push(gi); // 被包住的孔洞
  }
}

// 圈选核心：模型模式走 Worker；算法模式只处理圈选区域，避免把整幅画面当作主体猜。
async function runLassoBox(box, centroid) {
  if (lassoState.busy) return null;
  if (state.aiBusy) { $('extractStatus').textContent = 'AI 任务进行中，请稍候…'; return null; }
  lassoState.busy = true;
  setAIBusy(true);
  $('btnLassoRun').disabled = true;
  const report = (text) => { $('extractStatus').textContent = text; setButtonLoad('loadLasso', text); };
  const onStage = (s) => report(s);
  const onProgress = (recv, total) => {
    report(`下载 ${(recv / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)}MB`);
  };
  report(lassoState.mode === 'algorithm' ? '准备算法抠像…' : '准备模型…');
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
    if (ok === 0) {
      // 没识别出任何内容：留在第一步，让用户按提示直接重圈（不收起粉色工具）
      $('extractStatus').textContent = '圈选区域没有找到明显前景——试着圈大一点、或让圈更贴近角色';
      lassoRedraw();
      return found;
    }
    $('extractStatus').textContent = `${method}已识别 ${ok} 个角色 · 漏掉的部分可点选补块或用画笔刷开蒙版`;
    lassoState.stage = 'refine';
    lassoState.keepMode = false; lassoState.pickMode = false;
    lassoState.refineBox = { ...box };
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
    setAIBusy(false);
    setButtonLoad('loadLasso');
    if (lassoState.stage !== 'refine') $('btnLassoRun').disabled = !lassoReady();
  }
}

$('btnLasso').addEventListener('click', () => {
  if (!state.anime) return;
  prepareIndependentCutout();
  openLasso('extract');
});
$('btnEraseMask').addEventListener('click', () => openLasso('erase'));
$('btnLassoClose').addEventListener('click', closeLasso);
$('btnLassoClear').addEventListener('click', () => { lassoState.pts = []; lassoState.keepStrokes = []; $('btnLassoRun').disabled = true; lassoRedraw(); });
$('btnLassoKeep').addEventListener('click', () => {
  if (lassoState.mode === 'erase' || lassoState.stage !== 'refine') return;
  lassoState.keepMode = !lassoState.keepMode;
  lassoState.pickMode = false; lassoState.fillMode = false; lassoState.fillPts = []; lassoState.snapContour = null;
  updateLassoTip(); updateKeepBrushUI(); lassoRedraw();
});
$('btnLassoPick').addEventListener('click', () => {
  // 擦除模式下也可切到「点擦整块」；补块模式下只在 refine 阶段可用。
  if (lassoState.mode !== 'erase' && lassoState.stage !== 'refine') return;
  lassoState.pickMode = !lassoState.pickMode;
  lassoState.keepMode = false; lassoState.fillMode = false; lassoState.fillPts = []; lassoState.snapContour = null;
  updateLassoTip(); updateKeepBrushUI(); lassoRedraw();
});
$('btnLassoFill').addEventListener('click', () => {
  // 磁性描边：擦除模式随时可用；补块模式只在 refine 阶段可用。
  if (lassoState.mode !== 'erase' && lassoState.stage !== 'refine') return;
  lassoState.fillMode = !lassoState.fillMode;
  lassoState.keepMode = false; lassoState.pickMode = false;
  if (!lassoState.fillMode) { lassoState.fillPts = []; lassoState.snapContour = null; }
  updateLassoTip(); updateKeepBrushUI(); lassoRedraw();
});
$('btnLassoFillConfirm').addEventListener('click', () => {
  const snap = lassoState.snapContour;
  if (!snap) return;
  const erasing = lassoState.mode === 'erase';
  const idx = contourInteriorIndices(snap, { erase: erasing });
  lassoState.snapContour = null;
  if (!idx.length) {
    $('extractStatus').textContent = '这条轮廓围出的范围太小或没套到目标——重新描一遍';
    updateKeepBrushUI(); lassoRedraw(); return;
  }
  if (erasing) {
    const coverage = pushMaskOp({ type: 'erase', idx });
    rebuildLassoMaskOverlay();
    $('extractStatus').textContent = `已锁定擦除轮廓内 ${idx.length.toLocaleString()} 个目标像素 · 当前角色占画面 ${((coverage || 0) * 100).toFixed(0)}% · 可撤销`;
  } else {
    pushMaskOp({ type: 'keepRegion', idx: Uint32Array.from(idx) });
    rebuildLassoMaskOverlay();
    $('extractStatus').textContent = `已锁定轮廓内 ${idx.length.toLocaleString()} 个像素 · 可继续描边或撤销`;
  }
  updateKeepBrushUI(); lassoRedraw();
});
$('btnLassoUndo').addEventListener('click', undoMaskOp);
$('btnMaskUndo').addEventListener('click', undoMaskOp);
// 圈选弹窗内也支持常见的 Ctrl/Cmd + Z；只接管遮罩编辑，避免影响页面其它输入框。
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !$('lassoModal').hidden && state.maskOps.length) {
    e.preventDefault();
    undoMaskOp();
  }
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
    anchor = toImg(e);
    if (lassoState.mode === 'erase') {
      // 点擦：单击一下就整块擦除（与点选补块对称），不进入拖笔状态。
      if (lassoState.pickMode) { runEraseAt(anchor); return; }
      // 描边擦除：按住沿线稿描一段，松手吸附到真实轮廓、确认后擦掉内部。
      if (lassoState.fillMode) {
        lassoState.snapContour = null; updateKeepBrushUI();
        lassoState.drawing = true; lassoState.fillDrawing = true;
        lassoState.fillPts = [anchor]; lassoRedraw(); return;
      }
      // 橡皮笔刷：按住即开始一笔，拖过亮起部分即可擦除。
      lassoState.drawing = true;
      lassoState.eraseDrawing = true;
      lassoState.eraseStroke = { pts: [anchor], width: Number($('lassoKeepBrush').value) };
      lassoRedraw();
      return;
    }
    if (lassoState.stage === 'refine') {
      // 第二步：点选补块单击即生效；补画开始记一笔；两者都没开就忽略，
      // 绝不能改写第一步的选区 pts（否则「完成抠像」会被误禁用）
      if (lassoState.pickMode) { runPickAt(anchor); return; }
      // 磁性描边：按住沿线稿描一段，松手吸附到真实轮廓、确认后并入目标。
      if (lassoState.fillMode) {
        lassoState.snapContour = null; updateKeepBrushUI();
        lassoState.drawing = true; lassoState.fillDrawing = true;
        lassoState.fillPts = [anchor]; lassoRedraw(); return;
      }
      if (!lassoState.keepMode) return;
      lassoState.drawing = true;
      lassoState.keepDrawing = true;
      lassoState.keepStrokes.push({ pts: [anchor], width: Number($('lassoKeepBrush').value) });
    } else {
      lassoState.drawing = true;
      lassoState.keepDrawing = false;
      lassoState.pts = lassoState.shape === 'free' ? [anchor] : [];
    }
    lassoRedraw();
  });

  function runPickAt(pt) {
    const res = growRegionAt(pt[0], pt[1], { erase: false });
    if (!res) return;
    if (res.reason === 'outside') { $('extractStatus').textContent = '请点在第一步圈出的范围内'; return; }
    if (res.reason === 'already') { $('extractStatus').textContent = '这一块已经是保留目标了'; return; }
    if (res.reason === 'toobig') { $('extractStatus').textContent = '这一片长得太大（超过圈选框七成），多半是漏进了背景——换个更靠中心的点，或改用画笔补画'; return; }
    pushMaskOp({ type: 'keepRegion', idx: res.idx });
    rebuildLassoMaskOverlay();
    $('extractStatus').textContent = `已整块补入 ${res.idx.length.toLocaleString()} 个像素 · 可继续点选或撤销`;
    lassoRedraw();
  }

  function runEraseAt(pt) {
    const res = growRegionAt(pt[0], pt[1], { erase: true });
    if (!res) return;
    if (res.reason === 'outside') { $('extractStatus').textContent = '请点在画面内'; return; }
    if (res.reason === 'notarget') { $('extractStatus').textContent = '点擦只作用在亮着的目标上；这一点不是已识别的角色'; return; }
    if (res.reason === 'toobig') { $('extractStatus').textContent = '这一片太大了，换个点，或改用橡皮笔刷慢慢擦'; return; }
    const coverage = pushMaskOp({ type: 'erase', idx: res.idx });
    rebuildLassoMaskOverlay();
    $('extractStatus').textContent = `已整块擦除 ${res.idx.length.toLocaleString()} 个目标像素 · 当前角色占画面 ${((coverage || 0) * 100).toFixed(0)}% · 可撤销`;
    lassoRedraw();
  }
  c.addEventListener('pointermove', (e) => {
    if (!lassoState.drawing) return;
    const p = toImg(e);
    if (lassoState.fillDrawing) {
      const last = lassoState.fillPts[lassoState.fillPts.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 3) { lassoState.fillPts.push(p); lassoRedraw(); }
    } else if (lassoState.eraseDrawing) {
      const stroke = lassoState.eraseStroke;
      const last = stroke.pts[stroke.pts.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 1) { stroke.pts.push(p); lassoRedraw(); }
    } else if (lassoState.keepDrawing) {
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
    const lastEraseStroke = lassoState.eraseDrawing ? lassoState.eraseStroke : null;
    const lastKeepStroke = lassoState.keepDrawing ? lassoState.keepStrokes[lassoState.keepStrokes.length - 1] : null;
    const doingFill = lassoState.fillDrawing;
    lassoState.drawing = false; lassoState.keepDrawing = false; lassoState.eraseDrawing = false; lassoState.fillDrawing = false;
    if (doingFill) {
      const raw = lassoState.fillPts; lassoState.fillPts = [];
      const snap = magneticSnap(raw); // 吸附到真实线稿并闭合
      if (!snap) {
        $('extractStatus').textContent = '这一描太短了——沿着要选的部件线稿多描一段再松手';
        lassoRedraw();
        return;
      }
      // 不立刻提交：先把吸附结果画出来给用户看，点「锁定此轮廓」才生效。
      lassoState.snapContour = snap;
      updateKeepBrushUI();
      $('extractStatus').textContent = lassoState.mode === 'erase'
        ? '已吸附到线稿轮廓（青色）· 点「✓ 锁定擦除」确认，或重新描一遍'
        : '已吸附到线稿轮廓（青色）· 点「✓ 锁定此轮廓」确认，或重新描一遍';
      lassoRedraw();
      return;
    }
    if (lastEraseStroke) {
      lassoState.eraseStroke = null;
      const idx = selectedIndicesInStroke(lastEraseStroke);
      if (!idx.length) {
        $('extractStatus').textContent = '橡皮只会擦亮起的目标部分；这一笔没有碰到目标';
        lassoRedraw();
        return;
      }
      // 一整笔是一个可撤销操作；之后用补画刷回即可覆盖这一笔擦除。
      const coverage = pushMaskOp({ type: 'erase', idx });
      rebuildLassoMaskOverlay();
      $('extractStatus').textContent = `已擦除 ${idx.length.toLocaleString()} 个目标像素 · 当前角色占画面 ${((coverage || 0) * 100).toFixed(0)}% · 可撤销`;
      lassoRedraw();
      return;
    }
    if (lastKeepStroke && lassoState.stage === 'refine') {
      // 落笔即提交成一个可撤销的 keep 操作；提交后最终 alpha 变为不透明，暗蒙版随即消失。
      lassoState.keepStrokes = [];
      pushMaskOp({ type: 'keep', stroke: lastKeepStroke });
      rebuildLassoMaskOverlay();
      $('extractStatus').textContent = '已补上一笔 · 可继续补画、点选或撤销';
      lassoRedraw();
      return; // refine 阶段「完成抠像」保持可用
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
    const canvas = await launchViewfinder(state.anime.imgData, {
      onReferenceChange: async (file) => {
        const data = await fileToImageData(file);
        $('thumbAnime').src = data.url; $('thumbAnime').hidden = false;
        await handleAnimeData(data);
        return state.anime.imgData;
      },
    });
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

window.__qa = { state, renderFullRes, enterAlignMode, applyAlignCrop, alignState, recompute, runLassoBox, launchViewfinder, makeAnimeToSceneGif, makeAnimeToSceneApng, undoMaskOp, growRegionAt, contourInteriorIndices, magneticSnap, ensureEdgeCost, lassoState };

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
