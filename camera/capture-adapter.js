// capture-adapter.js — 三级拍照，返回带真实分辨率的 canvas。
// 探针实测(iPhone iOS18.7/Safari26.4)的关键取舍：
//   预览/grabFrame = 2160×4032(与取景框像素级同框)，takePhoto = 3024×4032(4:3，视野更宽，非所见即所得)。
//   巡礼要 WYSIWYG，故默认 wysiwyg=grabFrame；quality=takePhoto 作高画质选项（视野更大，接受与预览不完全一致）。
// 级别：
//   'wysiwyg' → grabFrame（默认，同框）
//   'quality' → takePhoto（更高像素，视野略宽）
//   两者都失败 → 从 video 元素抓帧兜底。
// 系统相机(C级)是独立入口（file input capture=environment），不在此函数内。

async function capturePhoto(session, opts = {}) {
  const level = opts.level || 'wysiwyg';
  const track = session.track;
  if (!track) throw new Error('摄像头未启动');

  // 高画质：优先 takePhoto
  if (level === 'quality' && typeof window.ImageCapture !== 'undefined') {
    try {
      const ic = new ImageCapture(track);
      const blob = await Promise.race([
        ic.takePhoto(),
        new Promise((_, rej) => setTimeout(() => rej(new Error('takePhoto 超时')), 6000)),
      ]);
      const bmp = await createImageBitmap(blob);
      const c = _bitmapToCanvas(bmp);
      bmp.close();
      return { canvas: c, width: c.width, height: c.height, via: 'takePhoto' };
    } catch (e) {
      // 失败则落到同框路径
      console.warn('takePhoto 失败，回退 grabFrame', e);
    }
  }

  // 同框：grabFrame
  if (typeof window.ImageCapture !== 'undefined') {
    try {
      const ic = new ImageCapture(track);
      const bmp = await ic.grabFrame();
      const c = _bitmapToCanvas(bmp);
      bmp.close();
      return { canvas: c, width: c.width, height: c.height, via: 'grabFrame' };
    } catch (e) {
      console.warn('grabFrame 失败，回退 video 抓帧', e);
    }
  }

  // 兜底：从 <video> 直接抓当前帧
  const v = session.video;
  if (!v || !v.videoWidth) throw new Error('无法从摄像头抓取画面');
  const c = document.createElement('canvas');
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  return { canvas: c, width: c.width, height: c.height, via: 'videoFrame' };
}

function _bitmapToCanvas(bmp) {
  const c = document.createElement('canvas');
  c.width = bmp.width; c.height = bmp.height;
  c.getContext('2d').drawImage(bmp, 0, 0);
  return c;
}

// 把拍摄 canvas 裁到目标宽高比（= 动画参考图比例），居中裁切。
// 保证"取景时对齐的构图"在成图里保持同样的画框。
function cropToAspect(canvas, aspect) {
  const cw = canvas.width, ch = canvas.height;
  const curAspect = cw / ch;
  let sx = 0, sy = 0, sw = cw, sh = ch;
  if (curAspect > aspect) { sw = Math.round(ch * aspect); sx = Math.round((cw - sw) / 2); }
  else { sh = Math.round(cw / aspect); sy = Math.round((ch - sh) / 2); }
  const out = document.createElement('canvas');
  out.width = sw; out.height = sh;
  out.getContext('2d').drawImage(canvas, sx, sy, sw, sh, 0, 0, sw, sh);
  return out;
}

export { capturePhoto, cropToAspect };
