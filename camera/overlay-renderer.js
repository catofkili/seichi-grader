// overlay-renderer.js — 把动画参考图叠在实时取景上，四种观察模式。
// 参考图预处理一次（缩放、预算轮廓），之后每帧只做轻量绘制。
// 模式：transparent(半透明) / outline(轮廓) / blink(闪烁) / split(左右分割)。

class OverlayRenderer {
  // refImage: ImageData | HTMLImageElement | HTMLCanvasElement（动画截图）
  constructor(refImage) {
    this.mode = 'transparent';
    this.opacity = 0.5;
    this.split = 0.5;         // split 模式分割线位置 0..1
    this.blinkOn = true;
    this._blinkTimer = 0;
    this.ref = this._toCanvas(refImage);
    this.outline = null;      // 惰性生成
  }

  _toCanvas(src) {
    const c = document.createElement('canvas');
    if (src instanceof ImageData) {
      c.width = src.width; c.height = src.height;
      c.getContext('2d').putImageData(src, 0, 0);
    } else {
      c.width = src.naturalWidth || src.width;
      c.height = src.naturalHeight || src.height;
      c.getContext('2d').drawImage(src, 0, 0);
    }
    return c;
  }

  get aspect() { return this.ref.width / this.ref.height; }

  setMode(mode) {
    this.mode = mode;
    if (mode === 'outline' && !this.outline) this.outline = this._buildOutline();
    if (mode === 'blink') this._startBlink(); else this._stopBlink();
  }
  setOpacity(v) { this.opacity = Math.max(0, Math.min(1, v)); }
  setSplit(v) { this.split = Math.max(0, Math.min(1, v)); }

  // Sobel 边缘 → 白色描边（透明底），用于轮廓模式。只算一次。
  _buildOutline() {
    const w = this.ref.width, h = this.ref.height;
    const src = this.ref.getContext('2d').getImageData(0, 0, w, h).data;
    const gray = new Float32Array(w * h);
    for (let i = 0, p = 0; i < src.length; i += 4, p++) {
      gray[p] = 0.299 * src[i] + 0.587 * src[i + 1] + 0.114 * src[i + 2];
    }
    const out = document.createElement('canvas'); out.width = w; out.height = h;
    const od = out.getContext('2d').createImageData(w, h);
    // 稳健归一化：先求 98 分位做上限，避免个别强边吃掉对比
    let maxMag = 1;
    const mags = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const p = y * w + x;
        const gx = -gray[p - w - 1] - 2 * gray[p - 1] - gray[p + w - 1]
                 + gray[p - w + 1] + 2 * gray[p + 1] + gray[p + w + 1];
        const gy = -gray[p - w - 1] - 2 * gray[p - w] - gray[p - w + 1]
                 + gray[p + w - 1] + 2 * gray[p + w] + gray[p + w + 1];
        const m = Math.hypot(gx, gy);
        mags[p] = m; if (m > maxMag) maxMag = m;
      }
    }
    const thr = maxMag * 0.18;
    for (let p = 0; p < w * h; p++) {
      const on = mags[p] > thr;
      const a = on ? Math.min(255, mags[p] / maxMag * 320) : 0;
      const i = p * 4;
      od.data[i] = 120; od.data[i + 1] = 235; od.data[i + 2] = 255; od.data[i + 3] = a; // 青色描边
    }
    out.getContext('2d').putImageData(od, 0, 0);
    return out;
  }

  _startBlink() {
    this._stopBlink();
    this._blinkTimer = setInterval(() => { this.blinkOn = !this.blinkOn; }, 500);
  }
  _stopBlink() { if (this._blinkTimer) { clearInterval(this._blinkTimer); this._blinkTimer = 0; } this.blinkOn = true; }

  // 把参考图按 cover 方式铺满目标框，返回绘制矩形（保持参考图比例、居中裁切）。
  _coverRect(dw, dh) {
    const s = Math.max(dw / this.ref.width, dh / this.ref.height);
    const w = this.ref.width * s, h = this.ref.height * s;
    return { x: (dw - w) / 2, y: (dh - h) / 2, w, h };
  }

  // 每帧调用：把叠加层画到 ctx（尺寸 dw×dh，通常是覆盖 video 的 canvas）。
  render(ctx, dw, dh) {
    ctx.clearRect(0, 0, dw, dh);
    const r = this._coverRect(dw, dh);
    ctx.save();
    if (this.mode === 'transparent') {
      ctx.globalAlpha = this.opacity;
      ctx.drawImage(this.ref, r.x, r.y, r.w, r.h);
    } else if (this.mode === 'outline') {
      ctx.globalAlpha = Math.max(0.7, this.opacity);
      ctx.drawImage(this.outline || this.ref, r.x, r.y, r.w, r.h);
    } else if (this.mode === 'blink') {
      if (this.blinkOn) { ctx.globalAlpha = 1; ctx.drawImage(this.ref, r.x, r.y, r.w, r.h); }
    } else if (this.mode === 'split') {
      const clipW = dw * this.split;
      ctx.beginPath(); ctx.rect(0, 0, clipW, dh); ctx.clip();
      ctx.globalAlpha = 1;
      ctx.drawImage(this.ref, r.x, r.y, r.w, r.h);
      ctx.restore(); ctx.save();
      // 分割线
      ctx.globalAlpha = 1; ctx.strokeStyle = 'rgba(120,235,255,.95)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(clipW, 0); ctx.lineTo(clipW, dh); ctx.stroke();
    }
    ctx.restore();
  }

  destroy() { this._stopBlink(); }
}

export { OverlayRenderer };
