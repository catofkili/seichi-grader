// camera-session.js — 摄像头会话管理：启动、停止、切换镜头、能力检测、前后台自动停复。
// 独立模块，不依赖主应用状态。真机探针结论（2026-07-12 iPhone iOS18.7/Safari26.4）：
//   - 后置三镜头(主摄/双广角/超广角)可用 deviceId 单独选
//   - 实时轨道 2160×4032、takePhoto 12MP、torch/zoom 可用
//   - 必须 HTTPS/secure context，否则 getUserMedia 不存在

// 相机可用性预检。返回 { ok, reason }。
function checkSupport() {
  if (!window.isSecureContext) {
    return { ok: false, reason: '需要 HTTPS 安全环境（当前是普通 http），摄像头不可用' };
  }
  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
    return { ok: false, reason: '此浏览器不支持网页摄像头' };
  }
  return { ok: true };
}

class CameraSession extends EventTarget {
  constructor() {
    super();
    this.stream = null;
    this.track = null;
    this.video = null;
    this.facing = 'environment';
    this.deviceId = null;
    this._suspended = false;      // 被前后台切换临时挂起
    this._onVisibility = this._handleVisibility.bind(this);
  }

  // 启动摄像头。opts: { facing, deviceId, video }。video 为可复用的 <video> 元素。
  async start(opts = {}) {
    const support = checkSupport();
    if (!support.ok) throw new Error(support.reason);
    this.stop(); // 幂等：先清掉旧轨道
    this.facing = opts.facing || this.facing;
    this.deviceId = opts.deviceId || null;

    const videoConstraint = this.deviceId
      ? { deviceId: { exact: this.deviceId }, width: { ideal: 4096 }, height: { ideal: 2160 } }
      : { facingMode: { ideal: this.facing }, width: { ideal: 4096 }, height: { ideal: 2160 } };

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: false });
    } catch (e) {
      // 精确 deviceId 或高分辨率约束失败时，退回最宽松约束再试一次
      if (this.deviceId || e.name === 'OverconstrainedError') {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: this.facing }, audio: false });
        this.deviceId = null;
      } else {
        throw this._friendlyError(e);
      }
    }
    this.stream = stream;
    this.track = stream.getVideoTracks()[0];

    const video = opts.video || this.video || document.createElement('video');
    video.playsInline = true; video.muted = true; video.autoplay = true;
    video.setAttribute('playsinline', ''); // iOS 必须，否则全屏播放劫持
    video.srcObject = stream;
    this.video = video;
    await video.play().catch(() => {}); // 某些浏览器 autoplay 已够，play() 抛出可忽略

    document.addEventListener('visibilitychange', this._onVisibility);
    this._suspended = false;
    this.dispatchEvent(new CustomEvent('started', { detail: this.getInfo() }));
    return this.getInfo();
  }

  stop() {
    document.removeEventListener('visibilitychange', this._onVisibility);
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.video) this.video.srcObject = null;
    this.stream = null; this.track = null;
    this._suspended = false;
  }

  // 切到下一个可用后置镜头（主摄/超广角轮换）。无多镜头时返回 false。
  async cycleLens() {
    const cams = await this.listRearCameras();
    if (cams.length < 2) return false;
    const curId = this.track && this.track.getSettings ? this.track.getSettings().deviceId : null;
    const idx = cams.findIndex((c) => c.deviceId === curId);
    const next = cams[(idx + 1) % cams.length];
    await this.start({ deviceId: next.deviceId, video: this.video });
    return next.label || '已切换镜头';
  }

  async listRearCameras() {
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      const cams = devs.filter((d) => d.kind === 'videoinput');
      const rear = cams.filter((d) => /back|rear|environment|后/i.test(d.label));
      return rear.length ? rear : cams; // 标签拿不到（未授权）时返回全部
    } catch { return []; }
  }

  getCapabilities() {
    return this.track && this.track.getCapabilities ? this.track.getCapabilities() : {};
  }

  getInfo() {
    const s = this.track && this.track.getSettings ? this.track.getSettings() : {};
    const caps = this.getCapabilities();
    return {
      width: s.width, height: s.height, frameRate: s.frameRate,
      facingMode: s.facingMode, label: this.track ? this.track.label : '',
      hasTorch: !!caps.torch,
      zoom: caps.zoom ? { min: caps.zoom.min, max: caps.zoom.max, step: caps.zoom.step || 0.1 } : null,
      hasImageCapture: typeof window.ImageCapture !== 'undefined',
    };
  }

  // 手电筒开关（后置常有）。不支持时返回 false。
  async setTorch(on) {
    if (!this.track || !this.getCapabilities().torch) return false;
    try { await this.track.applyConstraints({ advanced: [{ torch: !!on }] }); return true; }
    catch { return false; }
  }

  // 光学/数码变焦。value 落在 capabilities.zoom 范围内。
  async setZoom(value) {
    const z = this.getCapabilities().zoom;
    if (!z) return false;
    const v = Math.max(z.min, Math.min(z.max, value));
    try { await this.track.applyConstraints({ advanced: [{ zoom: v }] }); return true; }
    catch { return false; }
  }

  // 前后台：切后台立即停轨道（省电/发热/避免 iOS 挂起黑帧），回前台自动重建。
  async _handleVisibility() {
    if (document.hidden) {
      if (this.stream) {
        this._suspended = true;
        this.stream.getTracks().forEach((t) => t.stop());
        this.stream = null; this.track = null;
        this.dispatchEvent(new CustomEvent('suspended'));
      }
    } else if (this._suspended) {
      try {
        await this.start({ facing: this.facing, deviceId: this.deviceId, video: this.video });
        this.dispatchEvent(new CustomEvent('resumed', { detail: this.getInfo() }));
      } catch (e) {
        this.dispatchEvent(new CustomEvent('error', { detail: this._friendlyError(e) }));
      }
    }
  }

  _friendlyError(e) {
    const map = {
      NotAllowedError: '摄像头权限被拒绝——请在系统设置里允许本站访问相机',
      NotFoundError: '没找到摄像头',
      NotReadableError: '摄像头被其他 App 占用，请关闭后重试',
      OverconstrainedError: '摄像头不支持所请求的规格',
      SecurityError: '安全限制：需要 HTTPS 环境',
    };
    const err = new Error(map[e.name] || (e.message || String(e)));
    err.name = e.name || 'CameraError';
    return err;
  }
}

export { CameraSession, checkSupport };
