// platform.js — 基于能力而不是 UA 版本做设备分级。
// iPad Pro 会报告 MacIntel，Safari 又不提供 deviceMemory，所以必须结合触摸点判断。
const nav = navigator;
const platform = nav.platform || '';
const ua = nav.userAgent || '';
const touchPoints = nav.maxTouchPoints || 0;
const matches = (query) => typeof globalThis.matchMedia === 'function' && globalThis.matchMedia(query).matches;
const isIPhone = /iPhone|iPod/.test(platform) || /iPhone|iPod/.test(ua);
const isIPad = /iPad/.test(platform) || /iPad/.test(ua) || (platform === 'MacIntel' && touchPoints > 1);
const isAppleMobile = isIPhone || isIPad;
const coarse = matches('(pointer: coarse)') || touchPoints > 1;
const lowMemory = Number.isFinite(nav.deviceMemory) && nav.deviceMemory <= 4;
const narrow = matches('(max-width: 760px)');
const isMobile = isAppleMobile || narrow || lowMemory;

const profile = Object.freeze({
  isIPhone, isIPad, isAppleMobile, isMobile, coarse, lowMemory,
  previewMax: isIPhone ? 896 : isIPad ? 1100 : isMobile ? 960 : 1400,
  exportMaxPixels: isIPhone ? 8_000_000 : isIPad ? 12_000_000 : lowMemory ? 16_000_000 : 64_000_000,
  exportTile: isMobile ? 512 : 1024,
  wasmThreads: isAppleMobile ? 2 : (self.crossOriginIsolated ? Math.min(4, nav.hardwareConcurrency || 1) : 1),
});

export { profile };
