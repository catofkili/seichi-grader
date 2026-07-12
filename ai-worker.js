// One-shot AI worker. Terminating it after each job returns the whole WASM heap to WebKit,
// which Tensor.dispose()/Session.release() alone cannot guarantee on iOS.
import { extractCharactersAI, extractForegroundAI } from './ai-segment.js';
import { releaseAllSessions } from './ort-env.js';

self.onmessage = async (event) => {
  const { imageData, hires, samFallback, mobileModel } = event.data;
  const onProgress = (received, total) => self.postMessage({ type: 'progress', received, total });
  const onStage = (text) => self.postMessage({ type: 'stage', text });
  try {
    const mobileOpts = mobileModel ? { isnetModelUrl: './models/isnet-anime-512-fp16.onnx', isnetSize: 512 } : {};
    const seg = await extractCharactersAI(imageData, { hires, samFallback, ...mobileOpts, onProgress, onStage });
    let whole = null;
    if (!seg.chars.length) {
      onStage('未检测到角色，改用整图抠取…');
      whole = await extractForegroundAI(imageData, {
        modelUrl: mobileOpts.isnetModelUrl, inputSize: mobileOpts.isnetSize, onProgress, onStage,
      });
    }
    await releaseAllSessions();
    const buffers = [];
    for (const char of seg.chars) if (char.alpha?.buffer) buffers.push(char.alpha.buffer);
    if (whole?.alpha?.buffer) buffers.push(whole.alpha.buffer);
    self.postMessage({ type: 'done', seg, whole }, [...new Set(buffers)]);
  } catch (error) {
    await releaseAllSessions();
    self.postMessage({ type: 'error', error: { name: error?.name || 'Error', message: String(error?.message || error), stack: error?.stack || '' } });
  } finally {
    self.close();
  }
};
