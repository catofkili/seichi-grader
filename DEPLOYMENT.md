# 公开部署清单

## 推荐结构

- 网页代码：GitHub Pages、Cloudflare Pages 或 Netlify。
- 约 165MB 的 ONNX 权重：单独放在 Hugging Face 模型仓库或支持大文件与跨域请求的对象存储；不要把测试截图打进公开站点。
- 模型文件必须保持当前文件名，或同步修改 `detect.js`、`ai-segment.js`、`sam-segment.js` 中的 URL。

GitHub 仓库普通 Git 对单文件有 100MB 硬限制；当前最大模型约 84MB，虽未越线，但整个模型目录会显著拖慢 clone 与部署。代码与模型分离仍是更稳妥的做法。

## 必需响应头

项目根目录 `_headers` 已写入 COOP/COEP/CORP。Netlify 和部分 Pages 平台会读取它；其他平台需在控制台手动添加：

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: cross-origin
```

部署后在开发者工具执行 `crossOriginIsolated`，必须得到 `true`，ONNX Runtime 才会启用多线程 WASM。所有第三方脚本/模型响应还必须允许跨域或带合适的 CORP/CORS 头。

## 发布包素材白名单

只发布程序文件、说明文件、模型许可文本与权重。以下本地验收素材不得进入公开站点：

- `test-anime*`
- `test-izu-*`
- `test-dusk.jpg`
- `test-water.jpg`
- `test-photo-fake.jpg`
- `web-photo-*.jpg`
- `qa/`（开发验收页）

尤其是动画截图及其裁切/合成衍生图。两张 `web-photo-*` 的原始 Wikimedia 页面和作者署名未保存在项目里，因此不能靠猜测补署名，发布包应直接排除。

## 上线验收

1. 首次打开点击“缓存全部 AI 模型”，确认进度变为 `4/4`。
2. 断网刷新，确认模型状态为“已缓存”且 AI 抠图能启动。
3. 确认 `crossOriginIsolated === true`，桌面设备推理线程数不再是 1。
4. 用 48MP JPEG 验证导出尺寸；低内存手机应明确显示降级尺寸，不能静默缩图。
5. 在 iPhone 导入 HEIC，页面应给出 JPEG 转换提示；导入带 EXIF 旋转的 JPEG，预览方向应正确。
6. 清空站点数据，确认预设/缓存的首次使用体验；刷新后确认参数恢复。

