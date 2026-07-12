# 聖地巡礼 · 调色对比生成器

这是一个独立的纯前端小工具，和 `ui-preview-project1` 的日语学习网站没有关系。

## 一键调色能力

- CDF 亮度曲线匹配：把实景照片的非线性影调匹配到动画截图
- MKL 二维色彩迁移：同时处理 Lab 的 a/b 相关性，减少色相搬运错误
- 天空/地景分区：两侧都识别到天空时分别迁移，失败自动退回全局路径
- 字幕带保护：动画统计默认忽略底部 12%
- 饱和度软提升与高光 Bloom
- 预览保持 1400px 流畅，导出时在原始分辨率上分块重放同一套调色
- 洋葱皮构图对齐：拖拽、缩放、裁剪后再进入调色
- 角色合成去色渗、光照融合、可调软阴影与颗粒匹配
- 参数自动恢复、按作品保存风格预设
- Service Worker 持久缓存四个 AI 模型，并显示缓存状态
- 手机端内存保护、HEIC 明确提示与 EXIF 方向校正
- 导出 65³ `.cube` LUT（含影调、色彩、饱和度；不含空间相关的天空分区和 Bloom）

## 启动

双击：

```bash
./启动圣地巡礼工具.command
```

也可以手动运行普通静态服务，但这会使 ONNX Runtime 退回单线程：

```bash
python3 -m http.server 8126 --bind 127.0.0.1
```

然后打开：

```text
http://localhost:8126/
```

双击启动器时，如果 8126 已被其他本地项目占用，会自动改用 8127–8136 中的空闲端口；终端窗口会显示实际地址。
推荐始终使用双击启动器，因为它会发送 COOP/COEP 响应头，让 AI 推理启用多线程。

## 文件说明

- `index.html`：页面结构
- `style.css`：界面样式
- `app.js`：上传、预览、导出等主逻辑
- `color.js`：LAB 色彩迁移和 LUT 导出
- `qa/grade-qa.html`：调色开发验收页（真实素材、数值、性能、LUT 回验）
- `segment.js`：普通算法抠图
- `ai-segment.js`：浏览器内 AI 抠图（整图直抠 + 检测→裁剪→抠→合并流水线）
- `detect.js`：动画人物检测（YOLOv8s，先找角色框再抠，解决多人同框/小角色）
- `ort-env.js`：onnxruntime-web 加载与模型会话缓存
- `models/isnet-anime-fp16.onnx`：AI 抠图模型（ISNet-anime）
- `models/person-detect.onnx`：人物检测模型（deepghs/anime_person_detection）
- `sam-segment.js`：SlimSAM 点提示分割（ISNet 抠不出的小角色自动兜底）
- `models/sam-encoder.onnx` / `models/sam-decoder.onnx`：SAM 模型（Xenova/slimsam-77-uniform）
- `THIRD_PARTY_LICENSES.md`：模型与运行时许可来源
- `DEPLOYMENT.md`：公开部署、响应头与素材排除清单

## 开发验收

服务启动后打开：

```text
http://localhost:8126/qa/grade-qa.html
```

验收页会自动运行 CDF、天空检测、同图恒等、Bloom=0、65³ LUT 随机点回验和 1400px 性能测试，并展示旧版/新版视觉对比与天空掩膜。

正式页面也提供隐藏演示入口，便于走完整 UI 链路：

```text
http://localhost:8126/?qa-demo=1
```
