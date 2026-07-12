# 第三方模型与运行时许可

发布前核对日期：2026-07-12。此表用于记录项目内实际使用的模型来源；发布包仍应附带各许可证全文与版权声明。

| 组件 | 项目内文件 | 来源 | 许可 | 发布判断 |
|---|---|---|---|---|
| Anime Person Detection | `models/person-detect.onnx` | [deepghs/anime_person_detection](https://huggingface.co/deepghs/anime_person_detection) | MIT | 可再分发，需保留 MIT 版权与许可文本 |
| ISNet Anime | `models/isnet-anime-fp16.onnx`, `models/isnet-anime-512-fp16.onnx` | [SkyTNT/anime-segmentation](https://github.com/SkyTNT/anime-segmentation) / [模型仓库](https://huggingface.co/skytnt/anime-seg) | Apache-2.0 | 可再分发，需附 Apache-2.0、保留 NOTICE/修改说明（如有）；512 版由本项目等比修改固定输入/Resize 尺寸 |
| SlimSAM 77 uniform | `models/sam-encoder.onnx`, `models/sam-decoder.onnx` | [Xenova/slimsam-77-uniform](https://huggingface.co/Xenova/slimsam-77-uniform) | Apache-2.0 | 可再分发，需附 Apache-2.0、保留 NOTICE/修改说明（如有） |
| ONNX Runtime Web | CDN 运行时 | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | 可使用 CDN；发布时保留第三方声明 |

## 仍需人工确认的边界

- 上表确认的是代码/权重仓库声明，不替代对训练数据权利的法律判断。
- `isnet-anime-fp16.onnx` 是项目中的 FP16 转换版，发布说明应标注“由原模型转换为 FP16 ONNX，模型能力未获原作者背书”。
- 如果日后换了模型文件，必须同时更新来源、版本/提交号和许可证副本；不能只沿用文件名。
- 动画截图、字幕、角色抠图及其衍生测试图不因模型开源而获得再发布权。
