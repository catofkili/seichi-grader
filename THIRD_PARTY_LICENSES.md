# 第三方模型与运行时许可

最近核对日期：2026-07-13。此表用于记录项目内实际使用的模型来源；发布包仍应附带各许可证全文与版权声明。

| 组件 | 项目内文件 | 来源 | 许可 | 发布判断 |
|---|---|---|---|---|
| Anime Person Detection | `models/person-detect.onnx` | [deepghs/anime_person_detection](https://huggingface.co/deepghs/anime_person_detection) | MIT | 可再分发，需保留 MIT 版权与许可文本 |
| ISNet Anime | `models/isnet-anime-fp16.onnx`, `models/isnet-anime-512-fp16.onnx` | [SkyTNT/anime-segmentation](https://github.com/SkyTNT/anime-segmentation) / [模型仓库](https://huggingface.co/skytnt/anime-seg) | **权重许可未标注**；代码仓库为 Apache-2.0 | **公开发布阻塞项**：不能只凭训练代码仓库的 Apache-2.0 就断言 ONNX 权重可再分发。需取得权重作者的明确许可，或替换为模型卡明确标注可再分发许可的权重。 |
| SlimSAM 77 uniform | `models/sam-encoder.onnx`, `models/sam-decoder.onnx` | [Xenova/slimsam-77-uniform](https://huggingface.co/Xenova/slimsam-77-uniform) | Apache-2.0 | 可再分发，需附 Apache-2.0、保留 NOTICE/修改说明（如有） |
| ONNX Runtime Web | CDN 运行时 | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | 可使用 CDN；发布时保留第三方声明 |

## 2026-07-13 核查结论

- `deepghs/anime_person_detection` 的模型页明确标注 `mit`；`Xenova/slimsam-77-uniform` 的模型页明确标注 `apache-2.0`。
- `SkyTNT/anime-segmentation` 的**代码仓库**是 Apache-2.0，但当前链接的 `skytnt/anime-seg` **模型页没有许可证字段**。这不足以授权公开分发本项目内的 ISNet ONNX 权重、它们的分块，或由其转换出的 512 版本。
- 因此在取得书面/模型卡许可前，公开部署包应排除两个 ISNet 完整权重及所有 `.part00`–`.part02` 分块；开发和个人本地使用应另行判断。
- 当前仓库只有这份许可说明，没有随发布包附上 MIT/Apache-2.0 的许可证全文、上游 `NOTICE`（若有）或每个二进制的来源版本/校验值。即使解除 ISNet 问题，公开发布前也应补齐这些材料。

## 仍需人工确认的边界

- 上表确认的是代码/权重仓库声明，不替代对训练数据权利的法律判断。
- `isnet-anime-fp16.onnx` 是项目中的 FP16 转换版。只有在确认原始权重许可允许再分发和转换后，发布说明才应标注“由原模型转换为 FP16 ONNX，模型能力未获原作者背书”。
- 如果日后换了模型文件，必须同时更新来源、版本/提交号、SHA-256 和许可证副本；不能只沿用文件名。
- 动画截图、字幕、角色抠图及其衍生测试图不因模型开源而获得再发布权。
