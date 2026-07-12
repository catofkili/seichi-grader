# seichi-grader 调色升级开发规格书

版本 v1 · 2026-07-12 · 本文档自包含，实现者不需要本仓库之外的任何上下文。

## 0. 一句话目标

把「实景照片一键模仿动画截图的影调色调」从目前的全局 Reinhard 统计迁移，升级为：**亮度曲线自动匹配（CDF）+ 天空/地景分区迁移 + MKL 最优传输色彩映射 + Bloom/饱和度收尾**。共 4 个工作包（WP1–WP4），按编号顺序做，每个工作包独立可验收、可单独上线。

## 1. 项目背景与硬约束

- 本项目是纯前端静态站：无构建系统、无 npm、无框架，原生 ES Modules，`python3 -m http.server 8126` 直接跑。**禁止引入任何依赖、构建步骤或后端**——这是产品约束（公开部署零算力成本），不是偷懒。
- 浏览器目标：现代 Chrome/Safari。所有计算在访客端 Canvas/TypedArray 上完成。
- 用途：圣地巡礼（动画取景地）对比图制作。用户上传一张动画截图 + 一张实景照片，工具把照片调成动画的影调色调，可并排导出，也可导出 .cube 3D LUT 给 PR/达芬奇用。
- 典型素材：《摇曳露营△》这类风景动画截图——**大面积天空 + 地景**构图占绝对多数；截图常带**底部硬字幕**。
- 代码注释用中文，风格与现有文件一致。
- **不要动抠图流水线**：`segment.js`、`ai-segment.js`、`detect.js`、`sam-segment.js`、`ort-env.js`、`models/` 与本任务无关，一行都不要改。

## 2. 现状代码地图（调色链路）

```
index.html          UI。相关 id：
                    #mode(下拉 full/chroma) #strength(0-100) #strengthVal
                    #canvasOrig #canvasGraded #btnExportLut #status
color.js            全部调色算法（本次改造主战场）：
                    rgb2lab/lab2rgb          sRGB<->CIELAB(D65)
                    labStats(imgData,step)   L/a/b 均值+标准差（跳过 alpha<8 像素）
                    makeLabTransform(src,tgt,mode,strength)
                                             Reinhard: (x-μs)*σt/σs+μt；mode='chroma' 时 L 不动；
                                             返回 (L,a,b)=>[L',a',b'] 的纯函数
                    applyTransfer(imgData,tf) 逐像素套用，返回新 ImageData
                    extractPalette           k-means 主色（仅展示用，不参与迁移）
                    generateCubeLUT(tf,33)   把同一个纯函数烘成 .cube 文本
app.js              调色链路调用点：
                    L69  recompute()：photo+anime 都在时
                         labStats(photo,2) → makeLabTransform(srcStats, state.anime.stats, mode, strength)
                         → applyTransfer → 画到 #canvasGraded
                    L175 上传动画截图时缓存 state.anime.stats = labStats(imgData,2)
                    L108 harmonizedCutout()：抠出的角色向实景做 'chroma' 迁移（光照融合滑杆）
                         ——这条链路复用 makeLabTransform，改造时保持其可用
                    导出 LUT 按钮用当前 transform 调 generateCubeLUT
```

图片上传时已统一缩放到最长边 ≤1400（`fileToImageData`），所以逐像素全图操作的预算是 1400×787 ≈ 110 万像素。

## 3. 工作包总览

| WP | 内容 | 解决什么 | 预估 |
|----|------|---------|------|
| WP1 | 亮度 CDF 匹配 | 影调（暗部利落/中调亮/高光收敛的非线性曲线），现有线性迁移表达不了 | 0.5 天 |
| WP2 | 天空分区迁移 | 全局统计把"天空的蓝"平摊到地面、天空本身反而不够蓝 | 1 天 |
| WP3 | MKL 最优传输 | a/b 通道各自独立迁移丢掉了通道相关性，色相搬运不准 | 0.5 天 |
| WP4 | Bloom + 饱和度软提升 | 颜色映射之外的"动画感"（高光辉光、高饱和不溢出） | 0.5 天 |

依赖关系：WP2 依赖 WP1（分区内的迁移函数复用 WP1/WP3 的实现）；WP3 可与 WP2 互换顺序；WP4 完全独立。

## 4. WP1 亮度 CDF 匹配

### 4.1 算法

替换 L 通道的线性迁移，改为直方图匹配 + 平滑正则：

```
1. 对照片和动画截图各采样（sampleStep=2，跳过 alpha<8；动画侧还要跳过字幕带，见 §8.1），
   把 L∈[0,100] 装进 256 bin 直方图。
2. 各自算累积分布 cdfS(照片)、cdfT(动画)，归一化到 [0,1]。
3. 原始映射：map[i] = argmin_j |cdfT[j] - cdfS[i]| ，即 T(l) = QT(cdfS(l))。
4. 正则化（顺序执行，防断带/防极端拉伸）：
   a. 盒式平滑：半径 4 bin，跑 2 遍。
   b. 斜率限幅：逐点保证 0.25 ≤ (map[i]-map[i-1])/binWidth ≤ 4，超限截断。
   c. 重新单调化：map[i] = max(map[i], map[i-1])。
5. 强度混合：mapS[i] = (1-s)*identity[i] + s*map[i]（s = #strength/100）。
6. 查表用 1024 级线性插值 LUT（Float32Array(1024)），封成 (L)=>L' 纯函数。
```

### 4.2 集成

- `color.js` 新增 `makeLumaCdfMap(srcImgData, tgtImgData, opts) -> {mapL: (L)=>L', table: Float32Array}`，`opts.ignoreBottomRatio`（见 §8.1）。
- `makeLabTransform` 改造为可注入 L 映射：新增参数或新函数 `makeGradeTransform(srcStats, tgtStats, { mode, strength, mapL })`，当 `mapL` 存在且 mode='full' 时 L 通道走 `mapL`，否则保持现状。**保持旧签名可用**（harmonizedCutout 还在用）。
- `app.js` recompute()：photo/anime 任一变化时重算 `mapL` 并缓存（挂 `state.gradeCache`，上传新图时失效）；strength 滑杆变化只需重做第 5 步混合，不必重算直方图。
- `#mode` 下拉加第三项：`tone`＝「影调+色彩（推荐）」，即 mapL+色彩迁移；原 full/chroma 语义不变，默认选 tone。
- LUT 导出无需改动：mapL 是纯逐点函数，`generateCubeLUT` 直接兼容。

### 4.3 验收

- 用仓库根目录 `test-izu-scenery.jpg`（动画）配任意风光照片：结果图的 L 直方图与动画的 KS 距离显著小于 Reinhard 基线（写个临时 console 断言即可，不用留测试框架）。
- 天空渐变区无可见断带（放大看无色阶台阶）。
- strength=0 时输出与原照片逐像素一致（±1/255）。
- 导出 .cube 后在页面内做数值回验：对 4096 个随机 RGB，LUT 三线性插值结果 vs 直接 transform，最大误差 ≤ 2/255。

## 5. WP2 天空分区迁移

### 5.1 天空掩膜（启发式，禁止上分割模型）

对照片和动画截图各自执行（在 256px 宽的缩略图上算，最后放大回原尺寸）：

```
1. 候选像素打分：亮度高（L > 全图 60 分位）＋ 局部纹理低（5×5 窗口 L 标准差 < 6）
   ＋ 偏蓝/中性（Lab 的 b < 15）。三条全满足记 1，否则 0。
2. 从顶部 30% 行内的候选像素做 flood fill（4 邻接，只在候选像素上生长）。
3. 只保留接触图像顶边的连通域；总面积 < 全图 5% 时判定"无天空"。
4. 掩膜放大回原尺寸后做 ~20px 半径的羽化（box blur），得到软权重 wSky∈[0,1]。
```

### 5.2 分区迁移

两侧都有有效天空掩膜时：

```
tSky  = 用（照片天空像素 ↔ 动画天空像素）的统计/直方图构建的 transform
tLand = 用（照片非天空 ↔ 动画非天空）构建的 transform
输出像素 = wSky * tSky(pix) + (1-wSky) * tLand(pix)   // 在 Lab 域混合后再回 RGB
```

任一侧无天空 → 自动退回全局单 transform（WP1 的路径），UI 状态栏注明「天空分区：未启用」。

### 5.3 集成

- `color.js` 新增 `skyMask(imgData) -> {weight: Float32Array(w*h), valid: bool}`；`applyTransfer` 加一个带权双 transform 的变体 `applyTransferRegioned(imgData, tSky, tLand, weight)`。
- `labStats` / `makeLumaCdfMap` 加可选 `weightMask` 参数（带权采样），供分区统计复用。
- UI：`#mode` 附近加 checkbox `#skyRegion`「天空分区迁移」，默认勾选；无效时禁用并显示原因。
- **LUT 导出**：分区映射不是单一全局 LUT。`#btnExportLut` 保持只导出**全局** transform（不含分区），按钮旁加一行小字说明「LUT 为全局近似，不含天空分区」。不要试图导出双 LUT。
- 掩膜计算结果缓存进 `state.gradeCache`，滑杆变化不重算掩膜。

### 5.4 验收

- 准备 3 对素材自测：晴天蓝天、黄昏、阴天（动画侧可从用户已有素材取，路径见 §9）。
- 晴天对：地景（非天空区）的平均 a/b 相比全局迁移基线不再整体偏蓝（Δb 改善肉眼可见，截图对比留档）。
- 天空区平均色相与动画天空平均色相 ΔE*ab < 10。
- 阴天/夜景对：自动退回全局路径，不崩、状态栏提示正确。
- 照片顶部有白墙/水面反光时不把它们当天空（顶边连通性约束生效）。

## 6. WP3 MKL 最优传输（替换 a/b 的 Reinhard）

### 6.1 算法

L 通道继续走 WP1 的 CDF。a/b 二维用 Monge-Kantorovich 线性最优传输：

```
Σs, Σt = 照片、动画 ab 的 2×2 协方差矩阵（采样同 labStats）
T = Σs^(-1/2) · ( Σs^(1/2) · Σt · Σs^(1/2) )^(1/2) · Σs^(-1/2)
映射：[a',b'] = T · ([a,b] - μs) + μt
```

2×2 对称正定矩阵的平方根/逆平方根用特征分解闭式解（解一元二次求特征值，手写即可，几行代码）。

正则化：
- 求逆前 Σs += εI（ε=1e-4）。
- 对 T 做 SVD（2×2 闭式），奇异值钳到 [0.25, 4] 再重组——防极端拉伸把噪声放大成色斑。

strength 混合：`x' = x + (T·(x-μs)+μt - x) * s`（与现状一致的线性插值语义）。

### 6.2 集成

- `color.js`：`labStats` 扩展为可返回协方差（加 `cov: true` 选项，返回 `covAA, covAB, covBB`；注意保持默认返回结构不变，旧调用方不受影响）。
- `makeGradeTransform` 内部：mode='tone' 或 'full' 或 'chroma' 时 ab 一律走 MKL（Reinhard 的 ab 分支删除即可，MKL 在 Σ 对角时严格退化为 Reinhard，无兼容风险）。
- `harmonizedCutout()` 的 'chroma' 调用自动受益，无需改 app.js。
- LUT 导出天然兼容（仍是逐点纯函数）。

### 6.3 验收

- strength=100、素材=同一张图自迁移时，输出≈原图（T≈I，逐像素 ≤2/255）。
- 黄昏对（橙紫大斜向色分布）：MKL 结果的 ab 散点与动画的 2D 分布重叠度优于 Reinhard 基线（临时脚本算 2D 直方图相交即可）。
- 无新增色斑/出界溢色（lab2rgb 截断像素占比相比基线不升高 >1%）。

## 7. WP4 Bloom + 饱和度软提升

### 7.1 饱和度软提升（逐点，可进 LUT）

```
C = sqrt(a²+b²)
gain(C) = 1 + k * exp(-((C-35)/25)²)        // 中饱和区增益最大，k=滑杆值(0..0.5)
C' = C * gain(C)
软限幅：C' > 90 时 C' = 90 + (C'-90)*0.3    // 防溢出荧光色
a,b 按 C'/C 等比缩放（C=0 时跳过）
```

集成进 `makeGradeTransform` 的末端（在 MKL 之后），因此**自动进 LUT 导出**。

### 7.2 Bloom 辉光（空间操作，不进 LUT）

在调色完成后的输出上做：

```
1. 亮部掩膜：m = smoothstep(0.72, 0.92, luma)   // luma 用线性域 Y
2. 掩膜×原图 → 缩到 1/4 尺寸 → box blur(r≈min(w,h)/50) 跑 3 遍近似高斯 → 放大回原尺寸
3. screen 混合：out = 1-(1-base)*(1-bloom*gain)，gain=滑杆值(0..0.6)
```

### 7.3 集成

- `color.js` 新增 `applyBloom(imgData, gain)`（用离屏 canvas 的 drawImage 做缩放，blur 手写 box 即可）。
- `app.js` recompute() 末尾、画布绘制前调用；`invalidate` 时机同现有 gradedData。
- UI：调色面板加两个滑杆 `#satBoost`「饱和度 0–50%，默认 15%」、`#bloom`「辉光 0–60%，默认 25%」；实时重算。
- 导出调色图/对比图走同一条链路（含 bloom）；**LUT 导出含饱和度、不含 bloom**，按钮说明文字补一句。

### 7.4 验收

- 默认参数下：亮部边缘无明显光晕方块（1/4 下采样伪影）；把 bloom 拉到 0 输出与调色结果逐像素一致。
- 性能：1400px 图上 recompute 全链路（含 bloom）在 M 系 Mac 上 < 300ms（console.time 留档后删掉）。

## 8. 通用工程约定

### 8.1 字幕带保护（所有统计的公共前置）

动画截图常带底部硬字幕，会污染直方图/统计/天空检测。规则：**凡是对动画截图采样**（labStats、CDF 直方图、天空掩膜、调色板），默认忽略底部 12% 行。加全局 checkbox `#ignoreSub`「忽略底部字幕带」默认勾选，塞进动画上传区下方。照片侧不受影响。

### 8.2 缓存与性能

- 每张图的派生数据（stats、直方图、天空掩膜、mapL）挂 `state.gradeCache = { photo:{...}, anime:{...} }`，上传新图时对应侧整体失效。
- 滑杆（strength/satBoost/bloom/harmonize）只做轻量重混合/重套用，不允许触发直方图或掩膜重算。
- 全链路保持同步代码即可（预算内），不要引 Worker——增加复杂度不值得。

### 8.3 LUT 一致性纪律

`generateCubeLUT` 只接受**逐点纯函数**。凡新增的映射，先问"是不是纯逐点"：是（CDF、MKL、饱和度）→ 必须进 LUT；不是（天空分区、bloom）→ 明确排除并在 UI 注明。禁止让 LUT 导出和页面预览悄悄不一致而不告知用户。

## 9. 测试素材与验证方式

- 仓库根目录现成三张：`test-anime.png`、`test-izu-scenery.jpg`（风景动画截图）、`test-izu-multi.jpg`。
- 更多动画截图（461 张《摇曳露营》伊豆篇）在机主 `~/Downloads/伊豆.zip`；若该路径不存在，用上面三张即可覆盖验收。
- 实景照片：任意风光照均可（蓝天/黄昏/阴天各备一张）。
- 启动：项目根 `python3 -m http.server 8126 --bind 127.0.0.1`，浏览器开 `http://localhost:8126/`。
- 每个 WP 完成后：上传素材对 → 截图前后对比留档 → 核对该 WP 验收清单 → 再进下一个 WP。

## 10. 明确不要做的事

- 不引入任何 npm 依赖、打包器、CSS 框架、Worker、WASM 图像库。
- 不上神经网络方案（风格迁移网络/扩散模型/学习式 LUT）——已评估否决：结构伪影风险、模型体积、与"实景保持原构图"的产品目标冲突。
- 不动抠图相关文件（§1 列表）。
- 不做天空之外的语义分割（草地/建筑分区等）——收益边际递减，掩膜错误的代价反而大。
- 不改 `fileToImageData` 的 1400px 上限。
