# TikTok 短视频复刻 Phase 1 技术设计

| 项目 | 内容 |
| --- | --- |
| 对应 PRD | [`../prds/tiktok-short-video-replication-phase1-prd.md`](../prds/tiktok-short-video-replication-phase1-prd.md) |
| 设计版本 | 002 |
| 状态 | 已实施 |

## 1. 设计目标

本设计实现 PRD 版本 004 中的专用预处理能力，暂不新增流水线。代码以注册工具 `replication_preprocess` 接入 OpenMontage；创意决策仍由 Agent 和流水线技能负责，Python 只承担确定性媒体分析、规划、持久化和校验。

设计遵循四条不变量：以源文件 PTS 为真值、先冻结时间轴再导出、没有合法解就阻断、任何降级都可审计。

## 2. 工具边界与调用契约

`replication_preprocess` 继承 `BaseTool`，由注册表发现，使用统一的 `execute(params)` 接口。能力归类为 `analysis`，支持四个 `operation`：

| 操作 | 行为 | 媒体输出 |
| --- | --- | --- |
| `plan` | 探测、检测、关键帧、审核证据、Agent 判断应用、DAG 规划 | 关键帧与联系表；不导出视频片段 |
| `export` | 读取并校验指定的 `ready` 计划后导出 | 生成片段 |
| `validate` | 复核清单、时间轴、媒体、哈希和约束 | 无新增媒体 |
| `run` | 顺序执行 `plan -> export -> validate` | 仅在计划 `ready` 时导出 |

最小输入：

```json
{
  "operation": "run",
  "project_id": "product-demo-001",
  "output_path": "projects/product-demo-001/artifacts/replication/replication_preprocess.json",
  "source_path": "/absolute/path/source.mp4",
  "profile": "default-v1",
  "config_path": null,
  "parent_plan_revision": null,
  "manual_overrides": [],
  "review_submission": null
}
```

`source_path` 必须是本地普通文件；输出根目录固定解析为 `projects/<project_id>/`。`run` 在 `needs_review` 时正常返回计划和审核材料，但跳过 `export`；`blocked` 返回结构化原因；源视频致命解码错误返回失败的 `ToolResult`，不创建可消费的部分计划。

`get_status()` 只检查 FFmpeg/ffprobe、PyAV、PySceneDetect 及其 headless OpenCV 运行时。未安装 Torch/Transformers、无 GPU、无模型缓存不得影响工具状态。

## 3. 组件设计

```text
ReplicationPreprocess
├── VideoProbe                 # 流选择、PTS、VFR、旋转、音频与输入能力
├── SceneDetector              # PySceneDetect content_val 与首轮边界
├── LongSegmentRefiner         # 分层阈值、候选排序、强制切点
├── ClearFrameSelector         # 近首帧质量门槛与确定性降级
├── BoundaryReviewBuilder      # SceneDetect 指标、证据板与 Agent 审核请求
├── SegmentRegrouper           # DAG 构造与字典序最短路径
├── MediaExporter              # 从源文件精确重编码、原子写入
└── ManifestWriterValidator    # 数据契约、哈希、联系表与交叉校验
```

建议在 `tools/analysis/` 放置薄工具入口，在 `lib/replication_preprocess/` 放置无 OpenMontage UI 依赖的确定性模块。不得修改现有 `scene_detect`、`frame_sampler` 或 `video_trimmer` 的行为来承载本能力；它们的浮点秒、可选回退和 stream-copy 语义不满足本设计。

## 4. 两遍分析与时间轴冻结

### 4.1 精确探测

用 ffprobe 读取容器与流信息，再由 PyAV 顺序解码主视频流，记录实际帧 PTS。选择容器默认 disposition 的视频/音频流；多个视频流直接 `blocked`。旋转元数据归一化为导出像素方向，但源 PTS 不变。

内部时间统一为：

```text
TimePoint = { pts: int, time_base: { num: int, den: int }, seconds: decimal-string }
Interval  = [start_pts, end_pts)
duration  = (end_pts - start_pts) * num / den
```

不得用标称帧率推导时长。不同流的时间点比较前先转换为有理数；JSON 中 `seconds` 仅为展示字段，不参与规划。

### 4.2 第一遍：边界检测

显式构造 PySceneDetect `VideoStreamAv`，或在打开后验证后端身份；禁止 `open_video()` 静默回退到 OpenCV。使用 `ContentDetector` 与 `StatsManager` 在一次顺序解码中保存所有帧的 `content_val`，保留分数不低于最低阈值 20 的候选。

边界选择器在候选分数上重放分层策略：

1. 全时间轴先应用阈值 50、最短场景 12 帧；
2. 仅对持续时间达到档案上限的区间依次应用 45、40、35、30、25、20；
3. 同轮候选按“能满足端点 3.2 秒余量、阈值更高、`content_val` 更高、两侧更均衡、PTS 更早”排序；
4. 每次接受边界后递归处理子区间；未采用候选保留为 `suppressed` 诊断项；
5. 最低阈值后仍过长时进入强制切分。

强制切点优先搜索距当前区间起点 8～12 秒的帧，并在不超过 14.8 秒目标的前提下最大化“稳定、曝光、清晰、低运动”得分。若窗口无通过帧，选择窗口内最高分帧并记录 `forced_low_quality`。持续递归，直至全部区间严格短于档案上限。

完成后冻结 `boundaries` 与 `atomic_segments`；后续自动重组不能移动边界。人工覆盖会创建新修订版，并从最早受影响边界开始重算关键帧、边界证据、规划和导出。

### 4.3 第二遍：帧与邻接特征

分析帧统一按保持宽高比缩放至 320px 宽。对每个原子片段，在以下窗口逐帧取候选：

```text
[segment_start, min(segment_start + 1s,
                    segment_start + 30% * segment_duration)]
```

首版质量门槛为可配置经验值：

| 指标 | 默认通过条件 |
| --- | --- |
| Laplacian 方差 | `>= 80` |
| 8-bit 灰度均值 | `16..239` |
| 黑像素比例 | `< 0.95` |
| 白像素比例 | `< 0.95` |
| 相邻灰度平均绝对差 / 255 | `<= 0.12` |

选择最早全部通过的候选。若无候选通过，按下式选唯一最高分；并依次以 PTS 更早、帧索引更小打破平局：

```text
fallback_score = 0.40 * sharpness_norm
               + 0.25 * exposure_norm
               + 0.20 * stability_norm
               + 0.15 * proximity_norm
```

归一化定义为：`sharpness_norm=clip(laplacian_var/160,0,1)`；`exposure_norm=1-|mean-127.5|/127.5`；`stability_norm=1-clip(gray_delta/0.12,0,1)`；`proximity_norm=1-offset/window_duration`。结果标记 `low_quality_fallback`。窗口内无可解码帧则标记 `failed`，计划进入 `needs_review`。

## 5. SceneDetect 证据与 Agent 审核

### 5.1 边界指标

ContentDetector 在最低阈值扫描时通过 `StatsManager` 保存 `content_val`、`delta_hue`、`delta_sat`、`delta_lum` 和 `delta_edges`，事件通过 PySceneDetect 0.7 的 PTS-backed `FrameTimecode` 映射回 PyAV 原始帧账本。`content_val` 只作为像素突变强度和 DAG 次级代价，不计算或宣称语义相似度。

持续至少 80ms、两侧均恢复非纯色画面的黑/白场或明确淡变自动标记为硬边界；单帧闪光不判硬。系统为时长约束插入的 `forced` 边界默认 `forced_same_scene`。人工覆盖最终优先。

### 5.2 审核证据

每个采用的自然边界生成固定六帧证据：左右原子关键帧、边界前后约 500ms 上下文帧、边界前最后一帧和边界后第一帧，并额外生成左右中心遮罩背景视图。所有图片记录真实 PTS、项目相对路径和 SHA-256；总览联系表和单边界证据板均带边界 ID、阈值和 `content_val`。

首次 `plan` 产出 `boundary_review_request.json` 后返回 `needs_review/pending_agent`。Agent 按 `skills/meta/replication-boundary-review.md` 检查背景、空间结构、光照和时间连续性，提交：

```text
same_scene      -> soft，可跨越
different_scene -> hard，不可跨越
uncertain       -> unresolved，不可跨越
```

提交绑定请求哈希、父修订版、源哈希和配置指纹。最多两轮 Agent 审核；第二轮仅处理 `uncertain` 并扩展至约 ±1.5 秒上下文，之后仍不确定则 `needs_human`。Python 工具不调用 LLM；相同审核提交作为普通输入确定性重放。

## 6. DAG 全局重组

原子片段边界为 DAG 节点。普通边只覆盖从当前位置开始的 1～3 个连续原子片段，并需满足：

```text
profile.min_duration <= exact_duration < profile.max_duration
member_count <= 3
count(member.duration >= profile.min_duration) <= 1
not crosses_hard_boundary
not merges_two_independently_valid_atoms
```

自动规划不提供“借用相邻时长”、非法片段透传或 `needs_review` 边。删除边仅允许覆盖 `<1s` 原子片段，且必须同时存在 `allow_drop_under_1s=true` 和针对该原子片段的人工 `drop` 覆盖。

从起点到终点寻找完整路径，按以下代价向量做字典序最小化：

```text
(
  dropped_duration_pts,
  dropped_count,
  sum(internal_boundary_content_val),
  merged_boundary_count,
  ordered_edge_signature
)
```

最后一项由成员稳定 ID 顺序构成，保证平局确定性。不存在完整路径时生成 `ReviewItem` 和一组候选人工动作，但不自动应用，计划为 `needs_review`。

## 7. 数据模型

所有实体包含 `schema_version`，引用使用稳定 ID：

| 实体 | 关键字段 |
| --- | --- |
| `Source` | 文件哈希、流索引、time_base、首尾 PTS、VFR、旋转、音频信息 |
| `Boundary` | PTS、类型、阈值、content_val、hard 状态/原因、人工覆盖、诊断状态 |
| `AtomicSegment` | 半开 PTS 区间、左右边界、关键帧、场景组、序号 |
| `AdjacencyEdge` | 前后原子 ID、SceneDetect 强度、Agent 三态关系、硬边界结论 |
| `GenerationClip` | PTS 区间、成员 ID、关键帧、片内锚点、导出状态 |
| `DroppedInterval` | PTS 区间、人工决定、理由、对应音频区间 |
| `TimelineMap` | 源区间到输出 clip/offset 的映射或删除映射 |
| `BoundaryReviewRequest` | 请求/证据哈希、父修订版、边界指标与证据图片 |
| `BoundaryReviewSubmission` | Agent 三态判断、置信度、证据引用、理由与审核者类型 |
| `ReviewItem` | 原因、受影响实体、诊断、候选人工动作、解决状态 |
| `ExportReport` | 编码参数、实测时长、流信息、解码结果、文件哈希 |

稳定 ID 由 `source_sha256 + entity_type + exact_pts_interval + config_revision + tool_fingerprint` 计算；人类可读序号单独保存。审核提交哈希影响计划与生成片段 ID，但不改变边界和原子片段 ID。计划索引记录配置、审核协议、PySceneDetect/PyAV/FFmpeg 版本、父修订版和全部清单哈希。

边界属性归属于 `Boundary`，不得用单个片段级 `boundary_type` 混淆左右边界。生成片段的内部锚点使用相对该片段起点的精确 PTS/时间基表达。

## 8. 输出布局

不创建独立顶层 `run/`。所有结果遵循 OpenMontage 项目目录：

```text
projects/<project-id>/
├── artifacts/replication/
│   ├── replication_preprocess.json   # 原子更新的规范索引/消费入口
│   ├── revisions/rNNNN/              # 不可变计划与审核修订
│   │   ├── video.json
│   │   ├── boundaries.json
│   │   ├── atomic_segments.json
│   │   ├── scene_groups.json
│   │   ├── generation_clips.json
│   │   ├── dropped_intervals.json
│   │   ├── timeline_map.json
│   │   ├── quality_report.json
│   │   ├── review_items.json
│   │   ├── boundary_review_request.json
│   │   └── boundary_review_submission.json
│   └── exports/rNNNN/export_report.json
├── assets/images/replication/
│   └── revisions/rNNNN/
│       ├── keyframes/<segment-id>.jpg
│       └── review-round-N/<boundary-id>/...
└── assets/video/replication/clips/rNNNN/<clip-id>.mp4
```

清单先写临时文件，完成 schema、引用和哈希校验后原子替换。`replication_preprocess.json` 是下游唯一入口，声明计划状态和当前修订版；下游不得扫描目录猜测最新结果。

## 9. 媒体导出

每个生成片段直接从源文件独立解码和重编码，不使用递归产生的片段。默认输出：

- 视频：H.264 `libx264`、CRF 18、preset medium、`yuv420p`、MP4 `faststart`；
- 音频：保留对应源区间的内容，以 AAC 192 kbps 输出；保留支持的声道数和采样率，否则转为 48 kHz；
- 时间戳：输出视音频 PTS 归零，同时保留源 PTS 映射；
- 方向：把旋转元数据归一化到实际像素，避免下游重复旋转。

无音频源时输出视频流并记录。先写同目录临时文件，再执行 ffprobe、从头到尾解码、精确时长和流存在性检查，通过后原子重命名并记录 SHA-256。实测达到档案上限或低于下限时，不做静默尾裁；计划转为失效，重新规划或 `needs_review`。

片段状态为 `planned -> exporting -> validated`，失败为 `export_failed`。任何片段失败都使整次导出不可消费，不更新规范索引为成功状态。

## 10. 状态、异常与恢复

| 条件 | 结果 |
| --- | --- |
| 缺媒体依赖或不支持的输入能力 | `blocked`，无计划 |
| 自然边界尚未完成 Agent 审核 | `needs_review/pending_agent`，输出证据但不导出 |
| 两轮 Agent 审核后仍不确定 | `needs_review/needs_human`，不导出 |
| 源视频致命解码失败 | 工具失败，无部分成功结果 |
| 局部损坏帧但区间仍可分析 | 记录范围；影响边界/关键帧时 `needs_review` |
| 关键帧仅能降级选择 | 可继续，标记 `low_quality_fallback` |
| 关键帧窗口无可解码帧 | `needs_review` |
| DAG 无完整路径 | `needs_review`，不导出视频片段 |
| 人工覆盖已提交 | 新建计划修订版，废弃受影响导出引用 |
| 导出/验证失败 | 当前导出失败，规范索引不切换 |

关键帧状态固定为 `passed|low_quality_fallback|failed`；计划状态固定为 `ready|needs_review|blocked`；片段状态固定为 `planned|exporting|validated|export_failed`。

## 11. 依赖与版本策略

首个实现版本建议锁定：

```text
scenedetect-headless==0.7.1
av==17.1.0
```

PyAV 17.1.0 支持仓库的 Python 3.10 下限；PySceneDetect 必须使用 headless 包以避免引入 GUI 运行时。依赖通过 `requirements-replication.txt` 和 `openmontage[replication]` 隔离安装。运行不联网、不下载模型，也不要求 GPU。

实现依据：

- [PySceneDetect 后端说明](https://www.scenedetect.com/docs/latest/cli/backends.html)
- [PySceneDetect API 与版本固定建议](https://www.scenedetect.com/docs/latest/api.html)
- [StatsManager 帧指标接口](https://www.scenedetect.com/docs/head/api/stats_manager.html)
- [PyAV 17.1.0 Python 兼容信息](https://pypi.org/project/av/17.1.0/)

## 12. 配置基线

当前实现已经升级为严格的 v2 分层配置；包内基线、项目覆盖规则和阶段失效语义见
[`docs/replication-preprocess-config.md`](../../docs/replication-preprocess-config.md)。下面的 v1
片段仅保留为本设计初稿的历史记录。

```yaml
config_revision: replication-preprocess-v1
profile:
  name: default-v1
  min_duration_s: "3.2"
  max_duration_s: "15.0"
  max_exclusive: true
  forced_split_target_max_s: "14.8"
scene_detection:
  backend: pyav
  initial_threshold: 50
  threshold_step: 5
  minimum_threshold: 20
  min_scene_len_frames: 12
keyframe:
  analysis_width_px: 320
  search_window_s: "1.0"
  search_window_ratio: "0.30"
  laplacian_min: 80
  luma_min: 16
  luma_max: 239
  black_ratio_max: "0.95"
  white_ratio_max: "0.95"
  stability_delta_max: "0.12"
review:
  protocol_version: boundary-visual-review-v1
  max_agent_rounds: 2
  context_offset_s: "0.5"
  expanded_context_offset_s: "1.5"
regroup:
  max_atomic_segments: 3
  allow_merge_two_normal_segments: false
  allow_cross_hard_boundary: false
  allow_drop_under_1s: false
  drop_threshold_s: "1.0"
```

小数以字符串进入配置，再解析为 `Fraction`/`Decimal`，避免二进制浮点改变边界判定。

## 13. 验证策略

### 13.1 单元测试

- PTS、Fraction、半开区间、VFR 和非零首 PTS 运算；
- 分层阈值、12 帧抑制、候选排序、递归与强制切分；
- 清晰度/曝光/稳定性指标和确定性降级；
- 稳定 ID、修订失效和清单交叉引用；
- DAG 结果与小规模穷举解对照，验证字典序代价。

### 13.2 集成测试

用可重复生成的夹具覆盖 CFR/VFR、非零 PTS、旋转、音视频起点偏移、有/无音频、黑白/模糊起始帧、31 秒无切点、局部损坏帧，以及精确位于 3.2、14.8、15 秒附近的边界。媒体测试必须验证完整解码和时间轴映射，不能只检查文件存在。

### 13.3 契约与端到端测试

- 注册表发现、依赖状态与四种操作的返回契约；
- 所有 JSON schema 正反例、路径约束、哈希和原子索引切换；
- 显式验证 PyAV/PySceneDetect 缺失时阻断，Torch/Transformers/模型缓存缺失不影响可用性；
- 审核请求/提交 schema、证据哈希、路径逃逸、过期父修订和两轮不确定门禁；
- 覆盖 PRD 中 `3.3+4.1`、`1.4+2.0`、四个 2 秒、孤立 0.7 秒、硬边界间 1.8 秒、起始黑场和 VFR 样例；
- 相同源、配置、工具版本和审核提交得到相同计划及 ID，不要求不同 Agent 独立判断一致，也不要求跨 FFmpeg 版本媒体字节一致。

## 14. 实施顺序与完成条件

1. 定义版本化 schema、配置解析、PTS 类型和清单校验器；
2. 实现探测、两遍分析、分层边界与关键帧模块；
3. 实现 SceneDetect 边界证据、Agent 审核协议、硬边界和场景组；
4. 实现 DAG 规划、两轮审核、人工覆盖修订和联系表；
5. 实现精确媒体导出、原子发布及全量测试；
6. 注册工具并补充 `get_status()`、使用说明和契约测试。

完成条件是：PRD 的 FR-001～FR-011 均有自动化契约或行为测试；所有 `ready` 输出可被规范索引消费；`needs_review`、`blocked` 和失败路径均不会留下可误用的视频片段。新复刻流水线、Phase 2 场景替换和 Backlot 交互另立设计评审，不包含在本实施批次中。
