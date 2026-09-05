# 关键帧选取优化：实施设计与验收记录

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-09-05 |
| 需求基线 | [关键帧选取优化 PRD](../prds/tiktok-short-video-replication-keyframe-selection-prd.md) |
| 工具 | `replication_preprocess` 0.3.0 |
| 输出契约 | canonical index、原子片段与生成片段为 3.0；新增选帧文档为 1.0 |
| 运行环境 | Python 3.12+；沿用现有 PyAV、OpenCV、PySceneDetect、Pillow、FFmpeg，无新增模型依赖 |

## 1. 结构与职责

`analysis.py` 保留镜头检测、强制切点及质量指标；增加可选轻量描述符。
`selection.py` 实现有界候选、请求与提交校验、图片角色及时间绑定。
`frame_images.py` 按真实 PTS 流式保存预览、无损工作图及分页联系表。
`keyframe_workflow.py` 提供准备、观察、提交与重选的持久化操作。
`engine.py` 保留确定性时间轴规划，衔接独立审核、版本和导出。
`media_assets.py` 定义与选帧无关的视频资产身份。

Agent 依据 [选帧技能](../../skills/meta/replication-keyframe-selection.md) 作语义判断。
Python 不调用 LLM，不根据技术分自动宣布代表性审核通过。

## 2. 候选与资源默认值

新增 `representative_selection` 配置，缺省时由严格配置模型补齐；不改写现有
`default-v1.yaml`。候选数量、质量门槛和新颖度参数均在有效配置及请求中回显。

| 配置 | 默认值 |
| --- | --- |
| `initial_candidates` | 24 |
| `page_size` | 12 |
| `max_observation_rounds` | 3 |
| `observations_per_round` | 16 |
| `max_candidates` | 72 |
| `preview_long_edge_px` | 640 |
| `observation_radius_s` | `"0.25"` |
| 清晰度/平均亮度/黑白场比例门槛 | ≥80 / 16～239 / 分别 <0.95 |

首轮配额为 12 个等时间覆盖位置、6 个时间桶的技术最佳帧、最多 6 个变化补充。
位置按真实帧 PTS 最近邻定位，等距取较早帧。重复 PTS 合并理由，不为凑满复制图片。
配置改变候选上限时按二分之一、四分之一及剩余配额计算。

清晰度与曝光评分为 `Q=(8×min(sharpness/160,1)+5×exposure)/13`，其中
`exposure=1−|luma−127.5|/127.5`。运动量不参与代表帧淘汰或排序；原有强制切点规则不变。

描述符由 64 位 dHash 和 4×4 RGB 均值网格组成，距离为归一化汉明距离与 RGB
绝对差均值各占一半。补充候选依次优先技术合格、与已选画面的最小距离、相邻变化、
技术评分，最后以 PTS 打破平局；最小距离不超过 0.025 时停止补充。
近重复判断不移除固定时间覆盖证据。

定向观察默认在目标时间两侧各 0.25 秒内采样，并将范围限制在原子区间。
`full_resolution: true` 额外保存最近目标帧的原尺寸 PNG 供核实细节；它仍是观察证据。
每个片段每次调用只能包含一项观察请求。相同调用重试幂等；观察轮次保存在选帧任务中。
达到上限后仍缺证据的片段进入待人工处理，不自动换任务继续采样。

这些值是首版工程默认值，不是经过真实产品视频验证的最优参数或效果保证。

## 3. 调用与提交

沿用 `plan/run/export/validate`。在 `plan/run` 增加三个互斥输入：

| 输入 | 约束 |
| --- | --- |
| `keyframe_submission` | 按不可变请求提交全部请求项；与边界提交分开 |
| `keyframe_review_segment_ids` | 非空、不重复的原子片段 ID，启动明确的重选任务 |
| `keyframe_observation_requests` | 指定片段、请求哈希、源 PTS、理由；可选半径和原尺寸细节 |

选帧操作须带当前 canonical 的 `parent_plan_revision`。提交内部则复制原请求的
`request_id`、`request_sha256`、`protocol_version` 和来源 `parent_plan_revision`。
边界或选帧提交导致全局修订增加时，只要另一请求的输入未变，它仍可使用原请求提交。
证据、源文件、区间或相关配置变化时拒绝过期提交。

每项决定记录 `segment_id/status/rationale/evidence_refs/information_gaps`。
`ready` 需要一个 `primary_candidate_id`，可带有明确用途的 `supplementary_anchors`。
未解决项不包含采用锚点。技术不合格画面需要真实用户选择，并以 `reviewer.kind=human`
及 `quality_override_reason` 留痕；Agent 不得用这些字段伪造用户确认。

请求、提交、结果及观察输入均有发布的 JSON Schema；程序额外检查实际文件哈希、
候选身份、唯一主帧、证据引用和精确时间关系。

## 4. 状态、时间和边界

`plan_status` 只表达时间轴合法性。`anchor_status` 独立表达 `pending_agent`、
`needs_more_evidence`、`needs_human`、`ready` 或旧包的 `legacy_unreviewed`。
当前实际删除的原子片段不参与可交付锚点汇总。

`next_actions` 同时列出可以推进的边界与选帧任务；`next_action` 保留一个兼容入口。
时间轴合法时，锚点待审也可以导出视频。`validate` 通过表示包结构和引用有效，
不代表未解决的视觉审核已经完成。

每个已就绪原子片段保留一个 `primary_representative` 和按需的 `supplementary_anchor`。
锚点记录源 PTS、有理数时间基、原子片段内偏移；生成片段清单再记录生成片段内偏移。
秒数为派生展示值。一个生成片段可以含多个成员的主代表帧，剪辑边界始终独立保存。

新边界请求采用 `boundary-visual-review-v2`：原子片段中点观察、两侧最近帧及附近上下文
均固定取自源视频，角色为 `left_observation/right_observation` 等，不读取当前代表帧。
旧 v1 请求仍绑定旧证据；更换旧审核实际引用的代表帧时，仅相关边界重新待审。
原时间轴保存在历史与 `previous_timeline` 中，未重新核验前不作为当前可导出计划。

## 5. 图片工作基准和视频复用

原图工作基准是同一解码路径得到的 RGB24 画面，PNG 保存不再引入有损压缩。
源视频原本的压缩损失、RGB 转换和位深处理不在“无损保存”承诺范围内。
记录源颜色标签、实际转换路径、直角方向归一化、尺寸及运行库版本；不伪称未知标签为 sRGB。
明确 PQ/HLG 输入直接拒绝。旋转按 FFmpeg 显示方向验证；原尺寸图逐张保存释放。

选帧指纹独立于既有分析、边界审核、规划和导出指纹。已有 2.0 包再次 `run` 时自动准备
新候选，复制冻结区间及标识并保留审核、已批准删除记录和历史图片；旧图不直接改名为已审核。

视频资产指纹包括源哈希、精确区间、流选择、编码参数、方向/尺寸规则及 FFmpeg 版本，
不包括锚点、修订号或片段 ID。通过明确保存的导出历史与原始状态引用查找资产，
核对报告/文件哈希、流布局、时长和此前完整解码记录后复用原路径。
当前修订的新报告关联新锚点；损坏的个别视频只重做对应项，历史报告保留。

新导出使用 `-copyts` 及 PTS 裁剪，处理非零源起始时间。旧报告未提供媒体签名时，
只有已验证的零起始、秒数能精确表达边界的旧裁剪才直接认定等价；其他情况重新导出，
避免把旧时间处理误差当成可复用资产。这类重导出的原因是时间契约核验，而非修订号变化。

## 6. 验收方法与当前边界

自动化测试覆盖候选全区间与短暂变化、清晰运动帧、质量失败、加采上限、原尺寸细节、
双审核顺序、哈希与过期提交、多锚点时间、VFR/非零起始时间、旋转、PNG 像素一致、
旧包迁移、重选不转码及局部资产修复。候选和图片样例由测试即时创建，不读取其他生产项目。

使用实际转码调用监测验证资产复用；用 FFmpeg 解码结果验证显示方向；用同一路径解码
数组逐像素检查 PNG；通过原尺寸图片对象驻留检查防止全片图片累积。

真实产品视频上的语义代表性、困难操作覆盖率、人工介入率和生产耗时仍待样例评估。
当前没有接入或选择视频生成模型。后续消费者必须核验角色、数量、时间语义和图片规格，
不得静默删锚点或把任意时间参考图改作首帧。

### 2026-09-05 验证记录

- 首轮专项检查：57 项通过；后续补充的原尺寸观察、已批准删除区间和资产恢复场景纳入完整回归。
- 完整回归：1897 项通过、12 项跳过、3 项预期失败、1 项子测试通过；回环端口测试因沙箱限制在该次调用中排除，另行复验通过。
- 网络保护测试单独复验：6 项通过、1 项跳过，包含被排除的回环测试；外网保护保持启用。
- `make lint`、修改模块编译检查、`pip check` 及 `git diff --check` 通过。
- 完整回归有一项既有 Starlette/AnyIO 弃用警告；未新增依赖冲突。
- 用户已有 `default-v1.yaml` 内容哈希与实施前一致。

完整回归命令将素材缓存指向临时目录，避免无关测试写入用户目录：

```bash
OPENMONTAGE_CACHE_DIR=/private/tmp/openmontage-keyframe-regression-cache-20260905 \
  .venv/bin/python -m pytest tests/ -q \
  --deselect=tests/test_network_guard.py::TestGuardBlocksOutbound::test_loopback_still_permitted
```

在允许本地回环端口的测试环境运行：

```bash
.venv/bin/python -m pytest tests/test_network_guard.py -q
```

日常功能回归使用 `make test-replication`，已包含新增选帧及图片工作基准测试。
