# TikTok 短视频复刻 Phase 2 技术计划：直出场景替换 V2.1

## 1. 实现策略

公开工具 `replication_scene_replacement` 只路由到直出状态机。删除旧蒙版、保护图、
背景板与合成引擎及其 Schema；已有直出 V2.0 状态仅在下一次写操作时升级为新的
不可变 V2.1 revision。

规范索引与素材根目录分别为：

- `artifacts/replication/scene-replacement-v2/index.json`
- `assets/images/replication/scene-replacement-v2/`

本设计属于通用视频复刻流程，不包含任何具体商品的硬编码。

## 2. 公共操作与状态流

V2.1 提供：`initialize`、`apply_scene_plan`、`record_direction_selection`、
`prepare_edit`、`mark_dispatched`、`import_result`、`reconcile_execution`、
`prepare_scene_review`、`prepare_global_review`、`apply_review`、`publish`、
`validate`。

主流程为：

`initialize → apply_scene_plan → record_direction_selection → prepare_edit → apply_review(pre_generation) → mark_dispatched → image_gen → import_result → apply_review(generated_result) → sample → prepare_scene_review → group → prepare_global_review → global → publish`

`apply_scene_plan` 对每个物理场景保存恰好三个未选择的方向；
`record_direction_selection` 把人工选择绑定到当时展示的三个方向、顺序和摘要。
所有场景都有选择回执后，才允许基于所选场景生成 master anchor。

## 3. Prompt 与参考图

`prepare_edit` 从选择回执、方向内容、共享约束、锚点约束、受众政策、生成模式和
返修问题确定性构建最终 prompt。调用方只能提供可选创意补充，不能覆盖目标、最终
prompt、有序图片参考或场景设计引用。

返回的 `generation_packet` 输出 prompt 原文、SHA-256、prompt builder 版本、
`ordered_image_references` 与 `context_refs`。预生成自检通过后才能派发；Python
状态机不调用模型，Agent 必须把该 prompt 和图片引用原样传给内置 `image_gen`。

首次与从源重生成以源关键帧为 `edit_target`；局部调整以失败生成图为
`edit_target`、源图为 `invariant_reference`。扩展视角自动加入已采用 master anchor
和需要时的困难视角，调用方不能形成随意的链式参考。

## 4. 图像边界

模型生成要求固定为 `720x1280`、`9:16`。导入完整、不透明、可解码图像后：

- 已是 720x1280：保持整图；
- 宽高比相对 9:16 的差异不超过 5%（含边界）：整图 Lanczos 调整到 720x1280；
- 超过 5%：阻断并进入返修。

不得裁切、补边、创建蒙版、恢复源像素或合成。

## 5. 自检与返修路由

`generated_result` 必须覆盖完整规则矩阵，并由大模型自检：

- 仅 minor 且没有硬规则失败：`local_adjustment`；
- 任一 hard、major 或 critical：`source_regeneration`。

失败规则集合必须与问题记录一一对应。逻辑任务最多三次实际派发：一次 initial 加
最多两次返修。确认失败的模型调用仍计数；只有确认未运行的派发可退回计数。

## 6. 比较图、人工闸门与可读命名

Phase 1 delivery 的 `sequence-v1` 映射是显示名唯一来源：片段为 `S01.mp4`、
`S02.mp4`，关键帧为 `S01_K01.png`、`S01_K02.png`。内部稳定 ID 继续用于谱系和
状态引用。

每个物理场景完成后，`prepare_scene_review` 生成一张源图/结果成对的多帧对比板和
格位映射；人工 `group` pass 必须绑定当前 candidate ID、依赖摘要和这两份证据。

所有场景组审通过后，`prepare_global_review`：

1. 为每个采用结果冻结一个 720x1280、字节一致的 `Sxx_Kxx.png` 交付副本；
2. 按片段序号、片内关键帧序号生成一张总多帧对比图；
3. 保存总图格位映射与 accepted-assets manifest。

最终人工 `global` pass 必须绑定当前 candidate、总图哈希、完整有序锚点和三份证据。
`publish` 只接受这个已通过的候选，单独的 publish confirmation 不能绕过它。

## 7. Schema 与交付

V2.1 plan、edit request、review、state、delivery 和 tool Schema 覆盖：三个方向、
选择回执、引擎构建 prompt、直出归一化、严重度路由、场景/全局候选、可读名称和
最终人工闸门。Checkpoint Schema 接受直出 2.0/2.1 迁移数据，不再注册 V1 Schema。

交付记录每张采用图的源锚点和时间、显示名、场景与选择回执、完整 prompt 和哈希、
生成模式、所有计数尝试的完整 prompt、请求/结果/审核谱系。交付同时绑定场景对比、
总对比图、格位映射、accepted-assets manifest 和最终人工确认。

## 8. 测试

覆盖三个方向与选择回执、真实北美 30–50 岁受众约束、调用方不能覆盖 prompt/
参考图、720x1280 与 5% 边界、模型自检角色、minor 局部调整、hard/major/critical
从源重生成、失败调用重试、三次上限、场景对比、总多帧对比、可读名称、全局人工
发布闸门、不可变升级和哈希验证。旧蒙版/合成模块、Schema 与测试全部删除。
