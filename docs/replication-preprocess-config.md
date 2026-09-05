# TikTok 复刻预处理配置

`replication_preprocess` 的业务参数集中在包内默认 profile：

`lib/replication_preprocess/profiles/default-v1.yaml`

本配置只作用于 TikTok 复刻 Phase 1 的 `replication_preprocess`，不会改变通用
`tools/analysis/scene_detect.py` 或 `video_analyzer` 的行为。

项目只需要保存差异项。建议路径为：

`projects/<project-id>/config/replication-preprocess.yaml`

调用时必须显式传入 `config_path`；工具不会扫描项目目录、环境变量或其他隐式配置。例如：

```yaml
scene_detection:
  initial_threshold: 45
  threshold_step: 5
  minimum_threshold: 20

keyframe:
  laplacian_min: 70

export:
  crf: 20
  preset: slow
```

小数参数必须写成字符串，例如 `"3.2"`、`"0.30"`。整数、布尔值仍使用 YAML 原生类型。未知字段、类型错误、越界值及不支持的组合会在视频解码前被拒绝。

## 参数归属

| 分组 | 控制内容 |
| --- | --- |
| `profile` | 生成片段最短/最长时长，强制切分目标与搜索窗口 |
| `scene_detection` | SceneDetect 阈值阶梯、最小场景帧数、黑白分隔检测 |
| `keyframe` | 分析尺寸、关键帧窗口、清晰度/曝光/稳定性门槛与评分权重 |
| `review` | 两轮审查取帧距离、背景遮罩比例、JPEG 质量 |
| `regroup` | 最多合并原子片段数及短片段删除策略 |
| `export` | CRF、x264 preset、AAC 音频码率 |

以下实现约束是锁定的：PyAV 分析后端、`< 15s` 的开区间上限、两轮 `boundary-visual-review-v1`、禁止跨硬边界、禁止合并两个正常时长片段、H.264/AAC MP4 编码格式。覆盖文件可以重复写相同值，但不能改变它们。

完整解析结果符合 `schemas/replication/replication_preprocess_config.schema.json`。每次运行还会在结果和清单中记录：

- 默认 profile 的包内资源路径、项目覆盖文件的绝对路径及各自内容哈希；
- 完整 `config_fingerprint`；
- `analysis`、`review`、`planning`、`export` 四个阶段指纹。

## 重跑规则

使用 `operation: run` 时，工具会比较阶段指纹：

| 改动 | 自动执行 |
| --- | --- |
| `profile` / `scene_detection` / `keyframe` | 重新分析，并失效审查、规划和导出 |
| `review` | 复用边界和关键帧选择，重建审查证据并要求重新审查 |
| `regroup` | 复用已接受的分析/审查，只重建分组与生成片段计划 |
| `export` | 保留计划，只导出新的 variant |

不同导出配置写入 `assets/video/replication/clips/<plan-revision>/<export-fingerprint-prefix>/`，旧 variant 不会被覆盖。返回值中的 `executed_stages`、`reused_stages` 和 `invalidated_stages` 可用于确认实际行为。

## 易读交付命名

当时间轴计划、关键帧和媒体导出均通过审核后，工具会额外发布易读交付包：

- 视频按源时间轴命名为 `S01.mp4`、`S02.mp4`；
- 每个视频关联的关键帧按片内时间命名为 `S01_K01.png`、`S01_K02.png`；
- `K01` 表示片段中的第一张代表帧，不表示时间零点；
- 完整映射写入 `artifacts/replication/delivery/<plan-revision>/<delivery-fingerprint-prefix>/manifest.json`。

视频和图片位于各自 `assets/.../replication/delivery/` 目录，均为已验证规范媒体的独立、逐字节一致副本。内部 `clip_id`、`anchor_id` 和原始路径继续保留在清单中，用于缓存、审核和跨修订追溯。`clip_paths` 保持原有语义；调用方可通过 `delivery_clip_paths`、`delivery_keyframe_paths` 和 `delivery` 获取易读交付文件。
