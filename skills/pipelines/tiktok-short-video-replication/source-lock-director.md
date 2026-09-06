# Source Lock Director - TikTok Short Video Replication

## Goal

Produce `replication_source_snapshot`, binding this run to one verified,
immutable Phase 1 package.

## Process

1. Read the explicit Phase 1 canonical-index path. Do not guess it by scanning a
   project directory.
2. If Phase 1 is incomplete, block source lock and route the user back to the
   Phase 1 replication workflow. `replication_preprocess` is optional here and
   is not a prerequisite when a ready immutable Phase 1 delivery already exists.
   After Phase 1 publishes, resume this source-lock stage from its explicit index.
3. Require Phase 1 schema `3.0`, `tool=replication_preprocess`,
   `plan_status=ready`, `anchor_status=ready`, and a validated delivery package.
4. Call `replication_scene_replacement.initialize` with the direct-generation canonical output
   path `artifacts/replication/scene-replacement-v2/index.json`, upstream index
   path, and optional explicit plan revision. The retired mask/composite
   `scene-replacement/` namespace is unsupported and is never a resume source.
   If the chosen Phase 1 revision contains a supplementary anchor, also provide
   its exact `anchor_id`, purpose, observed gap, upstream plan revision, and the
   human confirmation that authorized that added edit.
5. Verify the returned source snapshot and referenced files, then checkpoint the
   compact `replication_source_snapshot` artifact.

## Checkpoint Artifact Mapping

Use the artifact returned by the successful `initialize` call verbatim:

```python
data = tool_result.data
checkpoint_artifacts = data["checkpoint_artifacts"]
artifacts = {
    "replication_source_snapshot": checkpoint_artifacts["replication_source_snapshot"]
}
```

Pass that mapping to `write_checkpoint` for `stage="source_lock"`. Do not hand-build
or copy fields from the larger state: the returned wrapper binds the exact canonical
index and source-snapshot hashes from the same Engine revision. A missing wrapper is
a contract failure, not permission to synthesize one.

## Snapshot Requirements

The snapshot must preserve the immutable delivery and plan-state references;
source-video hash; atomic and generation timeline; dropped segments; every
editable anchor's role, purpose, IDs, stable `edit_slot_id`, exact PTS/time base
and offsets; selection references; canonical PNG hash; and its dimensions,
format, orientation, and source-to-working color transform. Also freeze the
validated `sequence-v1` presentation map (`Sxx.mp4` ownership and `Sxx_Kxx`
keyframe names) so later stages never infer order from directories. Preserve the
revision-bound supplementary-anchor confirmations in the snapshot; do not infer
authorization from the anchor's mere presence.

The observed canonical-index hash records what was read during initialization,
but it is not a permanent dependency: that index legitimately advances. Active
dependencies point to immutable content. Seconds are display values; verify exact
time from PTS and time base.

## Rebase

Only `initialize` may create the first replacement revision without
`parent_replacement_revision`. A later upstream revision is an explicit rebase
against the current replacement revision. Compare timeline and anchor content
fingerprints and invalidate only changed dependencies. A new upstream revision
with byte-identical effective inputs may reuse prior results with recorded
evidence.

Reject pending, legacy-unreviewed, missing, mixed-revision, hash-mismatched, or
path-escaping inputs. Report the exact blocker rather than silently choosing a
different Phase 1 revision.
