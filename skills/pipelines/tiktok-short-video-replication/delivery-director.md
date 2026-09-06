# Delivery Director - TikTok Short Video Replication

## Goal

Publish `scene_replacement_delivery`, a validated manifest and review bundle for
the current accepted still-image anchors.

## Process

1. Load the current replacement index and validate it before publishing.
2. Confirm every source anchor belongs to one physical scene; every scene has a
   selected direction, accepted main reference, resolved difficult-view
   requirement, current adopted results, prepared paired evidence, and explicit
   group approval. Also confirm the current chronological total multi-frame
   comparison has a separate human `global` pass.
3. Call `replication_scene_replacement.publish` with the current parent revision.
4. Call `validate` on the resulting package and referenced files.
5. Checkpoint the compact `scene_replacement_delivery` artifact only after all
   deterministic validation passes.

Use the wrapper from the successful `validate` result, not a manually reconstructed
manifest summary or the earlier `publish` result:

```python
data = validated_tool_result.data
checkpoint_artifacts = data["checkpoint_artifacts"]
artifacts = {
    "scene_replacement_delivery": checkpoint_artifacts["scene_replacement_delivery"]
}
```

Pass that mapping to `write_checkpoint` for `stage="delivery"`. Do not hand-build or
copy its counts, fingerprint, manifest reference, or hash. A missing wrapper is a
contract failure and delivery must remain incomplete.

## Delivery Boundary

The current set contains only dependency-valid adopted direct results that passed
all hard semantic checks and required human confirmations. Candidates, failed
results, late results, superseded tasks, unresolved reviews, and stale assets
remain in history but never appear in `accepted_results`.

For each scene include the paired source/result comparison and its cell-binding
manifest. Include the final total multi-frame comparison, global binding manifest,
accepted-assets manifest, and the exact human global approval.
For each accepted result preserve source anchor, exact source time, scene design,
exact prompt and hash, generation mode, attempt history, request/result lineage,
media hashes, model self-review, and approval evidence.
Use readable ownership names in delivery: clips remain `S01.mp4`, `S02.mp4`, and
so on; keyframes are `S01_K01.png`, `S01_K02.png`, and so on. These are immutable
aliases bound to the internal stable anchor and result IDs, not replacements for
those IDs.
Bind the manifest to the immutable, pre-publication accepted state by its
project-relative path and hash; IDs without that state reference are not an
independently auditable delivery.

Set downstream video-model adaptation to `unverified` unless a separate stage
has actually validated it. This delivery does not promise person replacement,
motion generation, audio work, or final video assembly.
