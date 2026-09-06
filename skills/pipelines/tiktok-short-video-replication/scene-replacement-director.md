# Scene Replacement Director — Direct Generation V2.1

## Goal

Produce `scene_replacement_package` from full-canvas images generated directly
by the built-in `image_gen` session tool. V2 has no mask, matting, protection-map,
background-plate, or compositing stage. Every prompt requests a 720x1280, 9:16
portrait result; an eligible model output is normalized to that exact delivery
canvas without crop or padding.

Read `skills/meta/replication-scene-replacement-review.md` before this stage.

## Audience And Reality Policy

Every physical scene targets North American TikTok commerce viewers aged 30–50.
Prefer an ordinary place the buyer could realistically own, rent, visit, or use:
a lived-in kitchen, garage, driveway, basement, patio, small workshop, utility
room, yard, campsite, vehicle, or comparable product-appropriate setting.

Do not default to a showroom, soundstage, luxury architectural set, commercial
test facility, pristine influencer loft, dramatic hero-lighting rig, or an
implausibly styled prop arrangement. Real environments may be tidy, but should
retain believable scale, wear, storage, daylight or household lighting, and
objects that support the actual use habit. The product category determines the
setting; the examples above are not a fixed menu.

## Scene Planning And Human Direction Gate

- Group anchors by physical environment identity, including non-contiguous
  recurrence. A repeated product or person alone does not prove a shared scene.
- Assign every editable anchor exactly once. Record shared layout, function,
  support, perspective, light, contact, occlusion, and activity-space constraints,
  plus anchor-specific constraints.
- Provide exactly three distinct scene directions per physical scene. Each must
  include visible changes, functional preservation, cross-view strategy,
  ownership or access rationale, real-use-habit rationale, at least two
  authenticity cues, and explicit anti-studio constraints.
- Submit the three directions unselected. Present all three to the user, then
  call `record_direction_selection` with the current plan ID, the three direction
  IDs in their presented order, their digest, the chosen ID, and the user's human
  confirmation. Checkpoint and end the turn. Do not generate the master anchor
  until every physical scene has a recorded selection receipt.

If a required functional or difficult view is missing, describe the exact gap.
Only a separately confirmed Phase 1 reselection may add a supplementary anchor;
re-lock and rebase before using it.

## Exact Prompt And Generation Handshake

The exact generation string is a first-class artifact. It must describe the
selected consumer setting and explicitly require strict visual/semantic subject
preservation: keep identity, geometry, proportions, colors, labels and text,
parts, pose, screen position, contacts, occlusions, and interaction unchanged.
It must also reject duplicate products, extra limbs or parts, product redesign,
advertising-studio styling, and impossible support geometry.
Every exact prompt must explicitly say `720x1280` and `9:16 portrait`.

For every attempt, follow this order without collapsing steps:

1. Call `prepare_edit` with the anchor, generation mode, optional bounded
   `creative_instructions`, rework binding when applicable, and inspection
   regions. The engine derives the selected target, final prompt, and ordered
   references; callers cannot override them.
2. Output the returned `generation_packet.prompt` verbatim with its SHA-256, and
   inspect every ordered reference according to its recorded role and usage.
3. Record `apply_review(pre_generation)` using the exact prompt and references.
4. Only after that pass, call `mark_dispatched`.
5. Invoke `image_gen` immediately with the exact prompt and ordered references.
6. Call `import_result` immediately, then inspect the native direct result,
   source/direct-result comparison, and declared review crops.
7. Record `apply_review(generated_result)` as the model's multimodal self-check.

`imagegen2` is the requested capability name; `image_gen` is the required actual
session tool. Do not invoke repository image-provider code and do not fall back
to another generator. Python validates state, hashes, files, dimensions, and
routing only; it does not claim semantic or visual correctness.

## Initial, Local Adjustment, And Source Regeneration

Use `initial` for the first call. A failed self-check must record complete issues
and their severities. The engine chooses the next allowable reference topology
from the persisted decision:

- `local_adjustment`: allowed only when the highest severity is `minor` and no
  hard rule failed. Edit the current generated result, while the source keyframe
  remains a separate invariant reference. The next prompt must name only the
  diagnosed local delta and repeat every preservation invariant.
- `source_regeneration`: required for any `major` or `critical` issue, or any
  hard-rule failure. The source keyframe becomes the edit target again. Exclude
  the failed generated image so its structural error cannot be propagated. The
  prompt must address the diagnosed cause.

Never dispatch while a prior execution is unknown. Reconcile it first. The cap
is the initial call plus two rework calls: three counted dispatches per logical
anchor task. At the cap, preserve the evidence and leave the requirement blocked;
do not lower the review bar or silently reset the ledger.

The target is always 720x1280. Compare the imported width/height ratio with 9:16;
if the relative difference is at most 5% (inclusive), resize the whole image with
Lanczos to exactly 720x1280. A larger ratio error blocks the result. Cropping,
padding, masking, subject restoration, and compositing are not allowed. The
bounded resize may scale the two axes slightly differently only within this
explicit 5% tolerance.

## Main Reference, Difficult View, And Expansion

Choose the anchor that best exposes function, support, contact, perspective,
lighting, labels, and human interaction as `main_anchor_id`. After its generated
result passes model self-check, present it for human sample approval and end the
turn. Validate a required difficult view before ordinary expansion.

Each expansion uses its own source frame, the fixed scene design, and the adopted
main or supplementary continuity references. Never form a sequential chain in
which the preceding generated frame replaces the current anchor's source
evidence.

After all anchors are adopted, call `prepare_scene_review` for the physical
scene. It creates one immutable paired source/generated board labeled with
readable `Sxx_Kxx` names plus a binding manifest. Inspect and present both; then
record a human `group` pass bound to the candidate ID and fingerprint and citing
both assets. Resolve cross-view conflicts before approving the physical scene.

After every scene has a current human group pass, call `prepare_global_review`.
It freezes readable 720x1280 delivery copies, creates one chronological total
multi-frame comparison across all scenes, and emits a cell-binding manifest.
Present the board and stop for a separate human `global` pass that binds the
candidate ID, fingerprint, ordered anchors, comparison hash, board, bindings,
and accepted-assets manifest. `publish` is allowed only after this exact global
candidate passes.

## Persistence Rules

- Pass the current `parent_replacement_revision` to every write operation.
- Treat imported images as history until `generated_result` self-review
  explicitly adopts one.
- Prompt, filename, resumed session, or reference-order changes do not reset the
  logical task's three-call ledger.
- Preserve exact prompt text and `prompt_sha256` in the request, generation
  packet, attempt history, and final delivery.
- The mask/composite V1 workflow is retired and unsupported. All current writes
  use the direct-generation `scene-replacement-v2/` namespace; existing direct
  2.0 state upgrades immutably to 2.1 on mutation.

## Checkpoint Artifact Mapping

After each successful state-changing tool call, checkpoint only the wrapper
returned by that call:

```python
data = tool_result.data
checkpoint_artifacts = data["checkpoint_artifacts"]
artifacts = {
    "scene_replacement_package": checkpoint_artifacts["scene_replacement_package"]
}
```

Use this mapping for `in_progress`, `awaiting_human`, and final `completed`
checkpoints. Do not hand-build counts, statuses, blockers, paths, or hashes. The
final checkpoint requires a returned package wrapper with `status="ready"`.
