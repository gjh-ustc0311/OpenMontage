# Executive Producer — TikTok Short Video Replication V2.1

## When To Use

Use this beta pipeline after a source video has a validated Phase 1 replication
delivery and the user wants traceable scene-replaced still anchors. It does not
replace people, generate motion, edit audio, or assemble a final video.

## Contract

The Agent makes scene, prompt, multimodal self-review, severity, and rework
decisions. Python freezes hashes, exact prompts, reference order, execution
lineage, dimensions, reviews, and delivery state. V2 directly generates the
complete image; masks, subject protection maps, background plates, and
compositing are outside its contract.

The requested capability name is `imagegen2`; the only executable capability is
the built-in `image_gen` session tool. Never substitute a repository generator,
provider API, or fallback without a new explicit user choice.

## Stage Order

1. `source_lock` — validate and freeze one immutable Phase 1 revision in the V2
   namespace.
2. `scene_replacement` — plan three directions per physical scene, obtain human
   selection, directly generate, self-review, rework when required, and obtain
   sample and group approval, then obtain final approval of the total multi-frame
   comparison.
3. `delivery` — publish only the globally approved current results with exact prompts.

Read the matching Director before each stage. Read
`skills/meta/replication-scene-replacement-review.md` before scene planning or
image review.

## Governance

- Run registry preflight and separately confirm that the current session exposes
  built-in `image_gen`. A missing session tool blocks generation.
- Announce the actual tool, known model information, scope, and known cost before
  every generation call. Mark unknown values as unknown.
- Target North American TikTok buyers aged 30–50. Favor believable household,
  vehicle, outdoor, hobby, or small-business use appropriate to the product;
  reject advertising-stage overdesign.
- Each physical scene presents exactly three distinct directions. Persist the
  exact presented order and digest with `record_direction_selection`; only then
  may the selected scene's master anchor be generated. Direction, main-sample,
  supplementary-anchor, scene-group, and final-global approvals are separate
  human decisions.
- Before dispatch, expose the exact prompt and hash, inspect ordered references,
  and pass the pre-generation review. After import, the model performs the full
  generated-result self-check before the user sees the image.
- Every generation prompt must explicitly request a 720x1280, 9:16 portrait
  image. If the returned width/height ratio differs from 9:16 by no more than
  5%, normalize the whole image to exactly 720x1280; a larger error blocks it.
- Minor localized non-hard defects route to `local_adjustment` on the generated
  image with the source as invariant reference. Any hard, major, or critical
  defect routes to `source_regeneration` from the source keyframe, excluding the
  failed image.
- A logical task receives one initial call and at most two rework calls. Never
  reset the ledger because a prompt, filename, session, or reference changed.
- Do not crop, pad, mask, restore source pixels, or composite. The only allowed
  geometric normalization is the bounded whole-image resize above.
- Prepare a paired source/generated board for every physical scene. After all
  scene boards pass human review, prepare one chronological total multi-frame
  board covering every anchor and stop for a final human decision. `publish`
  cannot replace or bypass that approval.
- Preserve Phase 1 sequence naming through Phase 2: clips use `S01.mp4`,
  `S02.mp4`, and so on; owned keyframes use `S01_K01.png`, `S01_K02.png`, and
  so on. Internal stable IDs remain the traceability keys.

## Resume Rule

Load `artifacts/replication/scene-replacement-v2/index.json` before every action.
The retired mask/composite V1 namespace is unsupported and must never be
resumed. Existing direct-generation 2.0 state is upgraded immutably to 2.1 on
the next mutation. Resume an imported unreviewed result at self-review and a
reviewed result at its next unmet human gate. Reconcile unknown execution before
another call. Never infer current state by scanning asset directories.
