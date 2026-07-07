# Fix Patterns

**Read when**: After completing a bug fix.

---

This document defines the recording template and archival rules for fix experiences.

## Fix Entry Template

Each fix record uses the following format:

```markdown
### [Short Title]
- **Date**: YYYY-MM-DD
- **Symptom**: What the user observed (error message / abnormal behavior)
- **Root Cause**: Root cause analysis (one sentence)
- **Fix**: What was changed (files involved and key modifications)
- **Lesson**: Implications for future development (why this happened, how to prevent it)
- **Related Constraint**: If a new hard constraint was created, reference the constraint number (N/A if none)
```

## Archival Location Decision Table

Based on the fix type, write the fix entry to the appropriate document:

| Fix Type | Archival Location | Example |
|----------|------------------|---------|
| Violated an existing constraint | `constraints.md` — add "common violation case" under the relevant entry | Forgot to update registry path |
| Discovered a new hard constraint | `constraints.md` — new entry | Found ZeRO-2 + EMA incompatibility |
| Architecture / data-flow misunderstanding | `architecture.md` — relevant module section | Misunderstood preprocess_func call timing |
| Subsystem-specific pitfall | `topics/<topic>.md` — corresponding topic | Sampler boundary condition |
| Does not fit any of the above | This document's "Recorded Fix Patterns" section below | Append as a new record |

**Decision flow**: Check whether the fix matches the first four rows; if none match, fall back to this document.

## Recorded Fix Patterns

<!-- This section accumulates over time. Append new records at the end using the template above. -->

### Multi-modal batch homogeneity (R6)
- **Date**: 2026-04
- **Symptom**: Silent HF `Dataset.map` errors and inconsistent per-sample types in the `audios` column (sometimes `None`, sometimes `Tensor`, sometimes `List[Tensor]`); image/video columns had a latent batch-length mismatch when a sample contributed zero items.
- **Root Cause**: `_preprocess_batch` returned a mix of `None`, `Tensor`, and `List[Tensor]` for the same modality column, breaking Arrow's homogeneous-column requirement and forcing every downstream consumer to handle three input shapes.
- **Fix**: `data_utils/dataset.py:_preprocess_batch` now always emits `List[List[Media]]` per modality (`[]` for empty samples, `[item]` for single-item samples, multi as-is) and appends to BOTH `xx_args[xx]` and `batch[xx]` for every sample so the columns stay length-aligned. Mirrored the same shape on `models/abc.py:preprocess_func` (`audios` parameter) and `utils/audio.py` (`MultiAudioBatch` type alias).
- **Lesson**: HF Arrow demands homogeneous columns, and downstream consumers benefit from a single canonical type. When a column has variable cardinality per row, always represent it as `List[...]` even when the row is empty or has exactly one element. Never special-case "single item" by unwrapping.
- **Related Constraint**: N/A (codified in `topics/adapter_conventions.md` Gotcha #6 and the new "Multi-media batch homogeneity" bullet under Batch Dimension Convention).

### Non-abstract encoder defaults (R7)
- **Date**: 2026-04
- **Symptom**: Adding `encode_audio` as `@abstractmethod` on `BaseAdapter` would force one-line `pass` stubs on 11 existing concrete adapters, none of which consume audio. The first iteration of R6 actually shipped this — and the resulting "noise" diff dwarfed the real change.
- **Root Cause**: Incorrect default-discoverability assumption — abstract methods force every subclass to acknowledge a feature, even when the subclass doesn't use it.
- **Fix**: `models/abc.py` dropped `@abstractmethod` from all 4 encoders (`encode_prompt`, `encode_image`, `encode_video`, `encode_audio`); default body is `pass` returning `None`; `preprocess_func` skips integration when the called encoder returns `None`. The Round-6 stub overrides on 11 concrete adapters were reverted, leaving them byte-identical to `origin/main`.
- **Lesson**: When extending a base contract for a partial-coverage feature (where only some subclasses will participate), no-op default + opt-in override beats forcing every subclass to acknowledge it. Reserve `@abstractmethod` for invariants that ALL subclasses must implement (e.g. `load_pipeline`, `decode_latents`, `forward`, `inference`).
- **Related Constraint**: #12 (post-update text codifies "Optional encoder overrides (no-op default)").

### Launcher process count exceeds visible GPUs
- **Date**: 2026-05-20
- **Symptom**: Accelerate/DeepSpeed failed during initialization with `ValueError: device_id cuda:7 is out of range. Please use a device index less than the number of accelerators available: 4.`
- **Root Cause**: The launch configuration requested 8 local processes on a node where only 4 CUDA devices were visible, so local rank 7 mapped to a nonexistent `cuda:7`.
- **Fix**: `cli.py` now validates `num_processes / num_machines` against the visible local GPU count before launching, `train.py` validates externally supplied `LOCAL_RANK` before constructing `Accelerator`, and the OPD Wan2.1 example uses `num_processes: 4` with matching sampler geometry comments.
- **Lesson**: Distributed process geometry must match `CUDA_VISIBLE_DEVICES` per node. Validate before distributed initialization because the later DeepSpeed error hides the configuration-level cause.
- **Related Constraint**: N/A

### Scheduler config contains incompatible checkpoint keys
- **Date**: 2026-05-23
- **Symptom**: OPD teacher loading failed while replacing the checkpoint scheduler with `TypeError: FlowMatchEulerDiscreteScheduler.__init__() got an unexpected keyword argument 'beta_end'`.
- **Root Cause**: `scheduler/loader.py` passed the checkpoint scheduler config verbatim into the registered SDE scheduler, but the Wan reward teacher checkpoint can carry scheduler keys such as `beta_end` that are valid for other schedulers and invalid for `FlowMatchEulerDiscreteScheduler`.
- **Fix**: `scheduler/loader.py` now filters merged scheduler config keys against the target scheduler class and its parent constructors before instantiation, while preserving Flow-Factory SDE args such as `noise_level`, `sde_steps`, `num_sde_steps`, `seed`, and `dynamics_type`.
- **Lesson**: Checkpoint scheduler configs are not a stable constructor contract across scheduler families. When wrapping a diffusers scheduler class, instantiate from keys accepted by the target scheduler rather than blindly forwarding every serialized config field.
- **Related Constraint**: N/A

### Standalone inference skips LoRA component device move
- **Date**: 2026-05-23
- **Symptom**: Wan LoRA inference failed with `RuntimeError: Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type) should be the same` inside transformer `patch_embedding`.
- **Root Cause**: LoRA checkpoint loading stores wrapped target components in `BaseAdapter._components`, and `on_load_components()` skips cached components because training treats them as accelerator-managed after `accelerator.prepare()`; standalone inference never prepares them, so the wrapped transformer stayed on CPU.
- **Fix**: `inference.py` now moves all standalone inference components explicitly, including LoRA-wrapped cached components, and fails fast if any parameter or buffer remains off the accelerator device before generation starts.
- **Lesson**: `_components` does not always imply accelerator-managed. Standalone utilities that bypass trainer initialization must not rely on `on_load_components()` for LoRA-wrapped target modules.
- **Related Constraint**: #8

### UniPC eval sigmas remain on CPU
- **Date**: 2026-05-23
- **Symptom**: Wan inference reached the scheduler and failed with `Expected all tensors to be on the same device` in `multistep_uni_p_bh_update` while stacking `rks`.
- **Root Cause**: Diffusers' UniPC `set_timesteps(..., device=cuda)` moves `timesteps` to CUDA but stores `sigmas` on CPU; Flow-Factory's eval path calls the parent UniPC multistep update, which mixes scalars derived from CPU `sigmas` with CUDA sample-derived scalars.
- **Fix**: `UniPCMultistepSDEScheduler.set_timesteps()` now preserves the parent scheduler setup and then moves `sigmas` to the requested execution device.
- **Lesson**: Scheduler tensor device placement matters for multistep solvers, not just model weights and latents. Wrapper schedulers should normalize all tensors consumed by parent solver math before generation.
- **Related Constraint**: #20

### OPD supervision used reconstructed forward-process states
- **Date**: 2026-05-30
- **Symptom**: OPD teacher matching trained on latents reconstructed by adding fresh noise to the final student latent instead of on states from the student's actual denoising trajectory.
- **Root Cause**: The MVP reused the NFT/AWM final-latent forward-process pattern, which is not equivalent to replaying the solver states produced by the student rollout.
- **Fix**: `OPDTrainer` now defaults to trajectory-mode supervision: sampling stores selected student denoising latents, each sample records the supervised step indices, and optimization replays those stored latents/timesteps for both student and frozen teacher velocity prediction. The forward-process path remains available only as an explicit legacy mode.
- **Lesson**: OPD-style step supervision is path-dependent. A final latent plus fresh noise cannot represent the student's generated trajectory unless the solver path is exactly linear and the same rollout noise is reused, which is not a valid framework invariant.
- **Related Constraint**: N/A

### Sparse raw metadata columns poison Arrow schema
- **Date**: 2026-06-01
- **Symptom**: Dataset preprocessing failed during `Dataset.map` with `TypeError: Couldn't cast array of type string to null`.
- **Root Cause**: `_preprocess_batch` returned arbitrary raw CSV columns and nested dict metadata directly to Arrow; when a sparse raw column was all null in an early batch and later contained strings, PyArrow inferred an incompatible `null` feature.
- **Fix**: `data_utils/dataset.py` now keeps only canonical training inputs at the top level and serializes raw metadata rows as JSON strings; `OPDTrainer._attach_opd_context` accepts either legacy dict metadata or the new JSON metadata string.
- **Lesson**: Raw dataset metadata should cross the preprocessing cache boundary as an opaque, stable scalar unless the schema is explicitly declared. Nested Arrow structs are brittle for sparse, user-provided CSV columns.
- **Related Constraint**: N/A

### Missing prompt values reach model text cleaners
- **Date**: 2026-06-01
- **Symptom**: Wan VACE preprocessing failed inside `ftfy.fix_text` with `TypeError: object of type 'NoneType' has no len()`.
- **Root Cause**: The configured prompt column contained missing or whitespace-only values, and the dataset layer passed those invalid prompt rows to model-specific text encoders.
- **Fix**: `data_utils/dataset.py` now filters missing or empty prompts immediately after prompt-column normalization, logs the number of dropped rows plus sample indices, and invalidates the preprocessing cache via the output schema version.
- **Lesson**: Required text-conditioning columns should be validated at the dataset boundary. Model adapters should receive well-typed text batches, not raw CSV nulls.
- **Related Constraint**: N/A

### DeepSpeed prepare needs explicit micro batch size
- **Date**: 2026-06-01
- **Symptom**: Trainer initialization failed in `accelerator.prepare()` with `ValueError: When using DeepSpeed, accelerate.prepare() requires you to pass at least one ... dataloaders with batch_size ... or set train_micro_batch_size_per_gpu`.
- **Root Cause**: Flow-Factory intentionally does not pass the custom training dataloader into `accelerator.prepare()` (constraint #9), but the DeepSpeed config omitted `train_micro_batch_size_per_gpu`, so Accelerate could not infer it.
- **Fix**: `BaseTrainer` now writes `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']` from `train.per_device_batch_size` when the DeepSpeed config leaves it unset or `auto`.
- **Lesson**: When bypassing dataloader preparation for sampler correctness, DeepSpeed launch configs must provide the micro batch size explicitly or the trainer must inject it before `accelerator.prepare()`.
- **Related Constraint**: #9

### Media-only OPD context should not require prompt template text slot
- **Date**: 2026-06-01
- **Symptom**: OPD teacher construction could fail with `teacher.prompt_template must contain {context}` for configs using only media teacher context such as `first_frame_path`.
- **Root Cause**: `OPDContextBuilder` treated all context keys as prompt-text context even though media keys are consumed by adapter-side conditioning and intentionally skipped in `build_prompt`.
- **Fix**: `OPDContextBuilder` now requires `{context}` only when configured context keys include non-media text keys; pure media context can use `prompt_template: "{prompt}"`.
- **Lesson**: Context validation must match consumption paths. Media context and text context have different integration surfaces in OPD.
- **Related Constraint**: N/A

### Wan VACE LoRA wrappers hide transformer config
- **Date**: 2026-06-01
- **Symptom**: Wan VACE rollout failed with `AttributeError: 'dict' object has no attribute 'vace_layers'` while normalizing `control_hidden_states_scale`.
- **Root Cause**: After LoRA and distributed preparation, the trainable transformer can be wrapped by PEFT/DeepSpeed objects whose `.config` is wrapper metadata rather than the underlying `WanVACETransformer3DModel` config.
- **Fix**: `Wan2_VACE_Adapter` now reads transformer config values through common wrapper layers (`unwrap_model`, `module`, `base_model`, `model`, `get_base_model`) and supports both dict-style and attribute-style configs.
- **Lesson**: Adapter code that needs architectural config should not assume the trainable module object still exposes the raw diffusers model config after PEFT/distributed wrapping.
- **Related Constraint**: #20

### OPD velocity matching missed Flow-OPD time weighting
- **Date**: 2026-06-03
- **Symptom**: OPD trajectory supervision matched student and teacher velocity with a uniform per-step MSE, while Flow-OPD derives a time-weighted velocity MSE from the continuous reverse-KL objective under SDE rollout.
- **Root Cause**: The trainer preserved the correct on-policy student trajectory replay, but it dropped the KL-derived `w(t)` factor and the example configs used deterministic ODE rollout.
- **Fix**: `OPDTrainer` now defaults to `opd_loss_weighting='flow_opd'`, validates that this path uses Flow-SDE trajectory rollout, applies the Flow-OPD time weight to teacher velocity matching and optional v-space KL, and the OPD examples now configure Flow-SDE with all stochastic steps.
- **Lesson**: For Flow-OPD-style distillation, the dense target is not just "same state velocity MSE"; the relative weighting of trajectory steps is part of the objective derived from the SDE transition KL. ODE/unweighted OPD should remain an explicit legacy choice.
- **Related Constraint**: N/A

### Flow-OPD auto trajectory indices include final zero-variance step
- **Date**: 2026-06-08
- **Symptom**: OPD training with `examples/opd/lora/wan2_vace/pexels_first_frame_context.yaml` failed before sampling with `Invalid OPD trajectory step indices: [23]`.
- **Root Cause**: Automatic trajectory-step derivation used the closed `timestep_range` scheduler index span unchanged, so the default 24-step Flow-OPD config could select the final denoising step even though the default Flow-SDE scheduler only injects positive variance on non-final transitions.
- **Fix**: `OPDTrainer._resolve_trajectory_step_indices()` now caps automatically derived Flow-OPD step indices at `num_inference_steps - 2`, while explicit `opd_trajectory_indices` continue to fail fast if they request the final step. OPD docs and examples now document the non-final requirement.
- **Lesson**: Automatic config derivation must encode objective-specific boundary conditions. A validation that is correct for explicit user input can make a default config unrunnable if the auto path is not constrained by the same scheduler semantics.
- **Related Constraint**: N/A

### Launcher misdetects stale rank environment as external launch
- **Date**: 2026-06-24
- **Symptom**: Running `CUDA_VISIBLE_DEVICES=4,5,6,7 ff-train ... --num_processes 4` logged `Direct launch` and `World Size: 1`, so only one GPU was used.
- **Root Cause**: `cli.py` treated the mere presence of `RANK` as proof that an external launcher was already active, even when `WORLD_SIZE` was unset or `1`.
- **Fix**: `cli.py` now treats an external distributed launcher as active only when `WORLD_SIZE > 1`, logs the loaded CLI module path, and removes stale worker-rank variables before spawning `accelerate launch`.
- **Lesson**: Scheduler or shell environments may inject rank-like variables without spawning worker processes. Launcher detection should key off actual world size, not only rank presence.
- **Related Constraint**: #18

## Cross-refs

- `constraints.md` (archival target for constraint violations)
- `architecture.md` (archival target for data-flow misunderstandings)
- `ff-debug/SKILL.md` Phase 5 (knowledge capture workflow)
