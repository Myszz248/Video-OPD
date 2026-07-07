# Flow-Factory Agent Guide

## Project Identity

Flow-Factory is an online RL fine-tuning framework for diffusion and flow-matching models. The central invariant is **train-inference consistency**: rollout-time `adapter.inference()` and optimize-time `adapter.forward()` must agree whenever they receive the same latent state, timestep, conditioning, scheduler config, precision boundary, and model weights.

Use this file as the root instruction sheet for future training changes. Detailed rules live in `.agents/knowledge/`; this file tells agents what to read, which code paths matter, and how to avoid breaking the training stack.

## Required Context

At the start of any coding session, read:

- `.agents/knowledge/philosophy.md`
- `.agents/knowledge/constraints.md`
- `.agents/knowledge/architecture.md`
- `.agents/knowledge/README.md`

For training or inference changes, also read:

- `.agents/knowledge/topics/train_inference_consistency.md`
- `.agents/knowledge/topics/dtype_precision.md` when touching mixed precision, latent dtype, autocast, scheduler replay, or NaN/overflow behavior.
- `.agents/knowledge/topics/timestep_sigma.md` when touching timesteps, sigmas, `TimeSampler`, `scheduler.step()`, `timestep_range`, or `flow_match_sigma`.
- `.agents/knowledge/topics/samplers.md` when touching sampler choice, group size, batch geometry, gradient accumulation, async rewards, or distributed data flow.
- `.agents/knowledge/topics/sample_lifecycle.md` when touching `sample()`, `optimize()`, stored trajectories, CPU offload, high-resolution image, or video training.
- `.agents/knowledge/topics/adapter_conventions.md` when touching model adapters.

Use skills when the request matches them:

- `/ff-develop` for feature work and refactors.
- `/ff-debug` for errors, crashes, hangs, NaNs, OOMs, or wrong training behavior.
- `/ff-review` before committing multi-file or shared-infrastructure changes.
- `/ff-new-model`, `/ff-new-reward`, `/ff-new-algorithm` for new components.

## Local Execution Rules

This checkout is for static code and documentation work. Do not run local training, inference, tests, dependency installation, or model-loading commands unless the user explicitly says the local environment is ready.

Allowed static checks, preferably through the repo `.venv` when present:

- Python syntax checks that do not import/load model weights.
- YAML/config parsing that does not load models.
- `black --check src/`
- `isort --check src/`

All temporary or intermediate files must go under `.scratch/`. Do not place analysis notes, checklists, generated reports, or scratch scripts in the project root or tracked directories.

## Training And Inference Code Map

Primary entry points:

- `src/flow_factory/train.py`: CLI entry, loads `Arguments`, logs distributed setup, calls `load_trainer(config)`, then `trainer.start()`.
- `src/flow_factory/inference.py`: standalone inference CLI, loads or builds `Arguments`, constructs the adapter directly, loads preprocessing and inference modules, freezes components, calls `adapter.inference(compute_log_prob=False)`, and saves videos.
- `src/flow_factory/trainers/loader.py`: creates `Accelerator`, sets seed, loads adapter via `load_model`, resolves trainer registry, constructs the trainer.

Shared training skeleton:

- `src/flow_factory/trainers/abc.py`: `BaseTrainer` owns dataloader setup, optimizer setup, accelerator preparation, reward model loading, reward buffers, `AdvantageProcessor`, component loading, FSDP frozen-component synchronization, checkpoint save/load, and async reward cleanup.
- Only trainable modules plus optimizer are passed to `accelerator.prepare()`. The training dataloader is intentionally not prepared because custom samplers already handle distributed layout.
- Per-epoch trainer order is fixed: `sample()` -> `prepare_feedback()` -> `optimize()`; evaluation and checkpointing happen around that loop.

Model adapter skeleton:

- `src/flow_factory/models/abc.py`: `BaseAdapter` loads a diffusers pipeline, installs Flow-Factory scheduler, handles checkpoint load/save, LoRA/full fine-tuning, freezing, mixed precision, component offload/onload, EMA/reference parameters, and the adapter contract.
- Required abstract methods are exactly `load_pipeline()`, `decode_latents()`, `forward()`, and `inference()`.
- Optional encoders are `encode_prompt()`, `encode_image()`, `encode_video()`, and `encode_audio()`. They default to no-op `None`; override only the modalities a model actually consumes.
- `preprocess_func()` dispatches to the optional encoders and requires each active encoder to return a non-empty dict whose values are lists, tensors, or NumPy arrays.

Scheduler path:

- `src/flow_factory/scheduler/abc.py`: `SDESchedulerOutput` is the cross-adapter output contract for `forward()` and `scheduler.step()`.
- `src/flow_factory/scheduler/flow_match_euler_discrete.py` and `src/flow_factory/scheduler/unipc_multistep.py`: SDE/ODE step logic, train/rollout/eval mode, SDE step selection, noise levels, and log-prob replay.
- Coupled algorithms need stored rollout trajectories and matching scheduler replay. Decoupled algorithms usually keep final latents and resample training timesteps.

Reward and advantage path:

- `src/flow_factory/rewards/reward_processor.py`: pointwise/groupwise reward dispatch, media conversion, async reward workers, groupwise local vs distributed reward computation, and `RewardBuffer`.
- `src/flow_factory/advantage/advantage_processor.py`: communication-aware advantage computation. `group_contiguous` computes advantages locally with reduced metrics; `distributed_k_repeat` gathers rewards and unique IDs before scattering local advantages.

Data and sample path:

- `src/flow_factory/data_utils/dataset.py`: raw dataset loading, preprocessing, cache fingerprinting, modality column normalization, and collation.
- `src/flow_factory/data_utils/sampler.py`: distributed K-repeat samplers and group topology contracts.
- `src/flow_factory/samples/samples.py`: `BaseSample` dataclass, `unique_id`, tensor movement, stacking, shared fields, and task-level sample types.

Config path:

- `src/flow_factory/hparams/args.py`: top-level `Arguments`, nested config loading, sampler resolution, batch-geometry alignment, gradient-accumulation adjustment, and SDE defaults.
- `src/flow_factory/hparams/training_args.py`: shared and algorithm-specific training args.
- `src/flow_factory/hparams/model_args.py`, `data_args.py`, `scheduler_args.py`, `reward_args.py`, `teacher_args.py`, `log_args.py`: component-specific config groups.

## Six-Stage Pipeline

Never reorder or skip these stages:

1. Data preprocessing: `GeneralDataset` calls `adapter.preprocess_func()` and caches encoded fields.
2. K-repeat sampling: custom sampler repeats prompts by `group_size`.
3. Trajectory generation: `adapter.inference()` runs full denoising and stores required sample fields.
4. Reward computation: `RewardBuffer.finalize()` calls pointwise/groupwise rewards.
5. Advantage computation: trainer delegates to `AdvantageProcessor.compute_advantages()`.
6. Policy optimization: trainer replays or constructs training inputs and calls `adapter.forward()`.

Trainer method mapping:

- `sample()` covers stages 2-3.
- `prepare_feedback()` covers stages 4-5.
- `optimize()` covers stage 6. DPO forms chosen/rejected pairs at the start of `optimize()`, after advantages are stored.

## Algorithm Paths

Coupled path:

- `src/flow_factory/trainers/grpo.py`: GRPO uses SDE rollout, stores selected trajectories and old log-probs, then replays each train timestep in `optimize()`. The initial ratio `exp(new_log_prob - old_log_prob)` should be close to 1 before policy updates.
- `GRPOGuardTrainer` extends GRPO and additionally stores `next_latents_mean` for ratio normalization.
- Coupled trainers must not use ODE dynamics.

Decoupled path:

- `src/flow_factory/trainers/dpo.py`: rollout keeps final latents, rewards determine chosen/rejected pairs, optimization samples training timesteps and compares current vs reference velocity errors.
- `src/flow_factory/trainers/nft.py`: rollout keeps final latents; optimize samples timesteps, precomputes old velocity predictions under the sampling policy, then trains current policy.
- `src/flow_factory/trainers/awm.py`: rollout keeps final latents; optimize samples timesteps, precomputes old weighted log-probs, then applies PPO-style clipped matching loss.
- `src/flow_factory/trainers/dgpo.py`: uses `GroupDistributedSampler`, global micro-batches must be group-complete, and group-level loss relies on strict sampler topology.
- `src/flow_factory/trainers/opd.py`: student rollout stores final or selected trajectory states, teacher context is attached to samples, and optimize performs teacher/student velocity matching.
- Decoupled trainers may use ODE or SDE dynamics depending on algorithm semantics.

## Change Protocol For Training Work

Before editing:

- Identify whether the change touches trainer, adapter, scheduler, reward, advantage, samples, config, or distributed behavior.
- Search the codebase with `rg` before changing a contract.
- Read the relevant base class before concrete subclasses.
- List affected subclasses and call sites.

When editing shared contracts:

- `BaseTrainer` changes affect all trainers.
- `BaseAdapter` changes affect every adapter and standalone inference.
- `SDESchedulerOutput` or scheduler step changes affect rollout replay and all adapter `forward()` implementations.
- `BaseSample` or `_shared_fields` changes affect collation, reward processing, advantage grouping, CPU offload, and trainer optimize loops.
- `RewardProcessor`, `RewardBuffer`, or `AdvantageProcessor` changes affect every algorithm.
- Config field additions, removals, or renames must update dataclasses, all code accesses, and relevant `examples/` YAML files.

Implementation rules:

- Keep new trainers inheriting directly from `BaseTrainer`; `GRPOGuardTrainer -> GRPOTrainer` is the only sanctioned exception.
- Keep adapters inheriting directly from `BaseAdapter`; share adapter-family logic through helpers or mixins, not adapter-to-adapter inheritance.
- Preserve `adapter.forward()` as the atomic train-inference unit.
- Preserve stored rollout fields needed by optimize: latents, next latents, timesteps, log-probs, callback fields, prompt/conditioning embeddings, and trajectory index maps.
- Preserve component lifecycle: preprocessing modules are loaded for dataset encoding and offloaded before training; inference modules are loaded for runtime; prepared trainable modules are not manually moved.
- Preserve precision boundaries: trainable params use `model.master_weight_dtype`, frozen params/buffers use inference dtype, and latent storage goes through `adapter.cast_latents()`.
- Preserve distributed barriers around preprocessing, reward/eval/checkpoint boundaries, and any collective save/load path.
- Do not introduce silent fallbacks for invalid training states; raise with concrete values and context.

## Sampler And Distributed Invariants

Sampler choice is resolved in `Arguments.__post_init__()` before trainers consume config:

- `group_contiguous`: all K copies of a prompt group stay on the same rank. This enables local groupwise rewards and local advantage grouping, but requires `unique_sample_num_per_epoch % world_size == 0` and local batch tiling.
- `distributed_k_repeat`: K copies are globally shuffled across ranks. This has fewer geometry constraints but requires gather/scatter for groupwise rewards and advantages.
- `group_distributed`: used by DGPO. Every rank sees the same prompt sequence, each rank owns `group_size / world_size` copies, and each global micro-batch is group-complete.

Do not pass the training dataloader to `accelerator.prepare()`. If sampler geometry or gradient accumulation changes, update `hparams/args.py`, `data_utils/sampler.py`, examples, and `.agents/knowledge/topics/samplers.md` together.

DeepSpeed ZeRO-3 is unsupported for reward model sharding. Treat ZeRO-1/2 and FSDP paths carefully; only trainable modules and optimizer should be prepared.

## Inference Compatibility Checklist

Any change to adapters, checkpoints, precision, component names, model args, or `inference()` must be checked against both:

- Training rollout/eval: trainer calls `adapter.inference()` with batches, `compute_log_prob`, and `trajectory_indices`.
- Standalone CLI: `src/flow_factory/inference.py` constructs the adapter directly, loads preprocessing and inference modules, freezes components, filters kwargs, and calls `adapter.inference(compute_log_prob=False)`.

Standalone inference rejects `resume_type="state"` and expects model-only LoRA/full checkpoints. If checkpoint layout changes, update both adapter checkpoint code and `inference.py`.

## Verification Expectations

Use static verification locally unless the user explicitly enables the runnable environment:

- Syntax-check touched Python files without importing heavy model code.
- Run `black --check src/` and `isort --check src/` before committing when feasible.
- For config changes, parse representative YAML only if it does not load models.
- For docs-only changes, inspect rendered Markdown structure and `git diff`.

When a remote runnable environment is available, verify at least:

- One coupled path, usually GRPO.
- One decoupled path, usually NFT or AWM.
- At least two model adapters if adapter or scheduler contracts changed.
- Distributed behavior for the sampler path touched.
- Standalone inference if adapter `inference()`, checkpoint loading, or component lifecycle changed.

## Documentation And Commit Flow

Update documentation whenever behavior changes:

- New or changed API: update `guidance/`.
- New or changed config field: update all relevant `examples/` YAML files and comments.
- Architecture or hard invariant change: update `.agents/knowledge/architecture.md` or `.agents/knowledge/constraints.md`.
- New training pitfall or repeatable fix pattern: update the appropriate `.agents/knowledge/topics/` leaf.

Before commit:

- Run `/ff-review` for multi-file or shared-infrastructure changes.
- Run `black --check src/` and `isort --check src/` when local static checks are permitted.
- Commit messages must be concise and English.
- Do not batch unrelated fixes.
