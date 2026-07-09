# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/trainers/opd.py
from __future__ import annotations

import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm as tqdm_

from diffusers.utils.torch_utils import randn_tensor

from ..hparams import OPDTrainingArguments
from ..samples import BaseSample
from ..teachers import load_opd_teacher
from ..teachers.context import OPDContextBuilder
from ..utils.base import (
    create_generator,
    create_generator_by_prompt,
    filter_kwargs,
    to_broadcast_tensor,
)
from ..utils.dist import reduce_loss_info
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import (
    TimeSampler,
    flow_match_sigma,
    fraction_range_to_t_bounds,
)
from .abc import BaseTrainer

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
logger = setup_logger(__name__)


class OPDTrainer(BaseTrainer):
    """Step-level OPD trainer with frozen teacher velocity matching."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: OPDTrainingArguments
        self.time_sampling_strategy = self.training_args.time_sampling_strategy
        self.time_shift = self.training_args.time_shift
        self.num_train_timesteps = self.training_args.num_train_timesteps
        self.timestep_range = self.training_args.timestep_range
        self._validate_loss_weighting_config()
        self.teacher = load_opd_teacher(self.config, self.accelerator)
        self._validate_teacher_blend_config()
        self._offload_student_vae()

    @property
    def enable_reward_weighting(self) -> bool:
        """Return whether scalar rewards should modulate OPD loss."""
        return self.training_args.opd_reward_weight > 0.0

    @property
    def enable_kl_loss(self) -> bool:
        """Return whether student reference KL is enabled."""
        return self.training_args.opd_kl_beta > 0.0

    @property
    def use_trajectory_timesteps(self) -> bool:
        """Return whether OPD should supervise stored student rollout states."""
        return self.training_args.opd_timestep_mode == "trajectory"

    @property
    def use_teacher_student_latent_blend(self) -> bool:
        """Return whether teacher supervision should use blended teacher/student latents."""
        return self.training_args.opd_blend_teacher_student_latents

    def _load_student_vae(self) -> None:
        """Load the student VAE for rollout/evaluation stages that need it."""
        if "vae" in self.adapter._resolve_component_names(None):
            self.adapter.on_load_vae(self.accelerator.device)

    def _offload_student_vae(self) -> None:
        """Offload the student VAE outside rollout/evaluation stages."""
        if "vae" in self.adapter._resolve_component_names(None):
            self.adapter.off_load_vae()

    def _validate_teacher_blend_config(self) -> None:
        """Validate optional teacher-first-step latent blending requirements."""
        if not self.use_teacher_student_latent_blend:
            return
        if not self.use_trajectory_timesteps:
            raise ValueError(
                "`opd_blend_teacher_student_latents=true` requires "
                "`opd_timestep_mode='trajectory'`."
            )

    def _validate_loss_weighting_config(self) -> None:
        """Validate Flow-OPD loss weighting requirements."""
        if self.training_args.opd_loss_weighting != "flow_opd":
            return
        if not self.use_trajectory_timesteps:
            raise ValueError(
                "`opd_loss_weighting='flow_opd'` requires `opd_timestep_mode='trajectory'` "
                "so the loss is evaluated on stored student rollout states."
            )
        if self.config.scheduler_args.dynamics_type != "Flow-SDE":
            raise ValueError(
                "`opd_loss_weighting='flow_opd'` requires `scheduler.dynamics_type='Flow-SDE'` "
                "to match Flow-OPD's SDE on-policy objective. Set "
                "`opd_loss_weighting='uniform'` for legacy ODE/unweighted OPD."
            )
        if self.config.scheduler_args.noise_level <= 0:
            raise ValueError(
                "`opd_loss_weighting='flow_opd'` requires a positive `scheduler.noise_level`."
            )
        expected_sde_steps = list(range(max(0, self.training_args.num_inference_steps - 1)))
        raw_sde_steps = self.config.scheduler_args.sde_steps
        configured_sde_steps = (
            expected_sde_steps if raw_sde_steps is None else [int(step) for step in raw_sde_steps]
        )
        if sorted(configured_sde_steps) != expected_sde_steps:
            raise ValueError(
                "`opd_loss_weighting='flow_opd'` requires `scheduler.sde_steps` to cover "
                "every denoising transition except the final zero-variance transition. Set "
                "`scheduler.sde_steps: null` for the Flow-OPD default, or set "
                "`opd_loss_weighting='uniform'` for a custom sparse-SDE legacy objective."
            )
        num_sde_steps = self.config.scheduler_args.num_sde_steps
        if num_sde_steps is None:
            num_sde_steps = len(configured_sde_steps)
        if num_sde_steps < len(expected_sde_steps):
            raise ValueError(
                "`opd_loss_weighting='flow_opd'` requires `scheduler.num_sde_steps` to select "
                "all configured SDE steps on every rollout. Set `scheduler.num_sde_steps: null` "
                "for the Flow-OPD default, or set `opd_loss_weighting='uniform'` for randomized "
                "sparse-SDE legacy OPD."
            )

    def _validate_flow_opd_supervised_steps(self, step_indices: torch.Tensor) -> None:
        """Validate that Flow-OPD supervision uses positive-variance SDE transitions."""
        if self.training_args.opd_loss_weighting != "flow_opd":
            return
        max_sde_step = self._get_max_flow_opd_supervised_step(
            self.training_args.num_inference_steps
        )
        invalid = step_indices[step_indices > max_sde_step]
        if invalid.numel() > 0:
            invalid_steps = invalid.detach().cpu().tolist()
            raise ValueError(
                "`opd_loss_weighting='flow_opd'` cannot supervise the final denoising step "
                "because the default Flow-SDE scheduler has no positive transition variance "
                f"there. Invalid OPD trajectory step indices: {invalid_steps}. Reduce "
                "`timestep_range`, set explicit non-final `opd_trajectory_indices`, or use "
                "`opd_loss_weighting='uniform'` for legacy unweighted OPD."
            )

    def _get_max_flow_opd_supervised_step(self, num_inference_steps: int) -> int:
        """Return the last denoising step with positive Flow-SDE transition variance."""
        return num_inference_steps - 2

    def _compute_flow_opd_timestep_weight(
        self,
        timestep: torch.Tensor,
        timestep_next: torch.Tensor,
    ) -> torch.Tensor:
        """Return Flow-OPD's per-sample time weight for velocity matching."""
        sigma = flow_match_sigma(timestep.view(-1)).float()
        sigma_next = flow_match_sigma(timestep_next.view(-1)).float()
        delta_sigma = (sigma - sigma_next).clamp_min(1e-6)

        scheduler_sigmas = getattr(self.adapter.scheduler, "sigmas", None)
        if scheduler_sigmas is None or len(scheduler_sigmas) < 2:
            raise ValueError(
                "Flow-OPD timestep weighting requires scheduler sigmas to be initialized."
            )
        sigma_max = scheduler_sigmas[1].to(device=sigma.device, dtype=sigma.dtype)
        denominator_sigma = torch.where(
            torch.isclose(sigma, torch.ones_like(sigma)),
            sigma_max.expand_as(sigma),
            sigma,
        )

        noise_level = torch.as_tensor(
            self.config.scheduler_args.noise_level,
            device=sigma.device,
            dtype=sigma.dtype,
        )
        sigma_safe = sigma.clamp_min(1e-6)
        sde_std = torch.sqrt(sigma_safe / (1.0 - denominator_sigma).clamp_min(1e-6)) * noise_level
        drift_coeff = 1.0 + sde_std.pow(2) * (1.0 - sigma) / (2.0 * sigma_safe)
        return delta_sigma * drift_coeff.pow(2) / (2.0 * sde_std.pow(2).clamp_min(1e-6))

    def _get_opd_timestep_weight(
        self,
        timestep: torch.Tensor,
        timestep_next: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return optional per-sample OPD loss weights."""
        if self.training_args.opd_loss_weighting == "uniform":
            return None
        if self.training_args.opd_loss_weighting == "flow_opd":
            return self._compute_flow_opd_timestep_weight(timestep, timestep_next)
        raise ValueError(f"Unknown OPD loss weighting: {self.training_args.opd_loss_weighting}.")

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample OPD training timesteps in scheduler scale [0, 1000]."""
        device = self.accelerator.device
        strategy = self.time_sampling_strategy.lower()
        available = [
            "logit_normal",
            "uniform",
            "discrete",
            "discrete_with_init",
            "discrete_wo_init",
        ]

        if strategy == "logit_normal":
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                stratified=True,
            )
        if strategy == "uniform":
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
            )
        if strategy.startswith("discrete"):
            discrete_config = {
                "discrete": (True, False),
                "discrete_with_init": (True, True),
                "discrete_wo_init": (False, False),
            }
            if strategy not in discrete_config:
                raise ValueError(
                    f"Unknown time_sampling_strategy: {strategy}. Available: {available}"
                )
            include_init, force_init = discrete_config[strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_range=self.timestep_range,
                include_init=include_init,
                force_init=force_init,
            )
        raise ValueError(f"Unknown time_sampling_strategy: {strategy}. Available: {available}")

    def _select_evenly_spaced_step_indices(
        self,
        start_idx: int,
        end_idx: int,
        num_indices: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Select unique denoising step indices from a closed interval."""
        if start_idx > end_idx:
            raise ValueError(
                "No valid OPD trajectory steps remain after applying timestep settings: "
                f"start_idx({start_idx}) > end_idx({end_idx})."
            )
        available = end_idx - start_idx + 1
        if num_indices > available:
            raise ValueError(
                "`num_train_timesteps` exceeds available stored trajectory steps: "
                f"num_train_timesteps({num_indices}) > available_steps({available}). "
                "Increase `timestep_range`, reduce `num_train_timesteps`, or set explicit "
                "`opd_trajectory_indices`."
            )
        if num_indices == 1:
            return torch.tensor([start_idx], dtype=torch.long, device=device)

        indices = torch.linspace(start_idx, end_idx, num_indices, device=device).round().long()
        if torch.unique_consecutive(indices).numel() != num_indices:
            raise ValueError(
                "Failed to build unique OPD trajectory step indices from "
                f"[{start_idx}, {end_idx}] with num_train_timesteps({num_indices})."
            )
        return indices

    def _normalize_explicit_trajectory_step_indices(
        self,
        num_inference_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize configured OPD trajectory step indices to [0, num_inference_steps)."""
        raw_indices = self.training_args.opd_trajectory_indices
        if raw_indices is None:
            raise ValueError("Expected explicit `opd_trajectory_indices`, got None.")
        if not raw_indices:
            raise ValueError("`opd_trajectory_indices` must not be empty.")

        normalized = []
        for idx in raw_indices:
            step_idx = num_inference_steps + idx if idx < 0 else idx
            if not 0 <= step_idx < num_inference_steps:
                raise ValueError(
                    "`opd_trajectory_indices` entries are denoising step indices and must "
                    f"resolve into [0, {num_inference_steps}); got {idx} -> {step_idx}."
                )
            normalized.append(step_idx)
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"`opd_trajectory_indices` must resolve to unique step indices, got {raw_indices}."
            )
        return torch.tensor(normalized, dtype=torch.long, device=device)

    def _resolve_trajectory_step_indices(self) -> torch.Tensor:
        """Resolve denoising step indices whose stored student latents will be supervised."""
        device = self.accelerator.device
        num_inference_steps = self.training_args.num_inference_steps
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}.")
        if self.training_args.opd_trajectory_indices is not None:
            return self._normalize_explicit_trajectory_step_indices(
                num_inference_steps=num_inference_steps,
                device=device,
            )

        strategy = self.time_sampling_strategy.lower()
        discrete_config = {
            "discrete": (True, False),
            "discrete_with_init": (True, True),
            "discrete_wo_init": (False, False),
        }
        if strategy not in discrete_config:
            raise ValueError(
                "`opd_timestep_mode='trajectory'` requires discrete timestep selection, "
                f"got time_sampling_strategy={strategy!r}."
            )

        self.adapter.scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler_timesteps = self.adapter.scheduler.timesteps.float()
        t_min, t_max = fraction_range_to_t_bounds(*self.timestep_range)
        valid_mask = (scheduler_timesteps >= t_min - 1e-3) & (scheduler_timesteps <= t_max + 1e-3)
        valid_indices = torch.where(valid_mask)[0]
        if valid_indices.numel() == 0:
            raise ValueError(
                "`timestep_range` selects no scheduler timesteps for OPD trajectory mode: "
                f"timestep_range={self.timestep_range}, bounds=({t_min}, {t_max}), "
                f"scheduler_timesteps={scheduler_timesteps.detach().cpu().tolist()}."
            )

        min_idx = int(valid_indices.min().item())
        max_idx = int(valid_indices.max().item())
        if self.training_args.opd_loss_weighting == "flow_opd":
            max_flow_opd_step = self._get_max_flow_opd_supervised_step(num_inference_steps)
            max_idx = min(max_idx, max_flow_opd_step)
            if min_idx > max_idx:
                raise ValueError(
                    "`timestep_range` selects no positive-variance Flow-OPD trajectory steps: "
                    f"timestep_range={self.timestep_range}, bounds=({t_min}, {t_max}), "
                    f"selected_index_range=({int(valid_indices.min().item())}, "
                    f"{int(valid_indices.max().item())}), "
                    f"max_flow_opd_step({max_flow_opd_step}). Lower the `timestep_range` "
                    "upper bound, set explicit non-final `opd_trajectory_indices`, or use "
                    "`opd_loss_weighting='uniform'` for legacy unweighted OPD."
                )
        include_init, force_init = discrete_config[strategy]
        if force_init:
            if self.num_train_timesteps == 1:
                return torch.tensor([min_idx], dtype=torch.long, device=device)
            rest = self._select_evenly_spaced_step_indices(
                start_idx=min_idx + 1,
                end_idx=max_idx,
                num_indices=self.num_train_timesteps - 1,
                device=device,
            )
            return torch.cat([torch.tensor([min_idx], dtype=torch.long, device=device), rest])

        start_idx = min_idx if include_init else min_idx + 1
        return self._select_evenly_spaced_step_indices(
            start_idx=start_idx,
            end_idx=max_idx,
            num_indices=self.num_train_timesteps,
            device=device,
        )

    def _attach_opd_step_indices(
        self,
        samples: List[BaseSample],
        step_indices: Optional[torch.Tensor],
    ) -> None:
        """Attach OPD trajectory step indices to samples for shuffled replay."""
        if step_indices is None:
            return
        stored_step_indices = step_indices.detach().cpu()
        for sample in samples:
            if sample.all_latents is None or sample.latent_index_map is None:
                raise ValueError(
                    "OPD trajectory mode requires stored student latents and a latent index map. "
                    "Check `trajectory_indices` passed to adapter.inference()."
                )
            sample.extra_kwargs["opd_step_indices"] = stored_step_indices

    def _attach_opd_context(self, samples: List[BaseSample], batch: Dict[str, Any]) -> None:
        """Attach serialized teacher-only context from dataloader metadata to samples."""
        metadata = batch.get("metadata")
        if metadata is None:
            metadata = [{} for _ in samples]
        if len(metadata) != len(samples):
            raise ValueError(
                "Metadata/sample batch mismatch: "
                f"{len(metadata)} metadata rows vs {len(samples)} samples."
            )

        for sample, meta in zip(samples, metadata):
            if meta is None:
                meta = {}
            meta = OPDContextBuilder.normalize_context(meta)
            context = meta.get("opd_context")
            if context is None:
                context = {
                    key: meta[key] for key in self.training_args.teacher_context_keys if key in meta
                }
            sample.extra_kwargs["opd_context"] = OPDContextBuilder.serialize_context(context)

    def _attach_teacher_first_step_latents(
        self,
        samples: List[BaseSample],
        generator: Optional[torch.Generator],
    ) -> None:
        """Run one teacher rollout step from the student's stored initial noise."""
        if not self.use_teacher_student_latent_blend:
            return
        if not samples:
            return

        prompts = [sample.prompt for sample in samples]
        if any(prompt is None for prompt in prompts):
            raise ValueError(
                "Teacher first-step latent blending requires every sample to carry a prompt."
            )
        contexts = [sample.extra_kwargs.get("opd_context", "{}") for sample in samples]
        student_initial_latents = []
        for sample in samples:
            if sample.all_latents is None:
                raise ValueError(
                    "Teacher first-step latent blending requires stored student trajectory "
                    "latents on every sample."
                )
            student_initial_latents.append(sample.all_latents[0])
        student_initial_latents_batch = torch.stack(student_initial_latents, dim=0)
        negative_prompts = [sample.negative_prompt for sample in samples]
        batch = {
            "prompt": prompts,
            "negative_prompt": negative_prompts if any(
                negative_prompt is not None for negative_prompt in negative_prompts
            ) else None,
        }
        with torch.no_grad():
            teacher_encoded = self.teacher.encode_prompt(
                prompts=prompts,
                contexts=contexts,
                negative_prompts=batch["negative_prompt"],
                generator=generator,
            )
            teacher_first_step_latents = self.teacher.rollout_first_step_latents(
                batch=batch,
                contexts=contexts,
                student_initial_latents=student_initial_latents_batch,
                generator=generator,
                encoded_prompt=teacher_encoded,
            ).detach().cpu()

        for sample, teacher_latents in zip(samples, teacher_first_step_latents):
            sample.extra_kwargs["teacher_first_step_latent"] = teacher_latents

    def evaluate(self) -> None:
        """Evaluate the student model with text prompt only."""
        if self.test_dataloader is None:
            return

        self._load_student_vae()
        try:
            self.adapter.eval()
            self.eval_reward_buffer.clear()

            with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
                all_samples: List[BaseSample] = []
                for batch in tqdm(
                    self.test_dataloader,
                    desc="Evaluating",
                    disable=not self.show_progress_bar,
                ):
                    generator = create_generator_by_prompt(batch["prompt"], self.training_args.seed)
                    inference_kwargs = {
                        "compute_log_prob": False,
                        "generator": generator,
                        "trajectory_indices": None,
                        **self.eval_args,
                    }
                    inference_kwargs.update(**batch)
                    inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                    samples = self.adapter.inference(**inference_kwargs)
                    all_samples.extend(samples)
                    self.eval_reward_buffer.add_samples(samples)

                rewards = self.eval_reward_buffer.finalize(store_to_samples=True, split="pointwise")
                rewards = {
                    key: torch.as_tensor(value).to(self.accelerator.device)
                    for key, value in rewards.items()
                }
                gathered_rewards = {
                    key: self.accelerator.gather(value).cpu().numpy() for key, value in rewards.items()
                }

                if self.accelerator.is_main_process:
                    log_data = {
                        f"eval/reward_{key}_mean": np.mean(value)
                        for key, value in gathered_rewards.items()
                    }
                    log_data.update(
                        {
                            f"eval/reward_{key}_std": np.std(value)
                            for key, value in gathered_rewards.items()
                        }
                    )
                    log_data["eval_samples"] = all_samples
                    self.log_data(log_data, step=self.step)
                self.accelerator.wait_for_everyone()
        finally:
            self._offload_student_vae()

    def start(self):
        """Run the OPD six-stage training loop."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)
            if hasattr(self.teacher.adapter.scheduler, "set_seed"):
                self.teacher.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    "checkpoints",
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()

            samples = self.sample()
            self.prepare_feedback(samples)
            self.optimize(samples)
            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

        if self.log_args.save_freq > 0 and self.log_args.save_dir and self.epoch > 0:
            save_dir = os.path.join(
                self.log_args.save_dir,
                str(self.log_args.run_name),
                "checkpoints",
            )
            self.save_checkpoint(save_dir, epoch=self.epoch)

    def sample(self) -> List[BaseSample]:
        """Generate student rollouts and keep OPD-required trajectory states."""
        self._load_student_vae()
        if self.use_teacher_student_latent_blend:
            self.teacher.on_load_runtime_components()
        try:
            self.adapter.rollout()
            self.reward_buffer.clear()
            samples = []
            data_iter = iter(self.dataloader)
            trajectory_step_indices = None
            if self.use_trajectory_timesteps:
                trajectory_step_indices = self._resolve_trajectory_step_indices()
                self._validate_flow_opd_supervised_steps(trajectory_step_indices)
                trajectory_indices = trajectory_step_indices.detach().cpu().tolist()
            else:
                trajectory_indices = [-1]

            with torch.no_grad(), self.autocast():
                for batch_idx in tqdm(
                    range(self.training_args.num_batches_per_epoch),
                    desc=f"Epoch {self.epoch} Sampling",
                    disable=not self.show_progress_bar,
                ):
                    batch = next(data_iter)
                    sample_kwargs = {
                        **self.training_args,
                        "compute_log_prob": False,
                        # Pure OPD does not consume decoded media; keep decode only when
                        # reward models need image/video tensors during prepare_feedback().
                        "decode_media": self.enable_reward_weighting,
                        "trajectory_indices": trajectory_indices,
                        **batch,
                    }
                    sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                    sample_batch = self.adapter.inference(**sample_kwargs)
                    self._attach_opd_context(sample_batch, batch)
                    teacher_first_step_gen = create_generator(
                        self.training_args.seed,
                        self.epoch,
                        batch_idx,
                    )
                    self._attach_teacher_first_step_latents(
                        sample_batch,
                        generator=teacher_first_step_gen,
                    )
                    self._attach_opd_step_indices(sample_batch, trajectory_step_indices)
                    self._maybe_offload_samples_to_cpu(sample_batch)
                    samples.extend(sample_batch)
                    self.reward_buffer.add_samples(sample_batch)
            return samples
        finally:
            if self.use_teacher_student_latent_blend:
                self.teacher.off_load_runtime_components()
            self._offload_student_vae()

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func=None,
    ) -> torch.Tensor:
        """Compute optional scalar reward advantages for OPD reward weighting."""
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Compute optional scalar rewards and advantages."""
        if not self.enable_reward_weighting:
            return
        if not self.reward_models:
            raise ValueError(
                "`opd_reward_weight` is positive, but no training reward model is configured."
            )

        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        self.compute_advantages(samples, rewards, store_to_samples=True)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self.log_data(adv_metrics, step=self.step)

    def _compute_student_output(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        latents: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return student velocity/noise prediction for one latent batch."""
        t_flat = timestep.view(-1)
        if t_next is None:
            t_next = torch.zeros_like(t_flat)
        else:
            t_next = t_next.view(-1)
        excluded = {
            "all_latents",
            "timesteps",
            "advantage",
            "rewards",
            "opd_context",
            "opd_step_indices",
            "teacher_first_step_latent",
            "callback_index_map",
        }
        forward_kwargs = {
            **self.training_args,
            "t": t_flat,
            "t_next": t_next,
            "latents": latents,
            "compute_log_prob": False,
            "return_kwargs": ["noise_pred"],
            "noise_level": 0.0,
            **{k: v for k, v in batch.items() if k not in excluded},
        }
        forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
        output = self.adapter.forward(**forward_kwargs)
        return output.noise_pred

    def _compute_opd_loss(
        self,
        student_v_pred: torch.Tensor,
        teacher_v_pred: torch.Tensor,
        noised_latents: torch.Tensor,
        timestep: torch.Tensor,
        timestep_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-sample OPD loss."""
        if student_v_pred.shape != teacher_v_pred.shape:
            raise ValueError(
                "Teacher/student velocity shape mismatch: "
                f"student={tuple(student_v_pred.shape)}, teacher={tuple(teacher_v_pred.shape)}. "
                "Use latent-compatible teacher/student models for OPD."
            )

        reduce_dims = tuple(range(1, student_v_pred.ndim))
        if self.training_args.opd_loss_type == "velocity_mse":
            per_sample_loss = F.mse_loss(
                student_v_pred.float(),
                teacher_v_pred.float(),
                reduction="none",
            ).mean(dim=reduce_dims)
            if timestep_weight is not None:
                per_sample_loss = per_sample_loss * timestep_weight.to(per_sample_loss.device)
            return per_sample_loss
        if self.training_args.opd_loss_type == "x0_mse":
            sigma = to_broadcast_tensor(flow_match_sigma(timestep.view(-1)), noised_latents)
            student_x0 = noised_latents.float() - sigma.float() * student_v_pred.float()
            teacher_x0 = noised_latents.float() - sigma.float() * teacher_v_pred.float()
            return F.mse_loss(student_x0, teacher_x0, reduction="none").mean(dim=reduce_dims)
        raise ValueError(f"Unknown OPD loss type: {self.training_args.opd_loss_type}.")

    def _apply_reward_weighting(
        self,
        per_sample_loss: torch.Tensor,
        batch: Dict[str, Any],
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> torch.Tensor:
        """Apply optional advantage-based weighting to per-sample OPD loss."""
        if not self.enable_reward_weighting:
            return per_sample_loss
        if "advantage" not in batch:
            raise ValueError("OPD reward weighting is enabled, but batch has no `advantage` field.")

        adv = batch["advantage"]
        adv_clip_range = self.training_args.adv_clip_range
        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
        weights = torch.clamp(1.0 + self.training_args.opd_reward_weight * adv, min=0.0)
        loss_info["reward_weight"].append(weights.detach())
        return per_sample_loss * weights

    def _get_teacher_first_step_latents(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Return the stored teacher first-step latents for one stacked batch."""
        if "teacher_first_step_latent" not in batch:
            raise ValueError(
                "Teacher/student latent blending requires `teacher_first_step_latent` stored "
                "on every sample."
            )
        teacher_latents = batch["teacher_first_step_latent"]
        if not isinstance(teacher_latents, torch.Tensor):
            teacher_latents = torch.as_tensor(teacher_latents)
        return teacher_latents.to(device=device)

    def _blend_teacher_and_student_latents(
        self,
        student_latents: torch.Tensor,
        teacher_first_step_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Blend teacher first-step and student rollout latents for teacher supervision."""
        if student_latents.shape != teacher_first_step_latents.shape:
            raise ValueError(
                "Teacher/student latent blend requires matching shapes, got "
                f"student={tuple(student_latents.shape)} and "
                f"teacher={tuple(teacher_first_step_latents.shape)}."
            )

        teacher_weight, student_weight = self._resolve_teacher_student_blend_weights(
            device=student_latents.device,
            dtype=student_latents.dtype,
        )
        teacher_latents = teacher_first_step_latents.to(
            device=student_latents.device,
            dtype=student_latents.dtype,
        )
        return teacher_weight * teacher_latents + student_weight * student_latents

    def _resolve_teacher_student_blend_weights(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the effective teacher/student latent blend weights."""
        teacher_weight = torch.as_tensor(
            self.training_args.opd_teacher_blend_weight,
            device=device,
            dtype=dtype,
        )
        student_weight = torch.as_tensor(
            self.training_args.opd_student_blend_weight,
            device=device,
            dtype=dtype,
        )
        return teacher_weight, student_weight

    def _get_batch_trajectory_step_indices(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Return the shared OPD trajectory step indices for a stacked batch."""
        if "opd_step_indices" not in batch:
            raise ValueError(
                "OPD trajectory mode requires `opd_step_indices` stored on every sample."
            )
        step_indices = batch["opd_step_indices"]
        if not isinstance(step_indices, torch.Tensor):
            step_indices = torch.as_tensor(step_indices)
        step_indices = step_indices.to(device=device, dtype=torch.long)

        if step_indices.ndim == 2:
            reference = step_indices[0]
            if not torch.equal(step_indices, reference.unsqueeze(0).expand_as(step_indices)):
                raise ValueError(
                    "OPD trajectory step indices differ inside a training batch. "
                    "All samples in an epoch must share the same stored trajectory layout."
                )
            return reference
        if step_indices.ndim == 1:
            return step_indices
        raise ValueError(
            "`opd_step_indices` must have shape (num_steps,) or (batch, num_steps), "
            f"got {tuple(step_indices.shape)}."
        )

    def _get_trajectory_step_inputs(
        self,
        batch: Dict[str, Any],
        step_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch rollout latents and scheduler times for one stored denoising step."""
        latent_index_map = batch["latent_index_map"].to(device=device)
        compact_idx = int(latent_index_map[step_idx].item())
        if compact_idx < 0:
            latent_index_map_values = latent_index_map.detach().cpu().tolist()
            raise ValueError(
                "Requested OPD trajectory step was not stored: "
                f"step_idx({step_idx}), latent_index_map={latent_index_map_values}."
            )

        timesteps = batch["timesteps"].to(device=device)
        if timesteps.ndim == 1:
            t_flat = timesteps[step_idx].expand(batch_size)
            if step_idx + 1 < timesteps.shape[0]:
                t_next = timesteps[step_idx + 1].expand(batch_size)
            else:
                t_next = torch.zeros_like(t_flat)
        else:
            t_flat = timesteps[:, step_idx]
            if step_idx + 1 < timesteps.shape[1]:
                t_next = timesteps[:, step_idx + 1]
            else:
                t_next = torch.zeros_like(t_flat)

        latents = batch["all_latents"][:, compact_idx]
        return latents, t_flat, t_next

    def optimize(self, samples: List[BaseSample]) -> None:
        """Optimize student with teacher velocity matching on OPD timesteps."""
        self.teacher.on_load_runtime_components()
        try:
            device = self.accelerator.device
            per_device_batch_size = self.training_args.per_device_batch_size
            num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size

            for inner_epoch in range(self.training_args.num_inner_epochs):
                perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
                perm = torch.randperm(len(samples), generator=perm_gen)
                shuffled_samples = [samples[i] for i in perm]
                loss_info = defaultdict(list)

                for batch_idx in tqdm(
                    range(num_batches),
                    total=num_batches,
                    desc=f"Epoch {self.epoch} Training",
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    start = batch_idx * per_device_batch_size
                    batch_samples = [
                        sample.to(device)
                        for sample in shuffled_samples[start : start + per_device_batch_size]
                    ]
                    batch = BaseSample.stack(batch_samples)
                    batch_size = batch["all_latents"].shape[0]
                    contexts = batch.get("opd_context", ["{}"] * batch_size)
                    teacher_context_gen = create_generator(
                        self.training_args.seed,
                        self.epoch,
                        inner_epoch,
                        batch_idx,
                    )
                    with torch.no_grad():
                        teacher_encoded = self.teacher.encode_prompt(
                            prompts=batch["prompt"],
                            contexts=contexts,
                            negative_prompts=batch.get("negative_prompt"),
                            generator=teacher_context_gen,
                        )

                    self.adapter.train()
                    if self.use_trajectory_timesteps:
                        trajectory_step_indices = self._get_batch_trajectory_step_indices(batch, device)
                        num_timestep_updates = int(trajectory_step_indices.numel())
                        teacher_first_step_latents = (
                            self._get_teacher_first_step_latents(batch, device)
                            if self.use_teacher_student_latent_blend
                            else None
                        )
                        all_timesteps = None
                        clean_latents = None
                    else:
                        trajectory_step_indices = None
                        teacher_first_step_latents = None
                        num_timestep_updates = self.num_train_timesteps
                        all_timesteps = self._sample_timesteps(batch_size)
                        clean_latents = batch["all_latents"][:, -1]

                    media_reference_latents = (
                        batch["all_latents"][:, 0] if self.use_trajectory_timesteps else clean_latents
                    )
                    with torch.no_grad():
                        teacher_media_kwargs = self.teacher.prepare_media_forward_kwargs(
                            contexts=contexts,
                            latents=media_reference_latents,
                            generator=teacher_context_gen,
                        )

                    with self.autocast():
                        for t_idx in tqdm(
                            range(num_timestep_updates),
                            desc=f"Epoch {self.epoch} Timestep",
                            position=1,
                            leave=False,
                            disable=not self.show_progress_bar,
                        ):
                            with self.accelerator.accumulate(*self.adapter.trainable_components):
                                if self.use_trajectory_timesteps:
                                    step_idx = int(trajectory_step_indices[t_idx].item())
                                    noised_latents, t_flat, t_next = self._get_trajectory_step_inputs(
                                        batch=batch,
                                        step_idx=step_idx,
                                        batch_size=batch_size,
                                        device=device,
                                    )
                                else:
                                    t_flat = all_timesteps[t_idx]
                                    t_next = torch.zeros_like(t_flat)
                                    sigma = to_broadcast_tensor(flow_match_sigma(t_flat), clean_latents)
                                    noise = randn_tensor(
                                        clean_latents.shape,
                                        device=clean_latents.device,
                                        dtype=clean_latents.dtype,
                                    )
                                    noised_latents = (1 - sigma) * clean_latents + sigma * noise

                                student_v_pred = self._compute_student_output(
                                    batch=batch,
                                    timestep=t_flat,
                                    latents=noised_latents,
                                    t_next=t_next,
                                )
                                teacher_input_latents = noised_latents
                                if teacher_first_step_latents is not None:
                                    teacher_input_latents = self._blend_teacher_and_student_latents(
                                        student_latents=noised_latents,
                                        teacher_first_step_latents=teacher_first_step_latents,
                                    )
                                with torch.no_grad():
                                    teacher_v_pred = self.teacher.forward_velocity(
                                        batch=batch,
                                        contexts=contexts,
                                        latents=teacher_input_latents,
                                        timestep=t_flat,
                                        t_next=t_next,
                                        encoded_prompt=teacher_encoded,
                                        media_forward_kwargs=teacher_media_kwargs,
                                    )

                                timestep_weight = self._get_opd_timestep_weight(t_flat, t_next)
                                if timestep_weight is not None:
                                    loss_info["opd_time_weight"].append(timestep_weight.detach())
                                per_sample_loss = self._compute_opd_loss(
                                    student_v_pred=student_v_pred,
                                    teacher_v_pred=teacher_v_pred.detach(),
                                    noised_latents=noised_latents,
                                    timestep=t_flat,
                                    timestep_weight=timestep_weight,
                                )
                                weighted_loss = self._apply_reward_weighting(
                                    per_sample_loss=per_sample_loss,
                                    batch=batch,
                                    loss_info=loss_info,
                                )
                                opd_loss = self.training_args.opd_teacher_weight * weighted_loss.mean()
                                loss = opd_loss

                                if self.enable_kl_loss:
                                    with torch.no_grad(), self.adapter.use_ref_parameters():
                                        ref_v_pred = self._compute_student_output(
                                            batch=batch,
                                            timestep=t_flat,
                                            latents=noised_latents,
                                            t_next=t_next,
                                        )
                                    kl_div = F.mse_loss(
                                        student_v_pred.float(),
                                        ref_v_pred.float(),
                                        reduction="none",
                                    ).mean(dim=tuple(range(1, student_v_pred.ndim)))
                                    if timestep_weight is not None:
                                        kl_div = kl_div * timestep_weight.to(kl_div.device)
                                    kl_loss = self.training_args.opd_kl_beta * kl_div.mean()
                                    loss = loss + kl_loss
                                    loss_info["kl_div"].append(kl_div.detach())
                                    loss_info["kl_loss"].append(kl_loss.detach())

                                loss_info["opd_loss"].append(opd_loss.detach())
                                loss_info["unweighted_opd_loss"].append(per_sample_loss.detach())
                                loss_info["loss"].append(loss.detach())

                                self.accelerator.backward(loss)
                                if self.accelerator.sync_gradients:
                                    grad_norm = self.accelerator.clip_grad_norm_(
                                        self.adapter.get_trainable_parameters(),
                                        self.training_args.max_grad_norm,
                                    )
                                    self.optimizer.step()
                                    self.optimizer.zero_grad()
                                    loss_info = reduce_loss_info(self.accelerator, loss_info)
                                    loss_info["grad_norm"] = grad_norm
                                    self.log_data(
                                        {f"train/{k}": v for k, v in loss_info.items()},
                                        step=self.step,
                                    )
                                    self.step += 1
                                    loss_info = defaultdict(list)
        finally:
            self.teacher.off_load_runtime_components()
