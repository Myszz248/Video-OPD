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

# src/flow_factory/teachers/opd_teacher.py
from __future__ import annotations

import inspect
import os
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from PIL import Image

from .context import MEDIA_CONTEXT_KEYS, OPDContextBuilder
from ..hparams import DataArguments, OPDTrainingArguments, TeacherArguments
from ..models.abc import BaseAdapter
from ..samples import BaseSample, T2VSample
from ..utils.base import filter_kwargs


class OPDTeacher:
    """Frozen teacher wrapper for OPD step-level velocity targets."""

    _SUPPORTED_REQUIRED_FORWARD_ARGS = {
        "self",
        "t",
        "latents",
        "prompt_embeds",
    }

    def __init__(
        self,
        adapter: BaseAdapter,
        teacher_args: TeacherArguments,
        training_args: OPDTrainingArguments,
        data_args: DataArguments,
        accelerator: Accelerator,
    ):
        self.adapter = adapter
        self.teacher_args = teacher_args
        self.training_args = training_args
        self.data_args = data_args
        self.accelerator = accelerator
        self.context_builder = OPDContextBuilder(
            context_keys=training_args.teacher_context_keys,
            prompt_template=teacher_args.prompt_template,
            context_dropout=training_args.teacher_context_dropout,
        )

    def prepare(self) -> None:
        """Freeze teacher parameters, move runtime components to device, and validate forward."""
        for component_name in self.adapter._resolve_component_names(None):
            component = self.adapter.get_component(component_name)
            if component is not None and hasattr(component, "requires_grad_"):
                component.requires_grad_(False)
                component.eval()

        self.adapter.on_load_components(
            components=self.teacher_args.runtime_components,
            device=self.teacher_args.device,
        )
        self.adapter.eval()
        self._validate_forward_signature()
        self._validate_media_context_support()

    def _validate_forward_signature(self) -> None:
        """Fail fast when a teacher forward requires unsupported non-text inputs."""
        signature = inspect.signature(self.adapter.forward)
        required = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        unsupported = sorted(required - self._SUPPORTED_REQUIRED_FORWARD_ARGS)
        if unsupported:
            raise ValueError(
                "This OPD MVP supports text-conditioned teacher forwards only. "
                f"Teacher adapter {type(self.adapter).__name__}.forward requires unsupported "
                f"arguments: {unsupported}. Use a text-to-video teacher such as `wan2_t2v`, "
                "or add a teacher context encoder for those required inputs."
            )

    def _configured_media_context_keys(self) -> List[str]:
        """Return configured teacher context keys that represent media inputs."""
        return [key for key in self.training_args.teacher_context_keys if key in MEDIA_CONTEXT_KEYS]

    def _validate_media_context_support(self) -> None:
        """Fail fast when media context is configured but the teacher cannot consume it."""
        media_keys = self._configured_media_context_keys()
        if not media_keys:
            return

        if not self._supports_vace_context() and not self._supports_i2v_context():
            raise ValueError(
                "Teacher media context keys were configured, but the selected teacher adapter "
                f"{type(self.adapter).__name__} cannot consume media context. "
                f"Configured media keys: {media_keys}. Use `wan2_i2v`, `wan2_vace`, or add an "
                "adapter-specific OPD teacher context preparation method."
            )
        if "vae" not in self.adapter._resolve_component_names(self.teacher_args.runtime_components):
            raise ValueError(
                "Teacher media context requires `vae` in `teacher.runtime_components` so first-frame "
                "conditions can be encoded on the teacher device."
            )

    def _supports_vace_context(self) -> bool:
        """Return whether the teacher adapter exposes VACE control preparation."""
        return hasattr(self.adapter, "prepare_teacher_conditioning")

    def _supports_i2v_context(self) -> bool:
        """Return whether the teacher adapter can prepare Wan I2V first-frame conditions."""
        signature = inspect.signature(self.adapter.forward)
        return (
            "condition" in signature.parameters
            and hasattr(self.adapter, "prepare_latents")
            and hasattr(self.adapter.pipeline, "video_processor")
        )

    def _context_image_base_dir(self) -> str:
        """Return the base directory used for relative OPD context image paths."""
        if self.data_args.image_dir is not None:
            return os.path.expanduser(self.data_args.image_dir)
        data_root = os.path.expanduser(self.data_args.dataset_dir)
        if os.path.isfile(data_root):
            return os.path.dirname(data_root)
        return data_root

    def _resolve_context_image_path(self, value: Any) -> str:
        """Resolve one context image path from string or small metadata dict."""
        if isinstance(value, dict):
            for key in ("path", "image", "first_frame_path", "file"):
                if key in value:
                    value = value[key]
                    break
        if not isinstance(value, str) or not value:
            raise ValueError(
                "OPD media context values must be non-empty image paths or dicts containing "
                f"a path-like field, got {type(value)}."
            )
        path = os.path.expanduser(value)
        if os.path.isabs(path):
            return path
        return os.path.join(self._context_image_base_dir(), path)

    def _load_context_image(self, context: Any) -> Optional[Image.Image]:
        """Load the first configured first-frame image from one OPD context."""
        context_dict = OPDContextBuilder.normalize_context(context)
        image_keys = (
            "high_aesthetic_first_frame_path",
            "high_aesthetic_first_frame",
            "first_frame_path",
            "first_frame",
        )
        for key in image_keys:
            if key not in context_dict or context_dict[key] in (None, ""):
                continue
            path = self._resolve_context_image_path(context_dict[key])
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Configured OPD teacher context image does not exist: {path} "
                    f"(context key: {key})."
                )
            return Image.open(path).convert("RGB")
        return None

    def _latent_geometry(self, latents: torch.Tensor) -> Dict[str, int]:
        """Infer pixel-space video geometry from latent shape and teacher VAE scale factors."""
        if latents.ndim != 5:
            raise ValueError(f"Expected video latents with shape (B, C, T, H, W), got {latents.shape}.")
        temporal_scale = getattr(self.adapter.pipeline, "vae_scale_factor_temporal", 4)
        spatial_scale = getattr(self.adapter.pipeline, "vae_scale_factor_spatial", 8)
        return {
            "batch_size": latents.shape[0],
            "num_channels_latents": latents.shape[1],
            "num_frames": (latents.shape[2] - 1) * temporal_scale + 1,
            "height": latents.shape[3] * spatial_scale,
            "width": latents.shape[4] * spatial_scale,
        }

    def _prepare_i2v_media_kwargs(
        self,
        images: List[Image.Image],
        latents: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> Dict[str, torch.Tensor]:
        """Prepare Wan I2V adapter kwargs from first-frame teacher context."""
        geometry = self._latent_geometry(latents)
        image_tensor = self.adapter.pipeline.video_processor.preprocess(
            images,
            height=geometry["height"],
            width=geometry["width"],
        ).to(latents.device, dtype=torch.float32)
        latent_outputs = self.adapter.prepare_latents(
            image=image_tensor,
            batch_size=geometry["batch_size"],
            num_channels_latents=geometry["num_channels_latents"],
            height=geometry["height"],
            width=geometry["width"],
            num_frames=geometry["num_frames"],
            dtype=latents.dtype,
            device=latents.device,
            generator=generator,
            latents=latents,
        )
        media_kwargs: Dict[str, torch.Tensor]
        if getattr(self.adapter.pipeline.config, "expand_timesteps", False):
            _, condition, first_frame_mask = latent_outputs
            media_kwargs = {
                "condition": condition,
                "first_frame_mask": first_frame_mask,
            }
        else:
            _, condition = latent_outputs
            media_kwargs = {"condition": condition}

        transformer = self.adapter.pipeline.transformer
        if transformer is not None and getattr(transformer.config, "image_dim", None) is not None:
            image_encoded = self.adapter.encode_image(images, device=latents.device)
            media_kwargs["image_embeds"] = image_encoded["image_embeds"].to(latents.device)
        return media_kwargs

    def prepare_media_forward_kwargs(
        self,
        contexts: List[Any],
        latents: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> Dict[str, Any]:
        """Build adapter-specific teacher forward kwargs from media OPD context."""
        if not self._configured_media_context_keys():
            return {}

        images = [self._load_context_image(context) for context in contexts]
        if any(image is None for image in images):
            raise ValueError(
                "Teacher media context is enabled, but at least one sample has no first-frame "
                "image under first_frame_path/first_frame/high_aesthetic_first_frame_path."
            )
        loaded_images = [image for image in images if image is not None]

        if self._supports_vace_context():
            geometry = self._latent_geometry(latents)
            return self.adapter.prepare_teacher_conditioning(
                images=loaded_images,
                latents=latents,
                height=geometry["height"],
                width=geometry["width"],
                num_frames=geometry["num_frames"],
                generator=generator,
                conditioning_scale=self.teacher_args.extra_kwargs.get("conditioning_scale", 1.0),
            )
        return self._prepare_i2v_media_kwargs(
            images=loaded_images,
            latents=latents,
            generator=generator,
        )

    def encode_prompt(
        self,
        prompts: List[str],
        contexts: List[Any],
        negative_prompts: Optional[List[Optional[str]]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode teacher prompts after appending teacher-only context."""
        teacher_prompts = self.context_builder.build_prompts(
            prompts=prompts,
            contexts=contexts,
            generator=generator,
        )
        if self.teacher_args.negative_prompt is not None:
            negative_prompt_input = [self.teacher_args.negative_prompt] * len(teacher_prompts)
        else:
            negative_prompt_input = negative_prompts

        encode_kwargs = {
            "prompt": teacher_prompts,
            "negative_prompt": negative_prompt_input,
            "guidance_scale": self.training_args.teacher_guidance_scale,
            "device": self.teacher_args.device,
        }
        encode_kwargs = filter_kwargs(self.adapter.encode_prompt, **encode_kwargs)
        return self.adapter.encode_prompt(**encode_kwargs)

    def _prepare_rollout_initial_latents(
        self,
        reference_latents: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Prepare one batch of teacher rollout latents matching a reference latent shape."""
        if reference_latents.ndim != 5:
            raise ValueError(
                "Teacher first-step rollout expects video latents with shape (B, C, T, H, W), "
                f"got {tuple(reference_latents.shape)}."
            )
        if not hasattr(self.adapter.pipeline, "prepare_latents"):
            raise ValueError(
                f"Teacher adapter {type(self.adapter).__name__} has no `pipeline.prepare_latents()` "
                "method required for first-step latent rollout."
            )

        geometry = self._latent_geometry(reference_latents)
        prepare_latent_kwargs = {
            "batch_size": geometry["batch_size"],
            "num_channels_latents": geometry["num_channels_latents"],
            "height": geometry["height"],
            "width": geometry["width"],
            "num_frames": geometry["num_frames"],
            "dtype": torch.float32,
            "device": self.teacher_args.device,
            "generator": generator,
        }
        prepare_latent_kwargs = filter_kwargs(
            self.adapter.pipeline.prepare_latents,
            **prepare_latent_kwargs,
        )
        initial_latents = self.adapter.pipeline.prepare_latents(**prepare_latent_kwargs)
        return self.adapter.cast_latents(initial_latents.to(self.teacher_args.device))

    def _forward_output(
        self,
        batch: Dict[str, Any],
        contexts: List[Any],
        latents: torch.Tensor,
        timestep: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        encoded_prompt: Optional[Dict[str, torch.Tensor]] = None,
        media_forward_kwargs: Optional[Dict[str, Any]] = None,
        return_kwargs: Optional[List[str]] = None,
        noise_level: float = 0.0,
    ) -> Any:
        """Run one frozen-teacher forward step and return the raw adapter output."""
        prompts = batch["prompt"]
        negative_prompts = batch.get("negative_prompt")
        if negative_prompts is not None and not isinstance(negative_prompts, list):
            negative_prompts = [negative_prompts] * len(prompts)

        encoded = encoded_prompt
        if encoded is None:
            encoded = self.encode_prompt(
                prompts=prompts,
                contexts=contexts,
                negative_prompts=negative_prompts,
                generator=generator,
            )
        prompt_embeds = encoded["prompt_embeds"].to(device=latents.device)
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=latents.device)

        t_flat = timestep.view(-1)
        if t_next is None:
            t_next = torch.zeros_like(t_flat)
        else:
            t_next = t_next.view(-1)
        forward_kwargs = {
            "t": t_flat,
            "t_next": t_next,
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "guidance_scale": self.training_args.teacher_guidance_scale,
            "compute_log_prob": False,
            "return_kwargs": ["noise_pred"] if return_kwargs is None else return_kwargs,
            "noise_level": noise_level,
        }
        if media_forward_kwargs is None:
            media_forward_kwargs = self.prepare_media_forward_kwargs(
                contexts=contexts,
                latents=latents,
                generator=generator,
            )
        forward_kwargs.update(media_forward_kwargs)
        forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
        return self.adapter.forward(**forward_kwargs)

    def forward_velocity(
        self,
        batch: Dict[str, Any],
        contexts: List[Any],
        latents: torch.Tensor,
        timestep: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        encoded_prompt: Optional[Dict[str, torch.Tensor]] = None,
        media_forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Return teacher velocity/noise prediction for the provided noised latents."""
        output = self._forward_output(
            batch=batch,
            contexts=contexts,
            latents=latents,
            timestep=timestep,
            t_next=t_next,
            generator=generator,
            encoded_prompt=encoded_prompt,
            media_forward_kwargs=media_forward_kwargs,
            return_kwargs=["noise_pred"],
            noise_level=0.0,
        )
        return output.noise_pred

    def rollout_first_step_latents(
        self,
        batch: Dict[str, Any],
        contexts: List[Any],
        reference_latents: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        encoded_prompt: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run exactly one teacher denoising step and return the resulting latent batch."""
        num_inference_steps = self.training_args.num_inference_steps
        if num_inference_steps < 2:
            raise ValueError(
                "Teacher first-step rollout requires `num_inference_steps >= 2`, got "
                f"{num_inference_steps}."
            )

        device = self.teacher_args.device
        self.adapter.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.adapter.scheduler.timesteps
        initial_latents = self._prepare_rollout_initial_latents(
            reference_latents=reference_latents,
            generator=generator,
        )
        media_forward_kwargs = self.prepare_media_forward_kwargs(
            contexts=contexts,
            latents=initial_latents,
            generator=generator,
        )
        timestep = timesteps[0].expand(initial_latents.shape[0])
        timestep_next = timesteps[1].expand(initial_latents.shape[0])
        noise_level = self.adapter.scheduler.get_noise_level_for_timestep(timesteps[0])
        output = self._forward_output(
            batch=batch,
            contexts=contexts,
            latents=initial_latents,
            timestep=timestep,
            t_next=timestep_next,
            generator=generator,
            encoded_prompt=encoded_prompt,
            media_forward_kwargs=media_forward_kwargs,
            return_kwargs=["next_latents"],
            noise_level=noise_level,
        )
        return self.adapter.cast_latents(output.next_latents).detach()

    def _normalize_resume_step_index(self, step_idx: int, num_inference_steps: int) -> int:
        """Normalize one teacher-resume step index to [0, num_inference_steps)."""
        if step_idx < 0:
            step_idx += num_inference_steps
        if not 0 <= step_idx < num_inference_steps:
            raise ValueError(
                "Teacher resume step index is out of range: "
                f"step_idx({step_idx}), num_inference_steps({num_inference_steps})."
            )
        return step_idx

    def _validate_resume_timesteps(self, timesteps: torch.Tensor) -> None:
        """Ensure teacher scheduler timesteps match the stored student rollout."""
        self.adapter.scheduler.set_timesteps(int(timesteps.shape[0]), device=timesteps.device)
        teacher_timesteps = self.adapter.scheduler.timesteps.to(device=timesteps.device)
        if teacher_timesteps.shape != timesteps.shape or not torch.equal(
            teacher_timesteps, timesteps
        ):
            raise ValueError(
                "Teacher resume expects the teacher scheduler timesteps to match the stored "
                "student rollout timesteps exactly. "
                f"teacher_timesteps={teacher_timesteps.detach().cpu().tolist()}, "
                f"student_timesteps={timesteps.detach().cpu().tolist()}."
            )

    def resume_from_student_samples(
        self,
        samples: List[BaseSample],
        step_idx: int,
    ) -> List[T2VSample]:
        """Resume teacher denoising from stored student states and decode debug videos."""
        if not samples:
            return []

        self.adapter.eval()
        resumed_samples: List[T2VSample] = []

        with torch.no_grad():
            for sample in samples:
                if sample.prompt is None:
                    raise ValueError("Teacher resume debug requires every sample to have `prompt`.")
                if sample.timesteps is None or sample.all_latents is None or sample.latent_index_map is None:
                    raise ValueError(
                        "Teacher resume debug requires `timesteps`, `all_latents`, and "
                        "`latent_index_map` on every sample."
                    )

                timesteps = sample.timesteps.to(self.teacher_args.device)
                if timesteps.ndim != 1:
                    raise ValueError(
                        "Teacher resume debug expects per-sample 1D `timesteps`, got "
                        f"{tuple(timesteps.shape)}."
                    )
                resume_step_idx = self._normalize_resume_step_index(
                    step_idx=step_idx,
                    num_inference_steps=int(timesteps.shape[0]),
                )
                latent_index_map = sample.latent_index_map.to(device=self.teacher_args.device)
                compact_idx = int(latent_index_map[resume_step_idx].item())
                if compact_idx < 0:
                    raise ValueError(
                        "Requested teacher resume step was not stored in the student trajectory: "
                        f"step_idx({resume_step_idx}), "
                        f"latent_index_map={latent_index_map.detach().cpu().tolist()}."
                    )

                current_latents = sample.all_latents[compact_idx].unsqueeze(0).to(
                    self.teacher_args.device
                )
                context = sample.extra_kwargs.get("opd_context", "{}")
                prompts = [sample.prompt]
                negative_prompts = None
                if sample.negative_prompt is not None:
                    negative_prompts = [sample.negative_prompt]

                teacher_prompts = self.context_builder.build_prompts(
                    prompts=prompts,
                    contexts=[context],
                )
                encoded_prompt = self.encode_prompt(
                    prompts=prompts,
                    contexts=[context],
                    negative_prompts=negative_prompts,
                )
                media_forward_kwargs = self.prepare_media_forward_kwargs(
                    contexts=[context],
                    latents=current_latents,
                    generator=None,
                )
                self._validate_resume_timesteps(timesteps)

                for idx in range(resume_step_idx, int(timesteps.shape[0])):
                    t = timesteps[idx].view(1)
                    t_next = (
                        timesteps[idx + 1].view(1)
                        if idx + 1 < timesteps.shape[0]
                        else torch.zeros_like(t)
                    )
                    negative_prompt_embeds = encoded_prompt.get("negative_prompt_embeds")
                    if negative_prompt_embeds is not None:
                        negative_prompt_embeds = negative_prompt_embeds.to(
                            device=current_latents.device
                        )
                    forward_kwargs = {
                        "t": t,
                        "t_next": t_next,
                        "latents": current_latents,
                        "prompt_embeds": encoded_prompt["prompt_embeds"].to(
                            device=current_latents.device
                        ),
                        "negative_prompt_embeds": negative_prompt_embeds,
                        "guidance_scale": self.training_args.teacher_guidance_scale,
                        "compute_log_prob": False,
                        "return_kwargs": ["next_latents"],
                        "noise_level": self.adapter.scheduler.get_noise_level_for_timestep(t),
                        **media_forward_kwargs,
                    }
                    forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
                    output = self.adapter.forward(**forward_kwargs)
                    current_latents = self.adapter.cast_latents(output.next_latents)

                decoded_video = self.adapter.decode_latents(current_latents, output_type="pt")[0]
                resumed_samples.append(
                    T2VSample(
                        video=decoded_video,
                        height=sample.height,
                        width=sample.width,
                        prompt=f"[teacher resume step {resume_step_idx}] {sample.prompt}",
                        negative_prompt=sample.negative_prompt,
                        extra_kwargs={
                            "teacher_prompt": teacher_prompts[0],
                            "resume_step_idx": resume_step_idx,
                            "resume_timestep": float(timesteps[resume_step_idx].item()),
                        },
                    )
                )

        return resumed_samples
