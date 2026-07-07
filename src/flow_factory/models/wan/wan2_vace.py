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

# src/flow_factory/models/wan/wan2_vace.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

import torch
from accelerate import Accelerator
from peft import PeftModel
from PIL import Image

from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.pipelines.wan.pipeline_wan_vace import WanVACEPipeline

from ...hparams import Arguments
from ...samples import T2VSample
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
)
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import (
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter

logger = setup_logger(__name__)


@dataclass
class WanVACESample(T2VSample):
    """Text-to-video sample carrying VACE control latents for replay."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset({"control_hidden_states_scale"})

    control_hidden_states: Optional[torch.FloatTensor] = None
    control_hidden_states_scale: Optional[torch.FloatTensor] = None


class Wan2_VACE_Adapter(BaseAdapter):
    """Adapter for Wan VACE text/video/image-conditioned generation."""

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: WanVACEPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

    def load_pipeline(self) -> WanVACEPipeline:
        """Load the Wan VACE diffusers pipeline."""
        return WanVACEPipeline.from_pretrained(self.model_args.model_name_or_path)

    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = ["transformer", "transformer_2"],
        **kwargs,
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        """Apply LoRA to Wan VACE transformer components."""
        return super().apply_lora(target_modules=target_modules, components=components, **kwargs)

    @property
    def default_target_modules(self) -> List[str]:
        """Return default LoRA module patterns for Wan VACE transformers."""
        return [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            "ffn.net.0.proj",
            "ffn.net.2",
            "vace_blocks.0.proj_in",
            "vace_blocks.0.proj_out",
        ]

    @property
    def inference_modules(self) -> List[str]:
        """Return modules needed for VACE rollout and OPD replay."""
        if self.pipeline.config.boundary_ratio is None or self.pipeline.config.boundary_ratio <= 0:
            return ["transformer", "vae"]
        if self.pipeline.config.boundary_ratio >= 1:
            return ["transformer_2", "vae"]
        return ["transformer", "transformer_2", "vae"]

    @property
    def transformer_2(self) -> torch.nn.Module:
        """Return the optional low-noise Wan VACE transformer."""
        return self.get_component("transformer_2")

    @transformer_2.setter
    def transformer_2(self, module: torch.nn.Module) -> None:
        self.set_component("transformer_2", module)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Encode prompts with Wan's T5 encoder."""
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(text) for text in prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        mask = text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.pipeline.text_encoder(
            text_input_ids.to(device),
            mask.to(device),
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [embeds[:seq_len] for embeds, seq_len in zip(prompt_embeds, seq_lens)]
        return torch.stack(
            [
                torch.cat(
                    [
                        embeds,
                        embeds.new_zeros(max_sequence_length - embeds.size(0), embeds.size(1)),
                    ],
                )
                for embeds in prompt_embeds
            ],
            dim=0,
        )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 5.0,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode prompt and optional negative prompt for VACE CFG."""
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        results = {"prompt_embeds": prompt_embeds}

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                [negative_prompt] * len(prompt)
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            if len(negative_prompt) != len(prompt):
                raise ValueError(
                    "`negative_prompt` batch size must match `prompt` batch size, "
                    f"got {len(negative_prompt)} vs {len(prompt)}."
                )
            results["negative_prompt_embeds"] = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return results

    def encode_image(self, images: Any) -> None:
        """Return no image preprocessing output for VACE datasets."""
        return None

    def encode_video(self, videos: Any) -> None:
        """Return no video preprocessing output for VACE datasets."""
        return None

    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: Literal["pt", "pil", "np"] = "pil",
    ) -> torch.Tensor:
        """Decode Wan VACE latents into videos."""
        vae_dtype = self.pipeline.vae.dtype
        latents = latents.to(device=self.pipeline.vae.device, dtype=vae_dtype)
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(
            1,
            self.pipeline.vae.config.z_dim,
            1,
            1,
            1,
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.pipeline.vae.decode(latents, return_dict=False)[0]
        return self.pipeline.video_processor.postprocess_video(video, output_type=output_type)

    @staticmethod
    def _config_value(config: Any, key: str) -> Any:
        """Read one config value from dict-like or attribute-style configs."""
        if isinstance(config, dict):
            return config.get(key)
        return getattr(config, key, None)

    def _transformer_config_value(self, transformer: torch.nn.Module, key: str) -> Any:
        """Read a Wan transformer config value through PEFT/accelerate wrappers."""
        candidates = [transformer, self._unwrap(transformer)]
        seen = set()
        idx = 0
        while idx < len(candidates):
            candidate = candidates[idx]
            idx += 1
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            value = self._config_value(getattr(candidate, "config", None), key)
            if value is not None:
                return value
            if hasattr(candidate, "get_base_model"):
                candidates.append(candidate.get_base_model())
            if hasattr(candidate, "module"):
                candidates.append(candidate.module)
            if hasattr(candidate, "base_model"):
                candidates.append(candidate.base_model)
            if hasattr(candidate, "model"):
                candidates.append(candidate.model)
        return None

    def _normalize_conditioning_scale(
        self,
        conditioning_scale: Union[float, List[float], torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
        transformer: torch.nn.Module,
    ) -> torch.Tensor:
        """Normalize VACE conditioning scale to one tensor per control layer."""
        vace_layers = self._transformer_config_value(transformer, "vace_layers")
        if vace_layers is None:
            raise ValueError(
                "Could not read `vace_layers` from the Wan VACE transformer config. "
                f"Transformer type: {type(transformer).__name__}. This usually means the "
                "trainable transformer wrapper does not expose the underlying Wan config."
            )
        if isinstance(conditioning_scale, (int, float)):
            conditioning_scale = [float(conditioning_scale)] * len(vace_layers)
        if isinstance(conditioning_scale, list):
            if len(conditioning_scale) != len(vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {len(conditioning_scale)} does not match "
                    f"number of VACE layers {len(vace_layers)}."
                )
            conditioning_scale = torch.tensor(conditioning_scale)
        if conditioning_scale.ndim != 1 or conditioning_scale.size(0) != len(vace_layers):
            raise ValueError(
                "`conditioning_scale` must have shape (num_vace_layers,), got "
                f"{tuple(conditioning_scale.shape)}."
            )
        return conditioning_scale.to(device=device, dtype=dtype)

    def _first_frame_video_and_mask(
        self,
        image: Image.Image,
        height: int,
        width: int,
        num_frames: int,
    ) -> tuple[List[Image.Image], List[Image.Image]]:
        """Create VACE video/mask inputs that keep only the first frame fixed."""
        image = image.convert("RGB").resize((width, height))
        blank = Image.new("RGB", (width, height), (0, 0, 0))
        mask_keep = Image.new("L", (width, height), 0)
        mask_generate = Image.new("L", (width, height), 255)
        video = [image] + [blank.copy() for _ in range(num_frames - 1)]
        mask = [mask_keep] + [mask_generate.copy() for _ in range(num_frames - 1)]
        return video, mask

    def _prepare_conditioning_batch(
        self,
        videos: List[Optional[List[Image.Image]]],
        masks: List[Optional[List[Image.Image]]],
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        """Prepare VACE control hidden states for a batch."""
        control_batches = []
        for idx, (video, mask) in enumerate(zip(videos, masks)):
            sample_generator = generator[idx] if isinstance(generator, list) else generator
            video_tensor, mask_tensor, reference_images = self.pipeline.preprocess_conditions(
                video=video,
                mask=mask,
                reference_images=None,
                batch_size=1,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
            )
            conditioning_latents = self.pipeline.prepare_video_latents(
                video_tensor,
                mask_tensor,
                reference_images,
                sample_generator,
                device,
            )
            prepared_mask = self.pipeline.prepare_masks(
                mask_tensor,
                reference_images,
                sample_generator,
            )
            control_batches.append(torch.cat([conditioning_latents, prepared_mask], dim=1)[0])
        return torch.stack(control_batches, dim=0).to(device=device, dtype=dtype)

    def prepare_empty_conditioning(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        conditioning_scale: Union[float, List[float], torch.Tensor] = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Prepare text-only VACE control kwargs with no visible conditioning."""
        transformer = self.pipeline.transformer or self.pipeline.transformer_2
        control_hidden_states = self._prepare_conditioning_batch(
            videos=[None] * batch_size,
            masks=[None] * batch_size,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        control_hidden_states_scale = self._normalize_conditioning_scale(
            conditioning_scale=conditioning_scale,
            dtype=dtype,
            device=device,
            transformer=transformer,
        )
        return {
            "control_hidden_states": control_hidden_states,
            "control_hidden_states_scale": control_hidden_states_scale,
        }

    def prepare_teacher_conditioning(
        self,
        images: List[Image.Image],
        latents: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        conditioning_scale: Union[float, List[float], torch.Tensor] = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Prepare VACE first-frame teacher conditioning for OPD."""
        videos_and_masks = [
            self._first_frame_video_and_mask(
                image, height=height, width=width, num_frames=num_frames
            )
            for image in images
        ]
        videos = [item[0] for item in videos_and_masks]
        masks = [item[1] for item in videos_and_masks]
        transformer = self.pipeline.transformer or self.pipeline.transformer_2
        control_hidden_states = self._prepare_conditioning_batch(
            videos=videos,
            masks=masks,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=transformer.dtype,
            device=latents.device,
            generator=generator,
        )
        control_hidden_states_scale = self._normalize_conditioning_scale(
            conditioning_scale=conditioning_scale,
            dtype=transformer.dtype,
            device=latents.device,
            transformer=transformer,
        )
        return {
            "control_hidden_states": control_hidden_states,
            "control_hidden_states_scale": control_hidden_states_scale,
        }

    @torch.no_grad()
    def inference(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        control_hidden_states: Optional[torch.Tensor] = None,
        control_hidden_states_scale: Union[float, List[float], torch.Tensor] = 1.0,
        compute_log_prob: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
        decode_media: bool = True,
    ) -> List[WanVACESample]:
        """Generate VACE text-to-video samples and store replay conditioning."""
        device = self.device
        if self.pipeline.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale
        if (num_frames - 1) % self.pipeline.vae_scale_factor_temporal != 0:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.pipeline.vae_scale_factor_temporal}. "
                "Rounding to the nearest number."
            )
            num_frames = (
                num_frames
                // self.pipeline.vae_scale_factor_temporal
                * self.pipeline.vae_scale_factor_temporal
                + 1
            )
        num_frames = max(num_frames, 1)

        patch_size = (
            self.pipeline.transformer.config.patch_size
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.config.patch_size
        )
        h_multiple_of = self.pipeline.vae_scale_factor_spatial * patch_size[1]
        w_multiple_of = self.pipeline.vae_scale_factor_spatial * patch_size[2]
        calc_height = height // h_multiple_of * h_multiple_of
        calc_width = width // w_multiple_of * w_multiple_of
        if height != calc_height or width != calc_width:
            logger.warning(
                f"`height` and `width` must be multiples of ({h_multiple_of}, {w_multiple_of}) "
                f"for proper patchification. Adjusting ({height}, {width}) -> "
                f"({calc_height}, {calc_width})."
            )
            height, width = calc_height, calc_width

        if prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_embeds = encoded["prompt_embeds"]
            negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        else:
            prompt_embeds = prompt_embeds.to(device)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device)

        batch_size = prompt_embeds.shape[0]
        transformer = self.pipeline.transformer or self.pipeline.transformer_2
        transformer_dtype = transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        if control_hidden_states is None:
            control_kwargs = self.prepare_empty_conditioning(
                batch_size=batch_size,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=transformer_dtype,
                device=device,
                generator=generator,
                conditioning_scale=control_hidden_states_scale,
            )
            control_hidden_states = control_kwargs["control_hidden_states"]
            control_hidden_states_scale = control_kwargs["control_hidden_states_scale"]
        else:
            control_hidden_states = control_hidden_states.to(device=device, dtype=transformer_dtype)
            control_hidden_states_scale = self._normalize_conditioning_scale(
                conditioning_scale=control_hidden_states_scale,
                dtype=transformer_dtype,
                device=device,
                transformer=transformer,
            )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self.pipeline._num_timesteps = len(timesteps)

        num_channels_latents = transformer.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latents = self.cast_latents(latents)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(
                trajectory_indices, num_inference_steps
            )
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        for i, t in enumerate(timesteps):
            self.pipeline._current_timestep = t
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kwargs = list(
                set(["next_latents", "log_prob", "noise_pred"] + extra_call_back_kwargs)
            )
            current_compute_log_prob = compute_log_prob and current_noise_level > 0

            output = self.forward(
                t=t,
                t_next=t_next,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                control_hidden_states=control_hidden_states,
                control_hidden_states_scale=control_hidden_states_scale,
                attention_kwargs=attention_kwargs,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )
            latents = self.cast_latents(output.next_latents)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)
            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        self.pipeline._current_timestep = None
        decoded_videos = self.decode_latents(latents, output_type="pt") if decode_media else None
        extra_call_back_res = callback_collector.get_result()
        callback_index_map = callback_collector.get_index_map()
        all_latents = latent_collector.get_result()
        latent_index_map = latent_collector.get_index_map()
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None

        samples = [
            WanVACESample(
                timesteps=timesteps,
                all_latents=(
                    torch.stack([lat[b] for lat in all_latents], dim=0)
                    if all_latents is not None
                    else None
                ),
                log_probs=(
                    torch.stack([lp[b] for lp in all_log_probs], dim=0)
                    if all_log_probs is not None
                    else None
                ),
                latent_index_map=latent_index_map,
                log_prob_index_map=log_prob_index_map,
                video=decoded_videos[b] if decoded_videos is not None else None,
                height=height,
                width=width,
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_embeds=prompt_embeds[b],
                negative_prompt=(
                    negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt
                ),
                negative_prompt_embeds=(
                    negative_prompt_embeds[b] if negative_prompt_embeds is not None else None
                ),
                control_hidden_states=control_hidden_states[b],
                control_hidden_states_scale=control_hidden_states_scale,
                extra_kwargs={
                    **{k: v[b] for k, v in extra_call_back_res.items()},
                    "callback_index_map": callback_index_map,
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()
        return samples

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        control_hidden_states: Optional[torch.Tensor] = None,
        control_hidden_states_scale: Union[float, List[float], torch.Tensor] = 1.0,
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        noise_level: Optional[float] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "noise_pred",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ],
        boundary_timestep: Optional[float] = None,
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Run one Wan VACE denoising step."""
        t = t[0] if t.ndim == 1 else t
        if t_next is not None:
            t_next = t_next[0] if t_next.ndim == 1 else t_next

        batch_size = latents.shape[0]
        device = latents.device
        dtype = (
            self.pipeline.transformer.dtype
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.dtype
        )

        if boundary_timestep is None and self.pipeline.config.boundary_ratio is not None:
            boundary_timestep = (
                self.pipeline.config.boundary_ratio * self.scheduler.config.num_train_timesteps
            )
        if boundary_timestep is None or t >= boundary_timestep:
            pipeline_transformer = self.pipeline.transformer
            transformer = self.transformer
            current_guidance_scale = guidance_scale
        else:
            pipeline_transformer = self.pipeline.transformer_2
            transformer = self.transformer_2
            current_guidance_scale = (
                guidance_scale_2 if guidance_scale_2 is not None else guidance_scale
            )

        if control_hidden_states is None:
            geometry = {
                "num_frames": (latents.shape[2] - 1) * self.pipeline.vae_scale_factor_temporal + 1,
                "height": latents.shape[3] * self.pipeline.vae_scale_factor_spatial,
                "width": latents.shape[4] * self.pipeline.vae_scale_factor_spatial,
            }
            with torch.no_grad():
                control_kwargs = self.prepare_empty_conditioning(
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                    conditioning_scale=control_hidden_states_scale,
                    **geometry,
                )
            control_hidden_states = control_kwargs["control_hidden_states"]
            control_hidden_states_scale = control_kwargs["control_hidden_states_scale"]
        else:
            control_hidden_states = control_hidden_states.to(device=device, dtype=dtype)
            control_hidden_states_scale = self._normalize_conditioning_scale(
                conditioning_scale=control_hidden_states_scale,
                dtype=dtype,
                device=device,
                transformer=transformer,
            )

        if current_guidance_scale > 1.0 and negative_prompt_embeds is None:
            logger.warning(
                "Passed `guidance_scale` > 1.0, but no `negative_prompt_embeds` provided. "
                "Classifier-free guidance will be disabled."
            )
        do_classifier_free_guidance = (
            negative_prompt_embeds is not None and current_guidance_scale > 1.0
        )

        latent_model_input = latents.to(dtype)
        timestep = t.expand(batch_size)

        with pipeline_transformer.cache_context("cond"):
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                control_hidden_states=control_hidden_states,
                control_hidden_states_scale=control_hidden_states_scale,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

        if do_classifier_free_guidance:
            with pipeline_transformer.cache_context("uncond"):
                noise_uncond = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    control_hidden_states=control_hidden_states,
                    control_hidden_states_scale=control_hidden_states_scale,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

        return self.scheduler.step(
            noise_pred=noise_pred,
            timestep=t,
            latents=latents,
            timestep_next=t_next,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
            noise_level=noise_level,
        )
