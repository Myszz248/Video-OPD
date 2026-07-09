#!/usr/bin/env python
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

"""Run Wan2.1 VACE TI2V inference with student-trajectory and teacher-step mixing."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
import yaml
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from peft import PeftModel
from PIL import Image


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Wan2.1 VACE TI2V inference that stores the student trajectory, mixes the "
            "student target step with the teacher first-step latent, and lets the teacher "
            "finish denoising from the mixed state."
        )
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Local path or Hugging Face repo id for the student Wan2.1 VACE model.",
    )
    parser.add_argument(
        "--teacher-base-model",
        type=str,
        default=None,
        help="Optional teacher base model path. Defaults to --base-model.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the first-frame image used for TI2V conditioning.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt shared by student and teacher. Pass an empty string to disable it.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional student VACE LoRA checkpoint path.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=None,
        help="Optional teacher VACE LoRA checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint-type",
        choices=["lora"],
        default="lora",
        help="Compatibility flag. This script supports VACE LoRA checkpoints.",
    )
    parser.add_argument(
        "--teacher-checkpoint-type",
        choices=["lora"],
        default="lora",
        help="Compatibility flag. This script supports VACE LoRA checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional Flow-Factory YAML config used to resolve teacher defaults.",
    )
    parser.add_argument(
        "--target-components",
        type=str,
        default="transformer",
        help="Compatibility flag; VACE LoRA is always loaded onto the transformer.",
    )
    parser.add_argument(
        "--teacher-target-components",
        type=str,
        default="transformer",
        help="Compatibility flag; VACE LoRA is always loaded onto the transformer.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output video path, usually ending with .mp4.",
    )
    parser.add_argument("--height", type=int, default=480, help="Generated video height.")
    parser.add_argument("--width", type=int, default=832, help="Generated video width.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of generated frames.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps. Official Wan VACE uses 50 by default.",
    )
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Student CFG scale.")
    parser.add_argument(
        "--teacher-guidance-scale",
        type=float,
        default=None,
        help="Teacher CFG scale. Defaults to --guidance-scale.",
    )
    parser.add_argument(
        "--conditioning-scale",
        type=float,
        default=1.0,
        help=(
            "Compatibility flag kept for teacher default resolution. "
            "Student now runs prompt-only and does not consume image conditioning."
        ),
    )
    parser.add_argument(
        "--teacher-conditioning-scale",
        type=float,
        default=None,
        help="Teacher VACE conditioning scale. Defaults to --conditioning-scale.",
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=16.0,
        help="Student scheduler flow shift. Official Wan VACE uses 16 by default.",
    )
    parser.add_argument(
        "--teacher-flow-shift",
        type=float,
        default=None,
        help="Teacher scheduler flow shift. Defaults to --flow-shift.",
    )
    parser.add_argument(
        "--mix-step-index",
        type=int,
        required=True,
        help=(
            "Student post-step index to mix with teacher post-step 0. "
            "Negative values are supported, for example -2 means the second-to-last step."
        ),
    )
    parser.add_argument(
        "--student-latent-weight",
        type=float,
        default=0.5,
        help="Weight applied to the student target-step latent during mixing.",
    )
    parser.add_argument(
        "--teacher-first-step-weight",
        type=float,
        default=0.5,
        help="Weight applied to the teacher first-step latent during mixing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
        help="Student transformer/text encoder inference dtype.",
    )
    parser.add_argument(
        "--teacher-dtype",
        choices=["fp16", "bf16", "fp32"],
        default=None,
        help="Teacher transformer/text encoder inference dtype. Defaults to --dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device. Defaults to cuda.",
    )
    parser.add_argument(
        "--enable-model-cpu-offload",
        action="store_true",
        help="Enable diffusers model CPU offload instead of moving the full pipeline to device.",
    )
    parser.add_argument("--fps", type=int, default=16, help="Output video frame rate.")
    parser.add_argument("--quality", type=float, default=8.0, help="MP4 quality from 0 to 10.")
    parser.add_argument(
        "--macro-block-size",
        type=int,
        default=16,
        help="imageio macro block size. Use 1 to disable automatic resizing.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map a CLI dtype name to torch dtype."""
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_model_path(model_path: str) -> str:
    """Resolve a model path while allowing Hugging Face repo ids."""
    resolved_path = os.path.abspath(os.path.expanduser(model_path))
    return resolved_path if os.path.exists(resolved_path) else model_path


def resolve_lora_path(checkpoint: str) -> str:
    """Resolve a VACE LoRA checkpoint directory."""
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint))
    transformer_path = os.path.join(checkpoint_path, "transformer")
    if os.path.isdir(transformer_path):
        return transformer_path
    if os.path.isdir(checkpoint_path):
        return checkpoint_path
    raise FileNotFoundError(f"LoRA checkpoint directory not found: {checkpoint_path}")


def resolve_image_path(image_path: str) -> str:
    """Resolve and validate the first-frame image path."""
    resolved_path = os.path.abspath(os.path.expanduser(image_path))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"First-frame image not found: {resolved_path}")
    return resolved_path


def load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load one optional YAML config file as a dictionary."""
    if config_path is None:
        return {}
    resolved_path = os.path.abspath(os.path.expanduser(config_path))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML config to decode to a mapping, got {type(config).__name__}.")
    return config


def resolve_teacher_base_model(
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> str:
    """Resolve the teacher base model from CLI first, then YAML, then student base."""
    if args.teacher_base_model is not None:
        return args.teacher_base_model
    teacher_config = config.get("teacher")
    if isinstance(teacher_config, dict):
        model_name_or_path = teacher_config.get("model_name_or_path")
        if isinstance(model_name_or_path, str) and model_name_or_path.strip():
            return model_name_or_path
    return args.base_model


def resolve_teacher_guidance_scale(
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> float:
    """Resolve teacher guidance scale from CLI, then YAML, then student guidance."""
    if args.teacher_guidance_scale is not None:
        return args.teacher_guidance_scale
    train_config = config.get("train")
    if isinstance(train_config, dict):
        guidance_scale = train_config.get("teacher_guidance_scale")
        if isinstance(guidance_scale, (int, float)):
            return float(guidance_scale)
    return args.guidance_scale


def resolve_teacher_conditioning_scale(
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> float:
    """Resolve teacher conditioning scale from CLI, then YAML, then student conditioning."""
    if args.teacher_conditioning_scale is not None:
        return args.teacher_conditioning_scale
    teacher_config = config.get("teacher")
    if isinstance(teacher_config, dict):
        extra_kwargs = teacher_config.get("extra_kwargs")
        if isinstance(extra_kwargs, dict):
            conditioning_scale = extra_kwargs.get("conditioning_scale")
            if isinstance(conditioning_scale, (int, float)):
                return float(conditioning_scale)
    return args.conditioning_scale


def prepare_first_frame_vace_control(
    image: Image.Image,
    height: int,
    width: int,
    num_frames: int,
) -> tuple[List[Image.Image], List[Image.Image]]:
    """Create VACE control inputs that keep only the first frame fixed."""
    image = image.convert("RGB").resize((width, height))
    blank = Image.new("RGB", (width, height), (0, 0, 0))
    mask_keep = Image.new("L", (width, height), 0)
    mask_generate = Image.new("L", (width, height), 255)
    video = [image] + [blank.copy() for _ in range(num_frames - 1)]
    mask = [mask_keep] + [mask_generate.copy() for _ in range(num_frames - 1)]
    return video, mask


def prepare_blank_vace_control(
    height: int,
    width: int,
    num_frames: int,
) -> tuple[List[Image.Image], List[Image.Image]]:
    """Create prompt-only VACE control inputs matching official blank conditioning."""
    video = [
        Image.new("RGB", (width, height), (128, 128, 128))
        for _ in range(num_frames)
    ]
    mask = [Image.new("L", (width, height), 255) for _ in range(num_frames)]
    return video, mask


def move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors in lightweight runtime containers to a device."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        moved = [move_to_device(item, device) for item in value]
        return moved if any(new is not old for new, old in zip(moved, value)) else value
    if isinstance(value, tuple):
        moved = tuple(move_to_device(item, device) for item in value)
        return moved if any(new is not old for new, old in zip(moved, value)) else value
    return value


def move_scheduler_to_device(pipe: WanVACEPipeline, device: str | torch.device) -> None:
    """Move scheduler tensor state to the inference device without touching config."""
    target_device = torch.device(device)
    skip_names = {"config", "_internal_dict", "compatibles"}
    for name, value in vars(pipe.scheduler).items():
        if name in skip_names or name.startswith("_deprecated"):
            continue
        moved = move_to_device(value, target_device)
        if moved is not value:
            setattr(pipe.scheduler, name, moved)


def patch_scheduler_device_guard(pipe: WanVACEPipeline) -> None:
    """Keep scheduler state on the same device as the latent sample during step()."""
    original_step = pipe.scheduler.step

    def guarded_step(*args: Any, **kwargs: Any) -> Any:
        sample = kwargs.get("sample")
        if sample is None and len(args) >= 3:
            sample = args[2]
        if torch.is_tensor(sample):
            move_scheduler_to_device(pipe, sample.device)
        return original_step(*args, **kwargs)

    pipe.scheduler.step = guarded_step


def summarize_lora_state(transformer: torch.nn.Module) -> Dict[str, Any]:
    """Return basic diagnostics for loaded LoRA parameters."""
    lora_tensors = [
        param.detach().float()
        for name, param in transformer.named_parameters()
        if "lora_" in name
    ]
    total_params = sum(tensor.numel() for tensor in lora_tensors)
    abs_sum = sum(tensor.abs().sum().item() for tensor in lora_tensors)
    max_abs = max((tensor.abs().max().item() for tensor in lora_tensors), default=0.0)
    active_adapters = getattr(transformer, "active_adapters", None)
    if callable(active_adapters):
        active_adapters = active_adapters()
    return {
        "lora_tensor_count": len(lora_tensors),
        "lora_param_count": total_params,
        "lora_abs_sum": abs_sum,
        "lora_max_abs": max_abs,
        "active_adapters": active_adapters,
    }


def set_pipeline_eval_mode(pipe: WanVACEPipeline) -> None:
    """Put available Wan transformer modules into eval mode."""
    if getattr(pipe, "transformer", None) is not None:
        pipe.transformer.eval()
    if getattr(pipe, "transformer_2", None) is not None:
        pipe.transformer_2.eval()


def load_pipeline(
    *,
    base_model: str,
    checkpoint: Optional[str],
    dtype_name: str,
    device: str,
    enable_model_cpu_offload: bool,
    flow_shift: float,
    role_name: str,
) -> WanVACEPipeline:
    """Load WanVACEPipeline and attach optional VACE LoRA weights."""
    base_model = resolve_model_path(base_model)
    torch_dtype = get_torch_dtype(dtype_name)

    vae = AutoencoderKLWan.from_pretrained(
        base_model,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanVACEPipeline.from_pretrained(
        base_model,
        vae=vae,
        torch_dtype=torch_dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )

    if checkpoint is not None:
        lora_path = resolve_lora_path(checkpoint)
        print(f"Loading {role_name} VACE LoRA from: {lora_path}")
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            lora_path,
            torch_dtype=torch_dtype,
            is_trainable=False,
        )
        if hasattr(pipe.transformer, "set_adapter"):
            pipe.transformer.set_adapter("default")
        diagnostics = summarize_lora_state(pipe.transformer)
        print(f"{role_name} LoRA diagnostics: {json.dumps(diagnostics, ensure_ascii=False)}")
        if diagnostics["lora_tensor_count"] == 0 or diagnostics["lora_abs_sum"] == 0.0:
            raise ValueError(
                f"{role_name} LoRA checkpoint was loaded, but no non-zero LoRA parameters were found. "
                "Please check that the checkpoint points to the real adapter directory."
            )
    else:
        print(f"No {role_name} checkpoint provided; running the base VACE model only.")

    set_pipeline_eval_mode(pipe)
    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
        move_scheduler_to_device(pipe, device)
    patch_scheduler_device_guard(pipe)
    return pipe


def get_execution_device(pipe: WanVACEPipeline, fallback_device: str) -> torch.device:
    """Resolve the active execution device for one pipeline."""
    execution_device = getattr(pipe, "_execution_device", None)
    if execution_device is None:
        return torch.device(fallback_device)
    return execution_device


def normalize_num_frames(pipe: WanVACEPipeline, num_frames: int) -> int:
    """Mirror the Wan pipeline frame-count normalization."""
    temporal_scale = int(pipe.vae_scale_factor_temporal)
    if num_frames % temporal_scale != 1:
        num_frames = num_frames // temporal_scale * temporal_scale + 1
    return max(num_frames, 1)


def build_generator(seed: int, device: str) -> torch.Generator:
    """Create one deterministic torch generator."""
    return torch.Generator(device=device).manual_seed(seed)


def prepare_initial_latents(
    pipe: WanVACEPipeline,
    *,
    height: int,
    width: int,
    num_frames: int,
    generator: torch.Generator,
    fallback_device: str,
) -> torch.Tensor:
    """Prepare one shared initial latent tensor for student and teacher."""
    transformer = pipe.transformer if pipe.transformer is not None else pipe.transformer_2
    if transformer is None:
        raise ValueError("Wan VACE pipeline has neither `transformer` nor `transformer_2`.")
    device = get_execution_device(pipe, fallback_device)
    return pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=transformer.config.in_channels,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=torch.float32,
        device=device,
        generator=generator,
        latents=None,
    )


def normalize_mix_step_index(step_idx: int, num_inference_steps: int) -> int:
    """Normalize one post-step index to [0, num_inference_steps)."""
    if step_idx < 0:
        step_idx += num_inference_steps
    if not 0 <= step_idx < num_inference_steps:
        raise ValueError(
            f"`mix_step_index` must be within [0, {num_inference_steps}) after normalization, "
            f"got {step_idx}."
        )
    if step_idx >= num_inference_steps - 1:
        raise ValueError(
            "The mixed latent must still leave teacher denoising steps to run. "
            f"Got mix_step_index({step_idx}) with num_inference_steps({num_inference_steps})."
        )
    return step_idx


def reset_unipc_scheduler_state(scheduler: Any, begin_index: int) -> None:
    """Reset UniPC multistep history so continuation restarts cleanly at one later step."""
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(begin_index)
    if hasattr(scheduler, "model_outputs"):
        scheduler.model_outputs = [None] * len(scheduler.model_outputs)
    if hasattr(scheduler, "timestep_list"):
        scheduler.timestep_list = [None] * len(scheduler.timestep_list)
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    nested_solver = getattr(scheduler, "solver_p", None)
    if nested_solver is not None:
        reset_unipc_scheduler_state(nested_solver, begin_index)


def capture_teacher_first_step_latent(
    pipe: WanVACEPipeline,
    *,
    initial_latents: torch.Tensor,
    prompt: str,
    negative_prompt: Optional[str],
    video: List[Image.Image],
    mask: List[Image.Image],
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    conditioning_scale: float,
) -> torch.Tensor:
    """Run teacher for one effective denoising step and return its post-step latent."""
    teacher_first_step_latent: Optional[torch.Tensor] = None
    pipe._interrupt = False

    def capture_callback(
        callback_pipe: WanVACEPipeline,
        step: int,
        _timestep: int,
        callback_kwargs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        nonlocal teacher_first_step_latent
        if step == 0:
            teacher_first_step_latent = callback_kwargs["latents"].detach().cpu().clone()
            callback_pipe._interrupt = True
        return callback_kwargs

    with torch.no_grad():
        pipe(
            video=video,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            conditioning_scale=conditioning_scale,
            latents=initial_latents.clone(),
            output_type="latent",
            callback_on_step_end=capture_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    pipe._interrupt = False
    if teacher_first_step_latent is None:
        raise RuntimeError("Teacher first-step latent was not captured.")
    return teacher_first_step_latent


def continue_teacher_from_mixed_latents(
    pipe: WanVACEPipeline,
    *,
    mixed_latents: torch.Tensor,
    prompt: str,
    negative_prompt: Optional[str],
    video: List[Image.Image],
    mask: List[Image.Image],
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    start_step_index: int,
    guidance_scale: float,
    conditioning_scale: float,
) -> List[Any]:
    """Continue teacher denoising from a mixed latent using the original remaining schedule."""
    scheduler = pipe.scheduler
    original_set_timesteps = scheduler.set_timesteps

    original_set_timesteps(num_inference_steps, device=mixed_latents.device)
    full_timesteps = scheduler.timesteps.detach().clone()
    full_sigmas = scheduler.sigmas.detach().clone() if hasattr(scheduler, "sigmas") else None
    remaining_timesteps = full_timesteps[start_step_index + 1 :]
    if remaining_timesteps.numel() == 0:
        raise ValueError(
            "No teacher timesteps remain after the mixed step. "
            f"start_step_index={start_step_index}, num_inference_steps={num_inference_steps}."
        )
    remaining_sigmas = (
        full_sigmas[start_step_index + 1 :] if full_sigmas is not None else None
    )

    def custom_set_timesteps(
        _num_inference_steps: Optional[int] = None,
        device: Optional[str | torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ) -> None:
        original_set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
        scheduler.timesteps = remaining_timesteps.to(device=device or mixed_latents.device)
        if remaining_sigmas is not None:
            scheduler.sigmas = remaining_sigmas.to(device=device or mixed_latents.device)
        scheduler.num_inference_steps = int(remaining_timesteps.shape[0])

    scheduler.set_timesteps = custom_set_timesteps
    try:
        pipe._interrupt = False
        with torch.no_grad():
            return pipe(
                video=video,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=int(remaining_timesteps.shape[0]),
                guidance_scale=guidance_scale,
                conditioning_scale=conditioning_scale,
                latents=mixed_latents.clone(),
            ).frames[0]
    finally:
        pipe._interrupt = False
        scheduler.set_timesteps = original_set_timesteps


def collect_student_target_step_latent(
    pipe: WanVACEPipeline,
    *,
    video: List[Image.Image],
    mask: List[Image.Image],
    prompt: str,
    negative_prompt: Optional[str],
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    target_step_index: int,
    guidance_scale: float,
    conditioning_scale: Optional[float],
    latents: torch.Tensor,
) -> torch.Tensor:
    """Run student inference and store only the requested post-step latent on CPU."""
    target_latent: Optional[torch.Tensor] = None
    pipe._interrupt = False

    def capture_callback(
        callback_pipe: WanVACEPipeline,
        step: int,
        _timestep: int,
        callback_kwargs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        nonlocal target_latent
        if step == target_step_index:
            target_latent = callback_kwargs["latents"].detach().cpu().clone()
            callback_pipe._interrupt = True
        return callback_kwargs

    inference_kwargs: Dict[str, Any] = {
        "video": video,
        "mask": mask,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "latents": latents.clone(),
        "output_type": "latent",
        "callback_on_step_end": capture_callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
    if conditioning_scale is not None:
        inference_kwargs["conditioning_scale"] = conditioning_scale

    with torch.no_grad():
        pipe(**inference_kwargs)
    pipe._interrupt = False

    if target_latent is None:
        raise RuntimeError(
            "Student target-step latent was not captured. "
            f"target_step_index={target_step_index}, num_inference_steps={num_inference_steps}."
        )
    return target_latent


def build_metadata(
    args: argparse.Namespace,
    *,
    output_path: str,
    image_path: str,
    student_lora_path: Optional[str],
    teacher_lora_path: Optional[str],
    effective_num_frames: int,
    normalized_mix_step_index: int,
    resolved_teacher_base_model: str,
    resolved_teacher_guidance_scale: float,
    resolved_teacher_conditioning_scale: float,
    resolved_teacher_flow_shift: float,
    resolved_teacher_dtype: str,
) -> Dict[str, Any]:
    """Build JSON-serializable inference metadata."""
    return {
        "student": {
            "base_model": args.base_model,
            "checkpoint": (
                os.path.abspath(os.path.expanduser(args.checkpoint))
                if args.checkpoint is not None
                else None
            ),
            "resolved_lora_path": student_lora_path,
            "guidance_scale": args.guidance_scale,
            "conditioning_mode": "prompt_only_blank_vace_control",
            "conditioning_scale": None,
            "flow_shift": args.flow_shift,
            "dtype": args.dtype,
        },
        "teacher": {
            "base_model": resolved_teacher_base_model,
            "checkpoint": (
                os.path.abspath(os.path.expanduser(args.teacher_checkpoint))
                if args.teacher_checkpoint is not None
                else None
            ),
            "resolved_lora_path": teacher_lora_path,
            "conditioning_mode": "first_frame_image",
            "guidance_scale": resolved_teacher_guidance_scale,
            "conditioning_scale": resolved_teacher_conditioning_scale,
            "flow_shift": resolved_teacher_flow_shift,
            "dtype": resolved_teacher_dtype,
        },
        "config": (
            os.path.abspath(os.path.expanduser(args.config)) if args.config is not None else None
        ),
        "image_path": image_path,
        "output": output_path,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "effective_num_frames": effective_num_frames,
        "num_inference_steps": args.num_inference_steps,
        "mix_step_index": normalized_mix_step_index,
        "teacher_first_step_index": 0,
        "student_latent_weight": args.student_latent_weight,
        "teacher_first_step_weight": args.teacher_first_step_weight,
        "seed": args.seed,
        "device": args.device,
        "enable_model_cpu_offload": args.enable_model_cpu_offload,
        "fps": args.fps,
    }


def save_metadata(metadata: Dict[str, Any], output_path: str) -> str:
    """Save a JSON sidecar describing the inference run."""
    metadata_path = f"{os.path.splitext(output_path)[0]}.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata_path


def cleanup_cuda_cache() -> None:
    """Release cached CUDA memory if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """Run student-trajectory to teacher-mix Wan2.1 VACE inference and save the result."""
    args = parse_args()
    config = load_yaml_config(args.config)
    output_path = os.path.abspath(os.path.expanduser(args.output))
    image_path = resolve_image_path(args.image_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    teacher_base_model = resolve_teacher_base_model(args, config)
    teacher_guidance_scale = resolve_teacher_guidance_scale(args, config)
    teacher_conditioning_scale = resolve_teacher_conditioning_scale(args, config)
    teacher_flow_shift = (
        args.teacher_flow_shift if args.teacher_flow_shift is not None else args.flow_shift
    )
    teacher_dtype = args.teacher_dtype if args.teacher_dtype is not None else args.dtype
    negative_prompt: Optional[str] = args.negative_prompt
    if negative_prompt == "":
        negative_prompt = None

    student_lora_path = resolve_lora_path(args.checkpoint) if args.checkpoint is not None else None
    teacher_lora_path = (
        resolve_lora_path(args.teacher_checkpoint) if args.teacher_checkpoint is not None else None
    )

    generator_device = "cpu" if args.enable_model_cpu_offload else args.device
    student_latent_generator = build_generator(args.seed, generator_device)
    student_pipe = load_pipeline(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        dtype_name=args.dtype,
        device=args.device,
        enable_model_cpu_offload=args.enable_model_cpu_offload,
        flow_shift=args.flow_shift,
        role_name="student",
    )
    effective_num_frames = normalize_num_frames(student_pipe, args.num_frames)
    normalized_mix_step_index = normalize_mix_step_index(
        args.mix_step_index,
        args.num_inference_steps,
    )
    shared_initial_latents = prepare_initial_latents(
        student_pipe,
        height=args.height,
        width=args.width,
        num_frames=effective_num_frames,
        generator=student_latent_generator,
        fallback_device=args.device,
    )

    with Image.open(image_path) as input_image:
        teacher_video, teacher_mask = prepare_first_frame_vace_control(
            input_image,
            height=args.height,
            width=args.width,
            num_frames=effective_num_frames,
        )
    student_video, student_mask = prepare_blank_vace_control(
        height=args.height,
        width=args.width,
        num_frames=effective_num_frames,
    )

    student_target_step_latent = collect_student_target_step_latent(
        student_pipe,
        video=student_video,
        mask=student_mask,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=effective_num_frames,
        num_inference_steps=args.num_inference_steps,
        target_step_index=normalized_mix_step_index,
        guidance_scale=args.guidance_scale,
        conditioning_scale=None,
        latents=shared_initial_latents,
    )

    del student_pipe
    cleanup_cuda_cache()

    teacher_pipe = load_pipeline(
        base_model=teacher_base_model,
        checkpoint=args.teacher_checkpoint,
        dtype_name=teacher_dtype,
        device=args.device,
        enable_model_cpu_offload=args.enable_model_cpu_offload,
        flow_shift=teacher_flow_shift,
        role_name="teacher",
    )
    teacher_first_step_latent = capture_teacher_first_step_latent(
        teacher_pipe,
        initial_latents=shared_initial_latents.clone(),
        video=teacher_video,
        mask=teacher_mask,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=effective_num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=teacher_guidance_scale,
        conditioning_scale=teacher_conditioning_scale,
    )
    mixed_latents = (
        args.student_latent_weight
        * student_target_step_latent.to(
            device=shared_initial_latents.device,
            dtype=shared_initial_latents.dtype,
        )
        + args.teacher_first_step_weight
        * teacher_first_step_latent.to(
            device=shared_initial_latents.device,
            dtype=shared_initial_latents.dtype,
        )
    )
    frames = continue_teacher_from_mixed_latents(
        teacher_pipe,
        mixed_latents=mixed_latents,
        video=teacher_video,
        mask=teacher_mask,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=effective_num_frames,
        num_inference_steps=args.num_inference_steps,
        start_step_index=normalized_mix_step_index,
        guidance_scale=teacher_guidance_scale,
        conditioning_scale=teacher_conditioning_scale,
    )
    del teacher_pipe
    cleanup_cuda_cache()

    export_to_video(
        frames,
        output_path,
        fps=args.fps,
        quality=args.quality,
        macro_block_size=args.macro_block_size,
    )
    metadata = build_metadata(
        args,
        output_path=output_path,
        image_path=image_path,
        student_lora_path=student_lora_path,
        teacher_lora_path=teacher_lora_path,
        effective_num_frames=effective_num_frames,
        normalized_mix_step_index=normalized_mix_step_index,
        resolved_teacher_base_model=teacher_base_model,
        resolved_teacher_guidance_scale=teacher_guidance_scale,
        resolved_teacher_conditioning_scale=teacher_conditioning_scale,
        resolved_teacher_flow_shift=teacher_flow_shift,
        resolved_teacher_dtype=teacher_dtype,
    )
    metadata["teacher_first_step_latent_shape"] = list(teacher_first_step_latent.shape)
    metadata_path = save_metadata(metadata, output_path)
    print(f"Saved video to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
