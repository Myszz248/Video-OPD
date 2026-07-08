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

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from accelerate import Accelerator
from diffusers.utils import export_to_video

from .hparams import Arguments
from .inference import (
    _freeze_adapter,
    _load_components_for_inference,
    _resolve_generation_value,
    _sample_to_frames,
)
from .models.loader import load_model
from .teachers import load_opd_teacher
from .teachers.context import OPDContextBuilder
from .utils.base import filter_kwargs
from .utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for teacher-resume debug."""
    parser = argparse.ArgumentParser(
        description=(
            "Standalone OPD teacher-resume debug. Student trajectory latents are generated "
            "from text prompt only; teacher first-step latents are generated from prompt plus "
            "teacher image context, then optionally blended before teacher resume."
        )
    )
    parser.add_argument("--config", type=str, required=True, help="OPD YAML config.")
    parser.add_argument("--prompt", type=str, required=True, help="Student prompt text.")
    parser.add_argument(
        "--teacher-image",
        type=str,
        required=True,
        help="Teacher first-frame image path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output mp4 path for the teacher-resumed video.",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        required=True,
        help="Student denoising step index used to resume the teacher.",
    )
    parser.add_argument(
        "--teacher-image-key",
        type=str,
        default="first_frame_path",
        help="Teacher context key used for the image path.",
    )
    parser.add_argument(
        "--teacher-context-json",
        type=str,
        default=None,
        help="Optional JSON object merged into teacher-only context.",
    )
    parser.add_argument("--negative-prompt", type=str, default=None, help="Optional negative prompt.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional student checkpoint override.")
    parser.add_argument(
        "--resume-type",
        choices=["lora", "full"],
        default=None,
        help="Optional student checkpoint type override.",
    )
    parser.add_argument("--height", type=int, default=None, help="Override generation height.")
    parser.add_argument("--width", type=int, default=None, help="Override generation width.")
    parser.add_argument("--num-frames", type=int, default=None, help="Override generated frame count.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override denoising step count.",
    )
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override student CFG scale.")
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=None,
        help="Override second-stage CFG scale for Wan models.",
    )
    parser.add_argument(
        "--student-latent-weight",
        type=float,
        default=1.0,
        help="Weight applied to the extracted student latent before teacher resume.",
    )
    parser.add_argument(
        "--teacher-first-step-weight",
        type=float,
        default=0.0,
        help="Weight applied to the teacher first-step latent before teacher resume.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--fps", type=int, default=16, help="Saved video FPS.")
    parser.add_argument("--quality", type=float, default=8.0, help="MP4 quality from 0 to 10.")
    parser.add_argument(
        "--macro-block-size",
        type=int,
        default=16,
        help="imageio macro block size. Use 1 to disable automatic resizing.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default=None,
        help="Optional mixed precision override.",
    )
    parser.add_argument(
        "--inference-kwargs-json",
        type=str,
        default=None,
        help="Extra student adapter.inference kwargs as a JSON object.",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> Arguments:
    """Load config and apply standalone debug overrides."""
    config = Arguments.load_from_yaml(args.config)
    config.log_args.logging_backend = "none"

    if args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision
    if args.checkpoint is not None:
        config.model_args.resume_path = os.path.expanduser(args.checkpoint)
    if args.resume_type is not None:
        config.model_args.resume_type = args.resume_type
        config.model_args.finetune_type = args.resume_type

    context_keys = list(config.training_args.teacher_context_keys)
    if args.teacher_image_key not in context_keys:
        context_keys.append(args.teacher_image_key)
    config.training_args.teacher_context_keys = context_keys
    return config


def _parse_json_dict(raw_value: Optional[str], field_name: str) -> Dict[str, Any]:
    """Parse an optional JSON object argument."""
    if raw_value is None:
        return {}
    value = json.loads(raw_value)
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must decode to a JSON object.")
    return value


def _normalize_step_index(step_idx: int, num_inference_steps: int) -> int:
    """Normalize one denoising step index to [0, num_inference_steps)."""
    if step_idx < 0:
        step_idx += num_inference_steps
    if not 0 <= step_idx < num_inference_steps:
        raise ValueError(
            f"`step_index` must be within [0, {num_inference_steps}) after normalization, got {step_idx}."
        )
    return step_idx


def _build_teacher_context(args: argparse.Namespace) -> str:
    """Build serialized teacher-only context for one sample."""
    context = _parse_json_dict(args.teacher_context_json, "--teacher-context-json")
    context[args.teacher_image_key] = os.path.abspath(os.path.expanduser(args.teacher_image))
    return OPDContextBuilder.serialize_context(context)


def _resolve_blend_weights(args: argparse.Namespace) -> Tuple[float, float]:
    """Resolve and validate teacher/student latent blend weights."""
    student_weight = args.student_latent_weight
    teacher_weight = args.teacher_first_step_weight
    if student_weight < 0:
        raise ValueError(
            f"`student_latent_weight` must be non-negative, got {student_weight}."
        )
    if teacher_weight < 0:
        raise ValueError(
            f"`teacher_first_step_weight` must be non-negative, got {teacher_weight}."
        )
    if student_weight == 0 and teacher_weight == 0:
        raise ValueError(
            "At least one of `student_latent_weight` or `teacher_first_step_weight` must be positive."
        )
    return student_weight, teacher_weight


def _build_student_inference_kwargs(
    args: argparse.Namespace,
    config: Arguments,
    trajectory_step_idx: int,
    generator: Optional[torch.Generator],
) -> Dict[str, Any]:
    """Build student adapter.inference kwargs for one standalone debug run."""
    extra_kwargs = _parse_json_dict(args.inference_kwargs_json, "--inference-kwargs-json")
    num_frames = args.num_frames if args.num_frames is not None else 81
    kwargs: Dict[str, Any] = {
        "prompt": [args.prompt],
        "compute_log_prob": False,
        "trajectory_indices": [trajectory_step_idx],
        "generator": generator,
    }
    if args.negative_prompt is not None:
        kwargs["negative_prompt"] = [args.negative_prompt]

    optional_values = {
        "height": _resolve_generation_value(config, args, "height"),
        "width": _resolve_generation_value(config, args, "width"),
        "num_frames": num_frames,
        "num_inference_steps": _resolve_generation_value(config, args, "num_inference_steps"),
        "guidance_scale": _resolve_generation_value(config, args, "guidance_scale"),
        "guidance_scale_2": args.guidance_scale_2,
    }
    kwargs.update({key: value for key, value in optional_values.items() if value is not None})
    kwargs.update(extra_kwargs)
    return kwargs


def _resolve_output_path(output: str) -> str:
    """Resolve one concrete output mp4 path."""
    output_path = Path(os.path.expanduser(output))
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".mp4")
    os.makedirs(output_path.parent, exist_ok=True)
    return str(output_path)


def _build_resume_override_latents(
    *,
    args: argparse.Namespace,
    teacher: Any,
    student_sample: Any,
    step_idx: int,
    accelerator: Accelerator,
) -> Optional[torch.Tensor]:
    """Build one optional mixed latent override for teacher resume."""
    student_weight, teacher_weight = _resolve_blend_weights(args)
    if student_weight == 1.0 and teacher_weight == 0.0:
        return None
    if student_sample.prompt is None:
        raise ValueError("Teacher-resume blend debug requires the student sample prompt.")
    if student_sample.all_latents is None or student_sample.latent_index_map is None:
        raise ValueError(
            "Teacher-resume blend debug requires `all_latents` and `latent_index_map` on the student sample."
        )

    latent_index_map = student_sample.latent_index_map.to(device=accelerator.device)
    compact_idx = int(latent_index_map[step_idx].item())
    if compact_idx < 0:
        raise ValueError(
            "Requested teacher resume step was not stored in the student trajectory: "
            f"step_idx({step_idx}), latent_index_map={latent_index_map.detach().cpu().tolist()}."
        )

    student_latents = student_sample.all_latents[compact_idx].unsqueeze(0).to(teacher.teacher_args.device)
    if teacher_weight == 0.0:
        return (student_weight * student_latents).detach().cpu()

    context = student_sample.extra_kwargs.get("opd_context", "{}")
    negative_prompts = (
        [student_sample.negative_prompt] if student_sample.negative_prompt is not None else None
    )
    batch = {
        "prompt": [student_sample.prompt],
        "negative_prompt": negative_prompts,
    }
    reference_latents = student_sample.all_latents[0].unsqueeze(0).to(teacher.teacher_args.device)
    teacher_generator = None
    if args.seed is not None:
        # Keep teacher-first-step rollout deterministic without reusing the student's consumed RNG state.
        teacher_generator = torch.Generator(device=teacher.teacher_args.device).manual_seed(args.seed)

    teacher.on_load_runtime_components()
    try:
        with torch.no_grad():
            teacher_encoded = teacher.encode_prompt(
                prompts=batch["prompt"],
                contexts=[context],
                negative_prompts=batch["negative_prompt"],
                generator=teacher_generator,
            )
            teacher_first_step_latents = teacher.rollout_first_step_latents(
                batch=batch,
                contexts=[context],
                reference_latents=reference_latents,
                generator=teacher_generator,
                encoded_prompt=teacher_encoded,
            )
    finally:
        teacher.off_load_runtime_components()

    target_dtype = student_latents.dtype
    mixed_latents = (
        student_weight * student_latents.to(dtype=target_dtype)
        + teacher_weight * teacher_first_step_latents.to(dtype=target_dtype)
    )
    return mixed_latents.detach().cpu()


def run_debug(args: argparse.Namespace) -> str:
    """Run standalone student-rollout -> teacher-resume debug and save one mp4."""
    config = _load_config(args)
    accelerator = Accelerator(mixed_precision=config.mixed_precision)

    student_adapter = load_model(config=config, accelerator=accelerator)
    _load_components_for_inference(
        adapter=student_adapter,
        components=list(student_adapter.preprocessing_modules) + list(student_adapter.inference_modules),
        device=accelerator.device,
    )
    _freeze_adapter(student_adapter)
    student_adapter.eval()

    teacher = load_opd_teacher(config, accelerator)

    num_inference_steps = _resolve_generation_value(config, args, "num_inference_steps")
    if num_inference_steps is None:
        num_inference_steps = config.training_args.num_inference_steps
    trajectory_step_idx = _normalize_step_index(args.step_index, int(num_inference_steps))

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    inference_kwargs = _build_student_inference_kwargs(
        args=args,
        config=config,
        trajectory_step_idx=trajectory_step_idx,
        generator=generator,
    )
    inference_kwargs = filter_kwargs(student_adapter.inference, **inference_kwargs)

    with torch.no_grad(), accelerator.autocast():
        student_sample = student_adapter.inference(**inference_kwargs)[0]

    student_sample.extra_kwargs["opd_context"] = _build_teacher_context(args)
    override_latents = _build_resume_override_latents(
        args=args,
        teacher=teacher,
        student_sample=student_sample,
        step_idx=trajectory_step_idx,
        accelerator=accelerator,
    )
    teacher_sample = teacher.resume_from_student_samples(
        samples=[student_sample],
        step_idx=trajectory_step_idx,
        override_latents=[override_latents] if override_latents is not None else None,
    )[0]

    output_path = _resolve_output_path(args.output)
    frames = _sample_to_frames(teacher_sample)
    export_to_video(
        frames,
        output_video_path=output_path,
        fps=args.fps,
        quality=args.quality,
        macro_block_size=args.macro_block_size,
    )
    logger.info("Saved teacher-resumed debug video to %s", output_path)
    return output_path


def main() -> None:
    """Run the standalone teacher-resume debug CLI."""
    run_debug(parse_args())


if __name__ == "__main__":
    main()
