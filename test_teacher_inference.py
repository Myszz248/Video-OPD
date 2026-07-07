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

"""Standalone OPD comparison inference for the Wan2.1-VACE setup.

This script samples rows from the training CSV and generates three videos per
row:

1. Pretrained student model with text prompt only.
2. Pretrained/frozen teacher model with text prompt plus first-frame VACE context.
3. Fine-tuned student checkpoint with text prompt only.

Run on a single-GPU inference machine, for example:

    CUDA_VISIBLE_DEVICES=0 python test_teacher_inference.py \
        --config examples/opd/lora/wan2_vace/pexels_first_frame_context.yaml \
        --trained-checkpoint saves/<run_name>/checkpoints/checkpoint-10 \
        --num-samples 5 \
        --output-dir saves/opd_compare
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import torch
from accelerate import Accelerator
from PIL import Image

from diffusers.utils import export_to_video
from flow_factory.hparams import Arguments
from flow_factory.inference import (
    _freeze_adapter,
    _load_components_for_inference,
    _sample_to_frames,
)
from flow_factory.models.registry import get_model_adapter_class
from flow_factory.teachers.context import MEDIA_CONTEXT_KEYS
from flow_factory.utils.base import filter_kwargs
from flow_factory.utils.logger_utils import setup_logger

logger = setup_logger(__name__)

MethodName = Literal["original_student", "teacher_first_frame", "trained_student"]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for OPD comparison inference."""
    parser = argparse.ArgumentParser(description="Wan2.1-VACE OPD comparison inference")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/opd/lora/wan2_vace/pexels_first_frame_context.yaml",
        help="OPD YAML config used for training.",
    )
    parser.add_argument(
        "--trained-checkpoint",
        type=str,
        default=None,
        help=(
            "Fine-tuned student checkpoint path. If omitted, uses model.resume_path "
            "from the config."
        ),
    )
    parser.add_argument(
        "--trained-resume-type",
        choices=["lora", "full"],
        default=None,
        help="Checkpoint type for the fine-tuned model. If omitted, Flow-Factory auto-detects.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Override dataset CSV path. Defaults to data.dataset_dir from the config.",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of rows to sample.")
    parser.add_argument(
        "--take-first",
        action="store_true",
        help="Take the first valid rows instead of random sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="saves/opd_compare",
        help="Directory to write generated videos and manifest.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Defaults to eval.seed, then train.seed from the config.",
    )
    parser.add_argument("--fps", type=int, default=16, help="Frame rate for saved mp4 files.")
    parser.add_argument("--quality", type=float, default=8.0, help="MP4 quality from 0 to 10.")
    parser.add_argument(
        "--macro-block-size",
        type=int,
        default=16,
        help="imageio macro block size. Use 1 to disable automatic resizing.",
    )
    parser.add_argument("--height", type=int, default=None, help="Override generated video height.")
    parser.add_argument("--width", type=int, default=None, help="Override generated video width.")
    parser.add_argument(
        "--num-frames", type=int, default=None, help="Override generated frame count."
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override number of denoising steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override student/trained model CFG scale.",
    )
    parser.add_argument(
        "--teacher-guidance-scale",
        type=float,
        default=None,
        help="Override teacher CFG scale. Defaults to train.teacher_guidance_scale.",
    )
    parser.add_argument(
        "--conditioning-scale",
        type=float,
        default=None,
        help="Override teacher VACE conditioning scale.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default=None,
        help="Override config.mixed_precision for inference.",
    )
    return parser.parse_args()


def _config_value(config: Arguments, key: str) -> Any:
    """Read generation defaults from eval args first, then training args."""
    value = getattr(config.eval_args, key, None)
    if value is not None:
        return value
    return getattr(config.training_args, key, None)


def _resolve_arg(config: Arguments, args: argparse.Namespace, key: str) -> Any:
    """Resolve a CLI override or fall back to config generation defaults."""
    cli_value = getattr(args, key)
    return cli_value if cli_value is not None else _config_value(config, key)


def _resolve_seed(config: Arguments, args: argparse.Namespace) -> int:
    """Resolve random seed for CSV sampling and generation."""
    if args.seed is not None:
        return args.seed
    if config.eval_args.seed is not None:
        return config.eval_args.seed
    return config.training_args.seed


def resolve_first_frame_key(config: Arguments) -> str:
    """Return the configured media context key used for the first-frame image."""
    media_keys = [
        key for key in config.training_args.teacher_context_keys if key in MEDIA_CONTEXT_KEYS
    ]
    if not media_keys:
        raise ValueError(
            "No media context key found in train.teacher_context_keys. Expected a first-frame "
            f"key such as 'first_frame_path'. Got: {config.training_args.teacher_context_keys}."
        )
    if len(media_keys) > 1:
        logger.warning("Multiple media context keys found %s; using the first one.", media_keys)
    return media_keys[0]


def resolve_image_base_dir(config: Arguments, csv_path: str) -> str:
    """Mirror OPDTeacher's base-dir logic for resolving relative first-frame paths."""
    if config.data_args.image_dir is not None:
        return os.path.expanduser(config.data_args.image_dir)
    return os.path.dirname(os.path.expanduser(csv_path))


def resolve_image_path(value: str, base_dir: str) -> str:
    """Resolve one first-frame path against the dataset base directory."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a non-empty first-frame path string, got {value!r}.")
    path = os.path.expanduser(value.strip())
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def _set_pretrained_full_model(model_args: Any) -> None:
    """Disable fine-tuning wrappers so an adapter loads the raw pretrained model."""
    model_args.finetune_type = "full"
    model_args.target_components = []
    model_args.resume_path = None
    model_args.resume_type = None


def build_original_student_config(config: Arguments) -> Arguments:
    """Build config for pretrained student text-only inference."""
    student_config = deepcopy(config)
    _set_pretrained_full_model(student_config.model_args)
    return student_config


def build_trained_student_config(
    config: Arguments,
    checkpoint: Optional[str],
    resume_type: Optional[str],
) -> Arguments:
    """Build config for fine-tuned student text-only inference."""
    trained_config = deepcopy(config)
    resolved_checkpoint = checkpoint or trained_config.model_args.resume_path
    if resolved_checkpoint is None:
        raise ValueError(
            "Provide --trained-checkpoint or set model.resume_path in the config to generate "
            "with the fine-tuned student."
        )
    trained_config.model_args.resume_path = os.path.expanduser(resolved_checkpoint)
    if resume_type is not None:
        trained_config.model_args.resume_type = resume_type
    if trained_config.model_args.resume_type == "full":
        trained_config.model_args.finetune_type = "full"
    elif trained_config.model_args.resume_type == "lora":
        trained_config.model_args.finetune_type = "lora"
    return trained_config


def build_teacher_config(config: Arguments) -> Arguments:
    """Build config for frozen teacher first-frame inference."""
    teacher_args = config.teacher_args
    if teacher_args is None:
        raise ValueError("The config has no `teacher:` block; cannot run teacher comparison.")
    if teacher_args.model_name_or_path is None:
        raise ValueError("`teacher.model_name_or_path` must be set in the config.")

    teacher_config = deepcopy(config)
    teacher_config.model_args = deepcopy(teacher_args)
    _set_pretrained_full_model(teacher_config.model_args)
    return teacher_config


def load_adapter_for_inference(
    config: Arguments,
    accelerator: Accelerator,
    label: str,
    components: Optional[List[str]] = None,
) -> Any:
    """Load one adapter, move inference components to device, and freeze it."""
    adapter_cls = get_model_adapter_class(config.model_args.model_type)
    logger.info(
        "Loading %s adapter: %s (%s)",
        label,
        config.model_args.model_type,
        config.model_args.model_name_or_path,
    )
    adapter = adapter_cls(config=config, accelerator=accelerator)
    components_to_load = components or list(adapter.preprocessing_modules) + list(
        adapter.inference_modules
    )
    _load_components_for_inference(
        adapter=adapter,
        components=components_to_load,
        device=accelerator.device,
    )
    _freeze_adapter(adapter)
    adapter.eval()
    return adapter


def release_cuda_memory() -> None:
    """Clear Python and CUDA caches after a large adapter is deleted."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _valid_text(value: Any) -> bool:
    """Return whether a CSV value is usable prompt text."""
    return isinstance(value, str) and bool(value.strip())


def load_samples(
    csv_path: str,
    prompt_column: str,
    first_frame_key: str,
    image_base_dir: str,
    num_samples: int,
    take_first: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    """Read the CSV and return prompt/first-frame records."""
    if num_samples <= 0:
        raise ValueError(f"--num-samples must be positive, got {num_samples}.")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}.")

    dataframe = pd.read_csv(csv_path)
    for required_column in (prompt_column, first_frame_key):
        if required_column not in dataframe.columns:
            raise KeyError(
                f"Column {required_column!r} missing from CSV {csv_path}. "
                f"Available columns: {list(dataframe.columns)}."
            )

    dataframe = dataframe.reset_index(names="csv_index")
    valid_mask = dataframe[prompt_column].map(_valid_text) & dataframe[first_frame_key].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    filtered = dataframe[valid_mask]
    if len(filtered) < num_samples:
        raise ValueError(
            f"Only {len(filtered)} valid row(s) found in {csv_path}, but --num-samples={num_samples}."
        )
    selected = (
        filtered.head(num_samples)
        if take_first
        else filtered.sample(n=num_samples, random_state=seed)
    )

    records = []
    for sample_id, (_, row) in enumerate(selected.reset_index(drop=True).iterrows()):
        first_frame_path = resolve_image_path(row[first_frame_key], image_base_dir)
        if not os.path.exists(first_frame_path):
            raise FileNotFoundError(
                f"First-frame image not found for sampled row {row['csv_index']}: {first_frame_path}."
            )
        records.append(
            {
                "sample_id": sample_id,
                "csv_index": int(row["csv_index"]),
                "prompt": row[prompt_column].strip(),
                "first_frame_path": first_frame_path,
            }
        )
    return records


def build_common_inference_kwargs(
    config: Arguments,
    args: argparse.Namespace,
    prompt: str,
    generator: torch.Generator,
    guidance_scale: float,
) -> Dict[str, Any]:
    """Build common adapter.inference kwargs."""
    kwargs = {
        "prompt": [prompt],
        "height": _resolve_arg(config, args, "height"),
        "width": _resolve_arg(config, args, "width"),
        "num_frames": _resolve_arg(config, args, "num_frames"),
        "num_inference_steps": _resolve_arg(config, args, "num_inference_steps"),
        "guidance_scale": guidance_scale,
        "generator": generator,
        "compute_log_prob": False,
        "trajectory_indices": None,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def save_sample_video(sample: Any, output_path: str, args: argparse.Namespace) -> None:
    """Save one generated sample as mp4."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    frames = _sample_to_frames(sample)
    export_to_video(
        frames,
        output_video_path=output_path,
        fps=args.fps,
        quality=args.quality,
        macro_block_size=args.macro_block_size,
    )


def generate_text_only(
    adapter: Any,
    records: List[Dict[str, Any]],
    config: Arguments,
    args: argparse.Namespace,
    seed: int,
    method_name: MethodName,
    guidance_scale: float,
) -> Dict[int, str]:
    """Generate one text-only video per record."""
    outputs = {}
    with torch.no_grad(), adapter.accelerator.autocast():
        for record in records:
            generator = torch.Generator(device=adapter.device).manual_seed(
                seed + record["sample_id"]
            )
            inference_kwargs = build_common_inference_kwargs(
                config=config,
                args=args,
                prompt=record["prompt"],
                generator=generator,
                guidance_scale=guidance_scale,
            )
            inference_kwargs = filter_kwargs(adapter.inference, **inference_kwargs)
            logger.info(
                "[%s] sample %02d csv_index=%d prompt=%s",
                method_name,
                record["sample_id"],
                record["csv_index"],
                record["prompt"][:80],
            )
            generated = adapter.inference(**inference_kwargs)
            output_path = os.path.join(
                args.output_dir,
                f"sample_{record['sample_id']:02d}_{method_name}.mp4",
            )
            save_sample_video(generated[0], output_path, args)
            outputs[record["sample_id"]] = output_path
    return outputs


def generate_teacher_first_frame(
    adapter: Any,
    records: List[Dict[str, Any]],
    config: Arguments,
    args: argparse.Namespace,
    seed: int,
    guidance_scale: float,
    conditioning_scale: float,
) -> Dict[int, str]:
    """Generate one teacher first-frame-conditioned video per record."""
    outputs = {}
    height = _resolve_arg(config, args, "height")
    width = _resolve_arg(config, args, "width")
    num_frames = _resolve_arg(config, args, "num_frames")
    negative_prompt: Optional[str] = config.teacher_args.negative_prompt

    if not hasattr(adapter, "prepare_teacher_conditioning"):
        raise TypeError(
            f"Teacher adapter {type(adapter).__name__} does not expose `prepare_teacher_conditioning`."
        )

    with torch.no_grad(), adapter.accelerator.autocast():
        for record in records:
            with Image.open(record["first_frame_path"]) as first_frame:
                image = first_frame.convert("RGB")
            generator = torch.Generator(device=adapter.device).manual_seed(
                seed + record["sample_id"]
            )
            conditioning = adapter.prepare_teacher_conditioning(
                images=[image],
                latents=torch.empty(0, device=adapter.device),
                height=height,
                width=width,
                num_frames=num_frames,
                generator=generator,
                conditioning_scale=conditioning_scale,
            )
            inference_kwargs = build_common_inference_kwargs(
                config=config,
                args=args,
                prompt=record["prompt"],
                generator=generator,
                guidance_scale=guidance_scale,
            )
            if negative_prompt is not None:
                inference_kwargs["negative_prompt"] = [negative_prompt]
            inference_kwargs.update(
                {
                    "control_hidden_states": conditioning["control_hidden_states"],
                    "control_hidden_states_scale": conditioning["control_hidden_states_scale"],
                }
            )
            inference_kwargs = filter_kwargs(adapter.inference, **inference_kwargs)

            logger.info(
                "[teacher_first_frame] sample %02d csv_index=%d prompt=%s",
                record["sample_id"],
                record["csv_index"],
                record["prompt"][:80],
            )
            generated = adapter.inference(**inference_kwargs)
            output_path = os.path.join(
                args.output_dir,
                f"sample_{record['sample_id']:02d}_teacher_first_frame.mp4",
            )
            save_sample_video(generated[0], output_path, args)
            outputs[record["sample_id"]] = output_path
    return outputs


def save_manifest(
    records: List[Dict[str, Any]], outputs: Dict[str, Dict[int, str]], args: argparse.Namespace
) -> str:
    """Write a manifest with prompts, first-frame paths, and output videos."""
    manifest_records = []
    for record in records:
        sample_outputs = {
            method_name: method_outputs.get(record["sample_id"])
            for method_name, method_outputs in outputs.items()
        }
        manifest_records.append({**record, "outputs": sample_outputs})

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    """Generate original/teacher/trained comparison videos."""
    args = parse_args()
    config = Arguments.load_from_yaml(args.config)
    if args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision

    seed = _resolve_seed(config, args)
    csv_path = (
        args.csv if args.csv is not None else os.path.expanduser(config.data_args.dataset_dir)
    )
    prompt_column = getattr(config.data_args, "prompt_column", None) or "prompt"
    first_frame_key = resolve_first_frame_key(config)
    image_base_dir = resolve_image_base_dir(config, csv_path)

    records = load_samples(
        csv_path=csv_path,
        prompt_column=prompt_column,
        first_frame_key=first_frame_key,
        image_base_dir=image_base_dir,
        num_samples=args.num_samples,
        take_first=args.take_first,
        seed=seed,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    for record in records:
        first_frame_copy = os.path.join(
            args.output_dir, f"sample_{record['sample_id']:02d}_first_frame.png"
        )
        with Image.open(record["first_frame_path"]) as first_frame:
            first_frame.convert("RGB").save(first_frame_copy)
        record["first_frame_copy"] = first_frame_copy

    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    student_guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None else config.eval_args.guidance_scale
    )
    teacher_guidance_scale = (
        args.teacher_guidance_scale
        if args.teacher_guidance_scale is not None
        else config.training_args.teacher_guidance_scale
    )
    conditioning_scale = (
        args.conditioning_scale
        if args.conditioning_scale is not None
        else config.teacher_args.extra_kwargs.get("conditioning_scale", 1.0)
    )

    outputs: Dict[str, Dict[int, str]] = {}

    original_config = build_original_student_config(config)
    original_adapter = load_adapter_for_inference(original_config, accelerator, "original_student")
    outputs["original_student"] = generate_text_only(
        adapter=original_adapter,
        records=records,
        config=original_config,
        args=args,
        seed=seed,
        method_name="original_student",
        guidance_scale=student_guidance_scale,
    )
    del original_adapter
    release_cuda_memory()

    teacher_config = build_teacher_config(config)
    teacher_adapter = load_adapter_for_inference(teacher_config, accelerator, "teacher_first_frame")
    outputs["teacher_first_frame"] = generate_teacher_first_frame(
        adapter=teacher_adapter,
        records=records,
        config=teacher_config,
        args=args,
        seed=seed,
        guidance_scale=teacher_guidance_scale,
        conditioning_scale=conditioning_scale,
    )
    del teacher_adapter
    release_cuda_memory()

    trained_config = build_trained_student_config(
        config=config,
        checkpoint=args.trained_checkpoint,
        resume_type=args.trained_resume_type,
    )
    trained_adapter = load_adapter_for_inference(trained_config, accelerator, "trained_student")
    outputs["trained_student"] = generate_text_only(
        adapter=trained_adapter,
        records=records,
        config=trained_config,
        args=args,
        seed=seed,
        method_name="trained_student",
        guidance_scale=student_guidance_scale,
    )
    del trained_adapter
    release_cuda_memory()

    manifest_path = save_manifest(records, outputs, args)
    logger.info("Done. Generated %d comparison triplets in %s", len(records), args.output_dir)
    logger.info("Manifest saved to %s", manifest_path)


if __name__ == "__main__":
    main()
