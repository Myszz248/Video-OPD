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

"""Run prompt-only Wan2.1 VACE text-to-video inference with VACE LoRA weights."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
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
        description="Prompt-only Wan2.1 VACE text-to-video inference with VACE LoRA."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Local path or Hugging Face repo id for Wan2.1-VACE-1.3B-diffusers.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt. Pass an empty string to disable the default.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional VACE LoRA checkpoint path. Omit it to run the base VACE model.",
    )
    parser.add_argument(
        "--checkpoint-type",
        choices=["lora"],
        default="lora",
        help="Compatibility flag. This script supports VACE LoRA checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Compatibility flag; Flow-Factory YAML configs are not used by this VACE script.",
    )
    parser.add_argument(
        "--target-components",
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
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=16.0,
        help="Wan VACE scheduler flow shift. Official Wan VACE uses 16 by default.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
        help="Transformer/text encoder inference dtype.",
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


def resolve_lora_path(checkpoint: str) -> str:
    """Resolve a VACE LoRA checkpoint directory."""
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint))
    transformer_path = os.path.join(checkpoint_path, "transformer")
    if os.path.isdir(transformer_path):
        return transformer_path
    if os.path.isdir(checkpoint_path):
        return checkpoint_path
    raise FileNotFoundError(f"LoRA checkpoint directory not found: {checkpoint_path}")


def prepare_blank_vace_control(
    height: int,
    width: int,
    num_frames: int,
) -> tuple[List[Image.Image], List[Image.Image]]:
    """Create prompt-only VACE control inputs matching official prepare_source()."""
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


def load_pipeline(args: argparse.Namespace) -> WanVACEPipeline:
    """Load WanVACEPipeline and attach VACE LoRA weights."""
    base_model = os.path.abspath(os.path.expanduser(args.base_model))
    if not os.path.exists(base_model):
        base_model = args.base_model

    torch_dtype = get_torch_dtype(args.dtype)
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
        flow_shift=args.flow_shift,
    )

    if args.checkpoint is not None:
        lora_path = resolve_lora_path(args.checkpoint)
        print(f"Loading VACE LoRA from: {lora_path}")
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            lora_path,
            torch_dtype=torch_dtype,
            is_trainable=False,
        )
        if hasattr(pipe.transformer, "set_adapter"):
            pipe.transformer.set_adapter("default")
        diagnostics = summarize_lora_state(pipe.transformer)
        print(f"LoRA diagnostics: {json.dumps(diagnostics, ensure_ascii=False)}")
        if diagnostics["lora_tensor_count"] == 0 or diagnostics["lora_abs_sum"] == 0.0:
            raise ValueError(
                "LoRA checkpoint was loaded, but no non-zero LoRA parameters were found. "
                "Please check that --checkpoint points to the real adapter directory."
            )
    else:
        print("No --checkpoint provided; running the base VACE model only.")
    pipe.transformer.eval()

    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    return pipe


def build_metadata(
    args: argparse.Namespace,
    output_path: str,
    lora_path: Optional[str],
) -> Dict[str, Any]:
    """Build JSON-serializable inference metadata."""
    return {
        "base_model": args.base_model,
        "checkpoint": (
            os.path.abspath(os.path.expanduser(args.checkpoint))
            if args.checkpoint is not None
            else None
        ),
        "resolved_lora_path": lora_path,
        "output": output_path,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "seed": args.seed,
        "dtype": args.dtype,
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


def main() -> None:
    """Run Wan2.1 VACE prompt-only inference and save the result."""
    args = parse_args()
    output_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lora_path = resolve_lora_path(args.checkpoint) if args.checkpoint is not None else None
    pipe = load_pipeline(args)
    video, mask = prepare_blank_vace_control(args.height, args.width, args.num_frames)
    generator_device = "cpu" if args.enable_model_cpu_offload else args.device
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    if not args.enable_model_cpu_offload:
        move_scheduler_to_device(pipe, args.device)
    patch_scheduler_device_guard(pipe)

    negative_prompt: Optional[str] = args.negative_prompt
    if negative_prompt == "":
        negative_prompt = None

    with torch.no_grad():
        frames = pipe(
            video=video,
            mask=mask,
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).frames[0]

    export_to_video(
        frames,
        output_path,
        fps=args.fps,
        quality=args.quality,
        macro_block_size=args.macro_block_size,
    )
    metadata_path = save_metadata(build_metadata(args, output_path, lora_path), output_path)
    print(f"Saved video to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
