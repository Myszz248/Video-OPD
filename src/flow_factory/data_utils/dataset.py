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

# src/flow_factory/data_utils/dataset.py
import hashlib
import inspect
import json
import logging
import os
import shutil
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

import imageio.v3 as iio
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from datasets.utils.logging import disable_progress_bar
from PIL import Image
from torch.utils.data import Dataset

from ..utils.audio import load_audio
from ..utils.base import filter_kwargs, pil_image_to_tensor, tensor_to_pil_image
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__, rank_zero_only=True)


# ========================================================================================
# Protocol Definitions
# ========================================================================================


class TextEncodeCallable(Protocol):
    """Protocol for text encoding functions."""

    def __call__(self, prompt: Union[str, List[str]], **kwargs: Any) -> Dict[str, Any]: ...


class ImageEncodeCallable(Protocol):
    """Protocol for image encoding functions."""

    def __call__(
        self, image: Union[Image.Image, List[Image.Image]], **kwargs: Any
    ) -> Dict[str, Any]: ...


class VideoEncodeCallable(Protocol):
    """Protocol for video encoding functions."""

    def __call__(
        self, video: Union[List[Image.Image], List[List[Image.Image]]], **kwargs: Any
    ) -> Dict[str, Any]: ...


class PreprocessCallable(Protocol):
    """Protocol for preprocessing functions that handle multi-modal inputs."""

    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]],
        images: Optional[Union[Image.Image, List[Image.Image], List[List[Image.Image]]]],
        videos: Optional[
            Union[List[Image.Image], List[List[Image.Image]], List[List[List[Image.Image]]]]
        ],
        **kwargs: Any,
    ) -> Dict[str, Any]: ...


# ========================================================================================
# GeneralDataset Class
# ========================================================================================


class GeneralDataset(Dataset):
    """
    General-purpose dataset for multi-modal data (text, images, videos).

    Supports:
    - Loading from JSONL or TXT files
    - Optional preprocessing with caching
    - Distributed preprocessing across multiple GPUs
    - Automatic cache management and merging
    """

    OUTPUT_SCHEMA_VERSION = "metadata-json-prompt-filter-v1"

    @staticmethod
    def check_exists(dataset_dir: str, split: str) -> bool:
        """Check if dataset files exist for a given split."""
        dataset_dir = os.path.expanduser(dataset_dir)
        if os.path.isfile(dataset_dir):
            return split == "train" and os.path.splitext(dataset_dir)[1].lower() in {
                ".csv",
                ".json",
                ".jsonl",
                ".txt",
            }
        jsonl_path = os.path.join(dataset_dir, f"{split}.jsonl")
        json_path = os.path.join(dataset_dir, f"{split}.json")
        csv_path = os.path.join(dataset_dir, f"{split}.csv")
        txt_path = os.path.join(dataset_dir, f"{split}.txt")
        return (
            os.path.exists(jsonl_path)
            or os.path.exists(json_path)
            or os.path.exists(csv_path)
            or os.path.exists(txt_path)
        )

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        cache_dir: str = "~/.cache/flow_factory/datasets",
        enable_preprocess: bool = True,
        force_reprocess: bool = False,
        preprocessing_batch_size: int = 16,
        max_dataset_size: Optional[int] = None,
        preprocess_func: Optional[PreprocessCallable] = None,
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
        num_shards: Optional[int] = None,
        shard_index: Optional[int] = None,
        extra_hash_strs: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        video_dir: Optional[str] = None,
        audio_dir: Optional[str] = None,
        target_arrow_path: Optional[str] = None,
        prompt_column: str = "prompt",
        negative_prompt_column: Optional[str] = "negative_prompt",
        image_column: Optional[str] = None,
        video_column: Optional[str] = None,
        audio_column: Optional[str] = None,
        first_frame_column: Optional[str] = None,
        first_frame_aspect_ratio_tolerance: float = 0.2,
        target_resolution: Optional[tuple] = None,
        **kwargs,
    ):
        """
        Initialize GeneralDataset.

        Args:
            dataset_dir: Path to dataset directory
            split: Dataset split ('train', 'test', etc.)
            cache_dir: Directory for caching preprocessed data
            enable_preprocess: Whether to enable preprocessing
            force_reprocess: Force reprocessing even if cache exists
            preprocessing_batch_size: Batch size for preprocessing
            max_dataset_size: Limit dataset size to this many samples
            preprocess_func: Function to preprocess batches
            preprocess_kwargs: Additional kwargs for preprocess_func
            num_shards: Total number of shards for distributed preprocessing
            shard_index: Current shard index (0 to num_shards-1)
            extra_hash_strs: Extra strings concatenated into the cache
                fingerprint (e.g. model identifiers) so two runs that differ
                only in those strings get distinct caches.
            image_dir: Override for the image root directory. When ``None``,
                JSONL datasets default to ``{dataset_dir}/images`` and TXT
                datasets stay ``None`` (no image loading).
            video_dir: Override for the video root directory. Same default
                resolution as ``image_dir``, with ``{dataset_dir}/videos``.
            audio_dir: Override for the audio root directory. Same default
                resolution as ``image_dir``, with ``{dataset_dir}/audios``.
            target_arrow_path: If provided, route ``Dataset.map`` output directly
                to this Arrow file via ``cache_file_name=``. The orchestrator
                (``loader._create_or_load_dataset``) sets this so each rank's
                preprocessed bytes land at their final per-rank location and
                the main rank can metadata-merge them without re-serialization.
                When ``None``, HF falls back to its default cache path under
                ``~/.cache/huggingface/datasets`` (single-process / legacy).
            prompt_column: Raw dataset column to rename to ``prompt``.
            negative_prompt_column: Optional raw column to rename to ``negative_prompt``.
            image_column: Optional raw column to rename to ``image`` for student image conditioning.
            video_column: Optional raw column to rename to ``video`` for student video conditioning.
            audio_column: Optional raw column to rename to ``audio`` for student audio conditioning.
            first_frame_column: Optional raw column holding the teacher first-frame image path.
                When set together with ``target_resolution``, rows whose first-frame aspect ratio
                deviates from the target video aspect ratio by more than
                ``first_frame_aspect_ratio_tolerance`` are dropped before preprocessing. The column
                itself is left in place so the teacher context still resolves it. The filter only
                reads each image header (``PIL.Image.size``), never the full pixels, so it is cheap.
            first_frame_aspect_ratio_tolerance: Maximum allowed relative aspect-ratio deviation,
                measured as ``max(ar_image / ar_video, ar_video / ar_image) - 1``.
            target_resolution: Target video resolution as ``(height, width)`` used as the reference
                aspect ratio for first-frame filtering. Required when ``first_frame_column`` is set.
            **kwargs: Additional arguments (ignored)

        Note:
            ``image_dir``, ``video_dir`` and ``audio_dir`` are NOT included in
            the cache fingerprint. If your JSONL stores RELATIVE asset paths
            and you switch one of these directories between runs while
            keeping every other config bit identical, the existing cache will
            be reused with stale data. Set ``force_reprocess=True`` once after
            such a switch, or include the directory in ``extra_hash_strs``.
        """
        super().__init__()
        self.data_root = os.path.expanduser(dataset_dir)
        self.cache_dir = os.path.expanduser(cache_dir)
        self.split = split
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.image_dir = image_dir
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.prompt_column = prompt_column
        self.negative_prompt_column = negative_prompt_column
        self.image_column = image_column
        self.video_column = video_column
        self.audio_column = audio_column
        self.first_frame_column = first_frame_column
        self.first_frame_aspect_ratio_tolerance = first_frame_aspect_ratio_tolerance
        self.target_resolution = target_resolution

        if self.first_frame_column is not None:
            if self.target_resolution is None:
                raise ValueError(
                    "`first_frame_column` was set to "
                    f"{self.first_frame_column!r} but `target_resolution` is None. "
                    "First-frame aspect-ratio filtering needs the target video resolution "
                    "(height, width) to compare against."
                )
            if (
                not isinstance(self.target_resolution, (tuple, list))
                or len(self.target_resolution) != 2
            ):
                raise ValueError(
                    "`target_resolution` must be a (height, width) pair, got "
                    f"{self.target_resolution!r}."
                )
            target_height, target_width = self.target_resolution
            if (
                not isinstance(target_height, int)
                or not isinstance(target_width, int)
                or target_height <= 0
                or target_width <= 0
            ):
                raise ValueError(
                    "`target_resolution` must contain two positive integers (height, width), got "
                    f"{self.target_resolution!r}."
                )
            if (
                not isinstance(self.first_frame_aspect_ratio_tolerance, (int, float))
                or self.first_frame_aspect_ratio_tolerance < 0
            ):
                raise ValueError(
                    "`first_frame_aspect_ratio_tolerance` must be a non-negative number, got "
                    f"{self.first_frame_aspect_ratio_tolerance!r}."
                )

        if self.shard_index is not None and self.shard_index > 0:
            disable_progress_bar()

        raw_dataset = self._load_raw_dataset()

        if max_dataset_size is not None and len(raw_dataset) > max_dataset_size:
            raw_dataset = raw_dataset.select(range(max_dataset_size))
            logger.info(f"Dataset size limited to {max_dataset_size} samples.")

        if enable_preprocess:
            self.processed_dataset = self._preprocess_dataset(
                raw_dataset=raw_dataset,
                preprocess_func=preprocess_func,
                preprocess_kwargs=preprocess_kwargs or {},
                preprocessing_batch_size=preprocessing_batch_size,
                force_reprocess=force_reprocess,
                max_dataset_size=max_dataset_size,
                extra_hash_strs=extra_hash_strs,
                target_arrow_path=target_arrow_path,
            )
        else:
            self.processed_dataset = raw_dataset
            self.merged_cache_path = None

    def _load_raw_dataset_file(self, path: str) -> HFDataset:
        """Load a raw dataset from a supported file path."""
        extension = os.path.splitext(path)[1].lower()
        if extension == ".csv":
            return load_dataset("csv", data_files=path, split="train")
        if extension in {".json", ".jsonl"}:
            return load_dataset("json", data_files=path, split="train")
        if extension == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(prompts)} prompts from {path}")
            return HFDataset.from_dict({"prompt": prompts})
        raise ValueError(
            f"Unsupported dataset file extension {extension!r}. "
            "Expected one of .csv, .json, .jsonl, or .txt."
        )

    @staticmethod
    def _rename_optional_column(
        raw_dataset: HFDataset,
        source: Optional[str],
        target: str,
        required: bool = False,
    ) -> HFDataset:
        """Rename one raw column while preserving fail-fast column validation."""
        if source is None:
            if required and target not in raw_dataset.column_names:
                raise ValueError(
                    f"Dataset must contain column {target!r}; available columns: "
                    f"{raw_dataset.column_names}."
                )
            return raw_dataset
        if source == target:
            if required and target not in raw_dataset.column_names:
                raise ValueError(
                    f"Dataset must contain column {target!r}; available columns: "
                    f"{raw_dataset.column_names}."
                )
            return raw_dataset
        if source not in raw_dataset.column_names:
            raise ValueError(
                f"Configured column {source!r} for {target!r} does not exist. "
                f"Available columns: {raw_dataset.column_names}."
            )
        if target in raw_dataset.column_names:
            raise ValueError(
                f"Cannot rename {source!r} to {target!r} because {target!r} already exists. "
                "Use the canonical column name directly or remove the duplicate raw column."
            )
        return raw_dataset.rename_column(source, target)

    def _normalize_raw_columns(self, raw_dataset: HFDataset) -> HFDataset:
        """Normalize configured raw data columns to Flow-Factory canonical names."""
        raw_dataset = self._rename_optional_column(
            raw_dataset,
            self.prompt_column,
            "prompt",
            required=True,
        )
        raw_dataset = self._rename_optional_column(
            raw_dataset,
            self.negative_prompt_column,
            "negative_prompt",
            required=False,
        )
        raw_dataset = self._rename_optional_column(
            raw_dataset,
            self.image_column,
            "image",
            required=False,
        )
        raw_dataset = self._rename_optional_column(
            raw_dataset,
            self.video_column,
            "video",
            required=False,
        )
        raw_dataset = self._rename_optional_column(
            raw_dataset,
            self.audio_column,
            "audio",
            required=False,
        )
        raw_dataset = self._filter_invalid_prompts(raw_dataset)
        return self._filter_first_frame_aspect_ratio(raw_dataset)

    @staticmethod
    def _is_valid_prompt(value: Any) -> bool:
        """Return whether a raw prompt value can be encoded as text."""
        return isinstance(value, str) and bool(value.strip())

    def _filter_invalid_prompts(self, raw_dataset: HFDataset) -> HFDataset:
        """Drop rows with missing or empty prompts after column normalization."""
        invalid_indices = [
            idx
            for idx, prompt in enumerate(raw_dataset["prompt"])
            if not self._is_valid_prompt(prompt)
        ]
        if not invalid_indices:
            return raw_dataset

        total = len(raw_dataset)
        preview = invalid_indices[:10]
        if len(invalid_indices) == total:
            raise ValueError(
                "All dataset rows have missing or empty prompts after mapping "
                f"{self.prompt_column!r} to 'prompt'. Check the dataset column configuration."
            )

        logger.warning(
            "Dropping %d/%d dataset row(s) with missing or empty prompts after mapping "
            "%r to 'prompt'. First invalid row indices: %s.",
            len(invalid_indices),
            total,
            self.prompt_column,
            preview,
        )
        return raw_dataset.filter(
            self._is_valid_prompt,
            input_columns="prompt",
            desc=f"[Filtering {self.split} invalid prompts]",
        )

    def _first_frame_base_dir(self) -> str:
        """Return the base directory used to resolve relative first-frame paths.

        Mirrors ``OPDTeacher._context_image_base_dir`` so the row dropped here is exactly
        the row the teacher would otherwise have to consume: prefer ``image_dir`` when it
        resolves, else the dataset file's parent directory (or the dataset directory).
        """
        if self.image_dir is not None:
            return os.path.expanduser(self.image_dir)
        if os.path.isfile(self.data_root):
            return os.path.dirname(self.data_root)
        return self.data_root

    def _first_frame_within_tolerance(self, value: Any) -> bool:
        """Return whether one first-frame image's aspect ratio matches the target resolution.

        Raises:
            FileNotFoundError: When a non-empty path does not resolve to an existing file.
                A configured-but-missing first frame is a configuration error, not a row to
                silently skip.
        """
        if not isinstance(value, str) or not value.strip():
            # Missing first-frame path: this sample cannot provide a teacher first-frame
            # condition, so drop it (counted/warned by the caller) rather than crash later.
            return False

        path = _resolve_path(self._first_frame_base_dir(), os.path.expanduser(value.strip()))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"First-frame image configured via column {self.first_frame_column!r} does not "
                f"exist: {path} (raw value: {value!r})."
            )

        with Image.open(path) as image:
            width, height = image.size
        if height <= 0 or width <= 0:
            raise ValueError(
                f"First-frame image has a non-positive dimension ({width}x{height}): {path}."
            )

        target_height, target_width = self.target_resolution
        image_aspect_ratio = width / height
        target_aspect_ratio = target_width / target_height
        deviation = (
            max(
                image_aspect_ratio / target_aspect_ratio,
                target_aspect_ratio / image_aspect_ratio,
            )
            - 1.0
        )
        return deviation <= self.first_frame_aspect_ratio_tolerance

    def _filter_first_frame_aspect_ratio(self, raw_dataset: HFDataset) -> HFDataset:
        """Drop rows whose first-frame aspect ratio is too far from the target video resolution."""
        if self.first_frame_column is None or self.target_resolution is None:
            return raw_dataset

        if self.first_frame_column not in raw_dataset.column_names:
            raise ValueError(
                f"Configured `first_frame_column`={self.first_frame_column!r} does not exist. "
                f"Available columns: {raw_dataset.column_names}."
            )

        first_frame_values = raw_dataset[self.first_frame_column]
        keep_mask = [self._first_frame_within_tolerance(value) for value in first_frame_values]
        dropped = sum(1 for keep in keep_mask if not keep)
        if dropped == 0:
            return raw_dataset

        total = len(raw_dataset)
        if dropped == total:
            target_height, target_width = self.target_resolution
            raise ValueError(
                f"All {total} dataset row(s) were dropped by first-frame aspect-ratio filtering "
                f"against target resolution (height={target_height}, width={target_width}) with "
                f"tolerance {self.first_frame_aspect_ratio_tolerance}. Check the target resolution, "
                "the tolerance, or the first-frame images."
            )

        target_height, target_width = self.target_resolution
        logger.warning(
            "Dropping %d/%d %s row(s) whose first-frame aspect ratio deviates from the target "
            "video resolution (height=%d, width=%d, target_ar=%.4f) by more than %.3f.",
            dropped,
            total,
            self.split,
            target_height,
            target_width,
            target_width / target_height,
            self.first_frame_aspect_ratio_tolerance,
        )
        keep_indices = [idx for idx, keep in enumerate(keep_mask) if keep]
        return raw_dataset.select(keep_indices)

    def _load_raw_dataset(self) -> HFDataset:
        """Load raw dataset from JSONL, JSON, CSV, or TXT files."""
        if os.path.isfile(self.data_root):
            raw_dataset = self._load_raw_dataset_file(self.data_root)
            data_parent = os.path.dirname(self.data_root)
            self.image_dir = data_parent if self.image_dir is None else self.image_dir
            self.video_dir = data_parent if self.video_dir is None else self.video_dir
            self.audio_dir = data_parent if self.audio_dir is None else self.audio_dir
            return self._normalize_raw_columns(raw_dataset)

        jsonl_path = os.path.join(self.data_root, f"{self.split}.jsonl")
        json_path = os.path.join(self.data_root, f"{self.split}.json")
        csv_path = os.path.join(self.data_root, f"{self.split}.csv")
        txt_path = os.path.join(self.data_root, f"{self.split}.txt")

        if os.path.exists(jsonl_path):
            raw_dataset = load_dataset("json", data_files=jsonl_path, split="train")
            self.image_dir = (
                os.path.join(self.data_root, "images") if self.image_dir is None else self.image_dir
            )
            self.video_dir = (
                os.path.join(self.data_root, "videos") if self.video_dir is None else self.video_dir
            )
            self.audio_dir = (
                os.path.join(self.data_root, "audios") if self.audio_dir is None else self.audio_dir
            )
        elif os.path.exists(json_path):
            raw_dataset = load_dataset("json", data_files=json_path, split="train")
            self.image_dir = (
                os.path.join(self.data_root, "images") if self.image_dir is None else self.image_dir
            )
            self.video_dir = (
                os.path.join(self.data_root, "videos") if self.video_dir is None else self.video_dir
            )
            self.audio_dir = (
                os.path.join(self.data_root, "audios") if self.audio_dir is None else self.audio_dir
            )
        elif os.path.exists(csv_path):
            raw_dataset = load_dataset("csv", data_files=csv_path, split="train")
            self.image_dir = (
                os.path.join(self.data_root, "images") if self.image_dir is None else self.image_dir
            )
            self.video_dir = (
                os.path.join(self.data_root, "videos") if self.video_dir is None else self.video_dir
            )
            self.audio_dir = (
                os.path.join(self.data_root, "audios") if self.audio_dir is None else self.audio_dir
            )
        elif os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            raw_dataset = HFDataset.from_dict({"prompt": prompts})
            self.image_dir = None if self.image_dir is None else self.image_dir
            self.video_dir = None if self.video_dir is None else self.video_dir
            self.audio_dir = None if self.audio_dir is None else self.audio_dir
            logger.info(f"Loaded {len(prompts)} prompts from {txt_path}")
        else:
            raise FileNotFoundError(
                f"Could not find {jsonl_path}, {json_path}, {csv_path}, or {txt_path}"
            )

        return self._normalize_raw_columns(raw_dataset)

    def _preprocess_dataset(
        self,
        raw_dataset: HFDataset,
        preprocess_func: PreprocessCallable,
        preprocess_kwargs: Dict[str, Any],
        preprocessing_batch_size: int,
        force_reprocess: bool,
        max_dataset_size: Optional[int],
        extra_hash_strs: Optional[List[str]] = None,
        target_arrow_path: Optional[str] = None,
    ) -> HFDataset:
        """Apply preprocessing to raw dataset with caching.

        Args:
            target_arrow_path: If set, ``map()`` writes its Arrow output directly
                to this file via ``cache_file_name=`` (and reads it back on a
                cache hit). When ``None``, HF derives a path under its own
                ``~/.cache/huggingface/datasets`` cache (legacy behavior).

        Returns:
            Preprocessed HuggingFace Dataset.
        """
        self._preprocess_func = preprocess_func
        self._preprocess_kwargs = preprocess_kwargs

        self.merged_cache_path = self.compute_cache_path(
            dataset_dir=self.data_root,
            split=self.split,
            cache_dir=self.cache_dir,
            max_dataset_size=max_dataset_size,
            preprocess_func=preprocess_func,
            preprocess_kwargs=preprocess_kwargs,
            extra_hash_strs=extra_hash_strs,
        )

        if self.num_shards and self.num_shards > 1:
            if self.shard_index is None:
                raise ValueError(
                    f"shard_index must be set when num_shards > 1, "
                    f"got num_shards={self.num_shards}, shard_index=None"
                )
            raw_dataset = self._shard_dataset(raw_dataset, self.shard_index, self.num_shards)
            shard_fingerprint = (
                f"{os.path.basename(self.merged_cache_path)}"
                f"{self._shard_suffix(self.shard_index, self.num_shards)}"
            )
            # Display convention matches :meth:`_shard_suffix`: the second
            # number is the last shard index (``num_shards - 1``), not the total.
            desc = (
                f"[Preprocessing {self.split} dataset] "
                f"Shard {self.shard_index:04d}/{self.num_shards - 1:04d}"
            )
        else:
            shard_fingerprint = os.path.basename(self.merged_cache_path)
            desc = f"[Preprocessing {self.split} dataset]"

        os.makedirs(self.cache_dir, exist_ok=True)
        if target_arrow_path is not None:
            os.makedirs(os.path.dirname(target_arrow_path), exist_ok=True)

        processed_dataset = raw_dataset.map(
            self._preprocess_batch,
            batched=True,
            batch_size=preprocessing_batch_size,
            fn_kwargs={
                "image_dir": self.image_dir,
                "video_dir": self.video_dir,
                "audio_dir": self.audio_dir,
            },
            remove_columns=raw_dataset.column_names,
            new_fingerprint=shard_fingerprint,
            cache_file_name=target_arrow_path,
            desc=desc,
            load_from_cache_file=not force_reprocess,
        )

        try:
            processed_dataset.set_format(type="torch", columns=processed_dataset.column_names)
        except Exception:
            pass

        return processed_dataset

    def _shard_dataset(self, dataset: HFDataset, shard_index: int, num_shards: int) -> HFDataset:
        """
        Split dataset into shards for distributed preprocessing.

        Args:
            dataset: Full dataset to shard
            shard_index: Index of current shard (0 to num_shards-1)
            num_shards: Total number of shards

        Returns:
            Sharded subset of the dataset
        """
        shard_size = len(dataset) // num_shards
        start_idx = shard_index * shard_size
        end_idx = start_idx + shard_size if shard_index < num_shards - 1 else len(dataset)
        return dataset.select(range(start_idx, end_idx))

    def _preprocess_batch(
        self,
        batch: Dict[str, Any],
        image_dir: Optional[str],
        video_dir: Optional[str],
        audio_dir: Optional[str],
    ) -> Dict[str, Any]:
        """
        Preprocess a batch of samples.

        Workflow:
            1. Prepare prompt inputs (text)
            2. Load and prepare image inputs
            3. Load and prepare video inputs
            4. Load and prepare audio inputs
            5. Call preprocess function
            6. Move result tensors to CPU for caching
            7. Pack non-preprocessed columns into ``metadata``

        Args:
            batch: Dictionary with batch data.
            image_dir: Directory containing images (``None`` skips image loading).
                Per-sample paths are loaded as PIL Images and kept as a
                ``List[Image]``; the column-level ``images`` field is therefore
                always a ``MultiImageBatch`` of shape ``List[List[Image]]`` —
                single-image samples produce ``[Image]`` and empty samples
                produce ``[]``.
            video_dir: Directory containing videos (``None`` skips video loading).
                Same shape as ``image_dir``: column-level ``videos`` is a
                ``MultiVideoBatch`` (``List[List[VideoFrames]]``).
            audio_dir: Directory containing audio files (``None`` skips audio
                loading). Each per-sample list of paths is loaded via
                :func:`flow_factory.utils.audio.load_audio` and stored as a
                ``List[torch.Tensor]`` (one Tensor per audio clip), so the
                column-level ``audios`` field is always a ``MultiAudioBatch``
                of shape ``List[List[Tensor]]`` — single-audio samples produce
                ``[Tensor]`` and empty samples produce ``[]``.

        Returns:
            Dictionary with preprocessed data, plus an additional ``metadata``
            list carrying every non-preprocess column from ``batch``.

        Note:
            The ``[]``-for-empty contract is what keeps every column length
            equal to the input batch size, which HF ``Dataset.map(batched=True)``
            requires. Mixing in ``None`` or unwrapping single-element lists to a
            bare ``Tensor`` breaks Arrow's homogeneous-column requirement and
            forces every downstream consumer to handle three input shapes.
        """
        assert self._preprocess_func is not None, "Preprocess function must be provided."
        # The keys that are used in preprocess and maintained in the final results.
        PREPROCESS_KEYS = ("prompt", "negative_prompt", "images", "videos", "audios")

        # 1. Prepare prompt inputs (text)
        prompt = batch["prompt"]
        negative_prompt = batch.get("negative_prompt", None)
        prompt_args = {"prompt": prompt}
        if negative_prompt is not None:
            prompt_args["negative_prompt"] = negative_prompt

        # 2. Prepare image inputs (only when image_dir exists and batch has images)
        if "image" in batch:
            batch["images"] = batch.pop("image")  # Rename for consistency

        image_args = {"images": None}
        if image_dir is not None and "images" in batch:
            img_paths_list = batch["images"]
            batch["images"] = []  # Clear
            image_args["images"] = []
            for img_paths in img_paths_list:
                if not img_paths:
                    # Empty sample contributes [] to both args and batch so the
                    # column stays a homogeneous List[List[...]] (MultiImageBatch)
                    # and HF.map(batched=True) sees matching column lengths.
                    image_args["images"].append([])
                    batch["images"].append([])
                else:
                    if isinstance(img_paths, str):
                        img_paths = [img_paths]
                    images = [
                        Image.open(_resolve_path(image_dir, img_path)).convert("RGB")
                        for img_path in img_paths
                    ]
                    image_pts = [pil_image_to_tensor(img)[0] for img in images]
                    image_args["images"].append(images)
                    batch["images"].append(image_pts)

        # 3. Prepare video inputs (only when video_dir exists and batch has videos)
        if "video" in batch:
            batch["videos"] = batch.pop("video")  # Rename for consistency

        video_args = {"videos": None}
        if video_dir is not None and "videos" in batch:
            video_paths_list = batch["videos"]
            batch["videos"] = []  # Clear
            video_args["videos"] = []
            for video_paths in video_paths_list:
                if not video_paths:
                    # Empty sample contributes [] to both args and batch so the
                    # column stays a homogeneous List[List[...]] (MultiVideoBatch)
                    # and HF.map(batched=True) sees matching column lengths.
                    video_args["videos"].append([])
                    batch["videos"].append([])
                else:
                    if isinstance(video_paths, str):
                        video_paths = [video_paths]

                    videos = [
                        load_video_frames(_resolve_path(video_dir, video_path))
                        for video_path in video_paths
                    ]
                    video_pts = [pil_image_to_tensor(video) for video in videos]
                    video_args["videos"].append(videos)
                    batch["videos"].append(video_pts)

        # 4. Prepare audio inputs (only when audio_dir exists and batch has audios)
        if "audio" in batch:
            batch["audios"] = batch.pop("audio")  # Rename for consistency

        audio_args = {"audios": None}
        if audio_dir is not None and "audios" in batch:
            audio_paths_list = batch["audios"]
            batch["audios"] = []  # Clear
            audio_args["audios"] = []
            for audio_paths in audio_paths_list:
                if not audio_paths:
                    # Empty sample contributes [] to both args and batch so the
                    # column stays a homogeneous List[List[Tensor]] (MultiAudioBatch)
                    # and HF.map(batched=True) sees matching column lengths.
                    audio_args["audios"].append([])
                    batch["audios"].append([])
                else:
                    if isinstance(audio_paths, str):
                        audio_paths = [audio_paths]
                    audios = [
                        load_audio(_resolve_path(audio_dir, audio_path))
                        for audio_path in audio_paths
                    ]
                    # Always store as List[Tensor] (no single-audio unwrap) so
                    # downstream encode_audio sees a uniform type within the batch.
                    audio_args["audios"].append(audios)
                    batch["audios"].append(audios)

        # 5. Call preprocess function with filtered kwargs
        input_args = {
            **prompt_args,
            **image_args,
            **video_args,
            **audio_args,
            **self._preprocess_kwargs,
        }
        filtered_args = filter_kwargs(self._preprocess_func, **input_args)
        preprocess_res = self._preprocess_func(**filtered_args)

        # 6. Process results - move tensors to CPU for caching
        final_res = {}
        for k, v in preprocess_res.items():
            if isinstance(v, torch.Tensor):
                # Case A: Dense Batch Tensor
                # Move entire batch to CPU first (faster than moving slices), then unbind
                final_res[k] = list(torch.unbind(v.cpu(), dim=0))
            elif isinstance(v, list):
                # Case B: Ragged List (e.g. Flux image latents of varying sizes,
                # or nested lists like List[List[Tensor]] for multi-ref condition images)
                final_res[k] = [_move_to_cpu(x) for x in v]
            else:
                # Case C: Other types (None, int, etc)
                final_res[k] = v

        # 7. Prepare final results. Keep only canonical training inputs at the
        # top level; arbitrary raw CSV/JSON columns are stored as JSON metadata
        # to avoid PyArrow inferring unstable nested null schemas.
        retained_batch = {k: v for k, v in batch.items() if k in PREPROCESS_KEYS}
        batch_dict = {**retained_batch, **final_res}
        batch_dict["metadata"] = [
            json.dumps(
                {
                    k: _json_safe_metadata_value(v[idx])
                    for k, v in batch.items()
                    if k not in PREPROCESS_KEYS
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for idx in range(len(batch["prompt"]))
        ]

        return batch_dict

    @classmethod
    def load_merged(cls, merged_cache_path: str) -> "GeneralDataset":
        """
        Load preprocessed dataset from merged cache.

        Args:
            merged_cache_path: Path to merged cache directory

        Returns:
            GeneralDataset instance with loaded data
        """
        instance = cls.__new__(cls)
        instance.processed_dataset = load_from_disk(merged_cache_path)
        try:
            instance.processed_dataset.set_format(
                type="torch", columns=instance.processed_dataset.column_names
            )
        except Exception:
            pass
        return instance

    @staticmethod
    def compute_cache_path(
        dataset_dir: str,
        split: str,
        cache_dir: str,
        max_dataset_size: Optional[int],
        preprocess_func: Optional[Callable],
        preprocess_kwargs: Optional[Dict[str, Any]],
        extra_hash_strs: Optional[List[str]] = None,
        digits: int = 32,
    ) -> str:
        """Compute merged cache path by hashing all components.

        ``kwargs_hash`` is computed via *deep signature collection*: the set
        of relevant keys is the union of named parameters from
        ``preprocess_func`` and (when ``preprocess_func`` accepts ``**kwargs``
        and is a bound adapter method) all ``encode_*`` methods on the same
        adapter instance.  Keys outside this union (e.g.
        ``num_batches_per_epoch``, ``gradient_accumulation_steps``) are
        excluded — they are training-infrastructure fields that do not affect
        preprocessing output.

        To force a value into the cache key without adding it to any function
        signature, include it in ``extra_hash_strs``.

        Args:
            digits: Length of hash fingerprint (default: 32, max: 32)

        Returns:
            Cache path with fingerprint of specified length
        """
        dataset_name = os.path.basename(dataset_dir)
        cutoff_str = str(max_dataset_size) if max_dataset_size else "full"
        funcs_hash = _compute_encode_funcs_hash(preprocess_func, digits=16)
        hashable_kwargs = _select_cache_relevant_kwargs(preprocess_func, preprocess_kwargs)
        kwargs_hash = hashlib.md5(str(sorted(hashable_kwargs.items())).encode()).hexdigest()[:16]
        extra_hash = "|".join(extra_hash_strs) if extra_hash_strs else ""

        combined = f"{dataset_name}|{split}|{cutoff_str}|{funcs_hash}|{kwargs_hash}|{extra_hash}"
        fingerprint = hashlib.md5(combined.encode()).hexdigest()[: min(digits, 32)]

        logger.debug(
            "compute_cache_path: dataset=%s split=%s cutoff=%s funcs=%s kwargs=%s "
            "extra=%s hashable_keys=%s -> %s",
            dataset_name,
            split,
            cutoff_str,
            funcs_hash,
            kwargs_hash,
            extra_hash,
            sorted(hashable_kwargs),
            fingerprint,
        )
        return os.path.join(os.path.expanduser(cache_dir), fingerprint)

    @staticmethod
    def _shard_suffix(shard_idx: int, num_shards: int) -> str:
        """Per-rank suffix ``_shard{X:04d}of{Y:04d}`` where ``Y = num_shards - 1``.

        IMPORTANT: ``Y`` is the *last* shard index (inclusive), **not** the
        total shard count. The rank range covered is ``[0, Y]``, i.e.
        ``num_shards`` ranks in total. Example::

            _shard_suffix(shard_idx=0, num_shards=4) -> "_shard0000of0003"
            _shard_suffix(shard_idx=3, num_shards=4) -> "_shard0003of0003"

        Shared by:

          * the Arrow filename embedded in :meth:`build_part_arrow_path`
          * the HF ``Dataset.map(new_fingerprint=...)`` string in
            :meth:`_preprocess_dataset`

        Changing the format here keeps both in lockstep; no caller of this class
        may hand-craft the suffix.
        """
        return f"_shard{shard_idx:04d}of{num_shards - 1:04d}"

    @staticmethod
    def build_part_arrow_path(merged_cache_path: str, shard_idx: int, num_shards: int) -> str:
        """Deterministic per-rank Arrow file path inside ``{merged_cache_path}.tmp``.

        Single source of truth for the per-rank cache file layout. Called by:

          * the writer (each rank's ``Dataset.map(cache_file_name=...)`` target
            in :func:`flow_factory.data_utils.loader._create_or_load_dataset`)
          * :meth:`consolidate_parts` (reconstructs every rank's path to build
            the merged dataset's ``state.json``)

        Layout::

            {merged_cache_path}.tmp/_parts/rank_{X:04d}_of_{N:04d}/
                cache-{basename}{_shard_suffix(X, N)}.arrow

        Args:
            merged_cache_path: Final merged-cache directory (without the
                ``.tmp`` suffix). A leading ``~`` is expanded internally; no
                other normalization is applied, so the return value is
                absolute iff ``merged_cache_path`` is absolute after
                ``expanduser``. ``{merged_cache_path}.tmp`` is the build dir.
            shard_idx: Shard index (``0 <= shard_idx < num_shards``).
            num_shards: Total number of shards participating in preprocessing.

        Returns:
            Path to this rank's Arrow file inside the build directory.
        """
        merged_cache_path = os.path.expanduser(merged_cache_path)
        build_dir = merged_cache_path + ".tmp"
        merged_fp = os.path.basename(merged_cache_path)
        return os.path.join(
            build_dir,
            "_parts",
            f"rank_{shard_idx:04d}_of_{num_shards:04d}",
            f"cache-{merged_fp}{GeneralDataset._shard_suffix(shard_idx, num_shards)}.arrow",
        )

    @classmethod
    def consolidate_parts(
        cls,
        merged_cache_path: str,
        num_shards: int,
        split: Optional[str] = None,
    ) -> None:
        """Promote per-rank Arrow files into a valid HF dataset directory without copying data.

        Builds the top-level ``state.json`` and ``dataset_info.json`` that turn the
        directory ``merged_cache_path + ".tmp"`` (which already contains every rank's
        Arrow file under ``_parts/rank_*/``) into a structure ``load_from_disk`` can
        read, then atomically renames ``.tmp`` -> ``merged_cache_path``. No row data
        is re-serialized: each shard's bytes stay where ``Dataset.map(cache_file_name=...)``
        wrote them.

        Paths of the ``num_shards`` per-rank Arrow files are derived via
        :meth:`build_part_arrow_path`, so the writer and the consolidator cannot
        drift.

        Args:
            merged_cache_path: Final destination directory. The function reads from
                ``merged_cache_path + ".tmp"`` and renames it to this path on success.
                A leading ``~`` is expanded internally to keep ``build_dir`` and
                :meth:`build_part_arrow_path` outputs on the same form (otherwise
                ``os.path.relpath`` would cross forms and produce bogus prefixes).
            num_shards: Total number of per-rank Arrow files expected under the
                build directory, in rank order. Listed in the produced
                ``state.json`` as ``_data_files`` (relative to ``merged_cache_path``);
                ``load_from_disk`` will memory-map them in this order.
            split: Optional split tag stored as ``state["_split"]`` (round-trips to
                ``dataset.split`` after ``load_from_disk``).

        Raises:
            FileNotFoundError: If the build directory or any expected per-rank Arrow
                file is missing. The message includes ``merged_cache_path`` and
                ``num_shards`` to make distributed debugging tractable.
        """
        merged_cache_path = os.path.expanduser(merged_cache_path)
        build_dir = merged_cache_path + ".tmp"
        if not os.path.isdir(build_dir):
            raise FileNotFoundError(
                f"expected build dir for consolidation, missing: {build_dir} "
                f"(merged_cache_path={merged_cache_path})"
            )
        part_arrow_paths = [
            cls.build_part_arrow_path(merged_cache_path, i, num_shards) for i in range(num_shards)
        ]
        for p in part_arrow_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"expected per-rank arrow file, missing: {p} "
                    f"(merged_cache_path={merged_cache_path}, "
                    f"num_shards={num_shards})"
                )

        template = HFDataset.from_file(part_arrow_paths[0])
        state = {
            "_data_files": [{"filename": os.path.relpath(p, build_dir)} for p in part_arrow_paths],
            "_fingerprint": os.path.basename(merged_cache_path),
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": split,
        }
        with open(os.path.join(build_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        dataset_info = asdict(template.info)
        with open(os.path.join(build_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump({k: dataset_info[k] for k in sorted(dataset_info)}, f, indent=2)
        if os.path.exists(merged_cache_path):
            shutil.rmtree(merged_cache_path)
        os.replace(build_dir, merged_cache_path)

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.

        Stacks tensors with same shape, keeps ragged tensors as lists.

        Args:
            batch: List of samples

        Returns:
            Collated batch dictionary
        """
        if not batch:
            return {}

        collated_batch = {}
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            # Classify value types
            is_tensor = [isinstance(v, torch.Tensor) for v in values]
            is_list = [isinstance(v, list) for v in values]

            if all(is_tensor):
                # Case 1: All elements are tensors
                shapes = [v.shape for v in values]
                if all(s == shapes[0] for s in shapes):
                    # Same shape → stack into batch tensor
                    collated_batch[key] = torch.stack(values, dim=0)
                else:
                    # Different shapes → keep as List[Tensor]
                    collated_batch[key] = values

            elif any(is_tensor) and any(is_list):
                # Case 2: Mixed tensor/list → normalize to List[List[Tensor]]
                # Handles ragged data (e.g., multi-reference images): dataset auto-stacks same-shape cases,
                # while some samples may have images of differetn shapes and are kept as List[Tensor], which is inconstent
                collated_batch[key] = [
                    list(torch.unbind(v, dim=0)) if isinstance(v, torch.Tensor) else v
                    for v in values
                ]

            else:
                # Case 3: Other types (all lists, ints, strs, etc.)
                collated_batch[key] = values

        return collated_batch


# ========================================================================================
# Utility Functions
# ========================================================================================


def _move_to_cpu(obj):
    """Recursively move tensors to CPU within nested lists."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, list):
        return [_move_to_cpu(x) for x in obj]
    return obj


def _json_safe_metadata_value(value: Any) -> Any:
    """Convert metadata values to JSON-safe scalars while preserving context paths."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_metadata_value(v) for k, v in value.items()}
    return str(value)


def _resolve_path(base_dir: str, path: str) -> str:
    """Resolve path: use as-is if absolute, otherwise join with base_dir."""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_video_frames(video_path: str, fps: Optional[int] = None) -> List[Image.Image]:
    """
    Load video frames using imageio (diffusers standard).

    Args:
        video_path: Path to video file
        fps: If specified, resample video to this frame rate

    Returns:
        List of PIL Images representing video frames
    """
    frames = [Image.fromarray(frame) for frame in iio.imread(video_path)]

    if fps is not None:
        # Uniform resampling based on target fps
        metadata = iio.immeta(video_path)
        original_fps = metadata.get("fps", 30)
        step = original_fps / fps
        indices = [int(i * step) for i in range(int(len(frames) / step))]
        frames = [frames[i] for i in indices if i < len(frames)]

    return frames


def _compute_function_hash(func: Optional[Callable], digits: int = 16) -> str:
    """
    Compute stable hash for function caching.
    For bound methods, includes class name to distinguish subclass implementations.
    """
    _MAX_DIGITS = 32
    digits = min(digits, _MAX_DIGITS)

    if func is None:
        return "none" * 4

    # Extract class context for bound methods
    class_prefix = ""
    if hasattr(func, "__self__"):
        class_name = func.__self__.__class__.__qualname__
        class_prefix = f"{class_name}:"

    try:
        # Method 1: Source code + class context
        source = inspect.getsource(func)
        source = "".join(source.split())
        combined = class_prefix + source
        return hashlib.md5(combined.encode()).hexdigest()[:digits]
    except (TypeError, OSError):
        # Method 2: Module path + class context
        try:
            module = inspect.getmodule(func)
            module_name = module.__name__ if module else "unknown"
            func_name = getattr(func, "__qualname__", getattr(func, "__name__", "anonymous"))
            signature = class_prefix + f"{module_name}.{func_name}"
            return hashlib.md5(signature.encode()).hexdigest()[:digits]
        except:
            # Method 3: Fallback with class context
            logger.warning(f"Could not compute stable hash for {func}, using id() fallback")
            signature = class_prefix + str(id(func))
            return hashlib.md5(signature.encode()).hexdigest()[:digits]


_ENCODER_METHOD_NAMES = ("encode_prompt", "encode_image", "encode_video", "encode_audio")


def _collect_named_params(func: Optional[Callable]) -> set[str]:
    """Named (non-VAR_KEYWORD / VAR_POSITIONAL) parameter names, minus ``self``."""
    if func is None:
        return set()
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return set()
    return {
        p.name
        for p in sig.parameters.values()
        if p.kind
        not in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    } - {"self"}


def _select_cache_relevant_kwargs(
    preprocess_func: Optional[Callable],
    preprocess_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the subset of *preprocess_kwargs* that can affect preprocessing output.

    Collects named parameters from:
      1. ``preprocess_func`` itself
      2. If ``preprocess_func`` accepts ``**kwargs`` AND is a bound method,
         also every ``encode_*`` method on the same adapter instance
         (``encode_prompt``, ``encode_image``, ``encode_video``,
         ``encode_audio``) — because ``BaseAdapter.preprocess_func``
         forwards its ``**kwargs`` to these methods via ``filter_kwargs``.

    The union of these parameter names becomes the key-filter. Keys not in
    the union (e.g. ``num_batches_per_epoch``, ``gradient_accumulation_steps``)
    are excluded from the cache fingerprint.

    Safety properties:
      - Over-hash is safe (worst case: unnecessary re-preprocess).
      - Under-hash is dangerous (cache corruption). This approach can only
        over-hash (includes encoder params for encoders that might not run
        at runtime), never under-hash.
      - Falls back to the full dict when signature inspection fails.

    To make a value influence the cache key without adding it to any
    function signature, pass it via ``extra_hash_strs`` instead.
    """
    kwargs = preprocess_kwargs or {}
    if preprocess_func is None or not kwargs:
        return dict(kwargs)

    relevant_keys = _collect_named_params(preprocess_func)

    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(preprocess_func).parameters.values()
    )
    if has_var_kw and hasattr(preprocess_func, "__self__"):
        adapter = preprocess_func.__self__
        for name in _ENCODER_METHOD_NAMES:
            encoder = getattr(adapter, name, None)
            if callable(encoder):
                relevant_keys |= _collect_named_params(encoder)

    if not relevant_keys:
        return dict(kwargs)

    return {k: v for k, v in kwargs.items() if k in relevant_keys}


def _compute_encode_funcs_hash(*funcs: Optional[Callable], digits: int = 16) -> str:
    """
    Compute joint hash for multiple functions.

    Ensures cache is invalidated when any preprocessing logic changes.

    Args:
        *funcs: Variable number of functions to hash
        digits: Number of hash digits to return

    Returns:
        Hexadecimal hash string representing joint hash
    """
    _MAX_DIGITS = 32
    digits = min(digits, _MAX_DIGITS)
    individual_hashes = [_compute_function_hash(func) for func in funcs]
    combined_parts = [f"func{i}:{hash_val}" for i, hash_val in enumerate(individual_hashes)]
    combined = "|".join(combined_parts)
    return hashlib.md5(combined.encode()).hexdigest()[:digits]
