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

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ExportRecord:
    """Store one exported first-frame image together with its caption."""

    row_index: int
    source_image_path: Path
    exported_image_path: Path
    caption: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export the first N first-frame images referenced by a CSV file."
    )
    parser.add_argument("csv_path", help="Path to the source CSV file.")
    parser.add_argument("output_dir", help="Directory to save the exported first-frame images.")
    parser.add_argument(
        "--column",
        default="first_frame_path",
        help="CSV column containing first-frame image paths. Default: first_frame_path.",
    )
    parser.add_argument(
        "--caption-column",
        default="caption_short_paragraph",
        help="CSV column containing captions. Default: caption_short_paragraph.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Number of rows to export from the top of the CSV. Default: 30.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory when names collide.",
    )
    parser.add_argument(
        "--caption-file",
        default="captions.json",
        help="Caption metadata filename to write under output_dir. Default: captions.json.",
    )
    return parser.parse_args()


def resolve_image_path(csv_path: Path, raw_value: str) -> Path:
    """Resolve one image path using the same relative-path rule as OPD teacher context."""
    expanded = Path(os.path.expanduser(raw_value))
    if expanded.is_absolute():
        return expanded
    return csv_path.parent / expanded


def load_export_rows(
    csv_path: Path,
    image_column_name: str,
    caption_column_name: str,
    limit: int,
) -> List[dict]:
    """Load and validate the first ``limit`` CSV rows needed for export."""
    if limit <= 0:
        raise ValueError(f"`limit` must be positive, got {limit}.")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {csv_path}")
        if image_column_name not in reader.fieldnames:
            raise ValueError(
                f"Column {image_column_name!r} does not exist in {csv_path}. "
                f"Available columns: {reader.fieldnames}."
            )
        if caption_column_name not in reader.fieldnames:
            raise ValueError(
                f"Column {caption_column_name!r} does not exist in {csv_path}. "
                f"Available columns: {reader.fieldnames}."
            )

        rows: List[dict] = []
        for row_index, row in enumerate(reader, start=1):
            if len(rows) >= limit:
                break
            raw_image_value = row.get(image_column_name)
            if raw_image_value is None or not raw_image_value.strip():
                raise ValueError(
                    f"Row {row_index} has an empty value in column {image_column_name!r}."
                )
            raw_caption_value = row.get(caption_column_name)
            if raw_caption_value is None or not raw_caption_value.strip():
                raise ValueError(
                    f"Row {row_index} has an empty value in column {caption_column_name!r}."
                )

            image_path = resolve_image_path(csv_path, raw_image_value.strip())
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Row {row_index} references a missing first-frame image: {image_path}"
                )
            rows.append(
                {
                    "row_index": row_index,
                    "image_path": image_path,
                    "caption": raw_caption_value.strip(),
                }
            )

    if len(rows) < limit:
        raise ValueError(
            f"CSV only provides {len(rows)} usable row(s), fewer than requested limit({limit})."
        )
    return rows


def build_output_path(output_dir: Path, index: int, source_path: Path) -> Path:
    """Build a stable output filename that preserves order and extension."""
    suffix = source_path.suffix or ".png"
    stem = source_path.stem or "first_frame"
    filename = f"{index:02d}_{stem}{suffix}"
    return output_dir / filename


def export_first_frames(
    csv_path: Path,
    output_dir: Path,
    column_name: str,
    caption_column_name: str,
    limit: int,
    overwrite: bool,
) -> List[ExportRecord]:
    """Copy first-frame images and collect aligned caption metadata."""
    export_rows = load_export_rows(
        csv_path=csv_path,
        image_column_name=column_name,
        caption_column_name=caption_column_name,
        limit=limit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_records: List[ExportRecord] = []
    for index, export_row in enumerate(export_rows, start=1):
        image_path = export_row["image_path"]
        destination = build_output_path(output_dir, index=index, source_path=image_path)
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {destination}. Re-run with --overwrite to replace it."
            )
        shutil.copy2(image_path, destination)
        exported_records.append(
            ExportRecord(
                row_index=export_row["row_index"],
                source_image_path=image_path,
                exported_image_path=destination,
                caption=export_row["caption"],
            )
        )
    return exported_records


def write_caption_metadata(output_dir: Path, caption_file_name: str, records: List[ExportRecord]) -> Path:
    """Write aligned caption metadata to a JSON file under ``output_dir``."""
    caption_path = output_dir / caption_file_name
    payload = [
        {
            "index": index,
            "row_index": record.row_index,
            "image_file": record.exported_image_path.name,
            "source_image_path": str(record.source_image_path),
            "caption": record.caption,
        }
        for index, record in enumerate(records, start=1)
    ]
    with caption_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return caption_path


def main() -> None:
    """Run the first-frame export script."""
    args = parse_args()
    csv_path = Path(os.path.expanduser(args.csv_path)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    exported_records = export_first_frames(
        csv_path=csv_path,
        output_dir=output_dir,
        column_name=args.column,
        caption_column_name=args.caption_column,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    caption_path = write_caption_metadata(
        output_dir=output_dir,
        caption_file_name=args.caption_file,
        records=exported_records,
    )
    print(
        f"Exported {len(exported_records)} first-frame image(s) to {output_dir} "
        f"and wrote captions to {caption_path}"
    )


if __name__ == "__main__":
    main()
