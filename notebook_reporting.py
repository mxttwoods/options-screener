"""Shared reporting helpers for research notebooks.

Provides a consistent export workflow:
- Per-table CSV files
- Consolidated Excel workbook
- JSON manifest with metadata
- Optional ZIP share package
"""

from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return token or "artifact"


def _sanitize_sheet_name(value: str) -> str:
    # Excel sheet names are capped at 31 chars and disallow : \\ / ? * [ ]
    clean = re.sub(r"[:\\/?*\[\]]+", "_", str(value)).strip()
    clean = clean[:31]
    return clean or "Sheet1"


def _clean_frame_map(frames: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    clean: dict[str, pd.DataFrame] = {}
    for name, value in frames.items():
        if isinstance(value, pd.DataFrame) and not value.empty:
            clean[_sanitize_token(name)] = value.copy()
    return clean


def export_report_bundle(
    *,
    prefix: str,
    run_stamp: str,
    output_dir: str,
    frames: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    include_excel: bool = True,
    include_zip: bool = True,
) -> dict[str, Any]:
    """Export notebook artifacts to a timestamped run directory.

    Returns a dictionary of file paths for downstream display.
    """
    clean_frames = _clean_frame_map(frames)
    run_prefix = _sanitize_token(prefix)
    run_stamp = _sanitize_token(run_stamp)

    run_dir = Path(output_dir) / f"{run_prefix}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: list[str] = []
    for name, df in clean_frames.items():
        csv_path = run_dir / f"{run_prefix}_{name}.csv"
        df.to_csv(csv_path, index=False)
        csv_paths.append(str(csv_path))

    excel_path: str | None = None
    if include_excel and clean_frames:
        workbook_path = run_dir / f"{run_prefix}_report.xlsx"
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            for name, df in clean_frames.items():
                sheet = _sanitize_sheet_name(name)
                df.to_excel(writer, sheet_name=sheet, index=False)
        excel_path = str(workbook_path)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prefix": run_prefix,
        "run_stamp": run_stamp,
        "output_dir": str(run_dir),
        "tables": [
            {
                "name": name,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
            }
            for name, df in clean_frames.items()
        ],
        "metadata": dict(metadata or {}),
    }

    manifest_path = run_dir / f"{run_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    zip_path: str | None = None
    if include_zip:
        archive_path = run_dir / f"{run_prefix}_share_package.zip"
        with zipfile.ZipFile(
            archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for csv_file in csv_paths:
                path_obj = Path(csv_file)
                zf.write(path_obj, arcname=path_obj.name)
            if excel_path:
                excel_obj = Path(excel_path)
                zf.write(excel_obj, arcname=excel_obj.name)
            zf.write(manifest_path, arcname=manifest_path.name)
        zip_path = str(archive_path)

    return {
        "run_dir": str(run_dir),
        "csv_paths": csv_paths,
        "excel_path": excel_path,
        "manifest_path": str(manifest_path),
        "zip_path": zip_path,
    }
