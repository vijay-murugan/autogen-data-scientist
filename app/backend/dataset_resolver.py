import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import kagglehub
import pandas as pd
from fastapi import HTTPException
from app.core.config import WORKING_DIR

SUPPORTED_TABULAR_EXTENSIONS = {
    ".csv",
    ".parquet",
    ".json",
    ".jsonl",
    ".xlsx",
    ".xls",
}

# Subfolder under WORKING_DIR where per-session cleaned parquet files live.
# Exported so callers that wipe WORKING_DIR between runs can skip this folder
# instead of destroying the cleaned artifact they just created.
CLEANED_SESSIONS_SUBDIR = "cleaned_sessions"


@dataclass
class DatasetFileInfo:
    id: str
    name: str
    relative_path: str
    size_bytes: int
    file_type: str


_DATASET_MANIFEST_CACHE: Dict[str, List[DatasetFileInfo]] = {}
_DATASET_ROOT_CACHE: Dict[str, str] = {}
_CLEANED_SESSION_CACHE: Dict[str, Dict[str, str]] = {}


def normalize_dataset_ref(dataset_input: str) -> str:
    raw = (dataset_input or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="dataset_ref is required.")

    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "datasets":
            return f"{parts[1]}/{parts[2]}"
        raise HTTPException(
            status_code=400,
            detail="Invalid Kaggle dataset URL. Expected format: https://www.kaggle.com/datasets/<owner>/<dataset>",
        )

    if "/" not in raw:
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset_ref. Expected format: <owner>/<dataset>.",
        )

    owner, dataset = raw.split("/", 1)
    owner = owner.strip()
    dataset = dataset.strip()
    if not owner or not dataset:
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset_ref. Expected format: <owner>/<dataset>.",
        )
    return f"{owner}/{dataset}"


def _check_kaggle_credentials() -> None:
    has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    has_file = kaggle_json_path.exists()

    if has_file:
        try:
            content = json.loads(kaggle_json_path.read_text(encoding="utf-8"))
            has_file = bool(content.get("username") and content.get("key"))
        except Exception:
            has_file = False

    if not has_env and not has_file:
        raise HTTPException(
            status_code=400,
            detail=(
                "Kaggle credentials are missing. Set KAGGLE_USERNAME/KAGGLE_KEY "
                "or place ~/.kaggle/kaggle.json with username and key."
            ),
        )


def _list_tabular_files(dataset_root: str) -> List[DatasetFileInfo]:
    root_path = Path(dataset_root)
    files: List[DatasetFileInfo] = []

    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue
        extension = file_path.suffix.lower()
        if extension not in SUPPORTED_TABULAR_EXTENSIONS:
            continue

        relative_path = file_path.relative_to(root_path).as_posix()
        files.append(
            DatasetFileInfo(
                id=relative_path,
                name=file_path.name,
                relative_path=relative_path,
                size_bytes=file_path.stat().st_size,
                file_type=extension.lstrip("."),
            )
        )

    # deterministic ordering for stable UI
    files.sort(key=lambda item: (item.name.lower(), item.relative_path.lower()))
    return files


def _download_dataset(dataset_ref: str) -> str:
    _check_kaggle_credentials()
    try:
        dataset_root = kagglehub.dataset_download(dataset_ref)
    except Exception as exc:
        msg = str(exc).lower()
        if "403" in msg or "unauthorized" in msg or "forbidden" in msg:
            raise HTTPException(
                status_code=403,
                detail="Kaggle authentication failed or dataset is private.",
            ) from exc
        if "404" in msg or "not found" in msg:
            raise HTTPException(status_code=404, detail="Kaggle dataset was not found.") from exc
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download Kaggle dataset: {exc}",
        ) from exc

    return dataset_root


def get_dataset_manifest(dataset_input: str) -> Dict[str, object]:
    dataset_ref = normalize_dataset_ref(dataset_input)
    dataset_root = _DATASET_ROOT_CACHE.get(dataset_ref)

    if not dataset_root:
        dataset_root = _download_dataset(dataset_ref)
        _DATASET_ROOT_CACHE[dataset_ref] = dataset_root

    files = _DATASET_MANIFEST_CACHE.get(dataset_ref)
    if files is None:
        files = _list_tabular_files(dataset_root)
        _DATASET_MANIFEST_CACHE[dataset_ref] = files

    if not files:
        raise HTTPException(
            status_code=400,
            detail=(
                "No supported tabular files found in dataset. "
                "Supported formats: csv, parquet, json, jsonl, xlsx, xls."
            ),
        )

    return {
        "dataset_ref": dataset_ref,
        "files": [
            {
                "id": item.id,
                "name": item.name,
                "relative_path": item.relative_path,
                "size_bytes": item.size_bytes,
                "file_type": item.file_type,
            }
            for item in files
        ],
    }


def resolve_selected_file(dataset_input: str, selected_file: str) -> Dict[str, str]:
    manifest = get_dataset_manifest(dataset_input)
    dataset_ref = str(manifest["dataset_ref"])
    selected = (selected_file or "").strip()
    if not selected:
        raise HTTPException(status_code=400, detail="selected_file is required.")

    available_ids = {file_info["id"] for file_info in manifest["files"]}
    if selected not in available_ids:
        raise HTTPException(
            status_code=400,
            detail="selected_file is not part of the resolved dataset files.",
        )

    dataset_root = _DATASET_ROOT_CACHE[dataset_ref]
    absolute_path = str((Path(dataset_root) / selected).resolve())
    if not Path(absolute_path).exists():
        raise HTTPException(status_code=500, detail="Selected file could not be resolved locally.")

    return {
        "dataset_ref": dataset_ref,
        "dataset_path": absolute_path,
    }


def _load_dataframe(dataset_path: str) -> pd.DataFrame:
    extension = Path(dataset_path).suffix.lower()
    if extension == ".csv":
        return pd.read_csv(dataset_path)
    if extension == ".parquet":
        return pd.read_parquet(dataset_path)
    if extension == ".json":
        return pd.read_json(dataset_path)
    if extension == ".jsonl":
        return pd.read_json(dataset_path, lines=True)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(dataset_path)
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type for cleaning: {extension}.",
    )


def _sanitize_session_id(session_id: Optional[str]) -> str:
    value = (session_id or "").strip()
    if not value:
        return "default"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


def _session_cache_key(session_id: str, dataset_path: str) -> str:
    fingerprint = sha256(dataset_path.encode("utf-8")).hexdigest()[:12]
    return f"{session_id}:{fingerprint}"


def _clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, object]]:
    before_rows, before_cols = df.shape
    cleaned = df.copy()

    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.drop_duplicates()
    # Drop rows that are completely empty; keep partially-populated rows.
    cleaned = cleaned.dropna(how="all")

    # Remove leading/trailing spaces in text fields for cleaner grouping/filters.
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    after_rows, after_cols = cleaned.shape
    metadata = {
        "steps": [
            "trim_column_names",
            "drop_duplicates",
            "drop_fully_empty_rows",
            "trim_string_values",
        ],
        "rows_before": before_rows,
        "rows_after": after_rows,
        "cols_before": before_cols,
        "cols_after": after_cols,
        "rows_removed": max(before_rows - after_rows, 0),
    }
    return cleaned, metadata


def get_or_create_cleaned_session_file(
    dataset_path: str,
    session_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Create one cleaned dataset artifact per session and source dataset.
    Reuses an existing artifact for repeated calls in the same session.
    """
    normalized_session = _sanitize_session_id(session_id)
    cache_key = _session_cache_key(normalized_session, dataset_path)
    cached = _CLEANED_SESSION_CACHE.get(cache_key)
    if cached and Path(cached["cleaned_dataset_path"]).exists():
        return cached

    cleaned_dir = Path(WORKING_DIR) / CLEANED_SESSIONS_SUBDIR
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = sha256(dataset_path.encode("utf-8")).hexdigest()[:12]
    cleaned_path = cleaned_dir / f"{normalized_session}_{dataset_hash}_cleaned.parquet"
    metadata_path = cleaned_dir / f"{normalized_session}_{dataset_hash}_cleaning_meta.json"

    result: Dict[str, str] = {
        "dataset_path": dataset_path,
        "cleaned_dataset_path": dataset_path,
        "cleaning_status": "raw_fallback",
        "cleaning_message": "",
        "cleaning_metadata_path": "",
    }

    try:
        df = _load_dataframe(dataset_path)
        cleaned_df, cleaning_meta = _clean_dataframe(df)
        cleaned_df.to_parquet(cleaned_path, index=False)
        cleaning_meta["generated_at"] = datetime.utcnow().isoformat() + "Z"
        cleaning_meta["session_id"] = normalized_session
        cleaning_meta["source_dataset_path"] = dataset_path
        metadata_path.write_text(json.dumps(cleaning_meta, indent=2), encoding="utf-8")

        result.update(
            {
                "cleaned_dataset_path": str(cleaned_path.resolve()),
                "cleaning_status": "cleaned",
                "cleaning_message": "Dataset auto-cleaned and persisted for this session.",
                "cleaning_metadata_path": str(metadata_path.resolve()),
            }
        )
    except Exception as exc:
        result["cleaning_message"] = (
            "Auto-clean failed; using raw dataset path instead. "
            f"Reason: {exc}"
        )

    _CLEANED_SESSION_CACHE[cache_key] = result
    return result
