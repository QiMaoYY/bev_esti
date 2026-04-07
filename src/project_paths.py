import json
import os
import re
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / "data"
TABLES_DIR = DATA_ROOT / "bevplace_tables"
RUNS_DIR = WORKSPACE_ROOT / "BEVPlace2" / "runs"
DEBUG_OUTPUTS_DIR = REPO_ROOT / "debug_outputs"
ESTIMATE_OUTPUTS_DIR = DEBUG_OUTPUTS_DIR / "estimate_pose"
BATCH_OUTPUTS_DIR = DEBUG_OUTPUTS_DIR / "batch_evaluate"


def default_data_root() -> str:
    return str(DATA_ROOT.resolve())


def default_database_table() -> str:
    return str((TABLES_DIR / "database_samples.csv").resolve())


def default_query_table() -> str:
    return str((TABLES_DIR / "query_samples.csv").resolve())


def default_db_cache() -> str:
    return str((REPO_ROOT / "database_cache.npz").resolve())


def sanitize_name(name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip())
    return normalized or "unnamed"


def _is_bevplace_checkpoint(checkpoint_path: Path) -> bool:
    flags_path = checkpoint_path.parent / "flags.json"
    if not flags_path.exists():
        return False
    try:
        flags = json.loads(flags_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return flags.get("dataset") == "bevplace"


def find_latest_checkpoint() -> Optional[Path]:
    checkpoint_env = os.environ.get("BEV_ESTI_CHECKPOINT", "").strip()
    if checkpoint_env:
        return Path(checkpoint_env).expanduser().resolve()

    if not RUNS_DIR.exists():
        return None

    candidates = sorted(
        (path for path in RUNS_DIR.glob("*/model_best.pth.tar") if _is_bevplace_checkpoint(path)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].resolve()


def resolve_checkpoint(checkpoint_arg: str) -> str:
    if checkpoint_arg.strip():
        return str(Path(checkpoint_arg).expanduser().resolve())

    checkpoint = find_latest_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError(
            "Failed to auto-detect a bevplace checkpoint under ../BEVPlace2/runs. "
            "Please provide --checkpoint explicitly."
        )
    return str(checkpoint)


def default_estimate_output_dir(query_label: str) -> Path:
    return (ESTIMATE_OUTPUTS_DIR / sanitize_name(query_label)).resolve()


def build_range_tag(first_index: int, last_index: int) -> str:
    return f"range_{first_index:04d}_{last_index:04d}"


def default_batch_output_dir(first_index: int, last_index: int) -> Path:
    return (BATCH_OUTPUTS_DIR / build_range_tag(first_index, last_index)).resolve()
