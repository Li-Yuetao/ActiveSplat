import os
from pathlib import Path


def source_root(script_file: str) -> str:
    configured = os.environ.get("ACTIVESPLAT_SOURCE_DIR", "").strip()
    if configured and (Path(configured) / "config").is_dir():
        return str(Path(configured).resolve())

    script_path = Path(script_file).resolve()
    for parent in script_path.parents:
        if (parent / "config" / "datasets").is_dir() and (parent / "src").is_dir():
            return str(parent)
        candidate = parent / "src" / "activesplat"
        if (candidate / "config" / "datasets").is_dir():
            return str(candidate)
    raise RuntimeError("cannot locate ActiveSplat source tree; set ACTIVESPLAT_SOURCE_DIR")
