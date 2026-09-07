from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_thirdparty_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for rel_path, package_name in (
        (Path("thirdparty") / "mast3r", "mast3r"),
        (Path("thirdparty") / "in3d", "in3d"),
        (Path("sam3"), "sam3"),
    ):
        package_root = repo_root / rel_path
        if (package_root / package_name / "__init__.py").is_file():
            package_root_str = str(package_root)
            if package_root_str not in sys.path:
                sys.path.insert(0, package_root_str)


_ensure_repo_thirdparty_imports()
