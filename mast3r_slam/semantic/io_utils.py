from __future__ import annotations



import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:

    import orjson  # type: ignore
    _HAS_ORJSON = True
except Exception:  # pragma: no cover
    _HAS_ORJSON = False


def write_json_atomic(path: Path, payload: Any, *, indent: int | None = 2) -> None:

    path = Path(path).expanduser().resolve()

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = None
    use_orjson = _HAS_ORJSON and indent in (None, 2)
    try:
        if use_orjson:

            opt = orjson.OPT_NON_STR_KEYS
            if indent == 2:
                opt |= orjson.OPT_INDENT_2
            data = orjson.dumps(payload, option=opt)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = f.name
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        else:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = f.name
                json.dump(payload, f, ensure_ascii=False, indent=indent)
                f.flush()
                os.fsync(f.fileno())

        try:
            os.replace(tmp_path, path)
        except FileNotFoundError:


            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=indent)
    finally:

        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def append_jsonl(path: Path, payload: Any) -> None:

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")
