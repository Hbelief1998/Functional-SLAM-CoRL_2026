from __future__ import annotations



import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from mast3r_slam.semantic.prompt_loader import load_prompt

# Objects/tags to drop entirely from incremental results (non-manipulable or too generic).
_BANNED_OBJECTS: Set[str] = {
    "appliance",
    "box",
    "cardboard box",
    "container",
    "counter top",
    "countertop",
    "equipment",
    "device",
    "furniture",
    "utensil",
    "tool",
    "object",
    "thing",
    "home appliance",
    "kitchenware",
    "tableware",
}

# Objects that the bathroom-specific incremental prompt explicitly excludes.
# Keep these scene-scoped so kitchen/general incremental outputs are unchanged.
_BATHROOM_INCREMENTAL_BANNED_OBJECTS: Set[str] = {
    "medicine cabinet",
    "soap dispenser",
    "hair dryer",
    "exhaust fan",
    "lamp",
    "container box",
}

# Objects/units intentionally excluded from the livingroom-specific incremental
# prompt to avoid broad, long-tail UCO expansion.
_LIVINGROOM_INCREMENTAL_BANNED_OBJECTS: Set[str] = {
    "blind",
    "blinds",
    "speaker",
    "media device",
    "game console",
    "recliner",
    "curtain",
    "cord",
    "pull cord",
    "draw cord",
    "wand",
}

# Objects intentionally excluded from the bedroom-specific incremental prompt.
_BEDROOM_INCREMENTAL_BANNED_OBJECTS: Set[str] = {
    "bed",
    "pillow",
    "blanket",
    "quilt",
    "mattress",
    "clothes",
    "clothing",
    "book",
    "decoration",
    "carpet",
    "rug",
    "wall",
    "floor",
    "ceiling",
    "curtain",
    "blinds",
    "desk",
    "monitor",
    "speaker",
    "game console",
    "recliner",
    "cord",
    "wand",
}

# Generic category nouns (not concrete objects). Keep it small and general.
_GENERIC_NOUNS: Set[str] = {
    "appliance",
    "device",
    "equipment",
    "utensil",
    "tool",
    "object",
    "thing",
    "item",
    "stuff",
    "kitchenware",
    "tableware",
    "furniture",
}

# Generic modifiers commonly used in category labels.
_GENERIC_MODIFIERS: Set[str] = {
    "kitchen",
    "home",
    "household",
    "electric",
    "electrical",
    "generic",
    "misc",
}

# Carriers that are clearly not parts of the same object.
_BANNED_CARRIERS: Set[str] = {
    "floor",
    "ground",
    "wall",
    "ceiling",
    "counter",
    "counter top",
    "countertop",
    "table",
    "table top",
    "desk",
    "stand",
    "room",
    "light",
    "shelf surface",
}

# Only allow a tiny set of strong remote relations to keep outputs conservative.
_ALLOWED_REMOTE_REL: Set[tuple[str, str]] = {
    ("faucet", "sink"),
    ("sink", "faucet"),
}


_REMOTE_STRUCT_BLOCK: Set[str] = {
    "door",
    "wall",
    "floor",
    "ceiling",
    "window",
    "cabinet",
    "drawer",
    "countertop",
    "counter",
    "shelf",
    "table",
    "desk",
}


_REMOTE_SOURCE_KEYS = ("switch", "panel", "outlet", "socket", "faucet", "tap", "valve", "remote control")


_REMOTE_TARGET_HINTS = (
    "light",
    "lamp",
    "fan",
    "hood",
    "exhaust",
    "air",
    "ac",
    "sink",
    "shower",
    "bathtub",
    "television",
)


def _has_any(s: str, keys: tuple[str, ...]) -> bool:
    return any(k in s for k in keys)


def _singularize(tok: str) -> str:
    # very light singularization for plurals
    if tok.endswith("es") and len(tok) > 4:
        if tok[:-2] in _GENERIC_NOUNS:
            return tok[:-2]
        if tok[:-1] in _GENERIC_NOUNS:
            return tok[:-1]
        return tok[:-2]
    if tok.endswith("s") and len(tok) > 3:
        return tok[:-1]
    return tok


def _is_generic_category_label(name: str) -> bool:
    """
    True for generic category labels like:
      - 'kitchen utensil', 'home appliances', 'electrical device'
    False for concrete objects like:
      - 'dishwasher', 'knife', 'utensil holder', 'appliance cord'
    """
    if not name:
        return True

    # keep existing exact banned list as first gate
    if name in _BANNED_OBJECTS:
        return True

    toks = [t for t in name.split() if t]
    if not toks:
        return True

    last = _singularize(toks[-1])

    # single-token generic noun
    if len(toks) == 1 and last in _GENERIC_NOUNS:
        return True

    # short phrase ending in generic noun with only generic modifiers before it
    if len(toks) <= 3 and last in _GENERIC_NOUNS:
        if all(_singularize(t) in _GENERIC_MODIFIERS for t in toks[:-1]):
            return True

    # two-word "X object/thing/item" is almost always useless
    if len(toks) == 2 and last in {"object", "thing", "item"}:
        return True

    return False


def _strip_text(x: Any) -> str:
    return x.strip() if isinstance(x, str) else ""


def _unit_name(x: Any) -> str:
    if isinstance(x, dict):
        return _strip_text(x.get("unit"))
    return _strip_text(x)


def _parse_interactive_unit(x: Any) -> Optional[Dict[str, str]]:
    name = _unit_name(x)
    if not name:
        return None
    if isinstance(x, dict):
        return {
            "unit": name,
            "cu_relation": _strip_text(x.get("cu_relation")),
            "ou_relation": _strip_text(x.get("ou_relation")),
        }
    return {"unit": name, "cu_relation": "", "ou_relation": ""}


def _parse_direct_unit(x: Any) -> Optional[Dict[str, str]]:
    name = _unit_name(x)
    if not name:
        return None
    if isinstance(x, dict):
        return {"unit": name, "ou_relation": _strip_text(x.get("ou_relation"))}
    return {"unit": name, "ou_relation": ""}


def _is_remote_source(name: str, atlas_remote_set: Set[str]) -> bool:

    if name in atlas_remote_set:
        if _has_any(name, _REMOTE_SOURCE_KEYS):
            return True
        if _has_any(name, _REMOTE_TARGET_HINTS):
            return False
        return False
    return _has_any(name, _REMOTE_SOURCE_KEYS)


def _is_bad_remote_target(name: str) -> bool:

    return name in _REMOTE_STRUCT_BLOCK


def _atlas_fix_remote_direction(atlas: Dict[str, Any]) -> Dict[str, Any]:
    edges = atlas.get("remote_relation_candidates", []) or []
    fixed: List[Dict[str, str]] = []
    src_keys = ("switch", "panel", "outlet", "socket", "faucet", "tap", "valve", "remote control")

    def is_source(x: str) -> bool:
        x = _norm(x)
        return any(k in x for k in src_keys)

    for e in edges:
        a = _norm(e.get("from_object"))
        b = _norm(e.get("to_object"))
        if not a or not b or a == b:
            continue

        a_is, b_is = is_source(a), is_source(b)
        if (not a_is) and b_is:
            a, b = b, a
            a_is, b_is = b_is, a_is

        if a_is and b_is:
            continue

        if _is_bad_remote_target(b):
            continue

        rel_text = (e.get("relation") or "").strip()
        fixed.append({"from_object": a, "to_object": b, "relation": rel_text})

    atlas["remote_relation_candidates"] = fixed
    return atlas


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _norm_scene_type(scene: Any) -> str:
    s = _norm(str(scene) if scene is not None else "")
    s = s.replace("_", " ")
    s = " ".join(s.split())
    if s in {"living room", "livingroom"}:
        return "livingroom"
    if s in {"bed room", "bedroom"}:
        return "bedroom"
    if s in {"kitchen", "bathroom", "unknown"}:
        return s
    return s


def _prompt_digest(text: str, n: int = 8) -> str:
    """Stable digest for prompt text with light normalization."""
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join([ln.rstrip() for ln in s.split("\n")]).strip() + "\n"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def sanitize_incremental_output(
    payload: Dict[str, Any],
    allowed_objects: Optional[Set[str]] = None,
    allowed_remote_endpoints: Optional[Set[str]] = None,
    atlas_remote_endpoints: Optional[Set[str]] = None,
    extra_banned_objects: Optional[Set[str]] = None,
    max_remote_pairs: int = 8,
) -> Dict[str, Any]:
    """Sanitize incremental UC output to align with prompt rules.

    - Only keep objects within allowed_objects (if provided) and not in banned list.
    - Accept entries with either carriers or direct_interactive_units (or both).
    - Do NOT merge carrier units into direct units; preserve structure.
    - Preserve remote_relation_candidates with filtering:
        * endpoints must be in allowed_remote_endpoints (if provided)
        * at least one endpoint must be in atlas_remote_endpoints (if provided)
        * dedup undirected; cap total pairs to max_remote_pairs.
    - Lightly clip sizes to avoid runaway outputs.
    """

    allowed_norm = {_norm(o) for o in allowed_objects} if allowed_objects else None
    allowed_remote_norm = {_norm(o) for o in allowed_remote_endpoints} if allowed_remote_endpoints else None
    atlas_remote_norm = {_norm(o) for o in atlas_remote_endpoints} if atlas_remote_endpoints else None
    banned_norm = set(_BANNED_OBJECTS)
    if extra_banned_objects:
        banned_norm.update({_norm(o) for o in extra_banned_objects if _norm(o)})
    input_tags_norm = allowed_norm

    sanitized_objects: List[Dict[str, Any]] = []
    for obj in payload.get("objects", []) or []:
        name = _norm(obj.get("object"))
        if not name or name in banned_norm or _is_generic_category_label(name):
            continue
        if allowed_norm is not None and name not in allowed_norm:
            continue

        carriers: List[Dict[str, Any]] = []
        for carrier in obj.get("functional_carriers", []) or []:
            cname = _norm(carrier.get("carrier"))
            if not cname or cname in banned_norm or cname in _BANNED_CARRIERS or _is_generic_category_label(cname):
                continue
            oc_relation = _strip_text(carrier.get("oc_relation")) or "part of"
            unit_map: Dict[str, Dict[str, str]] = {}
            for raw_unit in carrier.get("interactive_units", []) or []:
                parsed = _parse_interactive_unit(raw_unit)
                if not parsed:
                    continue
                unit = parsed["unit"]
                if _norm(unit) in banned_norm:
                    continue
                if unit not in unit_map:
                    unit_map[unit] = parsed
                else:
                    if parsed.get("cu_relation") and len(parsed["cu_relation"]) > len(unit_map[unit].get("cu_relation", "")):
                        unit_map[unit]["cu_relation"] = parsed["cu_relation"]
                    if parsed.get("ou_relation") and len(parsed["ou_relation"]) > len(unit_map[unit].get("ou_relation", "")):
                        unit_map[unit]["ou_relation"] = parsed["ou_relation"]
            if not unit_map:
                continue
            units = list(unit_map.values())[:2]
            carriers.append({"carrier": cname, "oc_relation": oc_relation, "interactive_units": units})

        direct_unit_map: Dict[str, Dict[str, str]] = {}
        for raw_unit in obj.get("direct_interactive_units", []) or []:
            parsed = _parse_direct_unit(raw_unit)
            if not parsed:
                continue
            unit = parsed["unit"]
            if _norm(unit) in banned_norm:
                continue
            if unit not in direct_unit_map:
                direct_unit_map[unit] = parsed
            else:
                if parsed.get("ou_relation") and len(parsed["ou_relation"]) > len(
                    direct_unit_map[unit].get("ou_relation", "")
                ):
                    direct_unit_map[unit]["ou_relation"] = parsed["ou_relation"]
        direct_units = list(direct_unit_map.values())

        # Keep entry if it has either carriers or direct units.
        if not carriers and not direct_units:
            continue

        sanitized_objects.append(
            {
                "object": name,
                "functional_carriers": carriers[:2],
                "direct_interactive_units": direct_units[:3],
            }
        )

    dedup: Dict[str, Dict[str, Any]] = {}
    for obj in sanitized_objects:
        if obj["object"] not in dedup:
            dedup[obj["object"]] = obj

    # Remote relations: conservative filtering
    remote_map: Dict[tuple[str, str], str] = {}
    remote_out: List[Dict[str, str]] = []
    for rel in payload.get("remote_relation_candidates", []) or []:
        a = _norm(rel.get("from_object"))
        b = _norm(rel.get("to_object"))
        if not a or not b or a == b:
            continue
        if a in banned_norm or b in banned_norm:
            continue
        if _is_generic_category_label(a) or _is_generic_category_label(b):
            continue
        if allowed_remote_norm is not None and (a not in allowed_remote_norm or b not in allowed_remote_norm):
            continue
        if input_tags_norm is not None and (a not in input_tags_norm and b not in input_tags_norm):
            continue
        if not _is_remote_source(a, atlas_remote_norm or set()):
            continue
        if _is_bad_remote_target(b):
            continue
        if _is_remote_source(b, atlas_remote_norm or set()):
            continue

        key = (a, b)
        rel_text = (rel.get("relation") or "").strip()
        if key not in remote_map or (not remote_map[key] and rel_text):
            remote_map[key] = rel_text

        if len(remote_map) >= max_remote_pairs:
            break

    for (a, b), rel_text in remote_map.items():
        remote_out.append({"from_object": a, "to_object": b, "relation": rel_text})

    return {"objects": list(dedup.values())[:12], "remote_relation_candidates": remote_out}


def strip_code_fences(text: str) -> str:

    stripped = (text or "").strip()
    if stripped.startswith("```"):
        after = stripped[3:]
        newline = after.find("\n")
        stripped = after[newline + 1 :] if newline != -1 else ""
    if stripped.endswith("```"):
        stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def extract_json_from_text(text: str) -> Dict[str, Any]:

    s = strip_code_fences(text)
    try:
        return json.loads(s)
    except Exception:
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(s[i : j + 1])
        raise RuntimeError("Model output is not valid JSON:\n" + (text or ""))


def _pretty_error(resp) -> str:

    try:
        return json.dumps(resp.json(), ensure_ascii=False, indent=2)
    except Exception:
        return (resp.text or "").strip()


def call_deepseek_chat(
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
):

    if not api_key or "PASTE" in api_key:
        raise RuntimeError("DeepSeek API key is empty/placeholder. Set DEEPSEEK_API_KEY.")

    import requests

    endpoints = [
        f"{base_url.rstrip('/')}/chat/completions",
        f"{base_url.rstrip('/')}/v1/chat/completions",
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }

    messages = [
        {"role": "system", "content": "Output valid JSON only. Do not include any extra text or Markdown."},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_resp = None
    last_exc: Optional[Exception] = None

    for endpoint in endpoints:
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
                last_resp = resp


                if resp.status_code == 200:
                    return resp.json()


                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"HTTP {resp.status_code} Unauthorized/Forbidden at {endpoint}\n{_pretty_error(resp)}"
                    )


                if resp.status_code == 429 or (500 <= resp.status_code <= 599):
                    if attempt < max_retries:
                        time.sleep(1.5 ** attempt)
                        continue

                raise RuntimeError(f"HTTP {resp.status_code} at {endpoint}\n{_pretty_error(resp)}")

            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    time.sleep(1.5 ** attempt)
                    continue
                break

    if last_resp is not None:
        raise RuntimeError(
            f"DeepSeek request failed. Last HTTP {last_resp.status_code}\n{_pretty_error(last_resp)}"
        )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("DeepSeek request failed for unknown reason.")


def generate_atlas_with_retry(
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
    response_retries: int,
    max_tokens_cap: int,
):

    current_max_tokens = max_tokens
    for attempt in range(response_retries + 1):
        resp = call_deepseek_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=current_max_tokens,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
        )
        choice = resp["choices"][0]
        content = choice["message"]["content"]
        finish_reason = (choice.get("finish_reason") or "").lower()
        try:
            return extract_json_from_text(content)
        except Exception as exc:
            should_retry = attempt < response_retries
            if finish_reason == "length" and current_max_tokens < max_tokens_cap:
                should_retry = True
                current_max_tokens = min(int(current_max_tokens * 1.5), max_tokens_cap)
            if not should_retry:
                raise
            print(
                f"[deepseek] retry {attempt + 1}/{response_retries} due to parse error ({exc}). "
                f"finish_reason={finish_reason or 'unknown'}, next max_tokens={current_max_tokens}",
                file=sys.stderr,
            )
            time.sleep(min(2.0, 0.5 * (attempt + 1)))

    raise RuntimeError("Failed to obtain valid JSON from DeepSeek after retries.")


class DeepSeekRuntime:
    def __init__(
        self,
        api_key: Optional[str],
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout_sec: int = 60,
        max_retries: int = 2,
        response_retries: int = 2,
        max_tokens: int = 1600,
        max_tokens_cap: int = 4096,
        cache_dir: str = "logs/deepseek_cache",
        temperature: float = 0.0,
    ) -> None:

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            self.api_key = os.getenv("DEEPSEEK_KEY") or os.getenv("DEEPSEEK_TOKEN")
        if not self.api_key:
            raise RuntimeError("DeepSeek API key not found. Please set DEEPSEEK_API_KEY (or DEEPSEEK_TOKEN).")

        self.base_url = base_url
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.response_retries = response_retries
        self.max_tokens = max_tokens
        self.max_tokens_cap = max_tokens_cap
        self.temperature = temperature


        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Prompt versioning to invalidate stale caches when prompts change.
        base_version = "v4_20260112_scene_aware_uc"
        try:
            inc_tpl = load_prompt("incremental_uc_en.txt")
            scene_tpl = load_prompt("scene_classify_en.txt")
            inc_hash = _prompt_digest(inc_tpl)
            scene_hash = _prompt_digest(scene_tpl)
            self.prompt_version = f"{base_version}_scene{scene_hash}_inc{inc_hash}"
        except Exception as e:
            self.prompt_version = base_version
            print(f"[DeepSeek] warning: failed to hash incremental_uc_en.txt: {e}", file=sys.stderr)

        print(f"[DeepSeek] prompt_version={self.prompt_version}")

        # Simple HTTP call counters (cache hits do not increment).
        self.http_stats = {"scene": 0, "incremental": 0, "atlas": 0}

        self.prompts = {
            "kitchen": load_prompt("global_atlas_kitchen_en.txt"),
            "bathroom": load_prompt("global_atlas_bathroom_en.txt"),
            "livingroom": load_prompt("global_atlas_livingroom_en.txt"),
            "bedroom": load_prompt("global_atlas_bedroom_en.txt"),
        }

        # Atlas prompt digests: invalidate atlas cache when atlas prompt changes.
        self.atlas_prompt_digest = {k: _prompt_digest(v) for k, v in self.prompts.items()}
        print(f"[DeepSeek] atlas_prompt_digest={self.atlas_prompt_digest}")
        self.cache_only = os.getenv("DEEPSEEK_CACHE_ONLY", "0") == "1"
        if self.cache_only:
            print("[DeepSeek] cache-only mode enabled; API calls are disabled.")

    @staticmethod
    def _normalize_tags(tags_en: List[str]) -> List[str]:

        return sorted({t.strip().lower() for t in tags_en if t and t.strip()})

    def _cache_key(
        self,
        name: str,
        tags_en: List[str],
        atlas_version: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> Path:

        payload = {
            "name": name,
            "tags": self._normalize_tags(tags_en),
        }
        if atlas_version:
            payload["atlas_version"] = atlas_version
        if prompt_version:
            payload["prompt_version"] = prompt_version
        h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{name}_{h}.json"

    def _load_incremental_prompt_template(self, scene_type: str) -> tuple[str, str, str]:
        """
        Return (prompt_name, template, digest).

        Scene-specific prompt has priority:
          incremental_uc_<scene_type>_en.txt

        Fallback:
          incremental_uc_en.txt
        """
        scene_norm = _norm_scene_type(scene_type) or "unknown"
        scene_key = scene_norm.replace(" ", "_")
        candidate_names = []
        if scene_key not in ("", "unknown"):
            candidate_names.append(f"incremental_uc_{scene_key}_en.txt")
        candidate_names.append("incremental_uc_en.txt")

        last_missing = None
        for name in candidate_names:
            try:
                template = load_prompt(name)
            except FileNotFoundError as exc:
                last_missing = exc
                continue
            return name, template, _prompt_digest(template)

        if last_missing is not None:
            raise last_missing
        raise RuntimeError("No incremental UC prompt template found.")

    @staticmethod
    def _load_cache(path: Path) -> Optional[Dict[str, Any]]:

        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _save_cache(path: Path, payload: Dict[str, Any]) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def classify_scene(self, tags_en: List[str]) -> Dict[str, Any]:

        norm_tags = self._normalize_tags(tags_en)
        cache_path = self._cache_key("scene_classify", norm_tags, prompt_version=self.prompt_version)
        cached = self._load_cache(cache_path)
        if cached is not None:
            return cached
        if self.cache_only:
            raise RuntimeError(f"DeepSeek cache-only miss: {cache_path}")
        template = load_prompt("scene_classify_en.txt")
        prompt = template.replace("{tags_csv}", ", ".join(norm_tags))
        self.http_stats["scene"] += 1
        resp = call_deepseek_chat(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            user_prompt=prompt,
            temperature=self.temperature,
            max_tokens=320,
            timeout_sec=self.timeout_sec,
            max_retries=self.max_retries,
        )
        choice = resp["choices"][0]
        content = choice["message"]["content"]
        parsed = extract_json_from_text(content)
        scene = _norm_scene_type(parsed.get("scene_type"))
        parsed["scene_type"] = scene
        reason = None
        if scene not in ("kitchen", "bathroom", "livingroom", "bedroom", "unknown"):
            reason = f"invalid_scene:{scene}"
            parsed["scene_type"] = "unknown"
            parsed["confidence"] = 0.0
        if parsed.get("scene_type") == "unknown":
            parsed["confidence"] = 0.0
            if reason is None:
                reason = "unknown_scene"
        if reason:
            parsed["reason"] = reason
        self._save_cache(cache_path, parsed)
        return parsed

    def get_or_build_global_atlas(self, scene_type: str) -> Dict[str, Any]:

        scene_type = _norm_scene_type(scene_type)
        if scene_type not in self.prompts:
            raise RuntimeError(f"Unsupported scene_type: {scene_type}")
        pd = self.atlas_prompt_digest.get(scene_type, "")
        cache_path = self._cache_key(
            f"atlas_{scene_type}",
            [scene_type],
            prompt_version=f"atlas_{scene_type}_{pd}",
        )
        cached = self._load_cache(cache_path)
        if cached is not None:
            print(f"[DeepSeek][atlas][cache_hit] scene={scene_type} file={cache_path.name}")
            return cached
        if self.cache_only:
            raise RuntimeError(f"DeepSeek cache-only miss: {cache_path}")
        print(f"[DeepSeek][atlas][cache_miss] scene={scene_type} prompt_digest={pd} file={cache_path.name}")
        self.http_stats["atlas"] += 1
        atlas = generate_atlas_with_retry(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            user_prompt=self.prompts[scene_type],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_sec=self.timeout_sec,
            max_retries=self.max_retries,
            response_retries=self.response_retries,
            max_tokens_cap=self.max_tokens_cap,
        )
        atlas = _atlas_fix_remote_direction(atlas)
        self._save_cache(cache_path, atlas)
        return atlas

    def infer_incremental_uc(
        self,
        uc_tags_en: List[str],
        atlas_remote_objects_en: List[str],
        scene_type: str = "unknown",
    ) -> Dict[str, Any]:

        norm_tags = self._normalize_tags(uc_tags_en)
        scene_norm = _norm_scene_type(scene_type) or "unknown"
        atlas_remote_norm = self._normalize_tags(atlas_remote_objects_en or [])
        if len(norm_tags) == 0:
            return {"objects": [], "remote_relation_candidates": []}
        if scene_norm == "bathroom":
            extra_banned = _BATHROOM_INCREMENTAL_BANNED_OBJECTS
        elif scene_norm == "livingroom":
            extra_banned = _LIVINGROOM_INCREMENTAL_BANNED_OBJECTS
        elif scene_norm == "bedroom":
            extra_banned = _BEDROOM_INCREMENTAL_BANNED_OBJECTS
        else:
            extra_banned = None

        inc_prompt_name, template, inc_prompt_digest = self._load_incremental_prompt_template(scene_norm)
        inc_prompt_version = f"{self.prompt_version}|inc_prompt={inc_prompt_name}:{inc_prompt_digest}"
        remote_hash_src = ",".join(atlas_remote_norm)
        remote_hash = hashlib.md5(remote_hash_src.encode("utf-8")).hexdigest() if remote_hash_src else "none"
        cache_path = self._cache_key(
            "incremental_uc",
            norm_tags,
            atlas_version=f"scene={scene_norm}|remote={remote_hash}",
            prompt_version=inc_prompt_version,
        )
        cached = self._load_cache(cache_path)
        if cached is not None:
            return sanitize_incremental_output(
                cached,
                allowed_objects=set(norm_tags),
                allowed_remote_endpoints=set(norm_tags) | set(atlas_remote_norm),
                atlas_remote_endpoints=set(atlas_remote_norm),
                extra_banned_objects=extra_banned,
            )
        if self.cache_only:
            raise RuntimeError(f"DeepSeek cache-only miss: {cache_path}")

        current_max_tokens = self.max_tokens
        prompt = (
            template.replace("{scene_type}", scene_norm)
            .replace("{uc_tags_csv}", ", ".join(norm_tags))
            .replace("{atlas_remote_objects_csv}", ", ".join(atlas_remote_norm) if atlas_remote_norm else "none")
        )


        for attempt in range(self.response_retries + 1):
            if attempt == 0:
                self.http_stats["incremental"] += 1
            resp = call_deepseek_chat(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                user_prompt=prompt,
                temperature=self.temperature,
                max_tokens=current_max_tokens,
                timeout_sec=self.timeout_sec,
                max_retries=self.max_retries,
            )
            choices = resp.get("choices") if isinstance(resp, dict) else None
            if not choices:
                should_retry = attempt < self.response_retries
                if should_retry:
                    print(
                        f"[deepseek] incremental retry {attempt + 1}/{self.response_retries} "
                        "because response has no choices.",
                        file=sys.stderr,
                    )
                    time.sleep(min(2.0, 0.5 * (attempt + 1)))
                    continue
                print(
                    "[deepseek] incremental fallback: response has no choices; using empty incremental result.",
                    file=sys.stderr,
                )
                return {"objects": [], "remote_relation_candidates": []}
            choice = choices[0]
            content = choice["message"]["content"]
            finish_reason = (choice.get("finish_reason") or "").lower()
            try:
                parsed = extract_json_from_text(content)
                sanitized = sanitize_incremental_output(
                    parsed,
                    allowed_objects=set(norm_tags),
                    allowed_remote_endpoints=set(norm_tags) | set(atlas_remote_norm),
                    atlas_remote_endpoints=set(atlas_remote_norm),
                    extra_banned_objects=extra_banned,
                )
                self._save_cache(cache_path, sanitized)
                return sanitized
            except Exception as exc:
                should_retry = attempt < self.response_retries
                if finish_reason == "length" and current_max_tokens < self.max_tokens_cap:
                    current_max_tokens = min(int(current_max_tokens * 1.5), self.max_tokens_cap)
                    should_retry = True
                if not should_retry:
                    raise
                print(
                    f"[deepseek] incremental retry {attempt + 1}/{self.response_retries} due to parse error ({exc}). "
                    f"finish_reason={finish_reason or 'unknown'}, next max_tokens={current_max_tokens}",
                    file=sys.stderr,
                )
                time.sleep(min(2.0, 0.5 * (attempt + 1)))

        raise RuntimeError("Failed to obtain valid incremental UC JSON after retries.")
