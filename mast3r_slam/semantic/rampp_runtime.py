from __future__ import annotations



import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


class RamppTagger:
    """Lightweight RAM++ wrapper with lazy imports for spawn compatibility."""

    def __init__(self, ckpt_path: str, device: str = "cuda", image_size: int = 384, vit: str = "swin_l") -> None:
        # Avoid heavy imports at module import time to keep torch.multiprocessing spawn stable.
        repo_root = Path(__file__).resolve().parents[2]
        ram_repo = repo_root / "recognize-anything"
        if str(ram_repo) not in sys.path:
            sys.path.append(str(ram_repo))


        import torch
        from PIL import Image
        from ram import get_transform, inference_ram
        from ram.models import ram_plus

        ckpt = Path(ckpt_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"RAM++ checkpoint not found: {ckpt}")


        self.device = torch.device(device)
        self._torch = torch
        self._Image = Image
        self._inference = inference_ram
        self._transform = get_transform(image_size=image_size)
        self._model = ram_plus(pretrained=str(ckpt), image_size=image_size, vit=vit)
        self._model.eval()
        self._model = self._model.to(self.device)
        self._image_size = image_size

        print(f"[RAM++] Loaded checkpoint {ckpt} on {self.device}")

    @staticmethod
    def _split_tags(raw_tags: str) -> List[str]:

        if raw_tags is None:
            return []
        if isinstance(raw_tags, (list, tuple)):
            # Model API returns a string; still handle lists defensively.
            joined = " | ".join([str(t) for t in raw_tags if t is not None])
            raw_tags = joined
        if not isinstance(raw_tags, str):
            return []
        return [tag.strip() for tag in raw_tags.split("|") if tag.strip()]

    def infer(self, img_rgb_float01: np.ndarray) -> Dict[str, List[str]]:
        """Run RAM++ tagging on an RGB float image in [0, 1]."""

        np_img = np.asarray(img_rgb_float01)
        if np_img.ndim != 3 or np_img.shape[2] != 3:
            raise ValueError("Expected an HxWx3 RGB image for RAM++ inference")


        np_img = np.clip(np_img, 0.0, 1.0)
        img_uint8 = (np_img * 255.0).round().astype(np.uint8)
        pil_img = self._Image.fromarray(img_uint8)
        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)


        with self._torch.no_grad():
            tags_en_raw, tags_zh_raw = self._inference(tensor, self._model)

        return {
            "tags_en": self._split_tags(tags_en_raw),
            "tags_zh": self._split_tags(tags_zh_raw),
        }
