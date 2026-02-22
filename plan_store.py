# plan_store.py
import os
import re
import time
import tempfile
from typing import Dict, Any, Optional

import cv2
import numpy as np

try:
    import streamlit as st
except Exception:
    st = None


def _safe_name(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9א-ת._ -]+", "_", name)
    return name[:120] if len(name) > 120 else name


def get_assets_dir() -> str:
    """
    מחזיר תיקיית temp פר-session. נשמרת ב-session_state כדי לא להתחלף בכל rerun.
    """
    if st is not None:
        if "contech_assets_dir" not in st.session_state:
            base = tempfile.mkdtemp(prefix="contech_assets_")
            st.session_state.contech_assets_dir = base
        return st.session_state.contech_assets_dir

    # fallback (אם משתמשים בזה בלי Streamlit)
    base = tempfile.mkdtemp(prefix="contech_assets_")
    return base


def _write_png(path: str, arr: np.ndarray) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if arr is None:
        raise ValueError("Tried to write None array")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("cv2.imencode failed for PNG")
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return path


def _write_jpg(path: str, bgr: np.ndarray, quality: int = 85) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if bgr is None:
        raise ValueError("Tried to write None image")

    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed for JPG")
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return path


def persist_project_assets(
    plan_key: str,
    *,
    original_bgr: Optional[np.ndarray] = None,
    thick_walls: Optional[np.ndarray] = None,
    skeleton: Optional[np.ndarray] = None,
    concrete_mask: Optional[np.ndarray] = None,
    blocks_mask: Optional[np.ndarray] = None,
    flooring_mask: Optional[np.ndarray] = None,
    debug_layers: Optional[Dict[str, np.ndarray]] = None,
    keep_debug: bool = False,
) -> Dict[str, Any]:
    """
    שומר assets לדיסק ומחזיר dict קטן עם נתיבים.
    """
    base = get_assets_dir()
    safe = _safe_name(plan_key)
    stamp = str(int(time.time()))
    out_dir = os.path.join(base, safe, stamp)
    os.makedirs(out_dir, exist_ok=True)

    assets: Dict[str, Any] = {}

    if original_bgr is not None:
        assets["original_path"] = _write_jpg(os.path.join(out_dir, "original.jpg"), original_bgr)

        h, w = original_bgr.shape[:2]
        max_dim = 1200
        scale = max(h, w) / max_dim if max(h, w) > max_dim else 1.0
        if scale > 1.0:
            preview = cv2.resize(original_bgr, (int(w / scale), int(h / scale)))
        else:
            preview = original_bgr
        assets["preview_path"] = _write_jpg(os.path.join(out_dir, "preview.jpg"), preview, quality=80)

    if thick_walls is not None:
        assets["thick_walls_path"] = _write_png(os.path.join(out_dir, "thick_walls.png"), thick_walls)

    if skeleton is not None:
        assets["skeleton_path"] = _write_png(os.path.join(out_dir, "skeleton.png"), skeleton)

    if concrete_mask is not None:
        assets["concrete_mask_path"] = _write_png(os.path.join(out_dir, "concrete_mask.png"), concrete_mask)

    if blocks_mask is not None:
        assets["blocks_mask_path"] = _write_png(os.path.join(out_dir, "blocks_mask.png"), blocks_mask)

    if flooring_mask is not None:
        assets["flooring_mask_path"] = _write_png(os.path.join(out_dir, "flooring_mask.png"), flooring_mask)

    if keep_debug and debug_layers:
        dbg_dir = os.path.join(out_dir, "debug")
        os.makedirs(dbg_dir, exist_ok=True)
        dbg_paths = {}
        for k, arr in debug_layers.items():
            if isinstance(arr, np.ndarray):
                dbg_paths[k] = _write_png(os.path.join(dbg_dir, f"{_safe_name(k)}.png"), arr)
        assets["debug_layers_paths"] = dbg_paths

    assets["assets_dir"] = out_dir
    return assets


def _read_image(path: str, flags: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"Failed to decode image: {path}")
    return img


if st is not None:
    @st.cache_data(show_spinner=False)
    def load_bgr(path: str) -> np.ndarray:
        return _read_image(path, cv2.IMREAD_COLOR)

    @st.cache_data(show_spinner=False)
    def load_gray(path: str) -> np.ndarray:
        return _read_image(path, cv2.IMREAD_GRAYSCALE)
else:
    def load_bgr(path: str) -> np.ndarray:
        return _read_image(path, cv2.IMREAD_COLOR)

    def load_gray(path: str) -> np.ndarray:
        return _read_image(path, cv2.IMREAD_GRAYSCALE)
