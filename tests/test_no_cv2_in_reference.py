"""Bảo đảm thư mục reference/ KHÔNG dùng OpenCV (cv2)."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"
ROOT_DIR = REFERENCE_DIR.parent


def _reference_files() -> list[Path]:
    """Mọi file .py trong reference/."""
    files = sorted(REFERENCE_DIR.rglob("*.py"))
    assert files, "reference/ không có file .py nào"
    return files


def _cv2_uses(tree: ast.AST) -> list[str]:
    """Liệt kê mọi cách nạp cv2: import cv2, from cv2 import ..., __import__/import_module('cv2')."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] == "cv2"]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "cv2":
            found.append(f"from {node.module}")
        elif isinstance(node, ast.Call) and any(isinstance(a, ast.Constant) and a.value == "cv2" for a in node.args):
            found.append("dynamic import 'cv2'")
    return found


def test_no_cv2_import_in_source() -> None:
    """Phân tích cú pháp: không file nào trong reference/ nạp cv2."""
    for path in _reference_files():
        uses = _cv2_uses(ast.parse(path.read_text(encoding="utf-8")))
        assert not uses, f"{path.name} dùng cv2: {uses}"


def test_no_cv2_import_in_any_dependency() -> None:
    """Chạy reference/ trong tiến trình con có chặn cv2: phải nạp và chạy được bình thường.

    Cách này bắt cả trường hợp reference/ gián tiếp nạp cv2 qua module khác (ví dụ src/).
    """
    code = (
        "import sys; sys.modules['cv2'] = None\n"   # mọi lệnh import cv2 sẽ ném ImportError
        "import numpy as np\n"
        "from reference import matrices as RM, warp_ref as RW\n"
        "img = (np.arange(64 * 64) % 251).astype(np.uint8).reshape(64, 64)\n"
        "m = RM.rotation_matrix((31.5, 31.5), 30)\n"
        "RW.warp_affine_ref(img, m, (64, 64), 'cubic')\n"
        "RW.resize_ref(img, 2.0)\n"
        "h = RM.homography_from_4pts([[0, 0], [63, 0], [63, 63], [0, 63]], [[5, 3], [60, 0], [63, 63], [0, 58]])\n"
        "RW.warp_perspective_ref(img, h, (64, 64))\n"
        "assert 'cv2' not in [k for k, v in sys.modules.items() if v is not None]\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT_DIR, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
