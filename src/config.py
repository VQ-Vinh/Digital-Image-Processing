"""Hằng số dùng chung cho toàn dự án: đường dẫn, seed, kích thước ảnh, cấu hình hình vẽ."""
from __future__ import annotations

from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"
FIG_DIR: Path = RESULTS_DIR / "figures"
TABLE_DIR: Path = RESULTS_DIR / "tables"
REPORT_NUMBERS_PATH: Path = RESULTS_DIR / "report_numbers.md"

SEED: int = 2025

# Hai kích thước thử nghiệm (rộng, cao) theo quy ước dsize của OpenCV.
SIZE_SMALL: tuple[int, int] = (512, 512)
SIZE_LARGE: tuple[int, int] = (1920, 1080)

# Đo thời gian: trung bình 10 lần, bỏ 1 lần warm-up.
BENCH_REPEAT: int = 10
BENCH_WARMUP: int = 1

# Cấu hình hình vẽ.
FIG_DPI: int = 200
FIG_FONT: str = "DejaVu Sans"  # có đủ glyph tiếng Việt
TITLE_FONTSIZE: int = 10

UINT8_MAX: int = 255
