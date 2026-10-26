"""Ghi bảng kết quả ra hai định dạng: Markdown (dán vào báo cáo) và CSV (mở bằng Excel)."""
from __future__ import annotations

import csv
from pathlib import Path

from src.config import TABLE_DIR


def to_markdown(header: list[str], rows: list[list[str]]) -> str:
    """Chuyển header + rows thành bảng Markdown."""
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"

    sep = "|" + "|".join("---" for _ in header) + "|"
    return "\n".join([line(header), sep, *(line(r) for r in rows)])


def write_table(name: str, title: str, header: list[str], rows: list[list[str]],
                notes: list[str] | None = None, out_dir: Path = TABLE_DIR) -> tuple[Path, Path]:
    """Ghi bảng ra ``<name>.md`` và ``<name>.csv``. Trả về hai đường dẫn."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, csv_path = out_dir / f"{name}.md", out_dir / f"{name}.csv"
    parts = [f"**{title}**", "", to_markdown(header, rows)]
    if notes:
        parts += ["", *(f"- {n}" for n in notes)]
    md_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:  # BOM để Excel hiện đúng dấu
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return md_path, csv_path
