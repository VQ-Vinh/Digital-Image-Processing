"""Đo thời gian thực thi: trung bình BENCH_REPEAT lần, bỏ BENCH_WARMUP lần chạy đầu (warm-up)."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import BENCH_REPEAT, BENCH_WARMUP

MS_PER_S: float = 1000.0


@dataclass
class Timing:
    """Kết quả đo: trung bình và độ lệch chuẩn (ms)."""

    mean_ms: float
    std_ms: float
    repeat: int


def benchmark(fn: Callable[..., Any], *args: Any, repeat: int = BENCH_REPEAT,
              warmup: int = BENCH_WARMUP, **kwargs: Any) -> Timing:
    """Gọi fn(*args, **kwargs) warmup + repeat lần, chỉ tính thời gian repeat lần sau."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    samples = np.empty(repeat)
    for i in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        samples[i] = (time.perf_counter() - t0) * MS_PER_S
    return Timing(mean_ms=float(samples.mean()), std_ms=float(samples.std()), repeat=repeat)
