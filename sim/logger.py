"""
sim/logger.py
Structured logging (file + console) and per-pick CSV metrics.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict


LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(log_dir: str, name: str = "pick_place") -> logging.Logger:
    """Configure root logger: rotating file + coloured console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{ts}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FMT,
        datefmt=DATE_FMT,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(name)
    logger.info("=" * 60)
    logger.info("PyBullet Pick-and-Place  —  session start")
    logger.info(f"Log file : {log_file.resolve()}")
    logger.info("=" * 60)
    return logger


class MetricsLogger:
    """Appends one row per pick-place attempt to a CSV file."""

    FIELDS = [
        "timestamp", "cycle", "cube_id", "color_class",
        "pick_xy_x", "pick_xy_y", "place_xy_x", "place_xy_y",
        "attempts", "result", "duration_s", "note",
    ]

    def __init__(self, csv_path: str) -> None:
        self._path = Path(csv_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()
        self._successes = 0
        self._total     = 0

    # ------------------------------------------------------------------
    def record(self, **kwargs: Any) -> None:
        self._total += 1
        if kwargs.get("result") == "success":
            self._successes += 1
        row = {k: kwargs.get(k, "") for k in self.FIELDS}
        row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self._path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)

    # ------------------------------------------------------------------
    @property
    def success_rate(self) -> float:
        return self._successes / self._total if self._total else 0.0

    def summary(self) -> str:
        return (
            f"Picks total={self._total}  "
            f"success={self._successes}  "
            f"rate={self.success_rate:.1%}"
        )
