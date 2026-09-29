#!/usr/bin/env python3
"""Offline unit tests - no network required. These DO run in a sandboxed
build environment and were actually executed while building this feature
(see the project chat / delivery notes for the real output). They cover
everything that does not require reaching the internet:

  - the geostationary projection / region-crop math (region.py)
  - JSON state storage, atomic writes, duplicate detection (storage.py)
  - the pipeline's error handling when a source is not configured/implemented
  - config.py's .env parsing

They do NOT prove the Himawari URLs are still correct today, or that the
crop lines up with the real coastline - only test_connection.py, run with
real internet, can prove that.

Usage:
    python backend/satellite/test_offline.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satellite import region, storage, config  # noqa: E402
from satellite.pipeline import run_once  # noqa: E402
from satellite.sources.himawari_nict import HimawariNictSource  # noqa: E402
from satellite.sources.base import InvalidResponseError  # noqa: E402
from PIL import Image  # noqa: E402
import io  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
_results = []


def check(name, condition):
    _results.append((name, PASS if condition else FAIL))
    print(f"[{PASS if condition else FAIL}] {name}")


def test_region_math():
    # Mumbai (~19.07N, 72.87E) should be visible and project somewhere
    # inside a plausible full-disk image from Himawari (140.7E) - it's a
    # real, named point with known coordinates, used only as a math sanity
    # check (not a claim about the rendered picture's exact pixel layout).
    angles = region.latlon_to_scan_angles(19.07, 72.87)
    check("Mumbai projects to visible scan angles (not None)", angles is not None)

    limb = region.full_disk_limb_angle()
    check("Full-disk limb angle is a small positive number of radians",
          0.1 < limb < 0.2)

    coverage = region.latlon_box_to_pixel_box(-5, 30, 40, 100, 2200, 2200)
    check("North Indian Ocean box projects to a valid pixel box on a 2200x2200 disk",
          coverage.box is not None and coverage.box.valid())
    if coverage.box is not None:
        box = coverage.box
        check("Pixel box is within image bounds",
              0 <= box.left < box.right <= 2200 and 0 <= box.top < box.bottom <= 2200)
    check("Coverage correctly reports the box as only PARTIALLY visible from Himawari "
          "(the western Arabian Sea, lon<~60E, is beyond its horizon)",
          not coverage.fully_visible and 0 < coverage.fraction_visible < 1)

    # A point on the far side of the Earth from Himawari's viewpoint
    # (roughly over the Atlantic / the Americas) must not be "visible".
    far_side = region.latlon_to_scan_angles(0, -60)
    check("A point on the far side of the Earth is correctly reported as not visible",
          far_side is None)


def test_crop_on_synthetic_image():
    # Build a plain synthetic "full disk" (this is a synthetic test image,
    # not a real satellite picture - it exists only to prove the crop
    # function returns a correctly-shaped, non-empty image).
    synthetic_disk = Image.new("RGB", (2200, 2200), color=(10, 10, 40))
    cropped, coverage = region.crop_north_indian_ocean(synthetic_disk, -5, 30, 40, 100)
    check("crop_north_indian_ocean returns an image for a synthetic full disk", cropped is not None)
    if cropped is not None:
        check("Cropped image has positive dimensions", cropped.width > 0 and cropped.height > 0)
    check("crop_north_indian_ocean also returns a CoverageResult", hasattr(coverage, "fraction_visible"))


def test_storage_roundtrip():
    tmp_dir = Path(tempfile.mkdtemp(prefix="prism_tc_sat_test_"))
    original_data_dir = config.DATA_DIR
    try:
        config.DATA_DIR = tmp_dir
        config.RAW_DIR = tmp_dir / "raw"
        config.PROCESSED_DIR = tmp_dir / "processed"
        config.LATEST_DIR = tmp_dir / "latest"
        config.METADATA_DIR = tmp_dir / "metadata"
        config.HISTORY_FILE = config.METADATA_DIR / "history.json"
        config.STATE_FILE = config.METADATA_DIR / "state.json"
        config.ensure_dirs()

        state = storage.write_state({"status": "LIVE", "observation_time_utc": "2026-01-01T00:00:00Z"})
        check("write_state returns the merged state", state["status"] == "LIVE")

        reread = storage.read_state()
        check("read_state reflects what was written", reread["observation_time_utc"] == "2026-01-01T00:00:00Z")

        check("already_have_observation is True for the same timestamp",
              storage.already_have_observation("2026-01-01T00:00:00Z"))
        check("already_have_observation is False for a different timestamp",
              not storage.already_have_observation("2026-01-01T00:10:00Z"))

        storage.append_history({"source": "test", "observation_time_utc": "2026-01-01T00:00:00Z"})
        history = storage.read_history()
        check("append_history / read_history round-trips one entry", len(history) == 1)

        # Corrupted state file must not crash read_state().
        config.STATE_FILE.write_text("{not valid json", encoding="utf-8")
        recovered = storage.read_state()
        check("A corrupted state.json falls back to defaults instead of raising",
              recovered["status"] == "NO_DATA")
    finally:
        config.DATA_DIR = original_data_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_not_implemented_source_is_handled_cleanly():
    tmp_dir = Path(tempfile.mkdtemp(prefix="prism_tc_sat_test2_"))
    original = (config.DATA_DIR, config.RAW_DIR, config.PROCESSED_DIR,
                config.LATEST_DIR, config.METADATA_DIR, config.HISTORY_FILE,
                config.STATE_FILE, config.SATELLITE_SOURCE, config.SATELLITE_FALLBACK_SOURCE)
    try:
        config.DATA_DIR = tmp_dir
        config.RAW_DIR = tmp_dir / "raw"
        config.PROCESSED_DIR = tmp_dir / "processed"
        config.LATEST_DIR = tmp_dir / "latest"
        config.METADATA_DIR = tmp_dir / "metadata"
        config.HISTORY_FILE = config.METADATA_DIR / "history.json"
        config.STATE_FILE = config.METADATA_DIR / "state.json"
        config.SATELLITE_SOURCE = "mosdac"          # known not-implemented
        config.SATELLITE_FALLBACK_SOURCE = ""
        config.ensure_dirs()

        state = run_once()
        check("run_once() with an unimplemented source returns ERROR, not a crash",
              state.get("status") == "ERROR")
        check("Error message names the failing source",
              "mosdac" in (state.get("error") or "").lower() or "source" in (state.get("error") or "").lower())
    finally:
        (config.DATA_DIR, config.RAW_DIR, config.PROCESSED_DIR, config.LATEST_DIR,
         config.METADATA_DIR, config.HISTORY_FILE, config.STATE_FILE,
         config.SATELLITE_SOURCE, config.SATELLITE_FALLBACK_SOURCE) = original
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_tiny_but_valid_tile_is_not_rejected_as_truncated():
    # Regression test for a real bug found via test_connection.py on real
    # hardware: a corner tile of a full-disk image (far from the round
    # Earth disk, inside the square canvas) is almost entirely one flat
    # color and can legitimately PNG-compress to well under 512 bytes.
    # A byte-size-based "looks truncated" check used to reject this exact
    # case (observed live: a genuine 444-byte tile). The fix validates by
    # decoding + checking dimensions instead of counting bytes.
    src = HimawariNictSource()
    tiny_valid = Image.new("RGB", (src.tile_px, src.tile_px), (10, 10, 10))
    buf = io.BytesIO()
    tiny_valid.save(buf, format="PNG", optimize=True, compress_level=9)
    content = buf.getvalue()
    # PIL's own encoder isn't as aggressive as whatever NICT uses server-side
    # (the real corner tile observed live was 444 bytes for the same 550x550
    # size), but the point stands regardless of the exact byte count: a
    # small, fully-valid, correctly-sized tile must never be rejected just
    # for being small.
    tile = src._validate_tile("http://test/tile.png", content)
    check("a small-but-valid, correctly-sized tile is accepted, not rejected as truncated",
          tile.size == (src.tile_px, src.tile_px))

    truncated = content[: len(content) // 2]
    rejected = False
    try:
        src._validate_tile("http://test/truncated.png", truncated)
    except InvalidResponseError:
        rejected = True
    check("an actually-truncated tile (half the bytes cut off) is still correctly rejected",
          rejected)


if __name__ == "__main__":
    test_region_math()
    test_crop_on_synthetic_image()
    test_storage_roundtrip()
    test_not_implemented_source_is_handled_cleanly()
    test_tiny_but_valid_tile_is_not_rejected_as_truncated()

    failed = [name for name, result in _results if result == FAIL]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed.")
    if failed:
        print("FAILED:", ", ".join(failed))
        raise SystemExit(1)
    raise SystemExit(0)
