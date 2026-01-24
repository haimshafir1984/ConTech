import json
import os
import sys

from analyzer import FloorPlanAnalyzer
from floor_extractor import analyze_floor_and_rooms
from utils import calculate_area_m2


def load_samples(samples_path: str):
    if not os.path.exists(samples_path):
        print(f"Samples file not found: {samples_path}")
        return []
    with open(samples_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    return data.get("samples", [])


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    samples_path = os.path.join(repo_root, "ground_truth_samples.json")
    samples = load_samples(samples_path)

    if not samples:
        print("No samples configured. Add entries to ground_truth_samples.json.")
        return 0

    analyzer = FloorPlanAnalyzer()
    failures = 0

    for sample in samples:
        pdf_path = sample.get("pdf_path")
        expected_area = sample.get("expected_total_area_m2")
        tolerance_pct = sample.get("tolerance_pct", 10)

        if not pdf_path or expected_area is None:
            print("Skipping sample with missing pdf_path/expected_total_area_m2.")
            continue

        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(repo_root, pdf_path)

        print(f"Running sample: {pdf_path}")
        (
            _pix,
            _skel,
            walls_mask,
            original_img,
            meta,
            _concrete,
            _blocks_mask,
            _flooring,
            _debug_img,
        ) = analyzer.process_file(pdf_path, save_debug=False, crop_bbox=None)

        result = analyze_floor_and_rooms(
            walls_mask=walls_mask,
            original_image=original_img,
            meters_per_pixel=meta.get("meters_per_pixel"),
            meters_per_pixel_x=meta.get("meters_per_pixel_x"),
            meters_per_pixel_y=meta.get("meters_per_pixel_y"),
            llm_rooms=None,
            segmentation_method="watershed",
            min_room_area_px=500,
        )

        if result.get("success") and result.get("totals", {}).get("total_area_m2"):
            measured_area = result["totals"]["total_area_m2"]
        else:
            flooring_pixels = meta.get(
                "pixels_flooring_area_refined",
                meta.get("pixels_flooring_area", 0),
            )
            measured_area = calculate_area_m2(
                flooring_pixels,
                meters_per_pixel=meta.get("meters_per_pixel"),
                meters_per_pixel_x=meta.get("meters_per_pixel_x"),
                meters_per_pixel_y=meta.get("meters_per_pixel_y"),
            )

        if not measured_area:
            print("  -> Measurement failed (no area).")
            failures += 1
            continue

        error_pct = abs(measured_area - expected_area) / expected_area * 100.0
        status = "OK" if error_pct <= tolerance_pct else "FAIL"
        print(
            f"  -> expected={expected_area:.2f} measured={measured_area:.2f} "
            f"error={error_pct:.1f}% [{status}]"
        )

        if status == "FAIL":
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
