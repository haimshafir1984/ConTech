import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app


def _print_step(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="ConTech backend smoke test")
    parser.add_argument("--pdf", type=str, default="", help="Path to sample PDF for upload test")
    args = parser.parse_args()

    client = TestClient(app)

    plan_id = ""
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"[ERROR] PDF not found: {pdf_path}")
            return 1
        with pdf_path.open("rb") as f:
            resp = client.post(
                "/manager/workshop/upload",
                files={"file": (pdf_path.name, f, "application/pdf")},
            )
        if resp.status_code != 200:
            print("[ERROR] Upload failed")
            print(resp.status_code, resp.text)
            return 1
        upload_data = resp.json()
        plan_id = upload_data["summary"]["id"]
        _print_step("Upload", {"plan_id": plan_id, "plan_name": upload_data["summary"].get("plan_name")})
    else:
        plans_resp = client.get("/manager/workshop/plans")
        if plans_resp.status_code != 200:
            print("[ERROR] Failed to list plans")
            print(plans_resp.status_code, plans_resp.text)
            return 1
        plans = plans_resp.json().get("plans", [])
        if not plans:
            print("[ERROR] No plans found. Provide --pdf <path/to/file.pdf> for full smoke test.")
            return 1
        plan_id = plans[0]["id"]
        _print_step("Using Existing Plan", {"plan_id": plan_id, "plan_name": plans[0].get("plan_name")})

    draw_resp = client.get(f"/manager/drawing-data/{plan_id}")
    if draw_resp.status_code != 200:
        print("[ERROR] Drawing Data failed")
        print(draw_resp.status_code, draw_resp.text)
        return 1
    draw_data = draw_resp.json()
    _print_step(
        "Drawing Data",
        {
            "plan_id": draw_data.get("plan_id"),
            "scale_px_per_meter": draw_data.get("scale_px_per_meter"),
            "materials": draw_data.get("materials"),
        },
    )

    area_run_resp = client.post(
        f"/manager/area-analysis/{plan_id}/run",
        json={"segmentation_method": "watershed", "auto_min_area": True, "min_area_px": 500},
    )
    if area_run_resp.status_code != 200:
        print("[ERROR] Area Analysis run failed")
        print(area_run_resp.status_code, area_run_resp.text)
        return 1
    area_data = area_run_resp.json()
    _print_step(
        "Area Analysis",
        {
            "success": area_data.get("success"),
            "totals": area_data.get("totals"),
            "rooms_count": len(area_data.get("rooms", [])),
        },
    )

    overlay_resp = client.get(f"/manager/area-analysis/{plan_id}/overlay")
    if overlay_resp.status_code != 200:
        print("[ERROR] Overlay fetch failed")
        print(overlay_resp.status_code, overlay_resp.text)
        return 1
    print("\n=== Overlay ===")
    print(f"bytes={len(overlay_resp.content)} status={overlay_resp.status_code}")

    print("\n[OK] Smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
